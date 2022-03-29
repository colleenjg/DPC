import logging
from pathlib import Path
import time

import numpy as np
import torch
from tqdm import tqdm

from utils import data_utils, loss_utils, misc_utils, training_utils

# a few global variables
CRITERION_FCT = torch.nn.CrossEntropyLoss
TOPK = [1, 3, 5]

logger = logging.getLogger(__name__)


#############################################
def resize_input_seq(input_seq):
    # if the dataset is set to 'supervised' and 'test mode', 
    # each batch item is a sub-batch in which all sets of sequences 
    # share a label

    B, SUB_B, N, C, SL, H, W = input_seq.size()
    input_seq.view(B * SUB_B, N, C, SL, H, W)

    return input_seq, SUB_B


#############################################
def train_epoch(data_loader, model, optimizer, epoch_n=0, num_epochs=50, 
                log_idx=0, topk=TOPK, device="cpu", log_freq=5, 
                writer=None):
    """
    train_epoch(data_loader, model, optimizer)
    """
    
    losses = training_utils.AverageMeter()
    topk_meters = [training_utils.AverageMeter() for _ in range(topk)]
    model = model.to(device)
    model.train()

    loss_keys = [
        "detailed_loss", "loss_by_batch", "dot_by_batch", "target_by_batch"
        ]
    train_dict = {key: list() for key in loss_keys}

    supervised = hasattr(model, "num_classes")
    
    criterion = CRITERION_FCT()
    for idx, (input_seq, target) in enumerate(data_loader):
        start_time = time.perf_counter()
        input_seq = input_seq.to(device)
        input_seq_shape = input_seq.size
        [output_, mask_] = model(input_seq)
        
        # visualize 2 examples from the batch
        if writer is not None and idx % log_freq == 0:
            misc_utils.write_input_seq_tb(writer, input_seq, n=2, i=log_idx)
        del input_seq

        if supervised:
            B, N, num_classes = output_.size()
            output_flattened = output_.view(B * N, num_classes)
            target_flattened = target.repeat(1, N).view(-1)

        else:
            logger.debug(
                "Model called next.\n"
                f"Input sequence shape: {input_seq_shape} "
                "(expecting [10, 4, 3, 5, 128, 128]).\n"
                f"Score shape: {output_.shape} "
                "(expecting a 6D tensor: [B, PS, D2, B, PS, D2]).\n"
                f"Mask shape: {mask_.shape}"
            )
            target, (B, PS, D2) = training_utils.get_target_from_mask(mask_)
        
            # output is a 6d tensor: [B, PS, D2, B, PS, D2]
            output_flattened = output_.view(B * PS * D2, B * PS * D2)
            target_flattened = target.view(
                B * PS * D2, B * PS * D2
                ).argmax(dim=1)

        target_flattened = target_flattened.to(device)
        loss = criterion(output_flattened, target_flattened)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update meters, in-place
        losses.update(loss.item(), B)
        loss_utils.update_topk_meters(
            topk_meters, output_flattened, target_flattened, ks=topk
            )
        del target_flattened

        # collect some values
        criterion_measure = data_utils.CRITERION_FCT(reduction="none")       
        train_dict["loss_by_batch"].append(
            criterion_measure(
                output_flattened, target_flattened
                ).view(B, D2).mean(axis=1).to("cpu").tolist()
            )
        train_dict["dot_by_batch"].append(output_.to("cpu").tolist())
        train_dict["target_by_batch"].append(target.to("cpu").tolist())
        del output_

        # log results
        if idx % log_freq == 0:
            training_utils.log_weight_decay_prop(model)
            loss_avg, acc_avg, loss_val, acc_val, log_str = \
                loss_utils.get_stats(
                    losses, topk_meters, ks=topk, local=True, last=True
                )
            logger.info(
                f"Epoch: [{epoch_n}/{num_epochs - 1}]"
                f"[{idx}/{len(data_loader)}]\t"
                f"{log_str}\tT:{time.perf_counter() - start_time:.2f}"
                )
            
            if writer is not None:
                writer.add_scalar("local/loss", loss_val, log_idx)
                writer.add_scalar("local/accuracy", acc_val, log_idx)

            logger.debug(
                f"Previous sequence: {data_loader.prev_seq[-1]}"
                f"Batch sequences: {data_loader.prev_seq[-B:]}"
                f"Batch loss: {train_dict['loss_by_batch'][-1]}"
                )

            train_dict["detailed_loss"].append(losses.val)

            log_idx += 1
    
    train_dict["loss"]         = loss_avg
    train_dict["acc"]          = acc_avg
    train_dict["topk_meters"]  = topk_meters

    return train_dict, log_idx


#############################################
def val_or_test_epoch(data_loader, model, epoch_n=0, num_epochs=10, 
                      topk=data_utils.TOPK, device="cpu", output_dir=None):
    """
    val_or_test_epoch(data_loader, model)
    """

    losses = training_utils.AverageMeter()
    topk_meters = [training_utils.AverageMeter() for _ in range(topk)]
    model = model.to(device)
    model.eval()

    criterion = data_utils.CRITERION_FCT()

    confusion_mat = None
    supervised = hasattr(model, "num_classes")
    if supervised and output_dir is not None:
        confusion_mat = training_utils.ConfusionMeter(model.num_classes)
        Path(output_dir).mkdir(exist_ok=True)

    shared_pred = False
    with torch.no_grad():
        for idx, (input_seq, target) in tqdm(
            enumerate(data_loader), total=len(data_loader)
            ):

            if supervised and len(input_seq.size) == 7:
                shared_pred = True # applies to all batches
                input_seq, SUB_B = resize_input_seq(input_seq)

            input_seq = input_seq.to(device)
            [output_, mask_] = model(input_seq)
            del input_seq

            if supervised:
                num_classes = output_.size()[-1]
                if shared_pred:
                    B_comb, N = input_seq.size()[:2]
                    B = B_comb // SUB_B
                    USE_N = SUB_B * N

                    # group sequences that share a label
                    output_flattened = output_.view(B, USE_N, num_classes)

                    # for each batch item, average the softmaxed class 
                    # predictions across sequences
                    output_flattened = torch.mean(
                        torch.nn.functional.softmax(output_, 2),
                        1) # B x num_classes
                    target_flattened = target
                else:
                    # consider all sequences separately, even if they share a label 
                    output_flattened = output_.view(B * USE_N, num_classes)
                    target_flattened = target.repeat(1, USE_N).view(-1) # B * USE_N
            
            else:
                if idx == 0: # same across all batches
                    target, (B, B2, NS, NP, SQ) = data_utils.process_output(
                        mask_
                        )

                # [B, P, SQ, B, N, SQ]
                output_flattened = output_.view(B*NP*SQ, B2*NS*SQ)
                target_flattened = target.view(B*NP*SQ, B2*NS*SQ).to(device)
                target_flattened = target_flattened.to(int).argmax(dim=1)
                  
            target_flattened = target_flattened.to(device)
            loss = criterion(output_flattened, target_flattened)

            training_utils.update_topk_meters(
                topk_meters, output_flattened, target_flattened, 
                ks=data_utils.TOPK
                )

            losses.update(loss.item(), B)
            if confusion_mat is not None:
                _, pred = torch.max(output_flattened, 1)
                confusion_mat.update(pred, target_flattened.view(-1).byte())

    loss_avg, acc_avg, log_str = loss_utils.get_stats(
            losses, topk_meters, ks=topk, local=False
        )

    logger.info(f"Epoch: [{epoch_n}/{num_epochs - 1}] [val] {log_str}")

    val_dict = {
        "loss" : loss_avg,
        "acc"  : acc_avg,
        "topk_meters": topk_meters,
    }
    
    if confusion_mat is not None:
        confusion_mat.plot_mat(Path(output_dir, "confusion_matrix.svg"))
        training_utils.write_log(
            content=log_str,
            epoch=epoch_n,
            filename=Path(output_dir, "test_log.md")
            )

    return val_dict


#############################################
def train_full(train_loader, model, optimizer, output_dir=".", net_name=None, 
               dataset="gabors", num_epochs=10, scheduler=None, device="cpu", 
               val_loader=None, seed_info=None, unexp_epoch=10, log_freq=5, 
               use_tb=False, reload_kwargs=dict()):
    """
    train_full(train_loader, model, optimizer)
    """

    model = model.to(device)

    log_idx, best_acc, start_epoch = training_utils.load_checkpoint(
        model, optimizer, **reload_kwargs
        )

    # setup tools
    model_direc = Path(output_dir, "model")
    Path(model_direc).mkdir(parents=True, exist_ok=True)

    # check validation parameters
    save_best = False if val_loader is None else True
    is_best = False
    if save_best:
        best_acc = -np.inf if best_acc is None else best_acc
    else:
        best_acc = None
    test = reload_kwargs["test"] if "test" in reload_kwargs.keys() else False

    # tensorboard
    writer_train, writer_val = None, None
    if use_tb: 
        tb_dir = Path(output_dir, "tensorboard")
        writer_train, writer_val = misc_utils.init_tb_writers(
            tb_dir, val=save_best
            )

    ### main loop ###
    loss_dict = loss_utils.init_loss_dict(dataset)
    for epoch_n in range(start_epoch, num_epochs):
        if dataset == "gabors":
            if not train_loader.unexp and epoch_n > unexp_epoch:
                train_loader.unexp = True
                logger.info(f"Mode: {train_loader.mode}")
        
        train_dict = train_epoch(
            train_loader, 
            model, 
            optimizer, 
            epoch_n=epoch_n, 
            num_epochs=num_epochs,
            log_idx=log_idx, 
            device=device, 
            log_freq=log_freq,
            )

        loss_dict["train"]["epoch_n"].append(epoch_n)
        for key in train_dict.keys():
            if key in loss_dict["train"].keys():
                loss_dict["train"]["loss"].append(train_dict[key])
        if dataset == "gabors":
            loss_dict["train"]["seq"].append(train_loader.prev_seq)

        if save_best or test:
            val_dict = val_or_test_epoch(
                val_loader, 
                model, 
                epoch_n=epoch_n, 
                num_epochs=num_epochs, 
                device=device
                )
            if save_best:
                is_best = val_dict["acc"] > best_acc
                best_acc = max(val_dict["acc"], best_acc)

        if scheduler is not None:
            scheduler.step(epoch_n)

        # Save and log loss information
        loss_utils.save_loss_dict(
            loss_dict, output_dir=output_dir, seed=seed_info, dataset=dataset, 
            unexp_epoch=unexp_epoch
            )
        if use_tb:
            misc_utils.update_tb(
                writer_train, train_dict, epoch_n=epoch_n, 
                writer_val=writer_val, val_dict=val_dict
                )

        logger.debug(f"Epoch train loss: {train_dict['loss']}")
        if save_best:
            logger.debug(f"Epoch val loss: {val_dict['loss']}")

        # save checkpoint
        epoch_path = Path(model_direc, f"epoch{epoch_n}.pth.tar")
        training_utils.save_checkpoint(
            {
            "epoch_n": epoch_n,
            "net": net_name,
            "state_dict": model.state_dict(),
            "best_acc": best_acc,
            "optimizer": optimizer.state_dict(),
            "log_idx": log_idx
            }, 
            is_best, 
            filename=epoch_path, 
            keep_all=False
        )
 
    logger.info(
        f"Training from ep {start_epoch} to {num_epochs} finished"
        )

