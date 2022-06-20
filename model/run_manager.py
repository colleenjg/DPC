import logging
from pathlib import Path
import time

from matplotlib import pyplot as plt
import numpy as np
import torch

from dataset import dataset_3d, gabor_stimuli
from utils import loss_utils, misc_utils, training_utils

# a few global variables
CRITERION_FCT = torch.nn.CrossEntropyLoss
TOPK = [1, 3, 5]

logger = logging.getLogger(__name__)

TAB = "    "

#############################################
def resize_input_seq(input_seq):
    """
    resize_input_seq(input_seq)

    if the dataset is set to 'supervised' and 'test mode', 
    each batch item is a sub-batch in which all sets of sequences 
    share a label

    """

    B, SUB_B, N, C, SL, H, W = input_seq.size()
    input_seq = input_seq.reshape(B * SUB_B, N, C, SL, H, W)

    return input_seq, SUB_B


#############################################
def update_dataset_possizes(main_loader, val_loader=None, seed=None, incr=0):
    """
    update_dataset_possizes(main_loader)
    """

    if not main_loader.dataset.same_possizes:
        return

    if seed is None:
        seed = np.random.choice(int(2**32))
    
    main_loader.dataset.set_possizes(seed=seed + incr, reset=True)
    
    if val_loader is not None:
        val_loader.dataset.set_possizes(seed=seed + incr, reset=True)    

    return seed


#############################################
def update_unexp(main_loader, val_loader=None, epoch_n=0, unexp_epoch=0):
    """
    update_unexp(main_loader)
    """

    if not main_loader.dataset.unexp and epoch_n >= unexp_epoch:
        main_loader.dataset.unexp = True
        loader_mode = main_loader.dataset.mode
        dataset_str = f" {loader_mode} dataset"
        if val_loader is not None and not val_loader.dataset.unexp:
            val_loader.dataset.unexp = True
            dataset_str = (
                f"{loader_mode} and {val_loader.dataset.mode} datasets"
                )

        logger.info(f"Setting 'unexp' to True for {dataset_str}.", 
            extra={"spacing": "\n"}
            )


#############################################
def get_sup_target(dataset, sup_target):
    """
    get_sup_target(dataset, sup_target)
    """
    
    gabors = isinstance(dataset, gabor_stimuli.GaborSequenceGenerator)

    if gabors:
        sup_target = torch.moveaxis(sup_target, -1, 0)
        target_images = dataset.image_label_to_image(
            sup_target[0].to("cpu").numpy().reshape(-1)
            )
        target_images = np.asarray(target_images).reshape(
            sup_target[0].shape
            )
        sup_target = [target_images.tolist(), sup_target[1].tolist()]
    else:
        sup_target = sup_target.tolist()

    return sup_target


#############################################
def train_epoch(data_loader, model, optimizer, epoch_n=0, num_epochs=50, 
                log_idx=0, topk=TOPK, loss_weight=None, device="cpu", 
                log_freq=5, writer=None, train_off=False, save_by_batch=False):
    """
    train_epoch(data_loader, model, optimizer)
    """
    
    losses = loss_utils.AverageMeter()
    topk_meters = [loss_utils.AverageMeter() for _ in range(len(topk))]
    model = model.to(device)

    if epoch_n == 0 or train_off:
        log_str = " (pre-training baseline)" if epoch_n == 0 else "" 
        logger.info(
            f"Running epoch {epoch_n} with no weight updates{log_str}.", 
            extra={"spacing": "\n"}
            )
        spacing = ""
        train = False # estimate untrained performance
        model.eval()
    else:
        spacing = "\n" # for first log of the epoch
        train = True
        model.train()

    supervised = training_utils.get_num_classes(model) is not None

    train_dict = loss_utils.init_loss_dict(
        ks=topk, val=False, supervised=supervised, save_by_batch=save_by_batch
        )["train"]
    train_dict["epoch_n"] = epoch_n
    
    if loss_weight is not None:
        loss_weight = loss_weight.to(device)
    criterion = CRITERION_FCT(weight=loss_weight)
    criterion_no_reduction = CRITERION_FCT(weight=loss_weight, reduction="none")

    batch_times = []
    for idx, (input_seq, sup_target) in enumerate(data_loader):
        start_time = time.perf_counter()
        input_seq = input_seq.to(device)
        input_seq_shape = input_seq.size()
        [output_, mask_] = model(input_seq)
        
        # visualize 2 examples from the batch
        if writer is not None and idx % log_freq == 0:
            misc_utils.write_input_seq_tb(writer, input_seq, n=2, i=log_idx)
        del input_seq

        if supervised:
            target = sup_target
            B, N, num_classes = output_.size()
            output_flattened = output_.reshape(B * N, num_classes)
            target_flattened = target.repeat(1, N).reshape(-1)
            loss_reshape = (B, N)
        else:
            logger.debug(
                "Model called next.\n"
                f"Input sequence shape: {input_seq_shape} "
                "(expecting [B, N, C, SL, H, W]).\n"
                f"Output shape: {output_.shape} "
                "(expecting a 6D tensor: [B, PS, HW, B_per, PS, HW]).\n"
                f"Mask shape: {mask_.size()}"
            )

            # batch x pred step x dim squared x batch/GPU x pred step x dim squared
            (B, PS, HW, B_per, _, _) = mask_.size()
            flat_dim = B * PS * HW
            flat_dim_per = B_per * PS * HW # B_per: batch size per GPU
            target = training_utils.get_target_from_mask(mask_)
        
            # output is a 6d tensor: [B, PS, HW, B_per, PS, HW]
            output_flattened = output_.reshape(flat_dim, flat_dim_per)
            target_flattened = target.reshape(
                flat_dim, flat_dim_per).argmax(dim=1)
            loss_reshape = (B, PS, HW)

        target_flattened = target_flattened.to(device)
        loss = criterion(output_flattened, target_flattened)
        
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # update meters, in-place
        losses.update(loss.item(), B)
        loss_utils.update_topk_meters(
            topk_meters, output_flattened, target_flattened, ks=topk
            )

        # collect some values
        batch_loss = criterion_no_reduction(
            output_flattened, target_flattened
            ).to("cpu").reshape(loss_reshape)

        if not supervised:
            batch_loss = batch_loss.reshape(B, PS * HW).mean(axis=1)

        if save_by_batch:
            train_dict["batch_epoch_n"].append(epoch_n)
            train_dict["loss_by_batch"].append(losses.val)
            train_dict["loss_by_item"].append(batch_loss.to("cpu").tolist())
            # train_dict["output_by_batch"].append(output_.to("cpu").tolist())
            # train_dict["target_by_batch"].append(target.to("cpu").tolist())            
            if not supervised:
                train_dict["sup_target_by_batch"].append(
                    get_sup_target(data_loader.dataset, sup_target)
                    )

        del output_
        del target_flattened

        # record batch time
        stop_time = time.perf_counter()
        batch_time = stop_time - start_time
        batch_times.append(batch_time)

        # log results
        if idx % log_freq == 0 or idx == len(data_loader) - 1:
            loss_avg, acc_avg, loss_val, acc_val, log_str = \
                loss_utils.get_stats(
                    losses, topk_meters, ks=topk, local=True, last=True
                )
            
            mean_batch_time = misc_utils.format_time(
                np.mean(batch_times), sep_min=False
                )
            batch_times = []

            logger.info(
                f"Epoch: [{epoch_n}/{num_epochs}]"
                f"[{idx}/{len(data_loader) - 1}]{TAB}"
                f"{log_str}{TAB}Time: {mean_batch_time}/batch", 
                extra={"spacing": spacing}
                )
            spacing = "" # for all but first log of the epoch

            if writer is not None:
                writer.add_scalar("local/loss", loss_val, log_idx)
                writer.add_scalar("local/accuracy", acc_val, log_idx)

            log_str = f"Batch loss: {batch_loss[-1]}"

            logger.debug(log_str)

            if train and supervised and optimizer:
                training_utils.log_weight_decay_prop(model) # log decay info
            
            log_idx += 1

    train_dict["loss"] = loss_avg
    train_dict["acc"]  = acc_avg
    for i, k in enumerate(topk):
        train_dict[f"top{k}"] = topk_meters[i].local_avg

    return train_dict, log_idx


#############################################
def val_or_test_epoch(data_loader, model, epoch_n=0, num_epochs=10, 
                      topk=TOPK, loss_weight=None, device="cpu", test=False, 
                      output_dir=None, save_by_batch=False):
    """
    val_or_test_epoch(data_loader, model)
    """

    losses = loss_utils.AverageMeter()
    topk_meters = [loss_utils.AverageMeter() for _ in range(len(topk))]
    model = model.to(device)
    model.eval()

    if loss_weight is not None:
        loss_weight = loss_weight.to(device)
    criterion = CRITERION_FCT(weight=loss_weight)
    criterion_no_reduction = CRITERION_FCT(weight=loss_weight, reduction="none")

    confusion_mat = None
    num_classes = training_utils.get_num_classes(model)
    supervised = num_classes is not None

    val_dict = loss_utils.init_loss_dict(
        ks=topk, val=True, supervised=supervised, save_by_batch=save_by_batch
        )["val"]
    val_dict["epoch_n"] = epoch_n

    if supervised and output_dir is not None:
        confusion_mat = loss_utils.ConfusionMeter(num_classes)
        Path(output_dir).mkdir(exist_ok=True)
    if not supervised:
        chance_level = loss_utils.AverageMeter()

    shared_pred = False
    start_time = time.perf_counter()
    with torch.no_grad():
        for idx, (input_seq, sup_target) in enumerate(data_loader):
            if supervised and len(input_seq.size()) == 7:
                shared_pred = True # applies to all batches
                input_seq, SUB_B = resize_input_seq(input_seq)

            input_seq = input_seq.to(device)
            [output_, mask_] = model(input_seq)
            del input_seq

            if supervised:
                target = sup_target
                if shared_pred:
                    B_comb, N, num_classes = output_.size()
                    B = B_comb // SUB_B
                    USE_N = SUB_B * N

                    # group sequences that share a label
                    output_flattened = output_.reshape(B, USE_N, num_classes)

                    # for each batch item, average the softmaxed class 
                    # predictions across sequences
                    output_flattened = torch.mean(
                        torch.nn.functional.softmax(output_flattened, 2),
                        1) # B x num_classes
                    target_flattened = target.reshape(-1)
                    loss_reshape = (B, )

                else:
                    # consider all sequences separately, even if they share a label 
                    B, N, num_classes = output_.size()
                    output_flattened = output_.reshape(B * N, num_classes)
                    target_flattened = target.repeat(1, N).reshape(-1)
                    loss_reshape = (B, N)
            
            else:
                # batch x pred step x dim squared x batch/GPU x pred step x dim squared
                (B, PS, HW, B_per, _, _) = mask_.size()
                flat_dim = B * PS * HW
                flat_dim_per = B_per * PS * HW # B_per: batch size per GPU
                target = training_utils.get_target_from_mask(mask_)
                                
                # output is a 6d tensor: [B, PS, HW, B, PS, HW]
                output_flattened = output_.reshape(flat_dim, flat_dim_per)
                target_flattened = target.reshape(
                    flat_dim, flat_dim_per
                    ).argmax(dim=1)
                loss_reshape = (B, PS, HW)
                
                chance_level.update(1 / np.product(flat_dim_per), 1)

            target_flattened = target_flattened.to(device)
            loss = criterion(output_flattened, target_flattened)

            loss_utils.update_topk_meters(
                topk_meters, output_flattened, target_flattened, ks=topk
                )

            # collect some values
            batch_loss = criterion_no_reduction(
                output_flattened, target_flattened
                ).to("cpu").reshape(loss_reshape)
                
            if not supervised:
                batch_loss = batch_loss.mean(axis=2) # across HW dimension

            if save_by_batch:
                val_dict["batch_epoch_n"].append(epoch_n)
                val_dict["loss_by_batch"].append(losses.val)
                val_dict["loss_by_item"].append(batch_loss.to("cpu").tolist())
                # val_dict["output_by_batch"].append(output_.to("cpu").tolist())
                # val_dict["target_by_batch"].append(target.to("cpu").tolist())            
                if not supervised:
                    val_dict["sup_target_by_batch"].append(
                        get_sup_target(data_loader.dataset, sup_target)
                        )

            losses.update(loss.item(), B)
            if confusion_mat is not None:
                _, pred = torch.max(output_flattened, 1)
                confusion_mat.update(pred, target_flattened.reshape(-1).byte())

    if supervised:
        chance = 1 / num_classes
    else:
        chance = chance_level.avg
    loss_avg, acc_avg, log_str = loss_utils.get_stats(
            losses, topk_meters, ks=topk, local=False, chance=chance
        )

    stop_time = time.perf_counter()
    time_str = misc_utils.format_time(stop_time - start_time, sep_min=True)
    
    if test:
        base_str = f"[{epoch_n}] [test]"
    else:
        base_str = f"[{epoch_n}/{num_epochs}] [val]"
    
    logger.info(
        f"Epoch: {base_str}{TAB}{log_str}{TAB}Time: {time_str}", 
        extra={"spacing": "\n"}
        )

    val_dict["loss"] = loss_avg
    val_dict["acc"]  = acc_avg

    for i, k in enumerate(topk):
        val_dict[f"top{k}"] = topk_meters[i].avg
    
    mode_str = "test" if test else "val"
    if confusion_mat is not None:
        confusion_mat.plot_mat(
            Path(output_dir, f"{mode_str}_confusion_matrix.svg")
            )

    training_utils.write_log(
        content=log_str,
        epoch_n=epoch_n,
        filename=Path(output_dir, f"{mode_str}_log.md")
        )

    return val_dict


#############################################
def train_full(main_loader, model, optimizer, output_dir=".", net_name=None, 
               dataset="UCF101", num_epochs=10, topk=TOPK, scheduler=None, 
               device="cuda", val_loader=None, seed=None, unexp_epoch=10, 
               log_freq=5, use_tb=False, save_by_batch=False, 
               reload_kwargs=dict()):
    """
    train_full(train_loader, model, optimizer)
    """

    dataset = dataset_3d.normalize_dataset_name(dataset)

    topk = loss_utils.check_topk(
        topk, num_classes=training_utils.get_num_classes(model)
        )

    model = model.to(device)

    log_idx, best_acc, start_epoch_n = \
        training_utils.load_checkpoint(model, optimizer, **reload_kwargs)
    num_classes = training_utils.get_num_classes(model)
    supervised = num_classes is not None
    loss_weight = training_utils.class_weight(dataset, supervised)

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
    if test:
        val_loader = main_loader
        loss_dict = dict()
        save_best = False
        use_tb = False
    else:
        if use_tb: 
            tb_dir = Path(output_dir, "tensorboard")
            writer_train, writer_val = misc_utils.init_tb_writers(
                tb_dir, val=save_best
                )
    
        # check whether number of epochs has been reached
        if start_epoch_n >= num_epochs + 1:
            logger.info(
                (f"Model already trained to epoch {start_epoch_n} "
                f"(> {num_epochs}).")
                )
            return

        # initialize loss dictionary or load, if it exists
        loss_dict = loss_utils.init_loss_dict(
            output_dir, ks=topk, supervised=supervised, 
            save_by_batch=save_by_batch
            )

    ### main loop ###
    data_seed = seed
    for epoch_n in range(start_epoch_n, num_epochs + 1):
        start_time = time.perf_counter()

        if dataset == "Gabors":
            # update dataset possizes, if applicable
            data_seed = update_dataset_possizes(
                main_loader, val_loader, seed=data_seed, incr=epoch_n
                )
            
            update_unexp(
                main_loader, val_loader, epoch_n=epoch_n, 
                unexp_epoch=unexp_epoch
                )
        
        if not test:
            train_dict, log_idx = train_epoch(
                main_loader, 
                model, 
                optimizer, 
                epoch_n=epoch_n, 
                num_epochs=num_epochs,
                writer=writer_train,
                log_idx=log_idx, 
                topk=topk,
                loss_weight=loss_weight,
                device=device, 
                log_freq=log_freq,
                save_by_batch=save_by_batch,
                )

            for key in train_dict.keys():
                if key in loss_dict["train"].keys():
                    loss_dict["train"][key].append(train_dict[key])

        if save_best or test:
            val_dict = val_or_test_epoch(
                val_loader, 
                model, 
                epoch_n=epoch_n, 
                num_epochs=num_epochs, 
                topk=topk,
                loss_weight=loss_weight,
                device=device,
                output_dir=output_dir,
                test=test,
                )
            
            if test:
                loss_dict = val_dict
                return

            for key in val_dict.keys():
                if key in loss_dict["val"].keys():
                    loss_dict["val"][key].append(val_dict[key])            

            if save_best:
                is_best = val_dict["acc"] > best_acc
                best_acc = max(val_dict["acc"], best_acc)

        if epoch_n != 0 and scheduler is not None:
            scheduler.step()

        # Save and log loss information
        chance = None if num_classes is None else 1 / num_classes
        loss_utils.save_loss_dict(
            loss_dict, output_dir=output_dir, seed=seed, dataset=dataset, 
            unexp_epoch=unexp_epoch, plot=True, chance=chance
            )

        if use_tb:
            misc_utils.update_tb(
                writer_train, train_dict, epoch_n=epoch_n, 
                writer_val=writer_val, val_dict=val_dict, ks=topk
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

        stop_time = time.perf_counter()
        time_str = misc_utils.format_time(stop_time - start_time, sep_min=True)
        logger.info(
            f"Epoch: [{epoch_n}/{num_epochs}] "
            f"Total time: {time_str}"
            )

        # close plots
        plt.close("all")
 
    logger.info(
        f"Training from epoch {start_epoch_n} to {num_epochs} finished"
        )

