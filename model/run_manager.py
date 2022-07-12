import logging
from pathlib import Path
import time

from matplotlib import pyplot as plt
import numpy as np
import torch

from dataset import gabor_stimuli
from utils import checkpoint_utils, gabor_utils, loss_utils, misc_utils, \
    training_utils

# a few global variables
TOPK = [1, 3, 5]

logger = logging.getLogger(__name__)

TAB = "    "

#############################################
def train_epoch(data_loader, model, optimizer, epoch_n=0, num_epochs=50, 
                log_idx=0, topk=TOPK, loss_weight=None, device="cpu", 
                log_freq=5, writer=None, train_off=False, output_dir=None, 
                save_by_batch=False):
    """
    train_epoch(data_loader, model, optimizer)
    """
    
    losses, topk_meters = loss_utils.init_meters(n_topk=len(topk))
    is_gabors = gabor_stimuli.check_if_is_gabors(data_loader.dataset)
        
    model = model.to(device)

    train, spacing = training_utils.set_model_train_mode(
        model, epoch_n, train_off
        )
    _, supervised = training_utils.get_num_classes_sup(model)

    train_dict = loss_utils.init_loss_dict(
        ks=topk, val=False, supervised=supervised, save_by_batch=save_by_batch
        )["train"]
    train_dict["epoch_n"] = epoch_n

    if is_gabors and save_by_batch:
        gabor_loss_dict, gabor_acc_dict, gabor_conf_mat = \
            gabor_utils.init_gabor_records(
                data_loader.dataset, init_conf_mat=(output_dir is not None)
                )
        
    criterion, criterion_no_reduction = loss_utils.get_criteria(
        loss_weight=loss_weight, device=device
        )

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

        # get targets, and reshape for loss calculation
        output_flattened, target_flattened, loss_reshape, target = \
            training_utils.prep_loss(
                output_, mask_, sup_target, input_seq_shape, 
                supervised=supervised, is_gabors=is_gabors
                )

        target_flattened = target_flattened.to(device)
        loss = criterion(output_flattened, target_flattened)
        
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # update meters, in-place
        losses.update(loss.item(), input_seq_shape[0])
        loss_utils.update_topk_meters(
            topk_meters, output_flattened, target_flattened, ks=topk, 
            main_shape=loss_reshape
            )

        if save_by_batch:
            batch_loss = criterion_no_reduction(
                output_flattened, target_flattened
                ).to("cpu").reshape(loss_reshape)

            if not supervised:
                # take mean across spatial (HW) dimension
                batch_loss = batch_loss.mean(axis=2) 
            
            training_utils.add_batch_data(
                train_dict, 
                dataset=data_loader.dataset, 
                batch_loss=losses.val, 
                batch_loss_by_item=batch_loss.to("cpu").tolist(), 
                sup_target=sup_target, 
                output=output_.to("cpu").tolist(), 
                target=target.to("cpu").tolist(), 
                epoch_n=epoch_n
                )

            if is_gabors:
                gabor_utils.update_records(
                    data_loader.dataset,
                    gabor_loss_dict, 
                    gabor_acc_dict,
                    output=output_flattened.detach().cpu().numpy(),
                    sup_target=sup_target.detach().cpu().numpy(),
                    batch_loss=batch_loss,
                    supervised=supervised,
                    confusion_mat=gabor_conf_mat,
                    )

        del output_, target, sup_target, target_flattened, output_flattened

        # record batch time
        stop_time = time.perf_counter()
        batch_time = stop_time - start_time
        batch_times.append(batch_time)

        # log results
        if idx % log_freq == 0 or idx == len(data_loader) - 1:
            loss_avg, acc_avg, loss_val, acc_val, stats_str = \
                loss_utils.get_stats(
                    losses, topk_meters, ks=topk, local=True, last=True
                )
            
            training_utils.log_epoch(
                stats_str, duration=np.mean(batch_times), epoch_n=epoch_n, 
                num_epochs=num_epochs, batch_idx=idx, 
                n_batches=len(data_loader), spacing=spacing
                )
            batch_times = []
            spacing = "" # for all but first log of the epoch

            if writer is not None:
                writer.add_scalar("local/loss", loss_val, log_idx)
                writer.add_scalar("local/accuracy", acc_val, log_idx)

            if train and supervised and optimizer:
                training_utils.log_weight_decay_prop(model) # log decay info
            
            log_idx += 1

    train_dict["loss"] = loss_avg
    train_dict["acc"]  = acc_avg
    for i, k in enumerate(topk):
        train_dict[f"top{k}"] = topk_meters[i].local_avg

    if is_gabors and save_by_batch:
        train_dict["gabor_loss_dict"] = gabor_loss_dict
        train_dict["gabor_acc_dict"] = gabor_acc_dict

        if gabor_conf_mat is not None:
            gabor_utils.plot_save_gabor_conf_mat(
                gabor_conf_mat, mode="train", epoch_n=epoch_n, 
                unexp=data_loader.dataset.unexp, 
                U_prob=data_loader.dataset.U_prob, 
                output_dir=output_dir
            ) 

    return train_dict, log_idx


#############################################
def val_or_test_epoch(data_loader, model, epoch_n=0, num_epochs=10, 
                      topk=TOPK, loss_weight=None, device="cpu", test=False, 
                      output_dir=None, save_by_batch=False):
    """
    val_or_test_epoch(data_loader, model)
    """

    losses, topk_meters = loss_utils.init_meters(n_topk=len(topk))
    is_gabors = gabor_stimuli.check_if_is_gabors(data_loader.dataset)

    model = model.to(device)
    model.eval()

    criterion, criterion_no_reduction = loss_utils.get_criteria(
        loss_weight=loss_weight, device=device
        )

    confusion_mat = None
    num_classes, supervised = training_utils.get_num_classes_sup(model)

    mode = "test" if test else "val"
    val_dict = loss_utils.init_loss_dict(
        ks=topk, val=True, supervised=supervised, save_by_batch=save_by_batch
        )["val"]
    val_dict["epoch_n"] = epoch_n

    if is_gabors and save_by_batch:
        gabor_loss_dict, gabor_acc_dict, gabor_conf_mat = \
            gabor_utils.init_gabor_records(
                data_loader.dataset, init_conf_mat=(output_dir is not None)
                )

    if supervised and output_dir is not None:
        if is_gabors:
            confusion_mat = gabor_utils.init_gabor_conf_mat(data_loader.dataset)
            plot_mat_kwargs = dict()
        else:
            class_names = list(data_loader.dataset.class_dict_encode.keys())
            confusion_mat = loss_utils.ConfusionMeter(class_names)
            # only include class names for final epoch 
            # (if there are many class labels, saving to svg is very slow)
            plot_mat_kwargs = {"incl_class_names": (epoch_n == num_epochs)}
        Path(output_dir).mkdir(exist_ok=True, parents=True)

    if not supervised:
        chance_level = loss_utils.AverageMeter()

    shared_pred, SUB_B = False, None
    start_time = time.perf_counter()
    with torch.no_grad():
        for idx, (input_seq, sup_target) in enumerate(data_loader):
            if supervised and len(input_seq.size()) == 7:
                shared_pred = True # applies to all batches
                input_seq, SUB_B = training_utils.resize_input_seq(input_seq)

            input_seq = input_seq.to(device)
            [output_, mask_] = model(input_seq)
            del input_seq

            # get targets, and reshape for loss calculation
            output_flattened, target_flattened, loss_reshape, target = \
                training_utils.prep_loss(
                    output_, mask_, sup_target, supervised=supervised, 
                    shared_pred=shared_pred, SUB_B=SUB_B, is_gabors=is_gabors
                    )

            target_flattened = target_flattened.to(device)
            loss = criterion(output_flattened, target_flattened)

            # collect some values
            losses.update(loss.item(), len(output_))
            loss_utils.update_topk_meters(
                topk_meters, output_flattened, target_flattened, ks=topk, 
                main_shape=loss_reshape
                )

            if not supervised:
                chance = loss_utils.calc_chance(output_flattened, loss_reshape)
                chance_level.update(chance, 1)

            if confusion_mat is not None:
                _, pred = torch.max(output_flattened, 1)
                confusion_mat.update(
                    pred.detach().cpu().numpy(), 
                    target_flattened.reshape(-1).byte().detach().cpu().numpy()
                    )

            if save_by_batch:
                batch_loss = criterion_no_reduction(
                    output_flattened, target_flattened
                    ).to("cpu").reshape(loss_reshape)

                if not supervised:
                    # take mean across spatial (HW) dimension
                    batch_loss = batch_loss.mean(axis=2)

                training_utils.add_batch_data(
                    val_dict, 
                    dataset=data_loader.dataset, 
                    batch_loss=losses.val, 
                    batch_loss_by_item=batch_loss.to("cpu").tolist(),
                    sup_target=sup_target, 
                    output=output_.to("cpu").tolist(), 
                    target=target.to("cpu").tolist(), 
                    epoch_n=epoch_n
                    )

                if is_gabors:
                    gabor_utils.update_records(
                        data_loader.dataset,
                        gabor_loss_dict, 
                        gabor_acc_dict,
                        output=output_flattened.detach().cpu().numpy(),
                        sup_target=sup_target.detach().cpu().numpy(),
                        batch_loss=batch_loss,
                        supervised=supervised,
                        confusion_mat=gabor_conf_mat,
                        )

        del output_, target, sup_target, target_flattened, output_flattened


    chance = 1 / num_classes if supervised else chance_level.avg
    loss_avg, acc_avg, stats_str = loss_utils.get_stats(
            losses, topk_meters, ks=topk, local=False, chance=chance
        )

    stop_time = time.perf_counter()
    duration = stop_time - start_time

    training_utils.log_epoch(
        stats_str, duration=duration, epoch_n=epoch_n, num_epochs=num_epochs, 
        val=True, test=test, spacing="\n"
        )

    val_dict["loss"] = loss_avg
    val_dict["acc"]  = acc_avg
    for i, k in enumerate(topk):
        val_dict[f"top{k}"] = topk_meters[i].avg
    
    if confusion_mat is not None:
        confusion_mat.plot_mat(
            Path(output_dir, f"{mode}_confusion_matrix.svg"), 
            title=f"Epoch {epoch_n} ({mode})",
            **plot_mat_kwargs
            )
        val_dict["confusion_matrix"] = confusion_mat.get_storage_dict()

    if is_gabors and save_by_batch:
        val_dict["gabor_loss_dict"] = gabor_loss_dict
        val_dict["gabor_acc_dict"] = gabor_acc_dict
        if gabor_conf_mat is not None:
            gabor_utils.plot_save_gabor_conf_mat(
                gabor_conf_mat, mode=mode, epoch_n=epoch_n, 
                unexp=data_loader.dataset.unexp, 
                U_prob=data_loader.dataset.U_prob, 
                output_dir=output_dir
            ) 

    if output_dir is not None:
        training_utils.write_log(
            content=stats_str,
            epoch_n=epoch_n,
            filename=Path(output_dir, f"{mode}_log.md")
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

    dataset = misc_utils.normalize_dataset_name(dataset)
    is_gabors = gabor_stimuli.check_if_is_gabors(main_loader.dataset)
    
    topk = loss_utils.check_topk(
        topk, num_classes=training_utils.get_num_classes_sup(model)[0]
        )

    model = model.to(device)

    log_idx, best_acc, start_epoch_n = \
        checkpoint_utils.load_checkpoint(model, optimizer, **reload_kwargs)
    num_classes, supervised = training_utils.get_num_classes_sup(model)
    if is_gabors and supervised:
        gabor_utils.warn_supervised(main_loader.dataset)

    loss_weight = training_utils.class_weight(dataset, supervised)

    model_direc = misc_utils.init_model_direc(output_dir)

    # check val/test parameters
    writer_train, writer_val = None, None
    test = reload_kwargs["test"] if "test" in reload_kwargs.keys() else False
    if test:
        val_loader = main_loader
        save_best = False
    else:
        save_best = False if val_loader is None else True
        is_best = False
        best_acc = loss_utils.get_best_acc(best_acc, save_best=save_best)

        if use_tb: 
            writer_train, writer_val = misc_utils.init_tb_writers(
                Path(output_dir, "tensorboard"), val=save_best
                )
    
        # check whether number of epochs has been reached
        if training_utils.check_end(start_epoch_n, num_epochs):
            return

        # initialize loss dictionary or load, if it exists
        loss_dict = loss_utils.init_loss_dict(
            output_dir, ks=topk, val=save_best, supervised=supervised, 
            save_by_batch=save_by_batch
            )
        
        if save_by_batch and is_gabors:
            for key in loss_dict.keys():
                loader = main_loader if key == "train" else val_loader
                gabor_dict = gabor_utils.init_gabor_records(loader.dataset)[0]
                for gabor_key, sub_dict in gabor_dict.items():
                    loss_dict[key][gabor_key] = sub_dict

    ### main loop ###
    data_seed = seed
    for epoch_n in range(start_epoch_n, num_epochs + 1):
        start_time = time.perf_counter()

        if dataset == "Gabors":
            data_seed = gabor_utils.update_gabors(
                main_loader, val_loader, seed=data_seed, epoch_n=epoch_n, 
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
                output_dir=output_dir,
                save_by_batch=save_by_batch,
                )

            loss_utils.populate_loss_dict(
                src_dict=train_dict, target_dict=loss_dict["train"]
                )

        if test or save_best:
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
                save_by_batch=save_by_batch,
                )
            
            if test:
                # store confusion matrix data, if applicable, then all done
                if "confusion_matrix" in val_dict.keys():
                    loss_utils.save_confusion_mat_dict(
                        val_dict["confusion_matrix"], prefix="test", 
                        output_dir=output_dir, overwrite=True
                        )
                return

            if save_best:
                is_best = val_dict["acc"] > best_acc
                best_acc = max(val_dict["acc"], best_acc)
                if is_best and "confusion_matrix" in val_dict.keys():
                    # plot best confusion matrix
                    loss_utils.load_replot_conf_mat(
                        val_dict["confusion_matrix"], mode="val", suffix="best",
                        omit_class_names=True, epoch_n=epoch_n, 
                        output_dir=output_dir
                        )

            loss_utils.populate_loss_dict(
                src_dict=val_dict, 
                target_dict=loss_dict["val"],
                append_confusion_matrices=True,
                is_best=is_best
                )


        if epoch_n != 0 and scheduler is not None:
            scheduler.step()

        # Save and log loss information
        loss_utils.save_loss_dict(
            loss_dict, output_dir=output_dir, seed=seed, dataset=dataset, 
            unexp_epoch=unexp_epoch, num_classes=num_classes, plot=True
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
        checkpoint_utils.save_checkpoint(
            {
            "epoch_n": epoch_n,
            "net": net_name,
            "state_dict": model.state_dict(),
            "best_acc": best_acc,
            "optimizer": optimizer.state_dict(),
            "log_idx": log_idx
            }, 
            is_best, 
            filename=Path(model_direc, f"epoch{epoch_n}.pth.tar"), 
            keep_all=False
        )

        stop_time = time.perf_counter()
        duration = stop_time - start_time
        time_str = misc_utils.format_time(duration, sep_min=True, lead=False)
        logger.info(
            f"Epoch: [{epoch_n}/{num_epochs}] "
            f"Total time: {time_str}"
            )

        # close plots
        plt.close("all")
 
    # replot the best confusion matrix, with class names (can take a long time)
    if "confusion_matrix_best" in loss_dict["val"].keys():
        best_idx = np.argmax(loss_dict["val"]["acc"])
        best_epoch_n = loss_dict["val"]["epoch_n"][best_idx]
        loss_utils.load_replot_conf_mat(
            loss_dict["val"]["confusion_matrix_best"], mode="val", 
            suffix="best", omit_class_names=False, epoch_n=best_epoch_n, 
            output_dir=output_dir
            )

    logger.info(
        f"Training from epoch {start_epoch_n} to {num_epochs} finished."
        )

