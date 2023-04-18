import copy
import logging
from pathlib import Path
import time
from tqdm import tqdm

from matplotlib import pyplot as plt
import numpy as np
import torch

from dataset import gabor_sequences
from utils import checkpoint_utils, gabor_utils, loss_utils, misc_utils, \
    training_utils
from analysis import hook_analysis

# a few global variables
TOPK = [1, 3, 5]

logger = logging.getLogger(__name__)

TAB = "    "


#############################################
def train_epoch(dataloader, model, optimizer, epoch_n=0, num_epochs=50, 
                topk=TOPK, loss_weights=None, device="cpu", log_freq=5, 
                writer=None, log_idx=0, train_off=False, output_dir=None, 
                save_by_batch=False):
    """
    train_epoch(dataloader, model, optimizer)

    Runs one epoch of training on the model with the dataloader provided.

    If the dataset is a Gabor dataset and an output directory is provided, 
    confusion matrices for each epoch are generated and saved, whether the task 
    is supervised or not.

    Required args
    -------------
    - dataloader : torch data.DataLoader
        Torch dataloader to use for model training.
    - model : torch nn.Module or nn.DataParallel
        Dense CPC (DPC) or linear classification (LC) model to train.
    - optimizer : torch.optim object
        Torch optimizer.

    Optional args
    -------------
    - epoch_n : int (default=0)
        Current epoch number, used primarily for logging and storing progress. 
        If epoch_n is 0, the data is passed through the model without training, 
        in order to obtain a baseline.
    - num_epochs : int (default=50)
        Total number of training epochs, used for logging.
    - topk : list (default=TOPK)
        The top k accuracies to record (e.g., [1, 3, 5] for top 1, 3 and 5).
    - loss_weights : tuple (default=None)
        Class weights to provide to the loss function.
    - device : torch.device or str (default="cpu")
        Device on which to train the model.
    - log_freq : int (default=5)
        Batch frequency at which to log progress to the console and, 
        optionally, the writer.
    - writer : tensorboardX.SummaryWriter (default=None)
    - log_idx : int (default=0)
        Log index, used for writing to tensorboard, if writer is not None.
    - train_off : bool (default=False)
        If True, no training is done for this epoch 
        (e.g., to obtain a baseline).
    - output_dir : str or path (default=None)
        Output directory in which to save confusion matrices, if applicable. 
        If None, confusion matrices are not generated.
    - save_by_batch : bool (default=False)
        If True, loss and accuracy data is saved for each batch.
    
    Returns
    -------
    - train_dict : dict
        Dictionary recording training loss and accuracy data, with keys:

        'acc' (float)   : Final local accuracy average.
        'epoch_n' (int) : Epoch number.
        'loss' (float)  : Final local loss average.
        'top{k}' (float): Final local top k accuracy average.
        
        if save_by_batch:
        'avg_loss_by_batch' (list)          : average loss values for each batch
        'batch_epoch_n' (list)              : epoch number for each batch
        'loss_by_item' (3D list)            : loss values with dims: 
            batches x B x N
        'sup_target_by_batch' (3 or 5D list): supervised targets for each batch, 
            with dims: B x N (x L x [image type, mean ori] if Gabor 
            dataset).
        
        if Gabor dataset:
        'unexp'                 : Unexpected dataloader value for the epoch. 
        'gabor_loss_dict' (dict):  Gabor loss dictionary, with keys
            '{image_type}' (list)       : image type loss, for each batch
            '{mean ori}' (list)         : orientation loss, for each batch
            'image_types_overall' (list): overall image type loss, for each 
                                          batch
            'mean_oris_overall'   (list): overall mean ori loss, for each batch
            'overall'             (list): overall loss, for each batch
        'gabor_acc_dict' (dict) : Gabor top 1 accuracy dictionary, with the 
                                  same keys as 'gabor_loss_dict'.

    - log_idx : int
        Updated log index, for writing to tensorboard.
    """
    
    losses, topk_meters = loss_utils.init_meters(n_topk=len(topk))
    is_gabor = gabor_sequences.check_if_is_gabor(dataloader.dataset)
        
    model = model.to(device)
    model.train()

    train, spacing = training_utils.set_model_train_mode(
        model, epoch_n, train_off
        )
    _, supervised = training_utils.get_num_classes_sup(model)

    train_dict = loss_utils.init_loss_dict(
        dataloader.dataset, ks=topk, val=False, supervised=supervised, 
        save_by_batch=save_by_batch, 
        )["train"]
    train_dict["epoch_n"] = epoch_n

    if is_gabor:
        train_dict["unexp"] = dataloader.dataset.unexp

    if is_gabor and save_by_batch:
        gabor_conf_mat = None     
        if output_dir is not None:
            gabor_conf_mat = gabor_utils.init_gabor_conf_mat(dataloader.dataset)
        
    criterion, criterion_no_reduction = loss_utils.get_criteria(
        loss_weights=loss_weights, device=device
        )

    batch_times = []
    for idx, (input_seq, sup_target) in enumerate(dataloader):
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
                supervised=supervised, is_gabor=is_gabor
                )

        target_flattened = target_flattened.to(device)
        loss = criterion(output_flattened, target_flattened)
        
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.zero_grad()

        # update meters, in-place
        losses.update(loss.item(), input_seq_shape[0])
        loss_utils.update_topk_meters(
            topk_meters, output_flattened, target_flattened, ks=topk, 
            main_shape=loss_reshape
            )

        if save_by_batch:
            batch_loss = criterion_no_reduction(
                output_flattened, target_flattened
                ).reshape(loss_reshape)

            if not supervised:
                # take mean across spatial (D2) dimension
                batch_loss = batch_loss.mean(axis=2) 
            
            training_utils.add_batch_data(
                train_dict, 
                dataset=dataloader.dataset, 
                batch_loss=losses.val, 
                batch_loss_by_item=batch_loss.detach().cpu().tolist(), 
                sup_target=sup_target, 
                output=output_.detach().cpu().tolist(), 
                target=target.detach().cpu().tolist(), 
                epoch_n=epoch_n
                )

            if is_gabor:
                gabor_utils.update_records(
                    dataloader.dataset,
                    train_dict["gabor_loss_dict"], 
                    train_dict["gabor_acc_dict"],
                    output=output_flattened.detach().cpu(),
                    sup_target=sup_target.detach().cpu(),
                    batch_loss=batch_loss.detach().cpu(),
                    supervised=supervised,
                    confusion_mat=gabor_conf_mat,
                    )

        del output_, target, sup_target, target_flattened, output_flattened

        # record batch time
        stop_time = time.perf_counter()
        batch_time = stop_time - start_time
        batch_times.append(batch_time)

        # log results
        if idx % log_freq == 0 or idx == len(dataloader) - 1:
            tail = "\n" if idx == len(dataloader) - 1 else ""
            loss_avg, acc_avg, stats_str, loss_val, acc_val = \
                loss_utils.get_stats(
                    losses, topk_meters, ks=topk, local=True, incl_last=True
                )
            
            training_utils.log_epoch(
                stats_str, duration=np.mean(batch_times), epoch_n=epoch_n, 
                num_epochs=num_epochs, batch_idx=idx, 
                n_batches=len(dataloader), spacing=spacing, tail=tail
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

    if is_gabor and gabor_conf_mat is not None:
        gabor_utils.plot_save_gabor_conf_mat(
            gabor_conf_mat, mode="train", epoch_n=epoch_n, 
            output_dir=output_dir, skip_plot=True
        ) 

    return train_dict, log_idx


#############################################
def eval_epoch(dataloader, model, epoch_n=0, num_epochs=50, topk=TOPK, 
               loss_weights=None, device="cpu", test_suffix=False, 
               output_dir=None, save_by_batch=False):
    """
    eval_epoch(dataloader, model)

    Runs one epoch of evaluation on the model with the dataloader provided.

    If an output directory is provided and the model is run in the supervised 
    mode, the latest confusion matrix is saved. 
    
    Aditionally, if the dataset is a Gabor dataset, confusion matrices for each 
    epoch are generated and saved, whether the task is supervised or not.

    Required args
    -------------
    - dataloader : torch data.DataLoader
        Torch dataloader to for model evaluation.
    - model : torch nn.Module or nn.DataParallel
        Dense CPC (DPC) or linear classification (LC) model to train.

    Optional args
    -------------
    - epoch_n : int (default=0)
        Current epoch number, used primarily for logging and storing progress. 
        If epoch_n is 0, the data is passed through the model without training, 
        in order to obtain a baseline.
    - num_epochs : int (default=50)
        Total number of training epochs, used for logging.
    - topk : int (default=TOPK)
        The top k accuracies to record (e.g., [1, 3, 5] for top 1, 3 and 5).
    - loss_weights : tuple (default=None)
        Class weights to provide to the loss function.
    - device : torch.device or str (default="cpu")
        Device on which to train the model.
    - test_suffix : str (default=None)
        Suffix to use for test logs. Must be provided if mode is test. Also 
        used for logging.
    - output_dir : str or path (default=None)
        Output directory in which to save confusion matrices, if applicable. 
        If None, confusion matrices are not generated.
    - save_by_batch : bool (default=False)
        If True, loss and accuracy data is saved for each batch.
    
    Returns
    -------
    - eval_dict : dict
        Dictionary recording validation or test loss and accuracy data, with 
        keys:

        'acc' (float)   : Overall accuracy average.
        'epoch_n' (int) : Epoch number.
        'loss' (float)  : Overall loss average.
        'top{k}' (float): Overall top k accuracy average.
        
        if save_by_batch:
        'avg_loss_by_batch' (list)          : average loss values for each batch
        'batch_epoch_n' (list)              : epoch number for each batch
        'loss_by_item' (3D list)            : loss values with dims: 
            batches x B x N
        'sup_target_by_batch' (3 or 5D list): supervised targets for each batch, 
            with dims: B x N (x L x [image type, mean ori] if Gabor 
            dataset).
        
        if output_dir is not None:
        'confusion_mat' (dict): confusion matrix storage dictionary for the 
                                   epoch

        if Gabor dataset:
        'unexp'                 : Unexpected dataloader value for the epoch. 
        'gabor_loss_dict' (dict):  Gabor loss dictionary, with keys
            '{image_type}' (list)       : image type loss, for each batch
            '{mean ori}' (list)         : orientation loss, for each batch
            'image_types_overall' (list): overall image type loss, for each 
                                          batch
            'mean_oris_overall'   (list): overall mean ori loss, for each batch
            'overall'             (list): overall loss, for each batch
        'gabor_acc_dict' (dict) : Gabor top 1 accuracy dictionary, with the 
                                  same keys as 'gabor_loss_dict'.
    """

    losses, topk_meters = loss_utils.init_meters(n_topk=len(topk))
    is_gabor = gabor_sequences.check_if_is_gabor(dataloader.dataset)

    model = model.to(device)
    model.eval()

    criterion, criterion_no_reduction = loss_utils.get_criteria(
        loss_weights=loss_weights, device=device
        )

    confusion_mat = None
    num_classes, supervised = training_utils.get_num_classes_sup(model)

    test = test_suffix is not None
    test_suffix = misc_utils.format_addendum(test_suffix, is_suffix=True)
    mode = "test" if test else "val"
    
    eval_dict = loss_utils.init_loss_dict(
        dataloader.dataset, ks=topk, val=True, supervised=supervised, 
        save_by_batch=save_by_batch,
        )["val"]
    eval_dict["epoch_n"] = epoch_n

    gab_unexp_str = ""
    if is_gabor:
        unexp = dataloader.dataset.unexp
        eval_dict["unexp"] = unexp
        gab_unexp_str = "_unexp" if unexp else ""
        gabor_conf_mat = None     
        if save_by_batch and output_dir is not None:
            gabor_conf_mat = gabor_utils.init_gabor_conf_mat(dataloader.dataset)

    if supervised and output_dir is not None:
        if is_gabor:
            confusion_mat = gabor_utils.init_gabor_conf_mat(dataloader.dataset)
        else:
            class_names = list(dataloader.dataset.class_dict_encode.keys())
            confusion_mat = loss_utils.ConfusionMeter(class_names)
        Path(output_dir).mkdir(exist_ok=True, parents=True)

    if not supervised:
        chance_level = loss_utils.AverageMeter()
    
    shared_pred, SUB_B = False, None
    start_time = time.perf_counter()
    with torch.no_grad():
        for idx, (input_seq, sup_target) in tqdm(
            enumerate(dataloader), total=len(dataloader)
            ):
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
                    shared_pred=shared_pred, SUB_B=SUB_B, is_gabor=is_gabor
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
                    pred.detach().cpu(), 
                    target_flattened.reshape(-1).byte().detach().cpu()
                    )

            if save_by_batch:
                batch_loss = criterion_no_reduction(
                    output_flattened, target_flattened
                    ).reshape(loss_reshape)

                if not supervised:
                    # take mean across spatial (D2) dimension
                    batch_loss = batch_loss.mean(axis=2)

                training_utils.add_batch_data(
                    eval_dict, 
                    dataset=dataloader.dataset, 
                    batch_loss=losses.val, 
                    batch_loss_by_item=batch_loss.detach().cpu().tolist(),
                    sup_target=sup_target, 
                    output=output_.detach().cpu().tolist(), 
                    target=target.detach().cpu().tolist(), 
                    epoch_n=epoch_n
                    )

                if is_gabor:
                    gabor_utils.update_records(
                        dataloader.dataset,
                        eval_dict["gabor_loss_dict"], 
                        eval_dict["gabor_acc_dict"],
                        output=output_flattened.detach().cpu(),
                        sup_target=sup_target.detach().cpu(),
                        batch_loss=batch_loss.detach().cpu(),
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
        val=True, test=test, spacing=""
        )

    eval_dict["loss"] = loss_avg
    eval_dict["acc"]  = acc_avg
    for i, k in enumerate(topk):
        eval_dict[f"top{k}"] = topk_meters[i].avg
    
    if confusion_mat is not None:
        eval_dict["confusion_mat"] = confusion_mat.get_storage_dict()
        # plot only if test or last epoch (slow, especially if many classes)
        if test or epoch_n == num_epochs: 
            loss_utils.plot_conf_mat(
                confusion_mat, mode=mode, suffix=test_suffix, 
                epoch_n=epoch_n, output_dir=output_dir
                )

    if is_gabor and gabor_conf_mat is not None:
        gabor_utils.plot_save_gabor_conf_mat(
            gabor_conf_mat, mode=mode, epoch_n=epoch_n, 
            output_dir=output_dir, skip_plot=not(test)
        )

    if output_dir is not None:
        overwrite = True if test else False
        suffix = ""
        if test:
            suffix = f"{test_suffix}{gab_unexp_str}"

        training_utils.write_log(
            stats_str=stats_str,
            epoch_n=epoch_n,
            output_dir=output_dir,
            filename=f"{mode}_log{suffix}.md",
            overwrite=overwrite
            )


    return eval_dict


#############################################
def train_full(main_loader, model, optimizer, output_dir=".", net_name=None, 
               dataset="UCF101", num_epochs=10, topk=TOPK, scheduler=None, 
               device="cuda", val_loader=None, seed=None, unexp_epoch=10, 
               log_freq=5, use_tb=False, save_by_batch=False, 
               run_gabor_analysis=False, reload_kwargs=dict()):
    """
    train_full(train_loader, model, optimizer)

    Trains and evaluates a model.

    If the model is evaluated in supervised mode, via validation or test, a 
    confusion matrix is saved for the best and test run, respectively.

    Required args
    -------------
    - main_loader : torch data.DataLoader
        Main Torch dataloader to use for model training, or evaluation 
        if "test" key is a provided in reload_kwargs.
    - model : torch nn.Module or nn.DataParallel
        Dense CPC (DPC) or linear classification (LC) model to train.
    - optimizer : torch.optim object
        Torch optimizer.

    Optional args
    -------------
    - output_dir : str or path (default=".")
        Output directory in which to save training and evalution records.
    - net_name : str (default=None)
        Network name, saved in the model checkpoint.
    - dataset : str (default="UCF101")
        Dataset name, used to set some parameters.
    - num_epochs : int (default=50)
        Total number of epochs to train the model on. Ignored if test mode is 
        identified.
    - topk : list (default=TOPK)
        The top k accuracies to record (e.g., [1, 3, 5] for top 1, 3 and 5).
    - scheduler : torch optim.lr_scheduler object (default=None)
        Torch learning rate scheduler.
    - device : torch.device or str (default="cpu")
        Device on which to train the model.
    - val_loader : torch data.DataLoader (default=None)
        Validation Torch dataloader to use for evaluating model, and 
        identifying and recording the one the yields the best accuracy. 
        If None, no evaluation is performed. 
    - seed : int (default=None)
        Seed, for inclusion in the records. Also used to deterministically 
        update the positions and sizes attribute for the Gabors dataset, if 
        applicable.
    - unexp_epoch : int (default=10)
        Epoch as of which unexpected sequences are introduced, if the dataset 
        is a Gabors dataset.
    - log_freq : int (default=5)
        Batch frequency at which to log training progress to the console and, 
        optionally, the training writer.
    - use_tb : bool (default=False)
        If True, tensorboard is used.
    - save_by_batch : bool (default=False)
        If True, loss and accuracy data is saved for each batch.
    - reload_kwargs : dict (default=dict())
        Dictionary with keys for reloading checkpointed parameters into the 
        model (see checkpoint_utils.load_checkpoint()).
    """

    dataset = misc_utils.normalize_dataset_name(dataset)
    is_gabor = gabor_sequences.check_if_is_gabor(main_loader.dataset)
    
    topk = loss_utils.check_topk(
        topk, num_classes=training_utils.get_num_classes_sup(model)[0]
        )

    model = model.to(device)

    log_idx, best_acc, start_epoch_n, test_suffix, gabor_unexp = \
        checkpoint_utils.load_checkpoint(model, optimizer, **reload_kwargs)

    num_classes, supervised = training_utils.get_num_classes_sup(model)
    
    if is_gabor and supervised:
        gabor_utils.warn_supervised(main_loader.dataset)

    loss_weights = training_utils.class_weights(dataset, supervised)

    model_direc = misc_utils.init_model_direc(output_dir)

    # check val/test parameters
    writer_train, writer_val = None, None
    test = reload_kwargs["test"] if "test" in reload_kwargs.keys() else False
    if test:
        save_best = False
    else:
        save_best = False if val_loader is None else True
        is_best = False
        if is_gabor:
            best_accs = gabor_utils.get_best_acc(best_acc, save_best)
            del best_acc
        else:
            best_acc = loss_utils.get_best_acc(best_acc, save_best)

        if use_tb: 
            writer_train, writer_val = misc_utils.init_tb_writers(
                Path(output_dir, "tensorboard"), val=save_best
                )
    
        # check whether number of epochs has been reached
        if training_utils.check_end(start_epoch_n, num_epochs):
            return

        # initialize loss dictionary or load, if it exists
        loss_dict = loss_utils.init_loss_dict(
            main_loader.dataset, output_dir, ks=topk, val=save_best, 
            supervised=supervised, save_by_batch=save_by_batch, 
            )

    ### main loop ###
    data_seed = seed
    gab_unexp_str = ""
    for epoch_n in range(start_epoch_n, num_epochs + 1):
        start_time = time.perf_counter()

        gabor_epoch_str = None
        if is_gabor:
            data_seed, gabor_epoch_str = gabor_utils.update_gabors(
                main_loader, val_loader, seed=data_seed, epoch_n=epoch_n, 
                unexp_epoch=unexp_epoch, test=test, num_pre_post=3
                )
            gabor_unexp = main_loader.dataset.unexp
            gab_unexp_str = "_unexp" if gabor_unexp else ""
            if not test:
                best_acc = best_accs[int(gabor_unexp)]
            
            ep_run_analysis = (
                run_gabor_analysis and not supervised and 
                gabor_epoch_str is not None
                )
            if ep_run_analysis: # run analysis
                hook_analysis.collect_save_hook_data(
                    model, main_loader, optimizer, device=device, 
                    output_dir=output_dir, epoch_str=gabor_epoch_str, pre=True
                    )

        if not test:
            train_dict, log_idx = train_epoch(
                main_loader, 
                model, 
                optimizer, 
                epoch_n=epoch_n, 
                num_epochs=num_epochs,
                topk=topk,
                loss_weights=loss_weights,
                device=device, 
                log_freq=log_freq,
                writer=writer_train,
                log_idx=log_idx, 
                output_dir=output_dir,
                save_by_batch=save_by_batch,
                )

            loss_utils.populate_loss_dict(
                src_dict=train_dict, target_dict=loss_dict["train"]
                )

        if is_gabor and ep_run_analysis:
                hook_analysis.collect_save_hook_data(
                    model, main_loader, None, device=device, 
                    output_dir=output_dir, epoch_str=gabor_epoch_str, pre=False
                    )            

        if test or save_best:
            eval_loader = main_loader if test else val_loader
            val_dict = eval_epoch(
                eval_loader, 
                model, 
                epoch_n=epoch_n, 
                num_epochs=num_epochs, 
                topk=topk,
                loss_weights=loss_weights,
                device=device,
                output_dir=output_dir,
                test_suffix=test_suffix,
                save_by_batch=save_by_batch,
                )
            
            if test:
                conf_mat_suffix = f"{test_suffix}{gab_unexp_str}"
                # store confusion matrix data, if applicable, then all done
                if "confusion_mat" in val_dict.keys():
                    loss_utils.save_conf_mat_dict(
                        val_dict["confusion_mat"], prefix="test", 
                        suffix=conf_mat_suffix, output_dir=output_dir, 
                        overwrite=True
                        )
                return

            if save_best:
                is_best = val_dict["acc"] > best_acc
                best_acc = max(val_dict["acc"], best_acc)
                if is_gabor:
                    best_accs[int(gabor_unexp)] = best_acc

            loss_utils.populate_loss_dict(
                src_dict=val_dict, 
                target_dict=loss_dict["val"],
                append_conf_matrices=True,
                is_best=is_best,
                gabor_unexp=gabor_unexp,
                )


        if epoch_n != 0 and scheduler is not None:
            scheduler.step()

        # Save and log loss information
        loss_utils.save_loss_dict(
            loss_dict, output_dir=output_dir, seed=seed, dataset=dataset, 
            num_classes=num_classes, plot=True, log_best=(epoch_n == num_epochs)
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
        state_dict = {
            "epoch_n": epoch_n,
            "net": net_name,
            "state_dict": model.state_dict(),
            "best_acc": best_acc,
            "optimizer": optimizer.state_dict(),
            "log_idx": log_idx            
        }
        if is_gabor:
            state_dict["best_acc"] = best_accs
            state_dict["data_seed"] = data_seed + epoch_n

        checkpoint_utils.save_checkpoint(
            state_dict, is_best, 
            output_dir=model_direc,
            gabor_unexp=gabor_unexp,
            keep_all=False,
            suffix=gabor_epoch_str,
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

    # plot the best confusion matrix, with class names (can take a long time)
    loss_utils.plot_best_conf_mat(
        loss_dict["val"], mode="val", omit_class_names=False, 
        output_dir=output_dir
        )
    
    # plot Gabor confusion matrices by epoch, in parallel, if possible
    if is_gabor:
        parallel = False
        if main_loader.num_workers is not None and main_loader.num_workers > 1:
            parallel = True
        gabor_utils.load_plot_gabor_conf_mat(
            gabor_utils.get_gabor_conf_mat_dict_path(output_dir), 
            parallel=parallel
            )


    logger.info(
        f"Training from epoch {start_epoch_n} to {num_epochs} finished."
        )

