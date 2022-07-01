from datetime import datetime
import logging
import multiprocessing
import os
from pathlib import Path
import warnings

import numpy as np
import torch

from dataset import dataset_3d, gabor_stimuli
from utils import misc_utils

logger = logging.getLogger(__name__)

TAB = "    "

#############################################
def get_num_jobs(max_n=None, min_n=1):
    """
    get_num_jobs()

    Get number of jobs to run in parallel.
    """
    
    num_jobs = multiprocessing.cpu_count()

    if max_n is None:
        max_n = num_jobs

    os_max_n = os.getenv("OMP_NUM_THREADS")
    if os_max_n is None:
        os_max_n = num_jobs
    else:
        os_max_n = int(os_max_n) 

    num_jobs = min(num_jobs, max_n, os_max_n)
    num_jobs = int(max(num_jobs, min_n))

    return num_jobs


#############################################
def get_device(num_workers=None, cpu_only=False):
    """
    get_device()
    """

    device_name = "cuda" if torch.cuda.is_available() and not cpu_only else "cpu"
    device = torch.device(device_name)

    if num_workers is None:
        num_cores = multiprocessing.cpu_count()
        num_devices = torch.cuda.device_count()
        num_workers = min(num_devices * 2, num_cores) # approx rule of thumb
    
    return device, num_workers


#############################################
def allow_data_parallel(data_loader, device="cpu", supervised=False):
    """
    allow_data_parallel(data_loader, device="cpu")
    """
    
    allow_parallel = False
    if device.type != "cpu" and torch.cuda.device_count() > 1:
        if supervised:
            allow_parallel = True
        else:
            if data_loader.drop_last:
                allow_parallel = True
            else:
                warnings.warn(
                    "Only 1 GPU will be used, as the data loader is not set "
                    "to drop final batches, when they are smaller."
                    )

    return allow_parallel      


#############################################
def log_weight_decay_prop(model):
    """
    log_weight_decay_prop(model)
    """
    
    total_weight = 0.0
    decay_weight = 0.0
    for m in model.parameters():
        if m.requires_grad: 
            decay_weight += m.norm(2).data
        total_weight += m.norm(2).data
    perc = decay_weight / total_weight * 100
    logger.info(
        "Norm of weights to decay: "
        f"{decay_weight:.2f} / {total_weight:.2f} ({perc:05.2f}% of total)", 
        extra={"spacing": TAB}
        )


#############################################
def get_target_from_mask(mask):
    """
    get_target_from_mask(mask)
    
    Task mask as input, compute the target for contrastive loss
    """
    
    # mask meaning: 1: pos
    target = (mask == 1).to(int)
    target.requires_grad = False

    return target


#############################################
def check_grad(model):
    """
    check_grad(model)
    """

    model = get_model(model)
    
    logger.debug("\n===========Check Grad============")
    for name, param in model.named_parameters():
        logger.debug(name, param.requires_grad)
    logger.debug("=================================\n")


#############################################
def get_multistepLR_restart_multiplier(epoch, gamma=0.1, step=[10, 15, 20], 
                                       repeat=3):
    """
    get_multistepLR_restart_multiplier(epoch)

    Returns the multipier for LambdaLR, 
    0  <= ep < 10: gamma^0
    10 <= ep < 15: gamma^1 
    15 <= ep < 20: gamma^2
    20 <= ep < 30: gamma^0 ... repeat 3 cycles and then keep gamma^2
    """

    max_step = max(step)
    effective_epoch = epoch % max_step
    if epoch // max_step >= repeat:
        exp = len(step) - 1
    else:
        exp = len([i for i in step if effective_epoch >= i])
    return gamma ** exp


#############################################
def get_lr_lambda(dataset="UCF101", img_dim=224): 
    """
    get_lr_lambda()
    """
    
    dataset = dataset_3d.normalize_dataset_name(dataset)

    big = img_dim >= 224

    if dataset == "HMDB51":
        steps = [150, 250, 300] if big else [30, 40, 50]
        
    elif dataset == "UCF101":
        steps = [300, 400, 500] if big else [60, 80, 100]

    elif dataset == "Kinetics400":
        steps = [150, 250, 300] if big else [30, 40, 50]

    elif dataset == "MouseSim":
        steps = [300, 400, 500] if big else [60, 80, 100]


    else:
        raise ValueError(f"{dataset} dataset not recognized.")

    lr_lambda = lambda ep: get_multistepLR_restart_multiplier(
        ep, gamma=0.1, step=steps, repeat=1
        )

    return lr_lambda


#############################################
def get_model(model):
    """
    get_model(model)
    """
    
    # in case the model is wrapped with DataParallel()
    if hasattr(model, "device_ids"):
        model = model.module

    return model


#############################################
def get_num_classes_sup(model):
    """
    get_num_classes_sup(model)
    """
    
    model = get_model(model)
    num_classes = None
    
    supervised = hasattr(model, "num_classes")
    if supervised:
        num_classes = model.num_classes
    
    return num_classes, supervised


#############################################
def class_weight(dataset="MouseSim", supervised=True):

    weight = None
    dataset = dataset_3d.normalize_dataset_name(dataset)
    if supervised and dataset == "MouseSim":
        weight = [6, 1]
        logger.warning(
            f"Loss will be weighted per class as follows: {weight}"
            )

        weight = torch.Tensor(weight) # based on training set

    return weight


#############################################
def check_end(start_epoch_n=0, num_epochs=50):
    """
    check_end()
    """
    
    end = False
    if start_epoch_n >= num_epochs + 1:
        logger.info(
            (f"Model already trained to epoch {start_epoch_n} "
            f"(> {num_epochs}).")
            )
        end = True
    
    return end


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
def set_model_train_mode(model, epoch_n=0, train_off=False):
    """
    set_model_train_mode(model)
    """

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

    return train, spacing


#############################################
def prep_supervised_loss(output, target, shared_pred=False, SUB_B=None):
    """
    prep_supervised_loss(output, target)
    """
    
    if shared_pred:
        if SUB_B is None:
            raise ValueError("Must pass 'SUB_B' if shared_pred is True.")
        B_comb, N, num_classes = output.size()
        B = B_comb // SUB_B
        USE_N = SUB_B * N

        # group sequences that share a label
        output_flattened = output.reshape(B, USE_N, num_classes)

        # for each batch item, average the softmaxed class 
        # predictions across sequences
        output_flattened = torch.mean(
            torch.nn.functional.softmax(output_flattened, 2),
            1) # B x num_classes
        target_flattened = target.reshape(-1)
        loss_reshape = (B, )

    else:
        # consider all sequences separately, even if they share a label 
        B, N, num_classes = output.size()
        output_flattened = output.reshape(B * N, num_classes)
        target_flattened = target.repeat(1, N).reshape(-1)
        loss_reshape = (B, N)

    return output_flattened, target_flattened, loss_reshape, target


#############################################
def prep_self_supervised_loss(output, mask, input_seq_shape=None):
    """
    prep_self_supervised_loss(output, mask)
    """
    
    input_seq_str = ""
    if input_seq_shape is not None:
        input_seq_str = (
            f"Input sequence shape: {input_seq_shape} "
            "(expecting [B, N, C, SL, H, W]).\n"
        )

    logger.debug(
        f"Model called next.\n{input_seq_str}"
        f"Output shape: {output.shape} "
        "(expecting a 6D tensor: [B, PS, HW, B_per, PS, HW]).\n"
        f"Mask shape: {mask.size()}"
    )

    # batch x pred step x dim squared x batch/GPU x pred step x dim squared
    (B, PS, HW, B_per, _, _) = mask.size()
    flat_dim = B * PS * HW
    flat_dim_per = B_per * PS * HW # B_per: batch size per GPU
    loss_reshape = (B, PS, HW)

    target = get_target_from_mask(mask)

    # output is a 6d tensor: [B, PS, HW, B_per, PS, HW]
    output_flattened = output.reshape(flat_dim, flat_dim_per)
    target_flattened = target.reshape(
        flat_dim, flat_dim_per).argmax(dim=1)

    return output_flattened, target_flattened, loss_reshape, target


#############################################
def prep_loss(output, mask, sup_target=None, input_seq_shape=None, 
              supervised=False, shared_pred=False, SUB_B=None):
    """
    prep_loss(output, mask)
    """

    if supervised:
        if sup_target is None:
            raise ValueError("Must pass 'sup_target' if 'supervised' is True.")
        output_flattened, target_flattened, loss_reshape, target = \
            prep_supervised_loss(
                output, sup_target, shared_pred=shared_pred, SUB_B=SUB_B
                )

        return output_flattened, target_flattened, loss_reshape, target

    else:
        output_flattened, target_flattened, loss_reshape, target = \
            prep_self_supervised_loss(
                output, mask, input_seq_shape=input_seq_shape
                )

        return output_flattened, target_flattened, loss_reshape, target



############################################
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


############################################
def add_batch_data(data_dict, dataset, batch_loss, batch_loss_by_item, 
                   supervised=False, sup_target=None, output=None, target=None, 
                   epoch_n=0):
    """
    add_batch_data(data_dict, dataset, batch_loss, batch_loss_by_item)
    """
    
    data_dict["batch_epoch_n"].append(epoch_n)
    data_dict["loss_by_batch"].append(batch_loss)
    data_dict["loss_by_item"].append(batch_loss_by_item)

    if output is not None and "output_by_batch" in data_dict.keys():
        data_dict["output_by_batch"].append(output)

    if target is not None and "target_by_batch" in data_dict.keys():
        data_dict["target_by_batch"].append(target)

    if not supervised and sup_target is not None:
        data_dict["sup_target_by_batch"].append(
            get_sup_target(dataset, sup_target)
            )


#############################################
def log_epoch(stats_str, duration=None, epoch_n=0, num_epochs=50, val=False, 
              test=False, batch_idx=0, n_batches=None, spacing="\n"):
    """
    log_epoch(stats_str)
    """

    if test:
        epoch_str = f"Epoch [{epoch_n}] [test]"
        space_batch = " "
    elif val:
        epoch_str = f"Epoch [{epoch_n}/{num_epochs}] [val]"
        space_batch = " "
    else:
        epoch_str = f"Epoch: [{epoch_n}/{num_epochs}]"
        space_batch = ""
        

    time_str = ""
    if n_batches is None and duration is not None:
        time_str = misc_utils.format_time(duration, sep_min=True, lead=True)
    else:
        epoch_str = f"{epoch_str}{space_batch}[{batch_idx}/{n_batches - 1}]"
        if duration is not None:
            time_str = misc_utils.format_time(
                duration, sep_min=False, lead=True
                )
            time_str = f"{time_str}/batch"

    log_str = f"{epoch_str}{TAB}{stats_str}{TAB}{time_str}"

    logger.info(log_str, extra={"spacing": spacing})

    return log_str


#############################################
def write_log(content, epoch_n, filename):
    """
    write_log(content, epoch_n, filename)
    """
    
    open_mode = "a" if Path(filename).is_file() else "w"

    with open(filename, open_mode) as f:
        f.write(f"## Epoch {epoch_n}:\n")
        f.write(f"Time: {datetime.now()}\n")
        f.write(f"{content}\n\n")


