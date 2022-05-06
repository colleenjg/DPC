#!/usr/bin/env python

import copy
from datetime import datetime
import glob
import logging
import multiprocessing
import os
from pathlib import Path
import re
import warnings

import torch

from dataset import dataset_3d
from model import resnet_2d3d

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
def get_num_classes(model):
    """
    get_num_classes(model)
    """
    
    model = get_model(model)
    num_classes = None
    
    if hasattr(model, "num_classes"):
        num_classes = model.num_classes
    
    return num_classes


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
def load_key_from_checkpoint(checkpoint, keys):
    """
    load_key_from_checkpoint(checkpoint, keys)
    """

    if len(keys) != 2:
        raise ValueError("'keys' must have a length of 2.")

    try:
        value = checkpoint[keys[0]]
    except KeyError:
        value = checkpoint[keys[1]]

    return value


#############################################
def get_state_dict(model, state_dict):

    # in case there is a mismatch: model wrapped or not with DataParallel()
    state_dict = copy.deepcopy(state_dict)
    if not hasattr(model, "device_ids"):
        for key in list(state_dict.keys()):
            if key.startswith("module."):
                new_key = key[7:]
                state_dict[new_key] = state_dict.pop(key)
    else:
        for key in list(state_dict.keys()):
            if not key.startswith("module."):
                new_key = f"module.{key}"
                state_dict[new_key] = state_dict.pop(key)

    return state_dict


#############################################
def load_checkpoint(model, optimizer=None, resume=False, pretrained=False, 
                    test=True, lr=1e-3, reset_lr=False):
    """
    load_checkpoint(model)
    """

    if bool(resume) + bool(pretrained) + bool(test) > 1:
        raise ValueError("Only resume, pretrained or test can be True.")
    
    log_idx, start_epoch_n = 0, 0
    best_acc, old_lr = None, None

    if resume:
        if Path(resume).is_file():
            old_lr = None
            if "_lr" in str(resume):
                old_lr = float(re.search("_lr(.+?)_", resume).group(1))
            checkpoint = torch.load(resume, map_location=torch.device("cpu"))
            checkpoint_epoch_n = load_key_from_checkpoint(
                checkpoint, ["epoch_n", "epoch"]
                )
            start_epoch_n = checkpoint_epoch_n + 1
            log_idx = load_key_from_checkpoint(checkpoint, ["log_idx", "iteration"])
            best_acc = checkpoint["best_acc"]
            try:
                checkpoint["state_dict"] = get_state_dict(
                    model, checkpoint["state_dict"]
                    )
                model.load_state_dict(checkpoint["state_dict"])
            except RuntimeError as err:
                if "Missing key" in str(err):
                    model_type = "supervised"
                    if get_num_classes(model) is None:
                        model_type = "self-supervised"
                    raise RuntimeError(
                        f"{err}\nEnsure that you are resuming from a "
                        f"{model_type} model checkpoint."
                        )
                else:
                    raise err
            # if not resetting lr, load old optimizer
            if optimizer is not None and not reset_lr: 
                optimizer.load_state_dict(checkpoint["optimizer"])
            else: 
                # optimizer state is not reloaded
                old_lr_str = "(unknown)" if old_lr is None else f"of {old_lr}"
                if old_lr != lr:
                    lr_str = (f", with lr of {lr} instead of previous value "
                        f"{old_lr_str}")
                
                logger.info(
                    (f"==== Using new optimizer{lr_str} ====")
                    )
            logger.info(
                f"=> Loaded checkpoint to resume from: '{resume}' "
                f"(epoch {checkpoint_epoch_n})."
                )
        else:
            warnings.warn(f"No checkpoint found at '{resume}'.")

    elif pretrained or test:
        reload_model = pretrained if pretrained else test
        reload_str = "pretrained" if pretrained else "test"
        if reload_model == "random":
            logger.warning("Loading random weights.")
        elif Path(reload_model).is_file():
            checkpoint = torch.load(
                reload_model, map_location=torch.device("cpu")
                )
            checkpoint_epoch_n = load_key_from_checkpoint(
                checkpoint, ["epoch_n", "epoch"]
                )
            if test:
                try:
                    checkpoint["state_dict"] = get_state_dict(
                        model, checkpoint["state_dict"]
                        )
                    model.load_state_dict(checkpoint["state_dict"])
                    test_loaded = True
                except:
                    logger.warning(
                        "Weight structure does not match test model. Using "
                        "non-equal load."
                        )
                    test_loaded = False
            if pretrained or not test_loaded:
                checkpoint["state_dict"] = get_state_dict(
                    model, checkpoint["state_dict"]
                    )
                model = resnet_2d3d.neq_load_customized(
                    model, checkpoint["state_dict"]
                    )
            logger.info(
                f"=> Loaded {reload_str} checkpoint '{reload_model}' "
                f"(epoch {checkpoint_epoch_n})."
                )
        else: 
            warnings.warn(f"=> No checkpoint found at '{reload_model}'.")
    
    return log_idx, best_acc, start_epoch_n


#############################################
def save_checkpoint(state_dict, is_best=0, gap=1, filename=None, 
                    keep_all=False):
    """
    save_checkpoint(state_dict)
    """
    
    if filename is None:
        filename = Path("models", "checkpoint.pth.tar")
    
    torch.save(state_dict, filename)
    
    prev_epoch_n = state_dict["epoch_n"] - gap
    last_epoch_path = Path(
        Path(filename).parent, f"epoch{prev_epoch_n}.pth.tar"
        )
    if not keep_all and last_epoch_path.exists():
        last_epoch_path.unlink() # remove

    if is_best:
        all_past_best = glob.glob(
            str(Path(Path(filename).parent, "model_best_*.pth.tar"))
            )
        for past_best in all_past_best:
            if Path(past_best).exists():
                Path(past_best).unlink() # remove

        epoch_n = state_dict["epoch_n"]
        torch.save(
            state_dict, 
            Path(Path(filename).parent, f"model_best_epoch{epoch_n}.pth.tar")
            )


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

