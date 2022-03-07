#!/usr/bin/env python

from datetime import datetime
import glob
import logging
import multiprocessing
from pathlib import Path
import re

import torch
from torchvision import utils as vutils

from model import resnet_2d3d

logger = logging.getLogger(__name__)


#############################################
def get_num_jobs(max_n=None, min_n=1):
    """get number of jobs to run in parallel"""
    num_jobs = multiprocessing.cpu_count()
    if max_n is not None:
        num_jobs = min(num_jobs, max_n)
    num_jobs = int(max(num_jobs, min_n))

    return num_jobs


#############################################
def get_device(num_workers=None):

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    if num_workers is None:
        num_cores = multiprocessing.cpu_count()
        num_devices = torch.cuda.device_count()
        num_workers = min(num_devices * 2, num_cores) # approx rule of thumb
    
    return device, num_workers


#############################################
def log_weight_decay_prop(model):
    total_weight = 0.0
    decay_weight = 0.0
    for m in model.parameters():
        if m.requires_grad: 
            decay_weight += m.norm(2).data
        total_weight += m.norm(2).data
    logger.info(
        "Norm of weights to decay / total: "
        f"{decay_weight:.3f} / {total_weight:.3f}"
        )


#############################################
def get_target_from_mask(mask):
    """task mask as input, compute the target for contrastive loss"""
    # mask meaning: 1: pos
    
    (B, PS, D2, _, _, _) = mask.size() # [B x PS x D2 x B x PS x D2]
    target = (mask == 1).to(int)
    target.requires_grad = False
    return target, (B, PS, D2)


#############################################
def check_grad(model):
    logger.debug("\n===========Check Grad============")
    for name, param in model.named_parameters():
        logger.debug(name, param.requires_grad)
    logger.debug("=================================\n")


#############################################
def get_multistepLR_restart_multiplier(epoch, gamma=0.1, step=[10, 15, 20], 
                                       repeat=3):
    """
    Return the multipier for LambdaLR, 
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
        exp = len([i for i in step if effective_epoch>=i])
    return gamma ** exp


#############################################
def get_lr_lambda(dataset="hmdb51", img_dim=224): 
    if dataset == "hmdb51":
        steps = [150, 250, 300]
        
    elif dataset == "ucf101":
        if img_dim == 224:
            steps = [300, 400, 500]
        else: 
            steps = [60, 80, 100]
    else:
        steps = [150, 250, 300]

    lr_lambda = lambda ep: get_multistepLR_restart_multiplier(
        ep, gamma=0.1, step=steps, repeat=1
        )

    return lr_lambda


#############################################
def load_checkpoint(model, optimizer=None, resume=False, pretrained=False, 
                    test=True, lr=1e-3, reset_lr=False):

    if bool(resume) + bool(pretrained) + bool(test) > 1:
        raise ValueError("Only resume, pretrained or test can be True.")
    
    iteration, start_epoch = 0, 0
    best_acc, old_lr = None, None

    if resume:
        if Path(resume).is_file():
            old_lr = float(re.search("_lr(.+?)_", resume).group(1))
            logger.info(f"=> Loading checkpoint to resume from: '{resume}'")
            checkpoint = torch.load(resume, map_location=torch.device("cpu"))
            start_epoch = checkpoint["epoch_n"]
            iteration = checkpoint["iteration"]
            best_acc = checkpoint["best_acc"]
            model.load_state_dict(checkpoint["state_dict"])
            # if not resetting lr, load old optimizer
            if optimizer is not None and not reset_lr: 
                optimizer.load_state_dict(checkpoint["optimizer"])
            else: 
                logger.info(f"==== Changing lr from {old_lr} to {lr} ====")
            logger.info(
                f"=> Loaded checkpoint to resume from: '{resume}' "
                f"(epoch {checkpoint['epoch']})"
                )
        else:
            logger.warning(f"No checkpoint found at '{resume}'")

    elif pretrained or test:
        reload_model = pretrained if pretrained else test
        reload_str = "pretrained" if pretrained else "test"
        if reload_model == "random":
            logger.warning("Loading random weights.")
        elif Path(reload_model).is_file():
            logger.info(f"=> Loading {reload_str} checkpoint: '{reload_model}'")
            checkpoint = torch.load(
                reload_model, map_location=torch.device("cpu")
                )
            if test:
                start_epoch = checkpoint["epoch_n"]
                try: 
                    model.load_state_dict(checkpoint['state_dict'])
                    test_loaded = True
                except:
                    logger.warning(
                        "Weight structure does not match test model. Using "
                        "non-equal load."
                        )
                    test_loaded = False
            if pretrained or not test_loaded:
                model = resnet_2d3d.neq_load_customized(
                    model, checkpoint["state_dict"]
                    )
            logger.info(
                f"=> Loaded {reload_str} checkpoint '{reload_model}' "
                f"(epoch {checkpoint['epoch']})"
                )
        else: 
            logger.warning(f"=> No checkpoint found at '{reload_model}'")
    
    return iteration, best_acc, start_epoch


#############################################
def save_checkpoint(state_dict, is_best=0, gap=1, filename=None, 
                    keep_all=False):
    
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
    
    open_mode = "a" if Path(filename).is_file() else "w"

    with open(filename, open_mode) as f:
        f.write(f"## Epoch {epoch_n}:\n")
        f.write(f"time: {datetime.now()}\n")
        f.write(f"{content}\n\n")

