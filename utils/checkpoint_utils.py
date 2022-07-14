import copy
import glob
import logging
from pathlib import Path
import re
import warnings

import torch

from model import resnet_2d3d
from utils import training_utils

logger = logging.getLogger(__name__)

TAB = "    "


#############################################
def check_checkpoint(checkpoint_path, raise_err=True):
    """
    check_checkpoint(checkpoint_path)
    """

    if Path(checkpoint_path).is_file():
        checkpoint_exists = True
    else:
        checkpoint_exists = False
        err_str = f"No checkpoint found at '{checkpoint_path}'."
        if raise_err:
            raise OSError(err_str)
        else:
            warnings.warn(err_str)

    return checkpoint_exists
    

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
    """
    get_state_dict(model, state_dict)
    """

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
def load_resume_checkpoint(checkpoint_path, model, optimizer=None, 
                           lr=1e-3, reset_lr=False, raise_err=True):
    """
    load_resume_checkpoint(checkpoint_path, model)
    """

    log_idx, start_epoch_n = 0, 0
    best_acc = None

    # check if the checkpoint exists
    if not Path(checkpoint_path).is_file():
        err_str = f"No checkpoint found at '{checkpoint_path}'."
        if raise_err:
            raise OSError(err_str)
        else:
            warnings.warn(err_str)
            return log_idx, best_acc, start_epoch_n

    # load checkpoint
    old_lr = None
    if "_lr" in str(checkpoint_path):
        old_lr = float(re.search("_lr(.+?)_", checkpoint_path).group(1))
    checkpoint = torch.load(
        checkpoint_path, map_location=torch.device("cpu")
        )
    checkpoint_epoch_n = load_key_from_checkpoint(
        checkpoint, ["epoch_n", "epoch"]
        )
    start_epoch_n = checkpoint_epoch_n + 1
    log_idx = load_key_from_checkpoint(
        checkpoint, ["log_idx", "iteration"]
        )
    best_acc = checkpoint["best_acc"]
    try:
        checkpoint["state_dict"] = get_state_dict(
            model, checkpoint["state_dict"]
            )
        model.load_state_dict(checkpoint["state_dict"])
    except RuntimeError as err:
        if "Missing key" in str(err):
            model_type = "supervised"
            if not training_utils.get_num_classes_sup(model)[1]:
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
        f"=> Loaded checkpoint to resume from: '{checkpoint_path}' "
        f"(epoch {checkpoint_epoch_n})."
        )

    return log_idx, best_acc, start_epoch_n

    
#############################################
def load_pretrained_checkpoint(checkpoint_path, model, raise_err=True, 
                               test=False):
    """
    load_pretrained_checkpoint(checkpoint_path, model, optimizer)
    """

    # check if the checkpoint exists
    if not check_checkpoint(checkpoint_path, raise_err=raise_err):
        return

    # load checkpoint
    if test:
        reload_str = "test"
        prev_str = "model to test"
    else:
        reload_str = "pretrained"
        prev_str = "pretrained model"

    checkpoint = torch.load(
        checkpoint_path, map_location=torch.device("cpu")
        )
    checkpoint_epoch_n = load_key_from_checkpoint(
        checkpoint, ["epoch_n", "epoch"]
        )

    try:
        checkpoint["state_dict"] = get_state_dict(
            model, checkpoint["state_dict"]
            )
        model.load_state_dict(checkpoint["state_dict"])

    except RuntimeError as err:
        if not str(err).startswith("Error(s) in loading state_dict"):
            raise err

        logger.warning(
            f"Weight structures of the {prev_str} and target model not match. "
            "Using non-equal load.", extra={"spacing": "\n"}
            )

        checkpoint["state_dict"] = get_state_dict(
            model, checkpoint["state_dict"]
            )

        # Load in place
        resnet_2d3d.neq_load_customized(model, checkpoint["state_dict"])
            
    logger.info(
        f"=> Loaded {reload_str} checkpoint '{checkpoint_path}' "
        f"(epoch {checkpoint_epoch_n})."
        )

    return


#############################################
def load_checkpoint(model, optimizer=None, resume=False, pretrained=False, 
                    test=True, lr=1e-3, reset_lr=False):
    """
    load_checkpoint(model)
    """

    if bool(resume) + bool(pretrained) + bool(test) > 1:
        raise ValueError("Only resume, pretrained or test can be True.")
    
    log_idx, start_epoch_n = 0, 0
    best_acc = None

    if resume:
        log_idx, best_acc, start_epoch_n = load_resume_checkpoint(
            resume, model, optimizer, lr=lr, reset_lr=reset_lr, raise_err=False
            )
    elif pretrained or test:
        reload_model = pretrained if pretrained else test

        if test and reload_model == "random":
            logger.warning("Loading random weights.")

        else:
            load_pretrained_checkpoint(
                reload_model, model, raise_err=False, test=test
            )

    
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

