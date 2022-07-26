import copy
import glob
import logging
import numpy as np
from pathlib import Path
import re
import warnings

import torch

from model import resnet_2d3d
from utils import misc_utils, training_utils

logger = logging.getLogger(__name__)

TAB = "    "


#############################################
def check_checkpoint(checkpoint_path, raise_err=True):
    """
    check_checkpoint(checkpoint_path)

    Checks whether a checkpoint exists at a specific path, and either raises an 
    error or a warning if it doesn't exist.

    Required args
    -------------
    - checkpoint_path : str or path
        Path to the checkpoint.

    Optional args
    -------------
    - raise_error : bool (default=True)
        If True, an error is raised if the checkpoint does not exist. 
        Otherwise, a warning is produced.

    Returns
    -------
    - checkpoint_exists : bool
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

    Loads a specific key from a checkpoint, optionally trying several 
    possibilities, in order.  The first key that exists in the checkpoint 
    dictionary is used.

    Required args
    -------------
    - checkpoint : dict
        Checkpoint dictionary to load from.
    - keys : list
        Keys to try loading from checkpoint, in order.

    Returns
    -------
    - value : object
        Object stored under the specified key in the checkpoint dictionary.
    """

    loaded = False
    for key in keys:
        if key in checkpoint.keys():
            loaded = True
            value = checkpoint[key]
            break
    
    if not loaded:
        key_names = ", ".join([str(key) for key in keys])
        raise KeyError(
            "'checkpoint' does not contain any of the following keys: "
            f"{key_names}'"
            )

    return value


#############################################
def get_state_dict(model, state_dict):
    """
    get_state_dict(model, state_dict)

    Returns a copy of the state dictionary, with keys modified to be usable 
    with the model, depending on whether it is wrapped with nn.DataParallel().

    Required args
    -------------
    - model : nn.Module or nn.DataParallel
        Model.
    - state_dict : dict
        State dictionary.

    Returns
    -------
    - state_dict : dict
        State dictionary, with keys updated to be usable with the model.
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

    Loads a checkpoint in place to resume model training, and returns variables 
    to situate new start point.

    Required args
    -------------
    - checkpoint_path : str or path
        Path to the checkpoint.
    - model : nn.Module or nn.DataParallel
        Model to load checkpoint into.

    Optional args
    -------------
    - optimizer : torch.optim object (default=None)
        Torch optimizer.
    - lr : float (default=1e-3)
    - reset_lr : bool (default=False)
        If True, instead of attempting to update the state of the optimizer 
        with the saved optimizer checkpoint, the optimizer is used in its 
        initialized state.
    - raise_err : bool (default=raise_err)
        If True, an error is raised if no checkpoint is found under 
        checkpoint_path. Otherwise, A warning is created, the model state is 
        not updated, and default values are returned.

    Returns
    -------
    - log_idx : int
        Log index to resume from, for writing to tensorboard.
    - best_acc : float or list
        Best validation accuracy logged for the model 
        (or best validation accuracies for [exp, unexp] models, if trained on 
        the Gabor dataset).
    - start_epoch_n : int
        Epoch number to resume from.
    - gabor_unexp : bool
        Whether the model, if trained on the Gabor dataset, was presented with 
        unexpected sequences.
    """

    log_idx, start_epoch_n = 0, 0
    best_acc = None
    gabor_unexp = None

    # check if the checkpoint exists
    if not Path(checkpoint_path).is_file():
        err_str = f"No checkpoint found at '{checkpoint_path}'."
        if raise_err:
            raise OSError(err_str)
        else:
            warnings.warn(err_str)
            return log_idx, best_acc, start_epoch_n, gabor_unexp

    # load checkpoint
    old_lr = None
    if "_lr" in str(checkpoint_path):
        old_lr = float(re.search("_lr(.+?)_", str(checkpoint_path)).group(1))
    checkpoint = torch.load(
        checkpoint_path, map_location=torch.device("cpu")
        )
    checkpoint_epoch_n = load_key_from_checkpoint(
        checkpoint, ["epoch_n", "epoch"]
        )
    if hasattr(checkpoint, "gabor_unexp"):
        gabor_unexp = checkpoint["gabor_unexp"]

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

    if optimizer is not None:
        if reset_lr:
            lr_str = ""
            old_lr_str = "(unknown)" if old_lr is None else f"of {old_lr}"
            if old_lr != lr:
                lr_str = (f", with lr of {lr} instead of previous value "
                    f"{old_lr_str}")
            logger.info(f"Using new optimizer{lr_str}.")
        else: 
            # if not resetting lr, load old optimizer
            optimizer.load_state_dict(checkpoint["optimizer"])
    
    logger.info(
        f"=> Loaded checkpoint to resume from: '{checkpoint_path}' "
        f"(epoch {checkpoint_epoch_n})."
        )

    return log_idx, best_acc, start_epoch_n, gabor_unexp

    
#############################################
def load_pretrained_checkpoint(checkpoint_path, model, raise_err=True, 
                               test=False):
    """
    load_pretrained_checkpoint(checkpoint_path, model, optimizer)

    Loads a pretrained model checkpoint in place.

    Required args
    -------------
    - checkpoint_path : str or path
        Path to the checkpoint.
    - model : nn.Module or nn.DataParallel
        Model to load checkpoint into.

    Optional args
    -------------
    - raise_err : bool (default=raise_err)
        If True, an error is raised if no checkpoint is found under 
        checkpoint_path. Otherwise, A warning is created, the model state is 
        not updated, and default values are returned.
    - test : bool (default=False)
        If True, checkpoint is loaded for testing purposes. Used for logging.
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

    Loads a checkpoint in place to resume model training, and returns variables 
    to situate the start point.

    Required args
    -------------
    - model : nn.Module or nn.DataParallel
        Model to load checkpoint into.

    Optional args
    -------------
    - optimizer : torch.optim object (default=None)
        Torch optimizer.
    - resume : str or path (default=False)
        Path to checkpoint to resume from, if provided.
    - pretrained : str or path (default=False)
        Path to checkpoint storing the pretrained model to reload, if provided.
    - test : str or path (default=True)
        If provided, path to checkpoint storing model to test or 'random' if 
        testing random weights.
    - lr : float (default=1e-3)
        Learning rate against which to compare old learning rate.
    - reset_lr : bool (default=False)
        If True, instead of attempting to update the state of the optimizer 
        with the saved optimizer checkpoint, the optimizer is used in its 
        initialized state.

    Returns
    -------
    - log_idx : int
        Log index to start or resume from, for writing to tensorboard.
    - best_acc : float or list
        Best validation accuracy logged for the model 
        (or best validation accuracies for [exp, unexp] models, if trained on 
        the Gabor dataset).
    - start_epoch_n : int
        Epoch number to start or resume from.
    - test_suffix : str
        Suffix to use to identify the model on which a test was run.
    - gabor_unexp : bool
        Whether the model, if trained on the Gabor dataset, was presented with 
        unexpected sequences.
    """

    if bool(resume) + bool(pretrained) + bool(test) > 1:
        raise ValueError("Only resume, pretrained or test can be True.")
    
    log_idx, start_epoch_n = 0, 0
    best_acc = None
    gabor_unexp = None
    test_suffix = None

    if resume:
        log_idx, best_acc, start_epoch_n, gabor_unexp = load_resume_checkpoint(
            resume, model, optimizer, lr=lr, reset_lr=reset_lr, raise_err=True
            )
    elif pretrained or test:
        reload_model = pretrained if pretrained else test

        if test:
            test_suffix = misc_utils.get_test_suffix(reload_model)

        if test and reload_model == "random":
            logger.warning("Loading random weights.")

        else:
            load_pretrained_checkpoint(
                reload_model, model, raise_err=True, test=test
            )
    
    return log_idx, best_acc, start_epoch_n, test_suffix, gabor_unexp


#############################################
def get_epoch_number(model_path):
    """
    get_epoch_number(model_path)

    Returns the epoch number based on the model path.

    Required args
    -------------
    - model_path : str or path
        Path to the saved model, expected to take the form 'epochX.pth.tar' or 
        'epochX_{suffix}.pth.tar'.
    
    Returns
    -------
    - epoch_n : int
        Epoch number, extracted from the model path.
    """
    
    stem = Path(model_path).stem.replace("best_", "")
    nbr_part = stem.split("_")[0].split(".")[0]

    pre_nbr_part = "epoch"
    if pre_nbr_part not in nbr_part:
        raise ValueError(
            f"'model_path' should have '{pre_nbr_part}' in the first part "
            f"of the stem, but found {nbr_part}."
            )
    st = nbr_part.index(pre_nbr_part) + len(pre_nbr_part)

    nbr_part = nbr_part[st:]

    if not nbr_part.isdigit():
        raise NotImplementedError(
            "Expected model paths following the saved epoch name pattern "
            "to end in form 'epochX.pth.tar' or 'epochX_{suffix}.pth.tar', "
            f"but found {nbr_part} instead of a number. Cannot parse epoch "
            "number from name."
            )
    
    epoch_n = int(nbr_part)

    return epoch_n


#############################################
def save_checkpoint(state_dict, is_best=False, gap=None, filename=None, 
                    output_dir=".", epoch_n=0, gabor_unexp=None, 
                    keep_all=False):
    """
    save_checkpoint(state_dict)

    Saves checkpoint under specified name, optionally removing previous 
    versions.

    Required args
    -------------
    - state_dict : dict
        State dictionary to save as a checkpoint.

    Optional args
    -------------
    - is_best : bool (default=False)
        If True, the current model is the best model, and additionally saved 
        under the name 'model_best_epoch{epoch_n}.pth.tar", where epoch_n is 
        retrieved from the state dictionary. 
    - gap : int (default=None)
        The gap between the current epoch and the previous epoch to remove, 
        if keep_all is False. If None, any previous epoch matching the pattern 
        is removed.
    - filename : str or path (default=None)
        Filename under which to store checkpoint. If None, a default path name 
        is used.
    - gabor_unexp : bool (default=None)
        If not None, the value of the 'unexp' attribute of the Gabor dataset on 
        which the model was trained. It is added to the model state dictionary, 
        and to the save name pattern.
    - keep_all : bool (default=False)
        If False, previous epoch checkpoints are removed.
    """
    
    unexp_str = "_unexp" if gabor_unexp else ""
    if gabor_unexp is not None:
        state_dict["gabor_unexp"] = gabor_unexp

    if output_dir is None:
        output_dir = Path("models")

    if filename is None:
        filename = f"epoch{epoch_n}{unexp_str}.pth.tar"
    filename = Path(output_dir, filename)
    model_direc = filename.parent

    # save model
    torch.save(state_dict, str(filename))    

    # remove previous epochs, if applicable
    if not keep_all:
        if gap is None:
            all_existing = glob.glob(
                str(Path(model_direc, f"epoch*{unexp_str}.pth.tar"))
            )
            remove_ns = [get_epoch_number(ep_path) for ep_path in all_existing]
            for n in remove_ns:
                remove_path = Path(model_direc, f"epoch{n}{unexp_str}.pth.tar")
                if remove_path.is_file() and remove_path != filename:
                    Path(remove_path).unlink() # remove
        else:
            prev_epoch_n = state_dict["epoch_n"] - gap
            last_epoch_path = Path(
                model_direc, f"epoch{prev_epoch_n}{unexp_str}.pth.tar"
                )                
            if last_epoch_path.exists():
                last_epoch_path.unlink() # remove


    # store as best, and replace previous, if applicable
    if is_best:
        pattern = Path(model_direc, f"best{unexp_str}_epoch*.pth.tar")
        all_past_best = glob.glob(str(pattern))
        for past_best in all_past_best:
            if Path(past_best).exists():
                Path(past_best).unlink() # remove

        epoch_n = state_dict["epoch_n"]
        torch.save(
            state_dict, 
            Path(model_direc, f"best{unexp_str}_epoch{epoch_n}.pth.tar")
            )


#############################################
def find_last_checkpoint(output_dir, best=False, unexp=False, raise_none=True):
    """
    find_last_checkpoint(output_dir)

    Finds the checkpoint for the last epoch recorded, following the pattern 
    "epoch*.pth.tar" in the specified directory.

    Required args
    -------------
    - output_dir : str or path
        Directory in which to search recursively for a model.
    
    Optional args
    -------------
    - best : bool (default=False)
        If True, only 'best' models are returned.
    - unexp : bool (default=False)
        If True, unexpected model checkpoint is searched for. 
    - raise_none : bool (default=True)
        If True and no model is found, an error is raised. Otherwise, if no 
        model is found, a warning is thrown, but None is returned.

    Returns
    -------
    - model_path : path
        Path to the model with the last epoch number or
        None, if none is found, and raise_none if False.
    """


    if not Path(output_dir).exists():
        raise OSError(f"{output_dir} does not exist.")
    if not Path(output_dir).is_dir():
        raise OSError(f"{output_dir} is not a directory.")

    unexp_str = "_unexp" if unexp else ""

    epoch_pattern = f"epoch*{unexp_str}.pth.tar"
    if best:
        epoch_pattern = f"best_{unexp_str}epoch*.pth.tar"
    all_existing = glob.glob(
        str(Path(output_dir, "**", epoch_pattern)), recursive=True
        )
    if not len(all_existing):
        msg = (f"No models found recursively in {output_dir} with pattern "
            f"'{epoch_pattern}'.")
        if raise_none:
            raise ValueError(msg)
        else:
            warnings.warn(msg)
            return None

    epoch_ns = [get_epoch_number(full_path) for full_path in all_existing]

    model_path = Path(all_existing[np.argmax(epoch_ns)])

    return model_path

