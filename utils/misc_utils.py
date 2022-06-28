import copy
import yaml
import logging
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

from torchvision import transforms
from torchvision import utils as vutils

logger = logging.getLogger(__name__)

TAB = "    "


#############################################
def seed_all(seed=None, deterministic_algorithms=True):
    """
    seed_all()
    """

    if seed is not None:
        logger.debug(f'Random state seed: {seed}')

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        if deterministic_algorithms:
            torch.use_deterministic_algorithms(True)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' # for cuda > 10.2


#############################################
def format_time(duration, sep_min=True, lead=False):
    """
    format_time(duration)
    """
    
    if sep_min:
        minutes = int(duration // 60)
        seconds = duration - minutes * 60
        time_str = f"{minutes}m {seconds:05.3f}s"
    else:
        time_str = f"{duration:.3f}s"

    if lead:
        time_str = f"Time: {time_str}"

    return time_str


#############################################
def get_unique_filename(filename, overwrite=False):
    """
    get_unique_filename(filename)
    """
    
    if Path(filename).is_file():
        if overwrite:
            logger.warning(
                (f"{filename} already exists. Removing, as it will be "
                "overwritten."), 
                extra={"spacing": "\n"}
                )
            time.sleep(5) # to allow for skipping file removal.
            shutil.unlink(str(filename))
        else:
            i = 0
            parent = Path(filename).parent
            base_stem = Path(filename).stem
            suffix = Path(filename).suffix
            while Path(filename).is_file():
                i += 1
                filename = Path(parent, f"{base_stem}_{i:03}{suffix}")

    return filename


#############################################
def get_unique_direc(direc, overwrite=False):
    """
    get_unique_direc(direc)
    """
    
    if Path(direc).is_dir():
        if overwrite:
            logger.warning(
                f"{direc} already exists. Removing, as it will be overwritten.", 
                extra={"spacing": "\n"}
                )
            time.sleep(5) # to allow for skipping directory removal.
            shutil.rmtree(str(direc))
        else:
            i = 0
            base_direc = direc
            while Path(direc).is_dir():
                i += 1
                direc = Path(f"{base_direc}_{i:03}")

    return direc


#############################################
class BasicLogFormatter(logging.Formatter):
    """
    BasicLogFormatter()

    Basic formatting class that formats different level logs differently. 
    Allows a spacing extra argument to add space at the beginning of the log.
    """

    dbg_fmt  = "%(spacing)s%(levelname)s: %(module)s: %(lineno)d: %(msg)s"
    info_fmt = "%(spacing)s%(msg)s"
    wrn_fmt  = "%(spacing)s%(levelname)s: %(msg)s"
    err_fmt  = "%(spacing)s%(levelname)s: %(module)s: %(lineno)d: %(msg)s"
    crt_fmt  = "%(spacing)s%(levelname)s: %(module)s: %(lineno)d: %(msg)s"

    def __init__(self, fmt="%(spacing)s%(levelname)s: %(msg)s"):
        """
        Optional args:
            - fmt (str): default format style.
        """
        super().__init__(fmt=fmt, datefmt=None, style="%") 

    def format(self, record):

        if not hasattr(record, "spacing"):
            record.spacing = ""

        # Original format as default
        format_orig = self._style._fmt

        # Replace default as needed
        if record.levelno == logging.DEBUG:
            self._style._fmt = BasicLogFormatter.dbg_fmt
        elif record.levelno == logging.INFO:
            self._style._fmt = BasicLogFormatter.info_fmt
        elif record.levelno == logging.WARNING:
            self._style._fmt = BasicLogFormatter.wrn_fmt
        elif record.levelno == logging.ERROR:
            self._style._fmt = BasicLogFormatter.err_fmt
        elif record.levelno == logging.CRITICAL:
            self._style._fmt = BasicLogFormatter.crt_fmt

        # Call the original formatter class to do the grunt work
        formatted_log = logging.Formatter.format(self, record)

        # Restore default format
        self._style._fmt = format_orig

        return formatted_log


#############################################
def set_logger_level(logger, level="info"):
    """
    set_logger_level(logger)
    """

    if str(level).isdigit():
        logger.setLevel(int(level))
    elif isinstance(level, str) and hasattr(logging, level.upper()):
        logger.setLevel(getattr(logging, level.upper()))
    else:
        raise ValueError(f'{level} logging level not recognized.')


#############################################
def get_logger(name=None, level="info", fmt=None, skip_exists=True):
    """
    get_logger()

    Returns logger. 

    Optional args:
        - name (str)        : logger name. If None, the root logger is returned.
                              default: None
        - level (str)       : level of the logger ("info", "error", "warning", 
                               "debug", "critical")
                              default: "info"
        - fmt (Formatter)   : logging Formatter to use for the handlers
                              default: None
        - skip_exists (bool): if a logger with the name already has handlers, 
                              does nothing and returns existing logger
                              default: True

    Returns:
        - logger (Logger): logger object
    """

    # create one instance
    logger = logging.getLogger(name)

    if not skip_exists:
        logger.handlers = []

    # skip if logger already has handlers
    add_handler = True
    for hd in logger.handlers:
        if isinstance(hd, logging.StreamHandler):
            add_handler = False
            if fmt is not None:
                hd.setFormatter(fmt)

    if add_handler:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    set_logger_level(logger, level)

    return logger


#############################################
def get_logger_with_basic_format(**logger_kw):
    """
    get_logger_with_basic_format()

    Returns logger with basic formatting, defined by BasicLogFormatter class.

    Keyword args:
        - logger_kw (dict): keyword arguments for get_logger()
        
    Returns:
        - logger (Logger): logger object
    """


    basic_formatter = BasicLogFormatter()

    logger = get_logger(fmt=basic_formatter, **logger_kw)

    return logger


#############################################
def add_nested_dict(src_dict, target_dict, keys, nested_key, 
                    raise_missing=True):
    """
    add_nested_dict(src_dict, target_dict, keys, nested_key)
    """
    
    src_dict = copy.deepcopy(src_dict)

    if nested_key in target_dict.keys():
        raise ValueError(
            f"{nested_key} key already exists in the target dictionary."
            )
        
    target_dict[nested_key] = dict()
    
    missing = []
    for key in keys:
        if key in src_dict.keys():
            target_dict[nested_key][key] = src_dict.pop(key)
        else:
            missing.append(key)

    if len(missing):
        missing_str = (
            "The following keys were missing from the source dictionary: "
            f"{', '.join(missing)}"
            )
        if raise_missing:
            raise KeyError(missing_str)
        else:
            logger.warning(missing_str, extra={"spacing": "\n"})

    return src_dict


#############################################
def init_tb_writers(direc=".", val=False):
    """
    init_tb_writers()
    """

    from tensorboardX import SummaryWriter

    if direc is None:
        raise ValueError("direc cannot be None.")

    Path(direc).mkdir(exist_ok=True, parents=True)
    writer_train = SummaryWriter(logdir=str(Path(direc, "train")))
    writer_val = None
    if val:
        writer_val = SummaryWriter(logdir=str(Path(direc, "val")))
    return writer_train, writer_val


#############################################
def init_model_direc(output_dir="."):
    """
    init_model_direc()
    """
    
    model_direc = Path(output_dir, "model")
    Path(model_direc).mkdir(exist_ok=True, parents=True)

    return model_direc


#############################################
def denorm(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    denorm()
    """
    
    if len(mean) != 3:
        raise ValueError("'mean' must comprise 3 values.")
    if len(std) != 3:
        raise ValueError("'std' must comprise 3 values.")
    inv_mean = [-mean[i] / std[i] for i in range(3)]
    inv_std = [1 / i for i in std]
    
    return transforms.Normalize(mean=inv_mean, std=inv_std)


#############################################
def write_input_seq_tb(writer, input_seq, n=2, i=0):
    """
    write_input_seq_tb(writer, input_seq)

    Write input sequences to a tensorboard writer.
    """

    _, N, C, SL, H, W = input_seq.shape
    writer.add_image(
        "input_seq", 
        denorm(vutils.make_grid(
            input_seq[:n].transpose(2, 3).reshape(-1, C, H, W), 
            nrow=N * SL)
        ), i
    )


#############################################
def update_tb(writer_train, train_dict, epoch_n=0, writer_val=None, 
              val_dict=None, ks=[1, 2, 3]):
    """
    update_tb(writer_train, train_dict)
    """

    topk_datatypes = [f"accuracy/top{int(k)}" for k in ks]     
    datatypes = ["global/loss", "global/accuracy"] + topk_datatypes

    for mode in ["train", "val"]:
        if mode == "train":
            writer = writer_train                    
            data_dict = train_dict
        elif val_dict is not None:
            if writer_val is None:
                raise ValueError(
                    "Must provide writer_val if val_dict is provided."
                    )
            writer = writer_val
            data_dict = val_dict

        all_data = [
            data_dict["loss"], 
            data_dict["acc"], 
            *data_dict["topk_meters"]
            ]

        for datatype, data in zip(datatypes, all_data):
            writer.add_scalar(datatype, data, epoch_n)


#############################################
def save_hyperparameters(hyperparams, direc=None):
    """
    save_hyperparameters(hyperparams)
    """

    hyperparams = copy.deepcopy(hyperparams)
    hyperparam_dict = dict()

    # check parameters
    necessary_params = ["dataset", "overwrite", "resume", "supervised", "test"]
    if direc is None:
        necessary_params.append("output_dir")

    for param in necessary_params:
        if param not in hyperparams.keys():
            raise KeyError(f"{param} missing from hyperparams dictionary.")
    
    dataset = hyperparams["dataset"]
    overwrite = hyperparams["overwrite"]
    resume = hyperparams["resume"]
    supervised = hyperparams["supervised"]
    test = hyperparams["test"]
    if direc is None:
        direc = hyperparams["output_dir"]


    # set lists for collecting parameters
    ignore_params = ["diff_possizes", "no_gray", "not_save_best"]


    # set technical parameters
    technical_params = [
        "cpu_only", "data_parallel", "device_type", "num_workers", "plt_bkend", 
        "seed", "use_tb"
        ]
    hyperparams = add_nested_dict(
        hyperparams, hyperparam_dict, technical_params, nested_key="technical", 
        raise_missing=False
        )


    # model parameters
    model_params = [
        "batch_size", "model", "net", "pretrained", "resume", "supervised", 
        "test"
        ]

    training_params = ["lr", "num_epochs", "train_what", "wd"]
    if test:
        ignore_params = ignore_params + training_params
        ignore_params = ignore_params + ["dropout", "pred_step", "reset_lr"]
    else:
        model_params = model_params + training_params
        if resume:
            model_params = model_params + ["reset_lr"]
        else:
            ignore_params = ignore_params + ["reset_lr"]
        if supervised:
            model_params.append("dropout")
            ignore_params.append("pred_step")
        else:
            model_params.append("pred_step")
            ignore_params.append("dropout")

    hyperparams = add_nested_dict(
        hyperparams, hyperparam_dict, model_params, nested_key="model", 
        raise_missing=False
        )

    # set dataset parameters
    dataset_params = [
        "dataset", "img_dim", "num_seq", "no_transforms", "seq_len"
    ]

    ucf_hmdb_ms_params = ["ucf_hmdb_ms_ds"]
    if dataset in ["UCF101", "HMDB51", "MouseSim"]:
        dataset_params = dataset_params + ucf_hmdb_ms_params
    else:
        ignore_params = ignore_params + ucf_hmdb_ms_params        

    ms_params = ["eye"]
    if dataset == "MouseSim":
        dataset_params = dataset_params + ms_params
    else:
        ignore_params = ignore_params + ms_params

    gabor_params = [
        "diff_U_possizes", "gab_img_len", "gray", "num_gabors", 
        "same_possizes", "roll", "train_len", "U_prob", "unexp_epoch"
        ]
    if dataset == "Gabors":
        dataset_params = dataset_params + gabor_params
    else:
        ignore_params = ignore_params + gabor_params

    hyperparams = add_nested_dict(
        hyperparams, hyperparam_dict, dataset_params, nested_key="dataset", 
        raise_missing=False
        )


    # general parameters
    general_params = [
        "data_path_dir", "log_freq", "log_level", "output_dir", "save_best", 
        "save_by_batch"
        ]
    if resume:
        overwrite = False
        ignore_params.append("overwrite")
    else:
        general_params.append("overwrite")

    hyperparams = add_nested_dict(
        hyperparams, hyperparam_dict, general_params, nested_key="general", 
        raise_missing=False
        )
    
    # convert Path objects to strings
    for nested_key in hyperparam_dict.keys():
        for key in hyperparam_dict[nested_key].keys():
            val = hyperparam_dict[nested_key][key]
            if isinstance(val, Path):
                hyperparam_dict[nested_key][key] = str(val)

    # final check
    left_over = []
    for key in hyperparams.keys():
        if key not in ignore_params:
            left_over.append(key)
    
    if len(left_over):
        left_over_str = (
            "The following keys were left out of the hyperparameters "
            f"dictionary: {', '.join(left_over)}"
            )
        logger.warning(left_over_str, extra={"spacing": "\n"})


    # save hyperparameters (as yaml for legibility, as the file is light-weight)
    base_name = "hyperparameters.yaml"
    if test:
        base_name = f"test_{base_name}"
    if resume:
        base_name = f"resume_{base_name}"

    hyperparams_path = get_unique_filename(
        Path(direc, base_name), overwrite=overwrite
        )

    Path(hyperparams_path).parent.mkdir(exist_ok=True, parents=True)

    with open(hyperparams_path, "w") as f:
        yaml.dump(hyperparam_dict, f)

