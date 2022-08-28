import copy
import logging
import os
from pathlib import Path
import multiprocessing
import random
import shutil
import sys
import time
import warnings
import yaml

import numpy as np
import torch

from torchvision import transforms
from torchvision import utils as vutils

logger = logging.getLogger(__name__)

TAB = "    "


#############################################
class BasicLogFormatter(logging.Formatter):
    """
    Basic formatting class that formats different level logs differently. 
    Allows a spacing extra argument to add space at the beginning of the log.

    Class attributes
    ----------------
    - crt_fmt : str
        Formatting string for critical logs.
    - dbg_fmt : str
        Formatting string for debugging logs.
    - err_fmt : str
        Formatting string for error logs.
    - info_fmt : str
        Formatting string for information logs.
    - wrn_fmt : str
        Formatting string for warning logs.

    Methods
    -------
    - self.format(record):
        Returns formatted record for logging.

    """

    dbg_fmt  = "%(spacing)s%(levelname)s: %(module)s: %(lineno)d: %(msg)s"
    info_fmt = "%(spacing)s%(msg)s"
    wrn_fmt  = "%(spacing)s%(levelname)s: %(msg)s"
    err_fmt  = "%(spacing)s%(levelname)s: %(module)s: %(lineno)d: %(msg)s"
    crt_fmt  = "%(spacing)s%(levelname)s: %(module)s: %(lineno)d: %(msg)s"

    def __init__(self, fmt="%(spacing)s%(levelname)s: %(msg)s"):
        """
        BasicLogFormatter()

        Constructs a BasicLogFormatter object.

        Optional args
        -------------
        - fmt : str
            Default format style.
        """

        super().__init__(fmt=fmt, datefmt=None, style="%") 


    def format(self, record):
        """
        self.format(record)

        Formats record to log, depending on the record level number.

        Required args
        -------------
        - record : str
            Record to log.

        Returns
        -------
        - formatted_log : str
            Record, formatted for logging.
        """

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
def get_order_str(n=1):
    """
    get_order_str()

    Returns an order string (e.g., "1st") based on the digit passed
    """

    if int(float(n)) != float(n):
        raise ValueError(f"{n} must be an integer.")

    n = int(float(n))

    last_digit = int(str(n)[-1])
    if last_digit == 1:
        order_str = f"{n}st"
    elif last_digit == 2:
        order_str = f"{n}nd"
    elif last_digit == 3:
        order_str = f"{n}rd"
    else:
        order_str = f"{n}th"

    return order_str


#############################################
def format_addendum(text=None, is_suffix=True):
    """
    format_addendum()

    Formats text to be used as a prefix or a suffix.

    Optional args
    -------------
    - text : str (default=None)
        Text to be formatted. If None or "", an empty string is returned.
    - is_suffix : bool (default=True)
        If True, text will be formatted as a suffix. Otherwise, it will be 
        formatted as a prefix.

    Returns
    -------
    - text : str
        Text, formatted as a suffix or prefix.
    """

    if text is None or len(text) == 0:
        text = ""
    
    elif is_suffix and text[0] != "_":
        text = f"_{text}" 

    elif not is_suffix and text[-1] != "_":
        text = f"{text}_"
    
    return text


#############################################
def set_logger_level(logger, level="info"):
    """
    set_logger_level(logger)

    Sets the level of the logger.

    Required args
    -------------
    - logger : logging.Logger
        Logger for which to adjust logging level.

    Optional args
    -------------
    - level : str or int (default="info")
        Level at which to set logger.            
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

    Optional args
    -------------
    - name : str (default=None)
        Logger name. If None, the root logger is returned.
    - level : str (default="info")
        Level of the logger ("info", "error", "warning", "debug", "critical").
    - fmt : Formatter (default=None)
        Logging Formatter to use for the handlers.
    - skip_exists : bool (default=True)
        If a logger with the name already has handlers, does nothing and 
        returns existing logger.

    Returns
    -------
    - logger : Logger
        Logger object.
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

    Keyword args
    ------------
    **logger_kw : dict
        Keyword arguments for get_logger(), excluding 'fmt'.
        
    Returns
    -------
    - logger : Logger
        Logger object.
    """

    basic_formatter = BasicLogFormatter()

    logger = get_logger(fmt=basic_formatter, **logger_kw)

    return logger


#############################################
def get_num_jobs(max_n=None, min_n=1):
    """
    get_num_jobs()

    Get number of jobs to run in parallel.

    Optional args
    -------------
    - max_n : int (default=None)
        Maximum number of jobs allowed, if not None.
    - min_n : int (default=1)
        Minimum number of jobs allowed.

    Returns
    -------
    - num_jobs : int
        Number of jobs computed.
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
def renest(flat_list, target_shape):
    """
    renest(flat_list, target_shape)

    Places a flat list into the specified target shape.

    Required args
    -------------
    - flat_list : list
        List to be reshaped using target_shape.
    - target_shape : tuple
        Shape into which to reshape flat_list.

    Returns
    -------
    - reshape_list : list
        flat_list nested into the target shape.
    """

    if len(flat_list) != np.product(target_shape):
        n_vals = len(flat_list)
        exp_vals = np.product(target_shape)
        raise ValueError(
            "Based on 'target_shape', 'flat_list' should have a "
            f"length of {exp_vals}, but found {n_vals}."
            )

    if len(target_shape) == 1:
        reshaped_list = flat_list
    
    else:
        reshaped_list = flat_list
        for d in target_shape[::-1][:-1]:
            n = len(reshaped_list) // d
            reshaped_list = [
                reshaped_list[d * j : d * (j + 1)] for j in range(n)
            ]

    return reshaped_list


#############################################
def get_new_seed(seed=None, seed_none=False):
    """
    get_new_seed()

    Deterministically retrieves a new seed based on a previous one.

    Optional args
    -------------
    - seed : int (default=None)
        Seed based on which to select new seed.
    - seed_none : bool (default=False)
        If True, a seed is generated even if the input seed is None.

    Returns
    -------
    - seed : int
        New seed. 
    """

    if seed_none or seed is not None:
        temp_rng = np.random.RandomState(seed)
        seed = temp_rng.randint(2**32)

    return seed


#############################################
def seed_all(seed=None, deterministic_algorithms=True):
    """
    seed_all()

    If a seed is provided, seeds modules used for random processes, and 
    optionally also sets torch algorithms to behave deterministically.
    
    NOTE: Setting torch models to behave deterministically may cause errors, 
    depending on the torch modules used.

    Optional args
    -------------
    - seed : int (default=None)
        Seed to use in seeding modules (random, numpy, and torch).
    - deterministic_algorithms : bool (default=True)
        If True, torch algorithms are set to behave deterministically, if seed 
        is None.
    """

    if seed is not None:
        logger.debug(f'Random state seed: {seed}')

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        if deterministic_algorithms:
            torch.use_deterministic_algorithms(True)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' # for cuda > 10.2


#############################################
def get_torch_generator(seed, gen_none=False):
    """
    get_torch_generator()
    
    Returns a torch Generator.

    Optional args
    -------------
    - seed : int (default=None)
        Seed to use in seeding torch Generator.
    - gen_none : bool (default=False)
        If True, a torch Generator even if seed is None. Otherwise, None is 
        returned.
        
    Returns
    -------
    rng : torch.Generator
        Torch generator or None.
    """

    rng = None
    if gen_none or seed is not None:
        rng = torch.Generator()
        rng.manual_seed(seed)

    return rng

#############################################
def format_time(duration, sep_min=True, lead=False):
    """
    format_time(duration)

    Converts a duration in seconds into a string.

    Required args
    -------------
    - duration : float
        Time in seconds.

    Optional args
    -------------
    - sep_min : bool (default=True)
        If True, minutes are separated from seconds in the duration string.
    - lead : bool (default=False)
        If True, 'Time: ' is included in the string before the duration is 
        expressed.

    Returns
    -------
    - time_str : str
        Duration string.
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

    Returns a unique filename, based on the input filename provided.

    Required args
    -------------
    - filename : str or path
        Filename.

    Optional args
    -------------
    - overwrite : bool (default=False)
        If True, and the filename exists, the existing version is removed 
        (with a wait time of 5 seconds to allow cancelling the operation).

    Returns
    -------
    - filename : str or path
        Unique version of the filename.
    """
    
    if Path(filename).is_file():
        if overwrite:
            logger.warning(
                (f"{filename} already exists. Removing, as it will be "
                "overwritten."), 
                extra={"spacing": "\n"}
                )
            time.sleep(5) # to allow for skipping file removal.
            Path(filename).unlink()
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

    Returns a unique directory name, based on the input directory name 
    provided.

    Required args
    -------------
    - direc : str or path
        Directory name.

    Optional args
    -------------
    - overwrite : bool (default=False)
        If True, and the directory exists, it is removed 
        (with a wait time of 5 seconds to allow cancelling the operation).

    Returns
    -------
    - direc : str or path
        Unique version of the directory name.
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
def normalize_dataset_name(dataset_name="UCF101", short=False, eye="right"):
    """
    normalize_dataset_name()

    Normalizes the dataset name provided.

    Optional args
    -------------
    - dataset_name : str (default="UCF101")
        Dataset name.
    - short : bool (default=False)
        If True, the short version of the dataset name is returned.
    - eye : str (default="right")
        Additional information for specifying the short name of the MouseSim 
        dataset, namely the eye parameter.

    Returns
    -------
    - dataset_name : str
        Normalized dataset name.
    """

    if dataset_name.lower() == "ucf101":
        dataset_name = "UCF101"
    elif dataset_name.lower() == "hmdb51":
        dataset_name = "HMDB51"
    elif dataset_name.lower() in ["kinetics400", "k400"]:
        dataset_name = "k400" if short else "Kinetics400"
    elif dataset_name.lower() in ["gabors", "gabor"]:
        dataset_name = "Gabors"
    elif dataset_name.lower() == "mousesim":
        dataset_name = "MouseSim"
        if eye not in ["both", "left", "right"]:
            raise ValueError(f"{eye} value for 'eye' not recognized")
        if short and eye != "both":
            dataset_name = f"{dataset_name}_{eye}"
    else:
        raise ValueError(f"{dataset_name} dataset not recognized.")

    if short:
        dataset_name = dataset_name.lower()
    
    return dataset_name


#############################################
def get_num_classes(dataset="UCF101"):
    """
    get_num_classes()

    Returns the number of classes for a given dataset.

    Optional args
    -------------
    - dataset : torch data.Dataset or str (default="UCF101")
        Torch dataset for which to retrieve the number of classes or name of 
        the dataset.

    Returns
    -------
    - num_classes : int 
        Number of classes for the dataset.
    """

    if isinstance(dataset, str):
        dataset_name = normalize_dataset_name(dataset)

        if dataset_name == "UCF101":
            num_classes = 101
        elif dataset_name == "HMDB51":
            num_classes = 51
        elif dataset_name == "Kinetics400":
            num_classes = 400
        elif dataset_name == "MouseSim":
            num_classes = 2
        else:
            raise ValueError(
                f"Cannot retrieve number of classes for {dataset_name}."
                )
    else:
        if hasattr(dataset, "class_dict_decode"):
            num_classes = len(dataset.class_dict_decode)
        else:
            raise NotImplementedError(
                "Cannot retrieve number of classes for the dataset of type "
                f"{type(dataset)}."
                )

    return num_classes


#############################################
def add_nested_dict(src_dict, target_dict, keys, main_key, 
                    raise_missing=True):
    """
    add_nested_dict(src_dict, target_dict, keys, main_key)

    Adds keys from a source dictionary to a dictionary stored under the main 
    key provided in the target dictionary, in place. 

    Required args
    -------------
    - src_dict : dict
        Source dictionary from which to add keys.
    - target_dict : dict
        Target dictionary in which to add keys under the main key.
    - keys : list
        Keys to retrieve from the source dictionary.
    - main_key : str
        Main key under which to store new dictionary in target dictionary.

    Optional args
    -------------
    - raise_missing : bool (default=True)
        If True and some keys are missing from the source dictionary, an error 
        is raised. Otherwise, missing keys are ignored.

    Returns
    -------
    - src_dict : dict
        Copy of the input source dictionary, with keys removed if they were 
        copied to the target directory.
    """
    
    src_dict = copy.deepcopy(src_dict)

    if main_key in target_dict.keys():
        raise ValueError(
            f"{main_key} key already exists in the target dictionary."
            )
        
    target_dict[main_key] = dict()
    
    missing = []
    for key in keys:
        if key in src_dict.keys():
            target_dict[main_key][key] = src_dict.pop(key)
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

    Initializes tensorboard writers.

    Optional args
    -------------
    - direc : str or path (default=".") 
        Directory in which to initialize writer or wriers.
    - val : bool (default=False)
        If True, the validation writer (writer_val) is initialized. Otherwise, 
        it is set to None.

    Returns
    -------
    - writer_train : tensorboardX.SummaryWriter
        Tensorboard writer for training.
    - writer_val : tensorboardX.SummaryWriter or None
        Tensorboard writer for validation.
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

    Creates a model directory under the specified output directory, and returns 
    its path.

    Optional args
    -------------
    - output_dir : str or path (default=".")
        Directory in which to create model directory.

    Returns
    -------
    - model_direc : path
        Model directory, created under the output directory.
    """
    
    model_direc = Path(output_dir, "model")
    Path(model_direc).mkdir(exist_ok=True, parents=True)

    return model_direc


#############################################
def denorm_transform(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    denorm_transform()

    Returns a denormalization transform.

    Optional args
    -------------
    - mean : list (default=[0.485, 0.456, 0.406])
        List of mean values to use for the denormalization.
    - std : list (default=[0.229, 0.224, 0.225])
        List of mean values to use for the denormalization.

    Returns
    -------
    - transforms.Normalize
        Normalization transform to use for denormalization.
    """
    
    if len(mean) != len(std):
        raise ValueError(
            "'mean' and 'std' must comprise the same number of values."
            )
    inv_mean = [-mean[i] / std[i] for i in range(3)]
    inv_std = [1 / i for i in std]
    
    return transforms.Normalize(mean=inv_mean, std=inv_std)


#############################################
def write_input_seq_tb(writer, input_seq, n=2, i=0):
    """
    write_input_seq_tb(writer, input_seq)

    Writes input sequences to a tensorboard writer.

    Required args
    -------------
    - writer : tensorboardX.SummaryWriter
        Tensorboard writer to write sequence images to.
    - input_seq : 6D Tensor
        Sequences from which to select images to write to tensorboard writer, 
        after denormalization.

    Optional args
    -------------
    - n : int (default=2)
        Number of batch samples from which to record sequences.
    - i : int (default=0)
        Index under which to record sequences.
    """

    _, N, C, L, H, W = input_seq.shape
    writer.add_image(
        "input_seq", 
        denorm_transform(vutils.make_grid(
            input_seq[:n].transpose(2, 3).reshape(-1, C, H, W), 
            nrow=N * L)
        ), i
    )


#############################################
def update_tb(writer_train, train_dict, epoch_n=0, writer_val=None, 
              val_dict=None):
    """
    update_tb(writer_train, train_dict)

    Updates tensorboard writers from data dictionaries. 

    Required args
    -------------
    - writer_train : tensorboardX.SummaryWriter
        Tensorboard writer for the training data.
    - train_dict : dict
        Training dictionary, comprising loss and accuracy keys 
        (i.e., 'loss', 'acc', 'top{i}').

    Optional args
    -------------
    - epoch_n : int (default=0)
        Epoch number to tie to the data being written to the tensorboard.
    - writer_val : tensorboardX.SummaryWriter (default=None)
        Tensorboard writer for the validation data.
    - val_dict : dict (default=None)
        Validation dictionary, comprising loss and accuracy keys 
        (i.e., 'loss', 'acc', 'top{i}').
    """

    datatypes = ["global/loss", "global/accuracy"]

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

        topk_datatypes, topk_data = [], []
        for key, value in data_dict.items():
            if key.startswith("top"):
                topk_datatypes.append(f"accuracy/{key}")
                topk_data.append(value)
 
        all_data = [
            data_dict["loss"], 
            data_dict["acc"], 
            *topk_data
            ]

        use_datatypes = datatypes + topk_datatypes
        for datatype, data in zip(use_datatypes, all_data):
            writer.add_scalar(datatype, data, epoch_n)


#############################################
def get_technical_hyperparams():
    """
    get_technical_hyperparams()

    Returns a list of technical hyperparameter names.

    Returns
    -------
    - technical_params : list
        List of parameters to include in the technical hyperparameters 
        subdictionary.
    - ignore_params : list
        List of parameters to ignore when collecting hyperparameters.
    """

    technical_params = [
        "cpu_only", "data_parallel", "device_type", "num_workers", "plt_bkend", 
        "seed", "temp_data_dir", "use_tb"
        ]

    ignore_params = ["log_test_cmd"]

    return technical_params, ignore_params
    
    
#############################################
def get_model_hyperparams(supervised=False, resume=False, test=False):
    """
    get_model_hyperparams()

    Returns a list of model hyperparameter names.

    Optional args
    -------------
    - supervised : bool (default=False)
        If True, parameters not relevant to the supervised task are 
        ignored. Otherwise, those not relevant to the self-supervised task 
        are ignored.
    - resume : bool (default=False)
        If True, parameters not relevant to resuming model training are 
        ignored.
    - test : bool (default=False)
        If True, parameters not relevant to testing are ignored.

    Returns
    -------
    - model_params : list
        List of parameters to include in the model hyperparameters 
        subdictionary.
    - ignore_params : list
        List of parameters to ignore when collecting hyperparameters.
    """

    model_params = ["batch_size", "model", "net", "supervised"]
    ignore_params = []

    training_params = ["lr", "num_epochs", "train_what", "use_scheduler", "wd"]
    if test:
        model_params.append("test")
        ignore_params.extend(training_params)
        ignore_params.extend(
            ["dropout", "pred_step", "pretrained", "resume", "reset_lr"]
        )
    else:
        model_params.extend(training_params)
        ignore_params.append("test")
        if resume:
            model_params.extend(["reset_lr", "resume"])
            ignore_params.extend(["pretrained", "pred_step"])
        else:
            model_params.append("pretrained")
            ignore_params.extend(["reset_lr", "resume"])
            if supervised:
                ignore_params.append("pred_step")
            else:
                model_params.append("pred_step")

        if supervised:
            model_params.append("dropout")
        else:
            ignore_params.append("dropout")



    return model_params, ignore_params


#############################################
def get_dataset_hyperparams(dataset="UCF101"):
    """
    get_dataset_hyperparams()

    Returns a list of dataset hyperparameter names.

    Optional args
    -------------
    - dataset : str (default="UCF101")
        Dataset name, used to identify parameters to ignore or retain.

    Returns
    -------
    - dataset_params : list
        List of parameters to include in the dataset hyperparameters 
        subdictionary.
    - ignore_params : list
        List of parameters to ignore when collecting hyperparameters.
    """

    dataset_params = ["dataset", "img_dim", "num_seq_in", "no_augm", "seq_len"]
    ignore_params = ["diff_possizes", "no_gray", "num_seq"]

    if dataset in ["UCF101", "HMDB51", "MouseSim"]:
        dataset_params.append("ucf_hmdb_ms_ds")
    else:
        ignore_params.append("ucf_hmdb_ms_ds")        

    if dataset == "MouseSim":
        dataset_params.append("eye")
    else:
        ignore_params.append("eye")

    gabor_params = [
        "diff_U_possizes", "gab_img_len", "gray", "num_gabors", 
        "same_possizes", "roll", "train_len", "U_prob", "unexp_epoch"
        ]
    if dataset == "Gabors":
        dataset_params.extend(gabor_params)
    else:
        ignore_params.extend(gabor_params)
    
    return dataset_params, ignore_params


#############################################
def get_general_hyperparams(resume=False, test=False):
    """
    get_general_hyperparams()

    Returns a list of general hyperparameter names.

    Optional args
    -------------
    - resume : bool (default=False)
        If True, parameters not relevant to resuming model training are 
        ignored.
    - test : bool (default=False)
        If True, parameters not relevant to testing are ignored.

    Returns
    -------
    - general_params : list
        List of parameters to include in the general hyperparameters 
        subdictionary.
    - ignore_params : list
        List of parameters to ignore when collecting hyperparameters.
    """
    
    general_params = ["data_path_dir", "log_level", "output_dir"]
    ignore_params = ["not_save_best"]
    
    if test:
        ignore_params.extend(["log_freq", "save_best", "save_by_batch"])
    else:
        general_params.extend(["log_freq", "save_best", "save_by_batch"])

    if resume or test:
        ignore_params.extend(["overwrite", "suffix"])
    else:
        general_params.extend(["overwrite", "suffix"])

    return general_params, ignore_params


#############################################
def check_missed_hyperparams(hyperparams, ignore_params=[]):
    """
    check_missed_hyperparams(hyperparams)

    Checks for hyperparameters that are left over in the hyperparameters  
    dictionary. If any are not also listed in the parameters to ignore, a 
    warning is logged to the console. 

    Required args
    -------------
    - hyperparams : dict
        Hyperparameters dictionary containing any leftover keys.

    Optional args
    -------------
    - ignore_params : list (default=[])
        Hyperparameters in the dictionary that can be ignored, if left over.
    """

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


#############################################
def save_hyperparameters(hyperparams, direc=None):
    """
    save_hyperparameters(hyperparams)

    Creates a nested hyperparameters dictionary and saves it in a json file.

    Required args
    -------------
    - hyperparams : dict
        Dictionary containing hyperparameters to be stored (not nested).

    Optional args
    -------------
    - direc : str or path (default=None)
        Directory in which to save hyperparameters json file. If None, it is 
        inferred from hyperparams["output_dir"].
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
    ignore_params = []


    # set technical parameters
    technical_params, add_ignore_params = get_technical_hyperparams()

    hyperparams = add_nested_dict(
        hyperparams, hyperparam_dict, technical_params, main_key="technical", 
        raise_missing=False
        )
    ignore_params = ignore_params + add_ignore_params


    # set model parameters
    model_params, add_ignore_params = get_model_hyperparams(
        supervised=supervised, resume=resume, test=test
        )
    ignore_params = ignore_params + add_ignore_params

    hyperparams = add_nested_dict(
        hyperparams, hyperparam_dict, model_params, main_key="model", 
        raise_missing=False
        )


    # set dataset parameters
    dataset_params, add_ignore_params = get_dataset_hyperparams(dataset)
    ignore_params = ignore_params + add_ignore_params

    hyperparams = add_nested_dict(
        hyperparams, hyperparam_dict, dataset_params, main_key="dataset", 
        raise_missing=False
        )


    # general parameters
    general_params, add_ignore_params = get_general_hyperparams(
        resume=resume, test=test
        )
    ignore_params = ignore_params + add_ignore_params

    hyperparams = add_nested_dict(
        hyperparams, hyperparam_dict, general_params, main_key="general", 
        raise_missing=False
        )
    

    # convert Path objects to strings
    for main_key in hyperparam_dict.keys():
        for key in hyperparam_dict[main_key].keys():
            val = hyperparam_dict[main_key][key]
            if isinstance(val, Path):
                hyperparam_dict[main_key][key] = str(val)


    # final check
    check_missed_hyperparams(hyperparams, ignore_params=ignore_params)


    # save hyperparameters (as yaml for legibility, as the file is light-weight)
    base_name = "hyperparameters"
    if test:
        gab_unexp_str = ""
        dataset_dict = hyperparam_dict["dataset"]
        if "unexp" in dataset_dict.keys():
            gab_unexp_str = "_unexp" if dataset_dict["unexp"] else ""
        test_suffix = get_test_suffix(test)

        base_name = f"test_{base_name}_{test_suffix}{gab_unexp_str}"
        overwrite = True
    if resume:
        base_name = f"resume_{base_name}"
        overwrite = False

    hyperparams_path = get_unique_filename(
        Path(direc, f"{base_name}.yaml"), 
        overwrite=overwrite
        )

    Path(hyperparams_path).parent.mkdir(exist_ok=True, parents=True)

    with open(hyperparams_path, "w") as f:
        yaml.dump(hyperparam_dict, f)


#############################################
def load_hyperparameters(model_direc, prefix=None, suffix=None):
    """
    load_hyperparameters(model_direc)

    Loads hyperparameter dictionaries.

    Required args
    -------------
    - model_direc : str or Path
        Model directory from which to retrieve hyperparameters.
    
    Optional args
    -------------
    - prefix : str (default=None)
        Prefix with which to start the hyperparameters file, if applicable.
    - suffix : str (default=None)
        Suffix with which to end the hyperparameters file, if applicable.

    Returns
    -------
    - hp_dict : dict
        Nested hyperparameters dictionary.
    """

    if not Path(model_direc).is_dir():
        raise OSError(f"{model_direc} is not a directory")

    prefix_str = format_addendum(prefix, is_suffix=False)
    suffix_str = format_addendum(suffix, is_suffix=True)
    save_name = f"{prefix_str}hyperparameters{suffix_str}"

    hp_dict_path = Path(model_direc, f"{save_name}.yaml")

    if not hp_dict_path.is_file():
        raise OSError(f"{hp_dict_path} does not exist.")

    with open(hp_dict_path, "r") as f:
        hp_dict = yaml.safe_load(f)
    
    return hp_dict


#############################################
def get_test_suffix(test_path):
    """
    get_test_suffix(test_path)

    Returns a suffix to use to identify the model on which a test was run.

    Required args
    -------------
    - test_path : str or path
        Path to the model being tested.

    Returns
    -------
    - test_suffix : str
        Suffix to use to identify the model on which a test was run.
    """
    
    if test_path == "random":
        test_suffix = "random"
    else:
        test_suffix = Path(test_path).stem.split(".")[0]

    return test_suffix


#############################################
def update_resume_args(args, resume_dir):
    """
    update_resume_args(args, resume_dir)

    Identifies the checkpoint to resume from (last checkpoint recorded is best 
    for avoiding bugs, and ensuring consistency). 
    
    If possible, reverts certain args values based on the associated 
    hyperparameters file. Logs to the console the parameters that have been 
    reverted, and those that have been updated. 
    
    Parameters irrelevant to resuming training, and technical parameters, are 
    left as is.

    Required args
    -------------
    - args : argparse.Namespace
        Argparse namespace containing arguments used to set hyperparameters.
    - resume_dir : str or path
        Directory in which to search recursively for a model to resume from or
        path to the model to reload.

    Returns
    -------
    - args : argparse.Namespace
        Argparse namespace updated based on recorded hyperparameters.
    """

    args = copy.deepcopy(args)

    resume_dir = Path(resume_dir)
    if resume_dir.is_file():
        args.resume = resume_dir
        resume_dir = resume_dir.parent
        if resume_dir.stem == "model":
            resume_dir = resume_dir.parent

        if Path(resume_dir, "loss_data.json").is_file():
            raise OSError(
                "If resuming from a specific model, reloading from the loss "
                "data dictionary in the main directory will cause errors if "
                "the last recorded model is not reloaded.\nEither (1) specify "
                "the main model directory only, or (2) to resume specifically "
                "from this model, either place it in a new directory "
                "(optionally with the associated hyperparameters.json file), "
                "or use the 'pretrained' mode, instead of the 'resume' mode."
                )
    else:
        from utils.checkpoint_utils import find_last_checkpoint
        args.resume = find_last_checkpoint(resume_dir, raise_none=True)

    try:
        hp_dict = load_hyperparameters(args.resume.parent.parent)
    except OSError as err:
        warnings.warn(
            "The following error occurred will attempting to reload the "
            f"training hyperparameters to resume training: {err}\nTraining "
            "will be resumed without reloading. However, note that this can "
            "lead to bugs if the new parameters are not compatible with the "
            "original parameters used to initialize the model and associated "
            "files. The 'pretrained' mode may be preferred."
        )
        return args

    args_keys = args.__dict__.keys()

    # identify values to update or revert
    update_vals = []
    revert_vals = []
    for key, sub_dict in hp_dict.items():

        # technical: ignore all keys but "seed"
        if key == "technical" and "seed" in sub_dict.keys():
            if sub_dict["seed"] != args.seed: # update
                update_vals.append(("seed", sub_dict["seed"], args.seed))

        # dataset and model: go through all keys
        elif key in ["dataset", "model"]: # go through all
            update_keys = [
                "dropout", "lr", "num_epochs", "wd", "U_prob", "unexp_epoch"
                ]
            for subkey, item in sub_dict.items():
                if subkey in update_keys:
                    args_val = getattr(args, subkey)
                    if subkey == "wd" and args_val is None:
                        revert_vals.append([subkey, args_val, item])
                    elif item != args_val:
                        update_vals.append((subkey, item, args_val))

                elif subkey != "pretrained":
                    if subkey in args_keys:
                        args_val = getattr(args, subkey)
                        if item != args_val:
                            revert_vals.append((subkey, args_val, item))
                    setattr(args, subkey, item)

        # general: ignore some keys, reset some, and keep "save_best"
        elif key == "general":
            for subkey, item in sub_dict.items():
                if subkey == "save_best": # reinstate
                    if item != args.save_best:
                        revert_vals.append(("save_best", args.save_best, item))
                        args.save_best = item

    # log relevant changes
    log_str = ""
    changes_types = [
        "updated from their original settings to the newly specified ones", 
        "reverted back to their original settings"
        ]
    for change_type, vals in zip(changes_types, [update_vals, revert_vals]):
        if len(vals):
            log_str = f"{log_str}\n" if len(log_str) else log_str
            vals_str = f"\n{TAB}".join(
                [f"{key}: from {st} to {end}" for key, st, end in vals]
                )
            log_str = (
                f"{log_str}\nBased on the stored hyperparameters.json file, "
                f"the following values have been {change_type}:"
                f"\n{TAB}{vals_str}"
            )
    if len(log_str):
        logger.warning(log_str, extra={"spacing": "\n"})

    return args


#############################################
def log_test_cmd(args):
    """
    log_test_cmd(args)

    Logs to the console a command for running a test on the best model using 
    the same parameters. 

    Required args
    -------------
    - args : argparse.Namespace
        Argparse namespace containing arguments used to set hyperparameters.
    """
    
    if not args.supervised or args.test:
        return

    unexp = True if args.dataset == "Gabors" else False

    # get the model to test
    from utils.checkpoint_utils import find_last_checkpoint
    for best in [True, False]:
        test_checkpoint_path = find_last_checkpoint(
            args.output_dir, best=best, unexp=unexp, raise_none=False
            )
        if test_checkpoint_path is not None:
            break

    if test_checkpoint_path is None:
        return

    # collect the arguments to use
    args_dict = args.__dict__

    cmd = "python run_model.py"

    # mandatory test settings
    cmd = f"{cmd} --batch_size 1 --num_workers 1"

    # specific settings
    include = [
        "data_path_dir", "dataset", "img_dim", "log_level", "model", "net", 
        "num_seq_in", "plt_bkend", "seq_len",
        # "temp_data_dir" # not included, as it is expected to be temporary
        ]

    if args.seed is not None:
        include.append("seed")

    if args.dataset == "Gabors":
        include.extend(
            ["gab_img_len", "num_gabors", "train_len", "U_prob"]
            )
    elif args.dataset in ["HMDB51", "MouseSim", "UCF101"]:
        include.append("ucf_hmdb_ms_ds")
        if args.dataset == "MouseSim":
            include.append("eye")

    # append to cmd
    for key in include:
        val = args_dict[key]
        cmd = f"{cmd} --{key} {val}"


    include_bool = ["cpu_only", "no_augm", "use_tb"]
    if args.dataset == "Gabors":
        include_bool.extend(["diff_U_possizes", "roll"])

    # append to cmd
    for key in include_bool:
        val = args_dict[key]
        if val:
            cmd = f"{cmd} --{key}"

    if args.dataset == "Gabors":
        # check the converted (opposite) args, as they are definitely up to date
        if not args.same_possizes:
            cmd = f"{cmd} --diff_possizes"
        if not args.gray:
            cmd = f"{cmd} --no_gray"

        # set unexpected epoch to 0
        cmd = f"{cmd} --unexp_epoch 0"
    
    best_str = "best" if best else "last (no best found)"
    cmd = f"{cmd} --test {test_checkpoint_path}"
    logging.info(
        f"To test the {best_str} model, run:\n{cmd}", 
        extra={"spacing": "\n"}
        )

