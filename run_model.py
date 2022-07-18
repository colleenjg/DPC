#!/usr/bin/env python

import argparse
import copy
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from model import run_manager
from utils import data_utils, misc_utils, training_utils

logger = logging.getLogger(__name__)

TAB = "    "

############################################
def check_adjust_args(args):
    """
    check_adjust_args(args)

    Verifies consistency between arguments, update values for consistency, and 
    initializes additional arguments, as applicable.

    - Normalizes dataset name
    - Sets arguments:
        - args.supervised from args.model
        - args.use_scheduler from args.supervised
    - Checks that values are accepted and inter-compatible for:
        - args.model
        - args.train_what
        - args.test, given args.supervised
    - Sets opposite arguments:
        - args.same_possizes from args.diff_possizes
        - args.gray from args.not_gray
    - If needed, sets default values for:
        - args.wd
    - If args.test is not False, updates values for:
        - args.num_epochs = 0
        - args.batch_size = 1
        - args.save_best = False


    Required args
    -------------
    - args : argparse.Namespace
        Argparse namespace containing arguments provided by command line, 
        needed to run the model training or testing.

    Returns
    -------
    - args : argparse.Namespace
        Argparse namespace containing input arguments, adjusted and check, as 
        needed.
    """

    args = copy.deepcopy(args)

    # normalize the dataset name
    args.dataset = misc_utils.normalize_dataset_name(args.dataset, short=False)

    # adjust Gabor arguments
    args.same_possizes = not(args.diff_possizes)
    args.gray = not(args.no_gray)

    # check model, and add supervised argument
    if "dpc" in args.model:
        if args.model not in ["dpc-rnn", "dpc"]:
            raise NotImplementedError(
                "Only the 'dpc-rnn' model is implemented for Dense CPC."
                )
        args.model = "dpc-rnn"
        args.supervised = False
    
    elif "lc" in args.model:
        if args.model not in ["lc-rnn", "lc"]:
            raise NotImplementedError(
                "Only the 'lc-rnn' model is implemented for supervised "
                "learning."
                )
        args.model = "lc-rnn"
        args.supervised = True
    else:
        raise ValueError(f"{args.model} not recognized.")

    # check train_what argument
    if args.train_what not in ["all", "ft", "last"]:
        raise ValueError(
            "args.train_what must be 'all', 'ft' or 'last', but found "
            f"{args.train_what}'."
            )

    # set weight decay, if needed
    if args.wd is None:
        if args.supervised:
            args.wd = 1e-3
        else:
            args.wd = 1e-5

    # set whether to use scheduler
    if args.supervised:
        args.use_scheduler = True
    else:
        args.use_scheduler = False

    # adjust if test is True
    if args.test:
        if args.supervised:
            logger.warning("Setting batch_size to 1.", extra={"spacing": TAB})
            args.num_epochs = 0
            args.batch_size = 1
            args.save_best = False
        else:
            raise ValueError("Test can only be done in the supervised setting.")
    
    return args


#############################################
def get_output_directory(args):
    """
    get_output_directory(args)

    Returns the path to the directory in which output data will be stored.

    If training a new model, optionally from a pretrained one, creates the 
    directory and returns its path. If resuming from or testing a model, 
    verifies that the directory exists, and raises an error if it doesn't.

    Required args
    -------------
    - args : argparse.Namespace
        Argparse namespace containing arguments needed to create or identify 
        the output directory.

    Returns
    -------
    - output_dir : path
        Path to the directory in which to save the model and associated results.
    """

    if args.test == "random":
        output_dir = misc_utils.get_unique_direc(
            Path(args.output_dir, "test_random"), overwrite=args.overwrite
            )
    elif args.test or args.resume:
        use_path = args.test if args.test else args.resume
        if not Path(use_path).exists():
            raise OSError(f"{use_path} does not exist.")
        elif not Path(use_path).is_file():
            raise OSError(f"{use_path} is not a file.")
        output_dir = Path(use_path).parent
        if output_dir.stem == "model":
            output_dir = Path(output_dir).parent            
    else:
        pretrained_str = ""
        if args.pretrained:
            pretrained_str = Path(
                Path(Path(args.pretrained).parts[-1]).stem
                ).stem
            pretrained_str = f"_pt-{pretrained_str}"
        
        dataset_str = misc_utils.normalize_dataset_name(
            args.dataset, short=True, eye=args.eye
            )
        
        seed_str = ""
        if args.seed is not None:
            seed_str = f"_seed{args.seed}"
        
        ds_str = ""
        if dataset_str in ["UCF101", "HMDB51"]:
            ds_str = f"_ds{args.ucf_hmdb_ms_ds}"

        pred_str = ""
        if not args.supervised:
            pred_str = f"_pred{args.pred_step}"

        suffix_str = ""
        if len(args.suffix):
            suffix_str = args.suffix
            if suffix_str[0] != "_":
                suffix_str = f"_{args.suffix}"

        save_name = (
            f"{dataset_str}-{args.img_dim}_r{args.net[6:]}_{args.model}_"
            f"bs{args.batch_size}_lr{args.lr}_nseq{args.num_seq}{pred_str}_"
            f"len{args.seq_len}{ds_str}_train-{args.train_what}"
            f"{seed_str}{pretrained_str}{suffix_str}"
        )

        output_dir = Path(args.output_dir, save_name)
        output_dir = misc_utils.get_unique_direc(
            output_dir, overwrite=args.overwrite
            )

    logger.info(
        f"Results will be saved to {output_dir}.", extra={"spacing": "\n"}
        )

    return output_dir


#############################################
def get_dataloaders(args):
    """
    get_dataloaders(args)

    Retrieves the main dataloader, and optionally the validation dataloader, 
    based on the arguments detailing dataset and dataloader parameters.

    Required args
    -------------
    - args : argparse.Namespace
        Argparse namespace containing arguments needed to initialize datasets 
        and their dataloaders.

    Returns
    -------
    - main_loader : torch data.DataLoader
        The main dataloader to be run 
        (to be used for training, or testing , if applicable).
    - val_loader : torch data.DataLoader or None
        Additional dataloader for validation. or None if args.save_best is 
        False.
    """

    dataset = misc_utils.normalize_dataset_name(args.dataset)

    data_kwargs = dict()
    add_keys = ["batch_size", "img_dim", "num_workers", "supervised"]
    dataset_keys = ["data_path_dir", "dataset", "seq_len", "num_seq"]
    data_kwargs["transform"] = "default"
    data_kwargs["no_augm"] = args.no_augm
    
    allow_flip = True
    if dataset == "Gabors":
        allow_flip = False
        gabor_keys = [
            "num_gabors", "gab_img_len", "same_possizes", "gray", "roll", 
            "U_prob", "diff_U_possizes", "train_len"
            ]
        dataset_keys = dataset_keys + gabor_keys
    elif dataset in ["UCF101", "HMDB51", "MouseSim"]:
        dataset_keys = dataset_keys + ["ucf_hmdb_ms_ds"]
        if dataset == "MouseSim":
            dataset_keys = dataset_keys + ["eye"]

    for key in add_keys + dataset_keys:
        data_kwargs[key] = args.__dict__[key]

    # adjust, if args.test is True
    mode = "test" if args.test else "train"
    if args.supervised:
        data_kwargs["transform"] = data_utils.get_transform(
            None, args.img_dim, mode=mode, no_augm=args.no_augm, 
            allow_flip=allow_flip
            )

    # get the main dataloader
    main_loader = data_utils.get_dataloader(
        mode=mode, seed=args.seed, **data_kwargs
        )
    
    # get the validation dataloader, if applicable
    val_loader = None
    if args.save_best:
        mode = "val"
        if args.supervised and not args.no_augm:
            data_kwargs["transform"] = data_utils.get_transform(
                None, args.img_dim, mode=mode, 
                no_augm=args.no_augm, allow_flip=allow_flip
                )
        val_seed = misc_utils.get_new_seed(args.seed)
        val_loader = data_utils.get_dataloader(
            mode=mode, seed=val_seed, **data_kwargs
            )

    return main_loader, val_loader


#############################################
def run_training_or_test(args):
    """
    run_training_or_test(args)

    Runs the model training or testing, as applicable.

    Required args
    -------------
    - args : argparse.Namespace
        Argparse namespace containing arguments needed to run the model 
        training or testing.
    """

    args = copy.deepcopy(args)

    args = check_adjust_args(args)

    args.output_dir = get_output_directory(args)

    # get device and dataloaders
    device, args.num_workers = training_utils.get_device(
        args.num_workers, args.cpu_only
        )
    args.device_type = str(device.type)

    main_loader, val_loader = get_dataloaders(args)

    # get model
    model = training_utils.get_model(
        dataset=main_loader.dataset, supervised=args.supervised, 
        img_dim=args.img_dim, num_seq=args.num_seq, seq_len=args.seq_len, 
        network=args.net, pred_step=args.pred_step, dropout=args.dropout
        )

    allow_data_parallel = training_utils.allow_data_parallel(
        main_loader, device, args.supervised
        )
    args.data_parallel = (args.device_type != "cpu" and allow_data_parallel)
    if args.data_parallel:
        model = torch.nn.DataParallel(model)

    model = model.to(device)

    # get optimizer and scheduler
    optimizer, scheduler = training_utils.init_optimizer(
        model, lr=args.lr, wd=args.wd, dataset=args.dataset, 
        img_dim=args.img_dim, supervised=args.supervised, 
        use_scheduler=args.use_scheduler, train_what=args.train_what, 
        test=args.test
    )

    # get whether to save by batch
    args.dataset = misc_utils.normalize_dataset_name(args.dataset)
    args.save_by_batch = (args.dataset == "Gabors")

    # save hyperparameters
    misc_utils.save_hyperparameters(args.__dict__)

    ### train DPC model
    reload_keys = ["resume", "pretrained", "test", "lr", "reset_lr"]
    reload_kwargs = {key: args.__dict__[key] for key in reload_keys}

    run_manager.train_full(
        main_loader, 
        model, 
        optimizer, 
        output_dir=args.output_dir,
        net_name=args.net,
        dataset=args.dataset,
        num_epochs=args.num_epochs,
        scheduler=scheduler,
        device=device, 
        val_loader=val_loader,
        seed=args.seed,
        unexp_epoch=args.unexp_epoch,
        log_freq=args.log_freq,
        use_tb=args.use_tb,
        save_by_batch=args.save_by_batch,
        reload_kwargs=reload_kwargs,
        )


#############################################
def main(args):
    """
    main(args)

    Runs the main code for model training and testing.

    Required args
    -------------
    - args : argparse.Namespace
        Argparse namespace containing the arguments passed as input to the 
        script.
    """

    args = copy.deepcopy(args)

    args.save_best = not(args.not_save_best)

    plt.switch_backend(args.plt_bkend)

    if args.seed == -1:
        args.seed = None
    else:
        misc_utils.seed_all(args.seed, deterministic_algorithms=False)

    misc_utils.get_logger_with_basic_format(level=args.log_level)

    run_training_or_test(args)


#############################################
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", default="output_dir", 
        help=("directory for saving files (ignored if resuming from or "
            "testing on a checkpoint)"))
    parser.add_argument("--net", default="resnet18")
    parser.add_argument("--model", default="dpc-rnn")
    parser.add_argument("--suffix", default="", 
        help=("suffix string to include in model folder name (ignored if "
            "resuming from or testing a model)"))  

    # data parameters
    parser.add_argument("--data_path_dir", default=Path("process_data", "data"), 
        help="path to files indexing datasets") 
    parser.add_argument("--dataset", default="UCF101")
    parser.add_argument("--img_dim", default=128, type=int)
    parser.add_argument("--seq_len", default=5, type=int, 
        help="number of frames in each video block")
    parser.add_argument("--num_seq", default=8, type=int, 
        help="number of video blocks")
    parser.add_argument("--ucf_hmdb_ms_ds", default=3, type=int, 
        help="frame downsampling rate for UCF, HMDB and MouseSim datasets")

    # training parameters
    parser.add_argument("--train_what", default="all", 
        help="'all', 'last' or 'ft' (supervised only)")
    parser.add_argument("--batch_size", default=24, type=int)
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
    parser.add_argument("--wd", default=None,
        help=("weight decay (different defaults for supervised and "
            "self-supervised)"))
    parser.add_argument("--not_save_best", action="store_true", 
        help="if True, best model is not identified and saved")
    parser.add_argument("--num_epochs", default=10, type=int, 
        help=("total number of epochs to run (in addition to epoch 0 which "
            "computes a pre-training baseline)"))
    parser.add_argument("--no_augm", action="store_true", 
        help="if True, no augmentation transforms are used")
    
    # pretrained/resuming parameters
    parser.add_argument("--resume", default="", 
        help="path of model to resume from")
    parser.add_argument("--reset_lr", action="store_true", 
        help="if True, learning rate is reset when resuming training")
    parser.add_argument("--pretrained", default="", 
        help="path to pretrained model to initialize model with")

    # technical parameters
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--log_freq", default=5, type=int, 
        help="batch frequency at which to log output during training")
    parser.add_argument("--plt_bkend", default="agg", 
        help="matplotlib backend")
    parser.add_argument("--seed", default=-1, type=int, 
        help="seed to use (-1 for no seeding)")
    parser.add_argument("--overwrite", action="store_true", 
        help=("if True, existing model directories are removed and "
            "overwritten (ignored if loading a checkpoint)"))
    parser.add_argument("--use_tb", action="store_true", 
        help="if True, tensorboard is used")
    parser.add_argument("--log_level", default="info", 
        help="logging level, e.g., debug, info, error")
    parser.add_argument("--cpu_only", action="store_true", 
        help="run on CPU (very slow)")

    # MouseSim only
    parser.add_argument("--eye", default="left", 
        help="which eye to include (left, right or both)")

    # Gabors only
    parser.add_argument("--num_gabors", default=30, type=int,
        help="number of Gabor patches per image")
    parser.add_argument("--gab_img_len", default=3, type=int,
        help="number of frames per image")
    parser.add_argument("--diff_possizes", action="store_true", 
        help=("if True, new positions/sizes are selected for each sequence. "
            "Otherwise, they are fixed at the start of each epoch."))
    parser.add_argument("--no_gray", action="store_true", 
        help="if True, no gray images are included in the sequences")
    parser.add_argument("--roll", action="store_true", 
        help="if True, sequences starting images are rolled")
    parser.add_argument("--unexp_epoch", default=5, type=int,
        help="epoch at which to introduce unexpected U images")
    parser.add_argument("--U_prob", default=0.1, type=float,
        help="frequency at which to include unexpected U images")
    parser.add_argument("--diff_U_possizes", action="store_true", 
        help="if True, U positions/sizes are different from D ones")
    parser.add_argument("--train_len", default=1000, type=int,
        help="size of Gabors training dataset")

    # supervised only
    parser.add_argument("--dropout", default=0.5, type=float, 
        help="dropout proportion for the final dropout layer")
    parser.add_argument("--test", default="", 
        help=("if not False, must be 'random' or the path to a saved model, "
            "and supervised network will be run in test mode"))

    # self-supervised only
    parser.add_argument("--pred_step", default=3, type=int)

    args = parser.parse_args()

    main(args)

