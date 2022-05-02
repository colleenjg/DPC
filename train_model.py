#!/usr/bin/env python

import argparse
import copy
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.optim as optim

from dataset import gabor_stimuli, dataset_3d
from model import model_3d, run_manager
from utils import data_utils, misc_utils, training_utils

logger = logging.getLogger(__name__)

TAB = "    "

############################################
def check_adjust_args(args):
    """
    check_adjust_args(args)
    """

    args = copy.deepcopy(args)

    # normalize the dataset name
    args.dataset = dataset_3d.normalize_dataset_name(args.dataset, short=False)

    # check model, and add supervised argument
    if "dpc" in args.model:
        if args.model != "dpc-rnn":
            raise NotImplementedError(
                "Only the 'dpc-rnn' model is implemented for Dense CPC."
                )
        args.supervised = False
    
    elif "lc" in args.model:
        if args.model != "lc":
            raise NotImplementedError(
                "Only the 'lc' model is implemented for supervised learning."
                )
        args.supervised = True
    else:
        raise ValueError(f"{args.model} not recognized.")

    # check train_what argument
    if (args.train_what == "all" or 
        (args.supervised and args.train_what in ["ft", "last"])
        ):
        pass
    else:
        raise ValueError(
            f"'{args.train_what}' value for train_what is not recognized "
            f"for {args.model} model."
            )

    # set weight decay, if needed
    if args.wd is None:
        if args.supervised:
            args.wd = 1e-3
        else:
            args.wd = 1e-5

    # adjust if test is True
    if args.test:
        if args.supervised:
            logger.warning("Setting batch_size to 1.", extra={"spacing": TAB})
            args.batch_size = 1
            args.save_best = False
        else:
            args.test = False
    
    return args


#############################################
def set_path(args):
    """
    set_path(args)
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
        
        dataset_str = dataset_3d.normalize_dataset_name(
            args.dataset, short=True
            )
        
        seed_str = ""
        if args.seed is not None:
            seed_str = f"_seed{args.seed}"
        
        ds_str = ""
        if dataset_str in ["UCF101", "HMDB51"]:
            ds_str = f"_ds{args.ucf_hmdb_ds}" 

        save_name = (
            f"{dataset_str}-{args.img_dim}_r{args.net[6:]}_{args.model}_"
            f"bs{args.batch_size}_lr{args.lr}_nseq{args.num_seq}_"
            f"pred{args.pred_step}_len{args.seq_len}{ds_str}_"
            f"train-{args.train_what}{seed_str}{pretrained_str}"
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
    """

    data_kwargs = dict()
    add_keys = ["batch_size", "img_dim", "num_workers", "supervised"]
    if args.dataset == "gabors":
        dataset_keys = ["blank", "roll", "U_prob", "U_pos", "seed"]
        get_dataloader_fn = gabor_stimuli.get_dataloader
    else:
        dataset_keys = ["data_path_dir", "dataset", "seq_len", "ucf_hmdb_ds"]
        data_kwargs["transform"] = "default"
        get_dataloader_fn = data_utils.get_dataloader

    for key in add_keys + dataset_keys:
        data_kwargs[key] = args.__dict__[key]

    # adjust, if args.test is True
    mode = "test" if args.test else "train"
    if args.supervised:
        data_kwargs["transform"] = data_utils.get_transform(
            None, args.img_dim, mode=mode
            )

    # get the main dataloader
    main_loader = get_dataloader_fn(mode=mode, **data_kwargs)
    
    # get the validation dataloader, if applicable
    val_loader = None
    if args.save_best:
        mode = "val"
        if args.supervised:
            data_kwargs["transform"] = data_utils.get_transform(
                None, args.img_dim, mode=mode
                )
        val_loader = get_dataloader_fn(mode=mode, **data_kwargs)

    return main_loader, val_loader


#############################################
def get_model(supervised=False):
    """
    get_model()
    """

    if supervised:
        # get number of classes
        if args.dataset == "gabors":
            num_classes = gabor_stimuli.get_num_classes()
        else:
            num_classes = dataset_3d.get_num_classes(args.dataset)

        model = model_3d.LC(
            sample_size=args.img_dim, 
            num_seq=args.num_seq, 
            seq_len=args.seq_len, 
            network=args.net,
            num_classes=num_classes,
            dropout=args.dropout
            )
    
    else:
        model = model_3d.DPC_RNN(
            sample_size=args.img_dim, 
            num_seq=args.num_seq, 
            seq_len=args.seq_len, 
            network=args.net, 
            pred_step=args.pred_step
            )

    return model


#############################################
def set_gradients(model, supervised=False, train_what="all", lr=1e-3):
    """
    set_gradients(model)
    """

    model = training_utils.get_model(model)

    params = model.parameters()
    if train_what == "last":
        logger.info("=> Training only the last layer")
        if supervised:
            for name, param in model.named_parameters():
                if ("resnet" in name) or ("rnn" in name):
                    param.requires_grad = False
        else:
            for name, param in model.resnet.named_parameters():
                param.requires_grad = False

    elif supervised and train_what == "ft":
        logger.info("=> Finetuning backbone with a smaller learning rate")
        params = []
        for name, param in model.named_parameters():
            if ("resnet" in name) or ("rnn" in name):
                params.append({"params": param, "lr": lr / 10})
            else:
                params.append({"params": param})
    
    elif train_what != "all":
        raise ValueError(
            f"{train_what} value for train_what is not recognized."
            )
    else: 
        pass # train all layers

    training_utils.check_grad(model)

    return params


#############################################
def init_optimizer(model, lr=1e-3, wd=1e-5, dataset="gabors", img_dim=128, 
                   supervised=False, train_what="all", test=False):
    """
    init_optimizer(model)
    """

    if test:
        return None, None

    # set gradients
    params = set_gradients(model, supervised, train_what=train_what, lr=lr)

    # get optimizer and scheduler
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=wd)

    scheduler = None
    if supervised:
        lr_lambda = training_utils.get_lr_lambda(dataset, img_dim=img_dim)    
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    return optimizer, scheduler


#############################################
def run_training(args):
    """
    run_training(args)
    """

    args = check_adjust_args(args)

    output_dir = set_path(args)

    model = get_model(args.supervised)

    # get device
    device, args.num_workers = training_utils.get_device(
        args.num_workers, args.cpu_only
        )
    if device.type != "cpu":
        model = torch.nn.DataParallel(model)
    
    model = model.to(device)

    # get optimizer and scheduler
    optimizer, scheduler = init_optimizer(
        model, lr=args.lr, wd=args.wd, dataset=args.dataset, 
        img_dim=args.img_dim, supervised=args.supervised, 
        train_what=args.train_what, test=args.test
    )

    # get dataloaders
    main_loader, val_loader = get_dataloaders(args)

    ### train DPC model
    reload_keys = ["resume", "pretrained", "test", "lr", "reset_lr"]
    reload_kwargs = {key: args.__dict__[key] for key in reload_keys}

    run_manager.train_full(
        main_loader, 
        model, 
        optimizer, 
        output_dir=output_dir,
        net_name=args.net,
        dataset=args.dataset,
        num_epochs=args.num_epochs,
        scheduler=scheduler,
        device=device, 
        val_loader=val_loader,
        seed_info=args.seed,
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
    """

    args = copy.deepcopy(args)

    args.save_best = not(args.not_save_best)

    plt.switch_backend(args.plt_bkend)

    if args.seed == -1:
        args.seed = None
    else:
        misc_utils.seed_all(args.seed, deterministic_algorithms=False)

    misc_utils.get_logger_with_basic_format(level=args.log_level)

    run_training(args)


#############################################
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", default="output_dir", 
        help="directory for saving files (ignored if loading a checkpoint)")
    parser.add_argument("--net", default="resnet18")
    parser.add_argument("--model", default="dpc-rnn")

    # data parameters
    parser.add_argument("--data_path_dir", default=Path("process_data", "data"), 
        help="path to files indexing datasets") 
    parser.add_argument("--dataset", default="UCF101")
    parser.add_argument("--img_dim", default=128, type=int)
    parser.add_argument("--seq_len", default=5, type=int, 
        help="number of frames in each video block")
    parser.add_argument("--num_seq", default=8, type=int, 
        help="number of video blocks")
    parser.add_argument("--ucf_hmdb_ds", default=3, type=int, 
        help="frame downsampling rate for UCF and HMDB datasets")

    # training parameters
    parser.add_argument("--train_what", default="all", 
        help="'all', 'last' or 'ft' (supervised only)")
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
    parser.add_argument("--wd", default=None,
        help="weight decay (different defaults for supervised and self-supervised)")
    parser.add_argument("--not_save_best", action="store_true", 
        help="if True, best model is not identified and saved")
    parser.add_argument("--num_epochs", default=10, type=int, 
        help=("total number of epochs to run (in addition to epoch 0 which "
            "computes a pre-training baseline)"))
    parser.add_argument("--save_by_batch", action="store_true",
        help="if True, detailled batch/loss information is saved")
    
    # pretrained/resuming parameters
    parser.add_argument("--resume", default="", 
        help="path of model to resume")
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
    parser.add_argument('--log_level', default='info', 
                        help='logging level, e.g., debug, info, error')
    parser.add_argument('--cpu_only', action='store_true', 
                        help='run on CPU (very slow)')

    # supervised only
    parser.add_argument("--dropout", default=0.5, type=float, 
        help="dropout proportion")
    parser.add_argument("--test", default=False, 
        help=("if not False, must be 'random' or the path to a saved model, "
            "and supervised network will be run in test mode"))

    # unsupervised only
    parser.add_argument("--pred_step", default=3, type=int)

    # gabor arguments
    parser.add_argument("--unexp_epoch", default=10, type=int)
    parser.add_argument("--blank", default=False, type=bool)
    parser.add_argument("--roll", action="store_true")
    parser.add_argument("--U_prob", default=0.1, type=float)
    parser.add_argument("--U_pos", default="U")

    args = parser.parse_args()

    main(args)

