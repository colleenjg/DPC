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


############################################
def check_adjust_args(args):

    args = copy.deepcopy(args)

    # check model, and add supervised argument
    if "dpc" in args.model:
        if args.model != "dpc-rnn":
            raise NotImplementedError(
                "Only 'dpc-rnn' model is implemented for Dense CPC."
                )
        args.supervised = False
    
    elif "lc" in args.model:
        if args.model != "lc":
            raise NotImplementedError(
                "Only 'lc' model is implemented for supervised learning."
                )
        args.supervised = True
    else:
        raise ValueError(f"{args.model} not recognized.")

    # check train_what argument
    if (args.train_what == "all" or 
        (args.supervised and args.train_what == "ft") or 
        (args.supervised and args.train_what == "last")
        ):
        raise ValueError(
            f"{args.train_what} value for train_what is not recognized "
            f"for {args.model} model."
            )

    # adjust if test is True
    if args.test:
        if args.supervised:
            logger.warning("Setting batch_size to 1.")
            args.batch_size = 1
            args.save_best = False
        else:
            args.test = False
    
    return args


#############################################
def set_path(args):

    if args.test == "random":
        i = 0
        while Path(args.output_dir, f"test_random_{i:03}").exists():
            i += 1
            if i > 999:
                raise NotImplementedError(
                    "Not implemented for 1000+ random tests."
                )
        output_dir = Path(args.output_dir, f"test_random_{i:03}")
    elif args.test:
        output_dir = Path(args.test).parent
    elif args.resume: 
        output_dir = Path(args.resume).parent.parent
    else:
        lr_str = args.old_lr if args.old_lr is not None else args.lr
        pretrained_str = ""
        if args.pretrained:
            pretrained_parts = [
                str(part) for part in Path(args.pretrained).parts
                ]
            pretrained_parts[-1] = str(Path(pretrained_parts[-1]).stem)
            pretrained_str = "-".join(pretrained_parts)
        
        save_name = (
            f"{args.dataset}-{args.img_dim}_r{args.net[6:]}_{args.model}_"
            f"bs{args.batch_size}_lr{lr_str}_seq{args.num_seq}_"
            f"pred{args.pred_step}_len{args.seq_len}_ds{args.ds}_"
            f"train-{args.train_what}_pt{pretrained_str}"
        )
        output_dir = Path(args.output_dir, save_name)

    return output_dir


#############################################
def get_dataloaders(args):
    """
    get_dataloaders(args)
    """

    data_kwargs = dict()
    add_keys = ["batch_size", "img_dim", "num_workers", "seed", "supervised"]
    if args.dataset == "gabors":
        dataset_keys = ["blank", "roll", "U_prob", "U_pos"]
        get_dataloader_fn = gabor_stimuli.get_dataloader
    else:
        dataset_keys = ["dataset", "seq_len", "ucf_hmdb_ds"]
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

    params = model.parameters()
    if not supervised and train_what == "last":
        logger.info("=> Training only the last layer")
        for name, param in model.module.resnet.named_parameters():
            param.requires_grad = False

    elif supervised and train_what == "ft":
        logger.info("=> Finetuning backbone with a smaller lr")
        params = []
        for name, param in model.module.named_parameters():
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

    if supervised:
        return None, None

    # set gradients
    params = set_gradients(model, supervised, train_what=train_what, lr=lr)

    # get optimizer and scheduler
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=wd)

    scheduler = None
    if supervised:
        lr_lambda = data_utils.get_lr_lambda(dataset, img_dim=img_dim)    
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    return optimizer, scheduler


#############################################
def run_training(args):
    """
    run_training(args)
    """

    args = check_adjust_args(args)

    model = get_model(args.supervised)

    # get device
    device, args.num_workers = training_utils.get_device(args.num_workers)
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

    output_dir = set_path(args)

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
        misc_utils.seed_all(args.seed)

    misc_utils.get_logger_with_basic_format(level=args.log_level)

    run_training(args)


#############################################
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", default="output_dir", 
        help="directory for saving files")
    parser.add_argument("--net", default="resnet18")
    parser.add_argument("--model", default="dpc-rnn")

    # data parameters
    parser.add_argument("--dataset", default="ucf101")
    parser.add_argument("--img_dim", default=128, type=int)
    parser.add_argument("--seq_len", default=5, type=int, 
        help="number of frames in each video block")
    parser.add_argument("--num_seq", default=8, type=int, 
        help="number of video blocks")
    parser.add_argument("--ucf_hmdb_ds", default=3, type=int, 
        help="frame downsampling rate for UCF and HMDB datasets")

    # training parameters
    parser.add_argument("--train_what", default="all")
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
    parser.add_argument("--wd", default=1e-5, type=float, help="weight decay")
    parser.add_argument("--not_save_best", default="store_true", 
        help="if True, best model is not identified and saved")
    parser.add_argument("--num_epochs", default=10, type=int, 
        help="number of total epochs to run")
    
    # pretrained/resuming parameters
    parser.add_argument("--resume", default="", 
        help="path of model to resume")
    parser.add_argument("--reset_lr", action="store_true", 
        help="if True, learning rate is reset when resuming training")
    parser.add_argument("--pretrained", default="", 
        help="path of pretrained model")

    # technical parameters
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--log_freq", default=5, type=int, 
        help="frequency at which to log output during training")
    parser.add_argument("--plt_bkend", default="agg", 
        help="matplotlib backend")
    parser.add_argument("--seed", default=-1, type=int, 
        help="seed to use (-1 for no seeding)")
    parser.add_argument("--use_tb", action="store_true", 
        help="if True, tensorboard is used")
    parser.add_argument('--log_level', default='info', 
                        help='logging level, e.g., debug, info, error')

    # unsupervised only
    parser.add_argument("--pred_step", default=1, type=int)

    # gabor arguments
    parser.add_argument("--unexp_epoch", default=10, type=int)
    parser.add_argument("--blank", default=False, type=bool)
    parser.add_argument("--roll", action="store_true")
    parser.add_argument("--U_prob", default=0.1, type=float)
    parser.add_argument("--U_pos", default="U")

    args = parser.parse_args()

    main(args)

