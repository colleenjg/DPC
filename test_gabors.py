import argparse
import copy
import logging
from pathlib import Path
import time

import json
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

import matplotlib.pyplot as plt

from dataset import dataset_3d, data_utils
from model import model_3d, training_utils

logger = logging.getLogger(__name__)


def train_epoch(data_loader, model, optimizer, epoch_n=0, num_epochs=50, 
                iteration=0, device="cpu", log_freq=5, writer=None):
    
    topk = data_utils.TOPK
    losses = training_utils.AverageMeter()
    topk_meters = [training_utils.AverageMeter() for _ in range(topk)]
    model = model.to(device)
    model.train()

    detailed_loss = []
    loss_foreach_dict = {}
    dot_foreach = {}
    target_foreach = {}

    criterion = data_utils.CRITERION_FCT()

    for idx, input_seq in enumerate(data_loader):
        start_time = time.time()
        input_seq = input_seq.to(device)
        B = input_seq.size(0)
        [score_, mask_] = model(input_seq)

        logger.debug("Model called next.")
        logger.debug(
            f"Input sequence shape: {input_seq.shape} "
            "(expecting [10, 4, 3, 5, 128, 128])."
            )
        logger.debug(
            f"Score shape: {score_.shape} (expecting a 6D tensor: "
            "[B, P, SQ, B, N, SQ]"
            )
        logger.debug(f"Mask shape: {mask_.shape}")
        
        if idx == 0: 
            target_, (_, B2, NS, NP, SQ) = data_utils.process_output(mask_)
        
        # score is a 6d tensor: [B, P, SQ, B, N, SQ]
        score_flattened = score_.view(B*NP*SQ, B2*NS*SQ)
        target_flattened = target_.view(B*NP*SQ, B2*NS*SQ).to(device)
        target_flattened = target_flattened.to(int).argmax(dim=1)

        loss = criterion(score_flattened, target_flattened)
        top1, top3, top5 = train_utils.calc_topk_accuracy(
            score_flattened, target_flattened, (1, 3, 5)
            )

        accuracy_list[0].update(top1.item(), B)
        accuracy_list[1].update(top3.item(), B)
        accuracy_list[2].update(top5.item(), B)

        losses.update(loss.item(), B)
        accuracy.update(top1.item(), B)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        del loss

        criterion_measure = data_utils.CRITERION_FCT(reduction="none")       
        loss_foreach_dict[idx] = criterion_measure(
            score_flattened, target_flattened
            ).view(B, SQ).mean(axis=1).to("cpu").tolist()
        
        dot_foreach[idx] = score_.to("cpu").tolist()
        target_foreach[idx] = target_.to("cpu").tolist()
        
        del score_

        if idx % log_freq == 0:
            logger.info(
                f"Epoch: [{epoch_n}/{num_epochs - 1}][{idx}/{len(data_loader)}]\t"
                f"Loss: {losses.val:.6f} ({losses.local_avg:.4f})\t"
                f"Acc: top1 {top1:.4f}; top3 {top3:.4f}; top5 {top5:.4f}"
                f"T:{time.time() - start_time:.2f}\t"
                )
            
            if writer is not None:
                writer.add_scalar("local/loss", losses.val, iteration)
                writer.add_scalar("local/accuracy", accuracy.val, iteration)

            logger.debug(f"Previous sequence: {data_loader.prev_seq[-1]}")
            logger.debug(
                f"Batch sequences: {data_loader.prev_seq[-B:]}"
                )
            logger.debug(f"Batch loss: {loss_foreach_dict[idx]}")

            detailed_loss.append(losses.val)

            iteration += 1
    
    training_dict = {
        "train_loss"       : losses.local_avg,
        "train_acc"        : accuracy.local_avg,
        "train_acc_list"   : [i.local_avg for i in accuracy_list],
        "detailed_loss"    : detailed_loss, 
        "loss_foreach_dict": loss_foreach_dict,
        "dot_foreach"      : dot_foreach,
        "target_foreach"   : target_foreach,
    }

    return training_dict, iteration


def val_epoch(data_loader, model, epoch_n=0, num_epochs=10, device="cpu"):
    losses = train_utils.AverageMeter()
    topk_meters = [train_utils.AverageMeter() for _ in range(data_utils.TOPK)]
    model = model.to(device)
    model.eval()

    criterion = data_utils.CRITERION_FCT()

    with torch.no_grad():
        for idx, input_seq in tqdm(
            enumerate(data_loader), total=len(data_loader)
            ):
            input_seq = input_seq.to(device)
            B = input_seq.size(0)
            [score_, mask_] = model(input_seq)
            del input_seq

            if idx == 0: 
                target_, (_, B2, NS, NP, SQ) = data_utils.process_output(mask_)

            # [B, P, SQ, B, N, SQ]
            score_flattened = score_.view(B*NP*SQ, B2*NS*SQ)
            target_flattened = target_.view(B*NP*SQ, B2*NS*SQ).to(device)
            target_flattened = target_flattened.to(int).argmax(dim=1)
            
            loss = criterion(score_flattened, target_flattened)

            train_utils.update_topk_meters(
                topk_meters, score_flattened, target_flattened, 
                ks=data_utils.TOPK
                )

            losses.update(loss.item(), B)

    topk_str = ", ".join(
        [f"top{k} {topk_meter.avg:.4f}" 
        for (k, topk_meter) in zip(data_utils.TOPK, topk_meters)]
        )

    logger.info(
        f"Epoch: [{epoch_n}/{num_epochs - 1}] Loss: {losses.local_avg:.4f}\t"
        f"Acc: {topk_str}"
        )
    
    accuracy = topk_meters[data_utils.TOPK.index(1)]
    
    return losses.local_avg, accuracy.local_avg, topk_meters


def train_full(args, train_loader, model, optimizer, scheduler=None, 
               device="cpu", val_loader=None):

    model = model.to(device)

    iteration, best_acc, start_epoch = data_utils.load_checkpoint(
        model, optimizer, resume=args.resume, pretrained=args.pretrained, 
        test=args.test, lr=args.lr, reset_lr=args.reset_lr
        )

    # setup tools
    img_path, model_path = data_utils.set_path(args) # was global
    writer_train, writer_val = None, None
    if args.use_tb:
        from tensorboardX import SummaryWriter
        writer_train = SummaryWriter(logdir=str(Path(img_path, "train")))
        writer_val = SummaryWriter(logdir=str(Path(img_path, "val")))
    
    loss_dict = {"Training" : {},
                 "Validation" : {}}
    
    detailed_loss_dict = {}
    loss_foreach_full_dict = {}
    dot_foreach_dict = {}
    target_foreach_dict = {}
    
    ### main loop ###
    Path(args.save_dir).mkdir(exist_ok=True)

    for epoch_n in range(start_epoch, args.num_epochs):
        if args.dataset == "gabors":
            if not train_loader.unexp and epoch_n > args.unexpected_epoch:
                train_loader.unexp = True
                logger.info(f"mode: {train_loader.mode}")
    
        training_dict = train_epoch(
            train_loader, 
            model, 
            optimizer, 
            epoch_n=epoch_n, 
            num_epochs=args.num_epochs,
            iteration=iteration, 
            device=device, 
            log_freq=args.log_freq,
            )

        loss_dict["Training"][epoch_n] = training_dict["train_loss"]
        loss_foreach_full_dict[epoch_n] = training_dict["loss_foreach_dict"]
        dot_foreach_dict[epoch_n] = training_dict["dot_foreach"]        
        target_foreach_dict[epoch_n] = training_dict["target_foreach"]
        detailed_loss_dict[epoch_n] = training_dict["detailed_loss"]
        
        if args.save_best:
            val_loss, val_acc, val_acc_list = val_epoch(
                val_loader, 
                model, 
                epoch_n=epoch_n, 
                num_epochs=args.num_epochs, 
                device=device
                )
            is_best = val_acc > best_acc
            best_acc = max(val_acc, best_acc)
        else:
            is_best = False
            best_acc = None

        if scheduler is not None:
            scheduler.step(epoch_n)

        # Save to json
        data_names = ["loss", "loss_foreach", "dot_foreach", "target_for_each"]
        data_to_save = [
            detailed_loss_dict, 
            loss_foreach_full_dict, 
            dot_foreach_dict, 
            target_foreach_dict
            ]
        
        unexpected_epoch = "_"
        if args.dataset == "gabors":
            data_names.append("seq")
            data_to_save.append(train_loader.prev_seq)
            unexpected_epoch = f"{args.unexpected_epoch}_"

        for data, name in zip(data_to_save, data_names):
            full_path = Path(
                args.save_dir, 
                f"{name}_{unexpected_epoch}{args.seed}.json"
                )
            with json.load(full_path, "w"):
                json.dump(data, full_path)

        if args.use_tb:
            modes = ["train"]
            if args.save_best:
                modes.append("val")
            datatypes = [
                "global/loss", 
                "global/accuracy", 
                "accuracy/top1", 
                "accuracy/top3", 
                "accuracy/top5"
                ]

            for mode in modes:
                if mode == "train":
                    writer = writer_train
                    all_data = [
                        training_dict["train_loss"], 
                        training_dict["train_acc"], 
                        *training_dict["train_acc_list"]
                        ]
                elif mode == "val":
                    writer = writer_val
                    all_data = [val_loss, val_acc, *val_acc_list]
                for datatype, data in zip(datatypes, all_data):
                    writer.add_scalar(datatype, data, epoch_n)

        logger.debug(
            f"Epoch training loss: {loss_dict['Training'][epoch_n]}"
            )
        if args.save_best:
            logger.debug(f"Epoch validation loss: {val_loss}")

        # save checkpoint
        epoch_path = Path(model_path, f"epoch{epoch_n}.pth.tar")
        train_utils.save_checkpoint(
            {
            "epoch_n": epoch_n,
            "net": args.net,
            "state_dict": model.state_dict(),
            "best_acc": best_acc,
            "optimizer": optimizer.state_dict(),
            "iteration": iteration
            }, 
            is_best, 
            filename=epoch_path, 
            keep_all=False
        )
           
    logger.info(
        f"Training from ep {args.start_epoch} to {args.num_epochs} finished"
        )


def run_DPC(args):

    if args.model != "dpc-rnn":
        raise NotImplementedError(
            "Only 'dpc-rnn' model is implemented for Dense CPC."
            )

    ### get model ###
    model = model_3d.DPC_RNN(
        sample_size=args.img_dim, 
        num_seq=args.num_seq, 
        seq_len=args.seq_len, 
        network=args.net, 
        pred_step=args.pred_step
        )

    ### get device ###
    device, num_workers = train_utils.get_device(args.num_workers)
    model = torch.nn.DataParallel(model)
    model = model.to(device)

    ### set parameters ###
    if args.train_what == "last":
        for name, param in model.module.resnet.named_parameters():
            param.requires_grad = False
    else: 
        pass # train all layers

    logger.debug("\n===========Check Grad============")
    for name, param in model.named_parameters():
        logger.debug(name, param.requires_grad)
    logger.debug("=================================\n")

    params = model.parameters()
    
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wd)

    ### prepare dataloader arguments
    data_kwargs = {
        "batch_size"  : args.batch_size,
        "img_dim"     : args.img_dim,
        "num_seq"     : args.num_seq,
        "supervised"  : True,
        "num_workers" : num_workers,
        "seed"        : args.seed,
    }

    if args.dataset == "gabors":
        data_kwargs["blank"]  = args.blank
        data_kwargs["roll"]   = args.roll
        data_kwargs["U_prob"] = args.U_prob
        data_kwargs["U_pos"]  = args.U_pos

        get_data_loader_fn = gabor_stimuli.get_data_loader
    else:
        transform = data_utils.get_transform(args.dataset, args.img_dim)

        data_kwargs["transform"]   = transform
        data_kwargs["dataset"]     = args.dataset
        data_kwargs["seq_len"]     = args.seq_len
        data_kwargs["ucf_hmdb_ds"] = args.ucf_hmdb_ds

        get_data_loader_fn = data_utils.get_data_loader

    train_loader = get_data_loader_fn(mode="train", **data_kwargs)
    val_loader = None
    if args.save_best:
        val_loader = get_data_loader_fn(mode="val", **data_kwargs)

    ### train DPC model
    train_full(
        args, 
        train_loader, 
        model, 
        optimizer, 
        device=device, 
        val_loader=val_loader
        )


def main(args):

    args = copy.deepcopy(args)

    args.save_best = not(args.not_save_best)

    plt.switch_backend(args.plt_bkend)

    if args.seed == -1:
        args.seed = None
    else:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

    if args.debug:
        logger.setLevel(logging.DEBUG)

    run_DPC(args)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", default="save_dir", 
        help="directory for saving files")
    parser.add_argument("--net", default="resnet18")
    parser.add_argument("--model", default="dpc-rnn")
    parser.add_argument("--dataset", default="ucf101")
    parser.add_argument("--seq_len", default=5, type=int, 
        help="number of frames in each video block")
    parser.add_argument("--num_seq", default=8, type=int, 
        help="number of video blocks")
    parser.add_argument("--ucf_hmdb_ds", default=3, type=int, 
        help="frame downsampling rate for UCF and HMDB datasets")
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
    parser.add_argument("--wd", default=1e-5, type=float, help="weight decay")
    parser.add_argument("--not_save_best", default="store_true", 
        help="if True, best model is not identified and saved")
    parser.add_argument("--resume", default="", 
        help="path of model to resume")
    parser.add_argument("--pretrained", default="", 
        help="path of pretrained model")
    parser.add_argument("--num_epochs", default=10, type=int, 
        help="number of total epochs to run")
    parser.add_argument("--start-epoch", default=0, type=int, 
        help="manual epoch number (useful on restarts)")
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--log_freq", default=5, type=int, 
        help="frequency at which to log output during training")
    parser.add_argument("--reset_lr", action="store_true", 
        help="if True, learning rate is reset when resuming training")
    parser.add_argument("--prefix", default="tmp", 
        help="prefix of checkpoint filename")
    parser.add_argument("--train_what", default="all")
    parser.add_argument("--img_dim", default=128, type=int)
    parser.add_argument("--plt_bkend", default="agg", 
        help="matplotlib backend")
    parser.add_argument("--seed", default=-1, type=int, 
        help="seed to use (-1 for no seeding)")
    parser.add_argument("--debug", action="store_true", 
        help="if True, extra information is logged to the console")
    parser.add_argument("--use_tb", action="store_true", 
        help="if True, tensorboard is used")

    # unsupervised only
    parser.add_argument("--pred_step", default=1, type=int)

    # gabor arguments
    parser.add_argument("--unexpected_epoch", default=10, type=int)
    parser.add_argument("--blank", default=False, type=bool)
    parser.add_argument("--roll", action="store_true")
    parser.add_argument("--U_prob", default=0.1, type=float)
    parser.add_argument("--U_pos", default="U")


    args = parser.parse_args()

    main(args)
