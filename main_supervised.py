import argparse
import copy
import logging
from pathlib import Path
import time

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

import matplotlib.pyplot as plt

from dataset import dataset_3d, data_utils
from model import model_3d, training_utils

logger = logging.getLogger(__name__)


def train_epoch(data_loader, model, optimizer, epoch_n=0, num_epochs=10, 
                log_idx=0, topk=data_utils.TOPK, device="cpu", 
                decay_log_freq=5, writer=None):

    losses = training_utils.AverageMeter()
    topk_meters = [training_utils.AverageMeter() for _ in range(topk)]
    model = model.to(device)
    model.train()

    criterion = data_utils.CRITERION_FCT()

    for idx, (input_seq, target) in enumerate(data_loader):
        start_time = time.perf_counter()
        input_seq = input_seq.to(device)
        target = target.to(device) # class index
        B = input_seq.size(0)
        output, _ = model(input_seq)

        # visualize 2 examples from the batch
        if writer is not None and idx % decay_log_freq == 0:
            training_utils.write_input_seq(writer, input_seq, n=2, i=log_idx)
        del input_seq

        # separate sequences with shared label
        B, N, num_classes = output.size()
        output = output.view(B * N, num_classes)
        target = target.repeat(1, N).view(-1)
        
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update meters, in-place
        losses.update(loss.item(), B)
        training_utils.update_topk_meters(topk_meters, output, target, ks=topk)
        del target 

        if idx % decay_log_freq == 0:
            training_utils.decay_weights(model)

        # log results
        if idx % decay_log_freq == 0:
            loss_avg, acc_avg, loss_val, acc_val, log_str = \
                training_utils.get_stats(
                    losses, topk_meters, ks=topk, local=True, 
                    last=True
                )

            logger.info(
                f"Epoch: [{epoch_n}/{num_epochs - 1}]"
                f"[{idx}/{len(data_loader)}]\t"
                f"{log_str}\tT:{time.perf_counter() - start_time:.2f}"
                )

            if writer is not None:
                writer.add_scalar("local/loss", loss_val, log_idx)
                writer.add_scalar("local/accuracy", acc_val, log_idx)

            log_idx += 1

    return loss_avg, acc_avg, topk_meters


def val_or_test_epoch(data_loader, model, epoch_n=0, num_epochs=10, 
                      topk=data_utils.TOPK, device="cpu", shared_pred=False, 
                      save_dir=None):

    losses = training_utils.AverageMeter()
    topk_meters = [training_utils.AverageMeter() for _ in range(topk)]
    model = model.to(device)
    model.eval()

    if save_dir is not None:
        confusion_mat = training_utils.ConfusionMeter(model.num_classes)
        Path(save_dir).mkdir(exist_ok=True)

    criterion = data_utils.CRITERION_FCT()
    with torch.no_grad():
        for _, (input_seq, target) in tqdm(enumerate(data_loader)):
            input_seq = input_seq.to(device)

            n_dim = len(input_seq.size())
            SUB_B = None
            if n_dim == 7: 
                # if the dataset is set to 'supervised' and 'test mode', 
                # each batch item is a sub-batch in which all sets of sequences 
                # share a label
                B, SUB_B, N, C, SL, H, W = input_seq.size()
                input_seq.view(B * SUB_B, N, C, SL, H, W)
                USE_N = SUB_B * N # sequences that share a label

            elif n_dim == 6:
                B, N, C, SL, H, W = input_seq.size()
                USE_N = N
                
            else:
                raise RuntimeError(
                    "input_seq should have 6 dims: B x N x C x SL x H x W"
                    )
            
            target = target.to(device) # class index for each batch item
            output, _ = model(input_seq)
            del input_seq

            num_classes = output.size()[-1]
            
            if shared_pred:
                # group sequences that share a label
                output = output.view(B, USE_N, num_classes)
                # for each batch item, average the softmaxed class predictions 
                # across sequences
                output = torch.mean(
                    torch.nn.functional.softmax(output, 2),
                    1) # B x num_classes
            else:
                # consider all sequences separately, even if they share a label 
                output = output.view(B * USE_N, num_classes)
                target = target.repeat(1, USE_N).view(-1) # B * USE_N
            
            # in-place update of accuracy meters
            training_utils.update_topk_meters(
                topk_meters, output, target, ks=topk
                )
            loss = criterion(output, target)

            losses.update(loss.item(), B)

            _, pred = torch.max(output, 1)
            confusion_mat.update(pred, target.view(-1).byte())

    loss_avg, acc_avg, log_str = training_utils.get_stats(
        losses, topk_meters, ks=topk, local=False
        )

    logger.info(f"Epoch: [{epoch_n}/{num_epochs - 1}] {log_str}")

    if save_dir is not None:
        confusion_mat.plot_mat(Path(save_dir, "confusion_matrix.svg"))
        training_utils.write_log(
            content=log_str,
            epoch=epoch_n,
            filename=Path(save_dir, "test_log.md")
            )

    return loss_avg, acc_avg, topk_meters


def train_full(args, train_loader, model, optimizer, scheduler=None, 
               topk=data_utils.TOPK, device="cpu", val_loader=None):

    model = model.to(device)

    log_idx, best_acc, start_epoch = data_utils.load_checkpoint(
        model, optimizer, resume=args.resume, pretrained=args.pretrained, 
        test=args.test, lr=args.lr, reset_lr=args.reset_lr
        )

    # setup tools
    img_path, model_path = data_utils.set_path(args)
    writer_train = None
    if args.use_tb:
        from tensorboardX import SummaryWriter
        writer_train = SummaryWriter(logdir=str(Path(img_path, "train")))
        
    ### main loop ###
    Path(args.save_dir).mkdir(exist_ok=True)

    for epoch_n in range(start_epoch, args.num_epochs):
    
        train_epoch(
            train_loader, 
            model, 
            optimizer, 
            epoch_n=epoch_n, 
            num_epochs=args.num_epochs,
            log_idx=log_idx, 
            topk=topk,
            device=device, 
            decay_log_freq=args.decay_log_freq,
            writer=writer_train,
            )

        if args.save_best:
            _, val_acc, _ = val_or_test_epoch(
                val_loader, 
                model, 
                epoch_n=epoch_n, 
                num_epochs=args.num_epochs, 
                topk=topk,
                device=device
                )
            is_best = val_acc > best_acc
            best_acc = max(val_acc, best_acc)
        else:
            is_best = False
            best_acc = None

        if scheduler is not None:
            scheduler.step(epoch_n)

        # save checkpoint
        epoch_path = Path(model_path, f"epoch{epoch_n}.pth.tar")
        training_utils.save_checkpoint(
            {
            "epoch_n": epoch_n,
            "net": args.net,
            "state_dict": model.state_dict(),
            "best_acc": best_acc,
            "optimizer": optimizer.state_dict(),
            "log_idx": log_idx
            }, 
            is_best, 
            filename=epoch_path, 
            keep_all=False
        )
           
    logger.info(
        f"Training from ep {args.start_epoch} to {args.num_epochs} finished."
        )


def run_supervised(args):

    ### get number of classes ###
    if args.dataset == "gabors":
        raise NotImplementedError(
            "Supervised learning not implemented for Gabors dataset."
            )
    else:
        num_classes = dataset_3d.get_num_classes(args.dataset)

    ### get model ###
    if args.model != "lc":
        raise NotImplementedError(
            "Only 'lc' model is implemented for supervised learning."
            )

    model = model_3d.LC(
        sample_size=args.img_dim, 
        num_seq=args.num_seq, 
        seq_len=args.seq_len, 
        network=args.net,
        num_classes=num_classes,
        dropout=args.dropout
        )
    
    ### get device ###
    device, num_workers = training_utils.get_device(args.num_workers)
    model = torch.nn.DataParallel(model)
    model = model.to(device)

    ### set parameters ### 
    params = None
    if args.train_what == "ft":
        logger.info("=> Finetuning backbone with a smaller lr")
        params = []
        for name, param in model.module.named_parameters():
            if ("resnet" in name) or ("rnn" in name):
                params.append({"params": param, "lr": args.lr / 10})
            else:
                params.append({"params": param})
    else: 
        pass # train all layers
    
    logger.debug("\n===========Check Grad============")
    for name, param in model.named_parameters():
        logger.debug(name, param.requires_grad)
    logger.debug("=================================\n")

    ### prepare dataloader arguments
    data_kwargs = {
        "dataset"     : args.dataset,
        "img_dim"     : args.img_dim,
        "seq_len"     : args.seq_len,
        "num_seq"     : args.num_seq,
        "ucf_hmdb_ds" : args.ucf_hmdb_ds,
        "split_n"     : args.split_n,
        "supervised"  : True,
        "num_workers" : num_workers,
        "seed"        : args.seed,
    }

    if not args.test: # train supervised model
        if params is None: 
            params = model.parameters()

        data_kwargs["batch_size"] = args.batch_size

        optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wd)
        lr_lambda = data_utils.get_lr_lambda(args.dataset, img_dim=args.img_dim)    
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        transform = data_utils.get_transform(None, args.img_dim, mode="train")
        train_loader = data_utils.get_dataloader(
            transform, mode="train", **data_kwargs
            )

        val_loader = None
        if args.save_best:
            val_transform = data_utils.get_transform(
                None, args.img_dim, mode="val"
                )
            val_loader = data_utils.get_dataloader(
                val_transform, mode="val", **data_kwargs
                )

        train_full(
            args, 
            train_loader, 
            model, 
            optimizer, 
            scheduler=scheduler,
            topk=data_utils.TOPK,
            device=device, 
            val_loader=val_loader
            )

    else: # test model trained on supervised task
        logger.warning("Setting batch_size to 1.")
        data_kwargs["batch_size"] = 1

        _, _, start_epoch = data_utils.load_checkpoint(model, test=args.test)
        transform = data_utils.get_transform(None, args.img_dim, mode="test")
        test_loader = data_utils.get_data_loader(
            transform, mode="test", **data_kwargs
            )

        if args.test == "random":
            i = 0
            while Path(args.save_dir, f"test_random_{i:03}").exists():
                i += 1
                if i > 999:
                    raise NotImplementedError(
                        "Not implemented for 1000+ random tests."
                    )
            save_dir = Path(args.save_dir, f"test_random_{i:03}")
        else:
            save_dir = Path(args.test).parent
    
        val_or_test_epoch(
            test_loader, 
            model, 
            epoch_n=start_epoch, 
            num_epochs=start_epoch + 1,
            topk=data_utils.TOPK,
            device=device,
            shared_pred=True,
            save_dir=save_dir, 
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

    run_supervised(args)
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # model parameters
    parser.add_argument("--net", default="resnet18")
    parser.add_argument("--model", default="lc")
    parser.add_argument("--save_dir", default="save_dir", 
        help="directory for saving files")

    # learning parameters
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--num_epochs", default=10, type=int, 
        help="total number of epochs to run")
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
    parser.add_argument("--wd", default=1e-5, type=float, help="weight decay")
    parser.add_argument("--train_what", default="all")
    parser.add_argument("--not_save_best", default="store_true", 
        help="do not identify and save the best model during training")

    # data parameters
    parser.add_argument("--dataset", default="ucf101")
    parser.add_argument("--img_dim", default=128, type=int)
    parser.add_argument("--seq_len", default=5, type=int, 
        help="number of frames in each video sequence")
    parser.add_argument("--num_seq", default=8, type=int, 
        help="number of sequences drawn from the same video")
    parser.add_argument("--ucf_hmdb_ds", default=3, type=int, 
        help="frame downsampling rate for UCF and HMDB datasets")

    # pretrained / resuming
    parser.add_argument("--pretrained", default="", 
        help="path to the pretrained model")
    parser.add_argument("--resume", default="", 
        help="path to the model to resume from")
    parser.add_argument("--reset_lr", action="store_true", 
        help="if True, learning rate is reset when resuming training")

    # technical parameters
    parser.add_argument("--seed", default=-1, type=int, 
        help="seed to use (-1 for no seeding)")
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--decay_log_freq", default=5, type=int, 
        help=("frequency at which to decay weights and log output during "
            "training"))
    parser.add_argument("--use_tb", action="store_true", 
        help="if True, tensorboard is used")
    parser.add_argument("--debug", action="store_true", 
        help="if True, extra information is logged to the console")
    parser.add_argument("--plt_bkend", default="agg", 
        help="matplotlib backend")

    # supervised only
    parser.add_argument("--split_n", default=1, type=int, 
        help="data split number")
    parser.add_argument("--dropout", default=0.5, type=float)
    parser.add_argument("--test", default=None, 
        help="Path to model to use for testing, if any, (or 'random').")

    # gabor arguments
    parser.add_argument("--unexpected_epoch_n", default=10, type=int, 
        help="epoch number at which to introduce unexpected Gabor sequences")
    parser.add_argument("--incl_blank", action="store_true", 
        help="include blank frames")
    parser.add_argument("--roll", action="store_true", 
        help="use rolling sequences")
    parser.add_argument("--U_prob", default=0.1, type=float, 
        help=("probability of a Gabor sequence including an unexpected U "
        "frame instead of an expected D frame"))
    parser.add_argument("--U_pos", default="U", 
        help="Gabor patch positions to use for U frames")

    # gabor and supervised only
    parser.add_argument("--gabor_classe_dim", default="ori", 
        help="dimension along which to classify Gabor sequences, e.g. mean "
        "orientation or exp/unexp value for the most recent sequence"
        )


    args = parser.parse_args()

    main(args)
