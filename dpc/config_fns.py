import logging
from pathlib import Path
import re
import sys
import time

import torch
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm

sys.path.extend(["..", str(Path("..", "utils")), str(Path("..", "backbone"))])
import dataset_3d
from resnet_2d3d import neq_load_customized
import augmentation
import utils
import gabor_stimuli


# a few global variables
CRITERION_FCT = torch.nn.CrossEntropyLoss
TOPK = [1, 3, 5]

logger = logging.getLogger(__name__)


def get_transform(dataset, img_dim=256, mode="train"):
    
    if dataset == "ucf101": 
        # designed for ucf101: 
        # short size=256 -> rand crop to 224 x 224 -> scale to img_dim x img_dim 
        transform = transforms.Compose([
            augmentation.RandomHorizontalFlip(consistent=True),
            augmentation.RandomCrop(size=224, consistent=True),
            augmentation.Scale(size=(img_dim, img_dim)),
            augmentation.RandomGray(consistent=False, p=0.5),
            augmentation.ColorJitter(
                brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25, p=1.0
                ),
            augmentation.ToTensor(),
            augmentation.Normalize(),
        ])

    elif dataset == "kinetics400": 
        # designed for kinetics400:
        # short size=150 -> rand crop to img_dim x img_dim
        transform = transforms.Compose([
            augmentation.RandomSizedCrop(size=img_dim, consistent=True, p=1.0),
            augmentation.RandomHorizontalFlip(consistent=True),
            augmentation.RandomGray(consistent=False, p=0.5),
            augmentation.ColorJitter(
                brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25, p=1.0
                ),
            augmentation.ToTensor(),
            augmentation.Normalize(),
        ])

    elif dataset is None:
        if mode in ["train", "val"]:
            crop_p = 1.0 if mode == "train" else 0.3
            bright_cont_sat = 0.5 if mode == "train" else 0.2
            hue = 0.25 if mode == "train" else 0.1

            transform = transforms.Compose([
                augmentation.RandomSizedCrop(consistent=True, size=224, p=crop_p),
                augmentation.Scale(size=(img_dim, img_dim)),
                augmentation.RandomHorizontalFlip(consistent=True),
                augmentation.ColorJitter(
                    brightness=bright_cont_sat, 
                    contrast=bright_cont_sat, 
                    saturation=bright_cont_sat, 
                    hue=hue, 
                    p=0.3, 
                    consistent=True
                    ),
                augmentation.ToTensor(),
                augmentation.Normalize(),
            ])

        elif mode == "test":
            # -> rand crop to 224 x 224 -> scale to img_dim x img_dim
            transform = transforms.Compose([
                augmentation.RandomSizedCrop(consistent=True, size=224, p=0.0),
                augmentation.Scale(size=(img_dim, img_dim)),
                augmentation.ToTensor(),
                augmentation.Normalize(),
            ])
        
        else:
            raise ValueError("mode must be 'train', 'val' or 'test'.")

    return transform
    
    
def get_dataloader(transform, dataset="kinetics400", mode="train", 
                   batch_size=4, img_dim=128, seq_len=5, num_seq=8, 
                   ucf_hmdb_ds=3, split_n=1, supervised=False, 
                   num_workers=4):
    
    logger.info(f"Loading {mode} data...")
    if dataset == "kinetics400":
        use_big_K400 = img_dim > 140
        dataset = dataset_3d.Kinetics400_full_3d(
            mode=mode,
            transform=transform,
            seq_len=seq_len,
            num_seq=num_seq,
            downsample=5,
            big=use_big_K400,
            supervised=supervised,
            )

    elif dataset == "ucf101":
        dataset = dataset_3d.UCF101_3d(
            mode=mode,
            transform=transform,
            seq_len=seq_len,
            num_seq=num_seq,
            downsample=ucf_hmdb_ds,
            split_n=split_n,
            supervised=supervised,
            )
    
    elif dataset == "hmdb51":
        dataset = dataset_3d.HMDB51_3d(
            mode=mode,
            transform=transform,
            seq_len=seq_len,
            num_seq=num_seq,
            downsample=ucf_hmdb_ds,
            split_n=split_n,
            supervised=supervised,
            )        

    else:
        raise NotImplementedError(
            "get_data_loader() only implemented for the 'kinetics400', "
            "'ucf101', and 'hmbd51', datasets."
            )

    sampler = data.RandomSampler(dataset)

    data_loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=utils.worker_init_fn,
        pin_memory=True,
        drop_last=True
        )
    
    logger.info(f"{mode.capitalize()} dataset size: {len(dataset)}.")

    return data_loader



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
                model = neq_load_customized(model, checkpoint["state_dict"])
            logger.info(
                f"=> Loaded {reload_str} checkpoint '{reload_model}' "
                f"(epoch {checkpoint['epoch']})"
                )
        else: 
            logger.warning(f"=> No checkpoint found at '{reload_model}'")
    
    return iteration, best_acc, start_epoch


def process_output(mask):
    """task mask as input, compute the target for contrastive loss"""
    # dot product is computed on parallel gpus, so get less easy neg, bounded 
    # by batch size in each gpu
    # mask meaning: -2: omit, -1: temporal neg (hard), 0: easy neg, 
    # 1: pos, -3: spatial neg
    (B, NP, SQ, B2, NS, _) = mask.size() # [B, P, SQ, B, N, SQ]
    target = mask == 1
    target.requires_grad = False
    return target, (B, B2, NS, NP, SQ)


def set_path(args):

    if args.resume: 
        exp_path = Path(args.resume).parent.parent
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

        exp_path = Path(args.savedir, save_name)

    img_path = Path(exp_path, "img")
    model_path = Path(exp_path, "model")
    img_path.mkdir(parent=True, exist_ok=True)
    model_path.mkdir(parent=True, exist_ok=True)

    return img_path, model_path


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

