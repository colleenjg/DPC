import logging
import random

import numpy as np
import torch
from torch.utils import data
from torchvision import transforms

from dataset import augmentations, dataset_3d, gabor_stimuli
from utils import misc_utils


logger = logging.getLogger(__name__)


#############################################
def seed_workers(worker_id):
    """
    seed_workers(worker_id)

    Function for deterministically re-seeding the np.random and random modules 
    when individual data loader workers are intialized. 

    See: https://pytorch.org/docs/stable/notes/randomness.html#dataloader

    Required args
    -------------
    - worker_id : int
        Data loader worker ID (ignored).
    """

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed) # used for augmentations


#############################################
def get_transform(dataset=None, img_dim=256, mode="train", no_augm=False, 
                  allow_flip=True):
    """
    get_transform(dataset)

    Returns a torch Transform for the given dataset type.

    Optional args
    -------------
    - dataset : str (default=None)
        Dataset name, or None.
    - img_dim : int (default=256)
        Final image dimensions.
    - mode : str (default="train")
        Dataset mode (i.e., "train", "val" or "test").
    - no_augm : bool (default=False)
        If True, no augmentations are applied (other than the basics: 
        scaling, converting to tensor and normalizing).
    - allow_flip : bool (default=True)
        If True, horizontal flips are allowed among augmentations.

    Returns
    -------
    - transform : torch Transform
        torch Transforms for the dataset
    """

    if dataset is not None:
        dataset = misc_utils.normalize_dataset_name(dataset)

    if no_augm:
        transform_list = [
            augmentations.Scale(size=(img_dim, img_dim)),
            augmentations.ToTensor(),
            augmentations.Normalize(),
        ]

    elif dataset in ["UCF101", "HMDB51", "MouseSim", "Gabors"]: 
        # designed for ucf101 and hmdb51: 
        # short size is 256 for ucf101, and 240 for hmdb51 
        # -> rand crop to 224 x 224 -> scale to img_dim x img_dim 
        transform_list = [
            augmentations.RandomHorizontalFlip(consistent=True),
            augmentations.RandomCrop(size=224, consistent=True),
            augmentations.Scale(size=(img_dim, img_dim)),
            augmentations.RandomGray(consistent=False, p=0.5),
            augmentations.ColorJitter(
                brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25, p=1.0
                ),
            augmentations.ToTensor(),
            augmentations.Normalize(),
        ]

        if dataset == "Gabors":
            allow_flip = False

    elif dataset == "Kinetics400": 
        # designed for kinetics400:
        # short size=150 -> rand crop to img_dim x img_dim
        transform_list = [
            augmentations.RandomSizedCrop(size=img_dim, consistent=True, p=1.0),
            augmentations.RandomHorizontalFlip(consistent=True),
            augmentations.RandomGray(consistent=False, p=0.5),
            augmentations.ColorJitter(
                brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25, p=1.0
                ),
            augmentations.ToTensor(),
            augmentations.Normalize(),
        ]
    
    elif dataset == "Gabors":
        # no flip (to maintain orientations)
        # -> rand crop to 224 x 224 -> scale to img_dim x img_dim 
        transform_list = [
            augmentations.RandomCrop(size=224, consistent=True),
            augmentations.Scale(size=(img_dim, img_dim)),
            augmentations.RandomGray(consistent=False, p=0.5),
            augmentations.ColorJitter(
                brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25, p=1.0
                ),
            augmentations.ToTensor(),
            augmentations.Normalize(),
        ]    

    elif dataset is None: # e.g., used for supervised learning
        if mode in ["train", "val"]:
            crop_p = 1.0 if mode == "train" else 0.3
            bright_cont_sat = 0.5 if mode == "train" else 0.2
            hue = 0.25 if mode == "train" else 0.1

            transform_list = [
                augmentations.RandomSizedCrop(
                    consistent=True, size=224, p=crop_p
                    ),
                augmentations.Scale(size=(img_dim, img_dim)),
                augmentations.RandomHorizontalFlip(consistent=True),
                augmentations.ColorJitter(
                    brightness=bright_cont_sat, 
                    contrast=bright_cont_sat, 
                    saturation=bright_cont_sat, 
                    hue=hue, 
                    p=0.3, 
                    consistent=True
                    ),
                augmentations.ToTensor(),
                augmentations.Normalize(),
            ]

        elif mode == "test":
            # -> rand crop to 224 x 224 -> scale to img_dim x img_dim
            transform_list = [
                augmentations.RandomSizedCrop(consistent=True, size=224, p=0.0),
                augmentations.Scale(size=(img_dim, img_dim)),
                augmentations.ToTensor(),
                augmentations.Normalize(),
            ]
        
        else:
            raise ValueError("mode must be 'train', 'val' or 'test'.")
    
    else:
        raise ValueError(f"{dataset} dataset is not recognized.")
    
    if not allow_flip: # remove any flips
        transform_list = [
            transform for transform in transform_list 
            if not isinstance(transform, augmentations.RandomHorizontalFlip)
        ]

    transform = transforms.Compose(transform_list)

    return transform
    
    

#############################################
def get_dataloader(data_path_dir="process_data", transform=None, 
                   dataset="UCF101", mode="train", eye="both",
                   batch_size=4, img_dim=128, seq_len=5, num_seq=8, 
                   ucf_hmdb_ms_ds=3, split_n=1, supervised=False, 
                   num_workers=4, no_augm=False, seed=None, 
                   **gabor_kwargs):
    """
    get_dataloader()

    Returns a dataloader for a specific dataset. 

    Optional args
    -------------
    - data_path_dir : str or path (default="process_data")
        Directory where the data is stored (only needd if dataset is not Gabors)
    - transform : torch Transform or None (default=None)
        torch Transforms for the dataset
    - dataset : str (default="UCF101")
        Dataset naes
    - mode : str (default="train")
        Dataset mode (i.e., "train", "val" or "test").
    - eye : str (default="both")
        Eye(s) to which data should be cropped, if the dataset is MouseSim.
    - batch_size : int (default=4)
        Data loader batch size.
    - img_dim : int (default=128)
        Final image dimensions, used for determining which size version if the 
        Kinetics400 dataset should be used.
    - seq_len : int (default=5)
        Length of individual sequences to generate for the dataset.
    - num_seq : int (default=8)
        Number of sequences to generate per batch item for the dataset.
    - ucf_hmdb_ms_ds : bool (default=3)
        Level of temporal downsampling to use.
    - split_n : int (default=1)
        Dataset train/val(/test) split to use.
    - supervised : bool (default=False)
        If True, data loader is initialized for a supervised task.
    - num_workers : int (default=4)
        Number of workers that data loader should spawn.
    - no_augm : bool (default=False)
        If True, no augmentations are applied (other than the basics: 
        scaling, converting to tensor and normalizing).
    - seed : int (default=None)
        Seed for seeding the sampling generator for the dataloader.

    Keyword args
    ------------
    - gabor_kwargs : dict
        Additional initialization arguments for 
        gabor_stimuli.GaborSequenceGenerator

    Returns
    -------
    - dataloader : torch data.DataLoader
    """
    
    logger.info(f"Loading {mode} data...", extra={"spacing": "\n"})

    dataset = misc_utils.normalize_dataset_name(dataset)

    if mode == "test" and not supervised:
        raise ValueError(
            "'test' mode can only be used in a supervised context."
            )

    drop_last = True
    if transform == "default":
        transform = get_transform(
            dataset, img_dim, mode=mode, no_augm=no_augm,
            allow_flip=(dataset != "Gabors")
            )

    if dataset == "Kinetics400":
        use_big_K400 = img_dim > 140
        dataset = dataset_3d.Kinetics400_3d(
            data_path_dir=data_path_dir,
            mode=mode,
            transform=transform,
            seq_len=seq_len,
            num_seq=num_seq,
            downsample=5,
            big=use_big_K400,
            supervised=supervised,
            return_label=True,
            )

    elif dataset == "UCF101":
        dataset = dataset_3d.UCF101_3d(
            data_path_dir=data_path_dir,
            mode=mode,
            transform=transform,
            seq_len=seq_len,
            num_seq=num_seq,
            downsample=ucf_hmdb_ms_ds,
            split_n=split_n,
            supervised=supervised,
            return_label=True,
            )
    
    elif dataset == "HMDB51":
        dataset = dataset_3d.HMDB51_3d(
            data_path_dir=data_path_dir,
            mode=mode,
            transform=transform,
            seq_len=seq_len,
            num_seq=num_seq,
            downsample=ucf_hmdb_ms_ds,
            split_n=split_n,
            supervised=supervised,
            return_label=True,
            )

    elif dataset == "MouseSim":
        drop_last = False # Note: False is not compatible with DataParallel
        dataset = dataset_3d.MouseSim_3d(
            data_path_dir=data_path_dir,
            eye=eye,
            mode=mode,
            transform=transform,
            seq_len=seq_len,
            num_seq=num_seq,
            downsample=ucf_hmdb_ms_ds,
            supervised=supervised,
            return_label=True,
            ) 

    elif dataset == "Gabors":
        use_mode = mode
        if mode == "test":
            logger.warning(
                "Gabors dataset will be set to 'val' mode for testing, as "
                "shared supervised targets for the extended sequences "
                "generated in the 'test' mode have not been implemented."
                )
            use_mode = "val"

        dataset = gabor_stimuli.GaborSequenceGenerator(
            unexp=False,
            mode=use_mode,
            transform=transform,
            seq_len=seq_len,
            num_seq=num_seq,
            supervised=supervised,
            return_label=True,
            **gabor_kwargs
            ) 

    else:
        raise NotImplementedError(
            "get_dataloader() only implemented for the 'Kinetics400', "
            "'UCF101', 'HMDB51', 'MouseSim', and 'Gabors' datasets."
            )

    sampler = data.RandomSampler(dataset)
    generator = misc_utils.get_torch_generator(seed)

    dataloader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        generator=generator,
        worker_init_fn=seed_workers,
        pin_memory=True,
        drop_last=drop_last
        )
    
    logger.info(f"{mode.capitalize()} dataset size: {len(dataset)}.")

    return dataloader

