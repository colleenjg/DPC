import logging

import numpy as np
from torch.utils import data
from torchvision import transforms

from dataset import augmentations, dataset_3d


logger = logging.getLogger(__name__)


#############################################
def worker_init_fn(worker_id):
    """
    worker_init_fn(worker_id)
    """
    np.random.seed(np.random.get_state()[1][0] + worker_id)


#############################################
def get_transform(dataset, img_dim=256, mode="train"):
    """
    get_transform(dataset)
    """

    if dataset is not None:
        dataset = dataset_3d.normalize_dataset_name(dataset)
    
    if dataset == "UCF101": 
        # designed for ucf101: 
        # short size=256 -> rand crop to 224 x 224 -> scale to img_dim x img_dim 
        transform = transforms.Compose([
            augmentations.RandomHorizontalFlip(consistent=True),
            augmentations.RandomCrop(size=224, consistent=True),
            augmentations.Scale(size=(img_dim, img_dim)),
            augmentations.RandomGray(consistent=False, p=0.5),
            augmentations.ColorJitter(
                brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25, p=1.0
                ),
            augmentations.ToTensor(),
            augmentations.Normalize(),
        ])

    elif dataset == "Kinetics400": 
        # designed for kinetics400:
        # short size=150 -> rand crop to img_dim x img_dim
        transform = transforms.Compose([
            augmentations.RandomSizedCrop(size=img_dim, consistent=True, p=1.0),
            augmentations.RandomHorizontalFlip(consistent=True),
            augmentations.RandomGray(consistent=False, p=0.5),
            augmentations.ColorJitter(
                brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25, p=1.0
                ),
            augmentations.ToTensor(),
            augmentations.Normalize(),
        ])


    elif dataset == "HMDB51":
        raise NotImplementedError(
            "Dataset transform not implemented for HMDB51."
            ) 

    elif dataset is None:
        if mode in ["train", "val"]:
            crop_p = 1.0 if mode == "train" else 0.3
            bright_cont_sat = 0.5 if mode == "train" else 0.2
            hue = 0.25 if mode == "train" else 0.1

            transform = transforms.Compose([
                augmentations.RandomSizedCrop(consistent=True, size=224, p=crop_p),
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
            ])

        elif mode == "test":
            # -> rand crop to 224 x 224 -> scale to img_dim x img_dim
            transform = transforms.Compose([
                augmentations.RandomSizedCrop(consistent=True, size=224, p=0.0),
                augmentations.Scale(size=(img_dim, img_dim)),
                augmentations.ToTensor(),
                augmentations.Normalize(),
            ])
        
        else:
            raise ValueError("mode must be 'train', 'val' or 'test'.")
    
    else:
        raise ValueError(f"'{dataset}' dataset is not recognized.")

    return transform
    
    

#############################################
def get_dataloader(data_path_dir="process_data", transform=None, 
                   dataset="kinetics400", mode="train", 
                   batch_size=4, img_dim=128, seq_len=5, num_seq=8, 
                   ucf_hmdb_ds=3, split_n=1, supervised=False, 
                   num_workers=4):
    """
    get_dataloader()
    """
    
    logger.info(f"Loading {mode} data...", extra={"spacing": "\n"})

    dataset = dataset_3d.normalize_dataset_name(dataset)

    if transform == "default":
        transform = get_transform(dataset, img_dim, mode=mode)

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
            )

    elif dataset == "UCF101":
        dataset = dataset_3d.UCF101_3d(
            data_path_dir=data_path_dir,
            mode=mode,
            transform=transform,
            seq_len=seq_len,
            num_seq=num_seq,
            downsample=ucf_hmdb_ds,
            split_n=split_n,
            supervised=supervised,
            )
    
    elif dataset == "HMDB51":
        dataset = dataset_3d.HMDB51_3d(
            data_path_dir=data_path_dir,
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
            "get_data_loader() only implemented for the 'Kinetics400', "
            "'UCF101', and 'HMDB51', datasets."
            )

    sampler = data.RandomSampler(dataset)

    data_loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=True,
        drop_last=True
        )
    
    logger.info(f"{mode.capitalize()} dataset size: {len(dataset)}.")

    return data_loader

