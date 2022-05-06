import logging
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils import data
from tqdm import tqdm

logger = logging.getLogger(__name__)

TAB = "    "


#############################################
def pil_loader(path_name):
    """
    pil_loader(path_name)
    """

    with open(path_name, "rb") as f:
        with Image.open(f) as img:
            return img.convert("RGB")


#############################################
def normalize_dataset_name(dataset_name="UCF101", short=False):

    if dataset_name.lower() == "ucf101":
        dataset_name = "UCF101"
    elif dataset_name.lower() == "hmdb51":
        dataset_name = "HMDB51"
    elif dataset_name.lower() in ["kinetics400", "k400"]:
        dataset_name = "k400" if short else "Kinetics400"
    elif dataset_name.lower() == "gabors":
        dataset_name = "Gabors"
    else:
        raise ValueError(f"{dataset_name} dataset not recognized.")

    if short:
        dataset_name = dataset_name.lower()
    
    return dataset_name


#############################################
def get_num_classes(dataset=None, dataset_name="UCF101"):
    """
    get_num_classes()
    """

    if dataset is not None:
        if hasattr(dataset, "action_dict_decode"):
            num_classes = len(dataset.action_dict_decode)
        else:
            raise NotImplementedError(
                "Cannot retrieve number of classes for the dataset of type "
                f"{type(dataset)}."
                )
    else:
        dataset_name = normalize_dataset_name(dataset_name)

        if dataset_name == "UCF101":
            num_classes = 101
        elif dataset_name == "HMDB51":
            num_classes = 51
        elif dataset_name == "Kinetics400":
            num_classes = 400
        else:
            raise ValueError(
                f"Cannot retrieve number of classes for {dataset_name}."
                )

    return num_classes


#############################################
class GeneralDataset(data.Dataset):
    def __init__(self,
                 data_path_dir,
                 mode="train",
                 dataset_name="UCF101",
                 transform=None,
                 seq_len=10,
                 num_seq=5,
                 downsample=3,
                 epsilon=5,
                 unit_test=False,
                 split_n=1,
                 return_label=False,
                 supervised=False,
                 seed=233, # used in init only
                 ):

        self.data_path_dir = data_path_dir
        self.mode = mode
        self.dataset_name = dataset_name
        self.supervised = supervised
        self.transform = transform
        self.seq_len = seq_len
        self.num_seq = num_seq
        self.downsample = downsample
        self.epsilon = epsilon
        self.unit_test = unit_test
        self.split_n = split_n

        if supervised:
            return_label = True

        self.return_label = return_label
        self._set_action_dicts()
        self._set_video_info(mode=mode, drop_short=True, seed=seed)


    @property
    def dataset_dir(self):
        if not hasattr(self, "_dataset_dir"):
            self._dataset_dir = Path(self.data_path_dir, self.dataset_name)
            if not self._dataset_dir.is_dir():
                raise OSError(
                    f"Did not find dataset directory under {self._dataset_dir}."
                    )
        return self._dataset_dir


    @property
    def action_file_dir(self):
        return self.dataset_dir


    def _set_action_dicts(self):
        """
        self._set_action_dicts()
        """
        
        self.action_dict_encode = dict()
        self.action_dict_decode = dict()

        action_file = Path(self.action_file_dir, "classInd.txt")
        action_df = pd.read_csv(str(action_file), sep=" ", header=None)
        for _, row in action_df.iterrows():
            act_id, act_name = row
            act_id = int(act_id)
            self.action_dict_decode[act_id] = act_name
            self.action_dict_encode[act_name] = act_id


    def _drop_short_videos(self, video_info):
        """
        self._drop_short_videos(video_info)
        """
        
        drop_idx = []
        logger.info(
            "Filtering out videos that are too short", extra={"spacing": TAB}
            )
        for idx, row in tqdm(video_info.iterrows(), total=len(video_info)):
            _, vlen = row
            if vlen - self.num_seq * self.seq_len * self.downsample <= 0:
                drop_idx.append(idx) 
        video_info = video_info.drop(drop_idx, axis=0)

        return video_info


    def _set_video_info(self, mode="train", drop_short=True, seed=None):
        """
        self._set_video_info()
        """
        
        split_str = "" if self.split_n is None else f"{self.split_n:02}"
        if mode == "train":
            split = Path(self.dataset_dir, f"train_split{split_str}.csv")
        elif mode == "val":
            split = Path(self.dataset_dir, f"val_split{split_str}.csv")
            if not split.is_file():
                warnings.warn(
                    f"No validation split found under {split}, so using "
                    "the test split instead.")
                split = Path(self.dataset_dir, f"test_split{split_str}.csv")              
        elif mode == "test":
            split = Path(self.dataset_dir, f"test_split{split_str}.csv")
        else: 
            raise ValueError("Mode must be 'train', 'val' or 'test'.")
        
        if not split.is_file():
            raise OSError(f"{split} not found.")
        self.video_info = pd.read_csv(str(split), header=None)
        if drop_short:
            self.video_info = self._drop_short_videos(self.video_info)
        
        if self.unit_test: 
            n_sample = 32
            self.video_info = self.video_info.sample(
                n_sample, random_state=seed
                )


    def _crop_clips(self, t_seq):
        """
        self._crop_clips(t_seq)
        """
        
        (C, H, W) = t_seq[0].size()
        t_seq = torch.stack(t_seq, 0)

        # get all possible consecutive clips of length seq_len
        SL = t_seq.size(0)
        clips = []
        i = 0
        while i + self.seq_len <= SL:
            clips.append(t_seq[i : i + self.seq_len, :])
            i = i + self.seq_len

        num_poss_clips = len(clips) + 1 - self.num_seq
        step_size = self.num_seq // 2

        # half overlap
        clips = [
            torch.stack(clips[i : i + self.num_seq], 0).transpose(1, 2) 
            for i in range(0, num_poss_clips, step_size)
            ]
        t_seq = torch.stack(clips, 0)   

        return t_seq


    def idx_sampler(self, vlen, vpath, sample_num_seq=True, raise_none=False):
        """
        self.idx_sampler(vlen, vpath)
        
        sample sequences of indices from a video
        if sample_num_seq: num_seq x seq_len
        else: all consecutive sequences of length seq_len
        """

        last_poss_start = vlen - self.num_seq * self.seq_len * self.downsample
        if last_poss_start <= 0: 
            if raise_none:
                raise RuntimeError(
                    f"No indices identified with vlen {vlen} and vpath {vpath}."
                    )
            return None

        if sample_num_seq:
            n = 1
            # select a start index
            start_idx = np.random.choice(range(last_poss_start), n)

            # identify the start point for each sequence (consecutive)
            seq_start_idx = \
                start_idx + self.downsample * self.seq_len * np.expand_dims(
                    np.arange(self.num_seq), -1
                ) 
            # identify indices for each sequence (num_seq x seq_len)
            seq_idx = seq_start_idx + self.downsample * np.expand_dims(
                np.arange(self.seq_len), 0
                )
        else:
            # identify indices of a single, full length sequence (1D indices)
            all_idxs = np.arange(0, vlen, self.downsample) 
            n_frames = len(all_idxs)

            # identify all possible consecutive sequence indices (n x seq_len)
            seq_idx = []
            i = 0
            while i + self.seq_len <= n_frames:
                seq_idx.append(all_idxs[i : i + self.seq_len])
                i = i + self.seq_len
            seq_idx = np.asarray(seq_idx)

        return seq_idx, vpath

    
    def _load_transform_images(self, vpath, seq_idx):
        """
        self._load_transform_images(vpath, seq_idx)
        """

        seq = [
            pil_loader(Path(vpath, f"image_{i+1:05}.jpg")) for i in seq_idx
            ]
        if self.transform is not None:
            t_seq = self.transform(seq) # apply same transform
        else:
            t_seq = seq
            
        return t_seq


    def _select_seq_sub_batch(self, t_seq):
        """
        self._select_seq_sub_batch(t_seq)
        """
        
        num_poss = len(t_seq) - self.num_seq + 1
        step_size = self.num_seq // 2 # sub-batch of sequences overlap by half

        t_seq = torch.stack([
            t_seq[i : i + self.num_seq] for i in range(0, num_poss, step_size)
            ], 0)
        return t_seq


    def __getitem__(self, index):
        vpath, vlen = self.video_info.iloc[index]

        if (self.supervised and self.mode == "test"):
            seq_idx, vpath = self.idx_sampler(
                vlen, vpath, sample_num_seq=False, raise_none=True
                )
            seq_idx = seq_idx.reshape(len(seq_idx) * self.seq_len)
            t_seq = self._load_transform_images(vpath, seq_idx)

            (C, H, W) = t_seq[0].size()
            t_seq = torch.stack(t_seq, 0) # stack images

            # reshape/transpose to N_all x C x SL x H x W
            t_seq = t_seq.reshape(-1, self.seq_len, C, H, W).transpose(1, 2)
            
            # get a sub-batch: SUB_B x N x C x SL x H x W
            t_seq = self._select_seq_sub_batch(t_seq)
        else:
            seq_idx, vpath = self.idx_sampler(
                vlen, vpath, sample_num_seq=True, raise_none=True
                )
            # check shape, then concatenate sequences 
            exp_shape = (self.num_seq, self.seq_len)
            if seq_idx.shape != exp_shape:
                raise RuntimeError(
                    f"seq_idx shape should be {exp_shape}, but is "
                    f"{seq_idx.shape}."
                    )
            seq_idx = seq_idx.reshape(self.num_seq * self.seq_len)

            # get images
            t_seq = self._load_transform_images(vpath, seq_idx)
            (C, H, W) = t_seq[0].size()
            t_seq = torch.stack(t_seq, 0) # stack images
            
            # reshape/transpose to N x C x SL x H x W
            t_seq = t_seq.reshape(
                self.num_seq, self.seq_len, C, H, W
                ).transpose(1, 2)
        
        if self.return_label:
            vname = Path(vpath).parts[-2]
            action = self.encode_action(vname)
            label = torch.LongTensor([action])
            return t_seq, label
        else:
            return t_seq


    def __len__(self):
        return len(self.video_info)


    def encode_action(self, action_name):
        """
        self.encode_action(action_name)

        give action name, return category
        """
        
        return self.action_dict_encode[action_name]


    def decode_action(self, action_code):
        """
        self.decode_action(action_code)

        give action code, return action name
        """
        
        return self.action_dict_decode[action_code]


#############################################
class Kinetics400_3d(GeneralDataset):
    def __init__(self,
                 data_path_dir=Path("process_data", "data"),
                 mode="train",
                 transform=None,
                 seq_len=10,
                 num_seq=5,
                 downsample=3,
                 epsilon=5,
                 unit_test=False,
                 big=False,
                 return_label=False,
                 supervised=False,
                 seed=None
                ):

        if big: 
            size_str = "_256"
            logger.info("Using Kinetics400 data (256x256)")
        else: 
            size_str = ""
            logger.info("Using Kinetics400 data (150x150)")

        super().__init__(
            data_path_dir,
            mode=mode, 
            dataset_name=f"Kinetics400{size_str}",
            transform=transform,
            seq_len=seq_len,
            num_seq=num_seq,
            downsample=downsample,
            epsilon=epsilon,
            unit_test=unit_test,
            split_n=None,
            return_label=return_label,
            supervised=supervised,
            seed=seed,
            )


    @property
    def action_file_dir(self):
        if not hasattr(self, "_action_file_dir"):
            self._action_file_dir = Path(self.data_path_dir, "Kinetics400")
            if not self._action_file_dir.is_dir():
                raise OSError(
                    "Did not find action file directory under "
                    f"{self._action_file_dir}."
                    )
        return self._action_file_dir


#############################################
class UCF101_3d(GeneralDataset):
    def __init__(self,
                 data_path_dir=Path("process_data", "data"),
                 mode="train",
                 transform=None, 
                 seq_len=10,
                 num_seq=5,
                 downsample=3,
                 epsilon=5,
                 unit_test=False,
                 split_n=1,
                 return_label=False,
                 supervised=False,
                 seed=None
                 ):

        super().__init__(
            data_path_dir,
            mode=mode, 
            dataset_name="UCF101",
            transform=transform,
            seq_len=seq_len,
            num_seq=num_seq,
            downsample=downsample,
            epsilon=epsilon,
            unit_test=unit_test,
            split_n=split_n,
            return_label=return_label,
            supervised=supervised,
            seed=seed,
            )


#############################################
class HMDB51_3d(GeneralDataset):
    def __init__(self,
                 data_path_dir=Path("process_data", "data"),
                 mode="train",
                 transform=None,
                 seq_len=10,
                 num_seq=5,
                 downsample=1,
                 epsilon=5,
                 unit_test=False,
                 split_n=1,
                 return_label=False,
                 supervised=False,
                 seed=None
                 ):

        super().__init__(
            data_path_dir,
            mode=mode, 
            dataset_name="HMDB51",
            transform=transform,
            seq_len=seq_len,
            num_seq=num_seq,
            downsample=downsample,
            epsilon=epsilon,
            unit_test=unit_test,
            split_n=split_n,
            return_label=return_label,
            supervised=supervised,
            seed=seed,
            )

