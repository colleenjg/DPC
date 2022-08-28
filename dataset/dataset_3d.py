import logging
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils import data
import torchvision
from tqdm import tqdm

from dataset import augmentations

logger = logging.getLogger(__name__)

MAX_LEN = 500 # longest allowable test sequence

TAB = "    "


#############################################
def pil_loader(path_name):
    """
    pil_loader(path_name)

    Loads an image from file and converts to an RGB PIL Image.

    Required args
    -------------
    - path_name : str or path
        Path to image.

    Returns
    -------
    - RGB PIL Image
        RGB PIL image.
    """

    with open(path_name, "rb") as f:
        with Image.open(f) as img:
            return img.convert("RGB")


#############################################
class GeneralDataset(data.Dataset):
    """
    General dataset object that enables Dense CPC training.

    Attributes
    ----------
    - class_dict_decode : dict
        Dictionary for decoding class names from class labels.
    - class_dict_encode : dict
        Dictionary for encoding class names into class labels.
    - class_file_dir : path
        Directory under which file listing class labels is stored.
        ("classInd.txt")
    - data_path_dir : path
            Main directory in which the specific directory for the dataset 
            (dataset_dir) is located. 
    - dataset_dir : path
        Directory under which the split files are stored 
        (e.g., "train_split.csv").
    - dataset_name : str
        Dataset name used in dataset directory.
    - downsample : int
        Temporal downsampling to use.
    - mode : str
        Dataset mode (i.e., "train", "val", or "test") 
    - num_seq : int
        Number of consecutive sequences to return per sample
    - return_label : bool
        If True, when sequences are sampled, the associated label is returned 
        as well.
    - seq_len : int
        Number of frames per sequence.
    - split_n : int
        Dataset train/val(/test) split to use.
    - supervised : bool
        If True, dataset is set to supervised mode.
    - temp_data_dir : str or Path
        If provided, path to the new data directory to reset video paths to 
        point to (up to dataset name directory), after loading.
    - transform : torch Transform
        Transform to apply to the sequences sampled.
    - unit_test : bool
        If True, only a subsample of the full dataset is included in 
        self.video_info.
    - video_info : pd Dataframe
        Dataframe with 2 columns containing, for each video included in the 
        dataset given the mode and split: 
        - 0: The full path to the frame directory
        - 1: The total number of frames

    Methods
    -------
    - self.decode_class(class_label):
        Returns the class name for a given class label.
    - self.encode_class(class_name):
        Returns the class label for a given class name.
    - self.idx_sampler(vlen, vpath):
        Returns sequence indices for a specific video
    """
    
    def __init__(self,
                 data_path_dir,
                 mode="train",
                 dataset_name="UCF101",
                 transform=None,
                 seq_len=10,
                 num_seq=5,
                 downsample=3,
                 unit_test=False,
                 split_n=1,
                 return_label=False,
                 supervised=False,
                 seed=None,
                 temp_data_dir=None,
                 ):
        """
        GeneralDataset(data_path_dir)

        Constructs a dataset object.

        Required args
        -------------
        - data_path_dir : path
            Main directory in which the specific directory for the dataset 
            (containing pointers to the video frames for each split) is 
            located. 
        
        Optional args
        -------------
        - mode : str (default="train")
            Dataset mode (i.e., "train", "val", or "test").
        - dataset_name : str (default="UCF101")
            Dataset name used in dataset directory.
        - transform : torch Transform (default=None)
            Transform to apply to the sequences sampled.
        - seq_len : int (default=10)
            Number of frames per sequence.
        - num_seq : int (default=5)
            Number of consecutive sequences to return per sample.
        - downsample : int (default=3)
            Temporal downsampling to use.
        - unit_test : bool (default=False)
            If True, only a subsample of the full dataset is included in 
            self.video_info.
        - split_n : int (default=1)
            Dataset train/val(/test) split to use.
        - return_label : bool (default=False)
            If True, when sequences are sampled, the associated label is 
            returned as well.
        - supervised : bool (default=False)
            If True, dataset is set to supervised mode.
        - seed : int (default=None)
            Seed to use for the random process of sub-sampling the dataset, if 
            unit_test is True.
        - temp_data_dir : str or Path (default=None)
            If provided, path to the new data directory to reset video paths to 
            point to (up to dataset name directory), after loading.
        """

        self.data_path_dir = data_path_dir
        self.mode = mode
        self.dataset_name = dataset_name
        self.supervised = supervised
        self.seq_len = seq_len
        self.num_seq = num_seq
        self.downsample = downsample
        self.unit_test = unit_test
        self.split_n = split_n
        self.temp_data_dir = temp_data_dir

        if self.supervised:
            return_label = True

        self.return_label = return_label
        self._set_transform(transform)
        self._set_class_dicts()
        self._set_video_info(mode=mode, drop_short=True, seed=seed)


    @property
    def dataset_dir(self):
        """
        self.dataset_dir

        - path
            Directory under which the split files are stored 
            (e.g., "train_split.csv").
        """
        
        if not hasattr(self, "_dataset_dir"):
            self._dataset_dir = Path(self.data_path_dir, self.dataset_name)
            if not self._dataset_dir.is_dir():
                raise OSError(
                    f"Did not find dataset directory under {self._dataset_dir}."
                    )
        return self._dataset_dir


    @property
    def class_file_dir(self):
        """
        self.class_file_dir

        Directory under which file listing class labels is stored.
        ("classInd.txt")
        """
        
        return self.dataset_dir


    def _set_transform(self, transform=None):
        """
        self._set_transform()

        Sets the Tensor to use on the sequences after loading.

        Optional args
        -------------
        - transform : torch Transform (default=None)
            Transform to use. If None, a minimal transform is set to convert 
            images to tensor and normalize them.
        """

        if transform is None:
            self.transform = torchvision.transforms.Compose([
                augmentations.ToTensor(),
                augmentations.Normalize(),
            ])
        else:
            self.transform = transform


    def _set_class_dicts(self):
        """
        self._set_class_dicts()

        Sets attributes for encoding and decoding class names and labels.
        """
        
        self.class_dict_encode = dict()
        self.class_dict_decode = dict()

        class_file = Path(self.class_file_dir, "classInd.txt")
        class_df = pd.read_csv(str(class_file), sep=" ", header=None)
        for _, row in class_df.iterrows():
            act_id, act_name = row
            act_id = int(act_id)
            self.class_dict_decode[act_id] = act_name
            self.class_dict_encode[act_name] = act_id


    def _drop_short_videos(self, video_info):
        """
        self._drop_short_videos(video_info)

        Removes videos from the video info dataframe if they are too short to 
        allow a full sample to be generated.

        Required args
        -------------
        - video_info : pd Dataframe
            Dataframe with 2 columns containing, for each video included in the 
            dataset given the mode and split: 
            - 0: The full path to the frame directory
            - 1: The total number of frames
        """
        
        drop_idx = []
        logger.info(
            "Filtering out videos that are too short.", extra={"spacing": TAB}
            )
        for idx, row in tqdm(video_info.iterrows(), total=len(video_info)):
            _, vlen = row
            if vlen - self.num_seq * self.seq_len * self.downsample <= 0:
                drop_idx.append(idx) 
        video_info = video_info.drop(drop_idx, axis=0)

        return video_info


    def _update_video_paths(self, new_data_dir=None):
        """
        self._update_video_paths()

        Updates video paths in self.video_info to point to the new data 
        directory provided.

        Specifically, the new data directory replace the original path up to 
        the dataset name directory with no suffixes 
        (e.g., "Kinetics400" or "MouseSim"). If such a directory is not found 
        in the original paths, the replacement will fail. 

        Optional args
        -------------
        - new_data_dir : str or Path (default=None)
            If provided, path to the new data directory to reset video paths to 
            point to (up to dataset name directory). 
        """
        
        if new_data_dir is None:
            return

        if not Path(new_data_dir).is_dir():
            raise OSError(f"{new_data_dir} does not exist.")
        video_paths = [Path(video_path) for video_path in self.video_info[0]]
        orig_parts = video_paths[0].parts

        dataset_name = self.dataset_name.split("_")[0]
        if dataset_name in orig_parts:
            idx = orig_parts.index(dataset_name)
            first_path = Path(new_data_dir, *orig_parts[idx:])
            if video_paths[0].is_dir() and not first_path.is_dir():
                raise RuntimeError(
                    "Updating data paths with 'new_data_dir' failed: "
                    f"{first_path} does not exist."
                    )
            updated_video_paths = [
                str(Path(new_data_dir, *video_path.parts[idx:])) 
                for video_path in video_paths
                ]
            self.video_info[0] = updated_video_paths

        else:
            raise RuntimeError(
                "Could not align 'new_data_dir with existing video "
                "paths, as they do not contain the expected dataset name "
                f"directory {dataset_name}."
                )

    
    def _set_video_info(self, mode="train", drop_short=True, seed=None):
        """
        self._set_video_info()

        Sets the video info attribute.

        Optional args
        -------------
        - mode : str (default="train")
            Dataset mode (i.e., "train", "val", or "test").
        - drop_short : bool
            If True, videos that are too short to allow a full sample to be 
            generated are dropped.
        - seed : int (default=None)
            Seed to use for the random process of sub-sampling the dataset, if 
            self.unit_test is True.
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
        
        self._update_video_paths(self.temp_data_dir)
        

    def _load_transform_images(self, vpath, seq_idx):
        """
        self._load_transform_images(vpath, seq_idx)

        Loads images and applies transform.

        Required args
        -------------
        - vpath : path or str
            Path to the frames of a video, each labelled in the following 
            format: "image_00001.jpg"
        - seq_idx : array-like
            Indices of sequence frames to load, which are converted to the 
            frame numbers by adding 1, e.g., 0 indexes image 00001.

        Returns
        -------
        - t_seq: list of 3D Tensors
            List of transformed Tensor images, each with dims: C x H x W
        """

        seq = [
            pil_loader(Path(vpath, f"image_{i+1:05}.jpg")) for i in seq_idx
            ]

        t_seq = self.transform(seq) # apply same transform

        return t_seq


    def _select_seq_sub_batch(self, t_seq):
        """
        self._select_seq_sub_batch(t_seq)

        Returns sequence, split into sub-batches.

        Required args
        -------------
        - t_seq: list of 4D Tensors
            List of transformed Tensor sequences, each with dims: 
                L x C x H x W

        Returns
        -------
        - t_seq : 6D Tensor
            Tensor of all possible half-overlapping clips, with dims: 
                number of clips x N x L x C x H x W
        """
        
        num_poss = len(t_seq) - self.num_seq + 1
        step_size = self.num_seq // 2 # sub-batch of sequences overlap by half

        t_seq = torch.stack([
            t_seq[i : i + self.num_seq] for i in range(0, num_poss, step_size)
            ], 0)
        
        return t_seq
        
        
    def idx_sampler(self, vlen, vpath, sample_num_seq=True, raise_none=False):
        """
        self.idx_sampler(vlen, vpath)
        
        Returns indices to sample sequences from a video.

        Required args
        -------------
        - vlen : int
            Total number of frames in the video to sample from.
        - vpath : path or str
            Path to the frames of a video, each labelled in the following 
            format: "image_00001.jpg"            

        Optional args
        -------------
        - sample_num_seq : bool (default=True)
            If True, self.num_seq consecutive are sampled. Otherwise, all 
            possible sequences are retained. In both cases, sequences are 
            non-overlapping.
        - raise_none : bool (default=False)
            If True, an error is raised if the video is too short to sample 
            self.num_seq sequences from. Otherwise, if the video is too short, 
            None is returned.

        Returns
        -------
        - seq_idx : 2D array
            Indices of sequence frames to load, with dims: sequences x L
        """

        last_poss_start = vlen - self.num_seq * self.seq_len * self.downsample
        if last_poss_start <= 0: 
            if raise_none:
                raise RuntimeError(
                    f"No indices identified for vpath {vpath}, as it is too "
                    f"short to sample from ({vlen})."
                    )
            return None

        if sample_num_seq:
            n = 1
            # select a start index
            start_idx = np.random.choice(last_poss_start, n)

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

            if n_frames > MAX_LEN:
                logger.warning(
                    f"Sequence includes {n_frames} frames. Only the first "
                    f"{MAX_LEN} will be retained."
                    )
                n_frames = MAX_LEN

            # identify all possible consecutive sequence indices (n x seq_len)
            seq_idx = []
            i = 0
            while i + self.seq_len <= n_frames:
                seq_idx.append(all_idxs[i : i + self.seq_len])
                i = i + self.seq_len
            seq_idx = np.asarray(seq_idx)

        return seq_idx


    def encode_class(self, class_name):
        """
        self.encode_class(class_name)

        Returns the class label for a given class name.

        Required args
        -------------
        - class_name : str
            Class name for which to return the label.

        Returns
        -------
        - int
            Label corresponding to the class name provided.
        """
        
        return self.class_dict_encode[class_name]


    def decode_class(self, class_label):
        """
        self.decode_class(class_label)

        Returns the class name for a given class label.

        Required args
        -------------
        - class_label : int
            Class label for which to return the name.

        Returns
        -------
        - str
            Name corresponding to the class label provided.
        """
        
        return self.class_dict_decode[class_label]


    def sample_sequences(self, ix):
        """
        self.sample_sequences(ix)
    
        Returns sampled sequences and their shared class label. 
        
        Required args
        -------------
        - ix : int
            Index of video to sample

        Returns
        -------
        - t_seq : 5D Tensor
            Video sequences, 
            with dims: 
                number of seq x color channels (3) x seq len x height x width
        
        - label : int
            Shared class label for the video sequences.
        """

        vpath, vlen = self.video_info.iloc[ix]

        if (self.supervised and self.mode == "test"):
            seq_idx = self.idx_sampler(
                vlen, vpath, sample_num_seq=False, raise_none=True
                )
            seq_idx = seq_idx.reshape(len(seq_idx) * self.seq_len)
            t_seq = self._load_transform_images(vpath, seq_idx)

            (C, H, W) = t_seq[0].size()
            t_seq = torch.stack(t_seq, 0) # stack images

            # reshape/transpose to N_all x C x L x H x W
            t_seq = t_seq.reshape(-1, self.seq_len, C, H, W).transpose(1, 2)
            
            # get a sub-batch: SUB_B x N x C x L x H x W
            t_seq = self._select_seq_sub_batch(t_seq)

        else:
            seq_idx = self.idx_sampler(
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
            
            # reshape/transpose to N x C x L x H x W
            t_seq = t_seq.reshape(
                self.num_seq, self.seq_len, C, H, W
                ).transpose(1, 2)
        
        vname = Path(vpath).parts[-2]
        label = self.encode_class(vname)
        
        return t_seq, label


    def __getitem__(self, ix):
        """
        Returns sampled sequences, and, if self.return_label is True, their 
        shared class label. 
        
        Required args
        -------------
        - ix : int
            Index of video to sample

        Returns
        -------
        - t_seq : 5D Tensor
            Video sequences, 
            with dims: 
                number of seq x color channels (3) x seq len x height x width
        
        if self.return_label:
        - label : 1D Tensor
            Shared class label for the video sequences.
        """

        t_seq, label = self.sample_sequences(ix)

        if self.return_label:
            label = torch.LongTensor([label])
            return t_seq, label
        else:
            return t_seq


    def __len__(self):
        """
        Returns the dataset length (number of videos to sample from).
        """
        
        return len(self.video_info)


#############################################
class Kinetics400_3d(GeneralDataset):
    """
    Kinetics400_3d dataset object that enables Dense CPC training.

    See GeneralDataset for inherited attributes and methods.

    Attributes
    ----------
    - class_file_dir : path
        Directory under which file listing class labels is stored.
        ("classInd.txt")
    """
    
    def __init__(self,
                 data_path_dir=Path("process_data", "data"),
                 mode="train",
                 transform=None,
                 seq_len=10,
                 num_seq=5,
                 downsample=3,
                 unit_test=False,
                 big=False,
                 return_label=False,
                 supervised=False,
                 seed=None,
                 temp_data_dir=None,
                ):
        """
        Kinetics400_3d()

        Constructs a Kinetics400_3d dataset.

        Optional args
        -------------
        - data_path_dir : path (default=Path("process_data", "data"))
            Main directory in which the Kinetics400 directory (containing 
            pointers to the video frames for each split) is located. 
        - mode : str (default="train")
            Dataset mode (i.e., "train", "val", or "test").
        - transform : torch Transform (default=None)
            Transform to apply to the sequences sampled.
        - seq_len : int (default=10)
            Number of frames per sequence.
        - num_seq : int (default=5)
            Number of consecutive sequences to return per sample.
        - downsample : int (default=3)
            Temporal downsampling to use.
        - unit_test : bool (default=False)
            If True, only a subsample of the full dataset is included in 
            self.video_info.
        - big : bool (default=False):
            If True, the bigger version of the Kinetics400 dataset is used. 
        - return_label : bool (default=False)
            If True, when sequences are sampled, the associated label is 
            returned as well.
        - supervised : bool (default=False)
            If True, dataset is set to supervised mode.
        - seed : int (default=None)
            Seed to use for the random process of sub-sampling the dataset, if 
            unit_test is True.
        - temp_data_dir : str or Path (default=None)
            If provided, path to the new data directory to reset video paths to 
            point to (up to dataset name directory), after loading.
        """

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
            unit_test=unit_test,
            split_n=None,
            return_label=return_label,
            supervised=supervised,
            seed=seed,
            temp_data_dir=temp_data_dir,            
            )


    @property
    def class_file_dir(self):
        """
        self.class_file_dir

        Directory under which file listing class labels is stored.
        ("classInd.txt")
        """

        if not hasattr(self, "_class_file_dir"):
            self._class_file_dir = Path(self.data_path_dir, "Kinetics400")
            if not self._class_file_dir.is_dir():
                raise OSError(
                    "Did not find class file directory under "
                    f"{self._class_file_dir}."
                    )
        return self._class_file_dir


#############################################
class UCF101_3d(GeneralDataset):
    """
    UCF101_3d dataset object that enables Dense CPC training.

    See GeneralDataset for inherited attributes and methods.
    """

    def __init__(self,
                 data_path_dir=Path("process_data", "data"),
                 mode="train",
                 transform=None, 
                 seq_len=10,
                 num_seq=5,
                 downsample=3,
                 unit_test=False,
                 split_n=1,
                 return_label=False,
                 supervised=False,
                 seed=None,
                 temp_data_dir=None,
                 ):
        """
        UCF101_3d()

        Constructs a UCF101_3d dataset.

        Optional args
        -------------
        - data_path_dir : path (default=Path("process_data", "data"))
            Main directory in which the Kinetics400 directory (containing 
            pointers to the video frames for each split) is located. 
        - mode : str (default="train")
            Dataset mode (i.e., "train", "val", or "test").
        - transform : torch Transform (default=None)
            Transform to apply to the sequences sampled.
        - seq_len : int (default=10)
            Number of frames per sequence.
        - num_seq : int (default=5)
            Number of consecutive sequences to return per sample.
        - downsample : int (default=3)
            Temporal downsampling to use.
        - unit_test : bool (default=False)
            If True, only a subsample of the full dataset is included in 
            self.video_info.
        - split_n : int
            Dataset train/val(/test) split to use.
        - return_label : bool (default=False)
            If True, when sequences are sampled, the associated label is 
            returned as well.
        - supervised : bool (default=False)
            If True, dataset is set to supervised mode.
        - seed : int (default=None)
            Seed to use for the random process of sub-sampling the dataset, if 
            unit_test is True.
        - temp_data_dir : str or Path (default=None)
            If provided, path to the new data directory to reset video paths to 
            point to (up to dataset name directory), after loading.
        """

        super().__init__(
            data_path_dir,
            mode=mode, 
            dataset_name="UCF101",
            transform=transform,
            seq_len=seq_len,
            num_seq=num_seq,
            downsample=downsample,
            unit_test=unit_test,
            split_n=split_n,
            return_label=return_label,
            supervised=supervised,
            seed=seed,
            temp_data_dir=temp_data_dir,
            )


#############################################
class HMDB51_3d(GeneralDataset):
    """
    HMDB51_3d dataset object that enables Dense CPC training.

    See GeneralDataset for inherited attributes and methods.
    """

    def __init__(self,
                 data_path_dir=Path("process_data", "data"),
                 mode="train",
                 transform=None,
                 seq_len=10,
                 num_seq=5,
                 downsample=1,
                 unit_test=False,
                 split_n=1,
                 return_label=False,
                 supervised=False,
                 seed=None,
                 temp_data_dir=None,
                 ):
        """
        HMDB51_3d()

        Constructs a HMDB51_3d dataset.

        Optional args
        -------------
        - data_path_dir : path (default=Path("process_data", "data"))
            Main directory in which the Kinetics400 directory (containing 
            pointers to the video frames for each split) is located. 
        - mode : str (default="train")
            Dataset mode (i.e., "train", "val", or "test").
        - transform : torch Transform (default=None)
            Transform to apply to the sequences sampled.
        - seq_len : int (default=10)
            Number of frames per sequence.
        - num_seq : int (default=5)
            Number of consecutive sequences to return per sample.
        - downsample : int (default=3)
            Temporal downsampling to use.
        - unit_test : bool (default=False)
            If True, only a subsample of the full dataset is included in 
            self.video_info.
        - split_n : int
            Dataset train/val(/test) split to use.
        - return_label : bool (default=False)
            If True, when sequences are sampled, the associated label is 
            returned as well.
        - supervised : bool (default=False)
            If True, dataset is set to supervised mode.
        - seed : int (default=None)
            Seed to use for the random process of sub-sampling the dataset, if 
            unit_test is True.
        - temp_data_dir : str or Path (default=None)
            If provided, path to the new data directory to reset video paths to 
            point to (up to dataset name directory), after loading.
        """

        super().__init__(
            data_path_dir,
            mode=mode, 
            dataset_name="HMDB51",
            transform=transform,
            seq_len=seq_len,
            num_seq=num_seq,
            downsample=downsample,
            unit_test=unit_test,
            split_n=split_n,
            return_label=return_label,
            supervised=supervised,
            seed=seed,
            temp_data_dir=temp_data_dir,
            )


#############################################
class MouseSim_3d(GeneralDataset):
    """
    MouseSim_3d dataset object that enables Dense CPC training.

    See GeneralDataset for inherited attributes and methods.

    Attributes
    ----------
    - class_file_dir : path
        Directory under which file listing class labels is stored.
        ("classInd.txt")
    """

    def __init__(self,
                 data_path_dir=Path("process_data", "data"),
                 mode="train",
                 eye="right",
                 transform=None,
                 seq_len=10,
                 num_seq=5,
                 downsample=3,
                 unit_test=False,
                 return_label=False,
                 supervised=False,
                 seed=None,
                 temp_data_dir=None,
                 ):
        """
        MouseSim_3d()

        Constructs a MouseSim_3d dataset.

        Optional args
        -------------
        - data_path_dir : path (default=Path("process_data", "data"))
            Main directory in which the Kinetics400 directory (containing 
            pointers to the video frames for each split) is located. 
        - mode : str (default="train")
            Dataset mode (i.e., "train", "val", or "test").
        - eye : str (default="right")
            Eye(s) to which data should be cropped 
            (i.e., "left", "right", "both").
        - transform : torch Transform (default=None)
            Transform to apply to the sequences sampled.
        - seq_len : int (default=10)
            Number of frames per sequence.
        - num_seq : int (default=5)
            Number of consecutive sequences to return per sample.
        - downsample : int (default=1)
            Temporal downsampling to use.
        - unit_test : bool (default=False)
            If True, only a subsample of the full dataset is included in 
            self.video_info.
        - split_n : int
            Dataset train/val(/test) split to use.
        - return_label : bool (default=False)
            If True, when sequences are sampled, the associated label is 
            returned as well.
        - supervised : bool (default=False)
            If True, dataset is set to supervised mode.
        - seed : int (default=None)
            Seed to use for the random process of sub-sampling the dataset, if 
            unit_test is True.
        - temp_data_dir : str or Path (default=None)
            If provided, path to the new data directory to reset video paths to 
            point to (up to dataset name directory), after loading.
        """

        if eye == "both": 
            logger.info("Using MouseSim (both eyes) data")
            eye_str = ""
        elif eye in ["left", "right"]: 
            logger.info(f"Using MouseSim ({eye} eye) data")
            eye_str = f"_{eye}"
        else:
            raise ValueError(f"{eye} value for 'eye' not recognized.")

        super().__init__(
            data_path_dir,
            mode=mode,
            dataset_name=f"MouseSim{eye_str}",
            transform=transform,
            seq_len=seq_len,
            num_seq=num_seq,
            downsample=downsample,
            unit_test=unit_test,
            split_n=None,
            return_label=return_label,
            supervised=supervised,
            seed=seed,
            temp_data_dir=temp_data_dir,
            )

    @property
    def class_file_dir(self):
        """
        self.class_file_dir

        Directory under which file listing class labels is stored.
        ("classInd.txt")
        """
        if not hasattr(self, "_class_file_dir"):
            self._class_file_dir = Path(self.data_path_dir, "MouseSim")
            if not self._class_file_dir.is_dir():
                raise OSError(
                    "Did not find class file directory under "
                    f"{self._class_file_dir}."
                    )
        return self._class_file_dir
        