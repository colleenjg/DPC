#!/usr/bin/env python

import argparse
import glob
import io
import logging
from pathlib import Path
import shutil
import subprocess
import sys
import urllib
import zipfile

import numpy as np
from torchvision.datasets import kinetics
from tqdm import tqdm

sys.path.extend([
    "..", 
    str(Path("..", "..")), 
    str(Path("..", "..", "utils")), 
    str(Path("..", "..", "dataset"))
    ])
from utils import misc_utils, training_utils
from dataset import dataset_3d

logger = logging.getLogger(__name__)


UCF101_URLS = {
    "videos": "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar", # 6 GB
    "splits": "https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip",
}

HMDB51_URLS = {
    "videos": "http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar", # 2 GB
    "splits": "http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/test_train_splits.rar",
}


#############################################
def check_direc_load_rarfile(d_root, dataset="UCF101"):
    """
    check_direc_load_rarfile(d_root)

    Checks whether the dataset folder already exists, and loads the rarfile 
    module, checking that its requirements are installed.

    Required args:
        - d_root (Path):
            dataset folder to create (should not already exist)
    
    Optional args:
        - dataset (str):
            current dataset
            default: "UCF101"
    
    Returns:
        - rarfile (module): module for extracting files from a RAR file.
    """

    if Path(d_root).exists():
        raise NotImplementedError(
            f"Cannot download {dataset} data if {d_root} already exists. "
            "Remove directory and run again."
            )

    logger.info(f"Downloading and organizing data for {dataset}... ")

    try:
        from unrar import rarfile
    except LookupError as err:
        raise LookupError(
            f"{err} UnRAR can be installed following the instructions from "
            "https://pypi.org/project/unrar/")

    Path(d_root).mkdir(parents=True)

    return rarfile


#############################################
def check_num_classes(v_root, num_classes=10):
    """
    check_num_classes(v_root)

    Checks whether the number of video folders matches the number of classes.

    Required args:
        - v_root (Path):
            video folder, containing folders for each class
    
    Optional args:
        - num_classes (int):
            expected number of classes
            default: 10
    """

    contents = list(Path(v_root).iterdir())
    folders = [item for item in contents if item.is_dir()]
    others = [item for item in contents if not item.is_dir()]

    if len(folders) != num_classes:
        raise ValueError(
            f"Expected {num_classes} video folders in {v_root}, "
            f"but found {len(folders)}."
            )

    if len(others):
        raise ValueError(f"Found items that aren't folders in {v_root}.")


#############################################
def main_UCF101(d_root):
    """
    main_UCF101(d_root)

    Downloads data for the UCF101 dataset.

    Required args:
        - d_root (Path):
            main folder in which to save dataset
    """

    rarfile = check_direc_load_rarfile(d_root, dataset="UCF101")

    # get train/test split file
    for url_key in ["splits", "videos"]:
        args = ["wget", "-P", str(d_root), "--no-check-certificate", UCF101_URLS[url_key]]
        if url_key == "videos":
            logger.warning(
                f"Downloading from {UCF101_URLS[url_key]} can be buggy and "
                "slow, and may go through several re-connection attempts."
                )
        process = subprocess.Popen(args, stdout=subprocess.PIPE)
        process.wait()

        file_path = Path(d_root, Path(UCF101_URLS[url_key]).name)
        extract_targ = str(Path(d_root, url_key))
        if url_key == "splits":
            with zipfile.ZipFile(str(file_path), "r") as z:
                z.extractall(extract_targ)
        else:
            logger.warning(f"Opening RAR compressed file...")
            with rarfile.RarFile(str(file_path), "r") as rar:
                rar.extractall(extract_targ)
        file_path.unlink()

    # remove intermediate ucfTrainTestlist folder
    targ_folder = Path(d_root, "splits")
    split_files = glob.glob(str(Path(targ_folder, "**", "*.*")), recursive=True)
    for split_file in split_files:
        Path(split_file).rename(Path(targ_folder, Path(split_file).name))
    rm_folder = Path(d_root, "splits", "ucfTrainTestlist")
    if Path(rm_folder).is_dir():
        shutil.rmtree(rm_folder)

    # remove intermediate UCF-101 folder
    targ_folder = Path(d_root, "videos")
    video_folders = glob.glob(str(Path(targ_folder, "*", "*")))
    for video_folder in video_folders:
        Path(video_folder).rename(Path(targ_folder, Path(video_folder).name))
    rm_folder = Path(d_root, "videos", "UCF-101")
    if Path(rm_folder).is_dir():
        shutil.rmtree(rm_folder)

    check_num_classes(Path(d_root, "videos"), num_classes=101)


#############################################
def main_HMDB51(d_root):
    """
    main_HMDB51(d_root)

    Downloads data for the HMDB51 dataset.

    Required args:
        - d_root (Path):
            main folder in which to save dataset
    """

    # run checks and retrieves rarfile module
    rarfile = check_direc_load_rarfile(d_root, dataset="HMDB51")

    # get train/test split file
    for url_key in ["splits", "videos"]:
        args = ["wget", "-P", str(d_root), HMDB51_URLS[url_key]]
        process = subprocess.Popen(args, stdout=subprocess.PIPE)
        process.wait()
        rarfile_path = Path(d_root, Path(HMDB51_URLS[url_key]).name)
        extract_targ = str(Path(d_root, url_key))
        with rarfile.RarFile(str(rarfile_path)) as rar:
            rar.extractall(extract_targ)
        rarfile_path.unlink()

    # remove intermediate testTrainMulti_7030_splits folder
    targ_folder = Path(d_root, "splits")
    split_files = glob.glob(str(Path(targ_folder, "**", "*.*")), recursive=True)
    for split_file in split_files:
        Path(split_file).rename(Path(targ_folder, Path(split_file).name))
    rm_folder = Path(d_root, "splits", "testTrainMulti_7030_splits")
    if Path(rm_folder).is_dir():
        shutil.rmtree(rm_folder)

    # untar individual video folders
    logger.info("Untarring folders for each action...")
    video_rars = Path(d_root, "videos", "*.rar")
    all_rars = glob.glob(str(video_rars))
    for rarfile_path in tqdm(all_rars):
        rar = rarfile.RarFile(rarfile_path)
        rar.extractall(str(Path(d_root, "videos")))
        Path(rarfile_path).unlink()

    check_num_classes(Path(d_root, "videos"), num_classes=51)


#############################################
class KineticsMinimal(kinetics.Kinetics):
    """
    Modified Kinetics dataset class, that allows minimal downloading of the 
    Kinetics dataset.

    Adapted from torchvision.datasets.kinetics.Kinetics
    """

    _TAR_URLS = {
        "400": "https://s3.amazonaws.com/kinetics/400/{split}/k400_{split}_path.txt",
        "600": "https://s3.amazonaws.com/kinetics/600/{split}/k600_{split}_path.csv",
        "700": "https://s3.amazonaws.com/kinetics/700_2020/{split}/k700_2020_{split}_path.txt",
    }

    def __init__(
        self,
        root: str,
        num_classes: str = "400",
        split: str = "train",
        extensions: kinetics.Tuple[str, ...] = ("avi", "mp4"),
        download: bool = False,
        num_download_workers: int = 1,
        minimal: bool = True,
    ) -> None:

        self.num_classes = kinetics.verify_str_arg(num_classes, arg="num_classes", valid_values=["400", "600", "700"])
        self.extensions = extensions
        self.num_download_workers = num_download_workers

        self.root = root
        self.split_folder = kinetics.path.join(root, split)
        self.split = kinetics.verify_str_arg(split, arg="split", valid_values=["train", "val", "test"]) # added test

        # === allow for downloading only a minimal part of the datasets === #
        self.minimal = minimal
        
        # set number of tars to download, depending on the dataset and split
        # (implemented for Kinetics400 only)
        if self.minimal:
            partial_download_err = True
            if int(self.num_classes) == 400:
                if self.split == "train":
                    self._n_files = 14 # ~1.5 GB each
                    self._min_ex_per = 5
                    partial_download_err = False
                elif self.split == "val":
                    self._n_files = 3 # ~1.5 GB each
                    self._min_ex_per = 2
                    partial_download_err = False
                elif self.split == "test":
                    self._n_files = 4 # ~1.5 GB each
                    self._min_ex_per = 2
                    partial_download_err = False
            if partial_download_err:
                raise NotImplementedError(
                    "The minimum number of tars to download has not been "
                    f"determined for Kinetics{self.num_classes} {self.split} "
                    "to ensure that all action classes are included, with at "
                    f"least {self._min_ex_per} examples per class."
                )

        if download:
            self.download_and_process_videos()
            self._cleanup_check()
        # ================================================================= #        

    def _download_videos(self) -> None:
        """download tarballs containing the video to "tars" folder and extract them into the _split_ folder where
        split is one of the official dataset splits.
        Raises:
            RuntimeError: if download folder exists, breaks to prevent downloading entire dataset again.
        """

        if kinetics.path.exists(self.split_folder):
            raise RuntimeError(
                f"The directory {self.split_folder} already exists. "
                f"If you want to re-download or re-extract the images, delete the directory."
            )
        tar_path = kinetics.path.join(self.root, "tars")
        file_list_path = kinetics.path.join(self.root, "files")

        split_url = self._TAR_URLS[self.num_classes].format(split=self.split)
        split_url_filepath = kinetics.path.join(file_list_path, kinetics.path.basename(split_url))
        if not kinetics.check_integrity(split_url_filepath):
            kinetics.download_url(split_url, file_list_path)
        with open(split_url_filepath) as file:
            list_video_urls = [urllib.parse.quote(line, safe="/,:") for line in file.read().splitlines()]

        # === allow for downloading only part of the datasets === #
        n_urls = len(list_video_urls)
        if self.minimal:
            self._n_files = min(self._n_files, n_urls)
            list_video_urls = list_video_urls[:self._n_files]
            logger.info(f"Downloading from {self._n_files}/{n_urls} links.")
        else:
            logger.info(f"Downloading from {n_urls} links.")
        # ======================================================= # 

        if self.num_download_workers == 1:
            for video_url in list_video_urls:
                kinetics.download_and_extract_archive(video_url, tar_path, self.split_folder)
        else:
            part = kinetics.partial(kinetics._dl_wrap, tar_path, self.split_folder)
            poolproc = kinetics.Pool(self.num_download_workers)
            poolproc.map(part, list_video_urls)

        # === some cleanup === # 
        self._make_ds_structure()
        self._cleanup_check()
        shutil.rmtree(tar_path)
        shutil.rmtree(file_list_path)
        # ==================== # 


    # === additional cleanup and checks === # 
    def _cleanup_check(self) -> None:
        """
        self._cleanup_check()
        """
        
        # remove files outside of directories
        ns = []
        for item in Path(self.split_folder).iterdir():
            if item.is_file():
                item.unlink() # unsorted files
            else:
                ns.append(len(list(item.iterdir())))

        # check for errors
        err_msg = None
        n_empty = sum([n == 0 for n in ns])
        if n_empty != 0:
            err_msg = f"{n_empty} action directories are empty"
        elif len(ns) > int(self.num_classes):
            err_msg = (
                f"More action directories ({len(ns)}) than classes "
                f"({self.num_classes})."                
            )
        elif np.min(ns) < self._min_ex_per:
            n_below = sum(np.asarray(ns) < self._min_ex_per)
            min_n = np.min(ns)
            err_msg = (f"{n_below} action directories contain fewer than "
                f"{self._min_ex_per} examples (as few as {min_n}).")
        
        if err_msg is not None:
            raise NotImplementedError(f"Implementation error: {err_msg}")
    # ===================================== # 


#############################################
def main_Kinetics400(d_root, minimal=False, parallel=True):
    """
    main_Kinetics400(d_root)

    Downloads data for the Kinetics400 dataset.

    Required args:
        - d_root (Path):
            main folder in which to save dataset
    
    Optional args:
        - minimal (bool): 
            if True, a minimal subset of the Kinetics400 dataset is downloaded, 
            covering all classes.
            default: False
        - parallel (bool):
            if True, data is downloaded in parallel, instead of sequentially
            default: True
    """

    logger.info("Downloading and organizing data for Kinetics400... ")

    n_jobs = 1
    if parallel:
        n_jobs = training_utils.get_num_jobs()

    Path(d_root).mkdir(exist_ok=True, parents=True)
    for split in ["train", "val", "test"]:
        KineticsMinimal(
            root=d_root, split=split, minimal=minimal, 
            num_download_workers=n_jobs, download=True,
            )

    v_root = Path(d_root, "videos")
    v_root.mkdir(parents=True, exist_ok=True)
    for item in Path(d_root).iterdir():
        if "annotations" not in str(item):
            shutil.move(str(item), str(v_root))

    annot_root = Path(d_root, "annotations")
    for item in Path(annot_root).iterdir():
        Path(item).rename(str(item).replace(".csv", "_split.csv"))
    
    for video_dir in ["train", "val", "test"]:      
        src_direc = Path(d_root, "videos", video_dir)
        targ_direc = Path(d_root, "videos", f"{video_dir}_split")
        src_direc.rename(targ_direc)
        check_num_classes(targ_direc, num_classes=400)


#############################################
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--d_root", type=Path,
        help=("root directory in which to save datasets "
            "{args.d_root}/{args.dataset}/videos/{split names}.")
        )
    parser.add_argument("--minimal", action="store_true",
        help=("if True, only a minimal amount of the data is downloaded "
              "for Kinetics400, e.g. for testing"))

    parser.add_argument("--dataset", default="UCF101", help="dataset name")
    parser.add_argument('--log_level', default='info', 
                        help='logging level, e.g., debug, info, error')
    parser.add_argument("--not_parallel", action="store_true", 
                        help=("download data sequentially, instead of in "
                        "parallel (Kinetics400)"))

    args = parser.parse_args()

    misc_utils.get_logger_with_basic_format(level=args.log_level)

    args.dataset = dataset_3d.normalize_dataset_name(args.dataset)

    d_root = Path(args.d_root, args.dataset)

    if args.dataset == "UCF101":
        main_UCF101(d_root=d_root)

    elif args.dataset == "HMDB51":
        main_HMDB51(d_root=d_root)

    elif args.dataset == "Kinetics400":
        main_Kinetics400(
            d_root=d_root, minimal=args.minimal, parallel=not(args.not_parallel)
            )
    
    elif args.dataset == "MouseSim":
        raise ValueError(
            "Downloading data is not implemented for the MouseSim dataset."
            )

    elif args.dataset == "Gabors":
        raise ValueError(
            "Gabors are generated on the fly, and do not require downloading."
            )    

    else:
        raise ValueError(f"{args.dataset} dataset not recognized.")

