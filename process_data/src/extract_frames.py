#!/usr/bin/env python

import argparse
import logging
from pathlib import Path
import glob 
import sys
import warnings

import cv2
from joblib import delayed, Parallel
import numpy as np
from tqdm import tqdm 
from wurlitzer import pipes # catch C-level pipes in python

sys.path.extend([
    "..", 
    str(Path("..", "..")), 
    str(Path("..", "..", "utils")), 
    ])
from utils import misc_utils

logger = logging.getLogger(__name__)

TAB = "    "


#############################################
def get_split_width(width=224, half_wid="left"): 
    """
    get_split_width()
    
    Returns new width and slice for splitting image in half width-wise.

    Optional args
    -------------
    - width : int
        Original width
        (default=224
    - half_wid : str or bool (default="left")
        If not False, "left" or "right", depending on which side of the 
        image should be retained
    
    Returns
    -------
    - width : int
        Resliced width
    - wid_slice (tuple):
        Slice for slicing image width
    """

    wid_slice = slice(None)
    if half_wid:
        if half_wid == "left":
            split_pt = int(np.ceil(width / 2))
            wid_slice = slice(0, split_pt)
            width = split_pt
        elif half_wid == "right":
            split_pt = int(np.floor(width / 2))
            wid_slice = slice(split_pt, int(width))
            width = int(width - split_pt)
        else:
            raise ValueError(f"{half_wid} value for 'half_wid' not recognized.")

    return width, wid_slice


#############################################
def resize_dim(width, height, target):
    """
    resize_dim(width, height)
    
    Resizes, such that the smaller side is target, to keep the aspect ratio.

    Required args
    -------------
    - width : int
        Width
    - height : int
        Height
    - target : int
        Target length of the short side
    
    Returns
    -------
    - tuple
        Target width and height (w, h)
    """

    if width == 0 or height == 0:
        raise ValueError("width and height must be non-zero.")

    if width >= height:
        return (int(target * width / height), int(target))
    else:
        return (int(target), int(target * height / width)) 


#############################################
def extract_video_opencv(v_path, f_root, dim=None, half_wid=False):
    """
    extract_video_opencv(v_path, f_root)

    Extracts frames from a dataset video.

    Required args
    -------------
    - v_path : path
        Path to the video from which to extract frames
    - f_root : path 
        Main folder in which to save frames for the dataset
    
    Optional args
    -------------
    - dim : int (default=None)
        Dimension to use in resizing images (shortest side). If None, 
        a default value is used (see extract_video_opencv()).
    - half_wid : bool (default=False)
        If True, only one half of the frame width is retained 
        ("left" or "right")

    Returns
    -------
    - success_code : float
        Success code, where a value of 
        1.0 indicates a fully successful extraction, 
        0.5 indicates a partial extraction and 
        0.0 indicates a failed extraction
    """
    
    if dim is None:
        dim = 240

    v_path = Path(v_path)
    v_class = v_path.parent.name
    out_dir = Path(f_root, v_class, v_path.stem)
    out_dir.mkdir(exist_ok=True, parents=True)

    # ignore a wurlitzer warning
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", "Failed to set pipe buffer size", RuntimeWarning
            )
        with pipes() as (out, err):
            vidcap = cv2.VideoCapture(str(v_path))
    
    c_err = str(err.read())
    readable = vidcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if len(c_err) or not readable:
        base_str = f"Video {v_path} could not be read"
        if len(c_err):
            base_str = (
                f"{base_str} due to the following ffmpeg error: {c_err}"
            )
        else:
            base_str = f"{base_str}. "

        if len(c_err) and "error: -11" in str(c_err):
            extra_str = (
                "This error may be due to excessive process spawning. "
                "Try setting or lower the value of the environment variable "
                "OMP_NUM_THREADS before running, in order to limit the number "
                "of parallel processes being spawned, or running sequentially."
                )
            raise RuntimeError(f"{base_str}{extra_str}")
        else:
            logger.error(f"{base_str}Skipping...\n")

        success_code = 0
        return success_code

    vidcap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    nb_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
    height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float

    width, wid_slice = get_split_width(width, half_wid)

    new_dim = resize_dim(width, height, dim)

    count = 0
    success, image = vidcap.read()
    while success:
        count += 1
        image = cv2.resize(
            image[:, wid_slice], new_dim, interpolation=cv2.INTER_LINEAR
            )
        # quality from 0-100, 95 is default, high is good
        cv2.imwrite(
            str(Path(out_dir, f"image_{count:05}.jpg")), 
            image,
            [cv2.IMWRITE_JPEG_QUALITY, 80]
            )
        success, image = vidcap.read()
    
    if nb_frames - count > 1: # 1 missing frame is common
        logger.error(
            f"{Path(*Path(out_dir).parts[-2:])} only PARTIALLY successfully "
            f"extracted ({int(count/nb_frames)}%): {count}/{nb_frames} frames."
            )
        success_code = 0.5
    else:
        success_code = 1
    
    vidcap.release()

    return success_code


#############################################
def extract_videos_opencv(v_root, f_root, dim=None, video_ext="avi", 
                          half_wid=False, parallel=True):
    """
    extract_videos_opencv(v_root, f_root)

    Extracts frames from a dataset's videos, for one class at a time.

    Required args
    -------------
    - v_root : path
        Main folder containing videos for the dataset
    - f_root : path 
        Main folder in which to save frames for the dataset
    
    Optional args
    -------------
    - dim : int (default=None)
        Dimension to use in resizing images (shortest side). If None, 
        a default value is used (see extract_video_opencv()).
    - video_ext : str (default="avi")
        Video extension to use to identify videos for frame extraction
    - half_wid : bool (default=False)
        If True, only one half of the frame width is retained 
        ("left" or "right")
    - parallel : bool (default=True)
        If True, frames for different videos for a class are extracted in 
        parallel
    
    Returns
    -------
    - all_success_code : list
        Nested list of success codes for videos from each class 
        directory (class x video), where values of 
        1.0 indicate a fully successful extraction, 
        0.5 indicate a partial extraction and 
        0.0 indicate a failed extraction
    """

    if not Path(v_root).is_dir():
        raise ValueError(f"{v_root} is not a directory.")

    Path(f_root).mkdir(exist_ok=True, parents=True)

    v_act_root = glob.glob(str(Path(v_root, "*")))
    v_act_root = sorted([j for j in v_act_root if Path(j).is_dir()])
    n_direc = len(v_act_root)

    logger.info(
        f"\nExtracting frames from videos in {n_direc} directories in\n"
        f"{v_root}\nand saving to\n{f_root}."
        )

    all_success_codes = []
    for i, j in tqdm(enumerate(v_act_root), total=len(v_act_root)):
        v_paths = glob.glob(str(Path(j, f"*.{video_ext}")))
        v_paths = sorted(v_paths)

        v_class = Path(j).name # class name
        out_dir = Path(f_root, v_class)
        # if resuming, be sure to delete the last video folder, 
        # if the extraction was incomplete
        if out_dir.is_dir(): 
            logger.info(
                f"{out_dir} already exists. Skipping {v_class} class."
                )
            continue

        logger.info(f"Extracting frames from {len(v_paths)} {v_class} videos.")
        if parallel:
            n_jobs = misc_utils.get_num_jobs(len(v_paths))
            success_codes = Parallel(n_jobs=n_jobs)(
                delayed(extract_video_opencv)(
                    p, f_root, dim=dim, half_wid=half_wid
                    ) 
                for p in tqdm(v_paths, total=len(v_paths))
                )
        else:
            success_codes = []
            for p in tqdm(v_paths, total=len(v_paths)):
                success_codes.append(
                    extract_video_opencv(p, f_root, dim=dim, half_wid=half_wid)
                    )
        all_success_codes.append(success_codes)
    
    return all_success_codes
            

#############################################
def main_UCF101(v_root, f_root=None, parallel=True):
    """
    main_UCF101(v_root)

    Extracts frames from UCF101 dataset videos.

    Required args
    -------------
    - v_root : path
        Main folder containing videos for the dataset
    
    Optional args
    -------------
    - f_root : path (default=None)
        Main folder in which to save frames for the dataset. If None, 
        a default location is identified, based on v_root.
    - parallel : bool (default=True)
        If True, frames for different videos for a class are extracted in 
        parallel
    """
    
    logger.info("Extracting frames for UCF101... ")

    if f_root is None:
        f_root = Path(Path(v_root).parent, "frames")

    extract_videos_opencv(v_root, f_root, video_ext="avi", parallel=parallel)


#############################################
def main_HMDB51(v_root, f_root=None, parallel=True):
    """
    main_HMDB51(v_root)

    Extracts frames from HMDB51 dataset videos.

    Required args
    -------------
    - v_root : path
        Main folder containing videos for the dataset
    
    Optional args
    -------------
    - f_root : path (default=None)
        Main folder in which to save frames for the dataset. If None, 
        a default location is identified, based on v_root.
    - parallel : bool (default=True)
        If True, frames for different videos for a class are extracted in 
        parallel
    """
    
    logger.info("Extracting frames for HMDB51... ")

    if f_root is None:
        f_root = Path(Path(v_root).parent, "frames")

    extract_videos_opencv(v_root, f_root, video_ext="avi", parallel=parallel)


#############################################
def main_Kinetics400(v_root, f_root=None, dim=None, parallel=True):
    """
    main_Kinetics400(v_root)

    Extracts frames from Kinetics400 dataset videos.

    Required args
    -------------
    - v_root : path
        Main folder containing videos for the dataset
    
    Optional args
    -------------
    - f_root : path (default=None)
        Main folder in which to save frames for the dataset. If None, 
        a default location is identified, based on v_root.
    - dim : int (default=None)
        Dimension to use in resizing images (shortest side). If None, 
        a default value of 150 is used.
    - parallel : bool (default=True)
        If True, frames for different videos for a class are extracted in 
        parallel
    """
    
    logger.info("Extracting frames for Kinetics400... ")

    if f_root is None:
        f_root = Path(Path(v_root).parent, "frames")

    Path(f_root).mkdir(exist_ok=True, parents=True)

    if dim is None:
        dim = 150 # default value

    extract_videos_opencv(
        v_root, f_root, dim=dim, video_ext="mp4", parallel=parallel
        )


#############################################
def main_MouseSim(v_root, f_root=None, eye="all", parallel=True):
    """
    main_MouseSim(v_root)

    Extracts frames from MouseSim dataset videos.

    Required args
    -------------
    - v_root : path
        Main folder containing videos for the dataset
    
    Optional args
    -------------
    - eye : str (default="all")
        Eye views for which to extract frames 
        ('all', 'both', 'left' or 'right').
    - f_root : path (default=None)
        Main folder in which to save frames for the dataset. If None, 
        a default location is identified, based on v_root.
    - parallel : bool (default=True)
        If True, frames for different videos for a class are extracted in 
        parallel.
    """
    
    if eye == "all":
        eyes = [False, "left", "right"]
    elif eye == "both":
        eyes = [False]
    elif eye in ["left", "right"]:
        eyes = [eye]
    else:
        raise ValueError("'eye' must be 'all', 'both', 'left' or 'right'.")

    f_root_orig = f_root
    for eye in eyes:
        eye_str_pr = f" ({eye} eye only)" if eye else "" 
        eye_str = f"_{eye}" if eye else ""

        f_root = f_root_orig
        if f_root is None:
            f_root = Path(Path(v_root).parent, f"frames{eye_str}")
        else:
            f_root = Path(f"{str(f_root)}{eye_str}")

        logger.info(f"Extracting frames for MouseSim{eye_str_pr}... ")
        extract_videos_opencv(
            v_root, f_root, video_ext="mov", half_wid=eye, parallel=parallel
            )


#############################################
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--d_root", type=Path,
        help=("root directory for the datasets, which should have the "
            "structure {args.d_root}/{args.dataset}/videos."))
    parser.add_argument("--dataset", default="UCF101", help="dataset name")
    parser.add_argument("--k400_big", action="store_true", 
        help=("if True, and dataset is k400, the larger version frames "
            "are extracted and stored (256 instead of 150)."))
    parser.add_argument("--eye", default="all", 
        help=("if 'all', frames for each eye view ('left', 'right', 'both') "
            "are extracted and stored."))

    parser.add_argument("--log_level", default="info", 
        help="logging level, e.g., debug, info, error")
    parser.add_argument("--not_parallel", action="store_true", 
        help=("extract frames from videos sequentially, instead of "
            "in parallel (much slower)"))

    args = parser.parse_args()

    misc_utils.get_logger_with_basic_format(level=args.log_level)
    parallel = not(args.not_parallel)

    dataset = misc_utils.normalize_dataset_name(args.dataset)

    dim, dim_str = None, ""
    if dataset == "Kinetics400" and args.k400_big:
        dim = 256
        dim_str = f"_{dim}"

    # identify main paths
    v_root = Path(args.d_root, dataset, "videos")
    f_root = Path(args.d_root, dataset, f"frames{dim_str}")

    if not v_root.is_dir():
        raise OSError(f"{v_root} not found.")

    if dataset == "UCF101":
        main_UCF101(v_root=v_root, f_root=f_root, parallel=parallel)

    elif dataset == "HMDB51":
        main_HMDB51(v_root=v_root, f_root=f_root, parallel=parallel)

    elif dataset == "Kinetics400":
        main_Kinetics400(
            v_root=v_root, f_root=f_root, dim=dim, parallel=parallel
            )

    elif args.dataset == "MouseSim":
        main_MouseSim(
            v_root=v_root, f_root=f_root, eye=args.eye, parallel=parallel
            )

    elif args.dataset == "Gabors":
        raise ValueError(
            "Gabors are generated on the fly, and do not require frame "
            "extraction."
            )

    else:
        raise ValueError(f"{dataset} dataset not recognized.")

