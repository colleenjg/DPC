#!/usr/bin/env python

import argparse
import logging
from pathlib import Path
import glob 
import sys
import warnings

import cv2
from joblib import delayed, Parallel
from tqdm import tqdm 
from wurlitzer import pipes # catch C-level pipes in python

sys.path.extend([
    "..", 
    str(Path("..", "..")), 
    str(Path("..", "..", "utils")), 
    str(Path("..", "..", "dataset"))
    ])
from utils import misc_utils, training_utils
from dataset import dataset_3d

logger = logging.getLogger(__name__)


#############################################
def resize_dim(w, h, target):
    """
    resize (w, h)
    
    Resizes, such that the smaller side is target, to keep the aspect ratio.
    """

    if w == 0 or h == 0:
        raise ValueError("width and height must be non-zero.")

    if w >= h:
        return (int(target * w / h), int(target))
    else:
        return (int(target), int(target * h / w)) 


#############################################
def extract_video_opencv(v_path, f_root, dim=None):
    """
    extract_video_opencv(v_path, f_root)

    Extracts frames from a dataset video.

    Required args:
        - v_path (Path):
            path to the video from which to extract frames
        - f_root (Path): 
            main folder in which to save frames for the dataset
    
    Optional args:
        - dim (int):
            dimension to use in resizing images (shortest side). If None, 
            a default value is used (see extract_video_opencv()).
            default: None
    
    Returns:
        - success_code (float):
            success code, where a value of 
            1.0 indicates a fully successful extraction, 
            0.5 indicates a partial extraction and 
            0.0 indicates a failed extraction
    """
    
    if dim is None:
        dim = 240

    v_path = Path(v_path)
    v_class = v_path.parent.name
    out_dir = Path(f_root, v_class, v_path.stem)
    out_dir.mkdir(parents=True, exist_ok=True)

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
    new_dim = resize_dim(width, height, dim)

    count = 0
    success, image = vidcap.read()
    while success:
        count += 1
        image = cv2.resize(image, new_dim, interpolation=cv2.INTER_LINEAR)
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
                          parallel=True):
    """
    extract_videos_opencv(v_root, f_root)

    Extracts frames from a dataset's videos, for one action class at a time.

    Required args:
        - v_root (Path):
            main folder containing videos for the dataset
        - f_root (Path): 
            main folder in which to save frames for the dataset
    
    Optional args:
        - dim (int):
            dimension to use in resizing images (shortest side). If None, 
            a default value is used (see extract_video_opencv()).
            default: None
        - video_ext (str):
            video extension to use to identify videos for frame extraction
            default: "avi"
        - parallel (bool):
            if True, frames for different videos for an action class are 
            extracted in parallel
            default: True
    
    Returns:
        - all_success_code (list):
            nested list of success codes for videos from each action 
            directories (action class x video), where values of 
            1.0 indicate a fully successful extraction, 
            0.5 indicate a partial extraction and 
            0.0 indicate a failed extraction
    """

    if not Path(v_root).is_dir():
        raise ValueError(f"{v_root} is not a directory.")

    Path(f_root).mkdir(parents=True, exist_ok=True)

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

        v_class = Path(j).name # action class
        out_dir = Path(f_root, v_class)
        # if resuming, be sure to delete the last video folder, 
        # if the extraction was incomplete
        if out_dir.is_dir(): 
            logger.info(
                f"{out_dir} already exists. Skipping {v_class} action class."
                )
            continue

        logger.info(f"Extracting frames from {len(v_paths)} {v_class} videos.")
        if parallel:
            n_jobs = training_utils.get_num_jobs(len(v_paths))
            success_codes = Parallel(n_jobs=n_jobs)(
                delayed(extract_video_opencv)(p, f_root, dim=dim) 
                for p in tqdm(v_paths, total=len(v_paths))
                )
        else:
            success_codes = []
            for p in tqdm(v_paths, total=len(v_paths)):
                success_codes.append(
                    extract_video_opencv(p, f_root, dim=dim)
                    )
        all_success_codes.append(success_codes)
    
    return all_success_codes
            

#############################################
def main_UCF101(v_root, f_root=None, parallel=True):
    """
    main_UCF101(v_root)

    Extracts frames from UCF101 dataset videos.

    Required args:
        - v_root (Path):
            main folder containing videos for the dataset
    
    Optional args:
        - f_root (Path): 
            main folder in which to save frames for the dataset. If None, 
            a default location is identified, based on v_root.
            default: None
        - parallel (bool):
            if True, frames for different videos for an action class are 
            extracted in parallel
            default: True
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

    Required args:
        - v_root (Path):
            main folder containing videos for the dataset
    
    Optional args:
        - f_root (Path): 
            main folder in which to save frames for the dataset. If None, 
            a default location is identified, based on v_root.
            default: None
        - parallel (bool):
            if True, frames for different videos for an action class are 
            extracted in parallel
            default: True
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

    Required args:
        - v_root (Path):
            main folder containing videos for the dataset
    
    Optional args:
        - f_root (Path): 
            main folder in which to save frames for the dataset. If None, 
            a default location is identified, based on v_root.
            default: None
        - dim (int):
            dimension to use in resizing images (shortest side). If None, 
            a default value of 150 is used.
            default: None
        - parallel (bool):
            if True, frames for different videos for an action class are 
            extracted in parallel
            default: True
    """
    
    logger.info("Extracting frames for Kinetics400... ")

    if f_root is None:
        f_root = Path(Path(v_root).parent, "frames")

    Path(f_root).mkdir(parents=True, exist_ok=True)

    if dim is None:
        dim = 150 # default value

    for split in ["train", "val", "test"]:
        v_root_split = Path(v_root, f"{split}_split")
        f_root_split = Path(f_root, f"{split}_split")

        extract_videos_opencv(
            v_root_split, f_root_split, dim=dim, video_ext="mp4", 
            parallel=parallel
            )


#############################################
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--d_root", type=Path,
        help=("root directory for the datasets, which should have the "
            "structure {args.d_root}/{args.dataset}/videos.")
        )
    parser.add_argument("--dataset", default="UCF101", help="dataset name")
    parser.add_argument("--k400_big", action="store_true", 
                        help=("if True, and dataset is k400, the larger "
                        "version frames are stored (256 instead of 150)."))

    parser.add_argument("--log_level", default="info", 
                        help="logging level, e.g., debug, info, error")
    parser.add_argument("--not_parallel", action="store_true", 
                        help=("extract frames from videos sequentially, "
                        "instead of in parallel (much slower)"))

    args = parser.parse_args()

    misc_utils.get_logger_with_basic_format(level=args.log_level)
    parallel = not(args.not_parallel)

    dataset = dataset_3d.normalize_dataset_name(args.dataset)

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
    
    else:
        raise ValueError(f"{dataset} dataset not recognized.")

