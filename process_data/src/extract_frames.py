#!/usr/bin/env python

import argparse
import logging
from pathlib import Path
import glob 
import sys

import cv2
from joblib import delayed, Parallel
from tqdm import tqdm 

sys.path.extend(["..", str(Path("..", "utils"))])
from utils import training_utils

logger = logging.getLogger(__name__)


#############################################
def extract_video_opencv(v_path, f_root, dim=240):
    """
    v_path: single video path;
    f_root: root to store frames
    """
       
    v_path = Path(v_path)
    v_class = v_path.parent.name
    out_dir = Path(f_root, v_class, v_path.stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    vidcap = cv2.VideoCapture(v_path)
    nb_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
    height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
    if (width == 0) or (height==0): 
        logger.error(
            f"{v_path} was not successfully loaded, and will be skipped..."
            )
        return
    new_dim = resize_dim(width, height, dim)

    success, image = vidcap.read()
    count = 1
    while success:
        image = cv2.resize(image, new_dim, interpolation = cv2.INTER_LINEAR)
        # quality from 0-100, 95 is default, high is good
        cv2.imwrite(
            str(Path(out_dir, f"image_{count:05}.jpg")), 
            image,
            [cv2.IMWRITE_JPEG_QUALITY, 80]
            )
        success, image = vidcap.read()
        count += 1
    if nb_frames > count:
        logger.error(
            f"{Path(Path(out_dir).parts[-2:])} NOT extracted successfully: "
            f"{count}/{nb_frames} frames."
            )
    vidcap.release()


#############################################
def resize_dim(w, h, target):
    """
    resize (w, h), such that the smaller side is target, keep the aspect ratio
    """

    if w >= h:
        return (int(target * w / h), int(target))
    else:
        return (int(target), int(target * h / w)) 


#############################################
def main_UCF101(v_root, f_root):
    logger.info("Extracting for UCF101... ")
    logger.info(
        f"Extracting frames from videos in\n{v_root}"
        f"\nand saving to\n{f_root}."
        )
    
    Path(f_root).mkdir(parents=True, exist_ok=True)

    v_act_root = glob.glob(str(Path(v_root, "*")))
    v_act_root = sorted([j for j in v_act_root if Path(j).is_dir()])
    for _, j in tqdm(enumerate(v_act_root), total=len(v_act_root)):
        v_paths = glob.glob(str(Path(j, "*.avi")))
        v_paths = sorted(v_paths)
        n_jobs = training_utils.get_n_jobs(len(v_paths))
        Parallel(n_jobs=n_jobs)(
            delayed(extract_video_opencv)(p, f_root) 
            for p in tqdm(v_paths, total=len(v_paths))
            )


#############################################
def main_HMDB51(v_root, f_root):
    logger.info("Extracting for HMDB51... ")
    logger.info(
        f"Extracting frames from videos in\n{v_root}"
        f"\nand saving to\n{f_root}."
        )
    
    Path(f_root).mkdir(parents=True, exist_ok=True)

    v_act_root = glob.glob(str(Path(v_root, "*")))
    v_act_root = sorted([j for j in v_act_root if Path(j).is_dir()])
    for _, j in tqdm(enumerate(v_act_root), total=len(v_act_root)):
        v_paths = glob.glob(str(Path(j, "*.avi")))
        v_paths = sorted(v_paths)
        n_jobs = training_utils.get_n_jobs(len(v_paths))
        Parallel(n_jobs=n_jobs)(
            delayed(extract_video_opencv)(p, f_root) 
            for p in tqdm(v_paths, total=len(v_paths))
            )


#############################################
def main_Kinetics400(v_root, f_root, dim=150):
    logger.info("Extracting for Kinetics400... ")
    for basename in ["train_split", "val_split"]:
        v_root_split = Path(v_root, basename)
        if not v_root_split.exists():
            raise OSError(f"v_root does not exist: {v_root_split}")
        logger.info(
            f"Extracting frames from videos in\n{v_root_split}"
            f"\nand saving to\n{f_root}."
            )

        Path(f_root).mkdir(parents=True, exist_ok=True)

        v_act_root = glob.glob(str(Path(v_root_split, "*")))
        v_act_root = sorted([j for j in v_act_root if Path(j).is_dir()])

        for i, j in tqdm(enumerate(v_act_root), total=len(v_act_root)):
            v_paths = glob.glob(str(Path(j, "*.mp4")))
            v_paths = sorted(v_paths)
            v_class = Path(j).name # action class
            out_dir = Path(f_root, v_class)
            # if resuming, remember to delete the last video folders, 
            # if it is incomplete
            if out_dir.exists(): 
                logger.info(f"{out_dir} already exists.")
                continue
            logger.info(f"Extracting for {v_class} action class.")
            # dim = 150 (crop to 128 later) or 256 (crop to 224 later)
            n_jobs = training_utils.get_n_jobs(len(v_paths))
            Parallel(n_jobs=n_jobs)(
                delayed(extract_video_opencv)(p, f_root, dim=dim) 
                for p in tqdm(v_paths, total=len(v_paths)))


#############################################
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--v_root", default="data",
        help=("root directory for the videos, which should have the "
            "structure {args.v_root}/{args.dataset}/videos.")
        )

    parser.add_argument("--dataset", default="UCF101", help="dataset name")

    args = parser.parse_args()

    args.v_root = Path(args.v_root, args.dataset, "videos")
    f_root = Path(args.v_root, args.dataset, "frames")

    if not args.v_root.is_dir():
        raise OSError(f"{args.v_root} not found.")

    if args.dataset.lower() == "ucf101":
        main_UCF101(v_root=args.v_root, f_root=f_root)

    elif args.dataset.lower() == "hmdb51":
        main_HMDB51(v_root=args.v_root, f_root=f_root)

    elif args.dataset.lower() == "kinetics400":
        main_Kinetics400(v_root=args.v_root, f_root=f_root)
    
    elif args.dataset.lower() == "kinetics400_256":
        main_Kinetics400(v_root=args.v_root, f_root=f_root, dim=256)

    else:
        raise ValueError(f"{args.dataset} dataset not recognized.")

