#!/usr/bin/env python

import argparse
import glob
import logging
from pathlib import Path
import sys

import csv
from joblib import Parallel, delayed
import pandas as pd
import tqdm

sys.path.extend(["..", str(Path("..", "utils"))])
from utils import training_utils

logger = logging.getLogger(__name__)


#############################################
def write_list(data_list, target_path):
    with open(target_path, "w") as f:
        writer = csv.writer(f, delimiter=",")
        for row in data_list:
            if row: 
                writer.writerow(row)
    logger.info(f"Split saved to {target_path}.")


#############################################
def main_UCF101(f_root, splits_root, csv_root=None):
    """
    generate training/testing split, count number of available frames, 
    save in csv
    """

    if csv_root is None:
        csv_root = Path("..", "data", "ucf101")

    Path(csv_root).mkdir(exist_ok=True, parents=True)
    for which_split in [1, 2, 3]:
        train_set = []
        test_set = []
        train_split_file = Path(splits_root, f"trainlist{which_split:02}.txt")
        with open(train_split_file, "r") as f:
            for line in f:
                vpath = Path(f_root, line.split(" ")[0])
                vpath_no_suffix = Path(vpath.parent, vpath.stem)
                train_set.append(
                    [vpath_no_suffix, 
                    len(glob.glob(str(Path(vpath_no_suffix, "*.jpg"))))]
                    )

        test_split_file = Path(splits_root, f"testlist{which_split:02}.txt")
        with open(test_split_file, "r") as f:
            for line in f:
                vpath = Path(f_root, line.rstrip())
                vpath_no_suffix = Path(vpath.parent, vpath.stem)
                test_set.append(
                    [vpath_no_suffix, 
                    len(glob.glob(str(Path(vpath_no_suffix, "*.jpg"))))]
                    )

        write_list(
            train_set, str(Path(csv_root, f"train_split{which_split:02}.csv"))
            )
        write_list(
            test_set, str(Path(csv_root, f"test_split{which_split:02}.csv"))
            )


#############################################
def main_HMDB51(f_root, splits_root, csv_root=None):
    """
    generate training/testing split, count number of available frames, 
    save in csv
    """

    if csv_root is None:
        csv_root = Path("..", "data", "hmdb51")

    Path(csv_root).mkdir(exist_ok=True, parents=True)
    for which_split in [1, 2, 3]:
        train_set = []
        test_set = []
        pattern = f"_test_split{which_split}.txt"
        split_files = sorted(glob.glob(str(Path(splits_root, f"*{pattern}"))))
        assert len(split_files) == 51
        for split_file in split_files:
            action_name = Path(split_file).name.replace(pattern, "")
            with open(split_file, "r") as f:
                for line in f:
                    video_name = Path(line.split(" ")[0])
                    vpath = Path(
                        f_root, action_name, video_name.parent, video_name.stem
                        )
                    _type = line.split(" ")[1]
                    if _type == "1":
                        train_set.append(
                            [vpath, len(glob.glob(str(Path(vpath, "*.jpg"))))]
                            )
                    elif _type == "2":
                        test_set.append(
                            [vpath, len(glob.glob(str(Path(vpath, "*.jpg"))))]
                            )

        write_list(
            train_set, str(Path(csv_root, f"train_split{which_split:02}.csv"))
            )
        write_list(
            test_set, str(Path(csv_root, f"test_split{which_split:02}.csv"))
            )


### For Kinetics ###
#############################################
def get_split(root, split_path, mode):
    logger.info(f"Processing {mode} split...")
    logger.info(f"Checking {root}")
    split_list = []
    split_content = pd.read_csv(str(split_path)).iloc[:, 0:4]
    n_jobs = training_utils.get_n_jobs(len(split_content))
    split_list = Parallel(n_jobs=n_jobs)(
        delayed(check_exists)(row, root)
        for _, row in tqdm(split_content.iterrows(), total=len(split_content))
        )
    return split_list


#############################################
def check_exists(row, root):
    dirname = "_".join(
        [row["youtube_id"], f"{row['time_start']:06}", f"{row['time_end']:06}"]
        )
    full_dirname = Path(root, row["label"], dirname)
    if full_dirname.is_dir():
        n_frames = len(glob.glob(str(Path(full_dirname, "*.jpg"))))
        return [full_dirname, n_frames]
    else:
        return None


#############################################
def main_Kinetics400(mode, k400_path, f_root, csv_root):
    Path(csv_root).mkdir(exist_ok=True, parents=True)

    train_split_path = Path(k400_path, "kinetics_train", "kinetics_train.csv")
    val_split_path = Path(k400_path, "kinetics_val", "kinetics_val.csv")
    test_split_path = Path(k400_path, "kinetics_test", "kinetics_test.csv")

    if mode == "train":
        train_split = get_split(
            Path(f_root, "train_split"), train_split_path, "train"
            )
        write_list(train_split, Path(csv_root, "train_split.csv"))
    elif mode == "val":
        val_split = get_split(Path(f_root, "val_split"), val_split_path, "val")
        write_list(val_split, Path(csv_root, "val_split.csv"))
    elif mode == "test":
        test_split = get_split(f_root, test_split_path, "test")
        write_list(test_split, Path(csv_root, "test_split.csv"))
    else:
        raise ValueError("'mode' must be 'train', 'val' or 'test'.")


#############################################
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--f_root", default=None,
        help=("if different from args.v_root, root directory in which to save "
            "the csv with the list of paths, which will have the structure "
            "{args.f_root}/{args.dataset}/{splits name}.")
        )

    parser.add_argument("--dataset", default="UCF101", help="dataset name")

    args = parser.parse_args()

    splits_root = Path(args.f_root)
    args.f_root = Path(args.f_root, args.dataset, "frames")

    if not args.f_root.is_dir():
        raise OSError(f"{args.f_root} not found.")

    if args.dataset.lower() == "ucf101":
        splits_root = Path(splits_root, "splits_classification")
        main_UCF101(f_root=args.f_root, splits_root=splits_root)

    elif args.dataset.lower() == "hmdb51":
        splits_root = Path(splits_root, "split", "testTrainMulti_7030_splits")
        main_HMDB51(f_root=args.f_root, splits_root=splits_root)

    elif args.dataset.lower() in ["kinetics400", "kinetics400_256"]:
        k400_path = Path(args.f_root, "Kinetics")
        csv_root = Path(args.f_root, args.dataset)
        if not csv_root.is_file():
            raise OSError(f"{csv_root} not found.")

        main_Kinetics400(
            mode=args.mode,
            k400_path=k400_path, 
            f_root=args.f_root, 
            csv_root=csv_root
            )
    
    else:
        raise ValueError(f"{args.dataset} dataset not recognized.")

