#!/usr/bin/env python

import argparse
import glob
import logging
from pathlib import Path
import sys

import csv
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from tqdm import tqdm 

sys.path.extend([
    "..", 
    str(Path("..", "..")), 
    str(Path("..", "..", "utils")), 
    ])
from utils import misc_utils

logger = logging.getLogger(__name__)


#############################################
def write_list(data_list, target_path, delimiter=","):
    """
    write_list(data_list, target_path)

    Writes data to a CSV.

    Required args
    -------------
    - data_list : list
        List of data to write for each row 
    - target_path : path
        Path to csv file to which to write data_list
    
    Optional args
    -------------
    - delimiter : str (default=",")
        Column delimiter
    """
    
    with open(target_path, "w") as f:
        writer = csv.writer(f, delimiter=delimiter)
        for row in data_list:
            if row: 
                writer.writerow(row)
    logger.info(f"Split saved to {target_path}.")


#############################################
def write_class_index_file(class_names, csv_root, expected_n=None):
    """
    write_class_index_file(class_names, csv_root)

    Write class indices to a txt file.

    Required args
    -------------
    - class_names : list
        List of class names
    - csv_root : path
        Directory under which to save class indices file. 

    Optional args
    -------------
    - expected_n : int (default=None)
        If not None, expected number of unique class names
    """

    class_names = sorted(list(set(class_names)))

    if expected_n is not None and len(class_names) != expected_n:
        raise RuntimeError(
            f"Expected to find {expected_n} classes, but found "
            f"{len(class_names)}."
            )

    class_index_lines = [
        f"{idx} {class_name}" for idx, class_name in enumerate(class_names)
        ]
    class_index_lines = "\n".join(class_index_lines)

    target_path = Path(csv_root, "classInd.txt")
    with open(target_path, "w") as f:
        f.writelines(class_index_lines)

    logger.info(f"Class indices saved to {target_path}.")


#############################################
def main_UCF101(f_root, splits_root=None, csv_root=None):
    """
    main_UCF101(f_root)

    Creates a CSV for each split, listing the frames directory name, and number 
    of extracted frames for each included video.  

    Required args
    -------------
    - f_root : path
        Main folder in which frames for the dataset are stored.

    Optional args
    -------------
    - splits_root : path (default=None)
        Path to folder specifying video splits. If None, it is inferred 
        from f_root.
    - csv_root : path (default=None)
        Path in which to store CSVs for each split. If None, the parent 
        directory of f_root is used.
    """

    if splits_root is None:
        splits_root = Path(Path(f_root).parent, "splits")

    for root in [f_root, splits_root]:
        if not Path(root).is_dir():
            raise ValueError(f"{root} does not exist.")

    if csv_root is None:
        csv_root = Path(f_root).parent

    Path(csv_root).mkdir(exist_ok=True, parents=True)
    for split_n in [1, 2, 3]:
        train_set = []
        test_set = []
        train_split_file = Path(splits_root, f"trainlist{split_n:02}.txt")
        with open(train_split_file, "r") as f:
            for line in f:
                vpath = Path(f_root, line.split(" ")[0])
                vpath_no_suffix = Path(vpath.parent, vpath.stem)
                train_set.append(
                    [vpath_no_suffix, 
                    len(glob.glob(str(Path(vpath_no_suffix, "*.jpg"))))]
                    )

        test_split_file = Path(splits_root, f"testlist{split_n:02}.txt")
        with open(test_split_file, "r") as f:
            for line in f:
                vpath = Path(f_root, line.rstrip())
                vpath_no_suffix = Path(vpath.parent, vpath.stem)
                test_set.append(
                    [vpath_no_suffix, 
                    len(glob.glob(str(Path(vpath_no_suffix, "*.jpg"))))]
                    )

        write_list(
            train_set, str(Path(csv_root, f"train_split{split_n:02}.csv"))
            )
        write_list(
            test_set, str(Path(csv_root, f"test_split{split_n:02}.csv"))
            )
    
    # load class index file, and save a new version in csv_root
    class_index_file = Path(splits_root, "classInd.txt")
    if not class_index_file.is_file():
        raise OSError(
            f"Expected to find class index file under {class_index_file}."
            )
    class_index_df = pd.read_csv(str(class_index_file), sep=" ", header=None)
    write_class_index_file(class_index_df[1].tolist(), csv_root, expected_n=101)


#############################################
def main_HMDB51(f_root, splits_root=None, csv_root=None):
    """
    main_HMDB51(f_root)

    Creates a CSV for each split, listing the frames directory name, and number 
    of extracted frames for each included video.  

    Required args
    -------------
    - f_root : path
        Main folder in which frames for the dataset are stored.

    Optional args
    -------------
    - splits_root : path (default=None)
        Path to folder specifying video splits. If None, it is inferred 
        from f_root.
    - csv_root : path (default=None)
        Path in which to store CSVs for each split. If None, the parent 
        directory of f_root is used.
    """

    if splits_root is None:
        splits_root = Path(Path(f_root).parent, "splits")

    for root in [f_root, splits_root]:
        if not Path(root).is_dir():
            raise ValueError(f"{root} does not exist.")

    if csv_root is None:
        csv_root = Path(f_root).parent


    Path(csv_root).mkdir(exist_ok=True, parents=True)
    for split_n in [1, 2, 3]:
        class_names = []
        train_set = []
        test_set = []
        pattern = f"_test_split{split_n}.txt"
        split_files = sorted(glob.glob(str(Path(splits_root, f"*{pattern}"))))
        if len(split_files) != 51:
            raise RuntimeError(
                f"Expected 51 split files, but found {len(split_files)}."
                )
        for split_file in split_files:
            class_name = Path(split_file).name.replace(pattern, "")
            class_names.append(class_name)
            with open(split_file, "r") as f:
                for line in f:
                    video_name = Path(line.split(" ")[0])
                    vpath = Path(
                        f_root, class_name, video_name.parent, video_name.stem
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
            train_set, str(Path(csv_root, f"train_split{split_n:02}.csv"))
            )
        write_list(
            test_set, str(Path(csv_root, f"test_split{split_n:02}.csv"))
            )

    write_class_index_file(class_names, csv_root, expected_n=51)


### For Kinetics ###
#############################################
def check_exists(split_row, split_root):
    """
    check_exists(split_row, split_root)

    Checks whether a Kinetics400 video frames directory exists, and if so, 
    returns the directory name and the number of frames it contains.

    Required args
    -------------
    - split_row : list
        Data for reconstructing the video directory name
    - split_root : path
        Root directory for the split

    Returns
    -------
    - list
        Path to the frames directory and number of frames for the video, 
        or None if directory doesn't exist. 
    """
    
    dirname = "_".join(
        [split_row["youtube_id"], 
         f"{split_row['time_start']:06}", 
         f"{split_row['time_end']:06}"]
        )
    full_dirname = Path(split_root, split_row["label"], dirname)
    if full_dirname.is_dir():
        n_frames = len(glob.glob(str(Path(full_dirname, "*.jpg"))))
        return [full_dirname, n_frames]
    else:
        return None


#############################################
def get_split(f_root, split_path, mode="train"):
    """
    get_split(f_root, split_path)

    Returns list of video directory names and number of frames for each 
    Kinetics400 video.

    Required args
    -------------
    - f_root : path
        Main folder in which frames for the dataset are stored.
    - split_path : path
        Kinetics400 file specifying the split content

    Optional args
    -------------
    - mode : str (default="train")
        Data mode ("train" or "val")

    Returns 
    -------
    - split_list : list
        List of frames directory name and number of frames for each video 
        [full_dirname, n_frames], with None for missing videos.
    """

    for pathname in [f_root, split_path]:
        if not Path(pathname).exists():
            raise ValueError(f"{pathname} does not exist.")

    logger.info(f"Processing {mode} split...")
    logger.info(f"Checking {f_root}")
    split_list = []
    split_content = pd.read_csv(str(split_path)).iloc[:, 0:4]
    n_jobs = misc_utils.get_num_jobs(len(split_content))
    split_list = Parallel(n_jobs=n_jobs)(
        delayed(check_exists)(row, f_root)
        for _, row in tqdm(split_content.iterrows(), total=len(split_content))
        )
        
    return split_list


#############################################
def main_Kinetics400(f_root, splits_root=None, modes=["train", "val", "test"], 
                     csv_root=None):
    """
    main_Kinetics400(f_root)

    Creates a CSV for each split, listing the frames directory name, and number 
    of extracted frames for each included video.  

    Required args
    -------------
    - f_root : path
        Main folder in which frames for the dataset are stored.

    Optional args
    -------------
    - splits_root : path (default=None)
        Path to folder specifying video splits. If None, it is inferred 
        from f_root.
    - csv_root : path (default=None)
        Path in which to store CSVs for each split. If None, the parent 
        directory of f_root is used.
    """

    if splits_root is None:
        splits_root = Path(Path(f_root).parent, "annotations")

    for root in [f_root, splits_root]:
        if not Path(root).is_dir():
            raise ValueError(f"{root} does not exist.")

    if csv_root is None:
        csv_root = Path(f_root).parent

    Path(csv_root).mkdir(exist_ok=True, parents=True)

    if not isinstance(modes, list):
        modes = [modes]

    for mode in modes:
        if mode not in ["train", "val", "test"]:
            raise ValueError("'mode' must be 'train', 'val' or 'test'.")

        mode_path = Path(splits_root, f"{mode}_split.csv")    

        target_csv = Path(csv_root, f"{mode}_split.csv")
        if target_csv.is_file():
            logger.info(
                f"Skipping {mode} split, as {target_csv} already exists."
                )
            continue

        split = get_split(f_root, mode_path, mode)
        write_list(split, target_csv)

    # create class index file
    split_df = pd.read_csv(mode_path)
    class_names = [
        class_name.replace("(", "").replace(")", "").replace(" ", "_") 
        for class_name in split_df["label"].tolist()
        ]
    write_class_index_file(class_names, csv_root, expected_n=400)


### For MouseSim ###
#############################################
def main_MouseSim(f_root, csv_root=None, splits_seed=100, prop_test=0.2):
    """
    main_MouseSim(f_root)

    Creates a CSV for each split, listing the frames directory name, and number 
    of extracted frames for each included video.  

    Required args
    -------------
    - f_root : path
        Main folder in which frames for the dataset are stored.

    Optional args
    -------------
    - splits_seed : int (default=100)
        Random seed to use to split the videos into splits
    - csv_root : path (default=None)
        Path in which to store CSVs for each split. If None, the parent 
        directory of f_root is used.
    - prop_test : float (default=0.2)
        Proportion of videos per class to put into test split
    """

    splits_rng = np.random.RandomState(splits_seed)
    if prop_test <= 0 or prop_test >= 1:
        raise ValueError(
            "'prop_test' must be greater than 0 and smaller than 1."
            )

    if not Path(f_root).is_dir():
        raise ValueError(f"{f_root} does not exist.")

    if csv_root is None:
        csv_root = Path(f_root).parent

    csv_root_orig = csv_root
    f_root_orig = f_root
    found_eye = False
    for eye in [False, "left", "right"]:
        eye_str = f"_{eye}" if eye else ""
        f_root = f"{f_root_orig}{eye_str}"

        if not Path(f_root).is_dir():
            logger.warning(f"Did not find frames directory under '{f_root}'.")
            continue

        found_eye = True
        csv_root = f"{csv_root_orig}{eye_str}"
        Path(csv_root).mkdir(exist_ok=True, parents=True)

        class_names = [
            Path(direc).stem for direc in Path(f_root).iterdir() 
            if Path(direc).is_dir()
            ]
        train_split, test_split = [], []
        for class_name in class_names:
            video_direcs = list(sorted([
                Path(direc) for direc in Path(f_root, class_name).iterdir() 
                if Path(direc).is_dir()
                ]))

            n_videos = len(video_direcs)
            if n_videos < 2:
                raise RuntimeError("Expected at least 2 videos per class.")
            n_test = int(np.around(n_videos * prop_test))
            n_test = min([n_videos - 1, max([1, n_test])])

            test_idxs = np.sort(
                splits_rng.choice(n_videos, n_test, replace=False)
                )
            train_idxs = np.delete(np.arange(n_videos), test_idxs)

            for split, idxs in [("train", train_idxs), ("test", test_idxs)]:
                for i in idxs:
                    video_direc = video_direcs[i]
                    n_frames = len(
                        glob.glob(str(Path(video_direcs[i], "*.jpg")))
                        )
                    if split == "train":
                        split_list = train_split
                    elif split == "test":
                        split_list = test_split
                    else:
                        raise ValueError("'split' must be 'train' or 'test'.")
                    split_list.append([video_direc, n_frames])

        write_list(
            train_split, str(Path(csv_root, "train_split.csv"))
            )
        write_list(
            test_split, str(Path(csv_root, "test_split.csv"))
            )

        write_class_index_file(class_names, csv_root, expected_n=None)
    
    if not found_eye:
        raise ValueError("Did not find frames directory for any eye view.")


#############################################
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--d_root", type=Path,
        help=("root directory for the datasets, which should have the "
              "structure {args.d_root}/{args.dataset}."))
    parser.add_argument("--csv_root", type=Path, default=Path("..", "data"),
        help="root directory in which to save the csv with the list of paths.")

    parser.add_argument("--dataset", default="UCF101", help="dataset name")
    parser.add_argument("--k400_big", action="store_true", 
        help=("if True, and dataset is k400, the larger "
          "version frames are compiled (256 instead of 150)."))

    parser.add_argument("--log_level", default="info", 
        help="logging level, e.g., debug, info, error")

    args = parser.parse_args()

    misc_utils.get_logger_with_basic_format(level=args.log_level)

    dataset = misc_utils.normalize_dataset_name(args.dataset)

    dim, dim_str = None, ""
    if dataset == "Kinetics400" and args.k400_big:
        dim = 256
        dim_str = f"_{dim}"

    # identify main paths
    d_root = Path(args.d_root, dataset)
    f_root = Path(args.d_root, dataset, f"frames{dim_str}")
    csv_root = Path(args.csv_root, f"{dataset}{dim_str}")

    if dataset == "UCF101":
        splits_root = Path(d_root, "splits")
        main_UCF101(f_root=f_root, splits_root=splits_root, csv_root=csv_root)

    elif dataset == "HMDB51":
        splits_root = Path(d_root, "splits")
        main_HMDB51(f_root=f_root, splits_root=splits_root, csv_root=csv_root)

    elif dataset == "Kinetics400":
        splits_root = Path(d_root, "annotations")
        main_Kinetics400(
            f_root=f_root, splits_root=splits_root, csv_root=csv_root,
        )

    elif args.dataset == "MouseSim":
        main_MouseSim(f_root=f_root, csv_root=csv_root)

    elif args.dataset == "Gabors":
        raise ValueError(
            "Gabors are generated on the fly, and do not require CSV writing."
            )

    else:
        raise ValueError(f"{dataset} dataset not recognized.")

