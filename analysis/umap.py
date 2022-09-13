#!/usr/bin/env python
 
import argparse
import glob
import logging
from pathlib import Path
import sys

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pickle as pkl
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap

sys.path.extend(["..", str(Path("..", "utils")), str(Path("..", "analysis"))])
from analysis import analysis_utils, hooks
from utils import misc_utils

logger = logging.getLogger(__name__)

TAB = "    "


#############################################
def get_umap_df(data, labels, hook_module="encoder", hook_type="activations", 
                num_batches=10, pred_step=1):
    """
    get_umap_df(data, labels)

    Computes a 2D UMAP projetion for the data, and returns the results in a 
    dataframe.

    Required args
    -------------
    - data : 2D array
        Data on which to compute the UMAP projection, with dims: 
            blocks (num_batches * num_per * pred_step) x features
    - labels

    Optional args
    -------------
    - hook_module : str (default="encoder")
        Module for which dataframe is being created. 
    - hook_type : str (default="activations")
        Hook type for which dataframe is being created.

    Returns
    -------
    - umap_df : pd.DataFrame
        UMAP dataframe, with, for each datapoint, the columns:
        'dim0'       : value in the first UMAP dimension
        'dim1'       : value in the second UMAP dimension
        'image'      : Gabor image 
        'orientation': Gabor orientation
        'hook_module': hook module, i.e., 'encoder', 'contextual', 'predictive'
        'hook_type'  : hook type, i.e., 'activations' or 'errors'
        'pred_step'  : predicted step
        'batch_num'  : batch number
    """

    N_all = len(data)

    N_per = int(N_all / (num_batches * pred_step))
    if N_per != N_all / (num_batches * pred_step):
        raise ValueError(
            "'num_batches' or 'pred_step' is wrong, based on the shape of "
            "'data'."
            )
        
    pip = Pipeline([
        ('scaler', StandardScaler()), 
        ('umap', umap.UMAP(min_dist=0.4))
        ])
    
    proj = pip.fit_transform(data)

    umap_df = pd.DataFrame()
    umap_df["dim0"], umap_df["dim1"] = proj.T
    umap_df["image"], umap_df["orientation"] = list(zip(*labels))

    umap_df["orientation"] = [
        int(val) if val != "N/A" and int(val) == val else val 
        for val in umap_df["orientation"]
        ]

    umap_df["hook_module"] = hook_module
    umap_df["hook_type"] = hook_type
    
    pred_step_arr = np.empty([num_batches, N_per, pred_step])
    for i in range(pred_step):
        pred_step_arr[..., i] = i
    umap_df["pred_step"] = pred_step_arr.reshape(-1)

    batch_nums = np.empty([num_batches, N_per, pred_step])
    for i in range(num_batches):
        batch_nums[i] = i
    umap_df["batch_num"] = batch_nums.reshape(-1)

    return umap_df


#############################################
def plot_save_umaps(umap_df, epoch_str="0", pre=True, learn=False, 
                    output_dir=".", save_data=False):
    """
    plot_save_umaps(umap_df)

    Plots and saves UMAPs.

    Required args
    -------------
    - umap_df : pd.DataFrame
        UMAP dataframe, with, for each datapoint, the columns:
        'dim0'       : value in the first UMAP dimension
        'dim1'       : value in the second UMAP dimension
        'image'      : Gabor image 
        'orientation': Gabor orientation
        'hook_module': hook module, i.e., 'encoder', 'contextual', 'predictive'
        'hook_type'  : hook type, i.e., 'activations' or 'errors'
        'pred_step'  : predicted step
        'batch_num'  : batch number

    Optional args
    -------------
    - epoch_str : str (default: "0")
        String providing epoch information, for saving.
    - pre : bool (default: True)
        If True, hook data is being collected before epoch training in the main 
        loop.
    - learn : bool (default: False)
        If True, hook data is collected while model is being trained at a 
        slower rate. If True, UMAPs colour-coded by batch number are also 
        plotted.
    - output_dir : str or path (default=".")
        Main output directory in which to save collected data.
    - save_data : bool (default=False)
        If True, data is saved to h5 file in the output directory.
    """

    epoch_str, pre_str = analysis_utils.get_epoch_pre_str(epoch_str, pre, learn)

    if save_data:
       umap_dir = analysis_utils.save_dict_pkl(
            umap_df.to_dict(), output_dir=output_dir, epoch_str=epoch_str, 
            pre=pre, learn=learn, dict_type="umap", 
            )
    else:
        umap_dir = analysis_utils.get_dict_path(
            output_dir, dict_type="umap").parent
        umap_dir.mkdir(parents=True, exist_ok=True)

    for label in ["image", "orientation", "pred_step", "batch_num"]:
        palette = "viridis"

        unique = umap_df[label].unique()
        if len(unique) == 1:
            continue

        if label == "orientation" and "N/A" in unique.astype(str):
            no_na = unique[np.where(unique != "N/A")]
            idx_order = np.argsort(no_na.astype(float))
            order = [no_na[o] for o in idx_order] + ["N/A"]
        elif label == "batch_num" and not learn:
            continue
        else:
            order = sorted(unique)
            if label == "image" and "D" in unique and "U" not in order:
                order = list(order) + ["U"]
            if label == "image" and len(order) > 3 and "U" in order:
                palette = sns.color_palette("viridis", len(order) - 1)
                palette.append(sns.color_palette("tab10")[3])
        
        g = sns.relplot(
            data=umap_df, x="dim0", y="dim1", 
            hue=label, hue_order=order, 
            col="hook_module", col_order=hooks.HOOK_MODULES,
            row="hook_type", row_order=hooks.HOOK_TYPES,
            alpha=0.3, palette=palette, zorder=-10,
            facet_kws={"sharex": False, "sharey": False},
            )
        
        sns.despine(left=True, bottom=True)
        g.set(xticks=[], xticklabels=[], yticks=[], yticklabels=[])
        for m, hook_module in enumerate(hooks.HOOK_MODULES):
            for t, hook_type in enumerate(hooks.HOOK_TYPES):
                g.axes[t, m].set_title("")
                g.axes[t, m].set_xlabel("")
                if t == 0:
                    g.axes[t, m].set_title(hook_module.capitalize())
                if m == 0:
                    g.axes[t, m].set_ylabel(hook_type.capitalize())

        for subax in g.axes.ravel():
            subax.set_rasterization_zorder(-9)

        savepath = Path(umap_dir, f"{pre_str}_{epoch_str}_{label}.svg")
        savepath.parent.mkdir(exist_ok=True, parents=True)
        g.savefig(savepath, format="svg", bbox_inches="tight", dpi=600)


#############################################
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", default="output_dir", 
        help="directory in which to find UMAP pickles for replotting")
    parser.add_argument('--log_level', default='info', 
                        help='logging level, e.g., debug, info, error')
    args = parser.parse_args()

    misc_utils.get_logger_with_basic_format(level=args.log_level)

    umap_pickles = glob.glob(
        str(Path(args.output_dir, "**", "*umap*.pkl")), recursive=True
        )
    
    for umap_pickle in umap_pickles:
        logger.info(f"Replotting from {umap_pickle}.", extra={"spacing": "\n"})

        with open(umap_pickle, "rb") as f:
            data_dict = pkl.load(f)
        umap_pickle_dir = Path(umap_pickle).parent.parent
        
        for pre_learn_str, pre_dict in data_dict.items():
            pre = "pre" in pre_learn_str
            learn = "learn" in pre_learn_str
            logger.info(f"{pre_learn_str}/{epoch_str}", extra={"spacing": TAB})
            for epoch_str, umap_dict in pre_dict.items():
                plot_save_umaps(
                    pd.DataFrame(umap_dict), epoch_str=epoch_str, pre=pre, 
                    learn=learn, output_dir=umap_pickle_dir, save_data=False
                    )
                plt.close("all")


