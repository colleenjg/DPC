from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap

from analysis import analysis_utils, hooks


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
def plot_save_umaps(umap_df, output_dir=".", epoch_str="0", pre=True, 
                    learn=False):
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
    - output_dir : str or path (default=".")
        Main output directory in which to save collected data.
    - epoch_str : str (default: "0")
        String providing epoch information, for saving.
    - pre : bool (default: True)
        If True, hook data is being collected before epoch training in the main 
        loop.
    - learn : bool (default: False)
        If True, hook data is collected while model is being trained at a 
        slower rate. If True, UMAPs colour-coded by batch number are also 
        plotted.
    """

    epoch_str, pre_str = analysis_utils.get_epoch_pre_str(epoch_str, pre, learn)
    umap_dir = analysis_utils.save_dict_pkl(
        umap_df.to_dict(), output_dir=output_dir, epoch_str=epoch_str, pre=pre, 
        learn=learn, dict_type="umap", 
        )

    for label in ["image", "orientation", "pred_step", "batch_num"]:
        palette = "viridis"

        unique = umap_df[label].unique()
        if len(unique) == 1:
            continue

        if label == "orientation" and "N/A" in unique:
            order = np.sort(
                unique[np.where(unique != "N/A")]
                ).tolist() + ["N/A"]
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
            alpha=0.3, palette=palette,
            facet_kws={"sharex": False, "sharey": False}
            )
        
        sns.despine(left=True, bottom=True)
        g.set(xticks=[], xticklabels=[], yticks=[], yticklabels=[])
        for m, hook_module in enumerate(hooks.HOOK_MODULES):
            for t, hook_type in enumerate(hooks.HOOK_TYPES):
                g.axes[t, m].set_title("")
                if t == 0:
                    g.axes[t, m].set_title(hook_module.capitalize())
                if m == 0:
                    g.axes[t, m].set_ylabel(hook_type.capitalize())

        savepath = Path(umap_dir, f"{epoch_str}_{pre_str}_{label}.svg")
        g.savefig(savepath, format="svg", bbox_inches="tight", dpi=600)


