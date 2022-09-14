#!/usr/bin/env python

import argparse
import copy
import glob
import logging
import math
from pathlib import Path
import sys

from matplotlib import pyplot as plt
from matplotlib import ticker
import numpy as np
import pandas as pd
import pickle as pkl
import scipy.stats

sys.path.extend(["..", str(Path("..", "utils")), str(Path("..", "analysis"))])
from analysis import analysis_utils, hooks
from utils import misc_utils, plot_utils


TAB = "    "

logger = logging.getLogger(__name__)

# plotting parameters
HDASH = (0, (2, 2))
NUM_ROIS = 256 * 4 * 4
LIGHT_ALPHA = 0.6
DARK_ALPHA = 0.8
DASH_ALPHA = 0.6
LIGHT, HEAVY = 2.5, 3.5

COLORS = {
    "encoder": "#26783C", # green
    "contextual": "#A22D8A", # pink
    "predictive": "#2E3392" # purple
}
DASHSTYLE = {
    "activations": None,
    "errors": (0, (2, 2))
}

RESP_EP_NS = [-1, 0, 1, 2]
DIFF_EP_NS = [0, 1, 2]
CORR_EP_NS = [(0, 1), (1, 2)]


PLT_PARAMS = {
    "axes.labelsize"       : "xx-large", # xx-large axis labels
    "axes.linewidth"       : 3,
    "axes.spines.right"    : False,      # no axis spine on right
    "axes.spines.top"      : False,      # no axis spine at top
    "axes.titlesize"       : "xx-large", # xx-large axis title
    "figure.autolayout"    : True,       # adjusts layout
    "figure.facecolor"     : "w",        # figure facecolor
    "font.size"            : 12,         # basic font size value
    "errorbar.capsize"     : 6, 
    "lines.dashed_pattern" : [8.0, 4.0], # longer dashes
    "lines.linewidth"      : 2.5,        # thicker lines
    "lines.markeredgewidth": 2.5,        # thick marker edge widths 
                                            # (e.g., cap thickness) 
    "lines.markersize"     : 10,         # bigger markers
    "patch.linewidth"      : 2.5,        # thicker lines for patches
    "savefig.format"       : "svg",      # figure save format
    "savefig.bbox"         : "tight",    # tight cropping of figure
    "savefig.transparent"  : False,      # background transparency
    "xtick.labelsize"      : "x-large",  # x-large x-tick labels
    "xtick.major.size"     : 7, 
    "xtick.major.width"    : 3.5, 
    "ytick.labelsize"      : "x-large",  # x-large y-tick labels
    "ytick.major.size"     : 7, 
    "ytick.major.width"    : 3.5,
}


#############################################
def np_pearson_r(x, y, axis=0, nanpol=None):
    """
    np_pearson_r(x, y)

    Returns Pearson R correlation for two matrices, along the specified axis.

    Required args
    -------------
    - x : nd array
        First array to correlate
    - y : nd array
        Second array to correlate (same shape as x)
    
    Optional args
    -------------
    - axis : int (default=0)
        Axis along which to correlate values
    - nanpol : bool (default=None) 
        NaN policy

    Returns
    -------
    - corr_bounded : nd array
        Correlation array, with one less dimension than x/y.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if nanpol not in [None, "omit"]:
        raise ValueError("'nanpol' must be None or 'omit'.")

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")

    mean_fct = np.mean if nanpol is None else np.nanmean
    sum_fct = np.sum if nanpol is None else np.nansum


    x_centered = x - mean_fct(x, axis=axis, keepdims=True)
    y_centered = y - mean_fct(y, axis=axis, keepdims=True)
    x_centered_ss = sum_fct(x_centered * x_centered, axis=axis)
    y_centered_ss = sum_fct(y_centered * y_centered, axis=axis)

    corr = (
        sum_fct(x_centered * y_centered, axis=axis) /
        np.sqrt(x_centered_ss * y_centered_ss)
        )
    # bound the values to -1 to 1 (for precision problems) -> propagate NaNs
    corr_bounded = np.maximum(np.minimum(corr, 1.0), -1.0)

    return corr_bounded


#############################################
def bootstrapped_corr(data1, data2, seed=100, n_samples=1000):
    """
    bootstrapped_corr(data1, data2)

    Returns bootstrapped correlation values.

    Required args
    -------------
    - data1 : array-like
        Array of data points.
    - data2 : array-like
        Array with the same length as data1.

    Optional args
    -------------
    - seed : int (default=100)
        Seed to use for resampling the data.
    - n_samples : int (default=1000)
        Number of data resamplings to perform.

    Returns
    -------
    - bootstrapped_std : float
        Bootstrapped standard deviation computed across correlations of 
        resampled data.
    """
    randst = np.random.RandomState(seed)

    data = np.stack([data1, data2])
    n = data.shape[1]
    choices = np.arange(n)

    rand_data1, rand_data2 = data[
        :, randst.choice(choices, (n, n_samples), replace=True)
        ]
    rand_corrs = np_pearson_r(rand_data1, rand_data2, axis=0)

    bootstrapped_std = np.std(rand_corrs)

    return bootstrapped_std


#############################################
def get_mean_sem_USI(sub_dict, df, common_dict, targ_classes, ns, 
                     pop_indiv=False):
    """
    get_mean_sem_USI(sub_dict, df, common_dict, targ_classes, ns)

    Populates the input dataframe with data statistics from the dictionary 
    passed. Also returns a dictionary with individual data for aggregating 
    across experiments.

    Required args
    -------------
    - sub_dict : dict
        Hook statistics across trials for the class ('mean' and 'std'), 
        with dims: unique classes x P x C x D_out x D_out
    - df : pd.DataFrame
        Dataframe to which to add statistics across nodes, U-D differences, 
        USIs and p-values. 
    - common_dict : dict
        Dictionary with keys and values that are common to all data to be 
        collected and should be entered into each new row of the dataframe.
    - targ_classes : list
        List of tuples, where each tuple specifies the class Gabor image and 
        orientation. The final classes have 'any' for one or both values.
    - ns : 2D array
        The number of occurrences of each class, with dims: class x pred_step

    Optional args
    -------------
    - pop_indiv : bool (default=False)
        If True, an dictionary with individual data per node is returned with 
        means for each Gabor image, as well as U-D values and USIs.

    Returns
    -------
    - df : pd.DataFrame
        Updated dataframe, with common key columns, as well as
        'image'      : type of Gabor image (or 'U-D', 'USI', 'USI_abs')
        'orientation': Gabor orientation
        'num_trials' : number of trials used for this data
        'mean'       : mean across nodes
        'std'        : standard deviation across nodes
        'sem'        : SEM across nodes
    - indiv_dict : dict
        Dictionary with data for individual nodes, if pop_indiv is True, 
        with keys:
        '{image}': node means for the specified Gabor image

    """
    
    indiv_dict = dict()
    for i, ((image, ori), n) in enumerate(zip(targ_classes, ns)):
        for suffix in ["", "_abs"]:
            idx = len(df)
            for k, v in common_dict.items():
                df.loc[idx, k] = v
            df.loc[idx, "image"] = f"{image}{suffix}"
            df.loc[idx, "orientation"] = ori
            df.loc[idx, "num_trials"] = sum(n)

            mean, sems, stds = np.nan, np.nan, np.nan
            if n > 0:
                data = sub_dict["means"][i]
                if "abs" in suffix:
                    data = np.absolute(data)
                mean = data.mean()
                sems = scipy.stats.sem(data.reshape(-1))
                stds = data.reshape(-1).std()

            df.loc[idx, "mean"] = mean
            df.loc[idx, "sem"] = sems
            df.loc[idx, "std"] = stds

        if ori == "any" and (pop_indiv or image in ["D", "U"]):
            indiv_dict[image] = sub_dict["means"][i].reshape(-1)
            if image in ["D", "U"]:
                indiv_dict[f"{image}_stds"] = sub_dict["stds"][i].reshape(-1)
    
    # add diffs and USIs
    if "D" in indiv_dict.keys() and "U" in indiv_dict.keys():
        for image in ["U-D", "USI"]:
            idx = len(df)
            for k, v in common_dict.items():
                df.loc[idx, k] = v
            df.loc[idx, "image"] = image
            df.loc[idx, "orientation"] = "any"
            diffs = indiv_dict["U"] - indiv_dict["D"]

            if image == "U-D":                            
                df.loc[idx, "mean"] = diffs.mean()
                df.loc[idx, "sem"] = scipy.stats.sem(diffs)
                df.loc[idx, "std"] = diffs.std()
                all_data = diffs

            elif image == "USI":
                stds = [indiv_dict.pop(f"{im}_stds") for im in ["U", "D"]]
                div = np.sqrt(0.5 * np.sum(np.power(stds, 2), axis=0))
                all_USIs = diffs / div
                df.loc[idx, "mean"] = all_USIs.mean()
                df.loc[idx, "sem"] = scipy.stats.sem(all_USIs)
                df.loc[idx, "std"] = all_USIs.std()
                all_data = all_USIs
            abs_data = np.absolute(all_data)

            # also add absolute values
            idx = len(df)
            for k, v in common_dict.items():
                df.loc[idx, k] = v
            df.loc[idx, "image"] = f"{image}_abs"
            df.loc[idx, "orientation"] = "any"                
            df.loc[idx, "mean"] = abs_data.mean()
            df.loc[idx, "sem"] = scipy.stats.sem(abs_data)
            df.loc[idx, "std"] = abs_data.std()

            if pop_indiv:
                indiv_dict[image] = all_data
    
    for key in list(indiv_dict.keys()):
        if "stds" in key:
            indiv_dict.pop(key)
    
    if not pop_indiv:
        indiv_dict = dict()

    return df, indiv_dict


#############################################
def get_n_comps(n_vals=3):
    """
    get_n_comps()
    """

    if n_vals < 1:
        raise ValueError("n_vals must be strictly positive.")

    inter = 0
    if n_vals > 1:
        inter = int(
            math.factorial(n_vals) / 
            (math.factorial(2) * math.factorial(n_vals - 2))
            )
    n_comps = int(n_vals + inter)

    return n_comps


#############################################
def add_epoch_data(hook_df, hook_dicts, pre_str, epoch_str, common_dict, 
                   hook_module="encoder", hook_type="activations", 
                   resp_ep_ns=RESP_EP_NS, diff_ep_ns=DIFF_EP_NS):
    """
    add_epoch_data(hook_dicts, pre_str, epoch_str, common_dict)

    Add epoch data to the dataframe.

    Required args
    -------------
    - hook_df : pd.DataFrame
        Dataframe to which to add statistics across nodes, U-D differences, 
        USIs and p-values. 
    - hook_dicts : list
    Hook dictionaries for different models, each with keys:
    '{epoch_str}' (dict): dictionary with key specifying the epoch number, 
                            with 'unexp' in it, if the epoch included 
                            unexpected events.
        '{pre_str}' (dict): dictionary with key specifying whether data 
                            was collected before ('pre') or after ('post') 
                            epoch training.
            'losses' (dict)      : loss statistics for each class 
                                    ('mean' and 'std'), with dims: 
                                    unique classes x P x D_out x D_out
            'targ_classes' (list): List of tuples, where each tuple 
                                    specifies the class Gabor image and 
                                    orientation. The final classes have 
                                    'any' for one or both values.
            'ns' (2D array)      : The number of occurrences of each class, 
                                    with dims: class x pred_step
            for hook_module in ["contextual", "encoder", "predictive"]
                for hook_type in ["activations", "errors"]
                    '{hook_module}''{hook_type}' (5D array): 
                        hook statistics across trials for each class 
                        ('mean' and 'std'), with dims: 
                        unique classes x P x C x D_out x D_out
    - epoch_str : str
        String providing epoch information, for saving.    
    - pre_str : str
        String providing pre/post and learning information, for saving.
    - common_dict : dict
        Dictionary with keys and values that are common to all data to be 
        collected and should be entered into each new row of the dataframe.
    
    Optional args
    -------------
    - hook_module : str (default="encoder")
        Module for which dataframe is being created. 
    - hook_type : str (default="activations")
        Hook type for which dataframe is being created.
    - resp_ep_ns : list
        Session numbers for which to compute and plot responses.
    - diff_ep_ns : list
        Session numbers for which to compute and plot U/D differences and USIs.

    Returns
    -------
    - common_dict : dict
        Dictionary with keys and values that are common to all data to be 
        collected and should be entered into each new row of the dataframe.
    - ep_dict : dict
        Dictionary with USI, USI absolute, and difference data, if applicable.
        The keys are 'diffs', 'USI', 'USI_abs', each containing a tuple with 
        the dataframe index and the data.
    """

    epoch_n = analysis_utils.get_digit_in_key(epoch_str)
    common_dict = copy.deepcopy(common_dict)

    full_dict = dict()
    unexps = []
    for h, hook_dict in enumerate(hook_dicts):
        sub_dict = hook_dict[pre_str][epoch_str][hook_module][hook_type]
        unexps.append("unexp" in epoch_str)
        common_dict["unexp"] = unexps[-1]
        common_dict["model_n"] = h

        hook_df, indiv_dict = get_mean_sem_USI(
            sub_dict, hook_df, common_dict,
            hook_dict[pre_str][epoch_str]["targ_classes"], 
            hook_dict[pre_str][epoch_str]["ns"],
            pop_indiv=(epoch_n in resp_ep_ns)
            )
        for key, data in indiv_dict.items():
            if key not in full_dict.keys():
                full_dict[key] = []
            full_dict[key].append(data)

    # get the overall stats
    common_dict["model_n"] = "all"
    common_dict["unexp"] = unexps[0] if len(set(unexps)) == 1 else None

    ep_dict = dict()
    for suffix in ["", "_abs"]:
        for key, data in full_dict.items():
            idx = len(hook_df)
            for k, v in common_dict.items():
                hook_df.loc[idx, k] = v
            hook_df.loc[idx, "image"] = f"{key}{suffix}"
            hook_df.loc[idx, "orientation"] = "any"
            data = np.concatenate(data)
            if "abs" in suffix:
                data = np.absolute(data)
            hook_df.loc[idx, "mean"] = data.mean()
            hook_df.loc[idx, "sem"] = scipy.stats.sem(data)
            hook_df.loc[idx, "std"] = data.std()

            if key in ["USI", "U-D"]:
                hook_df.loc[idx, "pval"] = scipy.stats.ttest_rel(
                    data, np.zeros_like(data)
                    )[1]
                if epoch_n in diff_ep_ns:
                    ep_dict[f"{key}{suffix}"] = (data, idx)
                del data
    
    return common_dict, ep_dict


#############################################
def get_hook_df(hook_dicts, resp_ep_ns=RESP_EP_NS, diff_ep_ns=DIFF_EP_NS, 
                corr_ep_ns=CORR_EP_NS):
    """
    get_hook_df(hook_dicts)
    
    Compiles a dataframe with hook data.

    Required args
    -------------
    - hook_dicts : list
        Hook dictionaries for different models, each with keys:
        '{epoch_str}' (dict): dictionary with key specifying the epoch number, 
                              with 'unexp' in it, if the epoch included 
                              unexpected events.
            '{pre_str}' (dict): dictionary with key specifying whether data 
                                was collected before ('pre') or after ('post') 
                                epoch training.
                'losses' (dict)      : loss statistics for each class 
                                       ('mean' and 'std'), with dims: 
                                       unique classes x P x D_out x D_out
                'targ_classes' (list): List of tuples, where each tuple 
                                       specifies the class Gabor image and 
                                       orientation. The final classes have 
                                       'any' for one or both values.
                'ns' (2D array)      : The number of occurrences of each class, 
                                       with dims: class x pred_step
                for hook_module in ["contextual", "encoder", "predictive"]
                    for hook_type in ["activations", "errors"]
                        '{hook_module}''{hook_type}' (5D array): 
                            hook statistics across trials for each class 
                            ('mean' and 'std'), with dims: 
                            unique classes x P x C x D_out x D_out
    
    Optional args
    -------------
    - resp_ep_ns : list
        Epoch numbers for which to compute and plot responses.
    - diff_ep_ns : list
        Epoch numbers for which to compute and plot U/D differences and USIs.
    - corr_ep_ns : list
        Tuples of epochs for which to compute USI correlations.

    Returns
    -------
    - hook_df : pd.DataFrame
        Dataframe with hook data, with columns:
        - hook_module     : Module from which the data is taken, 
                            e.g., 'encoder', 'contextual' or 'predictive'
        - hook_type       : Type of data, e.g., 'activations' or 'errors'
        - unexp           : Whether epoch(s) included unexpected data.
        - pre_post        : Whether data is 'pre' or 'post' learning for the epoch.
        - epoch_n         : Epoch number(s).
        - model_n         : Model number (index).
        - image           : Gabor image (or 'U-D', 'USI', 'USI_abs', 'USI_corr').
        - orientation     : Gabor mean orientation.
        - num_trials      : Number of trials underlying data.
        - mean            : Mean across nodes.
        - sem             : SEM across nodes.
        - std             : Standard deviation across nodes.
        - corr            : Correlation between USIs.
        - corr_std        : Bootstrapped standard deviation of USI correlation.
        - pval            : Paired t-test p-value
        - pval_ep{e1}v{e2}: Paired t-test p-value between epochs.
    """

    hook_df = pd.DataFrame(
        columns=[
            "hook_module", "hook_type", "unexp", "pre", "learn", "epoch_n", 
            "model_n", "image", "orientation", "num_trials", "mean", "sem", 
            "std", "corr", "corr_std", "pval",
            ]
        )
    
    if not isinstance(hook_dicts, list):
        hook_dicts = [hook_dicts]

    all_pre_strs = [sorted(hook_dict.keys()) for hook_dict in hook_dicts]
    for k, keys in enumerate(all_pre_strs[:-1]):
        if keys != all_pre_strs[k + 1]:
            raise ValueError("All hook_dicts must have the same pre/post keys.")
    pre_strs = all_pre_strs[0]
            
    all_epoch_strs = [
        sorted(pre_dict.keys()) for hook_dict in hook_dicts 
        for pre_dict in hook_dict.values() 
        ]
    for k, keys in enumerate(all_epoch_strs[:-1]):
        if keys != all_epoch_strs[k + 1]:
            raise ValueError("All hook_dicts must have the same epoch keys.")
    epoch_strs = all_epoch_strs[0]

    epoch_ns = [
        analysis_utils.get_digit_in_key(key) for key in all_epoch_strs[0]
        ]
    pre_strs = np.sort(np.unique(all_pre_strs[0]))

    seed = 100
    for hook_module in hooks.HOOK_MODULES:
        for hook_type in hooks.HOOK_TYPES:
            for pre_str in pre_strs:
                gen_dict = {
                    key: dict() for key in ["U-D", "USI", "U-D_abs", "USI_abs"]
                    }
                for e in np.argsort(epoch_ns):
                    epoch_n = str(epoch_ns[e])           
                    common_dict = {
                        "hook_module": hook_module,
                        "hook_type": hook_type,
                        "epoch_n": epoch_n,
                        "pre": "pre" in pre_str,
                        "learn": "learn" in pre_str,
                    }
                    common_dict, ep_dict = add_epoch_data(
                        hook_df, hook_dicts, pre_str, epoch_strs[e], 
                        common_dict=common_dict, hook_module=hook_module, 
                        hook_type=hook_type, resp_ep_ns=resp_ep_ns, 
                        diff_ep_ns=diff_ep_ns
                        )
                    for key, ep_val in ep_dict.items():
                        gen_dict[key][int(epoch_n)] = ep_val

                # get the between epoch comps
                for key, use_dict in gen_dict.items():
                    if len(use_dict) == 0:
                        continue

                    for e, e1 in enumerate(diff_ep_ns):
                        for e2 in diff_ep_ns[e + 1:]:
                            if int(e1) in use_dict.keys() and int(e2) in use_dict.keys():
                                data1, idx1 = use_dict[e1]
                                data2, idx2 = use_dict[e2]
                                p_val_diff = scipy.stats.ttest_rel(data1, data2)[1]
                                hook_df.loc[idx1, f"pval_ep{e1}v{e2}"] = p_val_diff
                                hook_df.loc[idx2, f"pval_ep{e1}v{e2}"] = p_val_diff

                    if key == "USI": # USI correlations
                        for e1, e2 in corr_ep_ns:
                            if e1 in use_dict.keys() and e2 in use_dict.keys():
                                data1, _ = use_dict[int(e1)]
                                data2, _ = use_dict[int(e2)]

                                idx = len(hook_df)
                                for k, v in common_dict.items():
                                    hook_df.loc[idx, k] = v
                                hook_df.loc[idx, "image"] = "USI_corr"
                                hook_df.loc[idx, "orientation"] = "any"
                                hook_df.loc[idx, "epoch_n"] = f"{e1}v{e2}"

                                corr, pval = scipy.stats.pearsonr(data1, data2)
                                corr_std = bootstrapped_corr(
                                    data1, data2, seed=seed
                                    )
                                
                                hook_df.loc[idx, "corr"] = corr
                                hook_df.loc[idx, "corr_std"] = corr_std
                                hook_df.loc[idx, "pval"] = pval

                                seed += 1

    hook_df["epoch_n"] = hook_df["epoch_n"].astype(str)

    return hook_df


#############################################
def format_pval(pval, n_comps=1):
    """
    format_pval(pval)

    
    """

    if not np.isfinite(pval):
        return "NaN"

    pval = min(1, pval * n_comps)
    star = ""
    if pval < 0.001:
        star = "***"
    elif pval < 0.01:
        star = "**"
    elif pval < 0.05:
        star = "*"
    
    return f"{pval:.4f}{star}"


#############################################
def plot_responses(hook_data_df, ep_ns=RESP_EP_NS, pre=True, learn=False, 
                   exp_only=True, absolute=False, output_dir="."):
    """
    plot_responses(hook_data_df)


    """
    
    ep_ns = [ep_ns] if not isinstance(ep_ns, list) else ep_ns

    ncols = len(hooks.HOOK_MODULES)
    n_types = len(hooks.HOOK_TYPES)
    nrows = n_types * len(ep_ns)
    fig, ax = plt.subplots(nrows, ncols, figsize=[3.33 * ncols, 1.75 * nrows])

    if len(ep_ns) > 1:
        for axis in ["x", "y"]:
            for i in range(ncols):
                plot_utils.set_shared_axes(ax[:len(ep_ns), i], axes=axis)
                plot_utils.set_shared_axes(ax[len(ep_ns):, i], axes=axis)

    # prepare x axis
    all_letters = ["A", "B", "C", "D", "U", "G"]
    all_xs = [0, 1, 2, 2.65, 3.25, 4]
    xtick_letters = ["A", "B", "C", "D/U", "G"]
    x_ticks = np.arange(len(xtick_letters))
    half_span = (all_xs[4] - all_xs[3]) / 2
    start, stop = all_xs[4] - half_span, all_xs[4] + half_span

    suffix = "_abs" if absolute else ""

    for n, hook_module in enumerate(hooks.HOOK_MODULES):
        for t, hook_type in enumerate(hooks.HOOK_TYPES):
            for i, ep_n in enumerate(ep_ns):
                data = hook_data_df.loc[
                    (hook_data_df["orientation"] == "any") &
                    (hook_data_df["hook_module"] == hook_module) &
                    (hook_data_df["hook_type"] == hook_type) &
                    (hook_data_df["pre"] == pre) &
                    (hook_data_df["learn"] == learn) &
                    (hook_data_df["epoch_n"] == str(ep_n))
                ]

                if len(data) == 0:
                    raise RuntimeError("No matching data found.")

                row = t * len(ep_ns) + i
                ax[row, n].axhline(
                    0, ls=HDASH, alpha=DASH_ALPHA, color="k", zorder=-5
                    )

                for model_n in data["model_n"].unique():
                    model_data = data.loc[data["model_n"] == model_n]
                    lines = [
                        model_data.loc[model_data["image"] == f"{letter}{suffix}"] 
                        for letter in all_letters
                        ]
                    means, sems = [], []
                    for line in lines:
                        if len(line) == 0:
                            continue
                        elif len(line) != 1:
                            raise ValueError("Expected exactly one line.")
                        means.append(line["mean"].tolist()[0])
                        sems.append(line["sem"].tolist()[0])

                    xs = all_xs
                    if ep_n < 0:
                        xs = all_xs[:4] + [all_xs[-1]]

                    alpha = DARK_ALPHA if model_n == "all" else LIGHT_ALPHA / 2
                    lw = HEAVY if model_n == "all" else LIGHT
                    zorder = 3 if model_n == "all" else 1
                    marker = "." if model_n == "all" else None
                    
                    ax[row, n].plot(
                        xs, means, alpha=alpha, lw=lw, color=COLORS[hook_module], 
                        ls=DASHSTYLE[hook_type], zorder=zorder, marker=marker, 
                        ms=12
                        )
                    low = np.asarray(means) - np.asarray(sems)
                    high = np.asarray(means) + np.asarray(sems)
                    ax[row, n].fill_between(
                        xs, y1=low, y2=high, alpha=alpha / 2, 
                        color=COLORS[hook_module], lw=0
                        )

                if ep_n >= 0:
                    ax[row, n].axvspan(
                        start, stop, color="k", alpha=0.075, lw=0, zorder=-10
                        )
                ax[row, n].set_xticks(x_ticks)
                ax[row, n].set_xticklabels(xtick_letters)

    for i in range(ncols):
        for s, subax in enumerate([ax[0, i], ax[len(ep_ns), i]]):
            if s == 0:
                subax.yaxis.set_major_formatter(
                    ticker.FormatStrFormatter("%0.2f")
                )
            subax.yaxis.set_major_locator(ticker.MaxNLocator(2)) 
            plot_utils.expand_axis(subax, axis="x", pad_each=0.1)
            plot_utils.expand_axis(subax, axis="y", pad_each=0.04)
        for s, subax in enumerate(ax[:, i]):
            if s != len(ep_ns) - 1 and s != len(ep_ns) * 2 - 1:
                subax.xaxis.set_visible(False)
                subax.spines.bottom.set_visible(False)
        if exp_only: # force exponential
            for subax in ax[len(ep_ns) : , i]:
                subax.yaxis.get_major_formatter().set_powerlimits((0, 0))

    if output_dir is not None:
        _, pre_str = analysis_utils.get_epoch_pre_str(pre=pre, learn=learn)
        savepath = Path(output_dir, f"{pre_str}{suffix}_resp.svg")
        savepath.parent.mkdir(exist_ok=True, parents=True)
        fig.savefig(savepath, format="svg", bbox_inches="tight", dpi=300)

    return fig


#############################################
def plot_acr_epochs(hook_data_df, ep_ns=DIFF_EP_NS, pre=True, learn=False, 
                    image="U-D", log_stats=True, output_dir="."):
    """
    plot_acr_epochs(hook_data_df)


    """

    ep_ns = [ep_ns] if not isinstance(ep_ns, list) else ep_ns

    n_comps_per = get_n_comps(n_vals=len(ep_ns))
    if "abs" in image:
        n_comps_per = n_comps_per - len(ep_ns) # only compare between epochs
    n_comps = (n_comps_per * len(hooks.HOOK_TYPES) * len(hooks.HOOK_MODULES))

    if image not in hook_data_df["image"].tolist():
        raise ValueError(f"{image} not recognized.")
    
    nrows = len(hooks.HOOK_TYPES)
    ncols = len(hooks.HOOK_MODULES)

    col_wid = 1.9 if "abs" in image else 2.67 
    row_hei = 3 if "abs" in image else 4 
    fig, ax = plt.subplots(
        nrows, ncols, figsize=[ncols * col_wid, nrows * row_hei], sharex="col"
        )
    
    for ep_n in ep_ns:
        if "U" in image and int(ep_n) < 0:
            raise ValueError("ep_ns values must be greater or equal to 0.")

    log_str = f"{image}: Bonferroni corrected p-values:"
    for n, hook_module in enumerate(hooks.HOOK_MODULES):
        log_str = f"{log_str}\n{TAB}{hook_module.capitalize()}:"
        for t, hook_type in enumerate(hooks.HOOK_TYPES):
            log_str = f"{log_str}\n{TAB}{TAB}{hook_type.capitalize()}:"
            ax[t, n].axhline(
                0, ls=HDASH, alpha=DASH_ALPHA, color="k", zorder=-5
                )
            data = hook_data_df.loc[
                    (hook_data_df["orientation"] == "any") &
                    (hook_data_df["hook_module"] == hook_module) &
                    (hook_data_df["hook_type"] == hook_type) &
                    (hook_data_df["pre"] == pre) &
                    (hook_data_df["learn"] == learn)
                ]

            if len(data) == 0:
                raise RuntimeError("No matching data found.")

            for model_n in data["model_n"].unique():
                model_data = data.loc[data["model_n"] == model_n]
                ep_diffs = [
                    model_data.loc[
                        (model_data["image"] == image) & 
                        (model_data["epoch_n"] == str(ep_n))
                    ] for ep_n in ep_ns
                ]
                means, sems = [], []
                for line in ep_diffs:
                    if len(line) != 1:
                        raise ValueError("Expected exactly one line.")
                    means.append(line["mean"].tolist()[0])
                    sems.append(line["sem"].tolist()[0])

                alpha = DARK_ALPHA if model_n == "all" else LIGHT_ALPHA / 4
                lw = HEAVY if model_n == "all" else LIGHT
                zorder = 3 if model_n == "all" else 1
                marker = "." if model_n == "all" else None

                ax[t, n].errorbar(
                    np.arange(1, len(ep_ns) + 1), means, yerr=sems, 
                    alpha=alpha, lw=lw, color=COLORS[hook_module], 
                    ls=DASHSTYLE[hook_type], marker=marker, markersize=16, 
                    capsize=4, zorder=zorder
                )

                if model_n == "all":
                    full_tab = f"{TAB}{TAB}{TAB}"
                    if "abs" not in image:
                        for s, ep_n in enumerate(ep_ns):
                            pval = ep_diffs[s]["pval"].tolist()[0]
                            pval = format_pval(pval, n_comps=n_comps)
                            log_str = f"{log_str}\n{full_tab}ep {ep_n}: {pval}"
                    for i, s1 in enumerate(ep_ns):
                        for s2 in ep_ns[i + 1:]:
                            pval = ep_diffs[s1][f"pval_ep{s1}v{s2}"].tolist()[0]
                            pval = format_pval(pval, n_comps=n_comps)
                            log_str = \
                                f"{log_str}\n{full_tab}ep{s1}v{s2}: {pval}"

            plot_utils.expand_axis(ax[t, n], axis="y", pad_each=0.1)

    for i in range(ncols):
        plot_utils.expand_axis(ax[0, i], axis="x", pad_each=0.2)
        for j in range(2):
            ytick_max = ax[j, i].get_yticks()[0] 
            if ytick_max != 0 and ytick_max < 0.001: # force exponential
                ax[j, i].yaxis.get_major_formatter().set_powerlimits((0, 0))
            ax[j, i].yaxis.set_major_locator(ticker.MaxNLocator(3))
            if j == 0:
                ax[j, i].xaxis.set_visible(False)
                ax[j, i].spines.bottom.set_visible(False)

    if log_stats:
        logger.info(log_str)

    if output_dir is not None:
        _, pre_str = analysis_utils.get_epoch_pre_str(pre=pre, learn=learn)
        savepath = Path(output_dir, f"{pre_str}_{image}.svg")
        savepath.parent.mkdir(exist_ok=True, parents=True)
        fig.savefig(savepath, format="svg", bbox_inches="tight", dpi=300)

    return fig


#############################################
def get_corr_df(hook_data_df, ep_ns=CORR_EP_NS, pre=True, learn=False, 
                log_stats=True, output_dir="."):
    """
    get_corr_df(hook_data_df)


    """
    
    ep_ns = [ep_ns] if not isinstance(ep_ns, list) else ep_ns
    n_comps = (len(ep_ns) * len(hooks.HOOK_TYPES) * len(hooks.HOOK_MODULES))

    ep_col = "unexp. epochs"
    cols = ["hook type"] + hooks.HOOK_MODULES + [ep_col]
    corr_df = pd.DataFrame(columns=cols)

    for s1, s2 in ep_ns:
        if int(s1) < 0 or int(s2) < 0:
            raise ValueError("ep_ns values must be greater or equal to 0.")
    ep_ns_for_df = [f"{s1} vs {s2}" for s1, s2 in ep_ns]
    ep_ns = [f"{s1}v{s2}" for s1, s2 in ep_ns]

    log_str = "Correlations: Bonferroni corrected p-values:"
    for hook_module in hooks.HOOK_MODULES:
        log_str = f"{log_str}\n{TAB}{hook_module.capitalize()}:"
        for hook_type in hooks.HOOK_TYPES:
            log_str = f"{log_str}\n{TAB}{TAB}{hook_type.capitalize()}:"
            data = hook_data_df.loc[
                    (hook_data_df["orientation"] == "any") &
                    (hook_data_df["hook_module"] == hook_module) &
                    (hook_data_df["hook_type"] == hook_type) &
                    (hook_data_df["pre"] == pre) &
                    (hook_data_df["learn"] == learn) &
                    (hook_data_df["image"] == "USI_corr") &
                    (hook_data_df["model_n"] == "all")
                ]

            if len(data) == 0:
                raise RuntimeError("No matching data found.")

            ep_diffs = [
                data.loc[data["epoch_n"] == str(ep_n)] for ep_n in ep_ns
                ]
            corrs, stds = [], []
            for line in ep_diffs:
                if len(line) != 1:
                    raise ValueError("Should be exactly one line per line.")
                corrs.append(line["corr"].tolist()[0])
                stds.append(line["corr_std"].tolist()[0])

            for s, ep_n in enumerate(ep_ns):
                pval = ep_diffs[s]["pval"].tolist()[0] / 2
                corr_val = ep_diffs[s]["corr"].tolist()[0]
                if corr_val > 0:
                    pval = 1 - pval
                pval = format_pval(pval, n_comps=n_comps)
                star = ""
                if "*" in pval:
                    star = "".join([s for s in pval if s == "*"])
                
                log_str = f"{log_str}\n{TAB}{TAB}{TAB}ep {ep_n}: {pval}"
                
                lines = corr_df.loc[
                    (corr_df["hook type"] == hook_type) &
                    (corr_df[ep_col] == ep_ns_for_df[s])
                ]
                if len(lines) == 1:
                    corr_idx = lines.index[0]
                else:
                    corr_idx = len(corr_df)
                
                corr_df.loc[corr_idx, "hook type"] = hook_type
                corr_df.loc[corr_idx, ep_col] = ep_ns_for_df[s]
                corr_df.loc[corr_idx, hook_module] = f"{corr_val:.4f}{star}"
                

    if log_stats:
        logger.info(log_str)

    if output_dir is not None:
        _, pre_str = analysis_utils.get_epoch_pre_str(pre=pre, learn=learn)
        savepath = Path(output_dir, f"{pre_str}_corr_df.csv")
        savepath.parent.mkdir(exist_ok=True, parents=True)
        corr_df.to_csv(savepath)

    return corr_df


#############################################
def check_plotting(hook_df, resp_ep_ns=RESP_EP_NS, diff_ep_ns=DIFF_EP_NS, 
                   corr_ep_ns=CORR_EP_NS, pre=True, learn=False, 
                   raise_err=True):
    """
    check_plotting(hook_df)


    """

    corr_ep_ns = [int(i) for vals in corr_ep_ns for i in vals]
    ep_ns = set(resp_ep_ns + diff_ep_ns + corr_ep_ns)
    
    hook_df_sub = hook_df.loc[
        (hook_df["pre"] == pre) & (hook_df["learn"] == learn)
        ]

    plot_err = ""
    if not ep_ns.issubset(hook_df_sub["epoch_n"].unique()):
        plot_err = (
            "USI/diffs data cannot be plotted, as some epochs are missing "
            f"for pre={pre} and learn={learn}."
        )
    
        if raise_err:
            raise RuntimeError(plot_err)

    return plot_err


#############################################
def plot_all(hook_df, output_dir=None, resp_ep_ns=RESP_EP_NS, 
             diff_ep_ns=DIFF_EP_NS, corr_ep_ns=CORR_EP_NS, pre=True, 
             learn=False, log_stats=True):
    """
    plot_all(hook_df)


    """

    with plt.rc_context(PLT_PARAMS):
        for absolute in [False, True]:
            plot_responses(hook_df, ep_ns=resp_ep_ns, pre=pre, learn=learn, 
                        exp_only=True, output_dir=output_dir, absolute=absolute)

            suffix = "_abs" if absolute else ""
            for image in ["U-D", "USI"]:
                plot_acr_epochs(
                    hook_df, ep_ns=diff_ep_ns, pre=pre, learn=learn, 
                    image=f"{image}{suffix}", output_dir=output_dir, 
                    log_stats=log_stats
                    )

        get_corr_df(
            hook_df, ep_ns=corr_ep_ns, pre=pre, learn=learn, 
            output_dir=output_dir, log_stats=log_stats
            )


#############################################
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", default="output_dir", 
        help="directory in which to find hook pickles for replotting")
    parser.add_argument("--post", action="store_true", 
        help="plot post instead of pre data")
    parser.add_argument("--learn", action="store_true", 
        help="plot learning data")
    parser.add_argument('--log_level', default='info', 
                        help='logging level, e.g., debug, info, error')
    args = parser.parse_args()

    misc_utils.get_logger_with_basic_format(level=args.log_level)

    hook_pkls = glob.glob(
        str(Path(args.output_dir, "**", "*hook*.pkl")), recursive=True
        )
    
    logger.info(f"Compiling data from {hook_pkls} hook pickles.")
    
    hook_dicts = []
    for hook_pkl in hook_pkls:
        with open(hook_pkl, "rb") as f:
            hook_dicts.append(pkl.load(f))

    hook_df = get_hook_df(hook_dicts)

    plot_all(
        hook_df, output_dir=args.output_dir, resp_ep_ns=RESP_EP_NS, 
        diff_ep_ns=DIFF_EP_NS, corr_ep_ns=CORR_EP_NS, pre=not(args.post), 
        learn=args.learn, log_stats=True
        )

