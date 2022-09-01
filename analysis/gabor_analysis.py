import copy
import logging
from pathlib import Path
import re
import time

import h5py
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pickle as pkl
import scipy.stats
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import torch
import umap.umap_ as umap

from dataset import gabor_sequences
from utils import loss_utils, misc_utils, plot_utils, training_utils
from analysis import hooks

RSM_DIR = "rsms"
UMAP_DIR = "umaps"
HOOK_DIR = "hooks"
NUM_STEPS = 5

TAB = "    "

logger = logging.getLogger(__name__)


#############################################
def get_dicts(output_dir="."):
    """
    get_dicts()
    """

    hook_save_path = Path(output_dir, HOOK_DIR, "hook_data.pkl")
    umap_save_path = Path(output_dir, UMAP_DIR, "umap_data.pkl")

    data_dicts = []
    for save_path in [hook_save_path, umap_save_path]:
        save_path.parent.mkdir(parents=True, exist_ok=True)

        data_dict = dict()
        if save_path.is_file():
            with open(save_path, "rb") as f:
                data_dict = pkl.load(f)
        data_dicts.append(data_dict)
    
    rsm_save_path = dict()
    data_dicts.append(rsm_save_path)

    return data_dicts


#############################################
def load_rsm_dict(save_path):
    """
    load_rsm_dict(save_path)
    """

    if not Path(save_path).is_file():
        raise OSError(f"{save_path} is not a file.")

    data_dict = dict()
    with h5py.File(save_path, "r") as f:
        for key1 in f.keys():
            data_dict[key1] = dict()
            # rsm_vals, idxs_im/{}, idxs_ori/{}, {hook_names}/{hook_types} 
            for key2 in f[key1].keys(): 
                if key2 == "rsm_vals":
                    data_dict[key1][key2] = f[key1][key2][()]
                    continue
                data_dict[key1][key2] = dict()
                for key3 in f[key1][key2].keys():
                    data_dict[key1][key2][key3] = f[key1][key2][key3][()]
    
    return data_dict


#############################################
def save_rsm_h5(save_dict, output_dir="."):
    """
    save_rsm_h5(save_dict)
    """

    save_path = Path(output_dir, RSM_DIR, "rsm_data.h5")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    comp_kwargs = {"compression": "gzip", "chunks": True}
    with h5py.File(save_path, "a") as f:
        for key1, sub1 in save_dict.items():
            # rsm_vals, idxs_im/{}, idxs_ori/{}, {hook_names}/{hook_types} 
            for key2, sub2 in sub1.items(): 
                if key2 == "rsm_vals":
                    f.create_dataset(f"{key1}/{key2}", data=sub2)
                    continue
                for key3, sub3 in sub2.items():
                    kwargs = dict()
                    if "idxs" not in key2:
                        kwargs = comp_kwargs
                    f.create_dataset(
                        f"{key1}/{key2}/{key3}", data=sub3, **kwargs
                        )
    
    return save_path


#############################################
def save_dict(save_dict, output_dir=".", dict_type="hook_data"):
    """
    save_dict(save_dict)
    """

    if dict_type == "hook_data":
        save_path = Path(output_dir, HOOK_DIR, "hook_data.pkl")
    elif dict_type == "umap_data":
        save_path = Path(output_dir, UMAP_DIR, "umap_data.pkl")
    elif dict_type != "rsm_data":
        raise ValueError(f"{dict_type} not recognized.")
    
    if dict_type == "rsm_data": # append to file
        save_path = save_rsm_h5(save_dict, output_dir)
    else:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pkl.dump(save_dict, f)
    
    return save_path.parent


#############################################
def run_checks(model, dataloader):
    """
    run_checks(model, dataloader)
    """

    _, supervised = training_utils.get_num_classes_sup(model)
    is_gabor = gabor_sequences.check_if_is_gabor(dataloader.dataset)

    if not is_gabor:
        raise ValueError("Analysis applies only to Gabor datasets.")

    if supervised:
        raise NotImplementedError(
            "Analysis is not implemented for supervised models."
            )
    
    if dataloader.dataset.shift_frames:
        raise NotImplementedError(
            "Analysis requires dataset shift_frames to be set to False."
            )
        
    if dataloader.dataset.gab_img_len != dataloader.dataset.seq_len:
        raise NotImplementedError(
            "Analysis requires dataset gab_img_len to be equal to seq_len."
        )


#############################################
def get_umap_df(data, labels, name="encoder", hook_type="activations", 
                num_batches=10, pred_step=1):
    """
    get_umap_df(data, labels)
    """

    pip = Pipeline([
        ('scaler', StandardScaler()), 
        ('umap', umap.UMAP(min_dist=0.4))
        ])
    
    proj = pip.fit_transform(data)

    df = pd.DataFrame()
    df["dim0"], df["dim1"] = proj.T
    df["image"], df["orientation"] = list(zip(*labels))

    df["orientation"] = [
        int(val) if val != "N/A" and int(val) == val else val 
        for val in df["orientation"]
        ]

    df["name"] = name
    df["hook_type"] = hook_type

    batch_size = len(data) // (num_batches * pred_step)
    if num_batches * batch_size * pred_step != len(data):
        raise ValueError("'num_batches' or 'pred_step' is wrong.")
    
    pred_step_arr = np.empty([num_batches, batch_size, pred_step])
    for i in range(pred_step):
        pred_step_arr[..., i] = i
    df["pred_step"] = pred_step_arr.reshape(-1)

    batch_nums = np.empty([num_batches, batch_size, pred_step])
    for i in range(num_batches):
        batch_nums[i] = i
    df["batch_num"] = batch_nums.reshape(-1)

    return df


#############################################
def plot_save_umaps(umap_dict, ep_suffix, output_dir=".", plot_batch_num=False):
    """
    plot_save_umaps(umap_dict, ep_suffix)
    """

    umap_dir = save_dict(umap_dict, output_dir, dict_type="umap_data")
    df = umap_dict[ep_suffix]["umap_df"]

    for label in ["image", "orientation", "pred_step", "batch_num"]:
        palette = "viridis"

        unique = df[label].unique()
        if len(unique) == 1:
            continue

        if label == "orientation" and "N/A" in unique:
            order = np.sort(
                unique[np.where(unique != "N/A")]
                ).tolist() + ["N/A"]
        elif label == "batch_num" and not plot_batch_num:
            continue
        else:
            order = sorted(unique)
            if label == "image" and "D" in unique and "U" not in order:
                order = list(order) + ["U"]
            if label == "image" and len(order) > 3 and "U" in order:
                palette = sns.color_palette("viridis", len(order) - 1)
                palette.append(sns.color_palette("tab10")[3])
        
        g = sns.relplot(
            data=df, x="dim0", y="dim1", 
            hue=label, hue_order=order, 
            col="name", col_order=hooks.HOOK_NAMES,
            row="hook_type", row_order=hooks.HOOK_TYPES,
            alpha=0.3, palette=palette,
            facet_kws={"sharex": False, "sharey": False}
            )
        
        sns.despine(left=True, bottom=True)
        g.set(xticks=[], xticklabels=[], yticks=[], yticklabels=[])
        for n, name in enumerate(hooks.HOOK_NAMES):
            for t, hook_type in enumerate(hooks.HOOK_TYPES):
                g.axes[t, n].set_title("")
                if t == 0:
                    g.axes[t, n].set_title(name.capitalize())
                if n == 0:
                    g.axes[t, n].set_ylabel(hook_type.capitalize())

        savepath = Path(umap_dir, f"{ep_suffix}_{label}.svg")
        g.savefig(savepath, format="svg", bbox_inches="tight", dpi=600)


#############################################
def get_rsm_idxs(images, oris):
    """
    get_rsm_idxs(images, oris)
    """

    if len(images) != len(oris):
        raise ValueError("'images' and 'oris' should have the same length.")

    add_na = "N/A" in set(oris)
    unique_oris = [ori for ori in list(set(oris)) if ori != "N/A"]
    ori_sort = np.argsort(np.asarray(unique_oris).astype(float))
    unique_oris = [unique_oris[i] for i in ori_sort]
    if add_na:
        unique_oris.append("N/A")

    unique_images = list(sorted(set(images)))
    if "G" in unique_images:
        unique_images.remove("G")
        unique_images.append("G")

    all_full_idxs = []
    keys = ["sorter", "inner_labels", "outer_labels", "inner_ends", "outer_ends"]
    uniques = [unique_images, unique_oris]
    vals = [images, oris]
    for u, (outer_unique, outer_labels) in enumerate(zip(uniques, vals)):
        inner_unique = uniques[1 - u]
        inner_labels = vals[1 - u]
        full_idxs = {k: [] for k in keys}
        i, j = 0, 0
        for outer_label in outer_unique:
            outer_idxs = np.where(
                np.asarray(outer_labels).astype(str) == str(outer_label)
                )[0]
            if len(outer_idxs) == 0:
                continue
            i += len(outer_idxs)
            full_idxs["outer_labels"].append(outer_label)
            full_idxs["outer_ends"].append(i)
            for inner_label in inner_unique:
                inner_idxs = np.where(
                    np.asarray(inner_labels)[outer_idxs].astype(str) == str(inner_label)
                    )[0]
                if len(inner_idxs) == 0:
                    continue
                full_idxs["sorter"].extend(outer_idxs[inner_idxs].tolist())
                j += len(inner_idxs)
                full_idxs["inner_labels"].append(inner_label)
                full_idxs["inner_ends"].append(j)
        full_idxs["sorter"] = np.asarray(full_idxs["sorter"])
        for key, val in full_idxs.items():
            if "labels" in key:
                full_idxs[key] = np.asarray(val).astype("S10")
            elif "ends" in key:
                full_idxs[key] = np.asarray(val).astype(int)
        all_full_idxs.append(full_idxs)

    full_idxs_im, full_idxs_ori = all_full_idxs

    return full_idxs_im, full_idxs_ori


#############################################
def calculate_rsm_corr(features):
    """
    calculate_rsm_corr(features)

    Calculates representational similarity matrix (RSM) between for a feature 
    matrix using pairwise cosine similarity after data is centered. This 
    calculation is equivalent to pairwise Pearson correlations.

    Adapted from https://github.com/rsagroup/rsatoolbox/blob/main/rsatoolbox/rdm/calc.py

    Required args:
    - features (2D np array): feature matrix (items x features)

    Returns:
    - rsm (2D np array): similarity matrix 
        (nbr features items x nbr features items)
    """

    features = features.reshape(len(features), -1) # flatten
    
    features = features - np.mean(features, axis=1, keepdims=True)

    features /= np.sqrt(np.einsum("ij,ij->i", features, features))[:, None]

    rsm = np.einsum("ik,jk", features, features)

    if (rsm.max() > 1 or rsm.min() < -1):
        if rsm.max() < 1.01 and rsm.min() > -1.01:
            rsm = np.maximum(np.minimum(rsm, 1), -1)
        else:
            raise RuntimeError("'rsm' shouldn't go beneath -1 or above 1")

    return rsm


############################################
def mark_rsm_axes(ax, idx_dict):
    """
    mark_rsm_axes(ax, idx_dict)
    """

    for r, ax_row in enumerate(ax):
        for c, subax in enumerate(ax_row):
            for tick_type in ["outer", "inner"]:
                u = 0
                edge_marks = [0]
                label_dict = dict()
                labels = idx_dict[f"{tick_type}_labels"].astype(str)
                ends = idx_dict[f"{tick_type}_ends"].astype(int)
                for label, end in zip(labels, ends):
                    if "." in str(label) and float(label) == int(float(label)):
                        label = int(float(label))
                    label_dict[u + (end - u) / 2] = label
                    edge_marks.append(end)
                    u = end
                
                plot_utils.add_sets_of_ticks(
                    subax, label_dict, 
                    secondary=(tick_type=="inner"), 
                    remove_primary_ticks=True, 
                    rotate_secondary_x=True, 
                    rotate_primary_x=False, 
                    top=(r == 0), 
                    right=(c == len(ax_row) - 1)
                    )

                if tick_type == "outer" and len(edge_marks) > 2:
                    for edge in edge_marks[1:-1]:
                        subax.axhline(edge + 0.5, lw=1, color="k")
                        subax.axvline(edge + 0.5, lw=1, color="k")


#############################################
def expand_for_rare(rsm, idx_dict):
    """
    expand_for_rare(rsm, idx_dict)
    """

    u = 0
    lengths = []
    for end in idx_dict["inner_ends"]:
        lengths.append(end - u)
        u = end

    lengths = np.asarray(lengths)
    min_len = int(np.around(np.median(lengths)))
    if 0 in lengths:
        raise ValueError("No label should be for 0 items.")

    factors = np.asarray([
        1 if leng >= min_len else int(np.around(min_len / leng)) 
        for leng in lengths
        ])
    new_lengths = lengths * factors

    if (lengths == new_lengths).all():
        return rsm, idx_dict

    # update dictionary
    updates = []
    u, base_add = 0, 0
    new_idx_dict = copy.deepcopy(idx_dict)
    sorter = []
    for l, end in enumerate(new_idx_dict["inner_ends"]):
        diff = new_lengths[l] - lengths[l]
        if factors[l] == 1:
            sorter.extend(range(u, end))
        else:
            sorter.extend(np.repeat(np.arange(u, end), factors[l]).tolist())
        u = end

        base_add += diff
        new_end = end + base_add
        updates.append((end, new_end))
        new_idx_dict["inner_ends"][l] = new_end

    outer_ends = np.asarray(new_idx_dict["outer_ends"])
    for old, new in updates[::-1]:
        if old in outer_ends:
            l = outer_ends.tolist().index(old)
            new_idx_dict["outer_ends"][l] = new

    # RSMs
    rsm = rsm[sorter][:, sorter]

    return rsm, new_idx_dict


#############################################
def plot_save_rsms(rsm_dict, ep_suffix, output_dir="."):
    """
    plot_save_rsms(rsm_dict, ep_suffix)

    Plots representational similarity matrices.
    """

    rsm_dir = save_dict(rsm_dict, output_dir, dict_type="rsm_data")
    ep_rsm_dict = rsm_dict[ep_suffix]

    idxs_im = ep_rsm_dict["idxs_im"]
    idxs_ori = ep_rsm_dict["idxs_ori"]
    rsm_vals = ep_rsm_dict["rsm_vals"]
    rsm_vals = np.linspace(rsm_vals[0], rsm_vals[1], int(rsm_vals[2]))

    nrows, ncols = len(hooks.HOOK_TYPES), len(hooks.HOOK_NAMES)

    labels = ["image", "orientation"]
    for i, idx_dict in enumerate([idxs_im, idxs_ori]):
        fig, ax = plt.subplots(
            nrows, ncols, figsize=(ncols * 6, nrows * 6), 
            sharex=True, sharey=True, 
            squeeze=False,
            gridspec_kw={"wspace": 0.005, "hspace": 0.17}
            )

        fig.suptitle("Representational Similarity Matrices (RSMs)", y=0.96)

        cm_w = 0.03 / ncols
        cbar_ax = fig.add_axes([0.93, 0.11, cm_w, 0.78])

        sorter = idx_dict["sorter"]
        upper_idxs = np.triu_indices(len(sorter), k=0)
        lower_idxs = (upper_idxs[1], upper_idxs[0])
        rasterize_ax = []
        for n, name in enumerate(hooks.HOOK_NAMES):
            for t, hook_type in enumerate(hooks.HOOK_TYPES):
                rsm = np.empty((len(sorter), len(sorter)))
                rsm_binned = ep_rsm_dict[name][hook_type]
                rsm[upper_idxs] = rsm_binned
                rsm[lower_idxs] = rsm_binned
                rsm = rsm_vals[rsm.astype(int)]
                rsm = rsm[sorter][:, sorter]
                rsm, use_idx_dict = expand_for_rare(rsm, idx_dict)
                im = ax[t, n].imshow(
                    rsm, vmin=-1, vmax=1, interpolation="none", zorder=-10
                    )
                if t == 0:
                    ax[t, n].set_title(name.capitalize(), y=1.06)
                if n == 0:
                    ax[t, n].set_ylabel(
                        hook_type.capitalize(), fontsize="large"
                        )
                mark_rsm_axes(ax, use_idx_dict)
                rasterize_ax.append(ax[t, n])

        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label(label="Similarity", size="large")
        cbar_ax.yaxis.set_label_position("left")

        for subax in rasterize_ax:
            subax.set_rasterization_zorder(-9)

        savepath = Path(rsm_dir, f"{ep_suffix}_{labels[i]}.svg")
        fig.savefig(savepath, format="svg", bbox_inches="tight", dpi=300)

    return


#############################################
def get_idxs(targets, dataset):
    """
    get_idxs(targets, dataset)
    """

    factor = 10 # larger than the binary unexpected value

    targets_mod = targets[..., 0] * factor + targets[..., 1]
    num_pred = targets_mod.shape[1]
    unique = np.sort(np.unique(targets_mod))

    all_idxs = []
    ns = np.zeros((len(unique), num_pred))
    for t, target in enumerate(unique):
        all_idxs.append([])
        for p in range(num_pred):
            idxs = np.where(targets_mod[:, p] == target)[0]
            all_idxs[-1].append(idxs)
            ns[t, p] = len(idxs)
    
    labels = [val // factor for val in unique]
    unexps = [val - label * factor for val, label in zip(unique, labels)]
    seq_classes = dataset.image_label_to_class(labels, unexps)

    class_images, class_oris = zip(*seq_classes)

    # collate for images/oris
    more_ns = [[] for _ in range(num_pred)]
    for val in set(class_images + class_oris):
        if val in class_images:
            unique_idxs = [i for i, v in enumerate(class_images) if v == val]
            seq_classes.append((val, "any"))
        else:
            unique_idxs = [i for i, v in enumerate(class_oris) if v == val]
            seq_classes.append(("any", val))

        all_idxs.append([])
        for p in range(num_pred):
            idxs = [all_idxs[i][p] for i in unique_idxs]
            idxs = np.sort(np.concatenate(idxs))
            all_idxs[-1].append(idxs)
            more_ns[p].append(len(idxs))
    
    more_ns = np.asarray(more_ns).T
    ns = np.concatenate([ns, more_ns], axis=0)

    return all_idxs, ns, seq_classes


#############################################
def compile_data(ep_hook_dict, dataset):
    """
    compile_data(ep_hook_dict, dataset)

    Aggregate across batches and target classes.
    """

    ep_hook_dict = copy.deepcopy(ep_hook_dict)
    ep_umap_dict = dict()
    ep_rsm_dict = dict()

    targets = targets = torch.cat(ep_hook_dict.pop("targets")).numpy()
    all_idxs, ns, seq_classes = get_idxs(targets, dataset)

    seq_classes_flat = dataset.image_label_to_class(
        targets[..., 0].reshape(-1), targets[..., 1].reshape(-1)
        )
    idxs_im, idxs_ori = get_rsm_idxs(*list(zip(*seq_classes_flat)))
    rsm_bins = np.linspace(-1, 1 + 2 / 253, 251)
    rsm_vals = np.convolve(rsm_bins, [0.5, 0.5], "valid")
    rsm_vals = [rsm_vals.min(), rsm_vals.max(), len(rsm_vals) + 1]

    dfs = []
    for name in hooks.HOOK_NAMES:
        ep_rsm_dict[name] = dict()
        for hook_type in hooks.HOOK_TYPES:
            data = ep_hook_dict[name].pop(hook_type)
            num_batches = len(data)
            data = torch.cat(data).numpy()
            N, P, C, D, _ = data.shape

            # UMAP dataframe
            df = get_umap_df(
                data.mean(axis=(3, 4)).reshape(N * P, C), seq_classes_flat, 
                name=name, hook_type=hook_type, num_batches=num_batches, 
                pred_step=P
                )
            dfs.append(df)

            # RSM
            upp = np.triu_indices(N * P, k=0)
            rsm_upp = calculate_rsm_corr(data.reshape(N * P, -1))[upp]
            rsm_binned = np.maximum(np.digitize(rsm_upp, rsm_bins) - 1, 0)
            del rsm_upp

            ep_rsm_dict[name][hook_type] = rsm_binned.astype(np.uint8)

            # means and stds across trials
            new_shape = (len(seq_classes), P, C, D, D)
            ep_hook_dict[name][hook_type] = {
                stat: np.full(new_shape, np.nan) for stat in ["means", "stds"]
                }
            for i, pred_idxs in enumerate(all_idxs):
                for p, idxs in enumerate(pred_idxs):
                    if not len(idxs):
                        continue
                    ep_hook_dict[name][hook_type]["means"][i, p] = \
                        data[idxs, p].mean(axis=0)
                    ep_hook_dict[name][hook_type]["stds"][i, p] = \
                        data[idxs, p].std(axis=0)

    # hook data
    ep_hook_dict["seq_classes"] = seq_classes # unique, ordered classes
    ep_hook_dict["ns"] = ns # number per seq class
    ep_hook_dict["losses"] = torch.cat(ep_hook_dict["losses"]).numpy().tolist()
    
    # UMAP data
    ep_umap_dict["umap_df"] = pd.concat(dfs)

    # RSM data
    ep_rsm_dict["idxs_im"] = idxs_im
    ep_rsm_dict["idxs_ori"] = idxs_ori
    ep_rsm_dict["rsm_vals"] = rsm_vals

    return ep_hook_dict, ep_umap_dict, ep_rsm_dict


#############################################
def get_digit_in_key(key):
    """
    get_digit_in_key(key)
    """

    digits = re.findall(r"\d+", key)
    if len(digits) != 1:
        raise ValueError(f"Expected to find exactly one digit in {key}.")

    digit = digits[0]
    idx = key.index(digit)
    if idx > 0 and key[idx - 1] == "-":
        digit = f"-{digit}"

    digit = int(digit)

    return digit

#############################################
def np_pearson_r(x, y, axis=0, nanpol=None):
    """
    np_pearson_r(x, y)

    Returns Pearson R correlation for two matrices, along the specified axis.

    Required args:
        - x (nd array): first array to correlate
        - y (nd array): second array to correlate (same shape as x)
    
    Optional args:
        - axis (int)   : axis along which to correlate values
                         default: 0
        - nanpol (bool): NaN policy
                         default: None

    Returns:
        - corr_bounded (nd array): correlation array, with one less dimension 
                                   than x/y.
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
def get_mean_sem_USI_p_vals(sub_dict, df, common, seq_classes, ns, 
                            pop_indiv=False):
    """
    get_mean_sem_USI_p_vals(sub_dict, df, common, seq_classes, ns)
    """
    
    indiv_dict = dict()
    for i, ((image, ori), n) in enumerate(zip(seq_classes, ns)):
        idx = len(df)
        for k, v in common.items():
            df.loc[idx, k] = v
        df.loc[idx, "image"] = image
        df.loc[idx, "orientation"] = ori
        df.loc[idx, "num_trials"] = sum(n)

        mean, sems, stds = np.nan, np.nan, np.nan
        if n > 0:
            mean = sub_dict["means"][i].mean()
            sems = scipy.stats.sem(sub_dict["means"][i].reshape(-1))
            stds = sub_dict["means"][i].reshape(-1).std()
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
            for k, v in common.items():
                df.loc[idx, k] = v
            df.loc[idx, "image"] = image
            df.loc[idx, "orientation"] = "any"
            diffs = indiv_dict["U"] - indiv_dict["D"]

            if image == "U-D":                            
                df.loc[idx, "mean"] = diffs.mean()
                df.loc[idx, "sem"] = scipy.stats.sem(diffs)
                df.loc[idx, "std"] = diffs.std()

            elif image == "USI":
                stds = [indiv_dict.pop(f"{im}_stds") for im in ["U", "D"]]
                div = np.sqrt(0.5 * np.sum(np.power(stds, 2), axis=0))
                all_USIs = diffs / div
                df.loc[idx, "mean"] = all_USIs.mean()
                df.loc[idx, "sem"] = scipy.stats.sem(all_USIs)
                df.loc[idx, "std"] = all_USIs.std()
            
                if pop_indiv:
                    indiv_dict["USI"] = all_USIs
    
        for key in indiv_dict.keys():
            if "stds" in key:
                indiv_dict.pop(key)

    return df, indiv_dict


#############################################
def get_hook_df(hook_dicts):
    """
    get_hook_df(hook_dicts)
    """

    df = pd.DataFrame(
        columns=[
            "mean", "sem", "std", "name", "hook_type", "unexp", "pre", "sess_n",
            "image", "orientation", "num_trials", 
            ]
        )
    
    all_keys = [sorted(hook_dict.keys()) for hook_dict in hook_dicts]
    for k, keys in enumerate(all_keys[:-1]):
        if keys != all_keys[k + 1]:
            raise ValueError("All hook_dicts must have the same keys.")

    key_dict = dict()
    pres, sess_ns = [], []
    for key in all_keys[0]:
        pre = ("pre" in key)
        sess_n = get_digit_in_key(key)
        key_dict[(pre, sess_n)] = key
        pres.append(pre)
        sess_ns.append(sess_n)

    sess_ns = np.sort(np.unique(sess_ns))
    pres = np.sort(np.unique(pres))

    seed = 100
    for name in hooks.HOOK_NAMES:
        for hook_type in hooks.HOOK_TYPES:
            for pre in pres:
                USI_dict = dict()
                diff_dict = dict()
                for sess_n in sess_ns:
                    common = {
                        "hook_name": name,
                        "hook_type": hook_type,
                        "sess_n": sess_n,
                        "pre": pre,
                    }

                    full_dict = dict()
                    unexps = []
                    for h, hook_dict in enumerate(hook_dicts):
                        key = key_dict[(pre, sess_n)]
                        sub_dict = hook_dict[key][name][hook_type]
                        unexps.append("unexp" in key)
                        common["unexp"] = unexps[-1]
                        common["model_n"] = h

                        df, indiv_dict = get_mean_sem_USI_p_vals(
                            sub_dict, df, common,
                            hook_dict[key]["seq_classes"], 
                            hook_dict[key]["ns"],
                            pop_indiv=(sess_n in [-1, 0, 1, 2])
                            )
                        for key, data in indiv_dict.items():
                            if key not in full_dict.keys():
                                full_dict[key] = []
                            full_dict[key].append(data)

                    # get the overall stats
                    common["model_n"] = "all"
                    common["unexp"] = unexps[0] if len(set(unexps)) == 1 else None
                    for key, data in full_dict.items():
                        idx = len(df)
                        for k, v in common.items():
                            df.loc[idx, k] = v
                        df.loc[idx, "image"] = key
                        df.loc[idx, "orientation"] = "any"
                        data = np.concatenate(data)
                        df.loc[idx, "mean"] = data.mean()
                        df.loc[idx, "sem"] = scipy.stats.sem(data)
                        df.loc[idx, "std"] = data.std()

                        if key == "USI":
                            df.loc[idx, "pval"] = scipy.stats.ttest_rel(
                                data, np.zeros_like(data)
                                )[1]
                            if sess_n in [0, 1, 2]:
                                USI_dict[sess_n] = (data, idx)

                        if key == "U" and "D" in full_dict.keys():
                            idx = len(df)
                            for k, v in common.items():
                                df.loc[idx, k] = v
                            df.loc[idx, "image"] = "U-D"
                            df.loc[idx, "orientation"] = "any"
                            diffs = data - np.concatenate(full_dict["D"])
                            df.loc[idx, "mean"] = diffs.mean()
                            df.loc[idx, "sem"] = scipy.stats.sem(diffs)
                            df.loc[idx, "std"] = diffs.std()
                            df.loc[idx, "pval"] = scipy.stats.ttest_rel(
                                diffs, np.zeros_like(diffs)
                                )[1]

                            if sess_n in [0, 1, 2]:
                                diff_dict[sess_n] = (diffs, idx)
                            del diffs
                    del full_dict

            # get the between session comps
            for use_dict in diff_dict, USI_dict:
                if len(use_dict) == 0:
                    continue
                for sess1, sess2 in [(0, 1), (1, 2), (0, 2)]:
                    if sess1 in use_dict.keys() and sess2 in use_dict.keys():
                        data1, idx1 = use_dict[sess1]
                        data2, idx2 = use_dict[sess2]
                        p_val_diff = scipy.stats.ttest_rel(data1, data2)[1]
                        df.loc[idx1, f"sess{sess1}v{sess2}"] = p_val_diff
                        df.loc[idx2, f"sess{sess1}v{sess2}"] = p_val_diff

            if len(USI_dict):
                # add USI correlations
                for sess1, sess2 in [(0, 1), (1, 2)]:
                    if sess1 in use_dict.keys() and sess2 in use_dict.keys():
                        data1, _ = use_dict[sess1]
                        data2, _ = use_dict[sess2]

                        idx = len(df)
                        for k, v in common.items():
                            df.loc[idx, k] = v
                        df.loc[idx, "image"] = "USI_corr"
                        df.loc[idx, "orientation"] = "any"
                        df.loc[idx, "sess_n"] = f"{sess1}v{sess2}"

                        corr, pval = scipy.stats.pearsonr(data1, data2)
                        corr_std = bootstrapped_corr(data1, data1, seed=seed)
                        
                        df.loc[idx, "corr"] = corr
                        df.loc[idx, "pval"] = pval
                        df.loc[idx, "std"] = corr_std

                        seed += 1

    return df


#############################################
def bootstrapped_corr(data1, data2, seed=100, n_samples=1000):
    randst = np.random.RandomState(100)

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
def plot_save_hook_data(hook_dict, ep_suffix, output_dir="."):

    hook_dir = save_dict(hook_dict, output_dir, dict_type="hook_data")

    return

    hook_df = get_hook_df(hook_dict, ep_suffix)

    breakpoint()


#############################################
def plot_and_save_results(ep_hook_dict, dataset, ep_suffix, learn=False, 
                          output_dir="."):
    """
    plot_and_save_results(info_dict, dataset, ep_suffix)
    """

    logger.info(
        f"2/{NUM_STEPS}. Loading saved data..."
        )
    start_time = time.perf_counter()
    hook_dict, umap_dict, rsm_dict = get_dicts(output_dir)


    stop_time = time.perf_counter()
    time_str = misc_utils.format_time(stop_time - start_time, sep_min=True)
    logger.info(
        f"{TAB}[{time_str}] Saved data loaded."
        f"\n3/{NUM_STEPS}. Compiling data, and plotting and saving UMAP data..."
        )
    start_time = time.perf_counter()
    hook_dict[ep_suffix], umap_dict[ep_suffix], rsm_dict[ep_suffix] = \
        compile_data(ep_hook_dict, dataset)

    plot_save_umaps(umap_dict, ep_suffix, output_dir, plot_batch_num=learn)


    stop_time = time.perf_counter()
    time_str = misc_utils.format_time(stop_time - start_time, sep_min=True)
    logger.info(
        f"{TAB}[{time_str}] Data compiled, and UMAP data plotted and saved."
        f"\n4/{NUM_STEPS}. Plotting and saving RSM data..."
        )
    start_time = time.perf_counter() 
    plot_save_rsms(rsm_dict, ep_suffix, output_dir)


    stop_time = time.perf_counter()
    time_str = misc_utils.format_time(stop_time - start_time, sep_min=True)
    logger.info(
        f"{TAB}[{time_str}] RSM data plotted and saved."
        f"\n5/{NUM_STEPS}. Saving summarized hook data."
        )
    start_time = time.perf_counter() 
    plot_save_hook_data(hook_dict, ep_suffix, output_dir)


    stop_time = time.perf_counter()
    time_str = misc_utils.format_time(stop_time - start_time, sep_min=True)
    logger.info(f"{TAB}[{time_str}] Gabor analysis complete.\n")

    return


#############################################
def gabor_analysis(model, dataloader, optimizer, device="cuda", output_dir=".", 
                   ep_suffix="first", learn=False):
    """
    gabor_analysis(model, dataloader, optimizer)
    """

    run_checks(model, dataloader)

    # Do not use DataParallel
    model = training_utils.get_model_only(model).to(device)

    learn_str_pr = " (no learning)" 
    if learn:
        factor = 0.1
        model, optimizer = training_utils.get_model_optimizer(
            model, optimizer, factor=factor
            )
        learn_str_pr = f" (learning at a {factor}x rate.)"
        ep_suffix = f"{ep_suffix}_learning"
    else:
        model.eval()

    criterion, criterion_no_reduction = loss_utils.get_criteria(device=device)

    ep_hook_dict, activation_hooks, error_hooks = hooks.init_hook_dict(
        model, batch_size=dataloader.batch_size
        )

    for key in ["batch_losses", "losses", "targets"]:
        ep_hook_dict[key] = []

    logger.info(
        "Running extended Gabor analysis\n"
        f"1/{NUM_STEPS}. Recording activations and errors{learn_str_pr}...", 
        extra={"spacing": "\n"}
        )

    start_time = time.perf_counter()
    for idx, (input_seq, sup_target) in enumerate(dataloader):
        input_seq = input_seq.to(device)
        input_seq_shape = input_seq.size()
        [output_, mask_] = model(input_seq)

        # get targets, and reshape for loss calculation
        output_flattened, target_flattened, loss_reshape, _ = \
            training_utils.prep_loss(
                output_, mask_, sup_target, input_seq_shape, 
                supervised=False, is_gabor=True
                )
        D = int(np.sqrt(loss_reshape[-1]))
        loss_reshape = loss_reshape[:-1] + (D, D)

        target_flattened = target_flattened.to(device)
        loss = criterion(output_flattened, target_flattened)
        loss_full = criterion_no_reduction(
            output_flattened, target_flattened
            ).detach()
        
        model.zero_grad()
        if learn:
            optimizer.zero_grad()

        loss.backward()
        if learn:
            optimizer.step()

        model.zero_grad()

        # one target per block
        ep_hook_dict["targets"].append(sup_target[:, -model.pred_step:, 0].cpu())
        ep_hook_dict["losses"].append(loss_full.reshape(loss_reshape).cpu())
        ep_hook_dict["batch_losses"].append(loss.item())
    
    for hook in activation_hooks + error_hooks:
        hook.remove()

    stop_time = time.perf_counter()
    time_str = misc_utils.format_time(stop_time - start_time, sep_min=True)
    start_time = time.perf_counter() 
    logger.info(f"{TAB}[{time_str}] Activations and errors recorded.")

    plot_and_save_results(
        ep_hook_dict, dataloader.dataset, ep_suffix, 
        learn=learn, output_dir=output_dir
        )

