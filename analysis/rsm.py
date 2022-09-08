import copy
from pathlib import Path

import h5py
from matplotlib import pyplot as plt
import numpy as np

from analysis import analysis_utils, hooks
from utils import plot_utils


#############################################
def save_rsm_h5(ep_save_dict, output_dir=".", epoch_str="0", pre=True, 
                learn=False):
    """
    save_rsm_h5(ep_save_dict)


    Required args
    -------------
    - ep_save_dict : dict
        Dictionary in which epoch data is stored.

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
        slower rate.

    Returns
    -------
    - save_direc : Path
        Save directory.
    """

    save_path = analysis_utils.get_dict_path(output_dir, dict_type="rsm")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    epoch_str, pre_str = analysis_utils.get_epoch_pre_str(epoch_str, pre, learn)

    comp_kwargs = {"compression": "gzip", "chunks": True}
    with h5py.File(save_path, "a") as f:
        if pre_str in f.keys() and epoch_str in f[pre_str].keys():
            raise RuntimeError(
                f"RSM h5 file already contains a {pre_str}/{epoch_str} key."
                )
        main_keys = f"{pre_str}/{epoch_str}"
        # rsm_bin_vals, idxs_im/{}, idxs_ori/{}, {hook_modules}/{hook_types} 
        for key1, sub1 in ep_save_dict.items(): 
            if key1 == "rsm_bin_vals":
                f.create_dataset(f"{main_keys}/{key1}", data=sub1)
                continue
            for key2, sub2 in sub1.items():
                kwargs = dict()
                if "idxs" not in key2:
                    kwargs = comp_kwargs
                f.create_dataset(
                    f"{main_keys}/{key1}/{key2}", data=sub2, **kwargs
                    )
    
    save_direc = save_path.parent

    return save_direc


#############################################
def load_rsm_h5(save_path):
    """
    load_rsm_h5(save_path)

    Loads an RSM dictionary from an h5 file.

    Required args
    -------------
    - save_path : str or Path
        Path from which to load h5 file and build RSM dictionary.

    Returns
    -------
    - data_dict : dict
        Dictionary built from the RSM h5 file.
    """

    if not Path(save_path).is_file():
        raise OSError(f"{save_path} is not a file.")

    data_dict = dict()
    with h5py.File(save_path, "r") as f:
        for pre_str in f.keys():
            data_dict[pre_str] = dict()
            for epoch_str in f[pre_str].keys():
                data_dict[pre_str][epoch_str] = dict()
                # rsm_bin_vals, idxs_im/{}, idxs_ori/{}, {hook_modules}/{hook_types} 
                for key1 in f[pre_str][epoch_str].keys(): 
                    if key1 == "rsm_bin_vals":
                        data_dict[pre_str][epoch_str][key1] = \
                            f[pre_str][epoch_str][key1][()]
                        continue
                    data_dict[pre_str][epoch_str][key1] = dict()
                    for key2 in f[pre_str][epoch_str][key1].keys():
                        data_dict[pre_str][epoch_str][key1][key2] = \
                            f[pre_str][epoch_str][key1][key2][()]
    
    return data_dict


#############################################
def get_rsm_sorting_idxs(images, oris):
    """
    get_rsm_sorting_idxs(images, oris)

    Returns indices for nested sorting of RSM data by image and orientation.

    Required args
    -------------
    - images : array-like
        Image labels in their original order.
    - oris : array-like
        Orientation labels in their original order.

    Returns
    -------
    - idxs_im : dict
        Index dictionary, where the inner labels are Gabor orientations, and 
        the outer labels are Gabor images, with keys
        'inner_ends' (1D array)  : Final index for each inner label (exclusive)
        'inner_labels' (1D array): Inner labels
        'outer_ends' (1D array)  : Final index for each outer label (exclusive)
        'outer_labels' (1D array): Outer labels
        'sorter' (1D array)      : Sorting indices
    - idxs_ori : dict
        Index dictionary, where the inner labels are Gabor images, and 
        the outer labels are Gabor orientations, with the same keys as idxs_im.
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

    all_idxs = []
    keys = [
        "sorter", "inner_labels", "outer_labels", "inner_ends", "outer_ends"
        ]
    uniques = [unique_images, unique_oris]
    vals = [images, oris]
    for u, (outer_unique, outer_labels) in enumerate(zip(uniques, vals)):
        inner_unique = uniques[1 - u]
        inner_labels = vals[1 - u]
        idxs = {k: [] for k in keys}
        i, j = 0, 0
        for outer_label in outer_unique:
            outer_idxs = np.where(
                np.asarray(outer_labels).astype(str) == str(outer_label)
                )[0]
            if len(outer_idxs) == 0:
                continue
            i += len(outer_idxs)
            idxs["outer_labels"].append(outer_label)
            idxs["outer_ends"].append(i)
            for inner_label in inner_unique:
                inner_idxs = np.where(
                    np.asarray(inner_labels)[outer_idxs].astype(str) == str(inner_label)
                    )[0]
                if len(inner_idxs) == 0:
                    continue
                idxs["sorter"].extend(outer_idxs[inner_idxs].tolist())
                j += len(inner_idxs)
                idxs["inner_labels"].append(inner_label)
                idxs["inner_ends"].append(j)
        idxs["sorter"] = np.asarray(idxs["sorter"])
        for key, val in idxs.items():
            if "labels" in key:
                idxs[key] = np.asarray(val).astype("S10")
            elif "ends" in key:
                idxs[key] = np.asarray(val).astype(int)
        all_idxs.append(idxs)

    idxs_im, idxs_ori = all_idxs

    return idxs_im, idxs_ori


#############################################
def calculate_rsm_corr(features):
    """
    calculate_rsm_corr(features)

    Calculates representational similarity matrix (RSM) between for a feature 
    matrix using pairwise cosine similarity after data is centered. This 
    calculation is equivalent to pairwise Pearson correlations.

    Adapted from https://github.com/rsagroup/rsatoolbox/blob/main/rsatoolbox/rdm/calc.py

    Required args:
    - features : 2D array
        Feature matrix, with dims: items x features

    Returns:
    - rsm : 2D array
        Similarity matrix, with dims: nbr features items x nbr features items
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
def get_binned_rsm_data(data):
    """
    get_binned_rsm_data(data)

    Calculates RSM over the C and D dimensions, bins the data, and returns the 
    upper triangle and bin values. 

    Required args
    -------------
    - data : 5D array
        Data array with dims: N x P x C x D x D

    Returns
    -------
    - rsm_binned : 1D array
        Upper triangle RSM bin values
    - rsm_bin_vals : tuple
        Bin values, provided as (start, stop, number of bins) for use with 
        np.linspace
    """

    N, P, _, _, _ = data.shape

    rsm_bin_edges = np.linspace(-1, 1 + 2 / 253, 251)
    rsm_bin_centers = np.convolve(rsm_bin_edges, [0.5, 0.5], "valid")
    rsm_bin_vals = (
        rsm_bin_centers.min(), rsm_bin_centers.max(), len(rsm_bin_centers) + 1
    )

    upp = np.triu_indices(N * P, k=0)
    rsm_upp = calculate_rsm_corr(data.reshape(N * P, -1))[upp]
    rsm_binned = np.maximum(np.digitize(rsm_upp, rsm_bin_vals) - 1, 0)
    
    return rsm_binned, rsm_bin_vals


############################################
def mark_rsm_axes(ax, idx_dict):
    """
    mark_rsm_axes(ax, idx_dict)

    Marks RSM axes with inner and outer labels, and outer edge lines.

    - ax : 2D array
        Axis subplots to mark.
    - idx_dict : dict
        Index dictionary, where the inner and outer labels are Gabor 
        orientations and images, respectively, or v.v., with keys
        'inner_ends' (1D array)  : Final index for each inner label (exclusive)
        'inner_labels' (1D array): Inner labels
        'outer_ends' (1D array)  : Final index for each outer label (exclusive)
        'outer_labels' (1D array): Outer labels
        'sorter' (1D array)      : Sorting indices
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
                        subax.axhline(edge, lw=1, color="k")
                        subax.axvline(edge, lw=1, color="k")


#############################################
def expand_idxs_for_rare(idx_dict):
    """
    expand_idxs_for_rare(idx_dict)

    Expands index edges and adjusts sorting index to stretch underrepresented 
    categories to match the median width and height across categories.

    Required args
    -------------
    - idx_dict : dict
        Index dictionary, where the inner and outer labels are Gabor 
        orientations and images, respectively, or v.v., with keys
        'inner_ends' (1D array)  : Final index for each inner label (exclusive)
        'inner_labels' (1D array): Inner labels
        'outer_ends' (1D array)  : Final index for each outer label (exclusive)
        'outer_labels' (1D array): Outer labels
        'sorter' (1D array)      : Sorting indices
    
    Returns
    -------
    - exp_idx_dict : dict
        Index dictionary, expanded for underrepresented categories.
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
        return idx_dict

    # update dictionary
    updates = []
    u, base_add = 0, 0
    exp_idx_dict = copy.deepcopy(idx_dict)
    sorter = []
    for l, end in enumerate(exp_idx_dict["inner_ends"]):
        diff = new_lengths[l] - lengths[l]
        if factors[l] == 1:
            sorter.extend(range(u, end))
        else:
            sorter.extend(np.repeat(np.arange(u, end), factors[l]).tolist())
        u = end

        base_add += diff
        new_end = end + base_add
        updates.append((end, new_end))
        exp_idx_dict["inner_ends"][l] = new_end

    outer_ends = np.asarray(exp_idx_dict["outer_ends"])
    for old, new in updates[::-1]:
        if old in outer_ends:
            l = outer_ends.tolist().index(old)
            exp_idx_dict["outer_ends"][l] = new

    return exp_idx_dict


#############################################
def plot_save_rsms(ep_rsm_dict, epoch_str="0", pre=True, learn=False, 
                   output_dir="."):
    """
    plot_save_rsms(ep_rsm_dict)

    Plots representational similarity matrices for the epoch.

    Required args
    -------------
    - ep_rsm_dict : dict
        Epoch RSM dictionary, with keys
        'idxs_im' (dict)      : index dictionary, where the inner labels are 
                                Gabor orientations, and the outer labels are 
                                Gabor images, with keys
            'inner_ends' (1D array)  : Final index for each inner label (excl.)
            'inner_labels' (1D array): Inner labels
            'outer_ends' (1D array)  : Final index for each outer label (excl.)
            'outer_labels' (1D array): Outer labels
            'sorter' (1D array)      : Sorting indices
        'idxs_ori' (dict)     : index dictionary, where the inner labels are 
                                Gabor images, and the outer labels are Gabor 
                                orientations, with the same keys as idxs_im.
        'rsm_bin_vals' (tuple): bin values, provided as 
                                (start, stop, number of bins) for use with 
                                np.linspace
        for hook_module in ["contextual", "encoder", "predictive"]
            for hook_type in ["activations", "errors"]
                '{hook_module}''{hook_type}' (1D array): 
                    upper triangle RSM bin values

    Optional args
    -------------
    - epoch_str : str (default: "0")
        String providing epoch information, for saving.
    - pre : bool (default: True)
        If True, hook data is being collected before epoch training in the main 
        loop.
    - learn : bool (default: False)
        If True, hook data is collected while model is being trained at a 
        slower rate.
    - output_dir : str or path (default=".")
        Main output directory in which to save collected data.
    """

    epoch_str, pre_str = analysis_utils.get_epoch_pre_str(epoch_str, pre, learn)
    rsm_dir = save_rsm_h5(
        ep_rsm_dict, output_dir=output_dir, epoch_str=epoch_str, pre=pre, 
        learn=learn
        )

    idxs_im = ep_rsm_dict["idxs_im"]
    idxs_ori = ep_rsm_dict["idxs_ori"]
    rsm_bin_vals = ep_rsm_dict["rsm_bin_vals"]
    rsm_bin_vals = np.linspace(
        rsm_bin_vals[0], rsm_bin_vals[1], int(rsm_bin_vals[2])
        )

    nrows, ncols = len(hooks.HOOK_TYPES), len(hooks.HOOK_MODULES)

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

        upper_idxs = np.triu_indices(len(sorter), k=0)
        lower_idxs = (upper_idxs[1], upper_idxs[0])
        rasterize_ax = []
        for m, hook_module in enumerate(hooks.HOOK_MODULES):
            for t, hook_type in enumerate(hooks.HOOK_TYPES):
                exp_idx_dict = expand_idxs_for_rare(rsm, idx_dict)
                sorter = exp_idx_dict["sorter"]
                rsm = np.empty((len(sorter), len(sorter)))
                rsm_binned = ep_rsm_dict[hook_module][hook_type]
                rsm[upper_idxs] = rsm_binned
                rsm[lower_idxs] = rsm_binned
                rsm = rsm_bin_vals[rsm.astype(int)]
                rsm = rsm[sorter][:, sorter]
                im = ax[t, m].imshow(
                    rsm, vmin=-1, vmax=1, interpolation="none", zorder=-10
                    )
                if t == 0:
                    ax[t, m].set_title(hook_module.capitalize(), y=1.06)
                if m == 0:
                    ax[t, m].set_ylabel(
                        hook_type.capitalize(), fontsize="large"
                        )
                mark_rsm_axes(ax, exp_idx_dict)
                rasterize_ax.append(ax[t, m])

        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label(label="Similarity", size="large")
        cbar_ax.yaxis.set_label_position("left")

        for subax in rasterize_ax:
            subax.set_rasterization_zorder(-9)

        savepath = Path(rsm_dir, f"{epoch_str}_{pre_str}_{labels[i]}.svg")
        fig.savefig(savepath, format="svg", bbox_inches="tight", dpi=300)

    return


