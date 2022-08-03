#!/usr/bin/env python

import logging
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from utils import misc_utils

logger = logging.getLogger(__name__)

TAB = "    "
DEG = u"\u00b0"


#############################################
def get_cmap_colors(cmap_name="Blues", n_vals=5, min_n=5):
    """
    get_cmap_colors()

    Returns colors sampled evenly, with a dark bias, from the specified 
    colormap.

    Optional args
    -------------
    - cmap_name : str (default="Blues")
        Colormap name.
    - n_vals : int (default=5)
        Number of values to sample from the colormap.

    Returns
    -------
    - colors : list
        List of colors sampled from the colormap.
    """

    cmap = mpl.cm.get_cmap(cmap_name)
    samples = np.linspace(1.0, 0.3, max(min_n, n_vals))
    colors = [cmap(s) for s in samples]

    return colors


#############################################
def get_unexp_epoch_ranges(unexp_vals, epoch_ns=None, as_str=False):
    """
    get_unexp_epoch_ranges(unexp_vals)

    Returns ranges of epochs during which unexpected sequences occur, e.g. 
    [[start1, end1], [start2, end2], ...], where the start and end values are 
    inclusive.

    Required args
    -------------
    - unexp_vals : array-like
        List of boolean values indicating whether each epoch included 
        unexpected sequences or not.

    Optional args
    -------------
    - epoch_ns : array-like (default=None)
        Epoch numbers corresponding to unexp_vals. If not provided, indices 
        are reported instead.
    - as_str : bool (default=False)
        If True, unexpected epoch transition information is returned in a 
        string, instead of a nested list.

    Returns
    -------
    - unexp_ranges : list or str
        If as_str, a string listing the ranges of epochs with unexpected 
        sequences. If not, a nested list of start to end ranges. 
    """

    if epoch_ns is not None and len(epoch_ns) != len(unexp_vals):
        raise ValueError(
            "If provided, 'epoch_ns', must have the same length as "
            "'unexp_vals'."
            )
        
    unexp_vals = np.asarray(unexp_vals, dtype=int)
    n_vals = len(unexp_vals)
    if epoch_ns is None:
        epoch_ns = np.arange(n_vals)

    unexp_vals = np.insert(unexp_vals, 0, 0)
    change_vals = np.where(np.diff(unexp_vals) != 0)[0].tolist()
    change_vals.append(n_vals)

    unexp_ranges = []
    for i, val in enumerate(change_vals[:-1]):
        if not i % 2:
            unexp_ranges.append([val, change_vals[i + 1] - 1])

    for u, unexp_range in enumerate(unexp_ranges):
        unexp_ranges[u] = [epoch_ns[i] for i in unexp_range]

    if as_str:
        unexp_ranges_str = ""
        if len(unexp_ranges) == 1 and unexp_ranges[0][-1] == epoch_ns[-1]:
            unexp_ranges_str = f"unexp. epoch: {unexp_ranges[0][0]}"
        elif len(unexp_ranges):
            range_str = ", ".join([f"{v1}-{v2}" for v1, v2 in unexp_ranges])
            unexp_ranges_str = f"unexp. epochs: {range_str}"
        unexp_ranges = unexp_ranges_str

    return unexp_ranges


#############################################
def plot_acc(sub_ax, data_dict, epoch_ns, chance=None, cmap_name="Blues", 
             ls=None, data_label="train"):
    """
    plot_acc(sub_ax, data_dict, epoch_ns)

    Plots accuracy data.
    
    Required args
    -------------
    - sub_ax : plt subplot
        Subplot on which to plot accuracy data.
    - data_dict : dict
        Dictionary recording loss and accuracy information with keys:

        'acc' (list)    : Final local accuracy average per epoch.
        'epoch_n' (list): Epoch number per epoch.
        'loss' (list)   : Final local loss average per epoch.
        'top{k}' (list) : Final local top k accuracy average per epoch.
        
    - epoch_ns : array-like
        Epoch numbers to use for accuracy data x axis.

    Optional args
    -------------
    - chance : float (default=None)
        If not None, level at which a dashed horizontal chance line is plotted.
    - color : str (default="k")
        Color to use to plot the data.
    - ls : str (default=None)
        Linestyle to use to plot the data.
    - data_label : str (default="train")
        Data name to use in labelling the data in the plots.
    """

    if len(data_label) and data_label[-1] != " ":
        data_label = f"{data_label} "

    acc_keys = sorted([key for key in data_dict.keys() if "top" in str(key)])
    colors = get_cmap_colors(cmap_name, n_vals=len(acc_keys))
    if chance is not None:
        sub_ax.axhline(100 * chance, color="k", ls="dashed", alpha=0.8)

    lab_a = len(acc_keys) // 2
    for a, acc_key in enumerate(acc_keys):
        accuracies = 100 * np.asarray(data_dict[acc_key])
        label = None
        if a == lab_a:
            label = f"{data_label}{', '.join(acc_keys)}"
        sub_ax.plot(
            epoch_ns, accuracies, color=colors[a], ls=ls, alpha=0.8, 
            label=label
            )


#############################################
def plot_batches(batch_ax, data_dict, U_ax=None, data_label="train", 
                 cmap_name="Blues"):
    """
    plot_batches(batch_ax, loss_dict)

    Plots loss and accuracy data by batch, and optionally U sequence frequency 
    for the Gabors dataset.
    
    Required args
    -------------
    - batch_ax : plt subplot
        Subplot on which to plot batch data.
    - data_dict : dict
        Dictionary recording loss and accuracy information, under keys:
        'avg_loss_by_batch' (2D list)       : average loss values with dims:
                                              epochs x batches
        'batch_epoch_n' (2D list)           : epoch number, with dims:
                                              epochs x batches
        'sup_target_by_batch' (4 or 6D list): supervised targets with dims: 
            epochs x batches x B x N (x SL x [image type, mean ori] 
            if Gabor dataset).

    Optional args
    -------------
    - U_ax : plt subplot (default=None)
        Subplot on which to plot U sequence frequency by batch for the Gabors 
        dataset, if applicable.
    - data_label : str (default="train")
        Data name to use in labelling the data in the plots.
    - cmap_name : str (default="Blues")
        Name of the colormap to use to determine batch colors.
    """
    
    # transpose to number of batches x number of epochs
    avg_loss_by_batch = np.asarray(data_dict["avg_loss_by_batch"]).T
    epoch_ns = np.asarray(data_dict["batch_epoch_n"]).T

    num_batches = len(avg_loss_by_batch)
    colors = get_cmap_colors(cmap_name, n_vals=num_batches)

    if U_ax is not None:
        target_key = "sup_target_by_batch"

        # num batches x num epochs
        U_freqs_over_DU = np.empty_like(epoch_ns)

        # for loop duration ~= duration i first converting data to np.array
        for e, epoch_data in enumerate(data_dict[target_key]):
            for b, batch_data in enumerate(epoch_data):
                # B x N x SL x (image type, ori)
                batch_data = np.asarray(batch_data)[..., 0].reshape(-1)
                freqs = []
                for image_type in ["U", "D"]:
                    freqs.append(
                        (batch_data == image_type).astype(float).mean()
                        )
                U_freqs_over_DU[b, e] = (freqs[0] / (freqs[0] + freqs[1]) * 100)

    lab_b = len(avg_loss_by_batch) // 2
    for b, batch_losses in enumerate(avg_loss_by_batch):
        label = data_label if b == lab_b and len(data_label) else None
        batch_ax.plot(
            epoch_ns[b], batch_losses, color=colors[b], alpha=0.8, 
            label=label
            )
        if U_ax is not None:
            label = data_label if b == lab_b and len(data_label) else None
            U_ax.plot(
                epoch_ns[b], U_freqs_over_DU[b], color=colors[b], 
                alpha=0.8, label=label, marker="."
            )
    
    epoch_ns = epoch_ns.mean(axis=0)
    batch_ax.plot(
        epoch_ns, avg_loss_by_batch.mean(axis=0), color="k", 
        alpha=0.6, lw=2.5, 
        )
    if U_ax is not None:
        U_ax.plot(
            epoch_ns, U_freqs_over_DU.mean(axis=0), color="k", alpha=0.6, 
            lw=2.5,
            )


#############################################
def plot_gabor_data(sub_ax, data_dict, epoch_ns, datatype="image_types", 
                    data_label="train", cmap_name="Blues", is_accuracy=False, 
                    loss_max=None):
    """
    plot_gabor_data(sub_ax, data_dict, epoch_ns)

    Plot Gabor loss or accuracy data by class values.

    Required args
    -------------
    - sub_ax : plt subplot
        Subplot on which to plot accuracy data.
    - data_dict : dict
        Gabor loss or accuracy data dictionary, with keys:
        '{image_type}' (list)       : image type loss, with dims 
                                        epochs x batchs
        '{mean ori}' (list)         : orientation loss, with dims 
                                        epochs x batchs
        'image_types_overall' (list): overall image type loss, with dims 
                                        epochs x batchs
        'mean_oris_overall'   (list): overall mean ori loss, with dims 
                                        epochs x batchs
        'overall'             (list): overall loss, with dims 
                                        epochs x batchs
    - epoch_ns : array-like
        Epoch numbers to use for accuracy data x axis.


    Optional args
    -------------
    - datatype : str (default="image_types")
        Type of Gabor data to plot by class.
    - data_label : str (default="train")
        Data name to use in labelling the data in the plots.
    - cmap_name : str (default="Blues")
        Name of the colormap to use to determine datatype colors.
    - is_accuracy : bool (default=False)
        If True, the data provided is accuracy data. Otherwise, it is loss 
        data.
    - loss_max : float (default=None)
        If provided, maximum value to use when normalizing data, if it is loss 
        data.

    Returns
    -------
    - y_max : float
        Maximum y value used to normalize the data for stacked plotting.
    """

    if datatype not in ["image_types", "mean_oris"]:
        raise ValueError("'datatype' must be 'image_types' or 'mean_oris'.")

    plot_keys = []
    add_NA = False
    for key in data_dict.keys():
        if "overall" in str(key):
            pass
        elif "." in str(key) or str(key).isdigit():
            if datatype == "mean_oris":
                plot_keys.append(key)
        elif str(key) == "N/A":
            if datatype == "mean_oris":
                add_NA = True
        elif datatype == "image_types":
            plot_keys.append(key)

    if datatype == "mean_oris":
        sort_order = np.argsort([float(key) for key in plot_keys])
        plot_keys = [plot_keys[s] for s in sort_order]
    else:       
        plot_keys = sorted(plot_keys)
    if add_NA:
        plot_keys.append("N/A")
    plot_keys.insert(0, f"{datatype}_overall")
    plot_keys = plot_keys[::-1]

    colors = get_cmap_colors(cmap_name, n_vals=len(plot_keys) + 1)
    mult = 100 if is_accuracy else 1

    all_data = []
    for key in plot_keys:
        # nan_mean across batches for each epoch
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", "Mean of empty slice", RuntimeWarning
                )
            data = np.nanmean(np.asarray(data_dict[key]), axis=1) * mult
        all_data.append(data)
    all_data = np.asarray(all_data)

    y_max = loss_max
    if is_accuracy:
        y_max = 90
    elif loss_max is None:
        y_max = np.ceil(np.nanpercentile(all_data, 95) * 1.05) # allow some overlap

    incr = np.arange(len(plot_keys)).reshape(-1, 1)
    all_data = all_data / y_max + incr

    ytick_labels = []
    lab_k = len(plot_keys) // 2
    for k, key in enumerate(plot_keys):
        key_label = str(key)
        plot_kwargs = dict()
        if key_label.endswith("overall"):
            key_label = "overall"
            plot_kwargs = {"lw": 2.5}
        elif datatype == "mean_oris" and key_label != "N/A":
            key_label = float(key_label)
            if key_label == int(key_label):
                key_label = str(int(key_label))
            key_label = u"{}{}".format(key_label, DEG)
        
        ytick_labels.append(key_label)

        label = None
        if k == lab_k and len(data_label):
            label = data_label

        sub_ax.plot(
            epoch_ns, all_data[k], color=colors[k], alpha=0.8, 
            label=label, **plot_kwargs
            )

    # mark edges of each category
    edges = np.arange(len(plot_keys) + 1)
    for i, edge in enumerate(edges[:-1]):
        if i % 2 == 0:
            sub_ax.axhspan(
                edge, edges[i + 1], color="k", alpha=0.05, lw=0, zorder=-13
                )

    # label between edges
    label_pos = edges[:-1] + 0.5 # mid-way for each key
    sub_ax.set_yticks(label_pos)
    sub_ax.set_yticklabels(ytick_labels)

    # remove minor ticks
    sub_ax.tick_params("y", which="major", length=0, pad=5)

    return y_max



#############################################
def plot_from_loss_dict(loss_dict, num_classes=None, dataset="UCF101", 
                        plot_what="main"):
    """
    plot_from_loss_dict(loss_dict)

    Plots loss and accuracy information from loss dictionary.

    Required args
    -------------
    - loss_dict : dict
        Dictionary recording loss and accuracy information in dictionaries 
        stored under the 'train' and optionally the 'val' key, each with keys:

        'acc' (list)    : Final local accuracy average per epoch.
        'epoch_n' (list): Epoch number per epoch.
        'loss' (list)   : Final local loss average per epoch.
        'top{k}' (list) : Final local top k accuracy average per epoch.

        and optionally:
        'avg_loss_by_batch' (2D list)       : average loss values with dims:
                                              epochs x batches
        'batch_epoch_n' (2D list)           : epoch number, with dims:
                                              epochs x batches
        'loss_by_item' (4d list)            : loss values with dims: 
                                              epochs x batches x B x N
        'sup_target_by_batch' (4 or 6D list): supervised targets with dims: 
            epochs x batches x B x N (x SL x [image type, mean ori] 
            if Gabor dataset).
        'epoch_n_best' (int)                : Best epoch number.
        'epoch_n_best_unexp' (int)          : Best epoch number, when Gabors 
            dataset was set to include unexpected sequences.


        if Gabor dataset:
        'unexp' (list)          : for each epoch, whether unexpectes sequences 
                                  were included.
        'gabor_loss_dict' (list):  Gabor loss dictionary, with keys
            '{image_type}' (list)       : image type loss, with dims 
                                          epochs x batchs
            '{mean ori}' (list)         : orientation loss, with dims 
                                          epochs x batchs
            'image_types_overall' (list): overall image type loss, with dims 
                                          epochs x batchs
            'mean_oris_overall'   (list): overall mean ori loss, with dims 
                                          epochs x batchs
            'overall'             (list): overall loss, with dims 
                                          epochs x batchs
        'gabor_acc_dict' (list) : Gabor top 1 accuracy dictionary, with the 
                                  same keys as 'gabor_loss_dict'.

    Optional args
    -------------
    - num_classes : int (default=None)
        Number of classes, if applicable, used to calculate chance level.
     - dataset : str (default="UCF101")
        Dataset name, used to set some parameters.
    - unexp_epoch : int (default=10)
        Epoch as of which unexpected sequences are introduced, if the dataset 
        is a Gabors dataset.
    - plot_what : str (default="main")
        What to plot, i.e. the main loss and accuracy data ("main"), data by 
        batch ("by_batch") or data by Gabor split ("by_gabor").

    Returns
    -------
    - fig : plt.Figure
        Figure in which loss information is plotted.
    """

    modes = ["train"]
    colors = ["blue"]
    ls = ["dashed"]
    if "val" in loss_dict.keys():
        modes.append("val")
        colors.append("orange")
        ls.append(None)

    chance = None if num_classes is None else 1 / num_classes

    dataset = misc_utils.normalize_dataset_name(dataset)
    if plot_what == "main":
        nrows, ncols = 2, 1
    elif plot_what == "by_batch":
        nrows = 1 + int(dataset == "Gabors")
        ncols = len(modes)
    elif plot_what == "by_gabor":
        nrows, ncols = 2, 2
        if dataset != "Gabors":
            raise ValueError(
                "'plot_what' can only be 'by_gabor' if the dataset is a "
                "'Gabors' dataset."
                )
    else:
        raise ValueError(
            "Accepted values for 'plot_what' are 'main', 'by_batch', and "
            "'by_gabor'."
            ) 

    row_height = 6 if plot_what == "by_gabor" else 3.25 
    fig, ax = plt.subplots(
        nrows, ncols, figsize=[8.5 * ncols, row_height * nrows], sharex=True, 
        squeeze=False
        )
    ax = ax.reshape(nrows, ncols)

    gab_loss_maxes = [None, None]
    add_leg = []
    lab_best = 1
    poss_best_epoch_keys = ["epoch_n_best", "epoch_n_best_unexp"]
    best_epoch_key_colors = ["k", "darkred"]
    for m, mode in enumerate(modes):
        epoch_ns = loss_dict[mode]["epoch_n"]
        cmap_name = f"{colors[m].capitalize()}s"

        if ncols == len(modes) and plot_what != "gabors":
            ax[0, m].set_title(mode.capitalize())

        if plot_what == "by_batch":
            U_ax = ax[1, m] if nrows == 2 else None
            plot_batches(
                ax[0, m], loss_dict[mode], U_ax=U_ax, data_label="", 
                cmap_name=cmap_name
                )
        elif plot_what == "by_gabor":
            add_leg.extend([0])
            lab_best = 2
            for d, datatype in enumerate(["image_types", "mean_oris"]):
                ax[0, d].set_title(f"By {datatype[:-1].replace('_', ' ')}")
                for k, key in enumerate(["gabor_loss_dict", "gabor_acc_dict"]):
                    data_label = mode if d == 0 and k == 0 else ""
                    is_accuracy = (key == "gabor_acc_dict")
                    loss_max = plot_gabor_data(
                        ax[k, d], loss_dict[mode][key], epoch_ns, 
                        datatype=datatype, data_label=data_label, 
                        cmap_name=cmap_name, is_accuracy=is_accuracy, 
                        loss_max=gab_loss_maxes[d]
                    )
                    if not is_accuracy:
                        gab_loss_maxes[d] = loss_max
        else:
            if ncols == len(modes):
                data_label = ""
            else:
                data_label = f"{mode} "
                add_leg.extend([0, 1])

            ax[0, 0].plot(
                epoch_ns, loss_dict[mode]["loss"], color=colors[m], 
                ls=ls[m], label=f"{data_label}loss"
                )

            plot_acc(
                ax[1, 0], loss_dict[mode], epoch_ns, chance=chance, 
                cmap_name=cmap_name, ls=ls[m], data_label=data_label
                )

        if mode == "val":
            for b, best_key in enumerate(poss_best_epoch_keys):
                if best_key not in loss_dict[mode].keys():
                    continue
                unexp_str = " (unexp)" if best_key.endswith("_unexp") else ""
                best_epoch_n = loss_dict[mode][best_key]
                best_idx = np.argmax(loss_dict[mode]["acc"])
                best_idx = epoch_ns.index(best_epoch_n)
                for s, sub_ax in enumerate(ax.reshape(-1)):
                    label = None
                    if s == lab_best:
                        best_acc = 100 * loss_dict[mode]["acc"][best_idx]
                        label = f"ep {best_epoch_n}{unexp_str}: {best_acc:.2f}%"
                        add_leg.extend([lab_best])
                    sub_ax.axvline(
                        best_epoch_n, color=best_epoch_key_colors[b], 
                        ls="dashed", alpha=0.8, label=label
                        )

    min_x, max_x = ax[0, 0].get_xlim()
    for s, sub_ax in enumerate(ax.reshape(-1)):
        if s in add_leg:
            sub_ax.legend(fontsize="small", loc="upper left")
        sub_ax.spines["right"].set_visible(False)
        sub_ax.spines["top"].set_visible(False)
        if dataset == "Gabors":
            unexp_ranges = get_unexp_epoch_ranges(
                loss_dict["train"]["unexp"], loss_dict["train"]["epoch_n"]
                )
            for v1, v2 in unexp_ranges:
                sub_ax.axvspan(
                    v1, v2, color="red", alpha=0.08, lw=0, zorder=-12
                    )
        sub_ax.set_xlim(min_x, max_x)

    ax[0, 0].set_ylabel("Loss")
    for c in range(ncols):
        ax[-1, c].set_xlabel("Epoch")
    if (plot_what == "by_batch") and nrows == 2:
        ax[1, 0].set_ylabel("U/(D+U) frequency (%)")        
    else:
        ax[1, 0].set_ylabel("Accuracy (%)")

    return fig


#############################################
def add_colorbar(im, adj_aspect=True, norm=False, **cbar_kwargs):
    """
    add_colorbar(im)

    Adds a colorbar to the plot associated with the image provided.

    For adj_aspect, aspect ratios are computed for the default plt figure size.

    Required args
    -------------
    - im : mpl.image.AxesImage
        Confusion matrix axis image, with a colormap associated.

    Optional args
    -------------
    - adj_aspect : bool (default=True)
        If True, the colorbar width is adjusted to compensate for tick value 
        width to minimize the impact on the confusion matrix width.
    - norm : bool (default=False)
        If True, the confusion matrix has been normalized. A fixed number of 
        ticks (6) will be plotted on the colorbar.


    Keyword args
    ------------
    **cbar_kwargs : dict
        Keyword arguments for plt.colorbar().
    """
    
    aspect_ratios = {
        "1+"    : 12.6,
        "10+"   : 18,
        "100+"  : 32,
        "1000+" : 140,
    }
    
    if norm:
        adj_aspect = False # unnecessary

    fig = im.figure
    if adj_aspect:
        # ensure that the colormap aspect ratio keeps the widths of the 
        # main plot and overall figure constant
        cbar_kwargs["aspect"] = aspect_ratios["1+"]
        new_aspect = cbar_kwargs["aspect"]
        clims = im.get_clim()

    for _ in range(4):
        cm = fig.colorbar(im, **cbar_kwargs)
        
        if not norm:
            cm.ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        if adj_aspect:
            max_tick = max(
                [tick for tick in cm.ax.get_yticks() if tick <= clims[1]]
                )
            for min_val in [1, 10, 100, 1000]:
                if max_tick >= min_val:
                    new_aspect = aspect_ratios[f"{min_val}+"]

        if adj_aspect and (new_aspect != cbar_kwargs["aspect"]):
            cbar_kwargs["aspect"] = new_aspect
            cm.remove()
        else:
            break

    cm.set_label("Counts", rotation=270, labelpad=18)

    if norm:
        cm.ax.set_yticks(np.linspace(0, 1, 6))

