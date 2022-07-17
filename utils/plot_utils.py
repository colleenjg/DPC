#!/usr/bin/env python

import logging

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from utils import misc_utils

logger = logging.getLogger(__name__)

TAB = "    "


#############################################
def plot_acc(sub_ax, data_dict, epoch_ns, chance=None, color="k", ls=None, 
             data_label="train"):
    """
    plot_acc(sub_ax, data_dict, epoch_ns)

    Plots accuracy data.
    
    Required args
    -------------
    - sub_ax : plt subplot
        Subplot on which to plot accuracy data.
    - data_dict : dict
        Dictionary from which to retrieve accuracy data.
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
    if chance is not None:
        sub_ax.axhline(100 * chance, color="k", ls="dashed", alpha=0.8)
    for a, acc_key in enumerate(acc_keys):
        accuracies = 100 * np.asarray(data_dict[acc_key])
        alpha = 0.7 ** a
        label = None
        if a == 0:
            label = f"{data_label}{', '.join(acc_keys)}"
        sub_ax.plot(
            epoch_ns, accuracies, color=color, ls=ls, alpha=alpha, 
            label=label
            )


#############################################
def plot_batches(batch_ax, data_dict, epoch_ns, U_ax=None, data_label="train", 
                 colors="Blues"):
    """
    plot_batches(batch_ax, loss_dict, epoch_ns)

    Plots loss and accuracy data by batch, and optionally U sequence frequency 
    for the Gabors dataset.
    
    Required args
    -------------
    - batch_ax : plt subplot
        Subplot on which to plot batch data.
    - data_dict : dict
        Dictionary from which to retrieve batch data.
    - epoch_ns : array-like
        Epoch numbers to use for batch data x axis.

    Optional args
    -------------
    - U_ax : plt subplot (default=None)
        Subplot on which to plot U sequence frequency by batch for the Gabors 
        dataset, if applicable.
    - data_label : str (default="train")
        Data name to use in labelling the data in the plots.
    - colors : str (default="Blues")
        Name of the colormap to use to determine batch colors.
    """
    
    avg_loss_by_batch = np.asarray(data_dict["avg_loss_by_batch"]).T
    num_batches = len(avg_loss_by_batch)
    batch_cmap = mpl.cm.get_cmap(colors)
    cmap_samples = np.linspace(1.0, 0.3, num_batches)

    if U_ax is not None:
        target_key = "sup_target_by_batch"
        freqs = []
        for image_type in ["U", "D"]:
            # num_epochs x num_batches x B x N x SL x (image type, ori)
            freqs.append((
                np.asarray(data_dict[target_key])[:, :, ..., 0] == image_type
                ).astype(float).mean(axis=(2, 3, 4)).T)
        U_freqs_over_DU = freqs[0] / (freqs[0] + freqs[1]) * 100

    for b, batch_losses in enumerate(avg_loss_by_batch):
        batch_color = batch_cmap(cmap_samples[b])
        label = data_label if b == 0 and len(data_label) else None
        batch_ax.plot(
            epoch_ns, batch_losses, color=batch_color, alpha=0.8, label=label
            )
        if U_ax is not None:
            label = data_label if b == 0 and len(data_label) else None
            U_ax.plot(
                epoch_ns, U_freqs_over_DU[b], color=batch_color, 
                alpha=0.8, label=label, marker="."
            )
    
    batch_ax.plot(
        epoch_ns, avg_loss_by_batch.mean(axis=0), color="k", alpha=0.6, lw=2.5, 
        )
    if U_ax is not None:
        U_ax.plot(
            epoch_ns, U_freqs_over_DU.mean(axis=0), color="k", alpha=0.6, 
            lw=2.5,
            )


#############################################
def plot_loss_dict(loss_dict, num_classes=None, dataset="UCF101", 
                   unexp_epoch=10, by_batch=False):
    """
    plot_loss_dict(loss_dict)

    Plots loss and accuracy information from loss dictionary.

    Required args
    -------------
    - loss_dict : dict
        Dictionary containing loss and accuracy information. See 
        loss_utils.init_loss_dict() for information on keys and values.

    Optional args
    -------------
    - num_classes : int (default=None)
        Number of classes, if applicable, used to calculate chance level.
     - dataset : str (default="UCF101")
        Dataset name, used to set some parameters.
    - unexp_epoch : int (default=10)
        Epoch as of which unexpected sequences are introduced, if the dataset 
        is a Gabors dataset.
    - by_batch : bool (default=False)
        If True, loss and accuracy data is plotted by batch.

    Returns
    -------
    - fig : plt.Figure
        Figure in which loss information is plotted.
    """

    datatypes = ["train"]
    colors = ["blue"]
    ls = ["dashed"]
    if "val" in loss_dict.keys():
        datatypes.append("val")
        colors.append("orange")
        ls.append(None)

    chance = None if num_classes is None else 1 / num_classes

    dataset = misc_utils.normalize_dataset_name(dataset)
    plot_seq = by_batch and (dataset == "Gabors")
    nrows = 1 + int(plot_seq) if by_batch else 2
    ncols = len(datatypes) if by_batch else 1
    fig, ax = plt.subplots(
        nrows, ncols, figsize=[8.5 * ncols, 3.25 * nrows], sharex=True, 
        squeeze=False
        )
    ax = ax.reshape(nrows, ncols)

    for d, datatype in enumerate(datatypes):
        epoch_ns = loss_dict[datatype]["epoch_n"]  

        data_label = f"{datatype} "
        if ncols == len(datatypes):
            ax[0, d].set_title(datatype.capitalize())
            data_label = ""

        if by_batch:
            U_ax = ax[1, d] if plot_seq else None
            plot_batches(
                ax[0, d], loss_dict[datatype], epoch_ns, U_ax=U_ax, 
                data_label="", colors=f"{colors[d].capitalize()}s"
                )
        else:
            ax[0, 0].plot(
                epoch_ns, loss_dict[datatype]["loss"], color=colors[d], 
                ls=ls[d], label=f"{data_label}loss"
                )

            plot_acc(
                ax[1, 0], loss_dict[datatype], epoch_ns, chance=chance, 
                color=colors[d], ls=ls[d], data_label=data_label
                )

        if datatype == "val":
            best_idx = np.argmax(loss_dict[datatype]["acc"])
            best_epoch_n = epoch_ns[best_idx]
            for s, sub_ax in enumerate(ax.reshape(-1)):
                label = None
                if s == 1:
                    best_acc = 100 * loss_dict[datatype]["acc"][best_idx]
                    label = f"ep {best_epoch_n}: {best_acc:.2f}%"
                sub_ax.axvline(
                    best_epoch_n, color="k", ls="dashed", alpha=0.8, 
                    label=label
                    )

    min_x, max_x = ax[0, 0].get_xlim()
    for sub_ax in ax.reshape(-1):
        if not by_batch:
            sub_ax.legend(fontsize="small")
        sub_ax.spines["right"].set_visible(False)
        sub_ax.spines["top"].set_visible(False)
        if dataset == "Gabors" and unexp_epoch < max(epoch_ns):
            sub_ax.axvspan(
                unexp_epoch, max(epoch_ns), color="k", alpha=0.1, lw=0
                )
        sub_ax.set_xlim(min_x, max_x)

    ax[0, 0].set_ylabel("Loss")
    for c in range(ncols):
        ax[-1, c].set_xlabel("Epoch")
    if by_batch and nrows == 2:
        ax[1, 0].set_ylabel("U/(D+U) frequency (%)")        
    else:
        ax[1, 0].set_ylabel("Accuracy (%)")

    return fig


