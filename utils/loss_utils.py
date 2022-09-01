#!/usr/bin/env python

import argparse
from collections import deque
import copy
import json
import logging
from pathlib import Path
import sys
import warnings

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch

sys.path.extend(["..", str(Path("..", "utils"))])
from utils import misc_utils, plot_utils

logger = logging.getLogger(__name__)

TAB = "    "


#############################################
class AverageMeter(object):
    """
    Class for computing and storing average and current values, as new values 
    are added.

    Attributes
    ----------
    - self.avg : float
        Overall average to date.
    - self.count : int
        Total count to date.
    - self.history : list
        Full history (not reflecting counts).
    - self.local_history : deque 
        Recent history (not reflecting counts).
    - self.local_avg : float
        Average over self.local_history. May be biased if counts are not 
        consistent across values.
    - self.sum : float
        Sum across all values.
    - self.val : float
        Most recently added value.

    Methods
    -------
    - self.reset()
        Resets the attributes.
    - self.set_local_length()
        Sets the value of self.local_length, and adjusts length of 
        self.local_history, if necessary.
    - self.update(val)
        Updates attributes with the most recent value.
    """
    
    def __init__(self, local_length=5):
        """
        AverageMeter()

        Contructs an AverageMeter object.

        Optional args
        -------------
        - local_length : int (default=5)
            Maximum length at which to maintain self.local_history.
        """
        
        self.reset(local_length)


    def reset(self, local_length=5):
        """
        self.reset()

        Resets all attributes.

        Optional args
        -------------
        - local_length : int (default=5)
            Maximum length at which to maintain self.local_history.
        """
        
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.local_history = deque([])
        self.local_avg = 0
        self.history = []

        self.set_local_length(local_length)


    def set_local_length(self, local_length=5):
        """
        self.set_local_length()

        Sets the local_length attribute, and trims self.local_history, if 
        applicable.

        Optional args
        -------------
        - local_length : int (default=5)
            Maximum length at which to maintain self.local_history.
        """

        if hasattr(self, "local_length"):
            if local_length != self.local_length:
                self.local_length = int(local_length)
                while len(self.local_history) > self.local_length:
                    self.local_history.popleft()
        else:
            self.local_length = int(local_length)


    def update(self, val, n=1, add_to_history=False):
        """
        self.update(val)

        Updates the data stored in the average meter.

        Required args
        -------------
        - val : float
            Most recent value.

        Optional args
        -------------
        - n : int (default=1)
            Value by which to increment self.count. 
        - add_to_history : bool (default=False)
            If True, val is added to self.history.
        - local_length : int (default=5)
            Number of values to retain in self.local_history.
        """
        
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if add_to_history:
            self.history.append(val)
        if self.local_length > 0:
            self.local_history.append(val)
            if len(self.local_history) > self.local_length:
                self.local_history.popleft()
            self.local_avg = np.average(self.local_history)


    def __len__(self):
        """
        self.__len__()

        Returns the current count number for the average meter.  

        Returns
        -------
        - self.count : int
            Current count number.
        """
        
        return self.count


#############################################
class AccuracyTable(object):
    """
    Class for computing accuracy tables.
    
    Attributes
    ----------
    - self.df : pd.DataFrame
        Accuracy table dictionary, where each row index is a target label, and 
        the columns are 'correct' and 'count'. 

    Methods
    -------
    - self.log_table()
        Logs accuracy data at the INFO level.
    - self.update(pred, target)
        Updates accuracy table from prediction and target data.
    """

    def __init__(self):
        """
        AccuracyTable()

        Constructs an AccuracyTable object.
        """
        
        self.df = pd.DataFrame(columns=["correct", "count"])


    def update(self, pred, target):
        """
        self.update(pred, target)

        Updates accuracy table from prediction and target data.

        Required args
        -------------
        - pred : 1D Tensor
            Prediction values (int datatype).
        - target : 1D Tensor
            Target values (int datatype).
        """
        
        pred = torch.squeeze(pred)
        target = torch.squeeze(target)
        for i, j in zip(pred, target):
            i = int(i)
            j = int(j)
            if j not in self.df.index:
                self.df.loc[j] = [0, 0]
            self.df.loc[j, "count"] += 1
            if i == j:
                self.df.loc[j, "correct"] += 1


    def log_table(self, label=None):
        """
        self.log_table()

        Logs accuracy table values by index value at the INFO level.

        Optional args
        -------------
        - label : str (default=None)
            Label to lead accuracy logs with.
        """
        
        label_str = ""
        if label is not None and len(label):
            label_str = f"{label} "

        logger.info(f"{label_str}accuracy:".capitalize())
        for j in self.df.index:
            correct = self.df.loc[j, "correct"]
            count = self.df.loc[j, "count"]
            div = f"{correct}/{count}".rjust(7)
            acc = correct / count * 100
            logger.info(f"  [{j:03}]: {div} = {acc:05.2f}%")
        
        correct = self.df["correct"].sum()
        count = self.df["count"].sum()
        div = f"{correct}/{count}".rjust(7)
        acc = correct / count * 100
        logger.info(f"Overall: {div} = {acc:05.2f}%")


#############################################
class ConfusionMeter(object):
    """
    Class for computing and plotting classification performance in a confusion 
    matrix.

    Attributes
    ----------
    - self.class_names : array-like
        Ordered list of class names or None.
    - self.mat : 2D array
        Confusion matrix, with dimensions 
        self.num_classes (predicted) x self.num_classes (target).
    - self.num_classes : int
        Number of classes.
    - self.precision : list
        List for collecting precision scores.
    - self.recall : list
        List for collecting recall scores.

    Methods
    -------
    - self.add_labels(ax)
        Adds axis labels to the confusion matrix plot, optionally based on 
        class names.
    - self.annotate(ax)
        Adds class label annotations to the confusion matrix plot.
    - self.get_label_dict()
        Returns a label dictionary for associating class labels to names.
    - self.get_storage_dict()
        Returns the object data in a storage dictionary that can be directly 
        saved as a json file.
    - self.load_from_storage_dict()
        Reinitializes the current object based on a storage dictionary 
        previously generated by self.get_storage_dict(
    - self.log_mat()
        Logs confusion matrix shape at the INFO level.
    - self.get_norm_mat()
        Returns a copy of self.mat, normalized along the specified dimension, 
        with dimensions predicted self.num_classes x target self.num_classes. 
    - self.plot_mat()
        Plots and optionally saves the confusion matrix.
    - self.reinitialize_values()
        Reinitializes current object with new class names or a new number of 
        classes.
    - self.set_class_names()
        Initializes self.class_names from a list of class names.
    - self.update(pred, target)
        Updates confusion matrix based on prediction and target data.
    """

    def __init__(self, class_names=None, num_classes=5):
        """
        ConfusionMeter()

        Constructs ConfusionMeter object.

        Optional args
        -------------
        - class_names : array-like (default=None)
            Ordered list of class names.
        - num_classes : int (default=5)
            Number of classes, used if class_names is None, and ignored 
            otherwise.
        """
        
        self.reinitialize_values(class_names, num_classes=num_classes)


    def reinitialize_values(self, class_names=None, num_classes=5):
        """
        self.reinitialize_values()

        Reinitializes the attributes and properties of the current 
        ConfusionMeter object based on new class names or number of classes.

        Optional args
        -------------
        - class_names : array-like (default=None)
            Ordered list of class names.
        - num_classes : int (default=5)
            Number of classes, used if class_names is None, and ignored 
            otherwise.
        """
        
        if class_names is not None:
            num_classes = len(class_names)

        self.num_classes = num_classes
        self.set_class_names(class_names)
        self.mat = np.zeros((num_classes, num_classes)).astype(int)
        self.precision = []
        self.recall = []


    def set_class_names(self, class_names=None):
        """
        self.set_class_names()

        Sets the class names attribute.

        Optional args
        -------------
        - class_names : array-like (default=None)
            Ordered list of class names.
        """
        
        if class_names is not None and (len(class_names) != self.num_classes):
            raise ValueError(
                f"Number of class names ({len(class_names)}) does not match "
                f"'self.num_classes' ({self.num_classes})"
                )
        self.class_names = class_names


    def update(self, pred, target):
        """
        self.update(pred, target)

        Updates self.mat with new predicted and target values.

        Required args
        -------------
        - pred : torch Tensor or nd array
            Predicted values (int datatype).
        - target : torch Tensor or nd array
            Target values (int dataype).
        """

        pred = np.asarray(pred).squeeze()
        target = np.asarray(target).squeeze()
        for p, t in zip(pred.flat, target.flat):
            self.mat[p][t] += 1


    def log_mat(self):
        """
        self.log_mat()

        Logs confusion matrix shape at the INFO level.
        """
        
        logger.info(f"Confusion Matrix (target in columns):\n{self.mat}")


    def annotate(self, ax):
        """
        self.annotate(ax)

        Annotates a plotted confusion matrix by addin labels as text to each 
        box in the grid.

        Required args
        -------------
        - ax : plt subplot
            Axis on which to add annotations.
        """
        
        height, width = self.mat.shape
        for x in range(width):
            for y in range(height):
                ax.annotate(
                    str(int(self.mat[x][y])), xy=(y+1, x+1),
                    horizontalalignment="center", 
                    verticalalignment="center", fontsize=8
                    )


    def get_label_dict(self):
        """
        self.get_label_dict()

        Retrieves a label dictionary based on the ordered class names.

        Returns
        -------
        - label_dict : dict
            Dictionary where each key is a class label and each 
            item is the corresponding class name.
        """
        
        if self.class_names is None:
            raise ValueError(
                "Cannot include class names, as self.class_names is "
                "not set."
                )
        label_dict = {i: cl for i, cl in enumerate(self.class_names)}
    
        return label_dict


    def add_labels(self, ax, label_dict=None, secondary=False):
        """
        self.add_labels(ax)

        Adds axis labels to a plotted confusion matrix.

        Required args
        -------------
        - ax : plt subplot
            Axis on which to add the labels.

        Optional args
        -------------
        - label_dict : dict (default=None)
            If provided, dictionary where each key is a class label and each 
            item is the corresponding class name. Otherwise, default axis 
            labels are used, with minimum intervals of 1. 
        - secondary : bool (default=False)
            If True, labels are added as a secondary set, along the top and 
            right axes, instead of the bottom and left ones.
        """
        
        height, width = self.mat.shape
        if label_dict is None:
            xtick_diff = min(np.diff(ax.get_xticks()))
            if xtick_diff < 1:
                ax.set_xticks(np.arange(width) + 1)
            ytick_diff = min(np.diff(ax.get_yticks()))
            if ytick_diff < 1:
                ax.set_yticks(np.arange(height) + 1)
        else:
            use_label_dict = {k + 1: v for k, v in label_dict.items()}
            plot_utils.add_sets_of_ticks(
                ax, use_label_dict, secondary=secondary
                )


    def get_norm_mat(self, by="targ"):
        """
        self.get_norm_mat()

        Returns a copy of self.mat, normalized by class frequency along a 
        specified dimension.

        Optional args
        -------------
        - by : str (default="targ")
            Dimension by which to normalize the confusion matrix, 
            i.e., either 'targ' for target class frequencies, 'pred' for 
            predicted class frequencies or 'all' for all frequencies.

        Returns
        -------
        - norm_mat : 2D array
            Normalized confusion matrix, with dimensions 
            predicted self.num_classes x target self.num_classes. 
        """

        norm_mat = copy.deepcopy(self.mat)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", "invalid value", RuntimeWarning
                )
            if by == "targ":
                norm_mat = norm_mat / norm_mat.sum(axis=0, keepdims=True)
            elif by == "pred":
                norm_mat = norm_mat / norm_mat.sum(axis=1, keepdims=True)
            elif by == "all":
                norm_mat = norm_mat / norm_mat.sum()
        
        norm_mat[~np.isfinite(norm_mat)] = 0 # due to division by 0
        
        return norm_mat


    def plot_mat(self, save_path=None, incl_class_names=True, annotate=False, 
                 ax=None, title=None, vmax=None, adj_aspect=True, norm="targ", 
                 **cbar_kwargs):
        """
        self.plot_mat()

        Plots and optionally saves self.mat.

        Optional args
        -------------
        - save_path : str or path (default=None)
            If provided, path under which to save confusion matrix plot.
        - incl_class_names : bool (default=True)
            If True, class names are included in plotting the confusion matrix. 
            If there are hundreds of class names, plot saving may take several 
            minutes.
        - annotate : bool (default=False)
            If True, confusion matrix grid is annotated with predicted and 
            target class labels.
        - ax : plt subplot (default=None)
            Subplot on which to plot confusion matrix. If None, a figure and 
            associated subplot are initialized.
        - title : str (default=None)
            Plot title.
        - vmax : int (default=None)
            Max value of the colormap.
        - adj_aspect : bool (default=True)
            If True, the colorbar width is adjusted to compensate for tick 
            value width to minimize the impact on the confusion matrix width.
        - norm : str (default="targ")
            If not False, dimension by which to normalize the confusion matrix, 
            i.e., either 'targ' for target class frequencies, 'pred' for 
            predicted class frequencies or 'all' for all frequencies.

        Keyword args
        ------------
        **cbar_kwargs : dict
            Keyword arguments for plt.colorbar().

        Returns
        -------
        if save_path is None:
        - fig : plt Figure
            Confusion matrix plot.
        """

        plot_mat = self.mat
        if norm:
            plot_mat = self.get_norm_mat(norm)

        if vmax is not None and vmax < plot_mat.max():
            raise ValueError(
                "Confusion matrix contains values higher than 'vmax'."
                )
        elif plot_mat.max() < 1:
            vmax = 1

        if plot_mat.min() < 0:
            raise ValueError("Confusion matrox contains values below 0.")

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure


        im = ax.imshow(plot_mat,
            cmap=plt.cm.viridis,
            interpolation=None,
            extent=(
                0.5, 
                np.shape(plot_mat)[0] + 0.5, 
                np.shape(plot_mat)[1] + 0.5, 
                0.5
                ),
            vmin=0,
            vmax=vmax
            )
        
        ax.set_xlabel("Ground Truth")
        ax.set_ylabel("Prediction")

        if title is not None:
            ax.set_title(title, y=1.05)

        if annotate:
            self.annotate(ax)

        label_dict = self.get_label_dict() if incl_class_names else None            
        self.add_labels(ax, label_dict=label_dict)
        plot_utils.add_colorbar(
            im, adj_aspect=adj_aspect, norm=norm, **cbar_kwargs
            )

        if save_path is None:
            return fig
        else:
            if incl_class_names and (len(label_dict) > 60):
                logger.warning(
                    "Saving confusion matrix to svg with class labels may be "
                    f"slow, as this corresponds to {len(label_dict)} labels "
                    "per axis."
                    )
            Path(save_path).parent.mkdir(exist_ok=True, parents=True)
            fig.savefig(save_path, format="svg", bbox_inches="tight", dpi=600)
            plt.close(fig)
    

    def get_storage_dict(self):
        """
        self.get_storage_dict()
    
        Returns a storage dictionary containing the properties needed to reload 
        the ConfusionMeter object.

        Returns
        -------
        - storage_dict : dict
            Storage dictionary containing the main properties needed to reload 
            the ConfusionMeter object.
        """

        storage_dict = dict()
        storage_dict["num_classes"] = self.num_classes
        storage_dict["class_names"] = self.class_names
        storage_dict["precision"] = self.precision
        storage_dict["recall"] = self.recall

        idx = np.where(self.mat != 0)
        vals = self.mat[idx].astype(int).tolist()
        idx = tuple([ax_idx.tolist() for ax_idx in idx])
        storage_dict["mat_sparse"] = [idx, vals]

        return storage_dict


    def load_from_storage_dict(self, storage_dict):
        """
        self.load_from_storage_dict(storage_dict)

        Reinitializes the current ConfusionMeter object from a storage 
        dictionary generated with self.get_storage_dict().

        Required args
        -------------
        - storage_dict : dict
            Storage dictionary containing the main properties needed to reload 
            the ConfusionMeter object.
        """

        self.reinitialize_values(
            storage_dict["class_names"], 
            num_classes=storage_dict["num_classes"]
            )
        self.precision = storage_dict["precision"]
        self.recall = storage_dict["recall"]

        idx, vals = storage_dict["mat_sparse"]
        idx = tuple([np.asarray(ax_idx) for ax_idx in idx])
        self.mat[idx] = np.asarray(vals)


#############################################
def check_topk(ks=[1, 3, 5], num_classes=None):
    """
    check_topk()

    Checks top k values, and removes any that are too high for the total number 
    of classes.

    Optional args
    -------------
    - ks : list (default=[1, 3, 5])
        Top k values to record.
    - num_classes : int (default=None)
        Number of classes, used to remove incompatible top k values.

    Returns
    -------
    - ks : list
        Top k values to retain.
    """
    
    if num_classes is not None:
        ks = [k for k in ks if k < num_classes]

    return ks


#############################################
def init_meters(n_topk=3):
    """
    init_meters()

    Initializes Average meter objects for each top k value, and for the loss.

    Optional args
    -------------
    - n_topk : int (default=3)
        Number of top k Average meter objects to initialize.

    Returns
    -------
    - losses : AverageMeter
        Average meter object in which to collect loss data.
    - topk_meters : list
        List of average meter objects in which to collect top k data.
    """
    
    losses = AverageMeter()
    topk_meters = [AverageMeter() for _ in range(n_topk)]

    return losses, topk_meters



#############################################
def get_criteria(criterion="cross-entropy", loss_weights=None, device="cpu"):
    """
    get_criteria()

    Returns criterion functions, with and without reduction across batch items.

    Optional args
    -------------
    - criterion : str (default="cross-entropy")
        Criterion to use.
    - loss_weights : tuple (default=None)
        Class weights to provide to the loss function.
    - device : torch.device or str (default="cpu")
        Device on which to place loss weight, if applicable.

    Returns
    -------
    - criterion : torch loss function
        Criterion used to compute loss for backpropagation.
    - criterion_no_reduction : torch loss function
        Criterion used to compute loss for records, retaining individual values 
        for each batch item.
    """

    if criterion == "cross-entropy":
        criterion_fct = torch.nn.CrossEntropyLoss
    else:
        raise NotImplementedError("Only 'cross-entropy' loss is implemented.")

    if loss_weights is not None:
        loss_weights = loss_weights.to(device)
    criterion = criterion_fct(weight=loss_weights)
    criterion_no_reduction = criterion_fct(
        weight=loss_weights, reduction="none"
        )

    return criterion, criterion_no_reduction


#############################################
def get_stats(losses, topk_meters, ks=[1, 3, 5], local=False, incl_last=False, 
              chance=None):
    """
    get_stats(losses, topk_meters)

    Returns statistics retrieved from loss and top k meters, as data and in a 
    string to be used for logging.

    Required args
    -------------
    - losses : AverageMeter
        Average meter object for the loss data.
    - topk_meters : list
        List of average meter objects for each of the top k values.

    Optional args
    -------------
    - ks : list (default=[1, 3, 5])
        Top k values corresponding to the top k meters.
    - local : bool (default=False)
        If True, the local statistic is returned.
    - incl_last : bool (default=False)
        If True, the latest loss and accuracy values are returned and included 
        in the log string.
    - chance : float (default=None)
        Chance value to include in the log.

    Returns
    -------
    - loss_avg : float
        Average loss value.
    - acc_avg : float
        Average top 1 accuracy value.
    - log_str : str
        Statistics string to log.

    if last:
    - loss_val : float
        Latest loss value.
    - acc_val : float
        Latest top 1 accuracy value.
    """
    
    if len(topk_meters) != len(ks):
        raise ValueError(
            "Must provide as many topK meter objects as 'ks'."
            )

    avg_type = "local_avg" if local else "avg"

    topk_str = ", ".join(
        [f"top{k}: {100 * getattr(topk_meter, avg_type):05.2f}%" 
        for (k, topk_meter) in zip(ks, topk_meters)]
        )

    if 1 not in ks:
        raise ValueError("To calculate accuracy, ks must include 1.")
    accuracies = topk_meters[ks.index(1)]

    loss_avg = getattr(losses, avg_type)
    acc_avg = getattr(accuracies, avg_type)

    loss_val = losses.val
    acc_val = accuracies.val

    chance_str = ""
    if chance is not None:
        chance_str = f" (chance: {100 * chance:05.2f}%)"

    if incl_last:
        log_str = (
            f"Loss: {loss_val:.6f} (avg: {loss_avg:.4f}){TAB}"
            f"Acc{chance_str}: {100 * acc_val:07.4f}% (avg {topk_str})"
            )
        return loss_avg, acc_avg, log_str, loss_val, acc_val
    else:
        log_str = (
            f"Loss: {loss_avg:.4f}{TAB}"
            f"Acc{chance_str}: {topk_str}"
            )
        return loss_avg, acc_avg, log_str


#############################################
def get_dim_per_GPU(output, main_shape):
    """
    get_dim_per_GPU(output, main_shape)

    Returns the original shape of the output tensor's second dimension from the 
    original shape of the first dimension. 

    Required args
    -------------
    - output : 2D Tensor
        Output, with dimensions B * P * D_out2 x B_per * P * D_out2
    - main_shape : tuple
        Original shape of the first dimension of output (B, P, D_out2).

    Returns
    -------
    - dim_per_GPU : tuple
        Inferred original shape of the second dimension of output 
        (B_per, P, D_out2).
    """

    if len(main_shape) != 3:
        raise ValueError(
            ("Expected 'main_shape' to have 3 dimensions (B x P x D_out2), "
            f"but found {len(main_shape)}.")
        )

    B, P, D_out2 = main_shape
    if output.shape[0] != np.product(main_shape):
        raise ValueError(
            "The first dimension of 'output' should be the product of "
            "'main_shape'."
            )

    B_per = output.shape[1] / (P * D_out2)
    if B_per != int(B_per):
        raise RuntimeError(
            "Could not compute 'B_per' (number of examples per GPU)."
            )

    B_per = int(B_per)
    dim_per_GPU = (B_per, P, D_out2)

    return dim_per_GPU


#############################################
def calc_spatial_avg(values, main_shape):
    """
    calc_spatial_avg(values, main_shape)

    Calculates a spatial average over values, based on the shape information 
    provided.

    Required args
    -------------
    - values : 1 or 2D Tensor
        Values for which to compute the average along the spatial dimension, 
        where the first dimension can be reshaped using main_shape.
    - main_shape : tuple
        Original shape of the first dimension of values (B, P, D_out2).

    Returns
    -------
    - values : 1 or 2D Tensor
        Values after average has been taken along the spatial dimension.
    """

    if len(main_shape) != 3:
        raise ValueError("'main_shape' should comprise 3 values: (B, P, D_out2).")
    
    B, P, D_out2 = main_shape

    if len(values.shape) == 2: # output
        B_per, _, _ = get_dim_per_GPU(values, main_shape)
        reshape_tuple = (B, P, D_out2, B_per, P, D_out2)

        # take spatial mean (across D_out2 dimension)
        values = values.reshape(reshape_tuple).mean(axis=(2, 5))
        values = values.reshape(B * P, B_per * P)

    elif len(values.shape) == 1: # target
        if np.product(values.shape) != np.product(main_shape):
            raise ValueError(
                "The shape of 'values' should be the product of 'main_shape'."
                )
        # eliminate the D_out2 dimension
        values = values.reshape(main_shape).float()[..., 0] / D_out2
        values = values.reshape(B * P) 

    else:
        raise ValueError("Expected 'values' to have 1 or 2 dimensions.")

    return values


#############################################
def calc_chance(output, main_shape, spatial_avg=False):
    """
    calc_chance(output, main_shape)

    Calculates chance value from the output shape.

    Required args
    -------------
    - output : 2D Tensor
        Model output tensor, with dimensions B x number of classes.

    Optional args
    -------------
    - main_shape : tuple (default=None)
        If None, specifies the unflatted shape of the first dimension of the 
        output tensor (B, P, D_out2).
    - spatial_avg : bool (default=False)
        If True, chance is calculated after first collapsing the 
        spatial dimension of the output tensor, identified using main_shape.

    Returns
    -------
    - chance : float
        Chance value, computed from output dimensions.
    """

    dims_per_GPU = get_dim_per_GPU(output, main_shape)
    if spatial_avg:
        dims_per_GPU = dims_per_GPU[:1] # remove D_out2 dim
    chance = 1 / np.product(dims_per_GPU)

    return chance


#############################################
def get_predictions(output, keep_topk=1, spatial_avg=False, main_shape=None):
    """
    get_predictions(output)

    Returns predictions based on an output tensor.

    Required args
    -------------
    - output : 2D Tensor
        Model output tensor, with dimensions B x number of classes.

    Optional args
    -------------
    - keep_topk : int (default=1)
        Maximum top k values for which to retain class predictions.
    - spatial_avg : bool (default=False)
        If True, class predictions are calculated after first taking the 
        average along the spatial dimension of the output tensor, identified 
        using main_shape.
    - main_shape : tuple (default=None)
        If None, specifies the unflatted shape of the first dimension of the 
        output tensor, used to average across the spatial dimension 
        (B, P, D_out2). Required if spatial_avg is True. 
    
    Returns
    -------
    - pred : 2D Tensor
        Predictions, with dimensions number of items x keep_topk
    """

    if spatial_avg:
        if main_shape is None:
            raise ValueError(
                "If 'spatial_avg' is True, must pass 'main_shape'."
                )

        output = calc_spatial_avg(output, main_shape)

    _, pred = output.topk(keep_topk, 1, True, True)
    pred = pred.t()

    return pred


#############################################
def calc_topk_accuracy(output, target, ks=[1, 3, 5], spatial_avg=False, 
                       main_shape=None):
    """
    calc_topk_accuracy(output, target)

    Calculates top k accuracies for a model's output and target.

    Modified from: 
    https://gist.github.com/agermanidis/275b23ad7a10ee89adccf021536bb97e
    Given predicted and ground truth labels, calculate top-k accuracies.

    Required args
    -------------
    - output : 2D Tensor
        Model output tensor, with dimensions B x number of classes.
    - target : 1 or 2D Tensor
        Model target tensor, with dimensions B (x number of classes).

    Optional args
    -------------
    - ks : list (default=[1, 3, 5])
        Top k values for which to compute accuracies.
    - spatial_avg : bool (default=False)
        If True, accuracies are calculated after first taking the average along 
        the spatial dimension of target and output tensors, identified using 
        main_shape.
    - main_shape : tuple (default=None)
        If None, specifies the unflatted shape of the first dimension of the 
        output and target tensors, used to average across the spatial 
        dimension (B, P, D_out2). Required if spatial_avg is True. 
    
    Returns
    -------
    - accuracies : list
        Accuracies for each top k value.
    """

    pred = get_predictions(
        output, keep_topk=max(ks), spatial_avg=spatial_avg, 
        main_shape=main_shape
        )

    if spatial_avg:
        target = calc_spatial_avg(target, main_shape)

    batch_size = target.size(0)

    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    accuracies = []
    for k in ks:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        accuracies.append(correct_k.mul_(1 / batch_size))
    
    return accuracies


#############################################
def update_topk_meters(topk_meters, output, target, ks=[1, 3, 5], 
                       spatial_avg=False, main_shape=None):
    """
    update_topk_meters(topk_meters, output, target)

    Updates top k meters, in place.

    Required args
    -------------
    - topk_meters : list
        List of AccuracyMeter objects for each top k value.
    - output : 2D Tensor
        Model output tensor, with dimensions B x number of classes.
    - target : 1 or 2D Tensor
        Model target tensor, with dimensions B (x number of classes).

    Optional args
    -------------
    - ks : list (default=[1, 3, 5])
        Top k values paired to topk_meters.
    - spatial_avg : bool (default=False)
        If True, accuracies are calculated after first taking the average along 
        the spatial dimension of target and output tensors, identified using 
        main_shape.
    - main_shape : tuple (default=None)
        If None, specifies the unflatted shape of the first dimension of the 
        output and target tensors, used to average across the spatial 
        dimension. Required if spatial_avg is True. 
    """
    
    if len(topk_meters) != len(ks):
        raise ValueError(
            "Must provide as many topK meter objects as 'ks'."
            )

    top_vals = calc_topk_accuracy(
        output, target, ks, spatial_avg=spatial_avg, main_shape=main_shape
        )

    for t, top_val in enumerate(top_vals):
        topk_meters[t].update(top_val.item(), len(output))


#############################################
def calc_accuracy(output, target):
    """
    calc_accuracy(output, target)

    Calculates accuracy for one-hot data.

    Required args
    -------------
    - output : n+1 d Tensor
        Output values, interpreted as logits, with dimensions 
        B x number of classes.
    - target : nd Tensor
        Target class indices, with dimensions B.

    Returns
    -------
    - acc : float
        Accuracy.
    """

    target = target.squeeze()
    _, pred = torch.max(output, -1)

    if pred.shape != target.shape:
        raise RuntimeError(
            "'pred', calculated from 'output', should have the same shape as "
            "'target', once squeezed."
            )

    acc = torch.mean((pred == target).float())

    return acc


#############################################
def calc_accuracy_binary(output, target):
    """
    calc_accuracy_binary(output, target)

    Calculates accuracy from binarized data.

    Required args
    -------------
    - output : nd Tensor
        Output values, interpreted as logits, and binarized, with dimensions 
        B x number of classes.
    - target : nd Tensor
        Binary target values against which to compare output values, with 
        dimensions B x number of classes.

    Returns
    -------
    - acc : float
        Accuracy.
    """

    if output.shape != target.shape:
        raise ValueError("'output' and 'target' must have the same shape.")

    pred = output > 0
    acc = torch.mean((pred == target.byte()).float())
    return acc


#############################################
def get_best_acc(best_acc=None, save_best=True):
    """
    get_best_acc()

    Returns a best accuracy value to which to compare new accuracy values.

    Optional args
    -------------
    - best_acc : float (default=None)
        Initial best accuracy value based on which to calculate output. 
        If None, the output is set to -np.inf, unless save_best is False.
    - save_best : bool (default=True)
        If False, the output best accuracy is set to None. 

    Returns
    -------
    - best_acc : float
        Best accuracy value to use in comparison.
    """
    
    if save_best:
        best_acc = -np.inf if best_acc is None else best_acc
    else:
        best_acc = None

    return best_acc


##############################################
def init_loss_dict(dataset, direc=".", ks=[1, 3, 5], val=True, 
                   supervised=False, save_by_batch=False, light=True):
    """
    init_loss_dict(dataset)

    Initializes a loss dictionary, either from an existing file or from 
    scratch.

    Required args
    -------------
    - dataset : torch data.Dataset object
        Torch dataset object.
  
    Optional args
    -------------
    - direc : str or path (default=".")
        Directory from which to load loss dictionary, if it exists.
    - ks : list (default=[1, 3, 5])
        Top k accuracy values for which to include keys.
    - val : bool (default=True)
        If True, a validation sub-dictionary is included.
    - supervised : bool (default=False)
        If True, a confusion matrix key is created under the 'val' key, if val 
        is True.
    - save_by_batch : bool (default=False)
        If True, keys for individual batch data are included in the dictionary. 
    - light : bool (default=True)
        If True, keys for heavier data are excluded.

    Returns
    -------
    - loss_dict : dict
        Loss dictionary with the following keys:
        'train': sub-dictionary for the train mode, with keys:
            'acc'    : for accuracy values
            'epoch_n': for epoch numbers
            'loss'   : for loss values
            'top{k}' : for each set of top k values
            
            if save_by_batch:
            'avg_loss_by_batch'  : for average loss values for each batch
            'batch_epoch_n'      : for epoch numbers for batch data
            'loss_by_item'       : for loss values for each batch item
            'sup_target_by_batch': for supervised targets for each batch
            
                and if not light:
                'output_by_batch': for output values for each batch
                'target_by_batch': for target values for each batch
            
            if is_gabor:
            'unexp'          : for tracking whether epoch contained unexpected 
                               events
            'gabor_loss_dict': for gabor loss keys 
                               (see gabor_utils.init_gabor_records()).
            'gabor_acc_dict' : for gabor accuracy keys 
                               (see gabor_utils.init_gabor_records()).
        
        if val:
        'val': sub-dictionary for the val mode, with the same keys as 'train', 
               and, if supervised,
            'confusion_mat': for confusion matrices
    """

    loss_dict_path = Path(direc, "loss_data.json")

    if loss_dict_path.is_file():
        with open(loss_dict_path, "r") as f:
            loss_dict = json.load(f)
    else:
        loss_dict = dict()

    main_keys = ["train", "val"] if val else ["train"]
    
    topk_keys = [f"top{k}" for k in ks]
    shared_keys = ["epoch_n", "loss", "acc"] + topk_keys

    batch_keys = []
    if save_by_batch:
        batch_keys = [
            "batch_epoch_n", "avg_loss_by_batch", "loss_by_item", 
            "sup_target_by_batch"
            ]
        if not light:
            batch_keys.extend(["output_by_batch", "target_by_batch"])

    dataset_keys = []
    is_gabor = hasattr(dataset, "num_gabors")
    if is_gabor:
        from utils.gabor_utils import init_gabor_records
        gabor_loss_dict, gabor_acc_dict = init_gabor_records(dataset)
        dataset_keys = ["unexp", "gabor_loss_dict", "gabor_acc_dict"]

    for main_key in main_keys:
        if main_key not in loss_dict.keys():
            loss_dict[main_key] = dict()
        sub_keys = shared_keys + batch_keys + dataset_keys
        if main_key == "val" and supervised:
            sub_keys = sub_keys + ["confusion_mat"]
        for key in sub_keys:
            if key not in loss_dict[main_key].keys():
                if key == "gabor_loss_dict":
                    loss_dict[main_key][key] = copy.deepcopy(gabor_loss_dict)
                elif key == "gabor_acc_dict":
                    loss_dict[main_key][key] = copy.deepcopy(gabor_acc_dict)
                else:
                    loss_dict[main_key][key] = list()
    
    return loss_dict


#############################################
def populate_loss_dict(src_dict, target_dict, append_conf_matrices=True, 
                       is_best=False, gabor_unexp=None):
    """
    populate_loss_dict(src_dict, target_dict)

    Populates the target dictionary by appending data from the source 
    dictionary to each key shared by both dictionaries. 

    Required args
    -------------
    - src_dict : dict
        Dictionary from which to get data to store in the target dictionary.
        See init_loss_dict() for key details.
    - target_dict : dict
        Dictionary in which to store data. See init_loss_dict() for key details.

    Optional args
    -------------
    - append_conf_matrices : bool (default=True)
        If True, confusion matrix data is appended. Otherwise, it replaces any 
        previously stored value under the 'confusion_mat' key. Only applies 
        if the 'confusion_mat' key is present in the source and target 
        dictionary. 
    - is_best : bool (default=False)
        If True, epoch number is stored under the 'epoch_n_best' key in the 
        target dictionary. If the 'confusion_mat' key is present in the source 
        dictionary, the confusion matrix data is also saved under the 
        'confusion_mat_best'.
    - gabor_unexp : bool
        Whether the model, if trained on the Gabor dataset, was presented with 
        unexpected sequences. If so, the "_unexp" suffix is added to the 
        'confusion_mat_best' and 'epoch_n_best' keys.
    """

    confusion_mat_key = "confusion_mat"

    for loss_sub_dict in [src_dict, target_dict]:
        if "train" in loss_sub_dict.keys() or "val" in loss_sub_dict.keys():
            raise ValueError(
                "'src_dict' and 'target_dict' should be mode dictionaries, "
                "and not contain 'train' or 'val' keys."
                )

    for key, value in src_dict.items():
        if key not in target_dict.keys():
            continue
        if key == confusion_mat_key and not append_conf_matrices:
            continue
         # append the full dictionary, only if it's the confusion matrix
        if key == confusion_mat_key or not isinstance(value, dict):
            target_dict[key].append(src_dict[key])  
        else:
            if not isinstance(target_dict[key], dict):
                raise ValueError(
                    "'src_dict' and 'target_dict' structures "
                    "do not match."
                    )
            populate_loss_dict(value, target_dict[key])

    # add best epoch information
    gab_unexp_str = "_unexp" if gabor_unexp else ""
    if is_best:
        target_dict[f"epoch_n_best{gab_unexp_str}"] = src_dict["epoch_n"]

    # if not appending, replace stored confusion matrix, if applicable
    if confusion_mat_key in src_dict:
        if not append_conf_matrices:
            target_dict[confusion_mat_key] = src_dict[confusion_mat_key]
        if is_best:
            target_dict[f"confusion_mat_best{gab_unexp_str}"] = \
                src_dict[confusion_mat_key]


                            
#############################################
def plot_from_loss_dict(loss_dict, output_dir=".", seed=None, dataset="UCF101", 
                        num_classes=None, log_best=False):
    """
    plot_from_loss_dict(loss_dict)

    Plots loss data from loss dictionary.

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
            epochs x batches x B x N (x L x [image type, mean ori] 
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
    - output_dir : str or path (default=".")
        Directory in which to save loss data plot, if applicable.
    - seed : int (default=None)
        Seed used for model training, if applicable. Used in the plot title.
    - dataset : str (default="UCF101")
        Dataset name.
    - num_classes : int (default=None)
        Number of classes to use to compute chance in plotting loss, if 
        applicable.
    - log_best : bool (default=False)
        If True, the values for the best epoch are logged for the validation 
        set.
    """

    dataset = misc_utils.normalize_dataset_name(dataset)

    savename = "loss_data"

    seed_str_pr = "" if seed is None else f" (seed {seed})"
    unexp_str_pr = ""
    if dataset == "Gabors":
        unexp_str_pr = plot_utils.get_unexp_epoch_ranges(
            loss_dict["train"]["unexp"], loss_dict["train"]["epoch_n"], 
            as_str=True
            )
        if len(unexp_str_pr):
            unexp_str_pr = f" ({unexp_str_pr})"

    plot_what_vals = ["main"]
    if "avg_loss_by_batch" in loss_dict["train"].keys():
        plot_what_vals.append("by_batch")
    if "gabor_loss_dict" in loss_dict["train"].keys():
        plot_what_vals.append("by_gabor")

    for plot_what in plot_what_vals:
        use_log_best = log_best
        ext_str, ext_str_pr = "", ""
        y = 0.98
        if plot_what in ["by_batch", "by_gabor"]:
            use_log_best = False
            ext_str = f"_{plot_what}"
            if plot_what == "by_batch":
                y = 0.95
                ext_str_pr = " (average by batch)"
            else:
                y = 0.92
                ext_str_pr = " (split by class value comp.)"            

        fig = plot_utils.plot_from_loss_dict(
            loss_dict, num_classes=num_classes, dataset=dataset, 
            plot_what=plot_what, log_best=use_log_best
            )
        title = (f"{dataset[0].capitalize()}{dataset[1:]} dataset"
            f"{unexp_str_pr}{seed_str_pr}{ext_str_pr}")
        fig.suptitle(title, y=y)

        full_path = Path(output_dir, f"{savename}{ext_str}.svg")
        fig.savefig(full_path, bbox_inches="tight")


#############################################
def save_loss_dict(loss_dict, output_dir=".", seed=None, dataset="UCF101", 
                   num_classes=None, plot=True, log_best=False):
    """
    save_loss_dict(loss_dict)

    Saves a loss dictionary, and optionally plots loss data.

    Before saving the dictionary, checks for potential problems in the file.

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
            epochs x batches x B x N (x L x [image type, mean ori] 
            if Gabor dataset).
        
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
    - output_dir : str or path (default=".")
        Directory in which to save loss data and plot, if applicable.
    - seed : int (default=None)
        Seed used for model training, if applicable. Used in the plot title.
    - dataset : str (default="UCF101")
        Dataset name.
    - num_classes : int (default=None)
        Number of classes to use to compute chance in plotting loss, if 
        applicable.
    - plot : bool (default=True)
        If True, loss dictionary data is plotted.
    - log_best : bool (default=False)
        If True, the values for the best epoch are logged for the validation 
        set. Only applies if plot True.
    """
    
    full_path = Path(output_dir, f"loss_data.json")

    # check that there are a consistent number of batches per epoch, if recorded
    for subdict in loss_dict.values():
        if "avg_loss_by_batch" in subdict.keys():
            vals_count = subdict["avg_loss_by_batch"]
        elif "gabor_loss_dict" in subdict.keys():
            vals_count = subdict["gabor_loss_dict"]["overall"]
        else:
            continue
        num_batches_per_unique = []
        for vals in vals_count:
            n_vals_str = str(len(vals))
            if n_vals_str not in num_batches_per_unique:
                num_batches_per_unique.append(n_vals_str)
        if len(num_batches_per_unique) > 1:
            raise RuntimeError(
                "Found an inconsistent number of batches per epoch "
                f"({', '.join(num_batches_per_unique)}). This is likely due "
                "to resuming training with a different number of devices, "
                "leading to a change in batch size and number of batches per "
                "epoch. To address this, consider resuming with the same "
                "number of devices as before, or copying the model (and "
                "optionally the associated hyperparameters.json file) into a "
                "new folder, so that the existing loss dictionary is not "
                "reloaded."
                )

        
    with open(full_path, "w") as f:
        json.dump(loss_dict, f)

    if plot:
        plot_from_loss_dict(
            loss_dict,
            output_dir=output_dir,
            seed=seed,
            dataset=dataset,
            num_classes=num_classes,
            log_best=log_best
        )


#############################################
def get_loss_plot_kwargs(hp_dict, num_classes=None):
    """
    get_loss_plot_kwargs(hp_dict)

    Returns a dictionary of arguments for plotting loss with 
    plot_from_loss_dict(), excluding loss_dict and output_dir.

    Required args
    -------------
    - hp_dict : dict
        Nested hyperparameters dictionary.

    Optional args
    -------------
    - num_classes : int (default=None)
        Number of classes to use to compute chance in plotting loss, if 
        applicable. If None and hp_dict["model"]["supervised"] is True, it is 
        inferred from the hyperparameters.

    Returns
    -------
    - loss_plot_kwargs : dict
        Keyword arguments for plotting loss.
    """
    
    loss_plot_kwargs = dict()
    dataset = hp_dict["dataset"]["dataset"]
    dataset = misc_utils.normalize_dataset_name(dataset)
    loss_plot_kwargs["dataset"] = dataset

    if num_classes is None and hp_dict["model"]["supervised"]:
        if dataset == "Gabors":
            from utils.gabor_utils import NUM_MEAN_ORIS, get_num_classes
            num_classes = get_num_classes(
                num_mean_oris=NUM_MEAN_ORIS,
                gray=hp_dict["dataset"]["gray"],
                U_prob=hp_dict["dataset"]["U_prob"],
                diff_U_possizes=hp_dict["dataset"]["diff_U_possizes"]
            )
            logger.warning(
                f"Inferred number of Gabor classes: {num_classes}."
                )
        else:
            num_classes = misc_utils.get_num_classes(dataset)

    elif num_classes is not None:
        num_classes = int(num_classes)
    
    loss_plot_kwargs["num_classes"] = num_classes

    return loss_plot_kwargs


#############################################
def plot_conf_mat(conf_mat, mode="val", suffix=None, epoch_n=None, 
                  output_dir=".", **plot_mat_kwargs):
    """
    plot_conf_mat(conf_mat)

    Plots and saves the confusion matrix provided.
    
    Required args
    -------------
    - conf_mat : ConfusionMeter object
        ConfusionMeter object.

    Optional args
    -------------
    - mode : str (default="val")
        Mode to which the confusion meter is tied, used in the title and save 
        name.
    - suffix : str (default=None)
        Suffix to include in the save name only, if applicable.
    - epoch_n : int (default=None)
        Epoch number to include in the title only.
    - output_dir : str or path (default=".")
        Directory in which the plot will be saved.

    Keyword args
    ------------
    **plot_mat_kwargs : dict
        Additional confusion matrix plotting arguments for conf_mat.plot_mat()
    """

    suffix_str = misc_utils.format_addendum(suffix, is_suffix=True)

    U_prob_str = ""
    if hasattr(conf_mat, "unexp") and conf_mat.unexp:
        suffix_str = f"{suffix_str}_unexp"
        
        from utils.gabor_utils import get_U_prob_str
        U_prob_str = get_U_prob_str(conf_mat.U_prob, conf_mat.unexp)

    save_name = f"{mode}_confusion_mat{suffix_str}.svg"
    conf_mat_path = Path(output_dir, save_name)

    if suffix == "best":
        sub_str = f"{mode}, epoch {epoch_n}" if epoch_n is not None else mode
        title=f"Best model ({sub_str}{U_prob_str})"
    else:
        if epoch_n is None:
            title = mode.capitalize()
        else:
            title = f"Epoch {epoch_n} ({mode}{U_prob_str})" 
    
    conf_mat.plot_mat(conf_mat_path, title=title, **plot_mat_kwargs)


#############################################
def load_plot_conf_mat(conf_mat_dict, mode="val", suffix=None, 
                       omit_class_names=False, epoch_n=None, output_dir="."):
    """
    load_plot_conf_mat(conf_mat_dict)

    Loads and plots a ConfusionMeter from the input dictionary.

    Required args
    -------------
    - conf_mat_dict : dict
        ConfusionMeter storage dictionary.

    Optional args
    -------------
    - mode : str (default="val")
        Mode to which the confusion meter is tied, used in the title and save 
        name for the plot.
    - suffix : str (default=None)
        Suffix to include in the save name for the plot, if applicable.
    - omit_class_names : bool (default=False)
        If True, class names are omitted in plotting the confusion matrix. This 
        can save time, as plotting all the class names can take several minutes 
        if there are hundreds of them. 
    - epoch_n : int (default=None)
        Epoch number to include in the title only.
    - output_dir : str or path (default=".")
        Directory in which the plot will be saved.
    """

    if not isinstance(conf_mat_dict, dict):
        raise RuntimeError(f"Expected {conf_mat_dict} to be a dictionary.")
    
    plot_mat_kwargs = dict()
    if "unexp" in conf_mat_dict.keys():
        from utils.gabor_utils import GaborConfusionMeter
        conf_mat = GaborConfusionMeter(
            conf_mat_dict["class_names"],
            U_prob=conf_mat_dict["U_prob"],
            unexp=conf_mat_dict["unexp"]
            )
    else:
        conf_mat = ConfusionMeter(conf_mat_dict["class_names"])
        plot_mat_kwargs["incl_class_names"] = not(omit_class_names)

    conf_mat.load_from_storage_dict(conf_mat_dict)

    plot_conf_mat(
        conf_mat, mode=mode, suffix=suffix, epoch_n=epoch_n, 
        output_dir=output_dir, **plot_mat_kwargs
        )


#############################################
def plot_best_conf_mat(loss_sub_dict, mode="val", omit_class_names=True, 
                       output_dir="."):
    """
    plot_best_conf_mat(loss_sub_dict)
    """
    
    if "train" in loss_sub_dict.keys() or "val" in loss_sub_dict.keys():
        raise ValueError(
            "'loss_sub_dict' should be mode dictionaries, and not contain "
            "'train' or 'val' keys."
            )

    for key in ["confusion_mat_best", "confusion_mat_best_unexp"]:
        epoch_key = key.replace("confusion_mat", "epoch_n")
        
        if key in loss_sub_dict.keys():
            best_epoch_n = loss_sub_dict[epoch_key]
            load_plot_conf_mat(
                loss_sub_dict[key], mode=mode, suffix="best", 
                omit_class_names=omit_class_names, epoch_n=best_epoch_n, 
                output_dir=output_dir
                )


#############################################
def plot_conf_mats(loss_dict, epoch_n=-1, omit_class_names=False, 
                   output_dir="."):
    """
    plot_conf_mats(loss_dict)

    Plots confusion matrices from loss dictionary.

    Required args
    -------------
    - loss_dict : dict
        Dictionary recording loss and accuracy information in dictionaries 
        stored under the 'train' and optionally the 'val' key, each with keys:
        
        'acc' (list)    : Final local accuracy average per epoch.
        'epoch_n' (list): Epoch number per epoch.
        
        if 'val' key:
        'confusion_mat' (list or dict): Confusion matrix storage 
            dictionaries either for the final epoch or all epochs.
        
        and optionally:
        'confusion_mat_best' (dict)      : Confusion matrix storage 
            dictionaries for the best epoch.
        'confusion_mat_best_unexp' (dict): Confusion matrix storage 
            dictionaries for the best epoch, when Gabors dataset was set to 
            include unexpected sequences.
        'epoch_n_best' (int)                : Best epoch number.
        'epoch_n_best_unexp' (int)          : Best epoch number, when Gabors 
            dataset was set to include unexpected sequences.

    Optional args
    -------------
    - epoch_n : int (default=-1)
        Epoch number for which to plot confusion matrix.
    - omit_class_names : bool (default=False)
        If True, class names are omitted in plotting the confusion matrix. This 
        can save time, as plotting all the class names can take several minutes 
        if there are hundreds of them.
    - output_dir : str or path (default=".")
        Directory in which the plot will be saved.    

    """
    
    for mode, mode_dict in loss_dict.items():
        if "confusion_mat" in mode_dict.keys():
            conf_mat_dict = mode_dict["confusion_mat"]
            if isinstance(conf_mat_dict, list):
                if epoch_n == -1:
                    epoch_n = len(conf_mat_dict) - 1
                    conf_mat_dict = conf_mat_dict[-1]
                    suffix = ""
                elif epoch_n in mode_dict["epoch_n"]:
                    idx = mode_dict["epoch_n"].index(epoch_n)
                    conf_mat_dict = conf_mat_dict[idx]
                    suffix = f"ep_{epoch_n}"
                else:
                    raise ValueError(
                        f"No confusion matrix found for epoch {epoch_n} "
                        f"(mode {mode})."
                        )
            load_plot_conf_mat(
                conf_mat_dict, mode=mode, suffix=suffix, epoch_n=epoch_n,
                omit_class_names=omit_class_names, output_dir=output_dir
                )

        plot_best_conf_mat(
            mode_dict, mode=mode, omit_class_names=omit_class_names, 
            output_dir=output_dir
            )


#############################################
def save_conf_mat_dict(conf_mat_dict, prefix=None, suffix=None, output_dir=".", 
                       overwrite=True):
    """
    save_conf_mat_dict(conf_mat_dict)

    Saves a ConfusionMeter storage dictionary as a json file.

    Required args
    -------------
    - conf_mat_dict : dict
        ConfusionMeter storage dictionary.

    Optional args
    -------------
    - prefix : str (default=None)
        Prefix with which to start save name, if applicable.
    - output_dir : str or path (default=".")
        Directory in which to save storage dictionary.
    - overwrite : bool (default=True)
        If True, and the confusion matrix json file exists, it is overwritten. 
        Otherwise, a unique file name is identified.
    """

    prefix_str = misc_utils.format_addendum(prefix, is_suffix=False)
    suffix_str = misc_utils.format_addendum(suffix, is_suffix=True)
    save_name = f"{prefix_str}confusion_mat_data{suffix_str}"

    save_path = misc_utils.get_unique_filename(
        Path(output_dir, f"{save_name}.json"), 
        overwrite=overwrite
        )

    with open(save_path, "w") as f:
        json.dump(conf_mat_dict, f)


#############################################
def load_confusion_mat_dict(prefix=None, suffix=None, output_dir="."):
    """
    load_confusion_mat_dict()

    Loads a ConfusionMeter storage dictionary dictionary from the output 
    directory.

    Optional args
    -------------
    - prefix : str (default=None)
        Prefix with which to start the save name, if applicable.
    - suffix : str (default=None)
        Suffix with which to end the save name, if applicable.
    - output_dir : str or path (default=".")
        Directory from which to load the storage dictionary.

    Returns
    -------
    - conf_mat_dict : dict
        ConfusionMeter storage dictionary.
    """

    prefix_str = misc_utils.format_addendum(prefix, is_suffix=False)
    suffix_str = misc_utils.format_addendum(suffix, is_suffix=True)
    save_name = f"{prefix_str}confusion_mat_data{suffix_str}"

    save_path = Path(output_dir, f"{save_name}.json")
    if save_path.is_file():
        with open(save_path, "r") as f:
            conf_mat_dict = json.load(f)
    else:
        raise OSError(f"{save_path} is not a file.")

    return conf_mat_dict


#############################################
def load_loss_hyperparameters(model_direc, suffix=None): 
    """
    load_loss_hyperparameters(model_direc)

    Loads loss and hyperparameter dictionaries.

    Required args
    -------------
    - model_direc : str or Path
        Model directory from which to retrieve hyperparameters.
    
    Optional args
    -------------
    - suffix : str (default=None)
        Suffix in the hyperparameters filename, if applicable
    
    Returns
    -------
    - loss_dict : dict
        Dictionary recording loss and accuracy information in dictionaries 
        stored under the 'train' and optionally the 'val' key, each with keys:

        'acc' (list)    : Final local accuracy average per epoch.
        'epoch_n' (list): Epoch number per epoch.
        'loss' (list)   : Final local loss average per epoch.
        'top{k}' (list) : Final local top k accuracy average per epoch.

        if 'val' key:
        'confusion_mat' (list or dict)   : Confusion matrix storage 
            dictionaries either for the final epoch or all epochs.

        and optionally:
        'confusion_mat_best' (dict)      : Confusion matrix storage 
            dictionaries for the best epoch.
        'confusion_mat_best_unexp' (dict): Confusion matrix storage 
            dictionaries for the best epoch, when Gabors dataset was set to 
            include unexpected sequences.
        'epoch_n_best' (int)                : Best epoch number.
        'epoch_n_best_unexp' (int)          : Best epoch number, when Gabors 
            dataset was set to include unexpected sequences.

        and optionally:
        'avg_loss_by_batch' (2D list)       : average loss values with dims:
                                              epochs x batches
        'batch_epoch_n' (2D list)           : epoch number, with dims:
                                              epochs x batches
        'loss_by_item' (4d list)            : loss values with dims: 
                                              epochs x batches x B x N
        'sup_target_by_batch' (4 or 6D list): supervised targets with dims: 
            epochs x batches x B x N (x L x [image type, mean ori] 
            if Gabor dataset).
        
        if Gabor dataset:
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
    
    - hp_dict : dict
        Nested hyperparameters dictionary.
    """

    if not Path(model_direc).is_dir():
        raise ValueError(f"{model_direc} is not a directory")

    hp_dict =  misc_utils.load_hyperparameters(model_direc, suffix)

    suffix_str = misc_utils.format_addendum(suffix, is_suffix=True)
    loss_dict_path = Path(model_direc, f"loss_data{suffix_str}.json")
    if not loss_dict_path.is_file():
        raise OSError(f"{loss_dict_path} does not exist.")
    
    with open(loss_dict_path, "r") as f:
        loss_dict = json.load(f)

    return loss_dict, hp_dict


#############################################
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # replot from loss dictionary
    parser.add_argument("--model_direc", default=None,
        help=("directory in which loss dictionary and hyperparameters are "
            "stored, for replotting loss"))
    parser.add_argument("--file_suffix", default="",
        help="suffix, if any, of the loss dictionary and hyperparameter files.")
    parser.add_argument("--num_classes", default=None,
        help=("number of classes to use for plotting chance. If None, it is "
            "inferred from the dataset name information, for supervised "
            "settings."))

    # replot confusion matrix
    parser.add_argument("--conf_mat_epoch_n", default=-1, type=int,
        help="epoch for which to plot confusion matrix, if applicable.")
    parser.add_argument("--omit_class_names", action="store_true", 
        help=("if True, class names are omitted (greatly reduces svg save "
            "time, if there are many classes)."))
    parser.add_argument("--test", action="store_true", 
        help="plot confusion matrix for test results")
    parser.add_argument("--conf_mat_only", action="store_true", 
        help=("if True, only the confusion matrices are replotted."))

    # general arguments
    parser.add_argument("--output_dir", default=None, 
        help=("directory in which to save files. If None, it is inferred from "
        "another path argument)"))
    parser.add_argument('--log_level', default='info', 
                        help='logging level, e.g., debug, info, error')
    args = parser.parse_args()

    misc_utils.get_logger_with_basic_format(level=args.log_level)

    if args.model_direc is not None:
        if args.output_dir is None:
            args.output_dir = args.model_direc

        if args.test:
            conf_mat_dict = load_confusion_mat_dict(
                prefix="test", 
                suffix=args.file_suffix,
                output_dir=args.model_direc
                )

            load_plot_conf_mat(
                conf_mat_dict,
                mode="test",
                epoch_n=0,
                suffix=args.file_suffix,
                omit_class_names=args.omit_class_names, 
                output_dir=args.output_dir
                )

        else:
            loss_dict, hp_dict = load_loss_hyperparameters(
                args.model_direc, 
                suffix=args.file_suffix
                )

            # plot and save loss
            if not args.conf_mat_only:
                loss_plot_kwargs = get_loss_plot_kwargs(
                    hp_dict, 
                    num_classes=args.num_classes
                    )

                plot_from_loss_dict(
                    loss_dict, output_dir=args.output_dir, log_best=True,
                    **loss_plot_kwargs,
                    )

            # plot and save confusion matrix data
            plot_conf_mats(
                loss_dict, 
                epoch_n=args.conf_mat_epoch_n, 
                omit_class_names=args.omit_class_names, 
                output_dir=args.output_dir
                )

    else:
        breakpoint()
