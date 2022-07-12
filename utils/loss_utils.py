#!/usr/bin/env python

import argparse
from collections import deque
import json
import logging
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch
import yaml

sys.path.extend(["..", str(Path("..", "utils"))])
from utils import misc_utils, plot_utils

logger = logging.getLogger(__name__)

TAB = "    "

#############################################
class AverageMeter(object):
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.local_history = deque([])
        self.local_avg = 0
        self.history = []
        self.dict = dict() # save all data values here
        self.save_dict = dict() # save mean and std here, for summary table

    def update(self, val, n=1, history=0, step=5):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if history:
            self.history.append(val)
        if step > 0:
            self.local_history.append(val)
            if len(self.local_history) > step:
                self.local_history.popleft()
            self.local_avg = np.average(self.local_history)

    def dict_update(self, val, key):
        if key in self.dict.keys():
            self.dict[key].append(val)
        else:
            self.dict[key] = [val]

    def __len__(self):
        return self.count


#############################################
class AccuracyTable(object):
    """Compute accuracy for each class"""

    def __init__(self):
        self.dict = dict()

    def update(self, pred, target):
        pred = torch.squeeze(pred)
        target = torch.squeeze(target)
        for i, j in zip(pred, target):
            i = int(i)
            j = int(j)
            if j not in self.dict.keys():
                self.dict[j] = {"count": 0, "correct": 0}
            self.dict[j]["count"] += 1
            if i == j:
                self.dict[j]["correct"] += 1

    def log_table(self, label):
        for key in self.dict.keys():
            correct = self.dict[key]["correct"]
            count = self.dict[key]["count"]
            acc = correct / count
            logger.info(
                f"{label}: {key:2}, accuracy: {correct:3}/{count:3} = {acc:.6f}"
                )


#############################################
class ConfusionMeter(object):
    """Compute and show confusion matrix"""

    def __init__(self, class_names=None, num_classes=5):
        self.reinitialize_values(class_names, num_classes=num_classes)

    def reinitialize_values(self, class_names=None, num_classes=5):
        if class_names is not None:
            num_classes = len(class_names)

        self.num_classes = num_classes
        self.set_class_names(class_names)
        self.mat = np.zeros((num_classes, num_classes)).astype(int)
        self.precision = []
        self.recall = []

    def set_class_names(self, class_names=None):
        if class_names is not None and (len(class_names) != self.num_classes):
            raise ValueError(
                f"Number of class names ({len(class_names)}) does not match "
                f"'self.num_classes' ({self.num_classes})"
                )
        self.class_names = class_names

    def update(self, pred, target):
        pred = np.squeeze(pred)
        target = np.squeeze(target)
        for p, t in zip(pred.flat, target.flat):
            self.mat[p][t] += 1

    def log_mat(self):
        logger.info(f"Confusion Matrix (target in columns):\n{self.mat}")

    def annotate(self, ax):
        height, width = self.mat.shape
        for x in range(width):
            for y in range(height):
                ax.annotate(
                    str(int(self.mat[x][y])), xy=(y+1, x+1),
                    horizontalalignment="center", 
                    verticalalignment="center", fontsize=8
                    )

    def get_label_dict(self):
        if self.class_names is None:
            raise ValueError(
                "Cannot include class names, as self.class_names is "
                "not set."
                )
        label_dict = {i: cl for i, cl in enumerate(self.class_names)}
    
        return label_dict


    def add_labels(self, ax, label_dict=None, secondary=False):
        height, width = self.mat.shape
        if label_dict is None:
            xtick_diff = min(np.diff(ax.get_xticks()))
            if xtick_diff < 1:
                ax.set_xticks(np.arange(width) + 1)
            ytick_diff = min(np.diff(ax.get_yticks()))
            if ytick_diff < 1:
                ax.set_yticks(np.arange(height) + 1)
        else:
            ticks = np.sort(list(label_dict.keys()))
            n_ticks = len(ticks)
            fontsize = 10
            fact = 8 if secondary else 13
            if n_ticks >= fact:
                fontsize = np.max([np.min([10, int(10 - n_ticks / fact)]), 1])

            ax_x, ax_y = ax, ax
            x_rotation = "vertical"
            if secondary:
                ax_x = ax.secondary_xaxis("top")
                ax_y = ax.secondary_yaxis("right")
                ax_x.tick_params(axis="x", top=False, pad=0.2)
                ax_y.tick_params(axis="y", right=False, pad=0.2)
                x_rotation = None

            ax_x.set_xticks(
                ticks + 1, [label_dict[i] for i in ticks], rotation=x_rotation, 
                fontsize=fontsize, 
                )
            
            ax_y.set_yticks(
                ticks + 1, [label_dict[i] for i in ticks], fontsize=fontsize, 
                )

    def add_colorbar(self, im, adj_aspect=True, **cbar_kwargs):
        """
        add_colorbar(im)
        """
        
        aspect_ratios = {
            "1+"    : 13,
            "10+"   : 18,
            "100+"  : 32,
            "1000+" : 140,
        }
        
        fig = im.figure
        if adj_aspect:
            # ensure that the colormap aspect ratio keeps the widths of the 
            # main plot and overall figure constant
            cbar_kwargs["aspect"] = aspect_ratios["1+"]
            new_aspect = cbar_kwargs["aspect"]
            im.autoscale() # fix the colorbar limits?
            clims = im.get_clim()
            im.set_clim(clims)

        for _ in range(4):
            cm = fig.colorbar(im, **cbar_kwargs)
            cm.set_label("Counts", rotation=270, labelpad=18)
            cm.update_ticks() # doesn't quite work
            cm.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            yticks = cm.ax.get_yticks()

            max_tick = max(yticks)
            if adj_aspect:
                for min_val in [1, 10, 100, 1000]:
                    if min_val > max(clims): 
                        break
                    if max_tick >= min_val:
                        new_aspect = aspect_ratios[f"{min_val}+"]

            if adj_aspect and (new_aspect != cbar_kwargs["aspect"]):
                cbar_kwargs["aspect"] = new_aspect
                cm.remove()
            else:
                break


    def plot_mat(self, save_path=None, incl_class_names=True, annotate=False, 
                 ax=None, title=None, vmax=None, adj_aspect=True, 
                 **cbar_kwargs):


        if vmax is not None and vmax < self.mat.max():
            raise ValueError("self.mat contains values higher than 'vmax'.")
        elif self.mat.max() < 1:
            vmax = 1

        if self.mat.min() < 0:
            raise ValueError("self.mat contains values below 0.")

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        im = ax.imshow(self.mat,
            cmap=plt.cm.viridis,
            interpolation=None,
            extent=(
                0.5, 
                np.shape(self.mat)[0] + 0.5, 
                np.shape(self.mat)[1] + 0.5, 
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
        self.add_colorbar(im, adj_aspect=adj_aspect, **cbar_kwargs)

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
        self.get_storage_dict(storage_dict)
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
    """
    
    if num_classes is not None:
        ks = [k for k in ks if k < num_classes]

    return ks


#############################################
def init_meters(n_topk=3):
    """
    init_meters()
    """
    
    losses = AverageMeter()
    topk_meters = [AverageMeter() for _ in range(n_topk)]

    return losses, topk_meters



#############################################
def get_criteria(criterion="cross-entropy", loss_weight=None, device="cpu"):
    """
    get_criteria()
    """

    if criterion == "cross-entropy":
        criterion_fct = torch.nn.CrossEntropyLoss
    else:
        raise NotImplementedError("Only 'cross-entropy' loss is implemented.")

    if loss_weight is not None:
        loss_weight = loss_weight.to(device)
    criterion = criterion_fct(weight=loss_weight)
    criterion_no_reduction = criterion_fct(
        weight=loss_weight, reduction="none"
        )

    return criterion, criterion_no_reduction


#############################################
def get_stats(losses, topk_meters, ks=[1, 3, 5], local=False, last=False, 
              chance=None):
    """
    get_stats(losses, topk_meters)
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

    if last:
        log_str = (
            f"Loss: {loss_val:.6f} (avg: {loss_avg:.4f}){TAB}"
            f"Acc{chance_str}: {100 * acc_val:07.4f}% (avg {topk_str})"
            )
        returns = loss_avg, acc_avg, loss_val, acc_val, log_str
    else:
        log_str = (
            f"Loss: {loss_avg:.4f}{TAB}"
            f"Acc{chance_str}: {topk_str}"
            )
        returns = loss_avg, acc_avg, log_str

    return returns


#############################################
def get_dim_per_GPU(output, main_shape):
    """
    get_dim_per_GPU(output, main_shape)
    """

    if len(main_shape) != 3:
        raise ValueError(
            ("Expected 'main_shape' to have 3 dimensions (B x PS x HW), "
            f"but found {len(main_shape)}.")
        )

    B, PS, HW = main_shape
    if output.shape[0] != np.product(main_shape):
        raise ValueError(
            "The first dimension of 'output' should be the product of "
            "'main_shape'."
            )

    B_per = output.shape[1] / (PS * HW)
    if B_per != int(B_per):
        raise RuntimeError(
            "Could not compute 'B_per' (number of examples per GPU)."
            )

    B_per = int(B_per)
    dim_per_GPU = (B_per, PS, HW)

    return dim_per_GPU


#############################################
def calc_spatial_avg(values, main_shape):
    """
    calc_spatial_avg(values, main_shape)
    """
    
    B, PS, HW = main_shape

    if len(values.shape) == 2: # output
        B_per, _, _ = get_dim_per_GPU(values, main_shape)
        reshape_tuple = (B, PS, HW, B_per, PS, HW)

        # take spatial mean (across HW dimension)
        values = values.reshape(reshape_tuple).mean(axis=(2, 5))
        values = values.reshape(B * PS, B_per * PS)

    elif len(values.shape) == 1: # target
        if np.product(values.shape) != np.product(main_shape):
            raise ValueError(
                "The shape of 'values' should be the product of 'main_shape'."
                )
        # eliminate the HW dimension
        values = values.reshape(main_shape).float()[..., 0] / HW
        values = values.reshape(B * PS) 

    else:
        raise ValueError("Expected 'values' to have 1 or 2 dimensions.")

    return values


#############################################
def calc_chance(output, main_shape, spatial_avg=False):
    """
    calc_chance(output, main_shape)
    """

    dims_per_GPU = get_dim_per_GPU(output, main_shape)
    if spatial_avg:
        dims_per_GPU = dims_per_GPU[:1] # remove HW dim
    chance = 1 / np.product(dims_per_GPU)

    return chance


#############################################
def get_predictions(output, keep_topk=1, spatial_avg=False, main_shape=None):
    """
    get_predictions(output)
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
def calc_topk_accuracy(output, target, topk=(1,), spatial_avg=False, 
                       main_shape=None):
    """
    calc_topk_accuracy(output, target)

    output dim are: B * PS * HW x B_per * PS * HW => (32, 1, 16) x (32, 1, 16)
    target dim are: B * PS * HW

    Modified from: 
    https://gist.github.com/agermanidis/275b23ad7a10ee89adccf021536bb97e
    Given predicted and ground truth labels, calculate top-k accuracies.
    """

    pred = get_predictions(
        output, keep_topk=max(topk), spatial_avg=spatial_avg, 
        main_shape=main_shape
        )

    if spatial_avg:
        target = calc_spatial_avg(target, main_shape)

    batch_size = target.size(0)

    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(1 / batch_size))
    return res


#############################################
def update_topk_meters(topk_meters, output, target, ks=[1, 3, 5], 
                       spatial_avg=False, main_shape=None):
    """
    update_topk_meters(topk_meters, output, target)
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

    output: (B, N); target: (B)
    """

    target = target.squeeze()
    _, pred = torch.max(output, 1)
    return torch.mean((pred == target).float())


#############################################
def calc_accuracy_binary(output, target):
    """
    calc_accuracy_binary(output, target)

    output, target: (B, N), output is logits, before sigmoid
    """

    pred = output > 0
    acc = torch.mean((pred == target.byte()).float())
    return acc


#############################################
def get_best_acc(best_acc=None, save_best=True):
    """
    get_best_acc()
    """
    
    if save_best:
        best_acc = -np.inf if best_acc is None else best_acc
    else:
        best_acc = None

    return best_acc


##############################################
def init_loss_dict(direc=".", ks=[1, 3, 5], val=True, supervised=False, 
                   save_by_batch=False, light=True):
    """
    init_loss_dict()
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

    for main_key in main_keys:
        if main_key not in loss_dict.keys():
            loss_dict[main_key] = dict()
        sub_keys = shared_keys + batch_keys
        if main_key == "val" and supervised:
            sub_keys = sub_keys + ["confusion_matrix"]
        for key in sub_keys:
            if key not in loss_dict[main_key].keys():
                loss_dict[main_key][key] = list()
    
    return loss_dict


#############################################
def populate_loss_dict(src_dict, target_dict, append_confusion_matrices=True, 
                       is_best=False):
    """
    populate_loss_dict(src_dict, target_dict)
    """

    conf_matrix_key = "confusion_matrix"

    for key, value in src_dict.items():
        if key not in target_dict.keys():
            continue
        if key == conf_matrix_key and not append_confusion_matrices:
            continue
         # append the full dictionary, only if it's the confusion matrix
        if key == conf_matrix_key or not isinstance(value, dict):
            target_dict[key].append(src_dict[key])  
        elif isinstance(value, dict):
            if not isinstance(target_dict[key], dict):
                raise ValueError(
                    "'src_dict' and 'target_dict' structures "
                    "do not match."
                    )
            populate_loss_dict(value, target_dict[key])

    # if not appending, check if it should be replaced
    if conf_matrix_key in src_dict:
        if not append_confusion_matrices:
            target_dict[conf_matrix_key] = src_dict[conf_matrix_key]
        if is_best:
            target_dict["confusion_matrix_best"] = src_dict[conf_matrix_key]

                            
#############################################
def save_loss_dict_plot(loss_dict, output_dir=".", seed=None, dataset="UCF101", 
                        unexp_epoch=10, num_classes=None):
    """
    save_loss_dict_plot(loss_dict)
    """

    dataset = misc_utils.normalize_dataset_name(dataset)

    savename = "loss_data"

    seed_str_pr = "" if seed is None else f" (seed {seed})"
    unexp_str_pr = ""
    if dataset == "Gabors":
        unexp_str_pr = f" (unexp. epoch: {unexp_epoch})"

    by_batches = [False]
    if "avg_loss_by_batch" in loss_dict["train"].keys():
        by_batches.append(True)

    for by_batch in by_batches:
        if by_batch:
            batch_str = "_by_batch"
            batch_str_pr = " (average by batch)"
        else:
            batch_str, batch_str_pr = "", ""

        fig = plot_utils.plot_loss_dict(
            loss_dict, num_classes=num_classes, dataset=dataset, 
            unexp_epoch=unexp_epoch, by_batch=by_batch
            )
        title = (f"{dataset[0].capitalize()}{dataset[1:]} dataset"
            f"{unexp_str_pr}{seed_str_pr}{batch_str_pr}")
        fig.suptitle(title)

        full_path = Path(output_dir, f"{savename}{batch_str}.svg")
        fig.savefig(full_path, bbox_inches="tight")


#############################################
def save_loss_dict(loss_dict, output_dir=".", seed=None, dataset="UCF101", 
                   unexp_epoch=10, num_classes=None, plot=True):
    """
    save_loss_dict(loss_dict)
    """
    
    full_path = Path(output_dir, f"loss_data.json")
        
    with open(full_path, "w") as f:
        json.dump(loss_dict, f)

    if plot:
        save_loss_dict_plot(
            loss_dict,
            output_dir=output_dir,
            seed=seed,
            dataset=dataset,
            unexp_epoch=unexp_epoch,
            num_classes=num_classes
        )


#############################################
def get_loss_plot_kwargs(hp_dict, num_classes=None):
    """
    get_loss_plot_kwargs(hp_dict)
    """
    
    loss_dict_kwargs = dict()
    dataset = hp_dict["dataset"]["dataset"]
    dataset = misc_utils.normalize_dataset_name(dataset)
    loss_dict_kwargs["dataset"] = dataset

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
            num_classes = misc_utils.get_num_classes(
                dataset_name=dataset
                )

    elif num_classes is not None:
        num_classes = int(num_classes)
    
    loss_dict_kwargs["num_classes"] = num_classes
    if "unexp_epoch" in hp_dict["dataset"].keys():
        loss_dict_kwargs["unexp_epoch"] = hp_dict["dataset"]["unexp_epoch"]

    return loss_dict_kwargs


#############################################
def plot_conf_mat(conf_mat, mode="val", suffix="", epoch_n=None, output_dir=".", 
                  **plot_mat_kwargs):
    """
    plot_conf_mat(conf_mat)
    """

    suffix_str = f"_{suffix}" if len(suffix) else ""
    save_name = f"{mode}_confusion_matrix{suffix_str}.svg"
    conf_mat_path = Path(output_dir, save_name)

    if suffix == "best":
        sub_str = f"{mode}, epoch {epoch_n}" if epoch_n is not None else mode
        title=f"Best model ({sub_str})"
    else:
        if epoch_n is None:
            title = mode.capitalize()
        else:
            title = f"Epoch {epoch_n} ({mode})" 
    
    conf_mat.plot_mat(conf_mat_path, title=title, **plot_mat_kwargs)


#############################################
def load_replot_conf_mat(conf_mat_dict, mode="val", suffix="", 
                         omit_class_names=False, epoch_n=None, output_dir="."):
    """
    load_replot_conf_mat(conf_mat_dict)
    """

    if not isinstance(conf_mat_dict, dict):
        raise RuntimeError(f"Expected {conf_mat_dict} to be a dictionary.")
    
    plot_mat_kwargs = dict()
    if "mean_oris" in conf_mat_dict.keys():
        from utils.gabor_utils import GaborConfusionMeter
        conf_mat = GaborConfusionMeter(conf_mat_dict["class_names"])
    else:
        conf_mat = ConfusionMeter(conf_mat_dict["class_names"])
        plot_mat_kwargs["incl_class_names"] = not(omit_class_names)

    conf_mat.load_from_storage_dict(conf_mat_dict)

    plot_conf_mat(
        conf_mat, mode=mode, suffix=suffix, epoch_n=epoch_n, 
        output_dir=output_dir, **plot_mat_kwargs
        )


#############################################
def plot_conf_mats(loss_dict, epoch_n=-1, omit_class_names=False, 
                   output_dir="."):
    """
    plot_conf_mats(loss_dict)
    """
    
    for mode, mode_dict in loss_dict.items():
        if "confusion_matrix" in mode_dict.keys():
            conf_mat_dict = mode_dict["confusion_matrix"]
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
            load_replot_conf_mat(
                conf_mat_dict, mode=mode, suffix=suffix, epoch_n=epoch_n,
                omit_class_names=omit_class_names, output_dir=output_dir
                )

        if "confusion_matrix_best" in mode_dict.keys():
            best_idx = np.argmax(mode_dict["acc"])
            best_epoch_n = mode_dict["epoch_n"][best_idx]

            load_replot_conf_mat(
                mode_dict["confusion_matrix_best"], 
                mode=mode, 
                suffix="best", 
                omit_class_names=omit_class_names,
                epoch_n=best_epoch_n,
                output_dir=args.output_dir
                )


#############################################
def save_confusion_mat_dict(conf_mat_dict, prefix=None, output_dir=".", 
                            overwrite=True):
    """
    save_confusion_mat_dict(conf_mat_dict)
    """

    save_name = "confusion_matrix_data.json"
    if prefix is not None and len(prefix):
        save_name = f"{prefix}_{save_name}"

    save_path = Path(output_dir, save_name)
    save_path = misc_utils.get_unique_filename(
        Path(output_dir, save_name), overwrite=overwrite
        )

    with open(save_path, "w") as f:
        json.dump(conf_mat_dict, f)


#############################################
def load_confusion_mat_dict(prefix=None, suffix="", output_dir="."):
    """
    load_confusion_mat_dict(conf_mat_dict)
    """

    save_name = "confusion_matrix_data.json"

    if prefix is not None and len(prefix):
        save_name = f"{prefix}_{save_name}"

    if suffix is not None and len(suffix):
        save_name = f"{prefix}_{save_name}_{suffix}"

    save_path = Path(output_dir, save_name)
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
    """

    if not Path(model_direc).is_dir():
        raise ValueError(f"{model_direc} is not a directory")

    suffix_str = ""
    if suffix is not None and len(suffix):
        suffix_str = f"_{suffix}"

    loss_dict_path = Path(model_direc, f"loss_data{suffix_str}.json")
    hp_dict_path = Path(model_direc, f"hyperparameters{suffix_str}.yaml")

    if not loss_dict_path.is_file():
        raise OSError(f"{loss_dict_path} does not exist.")
    if not hp_dict_path.is_file():
        raise OSError(f"{hp_dict_path} does not exist.")
    
    with open(loss_dict_path, "r") as f:
        loss_dict = json.load(f)
    with open(hp_dict_path, "r") as f:
        hp_dict = yaml.safe_load(f)

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
                prefix="test", output_dir=args.model_direc
                )

            load_replot_conf_mat(
                conf_mat_dict,
                mode="test",
                epoch_n=0,
                omit_class_names=args.omit_class_names, 
                output_dir=args.output_dir
                )

        else:
            loss_dict, hp_dict = load_loss_hyperparameters(
                args.model_direc, suffix=args.file_suffix
                )

            # plot and save loss
            if not args.conf_mat_only:
                loss_plot_kwargs = get_loss_plot_kwargs(
                    hp_dict, num_classes=args.num_classes
                    )

                save_loss_dict_plot(
                    loss_dict, output_dir=args.output_dir, **loss_plot_kwargs
                    )

            # plot and save confusion matrix
            plot_conf_mats(
                loss_dict, 
                epoch_n=args.conf_mat_epoch_n, 
                omit_class_names=args.omit_class_names, 
                output_dir=args.output_dir
                )

    else:
        breakpoint()
