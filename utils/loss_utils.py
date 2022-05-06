from collections import deque
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

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
        self.dict = {} # save all data values here
        self.save_dict = {} # save mean and std here, for summary table

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
        self.dict = {}

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

    def __init__(self, num_class):
        self.num_class = num_class
        self.mat = np.zeros((num_class, num_class))
        self.precision = []
        self.recall = []

    def update(self, pred, target):
        pred, target = pred.cpu().numpy(), target.cpu().numpy()
        pred = np.squeeze(pred)
        target = np.squeeze(target)
        for p, t in zip(pred.flat, target.flat):
            self.mat[p][t] += 1

    def log_mat(self):
        logger.info(f"Confusion Matrix (target in columns):\n{self.mat}")

    def plot_mat(self, path, dictionary=None, annotate=False):
        fig, ax = plt.subplots()
        im = ax.imshow(self.mat,
            cmap=plt.cm.jet,
            interpolation=None,
            extent=(
                0.5, 
                np.shape(self.mat)[0] + 0.5, 
                np.shape(self.mat)[1] + 0.5, 
                0.5
                )
            )
        width, height = self.mat.shape
        if annotate:
            for x in range(width):
                for y in range(height):
                    ax.annotate(
                        str(int(self.mat[x][y])), xy=(y+1, x+1),
                        horizontalalignment="center", 
                        verticalalignment="center", fontsize=8
                        )

        if dictionary is not None:
            ax.set_xticks(
                np.arange(width) + 1, [dictionary[i] for i in range(width)],
                rotation="vertical"
                )
            ax.set_yticks(
                np.arange(height) + 1, [dictionary[i] for i in range(height)]
                )
        ax.set_xlabel("Ground Truth")
        ax.set_ylabel("Prediction")
        
        cm = fig.colorbar(im)
        cm.set_label("Counts", rotation=270, labelpad=18)
        
        fig.savefig(path, format="svg", bbox_inches="tight", dpi=600)
        plt.close(fig)


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
def calc_topk_accuracy(output, target, topk=(1,)):
    """
    calc_topk_accuracy(output, target)

    Modified from: 
    https://gist.github.com/agermanidis/275b23ad7a10ee89adccf021536bb97e
    Given predicted and ground truth labels, calculate top-k accuracies.
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(1 / batch_size))
    return res


#############################################
def update_topk_meters(topk_meters, output, target, ks=[1, 3, 5]):
    """
    update_topk_meters(topk_meters, output, target)
    """
    
    if len(topk_meters) != len(ks):
        raise ValueError(
            "Must provide as many topK meter objects as 'ks'."
            )

    top_vals = calc_topk_accuracy(output, target, ks)

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
def init_loss_dict(direc=".", dataset="UCF101", ks=[1, 3, 5], val=True, 
                   save_by_batch=False):
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

    train_keys = []
    if save_by_batch:
        train_keys = [
            "batch_epoch_n", "detailed_loss", "loss_by_batch", "dot_by_batch", 
            "target_by_batch"
            ]

    for main_key in main_keys:
        if main_key not in loss_dict.keys():
            loss_dict[main_key] = dict()
        sub_keys = shared_keys
        if main_key == "train":
            sub_keys = shared_keys + train_keys
            if dataset == "Gabors":
                sub_keys.append("seq")
        for key in sub_keys:
            if key not in loss_dict[main_key].keys():
                loss_dict[main_key][key] = list()
    
    return loss_dict


#############################################
def plot_loss_dict(loss_dict, chance=None):
    """
    plot_loss_dict(loss_dict)
    """

    fig, ax = plt.subplots(2, figsize=[8.5, 6.5], sharex=True)

    colors = ["blue", "orange"]
    ls = ["dashed", None]

    for d, datatype in enumerate(["train", "val"]):
        if datatype not in loss_dict.keys():
            continue
        epoch_ns = loss_dict[datatype]["epoch_n"]
        ax[0].plot(
            epoch_ns, loss_dict[datatype]["loss"], 
            color=colors[d], ls=ls[d], label=f"{datatype} loss"
            )
        acc_keys = sorted(
            [key for key in loss_dict[datatype].keys() if "top" in key]
            )
        if chance is not None:
            ax[1].axhline(100 * chance, color="k", ls="dashed", alpha=0.8)
        for a, acc_key in enumerate(acc_keys):
            accuracies = 100 * np.asarray(loss_dict[datatype][acc_key])
            alpha = 0.7 ** a
            label = None
            if a == 0:
                label=f"{datatype} {', '.join(acc_keys)}"
            ax[1].plot(
                epoch_ns, accuracies, color=colors[d], ls=ls[d], 
                alpha=alpha, label=label
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
                    best_epoch_n, color="k", ls="dashed", alpha=0.8, label=label
                    )

    for sub_ax in ax.reshape(-1):
        sub_ax.legend(fontsize="small")
        sub_ax.spines["right"].set_visible(False)
        sub_ax.spines["top"].set_visible(False)

    ax[0].set_ylabel("Loss")    
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy (%)")

    return fig


#############################################
def save_loss_dict(loss_dict, output_dir=".", seed=None, dataset="UCF101", 
                   unexp_epoch=10, plot=True, chance=None):
    """
    save_loss_dict(loss_dict)
    """

    seed_str_pr = ""
    if seed is not None:
        seed_str_pr = f" (seed {seed})"

    unexp_str_pr = ""
    if dataset == "Gabors":
        unexp_str_pr = f" (unexp. epoch: {unexp_epoch})"

    savename = "loss_data"
    full_path = Path(output_dir, f"loss_data.json")
        
    with open(full_path, "w") as f:
        json.dump(loss_dict, f)

    if plot:
        fig = plot_loss_dict(loss_dict, chance=chance)
        title = (f"{dataset[0].capitalize()}{dataset[1:]} dataset"
            f"{unexp_str_pr}{seed_str_pr}")
        fig.suptitle(title)

        full_path = Path(output_dir, f"{savename}.svg")
        fig.savefig(full_path, bbox_inches="tight")

