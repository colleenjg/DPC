from collections import deque
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

logger = logging.getLogger(__name__)


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
    """compute accuracy for each class"""
    def __init__(self):
        self.dict = {}

    def update(self, pred, tar):
        pred = torch.squeeze(pred)
        tar = torch.squeeze(tar)
        for i, j in zip(pred, tar):
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

    def update(self, pred, tar):
        pred, tar = pred.cpu().numpy(), tar.cpu().numpy()
        pred = np.squeeze(pred)
        tar = np.squeeze(tar)
        for p,t in zip(pred.flat, tar.flat):
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
                    plt.annotate(str(int(self.mat[x][y])), xy=(y+1, x+1),
                                 horizontalalignment="center",
                                 verticalalignment="center",
                                 fontsize=8)

        if dictionary is not None:
            ax.set_xticks([i+1 for i in range(width)],
                          [dictionary[i] for i in range(width)],
                          rotation="vertical")
            ax.set_yticks([i+1 for i in range(height)],
                       [dictionary[i] for i in range(height)])
        ax.set_xlabel("Ground Truth")
        ax.set_ylabel("Prediction")
        fig.colorbar(im)
        fig.savefig(path, format="svg", bbox_inches="tight", dpi=600)
        plt.close(fig)


#############################################
def get_stats(losses, topk_meters, ks=[1, 3, 5], local=False, last=False):

    if len(topk_meters) != len(ks):
        raise ValueError(
            "Must provide as many topK meter objects as 'ks'."
            )

    avg_type = "local_avg" if local else "avg"

    topk_str = ", ".join(
        [f"top{k} {topk_meter.get(avg_type):.4f}" 
        for (k, topk_meter) in zip(ks, topk_meters)]
        )

    if 1 not in ks:
        raise ValueError("To calculate accuracy, ks must include 1.")
    accuracies = topk_meters[ks.index(1)]

    loss_avg = losses.get(avg_type)
    acc_avg = accuracies.get(avg_type)

    loss_val = losses.val
    acc_val = accuracies.val

    if last:
        log_str = (
            f"Loss: {loss_val:.6f} ({loss_avg:.4f})\t"
            f"Acc: {acc_val:.6f} ({topk_str})"
            )
        returns = loss_avg, acc_avg, loss_val, acc_val, log_str
    else:
        log_str = (
            f"Loss: {loss_avg:.4f}\t"
            f"Acc: {topk_str}"
            )
        returns = loss_avg, acc_avg, log_str

    return returns


#############################################
def calc_topk_accuracy(output, target, topk=(1,)):
    """
    Modified from: 
    https://gist.github.com/agermanidis/275b23ad7a10ee89adccf021536bb97e
    Given predicted and ground truth labels, calculate top-k accuracies.
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(1 / batch_size))
    return res


#############################################
def update_topk_meters(topk_meters, output, target, ks=[1, 3, 5]):
    """updates topK meters in place"""
    if len(topk_meters) != len(ks):
        raise ValueError(
            "Must provide as many topK meter objects as 'ks'."
            )

    top_vals = calc_topk_accuracy(output, target, ks)

    for t, top_val in enumerate(top_vals):
        topk_meters[t].update(top_val.item(), len(output))


#############################################
def calc_accuracy(output, target):
    """output: (B, N); target: (B)"""
    target = target.squeeze()
    _, pred = torch.max(output, 1)
    return torch.mean((pred == target).float())


#############################################
def calc_accuracy_binary(output, target):
    """output, target: (B, N), output is logits, before sigmoid """
    pred = output > 0
    acc = torch.mean((pred == target.byte()).float())
    return acc


#############################################
def init_loss_dict(dataset="gabors"):
    
    shared_keys = ["epoch_n", "loss"]
    train_keys = [
        "detailed_loss", "loss_by_batch", "dot_by_batch", "target_by_batch"
        ]

    loss_dict = dict()    
    for main_key in ["train", "val"]:
        loss_dict[main_key] = dict()
        sub_keys = shared_keys
        if main_key == "train":
            sub_keys = shared_keys + train_keys
            if dataset == "gabors":
                sub_keys.append("seq")
        for key in sub_keys:
            loss_dict[main_key][key] = list()
    
    return loss_dict


#############################################
def save_loss_dict(loss_dict, output_dir=".", seed=None, dataset="gabors", 
                   unexp_epoch=10):

    seed_str = "" if seed is None else f"_{seed}"
    unexp_str = f"_{unexp_epoch}" if dataset == "gabors" else ""
    full_path = Path(
        output_dir, f"loss_data{unexp_str}{seed_str}.json"
        )
    with json.load(full_path, "w"):
        json.dump(loss_dict, full_path)

