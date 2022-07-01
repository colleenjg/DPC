from collections import deque
import json
import logging
from pathlib import Path

import matplotlib as mpl
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
        height, width = self.mat.shape
        if annotate:
            for x in range(width):
                for y in range(height):
                    ax.annotate(
                        str(int(self.mat[x][y])), xy=(y+1, x+1),
                        horizontalalignment="center", 
                        verticalalignment="center", fontsize=8
                        )

        if dictionary is None:
            xtick_diff = min(np.diff(ax.get_xticks()))
            if xtick_diff < 1:
                ax.set_xticks(np.arange(width) + 1)
            ytick_diff = min(np.diff(ax.get_yticks()))
            if ytick_diff < 1:
                ax.set_yticks(np.arange(height) + 1)
        else:
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
def calc_topk_accuracy(output, target, topk=(1,)):
    """
    calc_topk_accuracy(output, target)

    output dim are: B * PS * HW x B_per * PS * HW => (32, 1, 16) x (32, 1, 16)

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

    if not supervised:
        shared_keys.append("sup_target_by_batch")

    batch_keys = []
    if save_by_batch:
        batch_keys = [
            "batch_epoch_n", "loss_by_batch", "loss_by_item", 
            ]
        if not light:
            batch_keys.extend(["output_by_batch", "target_by_batch"])

    for main_key in main_keys:
        if main_key not in loss_dict.keys():
            loss_dict[main_key] = dict()
        sub_keys = shared_keys + batch_keys
        for key in sub_keys:
            if key not in loss_dict[main_key].keys():
                loss_dict[main_key][key] = list()
    
    return loss_dict


#############################################
def plot_acc(sub_ax, data_dict, epoch_ns, chance=None, color="k", ls=None, 
             data_label="train"):
    """
    plot_acc(sub_ax, data_dict, epoch_ns)
    """

    if len(data_label) and data_label[-1] != " ":
        data_label = f"{data_label} "

    acc_keys = sorted([key for key in data_dict.keys() if "top" in key])
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
    """
    
    loss_by_batch = np.asarray(data_dict["loss_by_batch"]).T
    n_batches = len(loss_by_batch)
    batch_cmap = mpl.cm.get_cmap(colors)
    cmap_samples = np.linspace(1.0, 0.3, n_batches)

    if U_ax is not None:
        target_key = "target_by_batch"
        if "sup_target_by_batch" in data_dict.keys():
            target_key = "sup_target_by_batch"
 
        freqs = []
        for gfr in ["U", "D"]:
            # retrieve only the Gabor frame (not orientations)
            freqs.append((
                np.asarray(data_dict[target_key])[:, :, 0] == gfr
                ).astype(float).mean(axis=(2, 3, 4)).T)
        U_freqs_over_DU = freqs[0] / (freqs[0] + freqs[1]) * 100

    for b, batch_losses in enumerate(loss_by_batch):
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
        epoch_ns, loss_by_batch.mean(axis=0), color="k", alpha=0.6, lw=2.5, 
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
    """

    datatypes = ["train"]
    colors = ["blue"]
    ls = ["dashed"]
    if "val" in loss_dict.keys():
        datatypes.append("val")
        colors.append("orange")
        ls.append(None)

    chance = None if num_classes is None else 1 / num_classes

    plot_seq = by_batch and "gabors" in dataset.lower()
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
        if "gabors" in dataset.lower() and unexp_epoch < max(epoch_ns):
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


#############################################
def save_loss_dict(loss_dict, output_dir=".", seed=None, dataset="UCF101", 
                   unexp_epoch=10, plot=True, num_classes=None):
    """
    save_loss_dict(loss_dict)
    """

    seed_str_pr = ""
    if seed is not None:
        seed_str_pr = f" (seed {seed})"

    unexp_str_pr = ""
    if "gabors" in dataset.lower():
        unexp_str_pr = f" (unexp. epoch: {unexp_epoch})"

    savename = "loss_data"
    full_path = Path(output_dir, f"loss_data.json")
        
    with open(full_path, "w") as f:
        json.dump(loss_dict, f)

    if plot:
        by_batches = [False]
        if "loss_by_batch" in loss_dict["train"].keys():
            by_batches.append(True)

        for by_batch in by_batches:
            if by_batch:
                batch_str = "_by_batch"
                batch_str_pr = " (by batch)"
            else:
                batch_str, batch_str_pr = "", ""

            fig = plot_loss_dict(
                loss_dict, num_classes=num_classes, dataset=dataset, 
                unexp_epoch=unexp_epoch, by_batch=by_batch
                )
            title = (f"{dataset[0].capitalize()}{dataset[1:]} dataset"
                f"{unexp_str_pr}{seed_str_pr}{batch_str_pr}")
            fig.suptitle(title)

            full_path = Path(output_dir, f"{savename}{batch_str}.svg")
            fig.savefig(full_path, bbox_inches="tight")

