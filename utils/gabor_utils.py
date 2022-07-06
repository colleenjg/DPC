#!/usr/bin/env python

import argparse
import copy
from pathlib import Path
import logging
import sys
import warnings

import json
from matplotlib import pyplot as plt
import numpy as np
import torch

sys.path.extend(["..", str(Path("..", "utils"))])
from utils import loss_utils, misc_utils

BASE_SEQ = ["A", "B", "C", "D"]
NUM_MEAN_ORIS = 8
U_ADJ = 90

DEG = u"\u00b0"
GABOR_CONF_MAT_DIREC = "gabor_confusion_matrices"


logger = logging.getLogger(__name__)


#############################################
def get_mean_oris(num_mean_oris=NUM_MEAN_ORIS):
    """
    get_mean_oris()
    """

    mean_oris = np.arange(0, 360, 360 / num_mean_oris)

    return mean_oris


#############################################
def get_mean_U_oris(mean_oris):
    """
    get_mean_U_oris(mean_oris)
    """

    mean_oris_U = (np.asarray(mean_oris) + U_ADJ) % 360

    return mean_oris_U



#############################################
def get_gabor_classes(num_mean_oris=NUM_MEAN_ORIS, gray=True, U_prob=0.1, 
                      diff_U_possizes=False):
    """
    get_gabor_classes()
    """

    mean_oris = np.arange(0, 360, 360 / num_mean_oris)
    frames = copy.deepcopy(BASE_SEQ)

    if U_prob > 0:
        mean_oris_U = (mean_oris + U_ADJ) % 360
        if diff_U_possizes:
            frames.append("U")
        else:
            frames[frames.index("D")] = "D/U"
            DU_mean_oris = np.sort(
                np.unique(np.concatenate([mean_oris, mean_oris_U]))
            )

    classes = []
    for frame in frames:
        if frame == "D/U":
            use_mean_oris = DU_mean_oris
        elif frame == "U":
            use_mean_oris = mean_oris_U
        else:
            use_mean_oris = mean_oris

        for mean_ori in use_mean_oris:
            classes.append((frame, mean_ori))
    
    if gray:
        classes.append(("G", "N/A"))

    return classes


#############################################
def get_num_classes(num_mean_oris=NUM_MEAN_ORIS, gray=True, U_prob=0.1, 
                    diff_U_possizes=False):
    """
    get_num_classes()
    """

    classes = get_gabor_classes(
        num_mean_oris, gray=gray, U_prob=U_prob, 
        diff_U_possizes=diff_U_possizes
        )

    num_classes = len(classes)

    return num_classes


#############################################
def get_image_class_and_label_dict(num_mean_oris=NUM_MEAN_ORIS, gray=True, 
                                   U_prob=0.1, diff_U_possizes=False, 
                                   class_to_label=True):
    """
    get_image_class_and_label_dict()
    """

    image_classes = get_gabor_classes(
        num_mean_oris, gray=gray, U_prob=U_prob, 
        diff_U_possizes=diff_U_possizes
        )
    
    if class_to_label:
        image_class_and_label = {
            im_cl: label for label, im_cl in enumerate(image_classes)
            }
    else:
        image_class_and_label = {
            label: im_cl for label, im_cl in enumerate(image_classes)
            }

    return image_class_and_label


#############################################
def image_class_to_label(image_class_and_label_dict, seq_images, 
                         seq_image_mean_oris, class_to_label=True):
    """
    image_class_to_label(image_class_and_label_dict, seq_images, 
                         seq_image_mean_oris)
    """

    seq_images = np.asarray(seq_images)
    seq_image_mean_oris = np.asarray(seq_image_mean_oris)

    if seq_images.shape != seq_image_mean_oris.shape:
        raise ValueError(
            "'seq_images' and 'seq_image_mean_oris' must have the same "
            "shape."
            )

    orig_shape = seq_images.shape
    seq_images = seq_images.reshape(-1)
    seq_image_mean_oris = seq_image_mean_oris.reshape(-1)

    if class_to_label:
        class_to_label_dict = image_class_and_label_dict
    else:
        class_to_label_dict = {
            im_cl: label for label, im_cl in image_class_and_label_dict.keys()
            }
    
    # if classes don't differentiate between D and U
    dict_images, _ = list(zip(*class_to_label_dict.keys()))
    if "D/U" in dict_images:
        seq_images = copy.deepcopy(seq_images).astype("<U3") # for "/" character
        seq_images[np.where(seq_images == "D")] = "D/U"
        seq_images[np.where(seq_images == "U")] = "D/U"

    seq_image_classes = [
        (i, o) if np.isfinite(o) else (i, "N/A") 
        for i, o in zip(seq_images, seq_image_mean_oris) 
        ]

    if not set(seq_image_classes).issubset(set(class_to_label_dict.keys())):
        raise ValueError(
            "'seq_images' and 'seq_image_mean_oris' contain combinations that "
            "do not occur in 'image_class_and_label_dict'."
            )

    seq_image_labels = np.asarray(
        [class_to_label_dict[im_cl] for im_cl in seq_image_classes]
    ).reshape(orig_shape)
        
    return seq_image_labels


#############################################
def get_unexp(image_types):
    """
    get_unexp(image_types)
    """

    image_types = np.asarray(image_types)

    if "D/U" in image_types:
        raise ValueError(
            "'image_types' must not contain 'D/U' values. "
            "Only 'D' or 'U', individually."
            )

    unexp = image_types == "U"

    return unexp


#############################################
def specify_image_types(image_types, unexp):
    """
    specify_image_types(image_types, unexp)
    """

    image_types = np.asarray(image_types)
    unexp = np.asarray(unexp).astype(bool)

    if image_types.shape != unexp.shape:
        raise ValueError("'image_types' and 'unexp' must have the same shape.")

    if "D/U" in image_types:
        image_types = copy.deepcopy(image_types)
        image_types[image_types == "D/U"] = "D"
        if (image_types[unexp] != "D").any():
            raise ValueError("Some unexpected image types are not D/Us.")
        image_types[unexp] = "U"
    
    if ((image_types == "U") != unexp).any():
        raise RuntimeError(
            "U images cannot be correctly inferred from 'image_types' and "
            "'unexp'."
            )

    return image_types


#############################################
def image_label_to_class(image_label_and_class_dict, seq_labels, 
                         label_to_class=True, seq_unexp=None):
    """
    image_label_to_class(image_class_and_label_dict, seq_labels)
    """

    seq_labels = np.asarray(seq_labels)

    orig_shape = seq_labels.shape
    seq_labels = seq_labels.reshape(-1)

    if label_to_class:
        label_to_class_dict = image_label_and_class_dict
    else:
        label_to_class_dict = {
            label: im_cl for im_cl, label in image_label_and_class_dict.keys()
            }

    # check that all labels exist in the dictionary
    if not set(seq_labels).issubset(set(label_to_class_dict.keys())):
        raise ValueError(
            "'seq_labels' contains labels not present in "
            "'image_label_and_class_dict'."
            )

    seq_classes = [
        label_to_class_dict[i] for i in seq_labels
    ]

    # convert to D or U instead of D/U, if applicable
    if seq_unexp is not None:
        seq_unexp = np.asarray(seq_unexp)
        if seq_unexp.shape != orig_shape:
            raise ValueError(
                "If provided, 'seq_unexp' must have the same shape as "
                "'seq_labels'."
                )

        image_types, mean_oris = list(zip(*seq_classes))
        image_types = specify_image_types(image_types, seq_unexp.reshape(-1))
        seq_classes = list(zip(image_types, mean_oris))

    seq_classes = misc_utils.renest(seq_classes, orig_shape)

    return seq_classes


#############################################
class GaborsConfusionMeter(loss_utils.ConfusionMeter):
    """Compute and show confusion matrix"""

    def __init__(self, class_names):
        self._set_properties = []
        self.reinitialize_values_gabors(class_names)
        super().__init__(class_names)


    def reset_properties(self):
        """
        self.reset_properties()
        """
        
        for attr_name in self._set_properties:
            delattr(self, attr_name)
        self._set_properties = []


    def reinitialize_values_gabors(self, class_names):
        """
        self.reinitialize_values_gabors(class_names)
        """
        
        self.reset_properties()
        class_names = [tuple(class_name) for class_name in class_names]
        super().reinitialize_values(class_names=class_names)

        image_types, mean_oris = list(zip(*class_names))

        self.image_types = sorted(set(image_types))
        if "N/A" in mean_oris:
            mean_oris = sorted(set([ori for ori in mean_oris if ori != "N/A"]))
            mean_oris.append("N/A")
        else:
            mean_oris = sorted(set(mean_oris))
        self.mean_oris = mean_oris


    @property
    def nest_idx(self):
        """
        self.nest_idx
        """
        if not hasattr(self, "_nest_idx"):
            nest_idx = []
            for mean_ori in self.mean_oris:
                for image_type in self.image_types:
                    label = (image_type, mean_ori)
                    if label in self.class_names:
                        idx = self.class_names.index((image_type, mean_ori))
                        nest_idx.append(idx)
            nest_idx = np.asarray(nest_idx)
            if len(nest_idx) != len(self.class_names):
                raise RuntimeError(
                    "'nest_idx' and 'self.class_names' lengths do not match."
                    )
            if len(np.unique(nest_idx)) != len(self.class_names):
                raise RuntimeError("Duplicate class names or indices found.")

            self._nest_idx = nest_idx
            self._set_properties.append("_nest_idx")
        
        return self._nest_idx


    @property
    def unnest_idx(self):
        """
        self.unnest_idx
        """
        if not hasattr(self, "_unnest_idx"):
            self._unnest_idx = np.argsort(self.nest_idx)
            self._set_properties.append("_unnest_idx")
        
        return self._unnest_idx


    def get_labels(self, nest_frames=False):
        """
        self.get_labels()
        """

        if nest_frames:
            labels = [self.class_names[i] for i in self.nest_idx]
        else:
            labels = [cl_name for cl_name in self.class_names]

        return labels


    def get_main_edges(self, nest_frames=False):
        """
        self.get_main_edges()
        """

        labels = self.get_labels(nest_frames)
        idx = 1 if nest_frames else 0

        main_edges = [0]
        curr_val = labels[0][idx]
        for l, label in enumerate(labels):
            if label[idx] != curr_val:
                main_edges.append(l)
                curr_val = label[idx]
        main_edges.append(len(labels))

        return main_edges


    def get_ori_str(self, ori):
        """
        self.get_ori_str(ori)
        """

        if ori != "N/A":
            ori = float(ori)
            if int(ori) == ori:
                ori = int(ori)
            ori = u"{}{}".format(ori, DEG)
        return ori


    def get_main_label_dict(self, nest_frames=False):
        """
        self.get_main_label_dict()
        """
        
        labels = self.get_labels(nest_frames)
        main_edges = self.get_main_edges(nest_frames)
        main_idx = 1 if nest_frames else 0

        outer_label_dict = dict()
        for e, edge in enumerate(main_edges[:-1]):
            mid_e = np.mean([edge, main_edges[e + 1]]) - 0.5
            label = labels[edge][main_idx]
            if nest_frames:
                label = self.get_ori_str(label)
            outer_label_dict[mid_e] = label
        
        return outer_label_dict


    def get_nested_label_dict(self, nest_frames=False):
        """
        self.get_nested_label_dict()
        """
        
        labels = self.get_labels(nest_frames)
        nested_idx = 0 if nest_frames else 1

        nested_label_dict = dict()
        for i, label in enumerate(labels):
            label = label[nested_idx]
            if not nest_frames:
                label = self.get_ori_str(label)
            nested_label_dict[i] = label

        return nested_label_dict


    def plot_mat(self, path=None, nest_frames=False):
        """
        self.plot_mat()
        """

        # accomodate longer main labels on right
        cbar_kwargs = dict({"pad": 0.1})

        try:
            if nest_frames: # nest self.mat, if needed
                self.mat = self.mat[self.nest_idx][:, self.nest_idx]
            fig = super().plot_mat(**cbar_kwargs)

        finally:
            if nest_frames: # unnest self.mat, if applicable
                self.mat = self.mat[self.unnest_idx][:, self.unnest_idx]

        ax = fig.axes[0]

        # add nested labels
        nested_dict = self.get_nested_label_dict(nest_frames)
        super().add_labels(ax, label_dict=nested_dict)

        # add main labels
        main_dict = self.get_main_label_dict(nest_frames)
        super().add_labels(ax, label_dict=main_dict, secondary=True)

        main_edges = self.get_main_edges(nest_frames)
        if len(main_edges) > 2:
            for edge in main_edges[1:-1]:
                ax.axhline(edge + 0.5, lw=1, color="k")
                ax.axvline(edge + 0.5, lw=1, color="k")

        if path is None:
            return fig
        else:
            Path(path).parent.mkdir(exist_ok=True, parents=True)
            fig.savefig(path, format="svg", bbox_inches="tight", dpi=600)
            plt.close(fig)


    def get_storage_dict(self):
        """
        self.get_storage_dict(storage_dict)
        """

        storage_dict = super().get_storage_dict()

        all_keys = ["image_types", "mean_oris"]
        all_data = [self.image_types, self.mean_oris]
        for key, data in zip(all_keys, all_data):
            if isinstance(data, np.ndarray):
                data = data.tolist()
            storage_dict[key] = data

        return storage_dict

    
    def load_from_storage_dict(self, storage_dict):
        """
        self.load_from_storage_dict(storage_dict)
        """

        self.reinitialize_values_gabors(storage_dict["class_names"])

        super().load_from_storage_dict(storage_dict)
        self.class_names = [
            tuple(class_name) for class_name in self.class_names
            ]

        stored_class_names = storage_dict["class_names"]
        if len(self.class_names) != len(stored_class_names):
            raise RuntimeError(
                "'self.class_names' does not have the same number of "
                "labels as the stored 'class_names'."
                )

        for new, stored in zip(self.class_names, stored_class_names):                
            if list(new) != list(stored):
                raise RuntimeError(
                    "'self.class_names' does not exactly match stored "
                    "'class_names'."
                    )


#############################################
def plot_save_gabor_conf_mat(gabor_conf_mat, mode="train", epoch_n=0, 
                             output_dir="."):
    """
    plot_save_gabor_conf_mat(gabor_conf_mat)
    """

    gabor_conf_mat_path = Path(
        output_dir, GABOR_CONF_MAT_DIREC, f"{mode}_{epoch_n:03}.svg"
        )
    
    gabor_conf_mat.plot_mat(gabor_conf_mat_path)

    gabor_conf_mat_dict_path = Path(
        output_dir, GABOR_CONF_MAT_DIREC, "gabor_confusion_mat_data.json"
    )

    gabor_conf_mat_dict = dict()
    if gabor_conf_mat_dict_path.is_file():
        with open(gabor_conf_mat_dict_path, "r") as f:
            gabor_conf_mat_dict = json.load(f)

    if mode not in gabor_conf_mat_dict.keys():
        gabor_conf_mat_dict[mode] = dict()

    if epoch_n in gabor_conf_mat_dict[mode]:
        raise RuntimeError(
            f"{epoch_n} epoch key for {mode} mode already exists."
            )
    
    gabor_conf_mat_dict[mode][f"epoch_{epoch_n}"] = \
        gabor_conf_mat.get_storage_dict()

    with open(gabor_conf_mat_dict_path, "w") as f:
        json.dump(gabor_conf_mat_dict, f)


#############################################
def load_replot_gabor_conf_mat(gabor_conf_mat_dict_path, nest_frames=False, 
                               output_dir=None):
    """
    load_replot_gabor_conf_mat(gabor_conf_mat_dict_path)
    """

    gabor_conf_mat_dict_path = Path(gabor_conf_mat_dict_path)
    if not gabor_conf_mat_dict_path.is_file():
        raise OSError(f"{gabor_conf_mat_dict_path} is not a file.")
    
    with open(gabor_conf_mat_dict_path, "r") as f:
        gabor_conf_mat_dict = json.load(f)

    if output_dir is None:
        save_dir = gabor_conf_mat_dict_path.parent
    else:
        save_dir = Path(output_dir, GABOR_CONF_MAT_DIREC)

    if not isinstance(gabor_conf_mat_dict, dict):
        raise RuntimeError(
            f"Expected {gabor_conf_mat_dict_path} to be storing a dictionary."
            )

    for mode_key, mode_dict in gabor_conf_mat_dict.items():
        if not isinstance(mode_dict, dict):
            raise RuntimeError(
                f"Expected {gabor_conf_mat_dict_path} to be storing "
                "nested dictionaries."
                )
        num_epochs = len(mode_dict)
        logger.info(
            f"Loading and plotting Gabor confusion matrices for {num_epochs} "
            f"epochs ({mode_key} mode).", extra={"spacing": "\n"}
            )
        for epoch_key, conf_mat_dict in mode_dict.items():
            if "class_names" not in conf_mat_dict.keys():
                raise RuntimeError(
                    "Expected GaborsConfusionMeter storage dictionaries two "
                    "levels in."
                    )
            if not epoch_key.startswith("epoch_"):
                raise KeyError(
                    "Expected to find a key of the form 'epoch_x', but "
                    f"found {epoch_key}"
                    )
            
            # load confusion matrix
            gabor_conf_mat = GaborsConfusionMeter(conf_mat_dict["class_names"])
            gabor_conf_mat.load_from_storage_dict(conf_mat_dict)

            epoch_n = int(epoch_key.replace("epoch_", ""))
            nest_frame_str = "_nest_fr" if nest_frames else ""
            save_name = f"{mode_key}{nest_frame_str}_{epoch_n:03}.svg"
            plot_path = Path(save_dir, save_name)
            gabor_conf_mat.plot_mat(plot_path, nest_frames=nest_frames)


#############################################
def init_gabor_conf_mat(dataset):
    """
    init_gabor_conf_mat(dataset)
    """
    
    gabor_classes = list(dataset.class_dict_encode.keys())
    confusion_mat = GaborsConfusionMeter(gabor_classes)

    return confusion_mat


#############################################
def init_gabor_records(dataset, init_conf_mat=True):
    """
    init_gabor_records(dataset)
    """
    
    gabor_classes = list(dataset.class_dict_encode.keys())
    if init_conf_mat:
        confusion_mat = init_gabor_conf_mat(dataset)
    else:
        confusion_mat = None

    # initialize loss/accuracy dictionaries
    loss_dict = dict()
    acc_dict = dict()

    for gabor_image, gabor_ori in gabor_classes:
        if gabor_image not in loss_dict.keys():
            loss_dict[gabor_image] = list()
            acc_dict[gabor_image] = list()
        if gabor_ori not in loss_dict.keys():
            loss_dict[gabor_ori] = list()
            acc_dict[gabor_ori] = list()
    
    for key in ["image_types_overall", "mean_oris_overall", "overall"]:
        acc_dict[key] = list()

    return loss_dict, acc_dict, confusion_mat


#############################################
def update_records(dataset, loss_dict, acc_dict, output, sup_target, 
                   batch_loss, supervised=False, confusion_mat=None):
    """
    update_records(dataset, loss_dict, acc_dict, output, sup_target, 
                   batch_loss)
    """

    if supervised:
        target_labels = get_gabor_sup_label(sup_target).reshape(-1)
        pred_labels = np.argmax(output, axis=1).reshape(-1)

    else:
        B, PS = batch_loss.shape
        HW = output.shape[0] / np.product(B * PS)
        if int(HW) != HW:
            raise RuntimeError(
                "Failed to calculate HW from 'output' and 'batch_loss' shapes."
                )
        HW = int(HW)

        main_shape = (B, PS, HW)

        # retrieve a prediction for each batch example / prediction step
        pred = loss_utils.get_predictions(
            torch.from_numpy(output), keep_topk=1, acc_avg_HW=True, 
            main_shape=main_shape
            )[0].reshape(-1)

        # retrieve target for first frame of each predicted sequences
        target_labels, _ = sup_target[:, -PS:, 0].reshape(B * PS, -1).T
        pred_labels = target_labels[pred]

    # update confusion matrix
    if confusion_mat is not None:
        confusion_mat.update(pred_labels, target_labels)

    # find proportion of correct supervised predictions
    label_decode_dict = dataset.class_dict_decode

    target_classes = image_label_to_class(label_decode_dict, target_labels)
    pred_classes = image_label_to_class(label_decode_dict, pred_labels)

    target_im_types, target_mean_oris = [
        np.asarray(val) for val in zip(*target_classes)
        ]
    target_mean_oris = target_mean_oris.astype(str)

    pred_im_types, pred_mean_oris = [
        np.asarray(val) for val in zip(*pred_classes)
        ]
    pred_mean_oris = pred_mean_oris.astype(str)
    
    correct_image_types = (target_im_types == pred_im_types)
    correct_mean_oris = (target_mean_oris == pred_mean_oris)
    correct_both = np.asarray(target_labels == pred_labels)
    if (correct_both != (correct_image_types * correct_mean_oris)).any():
        raise RuntimeError(
            "'correct_both' should match value inferred from "
            "'correct_image_types' and 'correct_mean_oris', but does not, "
            "suggesting a label interpretation error."
            )

    # add to meters
    batch_loss = batch_loss.reshape(-1)
    for key in loss_dict.keys():
        str_key = str(key)
        if str_key == "N/A" or "." in str_key or str_key.isdigit(): # oris
            idx = (target_mean_oris == str_key)
            n_correct = correct_mean_oris[idx].sum()
        else: # frames
            idx = (target_im_types == key)
            n_correct = correct_image_types[idx].sum()

        n_vals = sum(idx).item()
        if n_vals == 0:
            loss_dict[key].append(np.nan)
            acc_dict[key].append(np.nan)
        else:
            loss_dict[key].append(batch_loss[idx].mean().item())
            acc_dict[key].append(n_correct.item() / n_vals)

    keys = ["image_types_overall", "mean_oris_overall", "overall"]
    all_data = [correct_image_types, correct_mean_oris, correct_both]
    for key, data in zip(keys, all_data):
        n_total = len(data)
        acc_dict[key].append(data.sum().item() / n_total)


#############################################
def update_dataset_possizes(main_loader, val_loader=None, seed=None, incr=0):
    """
    update_dataset_possizes(main_loader)
    """

    if not main_loader.dataset.same_possizes:
        return

    if seed is None:
        seed = np.random.choice(int(2**32))
    
    main_loader.dataset.set_possizes(seed=seed + incr, reset=True)
    
    if val_loader is not None:
        val_loader.dataset.set_possizes(seed=seed + incr, reset=True)    

    return seed


#############################################
def update_unexp(main_loader, val_loader=None, epoch_n=0, unexp_epoch=0):
    """
    update_unexp(main_loader)
    """

    if not main_loader.dataset.unexp and epoch_n >= unexp_epoch:
        main_loader.dataset.unexp = True
        loader_mode = main_loader.dataset.mode
        dataset_str = f" {loader_mode} dataset"
        if val_loader is not None and not val_loader.dataset.unexp:
            val_loader.dataset.unexp = True
            dataset_str = (
                f"{loader_mode} and {val_loader.dataset.mode} datasets"
                )

        logger.info(f"Setting {dataset_str} to include unexpected sequences.", 
            extra={"spacing": "\n"}
            )


#############################################
def update_gabors(main_loader, val_loader=None, seed=None, epoch_n=0, 
                  unexp_epoch=0):
    """
    update_gabors(main_loader)
    """

    seed = update_dataset_possizes(
        main_loader, val_loader, seed=seed, incr=epoch_n
        )
    
    update_unexp(
        main_loader, val_loader, epoch_n=epoch_n, 
        unexp_epoch=unexp_epoch
        )

    return seed


############################################
def get_gabor_sup_target(dataset, sup_target):
    """
    get_gabor_sup_target(dataset, sup_target)
    """
    
    sup_target = torch.moveaxis(sup_target, -1, 0)
    target_images = dataset.image_label_to_image(
        sup_target[0].to("cpu").numpy().reshape(-1)
        )
    target_images = np.asarray(target_images).reshape(
        sup_target[0].shape
        )
    sup_target = [target_images.tolist(), sup_target[1].tolist()]

    return sup_target
    
    
#############################################
def get_gabor_sup_label(sup_target, warn=False):
    """
    get_gabor_sup_label(sup_target)
    """

    if len(sup_target.shape) != 4:
        raise ValueError(
            "'sup_target' should have 4 dimensions "
            "(B x N x SL x (label, unexp))."
            )

    B, N, SL, _ = sup_target.shape

    pred_idx = -1
    if warn:
        warnings.warn(
            "The supervised task for the Gabors dataset is currently "
            "implemented to predict the label (image type/mean orientation) "
            "of the final frame of the final sequence "
            f"(frame {SL} of {N} sequences)."
            )

    sup_target = sup_target[:, pred_idx, pred_idx, 0].reshape(B, 1)

    return sup_target


############################################
def get_gabor_sup_target_to_store(dataset, sup_target):
    """
    get_gabor_sup_target_to_store(dataset, sup_target)
    """
    
    seq_labels, seq_unexp = torch.moveaxis(sup_target, -1, 0)
    sup_target = dataset.image_label_to_class(seq_labels, seq_unexp)

    return sup_target
    

#############################################
def warn_supervised(dataset):
    """
    warn_supervised(dataset)
    """

    if dataset.mode == "test":
        raise ValueError(
            "Single supervised targets have not been set for Gabors "
            "dataset in 'test' mode. Use 'val' mode instead."
            )
    SL = dataset.seq_len
    B, N = 1, 1
    dummy_arr = np.empty([B, N, SL, 2])

    get_gabor_sup_label(dummy_arr, warn=True)


#############################################
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--gabor_conf_mat_dict_path", default=None,
        help="path of Gabor confusion matrix data to replot")
    parser.add_argument("--nest_frames", action="store_true",
        help="If True, Gabor frames are nested instead of orientations")

    parser.add_argument("--output_dir", default=None, 
        help=("directory in which to save files. If None, it is inferred from "
        "another path argument)"))
    parser.add_argument('--log_level', default='info', 
                        help='logging level, e.g., debug, info, error')
    args = parser.parse_args()

    misc_utils.get_logger_with_basic_format(level=args.log_level)


    if args.gabor_conf_mat_dict_path is None:
        breakpoint()
    else:
        load_replot_gabor_conf_mat(
            args.gabor_conf_mat_dict_path, 
            nest_frames=args.nest_frames, 
            output_dir=args.output_dir
            )
