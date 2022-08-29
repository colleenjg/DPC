import copy
import logging
from pathlib import Path
import time

import numpy as np
import pickle as pkl
import torch

from dataset import gabor_sequences
from utils import loss_utils, misc_utils, training_utils


logger = logging.getLogger(__name__)

#############################################
def init_layers_info_dict(model):
    """
    init_layers_info_dict(model)
    """

    layer_names = ["encoder", "recurrent", "predictive"]
    layers = [
        model.backbone.layer4[1].bn2,
        model.agg.cell_list[0].out_gate,
        model.network_pred[2],
    ]

    info_dict = dict()
    count_dict = dict()
    for name in layer_names:
        info_dict[name] = {
            "activations": [], 
            "errors": []
            }
        count_dict[name] = {
            "activations": 0,
            "errors": 0,
        }

    return info_dict, count_dict, layers, layer_names

#############################################
def reset_count_dict(count_dict):
    """
    reset_count_dict(count_dict)
    """

    for name in count_dict.keys():
        for hook_type in count_dict[name].keys():
            count_dict[name][hook_type] = 0

    return


#############################################
def extract_output(output, batch_size=32, pred_step=2, name="encoder", count=0):
    """
    extract_output(output)
    """

    if name == "recurrent" and count >= pred_step:
        return None

    if isinstance(output, tuple):
        if len(output) == 1:
            output = output[0]
        else:
            raise ValueError("Expected tuple of length 1.")

    output = output.detach()

    shape_tuple = output.shape
    if name == "encoder":
        num_seq = shape_tuple[0] // batch_size
        output = output.view(batch_size, num_seq, *shape_tuple[1:])[
            :, -pred_step :
            ]

    return output


#############################################
def init_hooks_dict(model, batch_size):
    """
    init_hooks_dict(model, batch_size)
    """

    info_dict, count_dict, layers, layer_names = init_layers_info_dict(model)
    pred_step = model.pred_step

    # register forward and backward hooks
    def get_hook(name, hook_type="activations"):
        """
        get_hook(name)
        """

        def hook(layer, inp, output):
            """
            hook(layer, inp, output)
            """

            subdict = info_dict[name][hook_type]

            count = count_dict[name][hook_type]
            output = extract_output(
                output, batch_size, pred_step=pred_step, name=name, count=count
            )

            if output is None:
                return

            subdict.append(output.cpu())

            if name in ["recurrent", "predictive"] and count == pred_step - 1:
                subdict[-pred_step] = torch.stack(subdict[-pred_step:], 1)
                if pred_step > 1:
                    del info_dict[name][hook_type][-pred_step + 1:]
            
            count_dict[name][hook_type] += 1
            if name == "encoder":
                reset_count_dict(count_dict)

        return hook


    activation_hooks = [
        layer.register_forward_hook(get_hook(name, "activations")) 
        for layer, name in zip(layers, layer_names)
        ]

    error_hooks = [
        layer.register_full_backward_hook(get_hook(name, "errors")) 
        for layer, name in zip(layers, layer_names)
        ]
    
    return info_dict, error_hooks, activation_hooks


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
            "'gabor_analysis' not implemented for supervised models."
            )
    
    if dataloader.dataset.shift_frames:
        raise NotImplementedError(
            "Dataset should be implemented with shift_frames set to False."
            )
        
    if dataloader.dataset.gab_img_len != dataloader.dataset.seq_len:
        raise NotImplementedError(
            "Dataset should be implemented with gab_img_len equal to seq_len."
        )


#############################################
def gabor_analysis(model, dataloader, device="cuda", output_dir=".", 
                   suffix="first"):
    """
    gabor_analysis(model, dataloader)
    """


    run_checks(model, dataloader)

    dataloader = copy.deepcopy(dataloader)

    # Do not use DataParallel
    model = training_utils.get_model_only(model).to(device)
    criterion, _ = loss_utils.get_criteria(device=device)

    info_dict, error_hooks, activation_hooks = init_hooks_dict(
        model, batch_size=dataloader.batch_size
        )

    info_dict["target"] = []

    start_time = time.perf_counter()
    for idx, (input_seq, sup_target) in enumerate(dataloader):
        input_seq = input_seq.to(device)
        input_seq_shape = input_seq.size()
        [output_, mask_] = model(input_seq)

        # get targets, and reshape for loss calculation
        output_flattened, target_flattened, _, _ = \
            training_utils.prep_loss(
                output_, mask_, sup_target, input_seq_shape, 
                supervised=False, is_gabor=True
                )

        target_flattened = target_flattened.to(device)
        loss = criterion(output_flattened, target_flattened)
        
        loss.backward()
        model.zero_grad()

        # one target per block
        info_dict["target"].append(sup_target[:, -model.pred_step:, 0])
    
    stop_time = time.perf_counter()
    time_str = misc_utils.format_time(stop_time - start_time, sep_min=True)
    logger.info(
        f"[{time_str}]  Accumulated activations and errors", 
        extra={"spacing": "\n"}
        )

    for hook in activation_hooks + error_hooks:
        hook.remove()
    
    info_dict = aggregate_data(info_dict, dataloader.dataset)

    save_path = Path(output_dir, "activ_errors", f"{suffix}.pkl")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(save_path), "wb") as f:
        pkl.dump(info_dict, f)



def aggregate_data(info_dict, dataset):
    """
    aggregate_data(info_dict, dataset)

    Aggregate across batches and target classes.
    """
    
    factor = 10 # larger than the binary unexpected value

    targets = torch.cat(info_dict.pop("target")).numpy()
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
    
    for name in info_dict.keys():
        if name == "target":
            continue
        hook_types = list(info_dict[name].keys())
        for hook_type in hook_types:
            data = torch.cat(info_dict[name].pop(hook_type)).numpy()
            new_shape = (len(unique), num_pred, *data.shape[2:])
            info_dict[name][hook_type] = {
                stat: np.full(new_shape, np.nan) for stat in ["means", "stds"]
                }
            
            for i, pred_idxs in enumerate(all_idxs):
                for p, idxs in enumerate(pred_idxs):
                    if not len(idxs):
                        continue
                    info_dict[name][hook_type]["means"][i, p] = \
                        data[idxs, p].mean(axis=0)
                    info_dict[name][hook_type]["stds"][i, p] = \
                        data[idxs, p].std(axis=0)
    
    labels = [val // factor for val in unique]
    unexps = [val - label * factor for val, label in zip(unique, labels)]

    seq_classes = dataset.image_label_to_class(labels, unexps)

    info_dict["seq_classes"] = seq_classes
    info_dict["ns"] = ns

    return info_dict

