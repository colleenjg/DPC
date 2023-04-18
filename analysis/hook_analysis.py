import copy
import logging
import time

import numpy as np
import pandas as pd
import pickle as pkl
import torch

from utils import loss_utils, misc_utils, training_utils
from analysis import analysis_utils, hooks, rsm, umap, usi_diffs

NUM_STEPS = 5

TAB = "    "

logger = logging.getLogger(__name__)


#############################################
def get_unique_class_idxs(targets, dataset):
    """
    get_unique_class_idxs(targets, dataset)
    
    Converts the list of targets for the dataset into indices for each class.

    Required args
    -------------
    - targets : 3D array
        Target array, structured as B x P x labels [class label, unexp]
    - dataset : torch data.Dataset
        Dataset on which data was collected to use to convert targets to 
        classes.
    
    Returns
    -------
    - all_idxs : list
        Indices of the original target array for each unique target class, and 
        prediction step, structured as class x pred_step  
    - ns : 2D array
        The number of occurrences
    - targ_classes : list
        List of tuples, where each tuple specifies the Gabor image and 
        orientation combination for each unique target. 
    """

    if len(targets.shape) != 3:
        raise ValueError(
            f"'target' should be 3D, but found {len(targets.shape)}D."
            )

    # larger than the binary unexpected value
    max_val = max(2, targets[..., 1].max() + 1)
    factor = int(10 ** np.ceil(np.log10(max_val)))

    # get unique values
    targets_mod = targets[..., 0] * factor + targets[..., 1]
    P = targets_mod.shape[1]
    unique = np.sort(np.unique(targets_mod))

    # collect indices for each target value combination
    all_idxs = []
    ns = np.zeros((len(unique), P))
    for t, target_mod in enumerate(unique):
        all_idxs.append([])
        for p in range(P):
            idxs = np.where(targets_mod[:, p] == target_mod)[0]
            all_idxs[-1].append(idxs)
            ns[t, p] = len(idxs)
    
    # get unique classes for each modified target
    labels = [val // factor for val in unique]
    unexps = [val - label * factor for val, label in zip(unique, labels)]
    targ_classes = dataset.image_label_to_class(labels, unexps)

    return all_idxs, ns, targ_classes


#############################################
def get_class_idxs(targets, dataset):
    """
    get_class_idxs(targets, dataset)

    Converts the list of targets for the dataset into indices for each class, 
    including combined classes, where one of both items (Gabor image and ori) 
    is set to 'any'. 

    Required args
    -------------
    - targets : 3D array
        Target array, structured as B x P x labels [class label, unexp]
    - dataset : torch data.Dataset
        Dataset on which data was collected to use to convert targets to 
        classes.
    
    Returns
    -------
    - all_idxs : list
        Indices of the original target array for each unique target class, and 
        prediction step, structured as class x pred_step  
    - ns : 2D array
        The number of occurrences of each class, with dims: class x pred_step
    - targ_classes : list
        List of tuples, where each tuple specifies the class Gabor image and 
        orientation. The final classes have 'any' for one or both values.
    """

    all_idxs, ns, targ_classes = get_unique_class_idxs(targets, dataset)
    class_images, class_oris = zip(*targ_classes)
    B, P, _ = targets.shape

    # combine data for image/ori values of 'any'
    comb_ns = [[] for _ in range(P)]
    for val in set(class_images + class_oris):
        if val in class_images:
            unique_idxs = [i for i, v in enumerate(class_images) if v == val]
            targ_classes.append((val, "any"))
        else:
            unique_idxs = [i for i, v in enumerate(class_oris) if v == val]
            targ_classes.append(("any", val))

        all_idxs.append([])
        for p in range(P):
            idxs = [all_idxs[i][p] for i in unique_idxs]
            idxs = np.sort(np.concatenate(idxs))
            all_idxs[-1].append(idxs)
            comb_ns[p].append(len(idxs))
    
    # add numbers for all data
    targ_classes.append(("any", "any"))
    all_idxs.append([])
    for p in range(P):
        all_idxs[-1].append(np.arange(B))
        if ns[:, p].sum() != B:
            raise NotImplementedError(
                "Implementation error. Sum across unique classes should be "
                f"the batch size ({B}), but found ({ns[:, p].sum()})."
                )
        comb_ns[p].append(ns[:, p].sum())
    
    # combine data
    comb_ns = np.asarray(comb_ns).T
    ns = np.concatenate([ns, comb_ns], axis=0)

    return all_idxs, ns, targ_classes


#############################################
def compile_data(ep_hook_dict, dataset):
    """
    compile_data(ep_hook_dict, dataset)

    Compiles data from the epoch hook dictionary, and returns a lighter hook 
    dictionary, as well as a UMAP and an RSM dictionary.

    Required args
    -------------
    - ep_hook_dict : dict
        Epoch hook dictionary, with keys:
        'losses' (list)      : losses for each batch, structured as 
                               B x P x D_out x D_out
        'targets' (list)     : targets for each batch, structured as 
                               B x P x labels [class label, unexp]
        for hook_module in ["contextual", "encoder", "predictive"]
            for hook_type in ["activations", "errors"]
                '{hook_module}''{hook_type}' (torch Tensor): 
                    hook values, with dims B x P x C x D_out x D_out
    - dataset : torch data.Dataset
        Dataset on which data was collected.
    
    Returns
    -------
    - ep_hook_dict : dict
        Lighter epoch hook dictionary, with keys:
        'losses' (dict)      : loss statistics for each class 
                               ('mean' and 'std'), with dims: 
                               unique classes x P x D_out x D_out
        'targ_classes' (list): List of tuples, where each tuple specifies the 
                               class Gabor image and orientation. The final 
                               classes have 'any' for one or both values.
        'ns' (2D array)      : The number of occurrences of each class, 
                               with dims: class x pred_step
        for hook_module in ["contextual", "encoder", "predictive"]
            for hook_type in ["activations", "errors"]
                '{hook_module}''{hook_type}' (5D array): 
                    hook statistics across trials for each class 
                    ('mean' and 'std'), with dims: 
                    unique classes x P x C x D_out x D_out
    - ep_umap_df : pd.DataFrame
        Epoch UMAP dataframe, with, for each datapoint, the columns:
        'dim0'       : value in the first UMAP dimension
        'dim1'       : value in the second UMAP dimension
        'image'      : Gabor image 
        'orientation': Gabor orientation
        'hook_module': hook module, i.e., 'encoder', 'contextual', 'predictive'
        'hook_type'  : hook type, i.e., 'activations' or 'errors'
        'pred_step'  : predicted step
        'batch_num'  : batch number
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
    """

    ep_hook_dict = copy.deepcopy(ep_hook_dict)
    ep_rsm_dict = dict()

    targets = ep_hook_dict.pop("targets")
    all_idxs, ns, targ_classes = get_class_idxs(targets, dataset)

    targ_classes_flat = dataset.image_label_to_class(
        targets[..., 0].reshape(-1), targets[..., 1].reshape(-1)
        )
    idxs_im, idxs_ori = rsm.get_rsm_sorting_idxs(*list(zip(*targ_classes_flat)))

    umap_dfs = []
    for hook_module in hooks.HOOK_MODULES:
        ep_rsm_dict[hook_module] = dict()
        for hook_type in hooks.HOOK_TYPES:
            data = ep_hook_dict[hook_module].pop(hook_type)
            B = len(data)
            data = torch.cat(data).numpy()
            N_all, P, C, D, _ = data.shape

            # UMAP dataframe
            umap_df = umap.get_umap_df(
                data.mean(axis=(3, 4)).reshape(N_all * P, C), targ_classes_flat, 
                hook_module=hook_module, hook_type=hook_type, 
                num_batches=B, pred_step=P
                )
            umap_dfs.append(umap_df)

            # RSM
            rsm_binned, rsm_bin_vals = rsm.get_binned_rsm_data(data)
            ep_rsm_dict[hook_module][hook_type] = rsm_binned.astype(np.uint8)
            del rsm_binned

            # means and stds across trials
            new_shape = (len(targ_classes), P, C, D, D)
            ep_hook_dict[hook_module][hook_type] = {
                stat: np.full(new_shape, np.nan) for stat in ["means", "stds"]
                }
            for i, pred_idxs in enumerate(all_idxs):
                for p, idxs in enumerate(pred_idxs):
                    if not len(idxs):
                        continue
                    ep_hook_dict[hook_module][hook_type]["means"][i, p] = \
                        data[idxs, p].mean(axis=0)
                    ep_hook_dict[hook_module][hook_type]["stds"][i, p] = \
                        data[idxs, p].std(axis=0)

    # collect loss data by class
    loss_shape = (len(targ_classes), P, D, D)
    losses = {stat: np.full(loss_shape, np.nan) for stat in ["means", "stds"]}
    for i, pred_idxs in enumerate(all_idxs):
        for p, idxs in enumerate(pred_idxs):
            losses["means"][i, p] = \
                ep_hook_dict["losses"][idxs, p].mean(axis=0)
            losses["stds"][i, p] = \
                ep_hook_dict["losses"][idxs, p].std(axis=0)
    ep_hook_dict["losses"] = losses

    # hooks data
    ep_hook_dict["targ_classes"] = targ_classes # unique, ordered classes
    ep_hook_dict["ns"] = ns # number per targ class
    
    # UMAP data
    ep_umap_df = pd.concat(umap_dfs)

    # RSM data
    ep_rsm_dict["idxs_im"] = idxs_im
    ep_rsm_dict["idxs_ori"] = idxs_ori
    ep_rsm_dict["rsm_bin_vals"] = rsm_bin_vals

    return ep_hook_dict, ep_umap_df, ep_rsm_dict


#############################################
def plot_save_hook_data(hook_dict, epoch_str="0", pre=True, learn=False, 
                        output_dir=".", save_data=False):
    """
    plot_save_hook_data(hook_dict)

    Saves hook data, and plots USI and U/D differences.

    Required args
    -------------
    - hook_dict : dict
        Epoch hook dictionary, with keys:
        'losses' (dict)      : loss statistics for each class 
                               ('mean' and 'std'), with dims: 
                               unique classes x P x D_out x D_out
        'targ_classes' (list): List of tuples, where each tuple specifies the 
                               class Gabor image and orientation. The final 
                               classes have 'any' for one or both values.
        for hook_module in ["contextual", "encoder", "predictive"]
            for hook_type in ["activations", "errors"]
                '{hook_module}''{hook_type}' (5D array): 
                    hook statistics for each class ('mean' and 'std'), 
                    with dims: unique classes x P x C x D_out x D_out

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
    - save_data : bool (default=False)
        If True, data is saved to h5 file in the output directory.
    """

    epoch_n = analysis_utils.get_digit_in_key(epoch_str)

    plot_kw = {
        "resp_ep_ns": usi_diffs.RESP_EP_NS,
        "diff_ep_ns": usi_diffs.DIFF_EP_NS,
        "corr_ep_ns": usi_diffs.CORR_EP_NS,
        "pre": pre,
        "learn": learn,
    }

    ep_n_keys = ["resp_ep_ns", "diff_ep_ns", "corr_ep_ns"]
    all_incl_ep_ns = np.unique(np.concatenate(
        [np.asarray(plot_kw[key]).reshape(-1) for key in ep_n_keys]
    ))

    if save_data or epoch_n in all_incl_ep_ns:
        analysis_utils.save_dict_pkl(
            hook_dict, output_dir=output_dir, epoch_str=epoch_str, pre=pre, 
            learn=learn, dict_type="hook", 
            )

    if epoch_n != all_incl_ep_ns.max(): # plot during last epoch
        return

    hook_path = analysis_utils.get_dict_path(output_dir, dict_type="hook")
    with open(hook_path, "rb") as f:
        hook_dict = pkl.load(f)
    
    if "post" in hook_dict.keys() and pre: # plot during post
        return

    hook_df = usi_diffs.get_hook_df(hook_dict)

    plot_err = usi_diffs.check_plotting(hook_df, raise_err=False, **plot_kw)
    if len(plot_err):
        logger.warning(plot_err)
        return

    usi_diffs.plot_all(
        hook_df, output_dir=hook_path.parent, log_stats=False, **plot_kw
        )
     

#############################################
def plot_and_save_results(ep_hook_dict, dataset, epoch_str="0", pre=True, 
                          learn=False, output_dir="."):
    """
    plot_and_save_results(ep_hook_dict, dataset)

    Plots and saves results for the epoch.

    Required args
    -------------
    - ep_hook_dict : dict
        Epoch hook dictionary, with keys:
        'losses' (list)      : losses for each batch, structured as 
                               B x P x D_out x D_out
        'targets' (list)     : targets for each batch, structured as 
                               B x P x labels [class label, unexp]
        for hook_module in ["contextual", "encoder", "predictive"]
            for hook_type in ["activations", "errors"]
                '{hook_module}''{hook_type}' (torch Tensor): 
                    hook values, with dims B x P x C x D_out x D_out
    - dataset : torch data.Dataset
        Dataset on which data was collected.

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

    logger.info(
        f"2/{NUM_STEPS}. Loading saved data..."
        )
    start_time = time.perf_counter()


    stop_time = time.perf_counter()
    time_str = misc_utils.format_time(stop_time - start_time, sep_min=True)
    logger.info(
        f"{TAB}[{time_str}] Saved data loaded."
        f"\n3/{NUM_STEPS}. Compiling data, and plotting and saving UMAP data..."
        )
    start_time = time.perf_counter()
    ep_hook_dict, ep_umap_df, ep_rsm_dict = compile_data(
        ep_hook_dict, dataset
        )
    umap.plot_save_umaps(
        ep_umap_df, epoch_str=epoch_str, pre=pre, learn=learn, 
        output_dir=output_dir, save_data=True
        )


    stop_time = time.perf_counter()
    time_str = misc_utils.format_time(stop_time - start_time, sep_min=True)
    logger.info(
        f"{TAB}[{time_str}] Data compiled, and UMAP data plotted and saved."
        f"\n4/{NUM_STEPS}. Plotting and saving RSM data..."
        )
    start_time = time.perf_counter() 
    rsm.plot_save_rsms(
        ep_rsm_dict, epoch_str, pre=pre, learn=learn, 
        output_dir=output_dir, save_data=True
        )


    stop_time = time.perf_counter()
    time_str = misc_utils.format_time(stop_time - start_time, sep_min=True)
    logger.info(
        f"{TAB}[{time_str}] RSM data plotted and saved."
        f"\n5/{NUM_STEPS}. Saving and plotting summarized USI/diffs data."
        )
    start_time = time.perf_counter() 
    plot_save_hook_data(
        ep_hook_dict, epoch_str, pre=pre, learn=learn, 
        output_dir=output_dir, save_data=False
        )


    stop_time = time.perf_counter()
    time_str = misc_utils.format_time(stop_time - start_time, sep_min=True)
    logger.info(f"{TAB}[{time_str}] Gabor analysis complete.\n")

    return


#############################################
def collect_save_hook_data(model, dataloader, optimizer, device="cuda", 
                           output_dir=".", epoch_str="0", pre=True, 
                           learn=False):
    """
    collect_save_hook_data(model, dataloader, optimizer)

    Collects and saves hook data for the epoch.

    Required args
    -------------
    - model : torch nn.Module or nn.DataParallel
        Model or wrapped model.
    - dataloader : torch data.DataLoader
        Torch dataloader to use for hook data collection.
    - optimizer : torch.optim object
        Torch optimizer.

    Optional args
    -------------
    - device : torch.device or str (default="cpu")
        Device on which to train the model.
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
    - model : torch nn.Module
        Model.
    """

    analysis_utils.run_analysis_checks(model, dataloader)

    # Do not use DataParallel
    model = training_utils.get_model_only(model).to(device)

    learn_str_pr = " (no learning)" 
    if learn:
        factor = 0.1
        model, optimizer = training_utils.get_new_model_optimizer(
            model, optimizer, factor=factor
            )
        learn_str_pr = f" (learning at a {factor}x rate.)"
        epoch_str = f"{epoch_str}_learning"
    else:
        model.eval()

    criterion, criterion_no_reduction = loss_utils.get_criteria(device=device)

    ep_hook_dict, activation_hooks, error_hooks = hooks.init_hook_dict(
        model, batch_size=dataloader.batch_size
        )

    for key in ["losses", "targets"]:
        ep_hook_dict[key] = []

    logger.info(
        "Running extended Gabor analysis\n"
        f"1/{NUM_STEPS}. Recording activations and errors{learn_str_pr}...", 
        extra={"spacing": "\n"}
        )

    start_time = time.perf_counter()
    for idx, (input_seq, sup_target) in enumerate(dataloader):
        input_seq = input_seq.to(device)
        input_seq_shape = input_seq.size()
        [output_, mask_] = model(input_seq)

        # get targets, and reshape for loss calculation
        output_flattened, target_flattened, loss_reshape, _ = \
            training_utils.prep_loss(
                output_, mask_, sup_target, input_seq_shape, 
                supervised=False, is_gabor=True
                )
        D = int(np.sqrt(loss_reshape[-1]))
        loss_reshape = loss_reshape[:-1] + (D, D)

        target_flattened = target_flattened.to(device)
        loss = criterion(output_flattened, target_flattened)
        loss_full = criterion_no_reduction(
            output_flattened, target_flattened
            ).detach()
        
        model.zero_grad()
        if learn:
            optimizer.zero_grad()

        loss.backward()
        if learn:
            optimizer.step()

        model.zero_grad()

        # retain one target per block
        ep_hook_dict["targets"].append(sup_target[:, -model.pred_step:, 0].cpu())
        ep_hook_dict["losses"].append(loss_full.reshape(loss_reshape).cpu())
    
    for hook in activation_hooks + error_hooks:
        hook.remove()

    for key in ["losses", "targets"]:
        ep_hook_dict[key] = torch.cat(ep_hook_dict[key]).numpy()

    stop_time = time.perf_counter()
    time_str = misc_utils.format_time(stop_time - start_time, sep_min=True)
    start_time = time.perf_counter() 
    logger.info(f"{TAB}[{time_str}] Activations and errors recorded.")

    plot_and_save_results(
        ep_hook_dict, dataloader.dataset, epoch_str=epoch_str, pre=pre, 
        learn=learn, output_dir=output_dir
        )

    return model
