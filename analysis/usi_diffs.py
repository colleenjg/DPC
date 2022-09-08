import numpy as np
import pandas as pd
import scipy.stats

from analysis import analysis_utils, hooks

#############################################
def np_pearson_r(x, y, axis=0, nanpol=None):
    """
    np_pearson_r(x, y)

    Returns Pearson R correlation for two matrices, along the specified axis.

    Required args
    -------------
    - x : nd array
        First array to correlate
    - y : nd array
        Second array to correlate (same shape as x)
    
    Optional args
    -------------
    - axis : int (default=0)
        Axis along which to correlate values
    - nanpol : bool (default=None) 
        NaN policy

    Returns
    -------
    - corr_bounded : nd array
        Correlation array, with one less dimension than x/y.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if nanpol not in [None, "omit"]:
        raise ValueError("'nanpol' must be None or 'omit'.")

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")

    mean_fct = np.mean if nanpol is None else np.nanmean
    sum_fct = np.sum if nanpol is None else np.nansum


    x_centered = x - mean_fct(x, axis=axis, keepdims=True)
    y_centered = y - mean_fct(y, axis=axis, keepdims=True)
    x_centered_ss = sum_fct(x_centered * x_centered, axis=axis)
    y_centered_ss = sum_fct(y_centered * y_centered, axis=axis)

    corr = (
        sum_fct(x_centered * y_centered, axis=axis) /
        np.sqrt(x_centered_ss * y_centered_ss)
        )
    # bound the values to -1 to 1 (for precision problems) -> propagate NaNs
    corr_bounded = np.maximum(np.minimum(corr, 1.0), -1.0)

    return corr_bounded


#############################################
def bootstrapped_corr(data1, data2, seed=100, n_samples=1000):
    """
    bootstrapped_corr(data1, data2)

    Returns bootstrapped correlation values.

    Required args
    -------------
    - data1 : array-like
        Array of data points.
    - data2 : array-like
        Array with the same length as data1.

    Optional args
    -------------
    - seed : int (default=100)
        Seed to use for resampling the data.
    - n_samples : int (default=1000)
        Number of data resamplings to perform.

    Returns
    -------
    - bootstrapped_std : float
        Bootstrapped standard deviation computed across correlations of 
        resampled data.
    """
    randst = np.random.RandomState(seed)

    data = np.stack([data1, data2])
    n = data.shape[1]
    choices = np.arange(n)

    rand_data1, rand_data2 = data[
        :, randst.choice(choices, (n, n_samples), replace=True)
        ]
    rand_corrs = np_pearson_r(rand_data1, rand_data2, axis=0)

    bootstrapped_std = np.std(rand_corrs)

    return bootstrapped_std


#############################################
def get_mean_sem_USI(sub_dict, df, common, targ_classes, ns, pop_indiv=False):
    """
    get_mean_sem_USI(sub_dict, df, common, targ_classes, ns)

    Populates the input dataframe with data statistics from the dictionary 
    passed. Also returns a dictionary with individual data for aggregating 
    across experiments.

    Required args
    -------------
    - sub_dict : dict
        Hook statistics across trials for the class ('mean' and 'std'), 
        with dims: unique classes x P x C x D_out x D_out
    - df : pd.DataFrame
        Dataframe to which to add statistics across nodes, U-D differences, 
        USIs and p-values. 
    - common : dict
        Dictionary with keys and values that are common to all data to be 
        collected and should be entered into each new row of the dataframe.
    - targ_classes : list
        List of tuples, where each tuple specifies the class Gabor image and 
        orientation. The final classes have 'any' for one or both values.
    - ns : 2D array
        The number of occurrences of each class, with dims: class x pred_step

    Optional args
    -------------
    - pop_indiv : bool (default=False)
        If True, an dictionary with individual data per node is returned with 
        means for each Gabor image, as well as U-D values and USIs.

    Returns
    -------
    - df : pd.DataFrame
        Updated dataframe, with common key columns, as well as
        'image'      : type of Gabor image (or 'U-D', 'USI', 'USI_abs')
        'orientation': Gabor orientation
        'num_trials' : number of trials used for this data
        'mean'       : mean across nodes
        'std'        : standard deviation across nodes
        'sem'        : SEM across nodes
    - indiv_dict : dict
        Dictionary with data for individual nodes, if pop_indiv is True, 
        with keys:
        '{image}': node means for the specified Gabor image

    """
    
    indiv_dict = dict()
    for i, ((image, ori), n) in enumerate(zip(targ_classes, ns)):
        idx = len(df)
        for k, v in common.items():
            df.loc[idx, k] = v
        df.loc[idx, "image"] = image
        df.loc[idx, "orientation"] = ori
        df.loc[idx, "num_trials"] = sum(n)

        mean, sems, stds = np.nan, np.nan, np.nan
        if n > 0:
            mean = sub_dict["means"][i].mean()
            sems = scipy.stats.sem(sub_dict["means"][i].reshape(-1))
            stds = sub_dict["means"][i].reshape(-1).std()
        df.loc[idx, "mean"] = mean
        df.loc[idx, "sem"] = sems
        df.loc[idx, "std"] = stds

        if ori == "any" and (pop_indiv or image in ["D", "U"]):
            indiv_dict[image] = sub_dict["means"][i].reshape(-1)
            if image in ["D", "U"]:
                indiv_dict[f"{image}_stds"] = sub_dict["stds"][i].reshape(-1)
    
    # add diffs and USIs
    if "D" in indiv_dict.keys() and "U" in indiv_dict.keys():
        for image in ["U-D", "USI"]:
            idx = len(df)
            for k, v in common.items():
                df.loc[idx, k] = v
            df.loc[idx, "image"] = image
            df.loc[idx, "orientation"] = "any"
            diffs = indiv_dict["U"] - indiv_dict["D"]

            if image == "U-D":                            
                df.loc[idx, "mean"] = diffs.mean()
                df.loc[idx, "sem"] = scipy.stats.sem(diffs)
                df.loc[idx, "std"] = diffs.std()

            elif image == "USI":
                stds = [indiv_dict.pop(f"{im}_stds") for im in ["U", "D"]]
                div = np.sqrt(0.5 * np.sum(np.power(stds, 2), axis=0))
                all_USIs = diffs / div
                df.loc[idx, "mean"] = all_USIs.mean()
                df.loc[idx, "sem"] = scipy.stats.sem(all_USIs)
                df.loc[idx, "std"] = all_USIs.std()

                # also add absolute values
                idx = len(df)
                for k, v in common.items():
                    df.loc[idx, k] = v
                df.loc[idx, "image"] = "USI_abs"
                df.loc[idx, "orientation"] = "any"                
                df.loc[idx, "mean"] = np.absolute(all_USIs).mean()
                df.loc[idx, "sem"] = scipy.stats.sem(np.absolute(all_USIs))
                df.loc[idx, "std"] = np.absolute(all_USIs).std()

                if pop_indiv:
                    indiv_dict["USI"] = all_USIs
                    indiv_dict["USI_abs"] = np.absolute(all_USIs)
    
        for key in indiv_dict.keys():
            if "stds" in key:
                indiv_dict.pop(key)
    
    if not pop_indiv:
        indiv_dict = dict()

    return df, indiv_dict


#############################################
def get_hook_df(hook_dicts):
    """
    get_hook_df(hook_dicts)
    
    Compiles a dataframe with hook data.

    Required args
    -------------
    - hook_dicts : list
        Hook dictionaries for different models, each with keys:
        '{epoch_str}' (dict): dictionary with key specifying the epoch number, 
                              with 'unexp' in it, if the epoch included 
                              unexpected events.
            '{pre_str}' (dict): dictionary with key specifying whether data 
                                was collected before ('pre') or after ('post') 
                                epoch training.
                'losses' (dict)      : loss statistics for each class 
                                       ('mean' and 'std'), with dims: 
                                       unique classes x P x D_out x D_out
                'targ_classes' (list): List of tuples, where each tuple 
                                       specifies the class Gabor image and 
                                       orientation. The final classes have 
                                       'any' for one or both values.
                'ns' (2D array)      : The number of occurrences of each class, 
                                       with dims: class x pred_step
                for hook_module in ["contextual", "encoder", "predictive"]
                    for hook_type in ["activations", "errors"]
                        '{hook_module}''{hook_type}' (5D array): 
                            hook statistics across trials for each class 
                            ('mean' and 'std'), with dims: 
                            unique classes x P x C x D_out x D_out

    Returns
    -------
    - hook_df : pd.DataFrame
        Dataframe with hook data, with columns:
        - hook_module     : Module from which the data is taken, 
                            e.g., 'encoder', 'contextual' or 'predictive'
        - hook_type       : Type of data, e.g., 'activations' or 'errors'
        - unexp           : Whether session(s) included unexpected data.
        - pre_post        : Whether data is 'pre' or 'post' learning for the epoch.
        - epoch_n         : Epoch number(s).
        - model_n         : Model number (index).
        - image           : Gabor image (or 'U-D', 'USI', 'USI_abs', 'USI_corr').
        - orientation     : Gabor mean orientation.
        - num_trials      : Number of trials underlying data.
        - mean            : Mean across nodes.
        - sem             : SEM across nodes.
        - std             : Standard deviation across nodes.
        - corr            : Correlation between USIs.
        - corr_std        : Bootstrapped standard deviation of USI correlation.
        - pval            : Paired t-test p-value
        - pval_ep{e1}v{e2}: Paired t-test p-value between sessions.
    """

    hook_df = pd.DataFrame(
        columns=[
            "hook_module", "hook_type", "unexp", "pre", "epoch_n", "model_n", 
            "image", "orientation", "num_trials", "mean", "sem", "std", "corr", 
            "corr_std", "pval",
            ]
        )
    
    all_pre_keys = [sorted(hook_dict.keys()) for hook_dict in hook_dicts]
    for k, keys in enumerate(all_pre_keys[:-1]):
        if keys != all_pre_keys[k + 1]:
            raise ValueError("All hook_dicts must have the same pre/post keys.")
    pre_keys = all_pre_keys[0]
            
    all_epoch_keys = [
        sorted(pre_dict.keys()) for hook_dict in hook_dicts 
        for pre_dict in hook_dict.values() 
        ]
    for k, keys in enumerate(all_epoch_keys[:-1]):
        if keys != all_epoch_keys[k + 1]:
            raise ValueError("All hook_dicts must have the same epoch keys.")
    epoch_keys = all_epoch_keys[0]

    epoch_ns = [
        analysis_utils.get_digit_in_key(key) for key in all_epoch_keys[0]
        ]
    pre_keys = np.sort(np.unique(all_pre_keys[0]))

    seed = 100
    for hook_module in hooks.HOOK_hook_moduleS:
        for hook_type in hooks.HOOK_TYPES:
            for pre_key in pre_keys:
                USI_dict = dict()
                diff_dict = dict()
                for e in np.argsort(epoch_ns):
                    epoch_n = epoch_ns[e]
                    epoch_key = epoch_keys[e]
                    common = {
                        "hook_hook_module": hook_module,
                        "hook_type": hook_type,
                        "epoch_n": epoch_n,
                        "pre_post": pre_key,
                    }

                    full_dict = dict()
                    unexps = []
                    for h, hook_dict in enumerate(hook_dicts):
                        sub_dict = hook_dict[pre_key][epoch_key][hook_module][hook_type]
                        unexps.append("unexp" in key)
                        common["unexp"] = unexps[-1]
                        common["model_n"] = h

                        hook_df, indiv_dict = get_mean_sem_USI(
                            sub_dict, hook_df, common,
                            hook_dict[key]["targ_classes"], 
                            hook_dict[key]["ns"],
                            pop_indiv=(epoch_n in [-1, 0, 1, 2])
                            )
                        for key, data in indiv_dict.items():
                            if key not in full_dict.keys():
                                full_dict[key] = []
                            full_dict[key].append(data)

                    # get the overall stats
                    common["model_n"] = "all"
                    common["unexp"] = unexps[0] if len(set(unexps)) == 1 else None
                    for key, data in full_dict.items():
                        idx = len(hook_df)
                        for k, v in common.items():
                            hook_df.loc[idx, k] = v
                        hook_df.loc[idx, "image"] = key
                        hook_df.loc[idx, "orientation"] = "any"
                        data = np.concatenate(data)
                        hook_df.loc[idx, "mean"] = data.mean()
                        hook_df.loc[idx, "sem"] = scipy.stats.sem(data)
                        hook_df.loc[idx, "std"] = data.std()

                        if key in ["USI", "USI_abs"]:
                            hook_df.loc[idx, "pval"] = scipy.stats.ttest_rel(
                                data, np.zeros_like(data)
                                )[1]
                            if epoch_n in [0, 1, 2]:
                                USI_dict[epoch_n] = (data, idx)

                        if key == "U" and "D" in full_dict.keys():
                            idx = len(hook_df)
                            for k, v in common.items():
                                hook_df.loc[idx, k] = v
                            hook_df.loc[idx, "image"] = "U-D"
                            hook_df.loc[idx, "orientation"] = "any"
                            diffs = data - np.concatenate(full_dict["D"])
                            hook_df.loc[idx, "mean"] = diffs.mean()
                            hook_df.loc[idx, "sem"] = scipy.stats.sem(diffs)
                            hook_df.loc[idx, "std"] = diffs.std()
                            hook_df.loc[idx, "pval"] = scipy.stats.ttest_rel(
                                diffs, np.zeros_like(diffs)
                                )[1]

                            if epoch_n in [0, 1, 2]:
                                diff_dict[epoch_n] = (diffs, idx)
                            del diffs
                    del full_dict

            # get the between session comps
            for use_dict in diff_dict, USI_dict:
                if len(use_dict) == 0:
                    continue
                for e1, e2 in [(0, 1), (1, 2), (0, 2)]:
                    if e1 in use_dict.keys() and e2 in use_dict.keys():
                        data1, idx1 = use_dict[e1]
                        data2, idx2 = use_dict[e2]
                        p_val_diff = scipy.stats.ttest_rel(data1, data2)[1]
                        hook_df.loc[idx1, f"pval_ep{e1}v{e2}"] = p_val_diff
                        hook_df.loc[idx2, f"pval_ep{e1}v{e2}"] = p_val_diff

            if len(USI_dict):
                # add USI correlations
                for e1, e2 in [(0, 1), (1, 2)]:
                    if e1 in use_dict.keys() and e2 in use_dict.keys():
                        data1, _ = use_dict[e1]
                        data2, _ = use_dict[e2]

                        idx = len(hook_df)
                        for k, v in common.items():
                            hook_df.loc[idx, k] = v
                        hook_df.loc[idx, "image"] = "USI_corr"
                        hook_df.loc[idx, "orientation"] = "any"
                        hook_df.loc[idx, "epoch_n"] = f"{e1}v{e2}"

                        corr, pval = scipy.stats.pearsonr(data1, data2)
                        corr_std = bootstrapped_corr(data1, data1, seed=seed)
                        
                        hook_df.loc[idx, "corr"] = corr
                        hook_df.loc[idx, "corr_std"] = corr_std
                        hook_df.loc[idx, "pval"] = pval

                        seed += 1

    return hook_df


