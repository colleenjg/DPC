from pathlib import Path
import re

import pickle as pkl

from analysis import hooks
from dataset import gabor_sequences
from utils import training_utils


RSM_DIR = "rsms"
UMAP_DIR = "umaps"
HOOK_DIR = "hooks"


#############################################
def get_epoch_pre_str(epoch_str="0", pre=True, learn=False): 
    """
    get_epoch_pre_str()
    

    Return formatted epoch and pre/post string.

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

    Returns
    -------
    - epoch_str : str
        String providing epoch information, for saving.    
    - pre_str : str
        String providing pre/post and learning information, for saving.    
    """

    epoch_str = str(epoch_str)
    pre_str = "pre" if pre else "post"
    if learn:
        pre_str = f"{pre_str}_learning"

    return epoch_str, pre_str



#############################################
def get_dict_path(output_dir=".", dict_type="hook"):
    """
    get_dict_path()

    Returns path to the specified dictionary.

    Optional args
    -------------
    - output_dir : str or path (default=".")
        Main output directory in which to save collected data.
    - dict_type : str (default="hook")
        Type of dictionary for which to retrieve path, 
        i.e., 'hook', 'umap' or 'rsm'
    
    Returns
    -------
    - save_path : path
        Path to the specified dictionary.
    """

    if dict_type == "hook":
        save_path = Path(output_dir, HOOK_DIR, "hook_data.pkl")
    elif dict_type == "umap":
        save_path = Path(output_dir, UMAP_DIR, "umap_data.pkl")
    elif dict_type == "rsm":
        save_path = Path(output_dir, RSM_DIR, "rsm_data.h5")
    else:
        raise ValueError(f"{dict_type} not recognized.")

    return save_path


#############################################
def get_pkl_dict(output_dir=".", epoch_str="0", pre=True, learn=False, 
                 dict_type="hook"):
    """
    get_pkl_dict()

    Returns a pickled dictionary, or a new one, if it does not yet exist. 
    Checks that the pre/post and epoch keys do not yet exist in the dictionary, 
    and raises an error if they do. 

    Optional args
    -------------
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
    - dict_type : str (default="hook")
        Type of dictionary for which to retrieve path, 
        i.e., 'hook', 'umap' or 'rsm'
    
    Returns
    -------
    - data_dict : dict
        Data dictionary, with the epoch and pre/post key added. 
    """

    epoch_str, pre_str = get_epoch_pre_str(epoch_str, pre, learn)

    if dict_type == "rsm":
        raise ValueError("Use rsm.load_rsm_h5() instead.")

    save_path = get_dict_path(output_dir, dict_type=dict_type)

    save_path.parent.mkdir(parents=True, exist_ok=True)

    data_dict = dict()
    if save_path.is_file():
        with open(save_path, "rb") as f:
            data_dict = pkl.load(f)
    
    if pre_str not in data_dict.keys():
        data_dict[pre_str] = dict()
    if epoch_str in data_dict[pre_str].keys():
        raise RuntimeError(
            f"Dictionary for {dict_type} already contains a "
            f"{pre_str}/{epoch_str} key."
            )

    return data_dict


#############################################
def save_dict_pkl(ep_save_dict, output_dir=".", epoch_str="0", pre=True, 
                  learn=False, dict_type="hook"):
    """
    save_dict_pkl(ep_save_dict)

    Saves dictionary for the epoch into the overall dictionary. 

    Required args
    -------------
    - ep_save_dict : dict
        Dictionary in which epoch data is stored.

    Optional args
    -------------
    - output_dir : str or path (default=".")
        Main Main output directory in which to save collected data.
    - epoch_str : str (default: "0")
        String providing epoch information, for saving.
    - pre : bool (default: True)
        If True, hook data is being collected before epoch training in the main 
        loop.
    - learn : bool (default: False)
        If True, hook data is collected while model is being trained at a 
        slower rate.
    - dict_type : str (default="hook")
        Type of dictionary for which to retrieve path, 
        i.e., 'hook', 'umap' or 'rsm'
    
    Returns
    -------
    - save_direc : Path
        Save directory.
    """

    epoch_str, pre_str = get_epoch_pre_str(epoch_str, pre, learn)

    data_dict = get_pkl_dict(
        output_dir, epoch_str=epoch_str, pre=pre, learn=learn, 
        dict_type=dict_type
        )
    data_dict[pre_str][epoch_str] = ep_save_dict

    save_path = get_dict_path(output_dir, dict_type=dict_type)
 
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pkl.dump(data_dict, f)

    save_direc = save_path.parent
    
    return save_direc


#############################################
def run_analysis_checks(model, dataloader):
    """
    run_analysis_checks(model, dataloader)

    Checks whether the analysis conditions are met:
    - Gabor dataset
    - Frames are not shifted for sequence starts, and the Gabor image length is 
    the same as the sequence length, ensuring that each video block is a single 
    Gabor image.
    - Task is unsupervised.

    Raises an error if these conditions are not met.

    Optional args
    -------------
    - model : torch nn.Module or nn.DataParallel
        Model or wrapped model.
    - dataloader : torch data.DataLoader
        Torch dataloader to use for hook data collection.
    """

    _, supervised = training_utils.get_num_classes_sup(model)
    is_gabor = gabor_sequences.check_if_is_gabor(dataloader.dataset)

    if not is_gabor:
        raise NotImplementedError(
            "Analysis is only implemented for Gabor datasets."
            )
   
    if dataloader.dataset.shift_frames:
        raise NotImplementedError(
            "Analysis requires dataset shift_frames to be set to False."
            )
        
    if dataloader.dataset.gab_img_len != dataloader.dataset.seq_len:
        raise NotImplementedError(
            "Analysis requires dataset gab_img_len to be equal to seq_len."
        )

    if supervised:
        raise NotImplementedError(
            "Analysis is not implemented for supervised models."
            )


#############################################
def get_digit_in_key(key):
    """
    get_digit_in_key(key)

    Retrieves a digit (positive or negative) for an epoch string key.

    Required args
    -------------
    - key : str
        Dictionary key containing a digit.
    """

    digits = re.findall(r"\d+", key)
    if len(digits) != 1:
        raise ValueError(f"Expected to find exactly one digit in {key}.")

    digit = digits[0]
    idx = key.index(digit)
    if idx > 0 and key[idx - 1] == "-":
        digit = f"-{digit}"

    digit = int(digit)

    return digit

