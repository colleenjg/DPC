import logging

import numpy as np

from dataset import gabor_stimuli
from utils import loss_utils


logger = logging.getLogger(__name__)


#############################################
def check_if_is_gabors(dataset):
    """
    check_if_is_gabors(dataset)
    """

    is_gabors = isinstance(dataset, gabor_stimuli.GaborSequenceGenerator)

    return is_gabors


#############################################
def init_gabor_records(dataset):
    """
    init_gabor_records(dataset)
    """
    
    gabor_mean_oris = dataset.all_mean_oris

    gabor_frames = list(dataset.image_type_dict.keys())

    loss_dict = dict()
    acc_dict = dict()

    for gabor_frame in gabor_frames:
        loss_dict[gabor_frame] = dict()
        acc_dict[gabor_frame] = dict()
        for gabor_mean_ori in gabor_mean_oris:
            loss_dict[gabor_frame][gabor_mean_ori] = loss_utils.AverageMeter()
            acc_dict[gabor_frame][gabor_mean_ori] = loss_utils.AverageMeter()

    n_categories = len(gabor_frames) * len(gabor_mean_oris)
    confusion_mat = loss_utils.ConfusionMeter(n_categories)

    return loss_dict, acc_dict, confusion_mat


#############################################
def update_records(loss_dict, acc_dict, output, target, batch_loss, 
                   supervised=False, sup_target=None, confusion_mat=None):

    if supervised:
        raise NotImplementedError(
            "Updating records is not implemented for supervised learning."
            )
    elif sup_target is None:
        raise ValueError("If 'supervised' is False, must pass 'sup_target'.")

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
        output, keep_topk=1, acc_avg_HW=True, main_shape=main_shape
        )

    breakpoint()

    # using sup_target, convert predictions and targets to 
    # gabor frame x gabor mean orientation
    
    # calculate accuracy based on these new labels/targets

    # obtain OVERALL accuracy, as well as broken down accuracy

    # update confusion matrix



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

