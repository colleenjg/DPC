from datetime import datetime
import logging
import multiprocessing
from pathlib import Path
import time
import warnings

import numpy as np
import torch
import torch.optim as optim

from model import model_3d
from utils import gabor_utils, misc_utils

logger = logging.getLogger(__name__)

TAB = "    "

#############################################
def get_device(num_workers=None, cpu_only=False):
    """
    get_device()

    Returns a device on which the code can be run, and the number of workers 
    that should be used, based on the input parameters and available resources.

    Optional args
    -------------
    - num_workers : int (default=None)
    - cpu_only : bool (default=False)

    Returns
    -------
    - device : torch.device
        Device on which to run code.
    - num_workers : int
        Number of workers.
    """

    if torch.cuda.is_available() and not cpu_only:
        device_name = "cuda"
    else:
        device_name = "cpu"

    device = torch.device(device_name)

    if num_workers is None:
        num_cores = multiprocessing.cpu_count()
        num_devices = torch.cuda.device_count()
        num_workers = min(num_devices * 2, num_cores) # approx rule of thumb
    
    return device, num_workers


#############################################
def check_batch_size(batch_size=32, device_type="cpu", resume=False, 
                     dataset="UCF101", save_by_batch=False):
    """
    check_batch_size()

    Verifies whether the batch size is compatible with splitting data across 
    GPUs, and updates it to an acceptable value.

    NOTE: If batch_size is 1, it is not changed.

    Optional args
    -------------
    - batch_size : int (default=32)
    - device_type : str (default="cpu")
    - resume : bool

    Returns
    -------
    - batch_size : int
        Batch size updated based on the number of available devices.
    """

    if batch_size == 1:
        return batch_size

    if device_type == "cuda":        
        num_devices = torch.cuda.device_count()
        per_GPU = batch_size / num_devices
        if int(per_GPU) != per_GPU:
            new_batch_size = int(np.ceil(per_GPU)) * num_devices
            
            dataset = misc_utils.normalize_dataset_name(dataset)
            resume_str = ""
            if resume and (save_by_batch or dataset == "Gabors"):
                resume_str = (
                    "\nNote that resuming model training with a different "
                    "batch size (e.g., incremented due to a change in the "
                    "number of devices available for training) may create "
                    "errors if this has resulted in a change in the number of "
                    "batches per epoch, as batch-wise information is being "
                    "recorded."
                )

            logger.warning(
                "To enable equal batch distribution across GPUs, incrementing"
                f" batch size from {batch_size} to {new_batch_size}."
                f"{resume_str}"
                )
            batch_size = new_batch_size

    return batch_size


#############################################
def allow_data_parallel(dataloader, device, supervised=False):
    """
    allow_data_parallel(dataloader, device)

    Check whether the available resources and the task being run are compatible 
    with training a model in parallel over multiple GPUs, using 
    torch nn.DataParallel.

    Required args
    -------------
    - dataloader : torch data.DataLoader:
        Dataloader with which code will be run.
    - device : torch.device
        Device on which code will be run.

    Optional args
    -------------
    - supervised : bool (default=False)
        If True, the target task is supervised. If not, dataloader.drop_last 
        must be True for the model to be run in parallel.

    Returns
    -------
    - allow_parallel : bool
        Whether the model should be trained in parallel over multiple GPUs.
    """
    
    allow_parallel = False
    if device.type != "cpu" and torch.cuda.device_count() > 1:
        if supervised:
            allow_parallel = True
        else:
            if dataloader.drop_last:
                allow_parallel = True
            else:
                warnings.warn(
                    "Only 1 GPU will be used, as the dataloader is not set "
                    "to drop final batches, when they are smaller."
                    )

    return allow_parallel      


#############################################
def get_model(dataset="UCF101", supervised=True, img_dim=224, num_seq_in=5, 
              seq_len=5, network="resnet18", pred_step=3, dropout=0.5):
    """
    get_model()

    Returns a model, based on the parameters provided.

    Required args
    -------------
    - dataset : torch data.Dataset or str (default="UCF101")
        Torch dataset for which to retrieve the number of classes or name of 
        the dataset, if supervised is True.

    Optional args
    -------------
    - supervised : bool (default=True)
        If True, the supervised model is returned. Otherwise, the 
        self-supervised model is returned.
    - img_dim : int (default=256)
        Image dimensions, when they are input to the model.
    - num_seq_in : int (default=5)
            Number of consecutive sequences to use as input.
    - seq_len : int (default=5)
        Number of frames per sequence.
    - network : str (default="resnet18")
        Backbone network on which to build the model.
    - pred_step : int (default=3)
        Number of steps ahead to predict.
    - dropout : float (default=0.1)
        Dropout proportion for the dropout layer in the final fully 
        connected layer.

    Returns
    -------
    - model : model_3d.LC_RNN or model_3d.DPC_RNN
        Model.
    """

    model_kwargs = {
        "input_size": img_dim,
        "seq_len"   : seq_len,
        "network"   : network,
    }

    if supervised:
        num_classes = misc_utils.get_num_classes(dataset)

        model = model_3d.LC_RNN(
            num_classes=num_classes,
            dropout=dropout,
            num_seq=num_seq_in,
            **model_kwargs
            )
    
    else:
        model = model_3d.DPC_RNN(
            num_seq_in=num_seq_in,
            pred_step=pred_step, 
            **model_kwargs
            )

    return model


#############################################
def get_model_only(model):
    """
    get_model_only(model)

    Returns the input model, unwrapped if it is wrapped in 
    torch nn.DataParallel.

    Required args
    -------------
    - model : torch nn.Module or nn.DataParallel
        Model or wrapped model, to unwrap.

    Returns
    -------
    - model : torch nn.Module
        Model, not wrapped in torch nn.DataParallel.
    """
    
    if hasattr(model, "device_ids"):
        model = model.module

    return model


#############################################
def get_target_from_mask(mask):
    """
    get_target_from_mask(mask)
    
    Computes a target for computing a contrastive loss from the mask returned 
    by a DPC model. 

    NOTE: target.requires_grad is set to False. 

    Required args
    -------------
    - mask : 6D Tensor
        Mask indicating the nature of each predicted-ground truth pair, 
        where 1s indicate positive pairs, with dims: 
            B x P x D_out2 x B x P x D_out2.

    Returns
    -------
    - target : 6D Tensor
        Target mask identifying all positive pairs by 1s, and all other pairs 
        (negative) by 0s, with dims:
            B x P x D_out2 x B x P x D_out2.
    """
    
    # mask meaning: 1: pos
    target = (mask == 1).to(int)
    target.requires_grad = False

    return target


#############################################
def log_weight_decay_prop(model):
    """
    log_weight_decay_prop(model)

    Applies weight decay, and logs the proportion of weights decayed over the 
    total weights of the model. 

    Required args
    -------------
    - model : torch nn.Module or nn.DataParallel
        Model or wrapped model.
    """
    
    total_weight = 0.0
    decay_weight = 0.0
    for m in model.parameters():
        if m.requires_grad: 
            decay_weight += m.norm(2).data
        total_weight += m.norm(2).data

    perc = decay_weight / total_weight * 100
    logger.info(
        "Norm of weights to decay: "
        f"{decay_weight:.2f} / {total_weight:.2f} ({perc:05.2f}% of total)", 
        extra={"spacing": TAB}
        )


#############################################
def check_grad(model):
    """
    check_grad(model)

    Logs, at the DEBUG level, each named parameter, and whether it requires 
    a gradient. 

    Required args
    -------------
    - model : torch nn.Module or nn.DataParallel
        Model or wrapped model.
    """
    
    logger.debug("\n===========Check Grad============")
    for name, param in model.named_parameters():
        logger.debug(name, param.requires_grad)
    logger.debug("=================================\n")


#############################################
def set_gradients(model, supervised=False, train_what="all", lr=1e-3):
    """
    set_gradients(model)

    Sets the gradients for the parameters, based on the type of training to be 
    done, and returns a dictionary, for use with an optimizer, that specifies 
    the learning rate to use for certain parameters.

    Required args
    -------------
    - model : torch nn.Module or nn.DataParallel
        Model or wrapped model.

    Optional args
    -------------
    - supervised : bool (default=False)
        Determines which parameters should have their gradients modified if 
        train_what in 'last' or 'ft'. If True, both ResNet and RNN parameters 
        are affected. Otherwise, only ResNet parameters are affected.
    - train_what : str (default="all")
        Specifies how the parameter gradients should be set for different 
        parameters. If 'all', no changes are made to the parameters gradient 
        settings. If 'ft', a lower learning rate is specified for affected 
        parameters. If 'last', affected parameters are set to not require a 
        gradient.
    - lr : float (default=1e-3)
        Original learning rate, based on which the lower learning rate used for 
        fine-tuning is calculated.

    Returns
    -------
    - params : dict
        Parameter dictionary for use with a torch optimizer, optionally 
        specifying a learning rate for certain parameters. 
    """

    params = model.parameters()

    if train_what in ["last", "ft"]:
        affected_params = []
        unaffected_params = []
        for name, param in model.named_parameters():
            if "resnet" in name.lower():
                affected_params.append(param)
            elif supervised and "rnn" in name.lower():
                affected_params.append(param)
            else:
                unaffected_params.append(param)

        if train_what == "ft":
            logger.info("=> Finetuning backbone with a smaller learning rate.")

            params = []
            for param in affected_params:
                params.append({"params": param, "lr": lr / 10})
            for param in unaffected_params:
                params.append({"params": param})

        else:
            logger.info("=> Training only the last layer.")
            for param in affected_params:
                param.requires_grad = False

    elif train_what != "all":
        raise ValueError(
            f"{train_what} value for train_what is not recognized."
            )
    else: 
        pass # train all layers

    check_grad(model)

    return params
    
    
#############################################
def get_multistepLR_restart_multiplier(epoch_n, gamma=0.1, steps=[10, 15, 20], 
                                       num_cycles=3):
    """
    get_multistepLR_restart_multiplier(epoch_n)

    Returns a learning rate multiplier (gamma) based on the following pattern:

        if ep_n in:
            [0      to step 1[ : gamma^0
            [step 1 to step 2[ : gamma^1
            [step 2 to step 3[ : gamma^2

            [step 3 + step 1 to step 3 + step 2[ : gamma^0
            ...

        After all restarts are completed, the highest exponent value is used.

    Required args
    -------------
    - epoch_n : int
        Epoch number for which to calculate adjusted gamma.

    Optional args
    -------------
    - gamma : float (default=0.1)
        Base gamma value.
    - steps : list (default[10, 15, 20])
        Epochs at which to modify gamma.
    - num_cycles : int (default=3)
        Number of cycles to go through before settling on a consistent value.

    Returns
    -------
    - float
        Gamma value to use for the specified epoch number.
    """

    max_step = max(steps)
    effective_epoch = epoch_n % max_step
    if epoch_n // max_step >= num_cycles: # beyond final repeat
        exp = len(steps) - 1
    else:
        exp = sum([True for i in steps if effective_epoch >= i])

    return gamma ** exp


#############################################
def get_lr_lambda(dataset="UCF101", img_dim=256): 
    """
    get_lr_lambda()

    Returns a lambda function to use to calculate the learning rate multiplier 
    for a given epoch number.

    Optional args
    -------------
    - dataset : str (default="UCF101")
        Dataset name.
    - img_dim : int (default=256)
        Image dimensions, when they are input to the model.

    Returns
    -------
    - lr_lambda : lambda function
        Lambda function for calculating the learning rate multiplier for a 
        given epoch number. 
    """
    
    dataset = misc_utils.normalize_dataset_name(dataset)

    big = img_dim >= 224

    if dataset == "HMDB51":
        steps = [150, 250, 300] if big else [30, 40, 50]
        
    elif dataset == "UCF101":
        steps = [300, 400, 500] if big else [60, 80, 100]

    elif dataset == "Kinetics400":
        steps = [150, 250, 300] if big else [30, 40, 50]

    elif dataset == "MouseSim":
        steps = [300, 400, 500] if big else [60, 80, 100]

    elif dataset == "Gabors":
        steps = [20, 30, 40] if big else [10, 15, 20]

    else:
        raise ValueError(f"{dataset} dataset not recognized.")

    lr_lambda = lambda ep_n: get_multistepLR_restart_multiplier(
        ep_n, gamma=0.1, steps=steps, num_cycles=1
        )

    return lr_lambda


#############################################
def init_optimizer(model, lr=1e-3, wd=1e-5, dataset="UCF101", img_dim=256, 
                   supervised=False, use_scheduler=False, train_what="all", 
                   test=False):
    """
    init_optimizer(model)

    Initializes a torch optimizer tied to the model parameters, and optionally 
    a learning rate scheduler.

    Required args
    -------------
    - model : torch nn.Module or nn.DataParallel
        Model or wrapped model.

    Optional args
    -------------
    - lr : float (1e-3)
        Learning weight to initialize the optimizer with.
    - wd : float (1e-5)
        Weight decay value to initialize the optimizer with.
    - dataset : str (default="UCF101")
        Dataset name.
    - img_dim : int (default=256)
        Image dimensions, when they are input to the model.
    - supervised : bool (default=False)
        Determines how to set parameter gradients.
    - use_scheduler : bool (default=False)
        If True, a learning rate scheduler is returned. 
    - train_what : str (default="all")
        Specifies which parameters to train, and whether to adjust some of 
        their learning rates. Accepted values include 'all', 'ft', and 'last'. 
    - test : bool (default=False)
        If True, None values are returned for the optimizer and scheduler. 

    Returns
    -------
    - optimizer : torch.optim object or None
        Optimizer or None if test is True.
    - scheduler : torch.optim.lr_scheduler object or None
        Torch learning rate scheduler, if use_scheduler, or None if test is 
        True.
    """

    if test:
        return None, None

    # set gradients
    params = set_gradients(model, supervised, train_what=train_what, lr=lr)

    # get optimizer and scheduler
    optimizer = optim.Adam(params, lr=lr, weight_decay=wd)

    scheduler = None
    if use_scheduler:
        lr_lambda = get_lr_lambda(dataset, img_dim=img_dim)    
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    return optimizer, scheduler
    
    
#############################################
def get_num_classes_sup(model):
    """
    get_num_classes_sup(model)

    Returns the number of classes and whether the model is set to supervised 
    mode.

    Required args
    -------------
    - model : torch nn.Module or nn.DataParallel
        Model or wrapped model.

    Returns
    -------
    - num_classes : int or None
        Number of classes, if the model is set to supervised mode, and None 
        otherwise.
    - supervised : bool
        Whether model is set to supervised mode.
    """
    
    model = get_model_only(model)
    num_classes = None
    
    supervised = hasattr(model, "num_classes")
    if supervised:
        num_classes = model.num_classes
    
    return num_classes, supervised


#############################################
def class_weights(dataset="MouseSim", supervised=True):
    """
    class_weights()

    Returns class weights for loss calculation, if training will be done in 
    supervised mode.

    Optional args
    -------------
    - dataset : str (default="MouseSim")
        Dataset name.
    - supervised : bool (default=True)
        Whether the model will be trained in supervised mode.

    Returns
    -------
    - class_weights : torch.Tensor or None
        Weights for each class if applicable, and None otherwise.
    """
    
    class_weights = None
    dataset = misc_utils.normalize_dataset_name(dataset)
    if supervised and dataset == "MouseSim":
        # weights were previously applied, based on training set proportions
        class_weights = None

    if class_weights is not None:
        logger.warning(
            f"Loss will be weighted per class as follows: {class_weights}."
            )

        class_weights = torch.Tensor(class_weights)

    return class_weights


#############################################
def check_end(start_epoch_n=0, num_epochs=50):
    """
    check_end()

    Checks whether training has already ended, based on the starting epoch 
    number, and target total number of epochs.

    Optional args
    -------------
    - start_epoch_n : int (default=0)
        Starting epoch number.
    - num_epochs : int (default=50)
        Target total number of epochs.

    Returns
    -------
    - end : bool
        Whether the training is already complete.
    """
    
    end = False
    if start_epoch_n >= num_epochs + 1:
        logger.info(
            (f"Model already trained to epoch {start_epoch_n} "
            f"(> {num_epochs}).")
            )
        end = True
    
    return end


#############################################
def resize_input_seq(input_seq):
    """
    resize_input_seq(input_seq)

    If the dataset is set in supervised and test mode, each batch item is a 
    sub-batch in which all sets of sequences share a label.

    Required args
    -------------
    - input_seq : 7D Tensor
        Input sequence to reshape, with dims: B, SUB_B, N, C, L, H, W

    Returns
    -------
    - input_seq : 6D Tensor
        Reshaped input sequence, with dims: B * SUB_B, N, C, L, H, W
    - SUB_B : int
        Input sequence sub-batch size, before reshaping.
    """

    if len(input_seq.shape) != 7:
        raise ValueError("'input_seq' should have 7 dimensions.")

    B, SUB_B, N, C, L, H, W = input_seq.shape
    input_seq = input_seq.reshape(B * SUB_B, N, C, L, H, W)

    return input_seq, SUB_B


#############################################
def set_model_train_mode(model, epoch_n=0, train_off=False):
    """
    set_model_train_mode(model)

    Sets the model to train or eval mode.

    Required args
    -------------
    - model : torch nn.Module or nn.DataParallel
        Model or wrapped model.

    Optional args
    -------------
    - epoch_n : int (default=0)
        Epoch number. If it is 0, model is set to eval mode to allow a baseline 
        to be computed.
    - train_off : bool (default=False)
        If True, regardless of the epoch number, model is set to eval mode.

    Returns
    -------
    - train : bool
        Whether model has been set to train mode.
    - spacing : str
        Spacing to use in logging epoch results.
    """

    if epoch_n == 0 or train_off:
        log_str = " (pre-training baseline)" if epoch_n == 0 else "" 
        logger.info(
            f"Running epoch {epoch_n} with no weight updates{log_str}.", 
            extra={"spacing": "\n"}
            )
        spacing = ""
        train = False # estimate untrained performance
        model.eval()
    else:
        spacing = "\n" # for first log of the epoch
        train = True
        model.train()

    return train, spacing


#############################################
def prep_supervised_loss(output, target, shared_pred=False, SUB_B=None):
    """
    prep_supervised_loss(output, target)

    Prepares target and output for supervised loss computing.

    If shared_pred is True, the output is reshaped to group the subsets of 
    batch items with the same prediction, and an average is computed over 
    their softmaxes.

    Required args
    -------------
    - output : 3D Tensor
        Model output, with dims: B or B * SUB_B x N x number of classes.
    - target : 2D Tensor
        Tensor specifying the target class for each sequence, 
        with dims: B x N.

    Optional args
    -------------
    - shared_pred : bool (default=False)
        If True, the target provides a shared prediction for subsets of batch 
        items, and the first dimension is B * SUB instead of B.
    - SUB_B : int (default=None)
        Number of batch items per subset that share a target prediction, 
        if shared_pred is True. 

    Returns
    -------
    - output_flattened : 2D Tensor
        Model output, flattened to group sequences across batches, 
            with dims: B x number of classes if shared_pred, else 
                       B * N x number of classes.
    - target_flattened : 1D Tensor
        Target values, flattened to group sequences across batches,            
            with dims: B if shared_pred, else B * N.
    - loss_reshape : tuple
        Shape to which loss can be reshaped to separate loss values by batch: 
        (B, ) if pred_shared is True, and (B, N) otherwise.
    """
    
    if shared_pred:
        if SUB_B is None:
            raise ValueError("Must pass 'SUB_B' if shared_pred is True.")
        B_comb, N, num_classes = output.size()
        B = B_comb // SUB_B
        USE_N = SUB_B * N

        # group sequences that share a label
        output_flattened = output.reshape(B, USE_N, num_classes)

        # for each batch item, average the softmaxed class 
        # predictions across sequences
        output_flattened = torch.mean(
            torch.nn.functional.softmax(output_flattened, 2),
            1) # B x num_classes
        target_flattened = target.reshape(-1)
        loss_reshape = (B, )

    else:
        # consider all sequences separately, even if they share a label 
        B, N, num_classes = output.size()
        output_flattened = output.reshape(B * N, num_classes)
        target_flattened = target.repeat(1, N).reshape(-1)
        loss_reshape = (B, N)

    return output_flattened, target_flattened, loss_reshape


#############################################
def prep_self_supervised_loss(output, mask, input_seq_shape=None):
    """
    prep_self_supervised_loss(output, mask)

    Prepares target and output for self-supervised loss computing.

    Required args
    -------------
    - output : 6D Tensor
        Model output tensor, with dims: 
            B x P x D_out2 x B x P x D_out2.
    - mask : 6D Tensor
        Mask indicating the nature of each predicted-ground truth pair, 
        where 1s indicate positive pairs, with dims: 
            B x P x D_out2 x B x P x D_out2.

    Optional args
    -------------
    - input_seq_shape : tuple (default=None)
        Input sequence shape (B, N, C, L, H, W), optionally used for logging 
        at the debugging level.

    Returns
    -------
    - output_flattened : 2D Tensor
        Flattened model output tensor, with dims: 
            B * P * D_out2 x B_per (per GPU) * P * D_out2
    - target_flattened : 2D Tensor
        Flattened target mask identifying all positive pairs by 1s, and all 
        other pairs (negative) by 0s, with the same dimensions as 
        output_flattened.
    - loss_reshape : tuple
        Shape to which loss can be reshaped to separate loss values by batch, 
        i.e. (B, P, D_out2)
    - target : 6D Tensor
        Unflattened target mask identifying all positive pairs by 1s, and all 
        other pairs (negative) by 0s, with dims: 
            B x P x D_out2 x B x P x D_out2.
    """
    
    input_seq_str = ""
    if input_seq_shape is not None:
        input_seq_str = (
            f"Input sequence shape: {input_seq_shape} "
            "(expecting [B, N, C, L, H, W]).\n"
        )

    logger.debug(
        f"Model called next.\n{input_seq_str}"
        f"Output shape: {output.shape} "
        "(expecting a 6D tensor: [B, P, D_out2, B_per, P, D_out2]).\n"
        f"Mask shape: {mask.size()}"
    )

    # batch x pred step x dim squared x batch/GPU x pred step x dim squared
    (B, P, D_out2, B_per, _, _) = mask.size()
    flat_dim = B * P * D_out2
    flat_dim_per = B_per * P * D_out2 # B_per: batch size per GPU
    loss_reshape = (B, P, D_out2)

    target = get_target_from_mask(mask)

    # output is a 6d tensor: [B, P, D_out2, B_per, P, D_out2]
    output_flattened = output.reshape(flat_dim, flat_dim_per)
    target_flattened = target.reshape(
        flat_dim, flat_dim_per).argmax(dim=1)

    return output_flattened, target_flattened, loss_reshape, target


#############################################
def prep_loss(output, mask=None, sup_target=None, input_seq_shape=None, 
              supervised=False, shared_pred=False, SUB_B=None, is_gabor=False):
    """
    prep_loss(output, mask)


    Required args
    -------------
    - output : 3 or 6D Tensor
        Model output. For details, see prep_supervised_loss() if 
        supervised is True, and prep_self_supervised_loss() otherwise.

    Optional args
    -------------
    - mask : 6D Tensor
        Mask indicating the nature of each predicted-ground truth pair, 
        where 1s indicate positive pairs. Required if supervised is False. 
            Dimensions: B x P x D_out2 x B x P x D_out2.
    - sup_target : 2 or 3D Tensor
        Tensor specifying the target class for each sequence. Required if 
        supervised is True. 
        Dimensions: B x N (x L x [label, unexp_label]).
    - input_seq_shape : tuple (default=None)
        Input sequence shape (B, N, C, L, H, W), optionally used for logging 
        at the debugging level, if supervised is False.
    - supervised : bool (default=False)
        Whether output and target are for the supervised task.
    - shared_pred : bool (default=False)
        If True, the target provides a shared prediction for subsets of batch 
        items, and the first dimension is B * SUB instead of B. Only applies if 
        supervised is True.
    - SUB_B : int (default=None)
        Number of batch items per subset that share a target prediction, 
        if shared_pred is True. 
    - is_gabor : bool (default=True)
        If True, the output and target are from the Gabors dataset. If so, the 
        specific supervised target to use for each sequence is returned from 
        the full targets provided by the dataset for each sequence item.    

    Returns
    -------
    - output_flattened : 2D Tensor
        Flattened output. For details, see prep_supervised_loss() if 
        supervised is True, and prep_self_supervised_loss() otherwise.
    - target_flattened : 1 or 2D Tensor
        Flattened target. For details, see prep_supervised_loss() if 
        supervised is True, and prep_self_supervised_loss() otherwise.
    - loss_reshape : tuple
        Shape to which loss can be reshaped to separate loss values by batch.
    - target : 2 or 6D Tensor
        Unflattened target. If supervised is True, dims are B x N. 
        See prep_self_supervised_loss() for details otherwise. 
    """

    if supervised:
        if sup_target is None:
            raise ValueError("Must pass 'sup_target' if 'supervised' is True.")
        if is_gabor:
            sup_target = gabor_utils.get_gabor_sup_label(sup_target)
        target = sup_target
        output_flattened, target_flattened, loss_reshape = \
            prep_supervised_loss(
                output, sup_target, shared_pred=shared_pred, SUB_B=SUB_B
                )

    else:
        if mask is None:
            raise ValueError("Must pass 'mask' if 'supervised' is False.")
        output_flattened, target_flattened, loss_reshape, target = \
            prep_self_supervised_loss(
                output, mask, input_seq_shape=input_seq_shape
                )

    return output_flattened, target_flattened, loss_reshape, target



############################################
def get_sup_target_to_store(dataset, sup_target):
    """
    get_sup_target_to_store(dataset, sup_target)

    Gets the supervised target to store based on the dataset type.

    If the dataset is a Gabor dataset, class labels in the input supervised 
    target tensor are converted to class names [image type, mean ori] before 
    the tensor is converted to a nested list. Also, targets are retained for 
    all sequence items.

    Required args
    -------------
    - dataset : torch data.Dataset object
        Torch dataset object, used to identify whether the dataset is a 
        Gabor dataset.
    - sup_target : 2 or 4D Tensor
        Tensor specifying the target class for each sequence. Required if 
        supervised is True. 
        Dimensions: B x N (x L x [label, unexp label] if Gabor dataset).

    Returns
    -------
    - sup_target : list
        Input sup_target converted to a nested list, or modified if Gabor 
        dataset to B x N x L x [image type, mean ori]. 
    """
    
    if isinstance(dataset, str):
        raise ValueError("'dataset' should be a dataset object, not a string.")

    is_gabor = hasattr(dataset, "num_gabors")

    if is_gabor:
        sup_target = gabor_utils.get_gabor_sup_target_to_store(
            dataset, sup_target
            )
    else:
        sup_target = sup_target.tolist()

    return sup_target


############################################
def add_batch_data(data_dict, dataset, batch_loss, batch_loss_by_item, 
                   sup_target=None, output=None, target=None, epoch_n=0):
    """
    add_batch_data(data_dict, dataset, batch_loss, batch_loss_by_item)

    Adds batch loss and accuracy data in place to a data dictionary, and 
    optionally output and target data.

    Required args
    -------------
    - data_dict : dict
        Data dictionary to which to append batch information, with keys:
        Dictionary recording loss and accuracy data, with keys:
        'avg_loss_by_batch' (list): average loss values for each batch
        'batch_epoch_n' (list)    : epoch number for each batch
        'loss_by_item' (3D list)  : loss values with dims batches x B x N
        
        and optionally:
        'output_by_batch' (4 or 7D array-like)    : output values for each 
            batch, with dims batch x B x N x number of classes if the task is 
            supervised or batch x B x P x D_out2 x B_per x P x D_out2, 
            otherwise.
        'sup_target_by_batch' (4 or 6D Tensor): supervised targets for each 
            batch, each with dims: batch x B x N (x L x [image type, mean ori] 
            if Gabor dataset).
        'target_by_batch' (3 or 7D array-like)    : target values for each 
            batch, each with dims batch x B x N if the task is supervised or 
            batch x B x P x D_out2 x B_per x P x D_out2, otherwise.

    - dataset : torch data.Dataset object
        Torch dataset object, used to identify whether the dataset is a 
        Gabor dataset.
    - batch_loss : float
        Loss for the batch for which data is being added.
    - batch_loss_by_item : list
        Batch losses separated by item.

    Optional args
    -------------
    - sup_target : 2 or 4D nested array-like
        List specifying the target class for each sequence. Required if 
        supervised is True. 
        Dimensions: B x N (x L x [label, unexp_label] if Gabor dataset).
    - output : 3 or 6D nested array-like
        Model output for the batch, with dims:
            B x N x number of classes if supervised, and 
            B x P x D_out2 x B x P x D_out2 otherwise.
    - target : 2, 4 or 6D nested array-like
        Target for the batch. See sup_target if supervised. 
        Otherwise, dimensions are B x P x D_out2 x B x P x D_out2. 
    - epoch_n : int (default=0)
        Epoch number associated with the batch for which data is being added.
    """
    
    n_items = np.asarray(batch_loss_by_item).size
    
    data_dict["batch_epoch_n"].append(epoch_n)
    data_dict["avg_loss_by_batch"].append(batch_loss / n_items)
    data_dict["loss_by_item"].append(batch_loss_by_item)

    if output is not None and "output_by_batch" in data_dict.keys():
        data_dict["output_by_batch"].append(output)

    if target is not None and "target_by_batch" in data_dict.keys():
        data_dict["target_by_batch"].append(target)

    if sup_target is not None:
        data_dict["sup_target_by_batch"].append(
            get_sup_target_to_store(dataset, sup_target)
            )


#############################################
def log_epoch(stats_str, duration=None, epoch_n=0, num_epochs=50, val=False, 
              test=False, batch_idx=0, n_batches=None, spacing="\n", tail=""):
    """
    log_epoch(stats_str)

    Logs epoch statistics at the INFO level.

    Required args
    -------------
    - stats_str : str
        String containing statistics for the epoch.

    Optional args
    -------------
    - duration : float (default=None)
        Duration of the epoch.
    - epoch_n : int (default=0)
        Epoch number.
    - num_epochs : int (default=50)
        Total number of training epochs.
    - val : bool (default=False)
        If True, a validation epoch is being logged.
    - test : bool (default=False)
        If True, model testing is being logged.
    - batch_idx : int (default=0)
        Batch index, if applicable.
    - n_batches : int (default=None)
        Total number of batches. If None, batch index is not logged.
    - spacing : str (default="\n")
        Spacing to add before the logged string.
    - tail : str (default="")
        Spacing to add after the logged string. 

    Returns
    -------
    - log_str : str
        Copy of the string that was logged.
    """

    if test:
        epoch_str = f"Epoch: [{epoch_n}] [test]"
        space_batch = " "
    elif val:
        epoch_str = f"Epoch: [{epoch_n}/{num_epochs}] [val]"
        space_batch = " "
    else:
        epoch_str = f"Epoch: [{epoch_n}/{num_epochs}]"
        space_batch = ""
        

    time_str = ""
    if n_batches is None and duration is not None:
        time_str = misc_utils.format_time(duration, sep_min=True, lead=True)
    else:
        epoch_str = f"{epoch_str}{space_batch}[{batch_idx}/{n_batches - 1}]"
        if duration is not None:
            time_str = misc_utils.format_time(
                duration, sep_min=False, lead=True
                )
            time_str = f"{time_str}/batch"

    log_str = f"{epoch_str}{TAB}{stats_str}{TAB}{time_str}{tail}"

    logger.info(log_str, extra={"spacing": spacing})

    return log_str


#############################################
def write_log(stats_str, epoch_n, filename, output_dir=".", overwrite=False):
    """
    write_log(stats_str, epoch_n, filename)

    Writes epoch statistics to a log file. If the file does not exist, it is 
    created. Otherwise, it is appended to, or optionally overwritten.

    Required args
    -------------
    - stats_str : str
        String containing statistics for the epoch.
    - epoch_n : int
        Epoch number.
    - filename : str or Path
        Path of log file in which to log the epoch statistics.
    
    Optional args
    -------------
    - output_dir : str or path (default=".")
        Directory in which to save the log file.
    - overwrite : bool (default=False)
        If True, and filename exists, the file is overwritten. Otherwise, it is 
        appended to.
    """
    
    filename = Path(output_dir, filename)
    filename.parent.mkdir(exist_ok=True, parents=True)

    open_mode = "w"
    if filename.is_file():
        if overwrite:
            logger.warning(
                (f"{filename} already exists. Removing, as it will be "
                "overwritten."), 
                extra={"spacing": "\n"}
                )            
            time.sleep(5) # to allow for skipping file removal.
            filename.unlink()
        else:
            open_mode = "a"

    with open(filename, open_mode) as f:
        f.write(f"## Epoch {epoch_n}:\n")
        f.write(f"Time: {datetime.now()}\n")
        f.write(f"{stats_str}\n\n")

