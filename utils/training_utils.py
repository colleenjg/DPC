from datetime import datetime
import logging
import multiprocessing

from pathlib import Path
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



    Optional args
    -------------
    - num_workers
    - cpu_only

    Returns
    -------
    - device
    - num_workers
    """

    device_name = "cuda" if torch.cuda.is_available() and not cpu_only else "cpu"
    device = torch.device(device_name)

    if num_workers is None:
        num_cores = multiprocessing.cpu_count()
        num_devices = torch.cuda.device_count()
        num_workers = min(num_devices * 2, num_cores) # approx rule of thumb
    
    return device, num_workers


#############################################
def allow_data_parallel(dataloader, device, supervised=False):
    """
    allow_data_parallel(dataloader, device)



    Required args
    -------------
    - dataloader : 
    - device : 

    Optional args
    -------------
    - supervised : bool (default=False)

    Returns
    -------
    - allow_parallel : bool
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
def get_model(dataset="UCF101", supervised=True, img_dim=224, num_seq=8, 
              seq_len=5, network="resnet18", pred_step=3, dropout=0.1):
    """
    get_model()



    Required args
    -------------
    - dataset : torch data.Dataset or str (default="UCF101")
        Torch dataset for which to retrieve the number of classes or name of 
        the dataset, if supervised is True.

    Optional args
    -------------
    - supervised : bool (default=True)
    - img_dim : int (default=224)
    - num_seq : int (default=8)
    - seq_len : int (default=5)
    - network : str (default="resnet18")
    - pred_step : int (default=3)
    - dropout : float (default=0.1)

    Returns
    -------
    - model
    """

    model_kwargs = {
        "sample_size": img_dim,
        "num_seq": num_seq,
        "seq_len": seq_len,
        "network": network,
    }

    if supervised:
        num_classes = misc_utils.get_num_classes(dataset)

        model = model_3d.LC_RNN(
            num_classes=num_classes,
            dropout=dropout,
            **model_kwargs
            )
    
    else:
        model = model_3d.DPC_RNN(pred_step=pred_step, **model_kwargs)

    return model


#############################################
def get_model_only(model):
    """
    get_model_only(model)



    Required args
    -------------
    - model : 

    Returns
    -------
    - model : 
    """
    
    # in case the model is wrapped with DataParallel()
    if hasattr(model, "device_ids"):
        model = model.module

    return model


#############################################
def log_weight_decay_prop(model):
    """
    log_weight_decay_prop(model)



    Required args
    -------------
    - model : 
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
def get_target_from_mask(mask):
    """
    get_target_from_mask(mask)
    
    Task mask as input, compute the target for contrastive loss

    Required args
    -------------
    - mask : 

    Returns
    -------
    - target : 
    """
    
    # mask meaning: 1: pos
    target = (mask == 1).to(int)
    target.requires_grad = False

    return target


#############################################
def check_grad(model):
    """
    check_grad(model)



    Required args
    -------------
    - model : 
    """

    model = get_model_only(model)
    
    logger.debug("\n===========Check Grad============")
    for name, param in model.named_parameters():
        logger.debug(name, param.requires_grad)
    logger.debug("=================================\n")


#############################################
def set_gradients(model, supervised=False, train_what="all", lr=1e-3):
    """
    set_gradients(model)



    Required args
    -------------
    - model : 

    Optional args
    -------------
    - supervised : bool (default=False)
    - train_what : str (default="all")
    - lr : float (default=1e-3)

    Returns
    -------
    - params
    """

    model = get_model_only(model)

    params = model.parameters()
    if train_what == "last":
        logger.info("=> Training only the last layer.")
        if supervised:
            for name, param in model.named_parameters():
                if ("resnet" in name.lower()) or ("rnn" in name.lower()):
                    param.requires_grad = False
        else:
            for name, param in model.resnet.named_parameters():
                param.requires_grad = False

    elif supervised and train_what == "ft":
        logger.info("=> Finetuning backbone with a smaller learning rate.")
        params = []
        for name, param in model.named_parameters():
            if ("resnet" in name.lower()) or ("rnn" in name.lower()):
                params.append({"params": param, "lr": lr / 10})
            else:
                params.append({"params": param})
    
    elif train_what != "all":
        raise ValueError(
            f"{train_what} value for train_what is not recognized."
            )
    else: 
        pass # train all layers

    check_grad(model)

    return params
    
    
#############################################
def get_multistepLR_restart_multiplier(epoch_n, gamma=0.1, step=[10, 15, 20], 
                                       repeat=3):
    """
    get_multistepLR_restart_multiplier(epoch)

    Returns the multipier for LambdaLR, 
    0  <= ep < 10: gamma^0
    10 <= ep < 15: gamma^1 
    15 <= ep < 20: gamma^2
    20 <= ep < 30: gamma^0 ... repeat 3 cycles and then keep gamma^2

    Required args
    -------------
    - epoch_n : 

    Optional args
    -------------
    - gamma : float (default=0.1)
    - step : list (default[10, 15, 20])
    - repeat : int (default=3)

    Returns
    -------
    - float
    """

    max_step = max(step)
    effective_epoch = epoch_n % max_step
    if epoch_n // max_step >= repeat:
        exp = len(step) - 1
    else:
        exp = len([i for i in step if effective_epoch >= i])

    return gamma ** exp


#############################################
def get_lr_lambda(dataset="UCF101", img_dim=224): 
    """
    get_lr_lambda()



    Optional args
    -------------
    - dataset : str (default="UCF101")
    - img_dim : int (default=224)

    Returns
    -------
    - lr_lambda : lambda function
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
        ep_n, gamma=0.1, step=steps, repeat=1
        )

    return lr_lambda


#############################################
def init_optimizer(model, lr=1e-3, wd=1e-5, dataset="UCF101", img_dim=128, 
                   supervised=False, train_what="all", test=False):
    """
    init_optimizer(model)
    """

    if test:
        return None, None

    # set gradients
    params = set_gradients(model, supervised, train_what=train_what, lr=lr)

    # get optimizer and scheduler
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=wd)

    scheduler = None
    if supervised:
        lr_lambda = get_lr_lambda(dataset, img_dim=img_dim)    
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    return optimizer, scheduler
    
    
#############################################
def get_num_classes_sup(model):
    """
    get_num_classes_sup(model)



    Required args
    -------------
    - model : 

    Returns
    -------
    - num_classes : int or None
    - supervised : bool
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



    Optional args
    -------------
    - dataset : str (default="MouseSim")
    - supervised : bool (default=True)

    Returns
    -------
    - class_weights : torch.Tensor or None
    """
    
    weights = None
    dataset = misc_utils.normalize_dataset_name(dataset)
    if supervised and dataset == "MouseSim":
        weights = [6, 1]
        logger.warning(
            f"Loss will be weighted per class as follows: {weights}."
            )

        weights = torch.Tensor(weights) # based on training set

    return weights


#############################################
def check_end(start_epoch_n=0, num_epochs=50):
    """
    check_end()



    Optional args
    -------------
    - start_epoch_n : int (default=0)
    - num_epochs : int (default=50)

    Returns
    -------
    - end
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

    if the dataset is set to 'supervised' and 'test mode', 
    each batch item is a sub-batch in which all sets of sequences 
    share a label

    Required args
    -------------
    - input_seq : 

    Returns
    -------
    - input_seq : 
    - SUB_B : 
    """

    B, SUB_B, N, C, SL, H, W = input_seq.size()
    input_seq = input_seq.reshape(B * SUB_B, N, C, SL, H, W)

    return input_seq, SUB_B


#############################################
def set_model_train_mode(model, epoch_n=0, train_off=False):
    """
    set_model_train_mode(model)


    Required args
    -------------
    - model :

    Optional args
    -------------
    - epoch_n : int (default=0)
    - train_off : bool (default=False)

    Returns
    -------
    - train : 
    - spacing : 
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



    Required args
    -------------
    - output : 
    - target : 

    Optional args
    -------------
    - shared_pred : bool (default=False)
    - SUB_B : int (default=None)

    Returns
    -------
    - output_flattened : 
    - target_flattened : 
    - loss_reshape : 
    - target : 
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

    return output_flattened, target_flattened, loss_reshape, target


#############################################
def prep_self_supervised_loss(output, mask, input_seq_shape=None):
    """
    prep_self_supervised_loss(output, mask)



    Required args
    -------------
    - output
    - mask

    Optional args
    -------------
    - input_seq_shape : tuple (default=None)

    Returns
    -------
    - output_flattened : 
    - target_flattened : 
    - loss_reshape : 
    - target : 
    """
    
    input_seq_str = ""
    if input_seq_shape is not None:
        input_seq_str = (
            f"Input sequence shape: {input_seq_shape} "
            "(expecting [B, N, C, SL, H, W]).\n"
        )

    logger.debug(
        f"Model called next.\n{input_seq_str}"
        f"Output shape: {output.shape} "
        "(expecting a 6D tensor: [B, PS, HW, B_per, PS, HW]).\n"
        f"Mask shape: {mask.size()}"
    )

    # batch x pred step x dim squared x batch/GPU x pred step x dim squared
    (B, PS, HW, B_per, _, _) = mask.size()
    flat_dim = B * PS * HW
    flat_dim_per = B_per * PS * HW # B_per: batch size per GPU
    loss_reshape = (B, PS, HW)

    target = get_target_from_mask(mask)

    # output is a 6d tensor: [B, PS, HW, B_per, PS, HW]
    output_flattened = output.reshape(flat_dim, flat_dim_per)
    target_flattened = target.reshape(
        flat_dim, flat_dim_per).argmax(dim=1)

    return output_flattened, target_flattened, loss_reshape, target


#############################################
def prep_loss(output, mask, sup_target=None, input_seq_shape=None, 
              supervised=False, shared_pred=False, SUB_B=None, is_gabors=False):
    """
    prep_loss(output, mask)


    Required args
    -------------
    - output
    - mask

    Optional args
    -------------
    - sup_target : 
    - input_seq_shape : tuple (default=None)
    - supervised : bool (default=False)
    - shared_pred : bool (default=False)
    - SUB_B : int (default=None)
    - is_gabors : bool (default=True)    

    Returns
    -------
    - output_flattened : 
    - target_flattened : 
    - loss_reshape : 
    - target : 
    """

    if supervised:
        if sup_target is None:
            raise ValueError("Must pass 'sup_target' if 'supervised' is True.")
        if is_gabors:
            sup_target = gabor_utils.get_gabor_sup_label(sup_target)
        output_flattened, target_flattened, loss_reshape, target = \
            prep_supervised_loss(
                output, sup_target, shared_pred=shared_pred, SUB_B=SUB_B
                )

    else:
        output_flattened, target_flattened, loss_reshape, target = \
            prep_self_supervised_loss(
                output, mask, input_seq_shape=input_seq_shape
                )

    return output_flattened, target_flattened, loss_reshape, target



############################################
def get_sup_target_to_store(dataset, sup_target):
    """
    get_sup_target_to_store(dataset, sup_target)

    Required args
    -------------
    - dataset : 
    - sup_target : 

    Returns
    -------
    - sup_target : 
    """
    
    if isinstance(dataset, str):
        raise ValueError("'dataset' should be a dataset object, not a string.")

    is_gabors = hasattr(dataset, "num_gabors")

    if is_gabors:
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



    Required args
    -------------
    - data_dict : 
    - dataset : 
    - batch_loss : 
    - batch_loss_by_item : 

    Optional args
    -------------
    - sup_target : 
    - output : 
    - target : 
    - epoch_n : int (default=0)
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
              test=False, batch_idx=0, n_batches=None, spacing="\n"):
    """
    log_epoch(stats_str)



    Required args
    -------------
    - stats_str : str

    Optional args
    -------------
    - duration : float (default=None)
    - epoch_n : int (default=0)
    - num_epochs : int (default=50)
    - val : bool (default=False)
    - test : bool (default=False)
    - batch_idx : int (default=0)
    - n_batches : int (default=None)
    - spacing : str (default="\n")

    Returns
    -------
    - log_str : str
    """

    if test:
        epoch_str = f"Epoch [{epoch_n}] [test]"
        space_batch = " "
    elif val:
        epoch_str = f"Epoch [{epoch_n}/{num_epochs}] [val]"
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

    log_str = f"{epoch_str}{TAB}{stats_str}{TAB}{time_str}"

    logger.info(log_str, extra={"spacing": spacing})

    return log_str


#############################################
def write_log(content, epoch_n, filename):
    """
    write_log(content, epoch_n, filename)

    Required args
    -------------
    - content
    - epoch_n
    - filename
    """
    
    filename = Path(filename)
    filename.parent.mkdir(exist_ok=True, parents=True)

    open_mode = "a" if filename.is_file() else "w"

    with open(filename, open_mode) as f:
        f.write(f"## Epoch {epoch_n}:\n")
        f.write(f"Time: {datetime.now()}\n")
        f.write(f"{content}\n\n")

