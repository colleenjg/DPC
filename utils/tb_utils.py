import logging
from pathlib import Path

from torchvision import transforms
from torchvision import utils as vutils

logger = logging.getLogger(__name__)


#############################################
def init_tb_writers(direc=".", val=False):
    from tensorboardX import SummaryWriter

    if direc is None:
        raise ValueError("direc cannot be None.")

    Path(direc).mkdir(exist_ok=True, parents=True)
    writer_train = SummaryWriter(logdir=str(Path(direc, "train")))
    writer_val = None
    if val:
        writer_val = SummaryWriter(logdir=str(Path(direc, "val")))
    return writer_train, writer_val


#############################################
def denorm(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    if len(mean) != 3:
        raise ValueError("'mean' must comprise 3 values.")
    if len(std) != 3:
        raise ValueError("'std' must comprise 3 values.")
    inv_mean = [-mean[i] / std[i] for i in range(3)]
    inv_std = [1 / i for i in std]
    return transforms.Normalize(mean=inv_mean, std=inv_std)


#############################################
def write_input_seq_tb(writer, input_seq, n=2, i=0):
    """
    Write input sequences to a tensorboard writer.
    """

    _, N, C, SL, H, W = input_seq.shape
    writer.add_image(
        "input_seq", 
        denorm(vutils.make_grid(
            input_seq[:n].transpose(2, 3).contiguous().view(-1, C, H, W), 
            nrow=N * SL)
        ), i
    )


#############################################
def update_tb(writer_train, train_dict, epoch_n=0, writer_val=None, 
                  valid_dict=None):

    datatypes = [
        "global/loss", 
        "global/accuracy", 
        "accuracy/top1", 
        "accuracy/top3", 
        "accuracy/top5"
        ]

    for mode in ["train", "val"]:
        if mode == "train":
            writer = writer_train                    
            data_dict = train_dict
        elif valid_dict is not None:
            if writer_val is None:
                raise ValueError(
                    "Must provide writer_val if valid_dict is provided."
                    )
            writer = writer_val
            data_dict = valid_dict

        all_data = [
            data_dict["loss"], 
            data_dict["acc"], 
            *data_dict["topk_meters"]
            ]

        for datatype, data in zip(datatypes, all_data):
            writer.add_scalar(datatype, data, epoch_n)

