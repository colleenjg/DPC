import logging
import math

import numpy as np
import torch
import torch.nn.functional as F

from model import convrnn, resnet_2d3d

logger = logging.getLogger(__name__)

TAB = "    "

#############################################
class DPC_RNN(torch.nn.Module):
    """DPC with RNN"""
    def __init__(self, sample_size, num_seq=8, seq_len=5, pred_step=3, 
                 network="resnet50"):
        
        super(DPC_RNN, self).__init__()
        
        logger.info("Loading DPC-RNN model")
        self.sample_size = sample_size
        self.num_seq = num_seq
        self.seq_len = seq_len
        self.pred_step = pred_step

        if self.num_seq <= self.pred_step:
            raise ValueError(
                f"'num_seq' ({num_seq}) must be strictly greater than "
                f"'pred_step' ({pred_step})."
                )

        self.last_duration = int(math.ceil(seq_len / 4))
        self.last_size = int(math.ceil(sample_size / 32))
        logger.info(
            f"Final feature map size: {self.last_size} x {self.last_size}", 
            extra={"spacing": TAB}
            )

        self.backbone, self.param = resnet_2d3d.select_resnet(
            network, track_running_stats=False
            )
        self.param["num_layers"] = 1 # param for GRU
        self.param["hidden_size"] = self.param["feature_size"] # param for GRU

        self.agg = convrnn.ConvGRU(
            input_size=self.param["feature_size"],
            hidden_size=self.param["hidden_size"],
            kernel_size=1,
            num_layers=self.param["num_layers"]
            )
        self.network_pred = torch.nn.Sequential(
            torch.nn.Conv2d(
                self.param["feature_size"], 
                self.param["feature_size"], 
                kernel_size=1, 
                padding=0
                ),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(
                self.param["feature_size"], 
                self.param["feature_size"], 
                kernel_size=1, 
                padding=0
                )
            )
        self.mask = None
        self.relu = torch.nn.ReLU(inplace=False)
        self._initialize_weights(self.agg)
        self._initialize_weights(self.network_pred)


    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if "bias" in name:
                torch.nn.init.constant_(param, 0.0)
            elif "weight" in name:
                torch.nn.init.orthogonal_(param, 1)
        # other resnet weights have been initialized in resnet itself


    def pred_steps(self, hidden):
        ### predict next steps from hidden layer ###
        # returns: [B, pred_step, D, last_size, last_size] #
        pred = []
        for _ in range(self.pred_step):
            # sequentially predict future
            p_tmp = self.network_pred(hidden)
            pred.append(p_tmp)
            _, hidden = self.agg(
                self.relu(p_tmp).unsqueeze(1), 
                hidden.unsqueeze(0)
                )
            hidden = hidden[:, -1, :]

        pred = torch.stack(pred, 1) # B, pred_step, xxx

        return pred


    def get_similarity_score(self, predicted, ground_truth):
        ### Get similarity score ###
        # predicted (pred): [B, pred_step, feature_size, last_size, last_size]
        # ground-truth (GT): [B, pred_step, feature_size, last_size, last_size]

        # Dot product for pred-GT pair taken along the feature dimension

        # Returns a 6D tensor, where:
        # first 3 dims are from pred (B, PS, D), and the
        # last 3 dims are from GT (B, PS, D). 

        if predicted.size() != ground_truth.size():
            raise ValueError(
                "predicted and ground_truth must have the same shape."
                )

        B, PS, FS, last_size, _ = predicted.size()
        HW = last_size ** 2

        predicted = torch.movedim(predicted, 2, -1).reshape(
            B * PS * HW, FS
            )
        ground_truth = torch.movedim(ground_truth, 2, -1).reshape(
            B * PS * HW, FS
            )

        score = torch.matmul(predicted, ground_truth.T).reshape(
            B, PS, HW, B, PS, HW
            )
        
        return score


    def reset_mask(self):
        self.mask = None


    def compute_mask(self, B, device="cpu"):
        # mask meaning: 
        # -3: spatial neg
        # (-2: omit)
        # -1: temporal neg (hard)
        # 0: easy neg
        # 1: pos

        # only recompute if needed
        if self.mask is not None and B == len(self.mask):            
            return

        PS = self.pred_step
        HW = self.last_size ** 2

        # creating mask with numpy to avoid a determinism indexing bug
        # default 0 (easy neg)
        mask = np.zeros((B, PS, HW, B, PS, HW), dtype=np.int8)

        # identify -3 (spatial neg)
        mask[np.arange(B), :, :, np.arange(B), :, :] = -3 # spatial neg
        
        # identify -1 (temporal neg (hard))
        for k in range(B):
            mask[
                k, :, np.arange(HW), 
                k, :, np.arange(HW)
                ] = -1
        
        # identify 1 (pos)
        mask = np.transpose(mask, (0, 2, 1, 3, 5, 4)).reshape(
            B * HW, PS, B * HW, PS
            )
        for j in range(B * HW):
            mask[
                j, np.arange(PS), 
                j, np.arange(PS)
                ] = 1

        mask = torch.tensor(
            mask, dtype=torch.int8, requires_grad=False
            ).to(device)

        # mask: B x PS x HW x B x PS x HW
        self.mask = mask.reshape(B, HW, PS, B, HW, PS).contiguous().permute(
            0, 2, 1, 3, 5, 4
        )

        return 


    def forward(self, batch):
        # batch: [B, N, C, SL, H, W]
        ### extract feature ###
        (B, N, C, SL, H, W) = batch.shape
        batch = batch.reshape(B * N, C, SL, H, W)
        feature = self.backbone(batch)
        del batch
        feature = F.avg_pool3d(
            feature, 
            (self.last_duration, 1, 1), 
            stride=(1, 1, 1)
            )

        feature = feature.reshape(
            B, N, self.param["feature_size"], self.last_size, self.last_size
            ) # [-inf, +inf)

        ground_truth = feature[:, N - self.pred_step :].contiguous() # GT

        feature = self.relu(feature) # [0, +inf)

        ### aggregate, predict future ###
        _, hidden = self.agg(feature[:, : N - self.pred_step].contiguous())
        # after tanh (-1, 1) get the hidden state of last layer/time step
        hidden = hidden[:, -1, :] 
        
        pred = self.pred_steps(hidden) # predict next steps
        del hidden

        ### Get similarity score ###
        score = self.get_similarity_score(pred, ground_truth)
        del ground_truth, pred

        self.compute_mask(B, device=feature.device)

        return [score, self.mask]


#############################################
class LC_RNN(torch.nn.Module):
    def __init__(self, sample_size, num_seq, seq_len, network="resnet18", 
                 dropout=0.5, num_classes=101):

        super(LC_RNN, self).__init__()
        self.sample_size = sample_size
        self.num_seq = num_seq
        self.seq_len = seq_len
        self.num_classes = num_classes

        self.last_duration = int(math.ceil(seq_len / 4))
        self.last_size = int(math.ceil(sample_size / 32))
        track_running_stats = True 

        self.backbone, self.param = resnet_2d3d.select_resnet(
            network, 
            track_running_stats=track_running_stats
            )
        self.param["num_layers"] = 1
        self.param["hidden_size"] = self.param["feature_size"]
        self.param["kernel_size"] = 1

        logger.info("=> Loading RNN + FC model (ConvRNN, kernel size: 1)")
        self.agg = convrnn.ConvGRU(
            input_size=self.param["feature_size"],
            hidden_size=self.param["hidden_size"],
            kernel_size=self.param["kernel_size"],
            num_layers=self.param["num_layers"]
            )
        self._initialize_weights(self.agg)

        self.final_bn = torch.nn.BatchNorm1d(self.param["feature_size"])
        self.final_bn.weight.data.fill_(1)
        self.final_bn.bias.data.zero_()

        self.final_fc = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(
                self.param["feature_size"], self.num_classes
                )
            )
        self._initialize_weights(self.final_fc)


    def forward(self, batch):
        # seq1: [B, N, C, SL, H, W]
        (B, N, C, SL, H, W) = batch.shape
        batch = batch.reshape(B * N, C, SL, H, W)
        feature = self.backbone(batch)
        del batch 
        feature = F.relu(feature)
        
        feature = F.avg_pool3d(feature, (self.last_duration, 1, 1), stride=1)
        feature = feature.reshape(
            B, N, self.param["feature_size"], self.last_size, self.last_size
            ) 
        context, _ = self.agg(feature)
        context = context[:, -1, :].unsqueeze(1)
        context = F.avg_pool3d(
            context, (1, self.last_size, self.last_size), stride=1
            ).squeeze(-1).squeeze(-1)
        del feature

        # [B, N, C] -> [B, C, N] for BN() (operates on dim 1), then [B, N, C] 
        context = self.final_bn(context.transpose(-1, -2)).transpose(-1, -2) 
        output = self.final_fc(context).reshape(B, -1, self.num_classes)

        return output, context


    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if "bias" in name:
                torch.nn.init.constant_(param, 0.0)
            elif "weight" in name:
                torch.nn.init.orthogonal_(param, 1)
                 
        # other resnet weights have been initialized in resnet_3d.py

