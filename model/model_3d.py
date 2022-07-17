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
    """
    Dense CPC RNN network.
    
    Attributes
    ----------
    - agg : convrnn.ConvGRU
        Convolution GRU neural network module.
    - backbone : ResNet2d3d_full
        Backbone ResNet network.
    - final_bn : BatchNorm1d
        Batch normalization module for the final layer.
    - final_fc : Sequential
        Final layer, composed of a dropout layer and a full connected layer.
    - last_duration: int
        Final temporal dimension of each feature (approx 1/4 of seq_len).
    - last_size : int
        Final spatial dimension of each feature (last_size x last_size) 
        (approx 1/32 of sample_size).
    - mask : 6D Tensor
        Mask indicating the nature of each predicted-ground truth pair, 
        with dims: B x PS x HW x B x PS x HW 
        (see self.compute_mask() for details).
    - network_pred : nn.Module
        Prediction network.
    - num_seq : int
        Number of sequences in a input batch sample (with shared labels).
    - param : dict
        Dictionary recording network parameters, including 
        "num_layers", "hidden_size", "feature_size", and "kernel_size".
    - pred_step : int
        Number of steps ahead to predict.
    - relu : nn.ReLU
        ReLU layer.
    - sample_size : int
        Height or width of input images (expected to be square).
    - seq_len : int
        Length of each sequence.

    Methods
    -------
    - self.compute_mask(B):
        Sets mask indicating the nature of each predicted-ground truth pair 
        (i.e., positive, negative spatial, easy/hard negative temporal, etc.).
    - self.forward(batch):
        Passes input through the network about produces output class 
        predictions and the final context.
    - self.get_similarity_score(predicted, ground_truth):
        Gets a similarity score for all possible pairs of predicted and ground 
        truth features.
    - self.pred_steps(hidden):
        Predicts features for subsequent timesteps from the hidden layer.
    - self.reset_mask()
        Resets the mask attribute.
    """

    def __init__(self, sample_size, num_seq=8, seq_len=5, pred_step=3, 
                 network="ResNet50"):
        """
        DPC_RNN(sample_size)

        Required args
        -------------
        - sample_size : int
            Height or width of input images (expected to be square).
        - seq_len : int (default=5)
            Length of each sequence.
        - pred_step : int (default=3)
            Number of steps ahead to predict.
        - network : str (default="ResNet50")
            Backbone network on which to build LC_RNN.
            
        Optional args
        -------------
        - num_seq : int (default=8)
            Number of sequences in a input batch sample (with shared labels).        
        """
        
        super(DPC_RNN, self).__init__()
        
        logger.info("Loading DPC-RNN model.")
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
            f"Final feature map size: {self.last_size} x {self.last_size}.", 
            extra={"spacing": TAB}
            )

        self.backbone, self.param = resnet_2d3d.select_ResNet(
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

        # Note that ResNet weights are initialized independently.
        self._initialize_weights(self.agg)
        self._initialize_weights(self.network_pred)


    def _initialize_weights(self, module):
        """
        self._initialize_weights(module)

        Initializes the weights and biases for the input module.

        Required args
        ------------
        - module (nn.Module):
            Module for which to initialize weights and biases.
        """
        for name, param in module.named_parameters():
            if "bias" in name:
                torch.nn.init.constant_(param, 0.0)
            elif "weight" in name:
                torch.nn.init.orthogonal_(param, 1)


    def pred_steps(self, hidden):
        """
        self.pred_steps(hidden)

        Predicts features for subsequent timesteps from the hidden layer.
        
        Returns
        -------
        - pred : 5D Tensor
            Predicted features, with dims: 
                B x PS x D (feature size) x last size x last size
        """

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

        pred = torch.stack(pred, 1)

        return pred


    def get_similarity_score(self, predicted, ground_truth):
        """
        self.get_similarity_score(predicted, ground_truth)

        Gets a similarity score for all possible pairs of predicted and ground 
        truth features.

        Specifically, calculates the dot product along the feature dimension.

        Required args
        -------------
        - predicted : 5D Tensor
            Predicted features, with dims: 
                B_pred x PS x D (feature size) x last size x last size
        - ground_truth : 5D Tensor
            Ground truth features, with dims: 
                B_gt x PS x D (feature size) x last size x last size


        Returns
        -------
        - score : 6D Tensor
            Dot product for each predicted-ground truth feature pair, 
            calculated along the feature dimension (D).
            Dimensions: B_pred x PS x HW x B_gt x PS x HW, 
            where HW = last_size ** 2
        """

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
        """
        self.reset_mask()

        Resets the mask attribute.
        """
        
        self.mask = None


    def compute_mask(self, B, device="cpu"):
        """
        self.compute_mask(B)

        Computes mask indicating the nature of each predicted-ground truth pair 
        (i.e., positive, negative spatial, easy/hard negative temporal, etc.).

        Only recomputes if a new value for B is provided.

        Mask dimensions: B x PS x HW x B x PS x HW

        Mask values: 
            -3 : spatial neg
           (-2 : omit)
            -1 : temporal neg (hard)
             0 : easy neg
             1 : pos

        Required args
        -------------
        - B : int
            Batch size (per GPU, if batches are split across parallel models).

        Optional args
        -------------
        - device : str or torch.device
            Device on which to place mask, when first computed.

        """

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

        self.mask = mask.reshape(B, HW, PS, B, HW, PS).contiguous().permute(
            0, 2, 1, 3, 5, 4
        )

        return 


    def forward(self, batch):
        """
        self.forward(batch)

        Passes batch through the network, and returns the score for the input, 
        as well as the mask for the batch size.

        Required args
        -------------
        - batch : 6D Tensor
            Input tensor with dims: B x N x C x SL x H x W.
            NOTE: B is the size of the batch sent one GPU, if the model is 
            split across GPUs. 

        Returns
        -------
        - score : 6D Tensor
            Similarity score for each predicted-ground truth pair, 
                with dims: B x PS x HW x B x PS x HW
        - self.mask : 6D Tensor      
            Mask indicating the nature of each predicted-ground truth pair, 
            with dims: B x PS x HW x B x PS x HW
            (see self.compute_mask() for details).
        """

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
    """
    Linear classification RNN.
    
    Attributes
    ----------
    - agg : convrnn.ConvGRU
        Convolution GRU neural network module.
    - backbone : ResNet2d3d_full
        Backbone ResNet network.
    - final_bn : BatchNorm1d
        Batch normalization module for the final layer.
    - final_fc : Sequential
        Final layer, composed of a dropout layer and a full connected layer.
    - last_duration: int
        Final temporal dimension of each feature (approx 1/4 of seq_len).
    - last_size : int
        Final spatial dimension of each feature (last_size x last_size) 
        (approx 1/32 of sample_size).
    - num_classes : int (default=101)
        Number of classes to the final prediction layer.
    - num_seq : int
        Number of sequences in a input batch sample (with shared labels).
    - param : dict
        Dictionary recording network parameters, including 
        "num_layers", "hidden_size", "feature_size", and "kernel_size".
    - sample_size : int
        Height or width of input images (expected to be square).
    - seq_len : int
        Length of each sequence.

    Methods
    -------
    - self.forward(batch):
        Passes input through the network about produces output class 
        predictions and the final context.
    """

    def __init__(self, sample_size, num_seq, seq_len, network="ResNet18", 
                 dropout=0.5, num_classes=101):
        """
        LC_RNN(sample_size, num_seq, seq_len)

        Constructs a LC_RNN object.

        Required args
        -------------
        - sample_size : int
            Height or width of input images (expected to be square).
        - num_seq : int
            Number of sequences in a input batch sample (with shared labels).
        - seq_len : int
            Length of each sequence.
        
        Optional args
        -------------
        - network : str (default="ResNet18")
            Backbone network on which to build LC_RNN.
        - dropout : float (default=0.5)
            Dropout proportion for the dropout layer in the final fully 
            connected layer.
        - num_classes : int (default=101)
            Number of classes to the final prediction layer.
        """

        super(LC_RNN, self).__init__()
        self.sample_size = sample_size
        self.num_seq = num_seq # N
        self.seq_len = seq_len # SL
        self.num_classes = num_classes

        self.last_duration = int(math.ceil(seq_len / 4))
        self.last_size = int(math.ceil(sample_size / 32))
        track_running_stats = True 

        self.backbone, self.param = resnet_2d3d.select_ResNet(
            network, 
            track_running_stats=track_running_stats
            )
        self.param["num_layers"] = 1
        self.param["hidden_size"] = self.param["feature_size"]
        self.param["kernel_size"] = 1

        logger.info("=> Loading RNN + FC model (ConvRNN, kernel size: 1).")
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
        
        # Note that ResNet weights are initialized independently.
        self._initialize_weights(self.final_fc)


    def forward(self, batch):
        """
        self.forward(batch)

        Passes batch through the network.

        Required args
        -------------
        - batch : 6D Tensor
            Input tensor with dims: B x N x C x SL x H x W

        Returns
        -------
        - output : 3D Tensor
            Network output, with dims: B x 1 x number of classes
        - context : list of 4D Tensors
            Final network context, with dims: B x 1 x feature size
        """
        
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
        """
        self._initialize_weights(module)

        Initializes the weights and biases for the input module.

        Required args
        ------------
        - module (nn.Module):
            Module for which to initialize weights and biases.
        """
        
        for name, param in module.named_parameters():
            if "bias" in name:
                torch.nn.init.constant_(param, 0.0)
            elif "weight" in name:
                torch.nn.init.orthogonal_(param, 1)
                 
