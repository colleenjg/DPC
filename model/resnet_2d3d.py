#!/usr/bin/env python

# modified from https://github.com/kenshohara/3D-ResNets-PyTorch
import argparse
import logging
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

sys.path.extend(["..", str(Path("..", "utils"))])
from utils import misc_utils

logger = logging.getLogger(__name__)

TAB = "    "


#############################################
def Conv3x3x3(in_planes, out_planes, stride=1, bias=False):
    """
    Conv3x3x3(in_planes, out_planes)

    Returns a 3D convolution module with paddig and a 3x3x3 kernel size.
    
    Required args
    -------------
    - in_planes : int
        Number of input planes (channels) to the convolution.
    - out_planes : int
        Number of output planes (channels) to the convolution.

    Optional args
    -------------
    - stride : int or tuple (default=1)
        Size of the convolution stride to use.
    - bias : bool (default=False)
        If True, convolution layer will include a bias unit.

    Returns
    -------
    - nn.Conv3d
        3D convolution module with padding and a 3x3x3 kernel size.
    """
    
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=bias
        )


#############################################
def Conv1x3x3(in_planes, out_planes, stride=1, bias=False):
    """
    Conv1x3x3(in_planes, out_planes)

    Returns a pseudo-2D convolution module with paddig and a 1x3x3 kernel size.
    
    Required args
    -------------
    - in_planes : int
        Number of input planes (channels) to the convolution.
    - out_planes : int
        Number of output planes (channels) to the convolution.

    Optional args
    -------------
    - stride : int (default=1)
        Size of the convolution stride to use.
    - bias : bool (default=False)
        If True, convolution layer will include a bias unit.

    Returns
    -------
    - nn.Conv3d
        Pseudo-2D convolution module with padding and a 1x3x3 kernel size.
    """
    
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=(1, 3, 3),
        stride=(1, stride, stride),
        padding=(0, 1, 1),
        bias=bias
        )


#############################################
def downsample_basic_block(x, num_out_planes, stride=1):
    """
    downsample_basic_block(x, num_out_planes)

    Returns the input, downsampled along the spatial dimension (H and W) and 
    padded with 0s along the planes dimension.
    
    Required args
    -------------
    - x : 5D Tensor 
        Input tensor, with dims: B x C_in (planes) x SL_in x H_in x W_in
    - num_out_planes : int
        Number of planes to expand the output to.

    Optional args
    -------------
    - stride : int (default=1)
        Stride to use in downsampling.

    Returns
    -------
    - out : 5D Tensor 
        dims: B x C_out (planes) x SL_out x H_out x W_out.
    """

    if len(x.size()) != 5:
        raise ValueError("'x' must have 5 dimensions.")
    
    if num_out_planes <= x.size(1):
        raise ValueError(
            "The number of planes in 'x' must be smaller than "
            f"'num_out_planes' ({num_out_planes}), but found {x.size(1)}."
            )
    
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), 
        num_out_planes - out.size(1), 
        out.size(2), 
        out.size(3),
        out.size(4)
        ).zero_().to(x.device)

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


#############################################
class BasicBlock3d(nn.Module):
    """
    Constructs a ResNet block comprising 2 3D convolutions, each with a kernel 
    size of 3x3x3.

    Class attributes
    ----------------
    - expansion : int
        Number of times to expand number of planes in final convolution.

    Attributes
    ----------
    - bn1 : nn.BatchNorm3d
        BatchNorm3d layer.
    - bn2 : nn.BatchNorm3d
        BatchNorm3d layer.
    - conv1 : Conv3x3x3
        3D convolution layer.
    - conv2 : Conv3x3x3
        3D convolution layer.
    - downsample : nn.Module
        Downsampling module applied to residual, if not None.
        and variance).
    - relu : nn.ReLU
        ReLU layer.
    - stride : int or tuple
        Size of the convolution stride to use.
    - use_final_relu : bool
        If True, a ReLU is applied to the output layer.

    Methods
    -------
    - self.forward(x):
        Passes the input through the block.
    """
    expansion = 1

    def __init__(self, in_planes, num_planes, stride=1, downsample=None, 
                 track_running_stats=True, use_final_relu=True):
        """
        BasicBlock3d(in_planes, num_planes)

        Constructs a BasicBlock3d object.

        Required args
        -------------
        - in_planes : int
            Number of input planes (channels) to the convolution.
        - num_planes : int
            Number of output planes (channels) for the first convolution, 
            expanded by self.expansion for the second.

        Optional args
        -------------
        - stride : int or tuple (default=1)
            Size of the convolution stride to use.
        - downsample : nn.Module (default=None)
            Downsampling module applied to residual, if not None.
        - track_running_stats : bool (default=True)
            If True, the BatchNorm modules will track statistics (running mean 
            and variance). Otherwise, batch statistics are used.
        - use_final_relu : bool (default=True)
            If True, a ReLU is applied to the output layer.
        """
        
        super(BasicBlock3d, self).__init__()
        bias = False
        self.use_final_relu = use_final_relu
        self.conv1 = Conv3x3x3(in_planes, num_planes, stride, bias=bias)
        self.bn1 = nn.BatchNorm3d(
            num_planes, track_running_stats=track_running_stats
            )

        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv3x3x3(
            num_planes, num_planes * self.expansion, bias=bias
            )
        self.bn2 = nn.BatchNorm3d(
            num_planes, track_running_stats=track_running_stats
            )
        
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        """
        self.forward(x)

        Passes the input through the block.

        Required args
        -------------
        - x : 5D Tensor 
            Input tensor, with dims: B x C_in (planes) x SL_in x H_in x W_in.

        Returns
        -------
        - out : 5D Tensor 
            Output tensor, with dims: 
                B x C_out (planes) x SL_out x H_out x W_out.
        """
        
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.use_final_relu: 
            out = self.relu(out)
    
        return out


#############################################
class BasicBlock2d(nn.Module):
    """
    Constructs a ResNet block comprising 2 pseudo-2D convolutions, each with a 
    kernel size of 1x3x3.

    Class attributes
    ----------------
    - expansion : int
        Number of times to expand number of planes in final convolution.

    Attributes
    ----------
    - bn1 : nn.BatchNorm3d
        BatchNorm3d layer.
    - bn2 : nn.BatchNorm3d
        BatchNorm3d layer.
    - conv1 : Conv1x3x3
        3D convolution layer.
    - conv2 : Conv1x3x3
        3D convolution layer.
    - downsample : nn.Module
        Downsampling module applied to residual, if not None.
        and variance).
    - relu : nn.ReLU
        ReLU layer.
    - stride : int or tuple
        Size of the convolution stride to use.
    - use_final_relu : bool
        If True, a ReLU is applied to the output layer.

    Methods
    -------
    - self.forward(x):
        Passes the input through the block.
    """
    
    expansion = 1

    def __init__(self, in_planes, num_planes, stride=1, downsample=None, 
                 track_running_stats=True, use_final_relu=True):
        """
        BasicBlock2d(in_planes, num_planes)

        Constructs a BasicBlock2d object.

        Required args
        -------------
        - in_planes : int
            Number of input planes (channels) to the convolution.
        - num_planes : int
            Number of output planes (channels) for the first convolution, 
            expanded by self.expansion for the second.

        Optional args
        -------------
        - stride : int or tuple (default=1)
            Size of the convolution stride to use.
        - downsample : nn.Module (default=None)
            Downsampling module applied to residual, if not None.
        - track_running_stats : bool (default=True)
            If True, the BatchNorm modules will track statistics (running mean 
            and variance). Otherwise, batch statistics are used.
        - use_final_relu : bool (default=True)
            If True, a ReLU is applied to the output layer.
        """
        
        super(BasicBlock2d, self).__init__()
        bias = False
        self.use_final_relu = use_final_relu
        self.conv1 = Conv1x3x3(in_planes, num_planes, stride, bias=bias)
        self.bn1 = nn.BatchNorm3d(
            num_planes, track_running_stats=track_running_stats
            )
        
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv1x3x3(
            num_planes, num_planes * self.expansion, bias=bias
            )
        self.bn2 = nn.BatchNorm3d(
            num_planes, track_running_stats=track_running_stats
            )
        
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        """
        self.forward(x)

        Passes the input through the block.

        Required args
        -------------
        - x : 5D Tensor 
            Input tensor, with dims: B x C_in (planes) x SL x H_in x W_in.

        Returns
        -------
        - out :  : 5D Tensor 
            Output tensor, with dims: B x C_out (planes) x SL x H_out x W_out.
        """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.use_final_relu: 
            out = self.relu(out)

        return out


#############################################
class Bottleneck3d(nn.Module):
    """
    Constructs a ResNet block comprising 3 3D convolutions, where the second 
    one has a kernel size of 3.

    Class attributes
    ----------------
    - expansion : int
        Number of times to expand number of planes in final convolution.

    Attributes
    ----------
    - bn1 : nn.BatchNorm3d
        BatchNorm3d layer.
    - bn2 : nn.BatchNorm3d
        BatchNorm3d layer.
    - bn3 : nn.BatchNorm3d
        BatchNorm3d layer.
    - conv1 : nn.Conv3d
        3D convolution layer with a kernel size of 1.
    - conv2 : nn.Conv3d
        3D convolution layer with a kernel size of 3.
    - conv3 : nn.Conv3d
        3D convolution layer with a kernel size of 1.
    - downsample : nn.Module
        Downsampling module applied to residual, if not None.
        and variance).
    - relu : nn.ReLU
        ReLU layer.
    - stride : int or tuple
        Size of the convolution stride to use.
    - use_final_relu : bool
        If True, a ReLU is applied to the output layer.

    Methods
    -------
    - self.forward(x):
        Passes the input through the bottleneck block.    
    """
    
    expansion = 4

    def __init__(self, in_planes, num_planes, stride=1, downsample=None, 
                 track_running_stats=True, use_final_relu=True):
        """
        Bottleneck3d(in_planes, num_planes)

        Constructs a Bottleneck3d object.

        Required args
        -------------
        - in_planes : int
            Number of input planes (channels) to the convolution.
        - num_planes : int
            Number of output planes (channels) for the first convolution, 
            expanded by self.expansion for the second.

        Optional args
        -------------
        - stride : int or tuple (default=1)
            Size of the convolution stride to use.
        - downsample : nn.Module (default=None)
            Downsampling module applied to residual, if not None.
        - track_running_stats : bool (default=True)
            If True, the BatchNorm modules will track statistics (running mean 
            and variance). Otherwise, batch statistics are used.
        - use_final_relu : bool (default=True)
            If True, a ReLU is applied to the output layer.
        """
        
        super(Bottleneck3d, self).__init__()
        bias = False
        self.use_final_relu = use_final_relu
        self.conv1 = nn.Conv3d(in_planes, num_planes, kernel_size=1, bias=bias)
        self.bn1 = nn.BatchNorm3d(
            num_planes, track_running_stats=track_running_stats
            )
        
        self.conv2 = nn.Conv3d(
            num_planes, num_planes, kernel_size=3, stride=stride, padding=1, 
            bias=bias
            )
        self.bn2 = nn.BatchNorm3d(
            num_planes, track_running_stats=track_running_stats
            )
        
        self.conv3 = nn.Conv3d(
            num_planes, num_planes * self.expansion, kernel_size=1, bias=bias
            )
        self.bn3 = nn.BatchNorm3d(
            num_planes * self.expansion, track_running_stats=track_running_stats
            )
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        """
        self.forward(x)

        Passes the input through the block.

        Required args
        -------------
        - x : 5D Tensor 
            Input tensor, with dims: B x C_in (planes) x SL_in x H_in x W_in.

        Returns
        -------
        - out :  : 5D Tensor 
            Output tensor, with dims: 
                B x C_out (planes) x SL_out x H_out x W_out.
        """
        
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.use_final_relu: 
            out = self.relu(out)

        return out


#############################################
class Bottleneck2d(nn.Module):
    """
    Constructs a ResNet block comprising 3 pseudo-2D convolutions, where the 
    second one has a kernel size of 1x3x3.

    Class attributes
    ----------------
    - expansion : int
        Number of times to expand number of planes in final convolution.

    Attributes
    ----------
    - bn1 : nn.BatchNorm3d
        BatchNorm3d layer.
    - bn2 : nn.BatchNorm3d
        BatchNorm3d layer.
    - bn3 : nn.BatchNorm3d
        BatchNorm3d layer.
    - conv1 : nn.Conv3d
        3D convolution layer with a kernel size of 1.
    - conv2 : nn.Conv3d
        3D convolution layer with a kernel size of 1x3x3.
    - conv3 : nn.Conv3d
        3D convolution layer with a kernel size of 1.
    - downsample : nn.Module
        Downsampling module applied to residual, if not None.
        and variance).
    - relu : nn.ReLU
        ReLU layer.
    - stride : int or tuple
        Size of the convolution stride to use.
    - use_final_relu : bool
        If True, a ReLU is applied to the output layer.

    Methods
    -------
    - self.forward(x):
        Passes the input through the bottleneck block.    
    """
    
    expansion = 4

    def __init__(self, in_planes, num_planes, stride=1, downsample=None, 
                 track_running_stats=True, use_final_relu=True):
        """
        Bottleneck2d(in_planes, num_planes)

        Constructs a Bottleneck2d object.

        Required args
        -------------
        - in_planes : int
            Number of input planes (channels) to the convolution.
        - num_planes : int
            Number of output planes (channels) for the first convolution, 
            expanded by self.expansion for the second.

        Optional args
        -------------
        - stride : int or tuple (default=1)
            Size of the convolution stride to use.
        - downsample : nn.Module (default=None)
            Downsampling module applied to residual, if not None.
        - track_running_stats : bool (default=True)
            If True, the BatchNorm modules will track statistics (running mean 
            and variance). Otherwise, batch statistics are used.
        - use_final_relu : bool (default=True)
            If True, a ReLU is applied to the output layer.
        """
        
        super(Bottleneck2d, self).__init__()
        bias = False
        self.use_final_relu = use_final_relu
        self.conv1 = nn.Conv3d(in_planes, num_planes, kernel_size=1, bias=bias)
        self.bn1 = nn.BatchNorm3d(
            num_planes, track_running_stats=track_running_stats
            )
        
        self.conv2 = nn.Conv3d(
            num_planes, num_planes, 
            kernel_size=(1, 3, 3), 
            stride=(1, stride, stride), 
            padding=(0, 1, 1), 
            bias=bias
            )
        self.bn2 = nn.BatchNorm3d(
            num_planes, track_running_stats=track_running_stats
            )
        
        self.conv3 = nn.Conv3d(
            num_planes, num_planes * self.expansion, kernel_size=1, bias=bias
            )
        self.bn3 = nn.BatchNorm3d(
            num_planes * self.expansion, 
            track_running_stats=track_running_stats
            )
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        """
        self.forward(x)

        Passes the input through the block.

        Required args
        -------------
        - x : 5D Tensor 
            Input tensor, with dims: B x C_in (planes) x SL x H_in x W_in.

        Returns
        -------
        - out :  : 5D Tensor 
            Output tensor, with dims: 
                B x C_out (planes) x SL x H_out x W_out.
        """
        
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.use_final_relu: out = self.relu(out)

        return out


#############################################
class ResNet2d3d_full(nn.Module):
    """
    Full ResNet2d3d network module.

    Attributes
    ----------
    - bn1 : nn.BatchNorm3d
        BatchNorm3d layer.
    - conv1 : nn.Conv3d
        Conv3d layer.
    - in_planes : int
        Tracks the next number of input planes, calculated from the most 
        recently created layer.
    - layer1 : nn.Sequential
        First layer of blocks.
    - layer2 : nn.Sequential
        Second layer of blocks.
    - layer3 : nn.Sequential
        Third layer of blocks.
    - layer4 : nn.Sequential
        Fourth layer of blocks.
    - maxpool : nn.MaxPool3d
        MaxPool3d layer.
    - relu : nn.ReLU
        ReLU layer.
    - track_running_stats : bool
        If True, the BatchNorm modules will track statistics (running mean 
        and variance). Otherwise, batch statistics are used.

    Methods
    -------
    - self.forward(x):
        Passes the input through the ResNet2d3d network.
    """
    
    def __init__(self, block_types, blocks_per, track_running_stats=True):
        """
        ResNet2d3d_full(block_types, blocks_per)

        Contructs a ResNet2d3d network.

        Required args
        -------------
        - block_types : list of nn.Module Class
            Types of blocks to use for each of the 4 layers, 
            i.e. Bottleneck2d, Bottleneck3d, BasicBlock2d or BasicBlock3d.
        - blocks_per : int or list
            Number of blocks to include in each of the 4 layers.

        Optional args
        -------------
        - track_running_stats : bool (default=True)
            If True, the BatchNorm modules will track statistics (running mean 
            and variance). Otherwise, batch statistics are used.
        """
        
        super(ResNet2d3d_full, self).__init__()
        self.in_planes = 64
        self.track_running_stats = track_running_stats
        bias = False
        self.conv1 = nn.Conv3d(
            3, 64, 
            kernel_size=(1, 7, 7), 
            stride=(1, 2, 2), 
            padding=(0, 3, 3), 
            bias=bias
            )
        self.bn1 = nn.BatchNorm3d(64, track_running_stats=track_running_stats)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)
            )

        # Create layers
        num_layers = 4
        if not isinstance(blocks_per, list):
            blocks_per = [blocks_per] * num_layers
 
        for var, name in zip(
            [block_types, blocks_per], ["block_types", "blocks_per"]
            ):
            if len(var) != num_layers:
                raise ValueError(
                    f"Expected the length of '{name}' to be {num_layers}, but "
                    f"found {len(var)}."
                    )

        num_planes_out = [64, 128, 256, 256]
        self.layer1 = self._make_layer(
            block_types[0], num_planes_out[0], blocks_per[0]
            )
        self.layer2 = self._make_layer(
            block_types[1], num_planes_out[1], blocks_per[1], stride=2
            )
        self.layer3 = self._make_layer(
            block_types[2], num_planes_out[2], blocks_per[2], stride=2
            )
        self.layer4 = self._make_layer(
            block_types[3], num_planes_out[3], blocks_per[3], stride=2, 
            is_final=True
            )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None: m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_layer(self, block_type, num_planes_out, num_blocks, stride=1, 
                    is_final=False):
        """
        self._make_layer(block_type, num_planes_out, num_blocks)

        Creates a ResNet layer.

        Required args
        -------------
        - block_type : nn.Module Class
            Type of block to use to make up the layer, 
            i.e. Bottleneck2d, Bottleneck3d, BasicBlock2d or BasicBlock3d.
        - num_planes_out : int
            Number of planes to output (i.e., channels)
        - num_blocks : int
            Number of blocks to create in the layer.

        Optional args
        -------------
        - stride : int (default=1)
            Size of the convolution stride to use.
        - is_final : bool (default=False)
            If True, the layer being created is the final layer of the ResNet 
            (and will not end with a ReLU).

        Returns
        -------
        - nn.Sequential
            ResNet layer.
        """
        
        downsample = None
        in_planes_calc = num_planes_out * block_type.expansion
        if stride != 1 or self.in_planes != in_planes_calc:
            # customized_stride to deal with 2d or 3d residual blocks
            if (block_type == Bottleneck2d) or (block_type == BasicBlock2d):
                customized_stride = (1, stride, stride)
            else:
                customized_stride = stride

            downsample = nn.Sequential(
                nn.Conv3d(
                    self.in_planes, 
                    num_planes_out * block_type.expansion, 
                    kernel_size=1, 
                    stride=customized_stride, 
                    bias=False
                    ), 
                nn.BatchNorm3d(
                    num_planes_out * block_type.expansion, 
                    track_running_stats=self.track_running_stats
                    )
                )

        layers = [
            block_type(
                self.in_planes, num_planes_out, stride, downsample, 
                track_running_stats=self.track_running_stats
                )
        ]

        self.in_planes = num_planes_out * block_type.expansion
        for i in range(1, num_blocks):
            use_final_relu = True
            if is_final and (i == num_blocks - 1):
                # if is the final block, no ReLU in the final output
                use_final_relu = False

            layers.append(
                block_type(
                    self.in_planes, num_planes_out, 
                    track_running_stats=self.track_running_stats,
                    use_final_relu=use_final_relu
                    )
                )
                
        return nn.Sequential(*layers)


    def forward(self, x):
        """
        self.forward(x)

        Passes the input through the ResNet2d3d network.

        Required args
        -------------
        - x : 5D Tensor
            Input tensor, with dims: B x C_in (planes=3) x SL_in x H_in x W_in.

        Returns
        -------
        - x : 5D Tensor
            Output tensor, with dims: 
                B x C_out (planes=256) x SL_out x H_out x W_out.            
        """
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x) 
        x = self.layer2(x) 
        x = self.layer3(x) 
        x = self.layer4(x)

        return x


### Full ResNet ###
#############################################
def ResNet18_2d3d_full(**kwargs):
    """
    ResNet18_2d3d_full(**kwargs)

    Constructs a ResNet-18 model.

    Keyword args
    ------------
    - kwargs : dict
        Keyword arguments for ResNet2d3d_full initialization 
        (excluding 'block_types', and 'blocks_per').

    Returns
    -------
    - model : ResNet2d3d_full
        A ResNet-18 model.
    """

    block_types = [BasicBlock2d, BasicBlock2d, BasicBlock3d, BasicBlock3d]
    blocks_per = [2, 2, 2, 2]

    model = ResNet2d3d_full(
        block_types=block_types,
        blocks_per=blocks_per,
        **kwargs
        )

    return model


#############################################
def ResNet34_2d3d_full(**kwargs):
    """
    ResNet34_2d3d_full(**kwargs)
    
    Constructs a ResNet-34 model. 

    Keyword args
    ------------
    - kwargs : dict
        Keyword arguments for ResNet2d3d_full initialization 
        (excluding 'block_types', and 'blocks_per').

    Returns
    -------
    - model : ResNet2d3d_full
        A ResNet-34 model.
    """

    block_types = [BasicBlock2d, BasicBlock2d, BasicBlock3d, BasicBlock3d]
    blocks_per = [3, 4, 6, 3]

    model = ResNet2d3d_full(
        block_types=block_types,
        blocks_per=blocks_per,
        **kwargs
        )

    return model


#############################################
def ResNet50_2d3d_full(**kwargs):
    """
    ResNet50_2d3d_full(**kwargs)

    Constructs a ResNet-50 model.

    Keyword args
    ------------
    - kwargs : dict
        Keyword arguments for ResNet2d3d_full initialization 
        (excluding 'block_types', and 'blocks_per').

    Returns
    -------
    - model : ResNet2d3d_full
        A ResNet-50 model.
    """

    block_types = [Bottleneck2d, Bottleneck2d, Bottleneck3d, Bottleneck3d]
    blocks_per = [3, 4, 6, 3]

    model = ResNet2d3d_full(
        block_types=block_types,
        blocks_per=blocks_per,
        **kwargs
        )

    return model


#############################################
def ResNet101_2d3d_full(**kwargs):
    """
    ResNet101_2d3d_full(**kwargs)

    Constructs a ResNet-101 model.

    Keyword args
    ------------
    - kwargs : dict
        Keyword arguments for ResNet2d3d_full initialization 
        (excluding 'block_types', and 'blocks_per').

    Returns
    -------
    - model : ResNet2d3d_full
        A ResNet-101 model.
    """

    block_types = [Bottleneck2d, Bottleneck2d, Bottleneck3d, Bottleneck3d]
    blocks_per = [3, 4, 23, 3]

    model = ResNet2d3d_full(
        block_types=block_types,
        blocks_per=blocks_per,
        **kwargs
        )

    return model


#############################################
def ResNet152_2d3d_full(**kwargs):
    """
    ResNet152_2d3d_full(**kwargs)

    Constructs a ResNet-152 model.

    Keyword args
    ------------
    - kwargs : dict
        Keyword arguments for ResNet2d3d_full initialization 
        (excluding 'block_types', and 'blocks_per').

    Returns
    -------
    - model : ResNet2d3d_full
        A ResNet-152 model.
    """

    block_types = [Bottleneck2d, Bottleneck2d, Bottleneck3d, Bottleneck3d]
    blocks_per = [3, 8, 36, 3]

    model = ResNet2d3d_full(
        block_types=block_types,
        blocks_per=blocks_per,
        **kwargs
        )

    return model


#############################################
def ResNet200_2d3d_full(**kwargs):
    """
    ResNet200_2d3d_full(**kwargs)

    Constructs a ResNet-200 model.

    Keyword args
    ------------
    - kwargs : dict
        Keyword arguments for ResNet2d3d_full initialization 
        (excluding 'block_types', and 'blocks_per').

    Returns
    -------
    - model : ResNet2d3d_full
        A ResNet-200 model.
    """

    block_types = [Bottleneck2d, Bottleneck2d, Bottleneck3d, Bottleneck3d]
    blocks_per = [3, 24, 36, 3]
    
    model = ResNet2d3d_full(
        block_types=block_types,
        blocks_per=blocks_per,
        **kwargs
        )

    return model


#############################################
def neq_load_customized(model, pretrained_dict):
    """
    neq_load_customized(model, pretrained_dict)
    
    Loads pretrained model parameters in a non-equal way, in place. To be used 
    when the model has a partially different set of parameters from the ones 
    stored in the pretrained parameter dictionary.

    Required args
    -------------
    - model : nn.Module
        Model into which to load the recorded parameters.
    - pretrained_dict : dict
        Pretrained model state dictionary in which pretrained model parameters 
        are stored under parameter name keys.
    """
    
    model_dict = model.state_dict()
    tmp = dict()

    log_str = "\n========= Weight loading ========="

    # Check for weights not used from pretrained file
    not_used = ""
    for k, v in pretrained_dict.items():
        if k in model_dict:
            tmp[k] = v
        else:
            not_used = f"{not_used}\n{TAB}{k}"

    if len(not_used):
        log_str = f"{log_str}\nWeights not used from pretrained file:{not_used}"

    # Check for weights not loaded into new model
    not_loaded = ""
    for k, v in model_dict.items():
        if k not in pretrained_dict:
            not_loaded = f"{not_loaded}\n{TAB}{k}"
    
    if len(not_loaded):
        if len(not_used):
            log_str = f"{log_str}\n---------------------------"
        log_str = f"{log_str}\nWeights not loaded into new model:{not_loaded}"
    
    log_str = f"{log_str}\n==================================="
    logger.warning(log_str, extra={"spacing": "\n"})

    del pretrained_dict
    model_dict.update(tmp)
    del tmp
    model.load_state_dict(model_dict)
    
    return model


#############################################
def select_ResNet(network="resnet18", track_running_stats=True):
    """
    select_ResNet(network)

    Returns the specified ResNet2d3d_full model.

    Optional args
    -------------
    - network : str (default="resnet18")
    - track_running_state : bool (default=True)

    Returns
    -------
    - model : ResNet2d3d_full
        ResNet2d3d_full network.
    - param : dict
        Parameter dictionary, with key "feature_size".
    """
    
    resnet_n = network.lower().replace("resnet", "")
    if not resnet_n.isdigit():
        raise ValueError(
            "network should have form 'resnet###', e.g. 'resnet18'."
            )

    resnet_n = int(resnet_n)

    if resnet_n == 18:
        resnet_fn = ResNet18_2d3d_full
        feat_size = 256
    elif resnet_n == 34:
        resnet_fn = ResNet34_2d3d_full
        feat_size = 256
    elif resnet_n == 50:
        resnet_fn = ResNet50_2d3d_full
        feat_size = 1024
    elif resnet_n == 101:
        resnet_fn = ResNet101_2d3d_full
        feat_size = 1024
    elif resnet_n == 152:
        resnet_fn = ResNet152_2d3d_full
        feat_size = 1024
    elif resnet_n == 200:
        resnet_fn = ResNet200_2d3d_full
        feat_size = 1024
    else:
        raise NotImplementedError(
            f"{network} network type is not implemented."
            )

    model = resnet_fn(track_running_stats=track_running_stats)
    param = {"feature_size": feat_size}

    return model, param


#############################################
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_level', default='info', 
                        help='logging level, e.g., debug, info, error')
    args = parser.parse_args()

    misc_utils.get_logger_with_basic_format(level=args.log_level)


    test_model = ResNet18_2d3d_full()
    test_data = torch.FloatTensor(4, 3, 16, 128, 128)
    nn.init.normal_(test_data)
    
    breakpoint()
    
    test_model(test_data)

