#!/usr/bin/env python

import argparse
from pathlib import Path
import sys

import torch
import torch.nn as nn

sys.path.extend(["..", str(Path("..", "utils"))])
from utils import misc_utils


#############################################
class ConvGRUCell(nn.Module):
    """
    Convolution GRU cell.
    """
    
    def __init__(self, input_size, hidden_size, kernel_size):
        """
        ConvGRUCell(input_size, hidden_size, kernel_size)

        Initializes a convolution GRU cell.

        Required args
        -------------
        - input_size : int
            Input layer size
        - hidden_size : int
            Hidden layer size
        - kernel_size : int  
            Convolution kernel size 
        """
        
        super(ConvGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        padding = kernel_size // 2

        self.reset_gate = nn.Conv2d(
            input_size + hidden_size, 
            hidden_size, 
            kernel_size, 
            padding=padding
            )
        self.update_gate = nn.Conv2d(
            input_size + hidden_size, 
            hidden_size, 
            kernel_size, 
            padding=padding
            )
        self.out_gate = nn.Conv2d(
            input_size + hidden_size, 
            hidden_size, 
            kernel_size, 
            padding=padding
            )

        nn.init.orthogonal_(self.reset_gate.weight)
        nn.init.orthogonal_(self.update_gate.weight)
        nn.init.orthogonal_(self.out_gate.weight)
        nn.init.constant_(self.reset_gate.bias, 0.)
        nn.init.constant_(self.update_gate.bias, 0.)
        nn.init.constant_(self.out_gate.bias, 0.)

    def forward(self, input_tensor, hidden_state):
        """
        self.forward(input_tensor, hidden_state)

        Passes input through the cell, and returns a new state.

        Required args
        -------------
        - input_tensor : 4D Tensor
            Input tensor with dimensions: B x hidden_size x H x W
        - hidden_state : 4D Tensor
            Hidden state tensor with dimensions: B x hidden_size x H x W
            Set to 0s if None.

        Returns
        -------
        - new_state : 4D Tensor
            New hidden state tensor with dimensions: B x hidden_size x H x W
        """
        
        if hidden_state is None:
            B, C, *spatial_dim = input_tensor.size()
            hidden_state = torch.zeros(
                [B, self.hidden_size, *spatial_dim]
                ).to(input_tensor.device)
        # [B, C, H, W]
        combined = torch.cat([input_tensor, hidden_state], dim=1) # concat in C
        update = torch.sigmoid(self.update_gate(combined))
        reset = torch.sigmoid(self.reset_gate(combined))
        out = torch.tanh(self.out_gate(
            torch.cat([input_tensor, hidden_state * reset], dim=1)
            ))
        new_state = hidden_state * (1 - update) + out * update
        return new_state


#############################################
class ConvGRU(nn.Module):
    """
    Convolution GRU neural network module.
    """
    
    def __init__(self, input_size, hidden_size, kernel_size, num_layers, 
                 dropout=0.1):
        """
        ConvGRU(input_size, hidden_size, kernel_size, num_layers)

        Initializes a multi-layer convolution GRU network.

        Required args
        -------------
        - input_size : int
            Input layer size
        - hidden_size : int
            Hidden layer size
        - kernel_size : int  
            Convolution kernel size 
        - num_layers : int
            Number of layers
        
        Optional args
        -------------
        - dropout : float
            Dropout rate for dropout layer
        """

        super(ConvGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.num_layers = num_layers

        cell_list = []
        for i in range(self.num_layers):
            if i == 0:
                input_dim = self.input_size
            else:
                input_dim = self.hidden_size
            cell = ConvGRUCell(
                input_dim, 
                self.hidden_size, 
                self.kernel_size
                )
            name = f"ConvGRUCell_{i:02}"

            setattr(self, name, cell)
            cell_list.append(getattr(self, name))
        
        self.cell_list = nn.ModuleList(cell_list)
        self.dropout_layer = nn.Dropout(p=dropout)


    def forward(self, x, hidden_state=None):
        """
        self.forward(x)

        Passes input through the GRU neural network.

        Required args
        -------------
        - x : 5D Tensor
            Input tensor with dimensions: B x SL x input_size x H x W
            
        Optional args
        -------------
        - hidden_state : list of 4D Tensors (default=None)
            Hidden state tensors for each layer, each with dimensions: 
            B x hidden_size x H x W
            Set to None for each, if None.

        Returns
        -------
        - layer_output :
            Final layer output with dimensions B x output_size x H x W
        - last_state_list : list of 4D Tensors
            Updated hidden state tensors for each layer, each with dimensions: 
            B x num_layers x H x W
        """
        
        [B, seq_len, *_] = x.size()

        if hidden_state is None:
            hidden_state = [None] * self.num_layers
        
        # input: image sequences [B, SL, C, H, W]
        current_layer_input = x 
        del x

        last_state_list = []

        for idx in range(self.num_layers):
            cell_hidden = hidden_state[idx]
            output_inner = []
            for t in range(seq_len):
                cell_hidden = self.cell_list[idx](
                    current_layer_input[:, t, :], cell_hidden
                    )
                # dropout in each time step
                cell_hidden = self.dropout_layer(cell_hidden)
                output_inner.append(cell_hidden)

            layer_output = torch.stack(output_inner, dim=1)
            current_layer_input = layer_output

            last_state_list.append(cell_hidden)

        last_state_list = torch.stack(last_state_list, dim=1)

        return layer_output, last_state_list


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_level', default='info', 
                        help='logging level, e.g., debug, info, error')
    args = parser.parse_args()

    misc_utils.get_logger_with_basic_format(level=args.log_level)


    test_crnn = ConvGRU(
        input_size=10, hidden_size=20, kernel_size=3, num_layers=2
        )
    # [B, seq_len, C, H, W], temporal axis=1
    test_data = torch.randn(4, 5, 10, 6, 6) 
    output, hn = test_crnn(test_data)

    breakpoint()

