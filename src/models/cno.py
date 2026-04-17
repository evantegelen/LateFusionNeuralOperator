import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Any, Optional, Callable
from .base_model import BasePDEModel
from .components.cno1d import CNO1d

# Wrapper classes for existing models
class CNO1DWrapper(BasePDEModel):
    def __init__(self, 
                 num_variables: int, 
                 num_parameters: int, 
                 periodic: bool = True, 
                 padding: Optional[int] = None):
        
        super().__init__()

        in_channels = num_variables + num_parameters + 1
        self.model = CNO1d(in_channels=in_channels,
                          out_channels=num_variables,
                          size = 128,                                
                          N_layers = 4,                      
                          N_res = 4,                          
                          N_res_neck = 4,                      
                          channel_multiplier = 16,      
                          use_bn = False,
                            )
        self.num_variables = num_variables
        self.num_parameters = num_parameters
        self.dim = 1
    
    def forward(self, states: torch.Tensor, parameters: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        parameters = parameters.unsqueeze(1).repeat(1, states.shape[1], 1)  # [B, X, P]
        input = torch.cat([states, parameters], dim=-1)  # [B, X, V + P]
        output = self.model(input, x)  # [B, X, 1, num_variables]
        output = output.squeeze(1)  # [B, X, num_channels]
        return output
