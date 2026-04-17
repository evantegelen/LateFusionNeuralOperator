import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Any, Optional, Callable
from .base_model import BasePDEModel
from .components.fno1d import FNO1d
from .components.fno2d import FNO2d

# Wrapper classes for existing models
class FNO1DWrapper(BasePDEModel):
    def __init__(self, 
                 num_variables: int, 
                 num_parameters: int, 
                 modes: int = 16, 
                 width: int = 64, 
                 periodic: bool = True, 
                 padding: Optional[int] = None):
        
        super().__init__()

        num_channels = num_variables + num_parameters + 1
        self.model = FNO1d(in_channels=num_channels, 
                           out_channels=num_variables, 
                           modes=modes, 
                           width=width, 
                           periodic=periodic, 
                           padding=padding)
        self.num_variables = num_variables
        self.dim = 1
    
    def forward(self, states: torch.Tensor, parameters: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        parameters = parameters.unsqueeze(1).expand(-1, states.shape[1], -1)  # [B, X, P]
        input_tensor = torch.cat([states, parameters], dim=-1)  # [B, X, V+P]
        
        output = self.model(input_tensor, x)  # [B, X, 1, num_variables]
        output = output.squeeze(-2)  # [B, X, num_channels]
        return output

class FNO2DWrapper(BasePDEModel):
    def __init__(self, 
                 num_variables: int, 
                 num_parameters: int, 
                 modes: int = 12, 
                 width: int = 32, 
                 periodic: bool = True, 
                 padding: Optional[int] = None):
        
        super().__init__()

        self.num_variables = num_variables
        self.dim = 2
        num_channels = num_variables + num_parameters + 2

        self.model = FNO2d(in_channels=num_channels, 
                           out_channels=num_variables, 
                           modes1=modes, 
                           modes2=modes, 
                           width=width, 
                           periodic=periodic, 
                           padding=padding)
        
    def forward(self, states: torch.Tensor, parameters: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        parameters = parameters.unsqueeze(1).unsqueeze(1).expand(-1, states.shape[1], states.shape[2], -1)  # [B, X, Y, P]
        input_tensor = torch.cat([states, parameters], dim=-1)  # [B, X, Y, V+P+1]

        output = self.model(input_tensor, x)  # [B, X, Y, num_variables]
        output = output.squeeze(-2)  # [B, X, Y, num_variables]
        return output