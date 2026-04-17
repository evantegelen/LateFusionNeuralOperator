import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Any, Optional, Callable
from abc import ABC, abstractmethod
from .components.fno1d import FNO1d
from .components.fno2d import FNO2d
from .components.libraries import PolynomialLibrary, SimpleLibrary
from .components.sparse_regression import SparseRegressionLayer
from .base_model import BasePDEModel
from .components.cno1d import CNO1d

class LateFusionFNO1dWrapper(BasePDEModel):
    def __init__(self, 
                 num_variables: int, 
                 num_parameters: int, 
                 num_hidden:int=4, 
                 modes: int = 16, 
                 width: int = 64, 
                 periodic: bool = True, 
                 padding: Optional[int] = None,
                 library_type: str = 'polynomial', 
                 order_states: int = 2,
                 order_parameters: int = 1,
                 sparsity_weight: float = 1e-3,
                 operator: str = 'FNO'):
        
        super().__init__()
        self.operator = operator
        self.fno = FNO1d(in_channels=num_variables+1, 
                         out_channels=num_hidden, 
                         modes=modes, 
                         width=width,
                         periodic=periodic, 
                         padding=padding)
        if self.operator == 'CNO':
            self.fno = CNO1d(in_channels=num_variables+1,
                             out_channels=num_hidden,
                             size= 128,
                             N_layers=4,
                             N_res=4,
                             N_res_neck=4,
                             channel_multiplier=16,
                             use_bn=False)
        if library_type == 'polynomial':
            self.library = PolynomialLibrary(order_states=int(order_states), 
                                            order_parameters=int(order_parameters),
                                            num_variables=num_variables, 
                                            num_parameters=num_parameters, 
                                            hidden_dim=int(num_hidden), 
                                            include_interactions=True,)
        elif library_type == 'simple':
            self.library = SimpleLibrary(num_variables=num_variables, 
                                         num_parameters=num_parameters, 
                                         hidden_dim=int(num_hidden))
        else:
            raise ValueError(f"Currently unsupported library type: {library_type}")
        
        self.regression = SparseRegressionLayer(library_size=self.library.library_size, 
                                                output_dim=num_variables, 
                                                l1_reg=float(sparsity_weight))
        
        self.sparsity_weight = float(sparsity_weight)
    
    def forward(self, states: torch.Tensor, parameters: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        fno_output = self.fno(states, x)
        if self.operator == 'CNO':
            fno_output = fno_output.squeeze(1)  # [B, X, hidden_dim]
        else:
            fno_output = fno_output.squeeze(-2)  # [B, X, hidden_dim]
        # Create library and regression output
        library_output = self.library(fno_output, parameters)  # [B, X, library_size]
        output = self.regression(library_output)  # [B, X, num_variables]
        return states + output
    
    def get_loss_components(self, pred: torch.Tensor, target: torch.Tensor, 
                           batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Add sparsity loss component"""
        sparsity_loss = self.regression.get_sparsity_loss()
        return {
            'sparsity_loss': self.sparsity_weight * sparsity_loss
        }
    
class LateFusionFNO2dWrapper(BasePDEModel):
    def __init__(self, 
                 num_variables: int, 
                 num_parameters: int, 
                 num_hidden:int=4, 
                 modes: int = 12, 
                 width: int = 32, 
                 periodic: bool = True, 
                 padding: Optional[int] = None,
                 library_type: str = 'polynomial', 
                 order_states: int = 2,
                 order_parameters: int = 1,
                 sparsity_weight: float = 1e-3):
        super().__init__()

        # Use FNO2d for 2D data
        self.fno = FNO2d(in_channels=num_variables+2, 
                         out_channels=num_hidden, 
                         modes1=modes,
                         modes2=modes, 
                         width=width,
                         periodic=periodic, 
                         padding=padding)
        self.library = PolynomialLibrary(order_states=int(order_states), 
                                         order_parameters=int(order_parameters),
                                         num_variables=num_variables, 
                                         num_parameters=num_parameters, 
                                         hidden_dim=int(num_hidden), 
                                         include_interactions=True,)
        self.regression = SparseRegressionLayer(library_size=self.library.library_size, 
                                                output_dim=num_variables, 
                                                l1_reg=float(sparsity_weight))
        self.dim = 2
        self.sparsity_weight = float(sparsity_weight)
    
    def forward(self, states: torch.Tensor, parameters: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            states: [B, X, Y, V] - Input states  
            parameters: [B, P] - Input parameters
            x: [B, X] - X coordinates
        """
        batch_size = states.shape[0]
        
        # Forward through FNO2D
        fno_output = self.fno(states, x)  # [B, X, Y, 1, H]
        fno_output = fno_output.squeeze(-2)  # [B, X, Y, H]
        # Reshape for library processing: [B, X*Y, H]
        batch_size, x_dim, y_dim, hidden_dim = fno_output.shape
        fno_output_flat = fno_output.view(batch_size, x_dim * y_dim, hidden_dim)
        # Create library and regression output
        library_output = self.library(fno_output_flat, parameters)  # [B, X*Y, library_size]
        regression_output = self.regression(library_output)  # [B, X*Y, num_variables]
        # Reshape back to 2D: [B, X, Y, V]
        output = regression_output.view(batch_size, x_dim, y_dim, -1)
        
        return states + output
    
    def get_loss_components(self, pred: torch.Tensor, target: torch.Tensor, 
                           batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Add sparsity loss component"""
        sparsity_loss = self.regression.get_sparsity_loss()
        return {
            'sparsity_loss': self.sparsity_weight * sparsity_loss
        }


