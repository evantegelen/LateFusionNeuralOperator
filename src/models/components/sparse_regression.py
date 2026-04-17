"""
Sparse regression layer for PDE discovery and modeling.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import torch.nn as nn
from typing import Optional

class SparseRegressionLayer(nn.Module):
    """Sparse regression layer with L1 regularization for feature selection"""
    
    def __init__(self, 
                 library_size: int, 
                 output_dim: int,
                 l1_reg: float = 0.01):
        """
        Args:
            library_size: Size of the function library
            output_dim: Output dimension (number of variables)
            l1_reg: L1 regularization strength for sparsity
        """
        super().__init__()
        
        self.library_size = library_size
        self.output_dim = output_dim
        self.l1_reg = l1_reg
        
        # Linear layer for regression
        self.regression = nn.Linear(library_size, output_dim, bias=False)
        
        # Initialize with small random weights
        nn.init.normal_(self.regression.weight, mean=0, std=0.01)
    
    def forward(self, library: torch.Tensor) -> torch.Tensor:
        "library: [B, X, L] - Function library, Returns:output: [B, X, V] - Regression output"
        return self.regression(library)
    
    def get_sparsity_loss(self) -> torch.Tensor:
        """Compute L1 regularization loss for sparsity"""
        return self.l1_reg * torch.sum(torch.abs(self.regression.weight))
    
    def get_active_coefficients(self) -> torch.Tensor:
        """Get mask of active (non-zero) coefficients"""
        return torch.abs(self.regression.weight) >= 0.0
    
    def get_coefficients(self) -> torch.Tensor:
        """Get regression coefficients"""
        return self.regression.weight.detach()



