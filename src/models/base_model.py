import torch
import torch.nn as nn
from typing import Dict, Any
from abc import ABC, abstractmethod

# Base Model Interface
class BasePDEModel(nn.Module, ABC):
    """Base class for PDE models to ensure consistent interface"""
    
    def __init__(self, **kwargs):
        super().__init__()
    
    @abstractmethod
    def forward(self, states: torch.Tensor, parameters: torch.Tensor, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Forward pass of the model"""
        pass
    
    def get_loss_components(self, pred: torch.Tensor, target: torch.Tensor, 
                           batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Override this for model-specific loss components"""
        return {}
    
 