from io import BytesIO
from PIL import Image
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import CometLogger
from typing import Dict, Any
import os
from src.utils.visualisations.visualise_predictions import visualise_predictions
from dotenv import load_dotenv  # Optional: if using .env file
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()  # Optional: if using .env file

from .base_model import BasePDEModel 
from .late_fusion import LateFusionFNO1dWrapper, LateFusionFNO2dWrapper
from .fno import FNO1DWrapper, FNO2DWrapper
from .cape_fno import CAPE_FNO1DWrapper, CAPE_FNO2DWrapper
from .cno import CNO1DWrapper

def create_model(model_name: str, model_config: Dict[str, Any]) -> BasePDEModel:
    """Factory function to create models"""
    
    models = {
        'fno1d': FNO1DWrapper,
        'late_fusion_fno1d': LateFusionFNO1dWrapper,
        'cape_fno1d': CAPE_FNO1DWrapper,
        'fno2d': FNO2DWrapper,
        'late_fusion_fno2d': LateFusionFNO2dWrapper,
        'cape_fno2d': CAPE_FNO2DWrapper,
        'cno1d': CNO1DWrapper,
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name](**model_config)

class PDELightningModule(pl.LightningModule):
    def __init__(self, 
                 model_name: str,
                 model_config: Dict[str, Any],
                 learning_rate: float = 1e-3,
                 log_hyperparameters: bool = True):
        super().__init__()

        if log_hyperparameters:
            self.save_hyperparameters()
        
        # Create model
        self.model = create_model(model_name, model_config)
        self.loss_fn = nn.MSELoss()

    def forward(self, states: torch.Tensor, parameters: torch.Tensor, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        return self.model(states, parameters, x)
        
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:

        pred = self.forward(batch['input_state'], batch['parameter'], batch['x'], batch.get('y', None))
        target = batch['target_state']
        
        loss = self.loss_fn(pred, target)

        # Add model-specific loss components (automatically handles both sparsity and embedding)
        additional_losses = self.model.get_loss_components(pred, target, batch)
        for loss_name, loss_value in additional_losses.items():
            loss = loss + loss_value
            self.log(f'train_{loss_name}', loss_value, prog_bar=True)  # Log individual components
        
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        pred = self.roll_out_predictions(batch)  # [B, T-1, X, num_variables]
        target = batch['state'][:, 1:, :, :]  # Shifted target states
        loss = self.loss_fn(pred, target)
        self.log('val_loss', loss, prog_bar=True)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        pred = self.roll_out_predictions(batch)  # [B, T-1, X, num_variables]
        target = batch['state'][:, 1:,...]  # Shifted target states
        loss = self.loss_fn(pred, target)
        self.log('test_loss', loss, prog_bar=True)
    
    def roll_out_predictions(self, batch: Dict[str, Any]) -> torch.Tensor:
        input_state = batch['state'][:, 0, :, :]
        pred_states = []
        #Loop through the full batch autoregressively
        for t in range(batch['state'].shape[1]-1):
            pred = self.forward(input_state, batch['parameter'], batch['x'])
            input_state = pred.squeeze(1)  # Update input_state for next time step
            pred_states.append(pred)
        
        pred = torch.stack(pred_states, dim=1)  # [B, T-1, X, num_variables]
        return pred
    
    
    def _log_figure_to_comet(self, fig, name: str, step: int) -> None:
        if not isinstance(self.logger, CometLogger):
            return
        if self.trainer is None or not self.trainer.is_global_zero:
            return

        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        buffer.seek(0)

        self.logger.experiment.log_image(
            image_data=Image.open(buffer),
            name=name,
            step=step,
        )

        buffer.close()
        plt.close(fig)

    def on_test_epoch_end(self) -> None:
        if self.trainer is None or self.trainer.datamodule is None:
            return
        
        test_loader = self.trainer.datamodule.test_dataloader()
        test_od_loader = self.trainer.datamodule.get_test_od_dataloaders()
        
        batch = next(iter(test_loader))
        batch = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in batch.items()}
        pred = self.roll_out_predictions(batch).cpu()   # [B, T-1, X, V]
        target = batch['state'][:, 1:, ...].cpu()      # [B, T-1, X, V]

        figs = visualise_predictions(pred, true=target, plot_index=0, save_path=None, show=False)
        for i, fig in enumerate(figs):
            self._log_figure_to_comet(fig, name=f"test_id/predictions_var{i}", step=self.global_step)

        # If an OD test loader is provided, run the same visualisation on one batch from it
        if test_od_loader is not None:
            od_batch = next(iter(test_od_loader))
            od_batch = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in od_batch.items()}
            pred = self.roll_out_predictions(od_batch).cpu()   # [B, T-1, X, V]
            target = od_batch['state'][:, 1:, ...].cpu()   # [B, T-1, X, V]
            
            figs = visualise_predictions(pred, true=target, plot_index=0, save_path=None, show=False)
            for i, fig in enumerate(figs):
                self._log_figure_to_comet(fig, name=f"test_od/predictions_var{i}", step=self.global_step)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

        # Halve learning rate  every 50 epochs
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=50,
            gamma=0.5,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
    
def create_comet_logger(
    project_name: str,
    experiment_name: str,
    model_name: str,
    equation_name: str,
    config: Dict[str, Any] = None,
    tags: list = None) -> CometLogger:

    """Create Comet logger with consistent naming"""
    if tags is None:
        tags = [model_name, equation_name]

    logger = CometLogger(
        api_key=os.getenv("COMET_API_KEY"),
        workspace=os.getenv("COMET_WORKSPACE"),
        project=project_name,
        name=experiment_name,
        offline_directory="comet_logs",
    )

    if config is not None:
        logger.log_hyperparams(config)
    if tags:
        logger.experiment.add_tags(tags)

    return logger