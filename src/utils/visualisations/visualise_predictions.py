import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib as mpl
import sys

"""
Utility function for visualising predictions during training and testing. Not used for evaluation.
"""

def _variable_save_path(save_path, variable_idx):
    if save_path is None:
        return None
    root, ext = os.path.splitext(save_path)
    ext = ext if ext else ".png"
    return f"{root}_var{variable_idx}{ext}"


def visualise_predictions(predicted, true=None, plot_index=0, save_path=None, show=False):
    """
    for visualising predictions during training, not correct figures sizes
    predicted: tensor [B, T, X, V] or [B, T, X, Y, V]
    true: optional tensor with same shape
    plot_index: batch index
    variable_idx: which variable/channel to plot (for multi-variable PDEs)
    """
    dimension = predicted.ndim - 3

    figs = []

    if dimension == 1:
        pred = predicted[plot_index].detach().cpu().numpy()  # [T, X, V]
        if true is not None:
            true = true[plot_index].detach().cpu().numpy()   # [T, X, V]

        num_variables = pred.shape[-1]

        for v in range(num_variables):
            pred_v = pred[..., v].T  # [X, T]

            vmin = float(np.min(pred_v))
            vmax = float(np.max(pred_v))
            if true is not None:
                true_v = true[..., v].T
                vmin = min(vmin, float(np.min(true_v)))
                vmax = max(vmax, float(np.max(true_v)))

            if true is not None:
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 5))
            else:
                fig, ax1 = plt.subplots(1, 1, figsize=(7, 5))

            im = ax1.imshow(pred_v, aspect='auto', origin='lower', vmin=vmin, vmax=vmax, cmap='viridis')
            ax1.set_title(f'Predicted - Variable {v}')
            ax1.set_xlabel('Time Step')
            ax1.set_ylabel('x')
            fig.colorbar(im, ax=ax1)

            if true is not None:
                im2 = ax2.imshow(true_v, aspect='auto', origin='lower', vmin=vmin, vmax=vmax, cmap='viridis')
                ax2.set_title(f'True - Variable {v}')
                ax2.set_xlabel('Time Step')
                ax2.set_ylabel('x')
                fig.colorbar(im2, ax=ax2)

                error_v = pred_v - true_v
                absmax = float(np.max(np.abs(error_v))) or 1e-8
                im3 = ax3.imshow(error_v, aspect='auto', origin='lower', vmin=-absmax, vmax=absmax, cmap='bwr')
                ax3.set_title(f'Error (Pred - True) - Variable {v}')
                ax3.set_xlabel('Time Step')
                ax3.set_ylabel('x')
                fig.colorbar(im3, ax=ax3)

            plt.tight_layout()

            variable_save_path = _variable_save_path(save_path, v)
            if variable_save_path:
                save_dir = os.path.dirname(variable_save_path)
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                fig.savefig(variable_save_path, dpi=150, bbox_inches="tight")

            if show:
                plt.show()
            else:
                plt.close(fig)

            figs.append(fig)

    elif dimension == 2:
        pred = predicted[plot_index].detach().cpu().numpy()  # [T, X, Y, V]
        if true is None:
            print("2D plotting expects true solution for 3x3 layout.")
            return
        true = true[plot_index].detach().cpu().numpy()       # [T, X, Y, V]

        T = pred.shape[0]
        t_idxs = [
            int(round(0.25 * (T - 1))),
            int(round(0.70 * (T - 1))),
            T - 1
        ]
        t_labels = ["t = 0.25T", "t = 0.7T", "t = T"]

        for v in range(pred.shape[-1]):
            pred_slices = [pred[t, :, :, v] for t in t_idxs]
            true_slices = [true[t, :, :, v] for t in t_idxs]
            err_slices = [p - g for p, g in zip(pred_slices, true_slices)]

            data_min = min(float(np.min(s)) for s in (pred_slices + true_slices))
            data_max = max(float(np.max(s)) for s in (pred_slices + true_slices))
            err_absmax = max(float(np.max(np.abs(s))) for s in err_slices) or 1e-8

            fig, axes = plt.subplots(3, 3, figsize=(13, 11))

            for j in range(3):
                im_true = axes[0, j].imshow(true_slices[j], origin='lower', cmap='viridis',
                                            vmin=data_min, vmax=data_max)
                axes[0, j].set_title(f"True ({t_labels[j]}) - Variable {v}")
                axes[0, j].set_xlabel("y")
                axes[0, j].set_ylabel("x")

                im_pred = axes[1, j].imshow(pred_slices[j], origin='lower', cmap='viridis',
                                            vmin=data_min, vmax=data_max)
                axes[1, j].set_title(f"Predicted ({t_labels[j]}) - Variable {v}")
                axes[1, j].set_xlabel("y")
                axes[1, j].set_ylabel("x")

                im_err = axes[2, j].imshow(err_slices[j], origin='lower', cmap='bwr',
                                           vmin=-err_absmax, vmax=err_absmax)
                axes[2, j].set_title(f"Error ({t_labels[j]}) - Variable {v}")
                axes[2, j].set_xlabel("y")
                axes[2, j].set_ylabel("x")

            fig.colorbar(im_pred, ax=axes[0:2, :], fraction=0.02, pad=0.02, label="State value")
            fig.colorbar(im_err, ax=axes[2, :], fraction=0.02, pad=0.02, label="Pred - True")
            plt.tight_layout()

            variable_save_path = _variable_save_path(save_path, v)
            if variable_save_path:
                save_dir = os.path.dirname(variable_save_path)
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                fig.savefig(variable_save_path, dpi=150, bbox_inches="tight")

            if show:
                plt.show()
            else:
                plt.close(fig)

            figs.append(fig)

    else:
        raise ValueError(f"Unsupported prediction dimensionality: {predicted.ndim}")

    return figs


