import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, List, Optional, Tuple
import matplotlib as mpl
import pandas as pd


def plot_id_od_heatmaps(
    bundle: Dict,
    max_models: int = 3,
    font_size: int = 10,
    model_order: Optional[List[str]] = None,
    panel_labels: Optional[tuple] = ("A", "B"),
) -> plt.Figure:
    """
    Exact notebook layout:
    - Two side-by-side 2x4 grids (ID on left, OD on right)
    - Top row: GT + up to 3 model predictions
    - Bottom row: 3 error maps
    - External colorbars and panel labels A/B
    """
    preds_id = bundle["predictions_dict_id"]
    preds_od = bundle["predictions_dict_od"]
    true_solution_id = bundle["true_solution_id"]
    true_solution_od = bundle["true_solution_od"]
    t_coord = bundle.get("t_coord", None)
    x_coord = bundle.get("x_coord", None)

    # For 1D problems, plot first variable
    true_vis_id = true_solution_id[:, :, 0].squeeze()
    true_vis_od = true_solution_od[:, :, 0].squeeze()

    preds_vis_id = {k: v[:, :, 0] for k, v in preds_id.items()}
    preds_vis_od = {k: v[:, :, 0] for k, v in preds_od.items()}

    # enforce plotting order if provided
    if model_order is None:
        model_names = list(preds_vis_id.keys())
    else:
        preferred = [m for m in model_order if m in preds_vis_id and m in preds_vis_od]
        remaining = [m for m in preds_vis_id.keys() if m not in preferred and m in preds_vis_od]
        model_names = preferred + remaining

    n_show = min(max_models, 3, len(model_names), len(preds_vis_id), len(preds_vis_od))


    # Shared scales for prediction panels
    v_min_id = true_vis_id.min() - 0.15 * abs(true_vis_id.max() - true_vis_id.min())
    v_max_id = true_vis_id.max() + 0.15 * abs(true_vis_id.max() - true_vis_id.min())
    v_min_od = true_vis_od.min() - 0.15 * abs(true_vis_od.max() - true_vis_od.min())
    v_max_od = true_vis_od.max() + 0.15 * abs(true_vis_od.max() - true_vis_od.min())

    # Error scales (separate for ID and OD)
    all_error_maps_id = [(preds_vis_id[m] - true_vis_id) for m in model_names[:n_show]]
    all_error_maps_od = [(preds_vis_od[m] - true_vis_od) for m in model_names[:n_show]]

    max_abs_error_id = max(float(np.abs(e).max()) for e in all_error_maps_id) if all_error_maps_id else 1.0
    max_abs_error_od = max(float(np.abs(e).max()) for e in all_error_maps_od) if all_error_maps_od else 1.0
    err_vmin_id, err_vmax_id = -max_abs_error_id, max_abs_error_id
    err_vmin_od, err_vmax_od = -max_abs_error_od, max_abs_error_od

    # Colormap with out-of-bounds in black
    cmap_viridis = plt.get_cmap("viridis").copy()
    cmap_viridis.set_over("black")
    cmap_viridis.set_under("black")

    # Figure layout (exact notebook geometry)
    fig = plt.figure(figsize=(10, 2.8))
    gs_1 = fig.add_gridspec(
        ncols=4, nrows=2,
        bottom=0.1, left=0.05, top=0.9, right=0.44,
        wspace=0.2, hspace=0.2
    )
    gs_2 = fig.add_gridspec(
        ncols=4, nrows=2,
        bottom=0.1, left=0.56, top=0.9, right=0.95,
        wspace=0.2, hspace=0.2
    )


    # Axes
    ax_1 = fig.add_subplot(gs_1[0, 0]); ax_2 = fig.add_subplot(gs_1[0, 1]); ax_3 = fig.add_subplot(gs_1[0, 2]); ax_4 = fig.add_subplot(gs_1[0, 3])
    ax_5 = fig.add_subplot(gs_1[1, 1]); ax_6 = fig.add_subplot(gs_1[1, 2]); ax_7 = fig.add_subplot(gs_1[1, 3])

    ax_8 = fig.add_subplot(gs_2[0, 0]); ax_9 = fig.add_subplot(gs_2[0, 1]); ax_10 = fig.add_subplot(gs_2[0, 2]); ax_11 = fig.add_subplot(gs_2[0, 3])
    ax_12 = fig.add_subplot(gs_2[1, 1]); ax_13 = fig.add_subplot(gs_2[1, 2]); ax_14 = fig.add_subplot(gs_2[1, 3])

    top_axes_id = [ax_1, ax_2, ax_3, ax_4]
    bottom_axes_id = [ax_5, ax_6, ax_7]
    top_axes_od = [ax_8, ax_9, ax_10, ax_11]
    bottom_axes_od = [ax_12, ax_13, ax_14]

    # Extent from coordinates
    t_extent = None
    if t_coord is not None and x_coord is not None:
        t_vals = t_coord.flatten() if np.ndim(t_coord) > 1 else t_coord
        x_vals = x_coord.flatten() if np.ndim(x_coord) > 1 else x_coord
        if len(t_vals) > 1:
            dt = (t_vals[-1] - t_vals[0]) / (len(t_vals) - 1)
            t_extent = [t_vals[0] - dt / 2, t_vals[-1] + dt / 2, x_vals[0], x_vals[-1]]
        else:
            t_extent = [0, 1, x_vals[0], x_vals[-1]]

    # Ground truth
    im_true_id = ax_1.imshow(true_vis_id.T, aspect="auto", origin="lower", cmap=cmap_viridis, vmin=v_min_id, vmax=v_max_id, extent=t_extent)
    ax_1.set_title("Ground Truth", fontsize=font_size)
    ax_1.set_xlabel("t", fontsize=font_size)
    ax_1.set_ylabel("x", fontsize=font_size)
    ax_1.tick_params(labelsize=font_size)

    im_true_od = ax_8.imshow(true_vis_od.T, aspect="auto", origin="lower", cmap=cmap_viridis, vmin=v_min_od, vmax=v_max_od, extent=t_extent)
    ax_8.set_title("Ground Truth", fontsize=font_size)
    ax_8.set_xlabel("t", fontsize=font_size)
    ax_8.set_ylabel("x", fontsize=font_size)
    ax_8.tick_params(labelsize=font_size)

    # Shared ticks
    all_axes = [ax_1, ax_2, ax_3, ax_4, ax_5, ax_6, ax_7, ax_8, ax_9, ax_10, ax_11, ax_12, ax_13, ax_14]
    if t_extent is not None:
        t_min, t_max = t_extent[0], t_extent[1]
        t_mid = (t_min + t_max) / 2
        xticks = [t_min, t_mid, t_max]
        xticklabels = [f"{t_min:.2f}", f"{t_mid:.2f}", f"{t_max:.2f}"]

        y_min, y_max = t_extent[2], t_extent[3]
        yticks = [y_min + frac * (y_max - y_min) for frac in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)]
        yticklabels = [f"{y:.1f}" for y in yticks]

        for ax in all_axes:
            ax.set_xticks(xticks)
            ax.set_yticks(yticks)

        ax_1.set_xticklabels(xticklabels); ax_8.set_xticklabels(xticklabels)
        ax_1.set_yticklabels(yticklabels); ax_8.set_yticklabels(yticklabels)

        for ax in [ax_2, ax_3, ax_4, ax_5, ax_6, ax_7, ax_9, ax_10, ax_11, ax_12, ax_13, ax_14]:
            ax.set_xticklabels([])
            ax.set_yticklabels([])

    def _pretty_name(model_name: str) -> str:
        if model_name == "fno":
            return "FNO"
        if model_name == "cape_fno":
            return "CAPE-FNO"
        if model_name == "late_fusion":
            return "Late Fusion"
        return model_name

    im_err_id = None
    im_err_od = None

    for i, model_name in enumerate(model_names):
        if i >= n_show:
            break

        plot_name = _pretty_name(model_name)

        # ID
        pred_id = preds_vis_id[model_name]
        err_id = pred_id - true_vis_id

        ax_pred_id = top_axes_id[i + 1]
        ax_pred_id.imshow(pred_id.T, aspect="auto", origin="lower", cmap=cmap_viridis, vmin=v_min_id, vmax=v_max_id, extent=t_extent)
        ax_pred_id.set_title(plot_name, fontsize=font_size)
        ax_pred_id.set_yticklabels([])

        ax_err_id = bottom_axes_id[i]
        im_err_id = ax_err_id.imshow(err_id.T, aspect="auto", origin="lower", cmap="bwr", vmin=err_vmin_id, vmax=err_vmax_id, extent=t_extent)
        ax_err_id.set_ylabel("")
        ax_err_id.set_yticklabels([])

        # OD
        pred_od = preds_vis_od[model_name]
        err_od = pred_od - true_vis_od

        ax_pred_od = top_axes_od[i + 1]
        ax_pred_od.imshow(pred_od.T, aspect="auto", origin="lower", cmap=cmap_viridis, vmin=v_min_od, vmax=v_max_od, extent=t_extent)
        ax_pred_od.set_title(plot_name, fontsize=font_size)
        ax_pred_od.set_xticklabels([])
        ax_pred_od.set_yticklabels([])

        ax_err_od = bottom_axes_od[i]
        im_err_od = ax_err_od.imshow(err_od.T, aspect="auto", origin="lower", cmap="bwr", vmin=err_vmin_od, vmax=err_vmax_od, extent=t_extent)
        ax_err_od.set_ylabel("")
        ax_err_od.set_yticklabels([])

    if im_err_id is None:
        im_err_id = ax_5.imshow(np.zeros_like(true_vis_id.T), aspect="auto", origin="lower", cmap="bwr", vmin=-1, vmax=1, extent=t_extent)
    if im_err_od is None:
        im_err_od = ax_12.imshow(np.zeros_like(true_vis_od.T), aspect="auto", origin="lower", cmap="bwr", vmin=-1, vmax=1, extent=t_extent)

    # External colorbars (exact positions)
    cax_top_id = fig.add_axes([0.45, 0.52, 0.008, 0.35])
    cbar_top_id = fig.colorbar(im_true_id, cax=cax_top_id, extend="both")
    cbar_top_id.set_label("$u$", fontsize=font_size)
    cbar_top_id.ax.tick_params(labelsize=font_size)

    cax_top_od = fig.add_axes([0.96, 0.52, 0.008, 0.35])
    cbar_top_od = fig.colorbar(im_true_od, cax=cax_top_od, extend="both")
    cbar_top_od.set_label("$u$", fontsize=font_size)
    cbar_top_od.ax.tick_params(labelsize=font_size)

    cax_bottom_id = fig.add_axes([0.448, 0.11, 0.008, 0.34])
    cbar_bottom_id = fig.colorbar(im_err_id, cax=cax_bottom_id)
    cbar_bottom_id.set_label("$u_{\\mathrm{pred}} - u_{\\mathrm{true}}$", fontsize=font_size)
    cbar_bottom_id.ax.tick_params(labelsize=font_size)

    cax_bottom_od = fig.add_axes([0.958, 0.11, 0.008, 0.34])
    cbar_bottom_od = fig.colorbar(im_err_od, cax=cax_bottom_od)
    cbar_bottom_od.set_label("$u_{\\mathrm{pred}} - u_{\\mathrm{true}}$", fontsize=font_size)
    cbar_bottom_od.ax.tick_params(labelsize=font_size)

    # Panel labels
    if panel_labels is not None and len(panel_labels) == 2:
        fig.text(0.005, 0.98, panel_labels[0], fontsize=font_size + 2, fontweight="bold", ha="left", va="top")
        fig.text(0.515, 0.98, panel_labels[1], fontsize=font_size + 2, fontweight="bold", ha="left", va="top")

    return fig


def plot_id_od_heatmaps_two_models(
    bundle: Dict,
    font_size: int = 10,
    model_order: Optional[List[str]] = None,
    panel_labels: Optional[tuple] = ("A", "B"),
) -> plt.Figure:
    """
    Two-model variant of the old-style heatmap layout:
    - ID (left) and OD (right)
    - Top row per domain: GT + 2 model predictions (3 columns)
    - Bottom row per domain: 2 error maps (under the prediction columns)
    - External colorbars and panel labels matching the original style
    """
    preds_id = bundle["predictions_dict_id"]
    preds_od = bundle["predictions_dict_od"]
    true_solution_id = bundle["true_solution_id"]
    true_solution_od = bundle["true_solution_od"]
    t_coord = bundle.get("t_coord", None)
    x_coord = bundle.get("x_coord", None)

    true_vis_id = true_solution_id[:, :, 0].squeeze()
    true_vis_od = true_solution_od[:, :, 0].squeeze()

    preds_vis_id = {k: v[:, :, 0] for k, v in preds_id.items()}
    preds_vis_od = {k: v[:, :, 0] for k, v in preds_od.items()}

    available_models = [m for m in preds_vis_id.keys() if m in preds_vis_od]
    if model_order is None:
        model_names = available_models
    else:
        preferred = [m for m in model_order if m in available_models]
        remaining = [m for m in available_models if m not in preferred]
        model_names = preferred + remaining

    model_names = model_names[:2]
    if len(model_names) < 2:
        raise ValueError("plot_id_od_heatmaps_two_models requires at least 2 overlapping models.")

    v_min_id = true_vis_id.min() - 0.15 * abs(true_vis_id.max() - true_vis_id.min())
    v_max_id = true_vis_id.max() + 0.15 * abs(true_vis_id.max() - true_vis_id.min())
    v_min_od = true_vis_od.min() - 0.15 * abs(true_vis_od.max() - true_vis_od.min())
    v_max_od = true_vis_od.max() + 0.15 * abs(true_vis_od.max() - true_vis_od.min())

    all_error_maps_id = [(preds_vis_id[m] - true_vis_id) for m in model_names]
    all_error_maps_od = [(preds_vis_od[m] - true_vis_od) for m in model_names]
    max_abs_error_id = max(float(np.abs(e).max()) for e in all_error_maps_id)
    max_abs_error_od = max(float(np.abs(e).max()) for e in all_error_maps_od)
    err_vmin_id, err_vmax_id = -max_abs_error_id, max_abs_error_id
    err_vmin_od, err_vmax_od = -max_abs_error_od, max_abs_error_od

    cmap_viridis = plt.get_cmap("viridis").copy()
    cmap_viridis.set_over("black")
    cmap_viridis.set_under("black")

    fig = plt.figure(figsize=(8.6, 2.8))
    gs_1 = fig.add_gridspec(
        ncols=3, nrows=2,
        bottom=0.1, left=0.05, top=0.9, right=0.43,
        wspace=0.2, hspace=0.2
    )
    gs_2 = fig.add_gridspec(
        ncols=3, nrows=2,
        bottom=0.1, left=0.56, top=0.9, right=0.94,
        wspace=0.2, hspace=0.2
    )

    ax_1 = fig.add_subplot(gs_1[0, 0]); ax_2 = fig.add_subplot(gs_1[0, 1]); ax_3 = fig.add_subplot(gs_1[0, 2])
    ax_4 = fig.add_subplot(gs_1[1, 1]); ax_5 = fig.add_subplot(gs_1[1, 2])

    ax_6 = fig.add_subplot(gs_2[0, 0]); ax_7 = fig.add_subplot(gs_2[0, 1]); ax_8 = fig.add_subplot(gs_2[0, 2])
    ax_9 = fig.add_subplot(gs_2[1, 1]); ax_10 = fig.add_subplot(gs_2[1, 2])

    top_axes_id = [ax_1, ax_2, ax_3]
    bottom_axes_id = [ax_4, ax_5]
    top_axes_od = [ax_6, ax_7, ax_8]
    bottom_axes_od = [ax_9, ax_10]

    t_extent = None
    if t_coord is not None and x_coord is not None:
        t_vals = t_coord.flatten() if np.ndim(t_coord) > 1 else t_coord
        x_vals = x_coord.flatten() if np.ndim(x_coord) > 1 else x_coord
        if len(t_vals) > 1:
            dt = (t_vals[-1] - t_vals[0]) / (len(t_vals) - 1)
            t_extent = [t_vals[0] - dt / 2, t_vals[-1] + dt / 2, x_vals[0], x_vals[-1]]
        else:
            t_extent = [0, 1, x_vals[0], x_vals[-1]]

    im_true_id = ax_1.imshow(true_vis_id.T, aspect="auto", origin="lower", cmap=cmap_viridis, vmin=v_min_id, vmax=v_max_id, extent=t_extent)
    ax_1.set_title("Ground Truth", fontsize=font_size)
    ax_1.set_xlabel("t", fontsize=font_size)
    ax_1.set_ylabel("x", fontsize=font_size)
    ax_1.tick_params(labelsize=font_size)

    im_true_od = ax_6.imshow(true_vis_od.T, aspect="auto", origin="lower", cmap=cmap_viridis, vmin=v_min_od, vmax=v_max_od, extent=t_extent)
    ax_6.set_title("Ground Truth", fontsize=font_size)
    ax_6.set_xlabel("t", fontsize=font_size)
    ax_6.set_ylabel("x", fontsize=font_size)
    ax_6.tick_params(labelsize=font_size)

    all_axes = [ax_1, ax_2, ax_3, ax_4, ax_5, ax_6, ax_7, ax_8, ax_9, ax_10]
    if t_extent is not None:
        t_min, t_max = t_extent[0], t_extent[1]
        t_mid = (t_min + t_max) / 2
        xticks = [t_min, t_mid, t_max]
        xticklabels = [f"{t_min:.2f}", f"{t_mid:.2f}", f"{t_max:.2f}"]

        y_min, y_max = t_extent[2], t_extent[3]
        yticks = [y_min + frac * (y_max - y_min) for frac in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)]
        yticklabels = [f"{y:.1f}" for y in yticks]

        for ax in all_axes:
            ax.set_xticks(xticks)
            ax.set_yticks(yticks)

        ax_1.set_xticklabels(xticklabels); ax_6.set_xticklabels(xticklabels)
        ax_1.set_yticklabels(yticklabels); ax_6.set_yticklabels(yticklabels)

        for ax in [ax_2, ax_3, ax_4, ax_5, ax_7, ax_8, ax_9, ax_10]:
            ax.set_xticklabels([])
            ax.set_yticklabels([])

    def _pretty_name(model_name: str) -> str:
        if model_name == "fno":
            return "FNO"
        if model_name == "cape_fno":
            return "CAPE-FNO"
        if model_name == "late_fusion":
            return "Late Fusion - CNO"
        if model_name == "cno":
            return "CNO"
        return model_name

    im_err_id = None
    im_err_od = None

    for i, model_name in enumerate(model_names):
        pred_id = preds_vis_id[model_name]
        err_id = pred_id - true_vis_id
        ax_pred_id = top_axes_id[i + 1]
        ax_pred_id.imshow(pred_id.T, aspect="auto", origin="lower", cmap=cmap_viridis, vmin=v_min_id, vmax=v_max_id, extent=t_extent)
        ax_pred_id.set_title(_pretty_name(model_name), fontsize=font_size)
        ax_pred_id.set_yticklabels([])

        ax_err_id = bottom_axes_id[i]
        im_err_id = ax_err_id.imshow(err_id.T, aspect="auto", origin="lower", cmap="bwr", vmin=err_vmin_id, vmax=err_vmax_id, extent=t_extent)
        ax_err_id.set_ylabel("")
        ax_err_id.set_yticklabels([])

        pred_od = preds_vis_od[model_name]
        err_od = pred_od - true_vis_od
        ax_pred_od = top_axes_od[i + 1]
        ax_pred_od.imshow(pred_od.T, aspect="auto", origin="lower", cmap=cmap_viridis, vmin=v_min_od, vmax=v_max_od, extent=t_extent)
        ax_pred_od.set_title(_pretty_name(model_name), fontsize=font_size)
        ax_pred_od.set_yticklabels([])

        ax_err_od = bottom_axes_od[i]
        im_err_od = ax_err_od.imshow(err_od.T, aspect="auto", origin="lower", cmap="bwr", vmin=err_vmin_od, vmax=err_vmax_od, extent=t_extent)
        ax_err_od.set_ylabel("")
        ax_err_od.set_yticklabels([])

    if im_err_id is None:
        im_err_id = ax_4.imshow(np.zeros_like(true_vis_id.T), aspect="auto", origin="lower", cmap="bwr", vmin=-1, vmax=1, extent=t_extent)
    if im_err_od is None:
        im_err_od = ax_9.imshow(np.zeros_like(true_vis_od.T), aspect="auto", origin="lower", cmap="bwr", vmin=-1, vmax=1, extent=t_extent)

    cax_top_id = fig.add_axes([0.445, 0.52, 0.008, 0.35])
    cbar_top_id = fig.colorbar(im_true_id, cax=cax_top_id, extend="both")
    cbar_top_id.set_label("$u$", fontsize=font_size)
    cbar_top_id.ax.tick_params(labelsize=font_size)

    cax_top_od = fig.add_axes([0.945, 0.52, 0.008, 0.35])
    cbar_top_od = fig.colorbar(im_true_od, cax=cax_top_od, extend="both")
    cbar_top_od.set_label("$u$", fontsize=font_size)
    cbar_top_od.ax.tick_params(labelsize=font_size)

    cax_bottom_id = fig.add_axes([0.443, 0.11, 0.008, 0.34])
    cbar_bottom_id = fig.colorbar(im_err_id, cax=cax_bottom_id)
    cbar_bottom_id.set_label("$u_{\\mathrm{pred}} - u_{\\mathrm{true}}$", fontsize=font_size)
    cbar_bottom_id.ax.tick_params(labelsize=font_size)

    cax_bottom_od = fig.add_axes([0.943, 0.11, 0.008, 0.34])
    cbar_bottom_od = fig.colorbar(im_err_od, cax=cax_bottom_od)
    cbar_bottom_od.set_label("$u_{\\mathrm{pred}} - u_{\\mathrm{true}}$", fontsize=font_size)
    cbar_bottom_od.ax.tick_params(labelsize=font_size)

    if panel_labels is not None and len(panel_labels) == 2:
        fig.text(0.005, 0.98, panel_labels[0], fontsize=font_size + 2, fontweight="bold", ha="left", va="top")
        fig.text(0.515, 0.98, panel_labels[1], fontsize=font_size + 2, fontweight="bold", ha="left", va="top")

    return fig

def plot_param_vs_rmse_three_models_id_od(
    traj_df: pd.DataFrame,
    id_loader_name: str = "test_id",
    od_loader_name: str = "test_od",
    model_order: Optional[List[str]] = None,
    figsize=(10, 2.5),
    fontsize: int = 8,
    param_key: str = "parameter",
):
    """
    3 subplots (one per model), each panel overlays ID + OD:
    - light points: per-seed/per-trajectory
    - darker points: mean over seeds (after seed-wise averaging per parameter)
    """
    df = traj_df[traj_df["loader"].isin([id_loader_name, od_loader_name])].copy()
    if df.empty:
        raise ValueError(f"No rows found for loaders: {id_loader_name}, {od_loader_name}")

    if model_order is None:
        model_order = sorted(df["model"].unique().tolist())[:3]
    else:
        model_order = model_order[:3]

    # colors: (seed_points_color, mean_points_color)
    color_map = {
        id_loader_name: ("#737a7e", "#5face4", "#27516e"),  # light/dark blue
        od_loader_name: ("#737a7e", "#45792D", "#1a380d"),  # light/dark orange
    }

    # Create figure first, reserve room at the bottom for an external legend axis
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows=1, ncols=3, left=0.06, right=0.98, top=0.95, bottom=0.3, wspace=0.08)

    # create first axis, then share its y-axis for the others so they align
    ax0 = fig.add_subplot(gs[0, 0])
    axes = [ax0]
    for i in range(1, 3):
        axes.append(fig.add_subplot(gs[0, i], sharey=ax0))
    axes = np.array(axes)

    # hide y-axis tick labels on the right panels (keep only leftmost numbers)
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)

    max_mean = 0.0

    for ax, model_name in zip(axes, model_order):
        dm = df[df["model"] == model_name].copy()

        for loader_name in [id_loader_name, od_loader_name]:
            d = dm[dm["loader"] == loader_name].copy()
            if d.empty:
                continue

            seed_c, mean_c, line_c = color_map[loader_name]

            # Seed/trajectory points
            ax.scatter(
                d["param_value"],
                d["traj_rmse"],
                s=20,
                c=seed_c,
                alpha=0.35,
                linewidths=0,
                zorder=1,
                rasterized=True,
                label=f"{loader_name} seeds",
            )

            # Mean over seeds: first mean over trajectories within (seed, param), then across seeds
            seed_param = (
                d.groupby(["seed", "param_value"], as_index=False)
                .agg(rmse_seed_mean=("traj_rmse", "mean"))
            )
            mean_over_seeds = (
                seed_param.groupby("param_value", as_index=False)
                .agg(rmse_mean=("rmse_seed_mean", "mean"))
                .sort_values("param_value")
            )

            # update global max mean for y-limit later
            if not mean_over_seeds.empty:
                max_mean = max(max_mean, float(mean_over_seeds["rmse_mean"].max()))

            ax.scatter(
                mean_over_seeds["param_value"],
                mean_over_seeds["rmse_mean"],
                s=20,
                c=mean_c,
                alpha=0.95,
                zorder=2,
                rasterized=True,
                label=f"{loader_name} mean",
                edgecolors=line_c,     # outline color
                linewidths=0.7,     # outline width
                marker="o",
            )

        if model_name == "fno":
            pretty_name = "FNO"
        elif model_name == "cape_fno":
            pretty_name = "CAPE-FNO"
        elif model_name == "late_fusion":
            pretty_name = "Late Fusion"
        else:
            pretty_name = model_name

        # dashed horizontal line at rmse=0 to indicate perfect predictions
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7, zorder=0)
        ax.set_title(pretty_name, fontsize=fontsize)
        ax.set_xlabel(param_key, fontsize=fontsize)
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=fontsize)

    # set y label and y-limits (lower bound slightly below 0, upper bound = max mean error)
    axes[0].set_ylabel("RMSE", fontsize=fontsize)
    upper = float(max_mean)
    # ensure some minimal upper bound if nothing plotted (shouldn't happen due to earlier checks)
    if upper <= 0:
        upper = 1.0
    for ax in axes:
        #ax.set_ylim(-0.1,1)
        ax.set_ylim(0.0 - 0.1 * upper, upper)

    # Build legend handles for mean markers only (ID/OD)
    blue_face, blue_edge = color_map[id_loader_name][1], color_map[id_loader_name][2]
    green_face, green_edge = color_map[od_loader_name][1], color_map[od_loader_name][2]
    handles = [
        mpl.lines.Line2D([0], [0], marker="o", color="w", markerfacecolor=blue_face, markeredgecolor=blue_edge, markersize=6, linestyle=""),
        mpl.lines.Line2D([0], [0], marker="o", color="w", markerfacecolor=green_face, markeredgecolor=green_edge, markersize=6, linestyle=""),
    ]
    labels = ["In-domain", "Out-domain"]

    # Place a compact legend in a dedicated axis below the grid (a bit outside the main grid)
    legend_ax = fig.add_axes([0.42, 0.03, 0.2, 0.12])
    legend_ax.axis("off")
    legend_ax.legend(handles, labels, loc="center", ncol=2, fontsize=fontsize, framealpha=0.2)

    # Return figure
    return fig

def plot_id_od_heatmaps_2d_lasttime(
    bundle: Dict,
    max_models: int = 3,
    font_size: int = 10,
    model_order: Optional[List[str]] = None,
    panel_labels: Optional[tuple] = ("A", "B"),
) -> plt.Figure:
    """
    Plot 2D predictions at the final time step with layout matching 1D case.
    
    Layout:
    - Two side-by-side grids (ID on left, OD on right)
    - Top row: GT + up to 3 model predictions
    - Bottom row: 3 error maps
    - Each cell is a 2D heatmap (x, y axes)
    
    Args:
        bundle: Dict containing predictions_dict_id/od, true_solution_id/od, and optionally x_coord, y_coord, t_coord
        max_models: Maximum number of models to display
        font_size: Font size for labels
        model_order: Optional list specifying order to display models
    
    Returns:
        plt.Figure: The generated figure
    """
    # Extract data from bundle
    preds_id = bundle["predictions_dict_id"]
    preds_od = bundle["predictions_dict_od"]
    true_solution_id = bundle["true_solution_id"]
    true_solution_od = bundle["true_solution_od"]
    x_coord = bundle.get("x_coord", None)
    y_coord = bundle.get("y_coord", None)

    # Extract first variable and last time step for 2D problems
    # Shape: [T, X, Y, Variables] -> [X, Y] at last time step
    true_vis_id = true_solution_id[-1, :, :, 0].squeeze()
    true_vis_od = true_solution_od[-1, :, :, 0].squeeze()

    preds_vis_id = {k: v[-1, :, :, 0].squeeze() for k, v in preds_id.items()}
    preds_vis_od = {k: v[-1, :, :, 0].squeeze() for k, v in preds_od.items()}

    # Enforce plotting order
    if model_order is None:
        model_names = list(preds_vis_id.keys())
    else:
        preferred = [m for m in model_order if m in preds_vis_id and m in preds_vis_od]
        remaining = [m for m in preds_vis_id.keys() if m not in preferred and m in preds_vis_od]
        model_names = preferred + remaining

    n_show = min(max_models, 3, len(model_names), len(preds_vis_id), len(preds_vis_od))

    # Compute color scales for visualization
    # ID scales
    v_min_id = true_vis_id.min() - 0.15 * abs(true_vis_id.max() - true_vis_id.min())
    v_max_id = true_vis_id.max() + 0.15 * abs(true_vis_id.max() - true_vis_id.min())
    
    # OD scales
    v_min_od = true_vis_od.min() - 0.15 * abs(true_vis_od.max() - true_vis_od.min())
    v_max_od = true_vis_od.max() + 0.15 * abs(true_vis_od.max() - true_vis_od.min())

    # Error scales (separate for ID and OD)
    all_error_maps_id = [(preds_vis_id[m] - true_vis_id) for m in model_names[:n_show]]
    all_error_maps_od = [(preds_vis_od[m] - true_vis_od) for m in model_names[:n_show]]

    max_abs_error_id = max(float(np.abs(e).max()) for e in all_error_maps_id) if all_error_maps_id else 1.0
    max_abs_error_od = max(float(np.abs(e).max()) for e in all_error_maps_od) if all_error_maps_od else 1.0
    err_vmin_id, err_vmax_id = -max_abs_error_id, max_abs_error_id
    err_vmin_od, err_vmax_od = -max_abs_error_od, max_abs_error_od

    # Colormap with out-of-bounds in black
    cmap_viridis = plt.get_cmap("viridis").copy()
    cmap_viridis.set_over("black")
    cmap_viridis.set_under("black")

    # Figure layout (matching 1D structure)
    fig = plt.figure(figsize=(10, 2.8))
    gs_1 = fig.add_gridspec(
        ncols=4, nrows=2,
        bottom=0.1, left=0.05, top=0.9, right=0.43,
        wspace=0.2, hspace=0.2
    )
    gs_2 = fig.add_gridspec(
        ncols=4, nrows=2,
        bottom=0.1, left=0.57, top=0.9, right=0.95,
        wspace=0.2, hspace=0.2
    )

    # Create axes for top row (GT + model predictions)
    ax_1 = fig.add_subplot(gs_1[0, 0]); ax_2 = fig.add_subplot(gs_1[0, 1]); ax_3 = fig.add_subplot(gs_1[0, 2]); ax_4 = fig.add_subplot(gs_1[0, 3])
    # Create axes for bottom row (error maps)
    ax_5 = fig.add_subplot(gs_1[1, 1]); ax_6 = fig.add_subplot(gs_1[1, 2]); ax_7 = fig.add_subplot(gs_1[1, 3])

    ax_8 = fig.add_subplot(gs_2[0, 0]); ax_9 = fig.add_subplot(gs_2[0, 1]); ax_10 = fig.add_subplot(gs_2[0, 2]); ax_11 = fig.add_subplot(gs_2[0, 3])
    ax_12 = fig.add_subplot(gs_2[1, 1]); ax_13 = fig.add_subplot(gs_2[1, 2]); ax_14 = fig.add_subplot(gs_2[1, 3])

    top_axes_id = [ax_1, ax_2, ax_3, ax_4]
    bottom_axes_id = [ax_5, ax_6, ax_7]
    top_axes_od = [ax_8, ax_9, ax_10, ax_11]
    bottom_axes_od = [ax_12, ax_13, ax_14]

    # Extent from coordinates (for spatial domain)
    # Default to [0, 1] for both axes if coordinates not provided
    if x_coord is not None and y_coord is not None:
        x_vals = x_coord.flatten() if np.ndim(x_coord) > 0 else np.array([x_coord])
        y_vals = y_coord.flatten() if np.ndim(y_coord) > 0 else np.array([y_coord])
        
        if len(x_vals) > 1:
            x_min, x_max = x_vals[0], x_vals[-1]
        else:
            x_min, x_max = 0.0, 1.0
            
        if len(y_vals) > 1:
            y_min, y_max = y_vals[0], y_vals[-1]
        else:
            y_min, y_max = 0.0, 1.0
    else:
        # Default to [0, 1] for both axes
        x_min, x_max = 0.0, 1.0
        y_min, y_max = 0.0, 1.0
    
    # extent format for imshow: [left, right, bottom, top]
    extent = [x_min, x_max, y_min, y_max]

    # Ground truth
    im_true_id = ax_1.imshow(true_vis_id.T, aspect="auto", origin="lower", cmap=cmap_viridis, vmin=v_min_id, vmax=v_max_id, extent=extent)
    ax_1.set_title("Ground Truth", fontsize=font_size)
    ax_1.set_xlabel("x", fontsize=font_size)
    ax_1.set_ylabel("y", fontsize=font_size)
    ax_1.tick_params(labelsize=font_size)

    im_true_od = ax_8.imshow(true_vis_od.T, aspect="auto", origin="lower", cmap=cmap_viridis, vmin=v_min_od, vmax=v_max_od, extent=extent)
    ax_8.set_title("Ground Truth", fontsize=font_size)
    ax_8.set_xlabel("x", fontsize=font_size)
    ax_8.set_ylabel("y", fontsize=font_size)
    ax_8.tick_params(labelsize=font_size)

    # Helper to format model names
    def _pretty_name(model_name: str) -> str:
        if model_name == "fno":
            return "FNO"
        if model_name == "cape_fno":
            return "CAPE-FNO"
        if model_name == "late_fusion":
            return "Late Fusion"
        return model_name

    # Plot models and error maps
    im_err_id = None
    im_err_od = None

    for i, model_name in enumerate(model_names[:n_show]):
        plot_name = _pretty_name(model_name)

        # ID predictions
        pred_id = preds_vis_id[model_name]
        err_id = pred_id - true_vis_id

        ax_pred_id = top_axes_id[i + 1]
        ax_pred_id.imshow(pred_id.T, aspect="auto", origin="lower", cmap=cmap_viridis, vmin=v_min_id, vmax=v_max_id, extent=extent)
        ax_pred_id.set_title(plot_name, fontsize=font_size)
        ax_pred_id.set_xlabel("")
        ax_pred_id.set_ylabel("")
        ax_pred_id.set_yticklabels([])
        ax_pred_id.set_xticklabels([])
        ax_pred_id.tick_params(labelsize=font_size)

        ax_err_id = bottom_axes_id[i]
        im_err_id = ax_err_id.imshow(err_id.T, aspect="auto", origin="lower", cmap="bwr", vmin=err_vmin_id, vmax=err_vmax_id, extent=extent)
        ax_err_id.set_xlabel("")
        ax_err_id.set_ylabel("")
        ax_err_id.set_yticklabels([])
        ax_err_id.set_xticklabels([])
        ax_err_id.tick_params(labelsize=font_size)

        # OD predictions
        pred_od = preds_vis_od[model_name]
        err_od = pred_od - true_vis_od

        ax_pred_od = top_axes_od[i + 1]
        ax_pred_od.imshow(pred_od.T, aspect="auto", origin="lower", cmap=cmap_viridis, vmin=v_min_od, vmax=v_max_od, extent=extent)
        ax_pred_od.set_title(plot_name, fontsize=font_size)
        ax_pred_od.set_xlabel("")
        ax_pred_od.set_ylabel("")
        ax_pred_od.set_yticklabels([])
        ax_pred_od.set_xticklabels([])
        ax_pred_od.tick_params(labelsize=font_size)

        ax_err_od = bottom_axes_od[i]
        im_err_od = ax_err_od.imshow(err_od.T, aspect="auto", origin="lower", cmap="bwr", vmin=err_vmin_od, vmax=err_vmax_od, extent=extent)
        ax_err_od.set_xlabel("")
        ax_err_od.set_ylabel("")
        ax_err_od.set_yticklabels([])
        ax_err_od.set_xticklabels([])
        ax_err_od.tick_params(labelsize=font_size)

    # Create dummy images for missing colorbars if needed
    if im_err_id is None:
        im_err_id = ax_5.imshow(np.zeros_like(true_vis_id.T), aspect="auto", origin="lower", cmap="bwr", vmin=-1, vmax=1, extent=extent)
    if im_err_od is None:
        im_err_od = ax_12.imshow(np.zeros_like(true_vis_od.T), aspect="auto", origin="lower", cmap="bwr", vmin=-1, vmax=1, extent=extent)

    # External colorbars (exact positions)
    cax_top_id = fig.add_axes([0.44, 0.52, 0.008, 0.35])
    cbar_top_id = fig.colorbar(im_true_id, cax=cax_top_id, extend="both")
    cbar_top_id.set_label("$u$", fontsize=font_size)
    cbar_top_id.ax.tick_params(labelsize=font_size)

    cax_top_od = fig.add_axes([0.96, 0.52, 0.008, 0.35])
    cbar_top_od = fig.colorbar(im_true_od, cax=cax_top_od, extend="both")
    cbar_top_od.set_label("$u$", fontsize=font_size)
    cbar_top_od.ax.tick_params(labelsize=font_size)

    cax_bottom_id = fig.add_axes([0.438, 0.11, 0.008, 0.34])
    cbar_bottom_id = fig.colorbar(im_err_id, cax=cax_bottom_id)
    cbar_bottom_id.set_label("$u_{\\mathrm{pred}} - u_{\\mathrm{true}}$", fontsize=font_size)
    cbar_bottom_id.ax.tick_params(labelsize=font_size)

    cax_bottom_od = fig.add_axes([0.958, 0.11, 0.008, 0.34])
    cbar_bottom_od = fig.colorbar(im_err_od, cax=cax_bottom_od)
    cbar_bottom_od.set_label("$u_{\\mathrm{pred}} - u_{\\mathrm{true}}$", fontsize=font_size)
    cbar_bottom_od.ax.tick_params(labelsize=font_size)

    # Panel labels
    if panel_labels is not None and len(panel_labels) == 2:
        fig.text(0.005, 0.98, panel_labels[0], fontsize=font_size + 2, fontweight="bold", ha="left", va="top")
        fig.text(0.515, 0.98, panel_labels[1], fontsize=font_size + 2, fontweight="bold", ha="left", va="top")

    return fig


def plot_stacked_model_heatmaps(
    bundle: Dict,
    model_order: Optional[List[str]] = None,
    font_size: int = 9,
    row_height: float = 2.1,
) -> plt.Figure:
    """
    Create one stacked figure with one row per model and six panels per row:
    [ID GT, ID Pred, ID Error, OD GT, OD Pred, OD Error].

    Row-group labels use A1/A2, B1/B2, C1/C2, ... where:
    - X1 marks the in-domain group of row X
    - X2 marks the out-domain group of row X
    """
    preds_id = bundle["predictions_dict_id"]
    preds_od = bundle["predictions_dict_od"]
    true_solution_id = bundle["true_solution_id"]
    true_solution_od = bundle["true_solution_od"]

    available_models = [m for m in preds_id.keys() if m in preds_od]
    if not available_models:
        raise ValueError("No overlapping model predictions found in ID/OD bundle.")

    if model_order is None:
        model_names = available_models
    else:
        preferred = [m for m in model_order if m in available_models]
        remaining = [m for m in available_models if m not in preferred]
        model_names = preferred + remaining

    n_models = len(model_names)
    fig, axes = plt.subplots(n_models, 6, figsize=(12.0, row_height * n_models), squeeze=False)

    def _pretty_name(model_name: str) -> str:
        if model_name == "fno":
            return "FNO"
        if model_name == "cape_fno":
            return "CAPE-FNO"
        if model_name == "late_fusion":
            return "Late Fusion"
        return model_name

    def _extract_vis(arr: np.ndarray) -> np.ndarray:
        # 1D rollout shape: [T, X, C] -> [T, X]
        if arr.ndim == 3:
            return arr[:, :, 0].squeeze()
        # 2D rollout shape: [T, X, Y, C] -> [X, Y] at final time
        if arr.ndim >= 4:
            return arr[-1, :, :, 0].squeeze()
        raise ValueError(f"Unsupported trajectory shape: {arr.shape}")

    true_id = _extract_vis(true_solution_id)
    true_od = _extract_vis(true_solution_od)
    is_1d = true_solution_id.ndim == 3

    cmap_state = plt.get_cmap("viridis").copy()
    cmap_state.set_over("black")
    cmap_state.set_under("black")

    for row_idx, model_name in enumerate(model_names):
        pred_id = _extract_vis(preds_id[model_name])
        pred_od = _extract_vis(preds_od[model_name])
        err_id = pred_id - true_id
        err_od = pred_od - true_od

        vmin_id = min(float(true_id.min()), float(pred_id.min()))
        vmax_id = max(float(true_id.max()), float(pred_id.max()))
        vmin_od = min(float(true_od.min()), float(pred_od.min()))
        vmax_od = max(float(true_od.max()), float(pred_od.max()))

        err_abs_id = max(float(np.abs(err_id).max()), 1e-12)
        err_abs_od = max(float(np.abs(err_od).max()), 1e-12)

        row_axes = axes[row_idx]
        ims = []
        ims.append(row_axes[0].imshow(true_id.T, aspect="auto", origin="lower", cmap=cmap_state, vmin=vmin_id, vmax=vmax_id))
        ims.append(row_axes[1].imshow(pred_id.T, aspect="auto", origin="lower", cmap=cmap_state, vmin=vmin_id, vmax=vmax_id))
        ims.append(row_axes[2].imshow(err_id.T, aspect="auto", origin="lower", cmap="bwr", vmin=-err_abs_id, vmax=err_abs_id))
        ims.append(row_axes[3].imshow(true_od.T, aspect="auto", origin="lower", cmap=cmap_state, vmin=vmin_od, vmax=vmax_od))
        ims.append(row_axes[4].imshow(pred_od.T, aspect="auto", origin="lower", cmap=cmap_state, vmin=vmin_od, vmax=vmax_od))
        ims.append(row_axes[5].imshow(err_od.T, aspect="auto", origin="lower", cmap="bwr", vmin=-err_abs_od, vmax=err_abs_od))

        if row_idx == 0:
            row_axes[0].set_title("ID GT", fontsize=font_size)
            row_axes[1].set_title("ID Pred", fontsize=font_size)
            row_axes[2].set_title("ID Error", fontsize=font_size)
            row_axes[3].set_title("OD GT", fontsize=font_size)
            row_axes[4].set_title("OD Pred", fontsize=font_size)
            row_axes[5].set_title("OD Error", fontsize=font_size)

        for ax in row_axes:
            ax.tick_params(labelsize=font_size - 1)

        for ax in [row_axes[1], row_axes[2], row_axes[4], row_axes[5]]:
            ax.set_yticklabels([])

        if row_idx < n_models - 1:
            for ax in row_axes:
                ax.set_xticklabels([])

        if is_1d:
            row_axes[0].set_ylabel("x", fontsize=font_size)
            row_axes[0].set_xlabel("t" if row_idx == n_models - 1 else "", fontsize=font_size)
            row_axes[3].set_ylabel("x", fontsize=font_size)
            row_axes[3].set_xlabel("t" if row_idx == n_models - 1 else "", fontsize=font_size)
        else:
            row_axes[0].set_ylabel("y", fontsize=font_size)
            row_axes[0].set_xlabel("x" if row_idx == n_models - 1 else "", fontsize=font_size)
            row_axes[3].set_ylabel("y", fontsize=font_size)
            row_axes[3].set_xlabel("x" if row_idx == n_models - 1 else "", fontsize=font_size)

        for ax in [row_axes[1], row_axes[2], row_axes[4], row_axes[5]]:
            ax.set_xlabel("")
            ax.set_ylabel("")

        letter = chr(ord("A") + row_idx)
        row_axes[0].text(-0.25, 1.12, f"{letter}1", transform=row_axes[0].transAxes, fontsize=font_size + 2, fontweight="bold")
        row_axes[3].text(-0.25, 1.12, f"{letter}2", transform=row_axes[3].transAxes, fontsize=font_size + 2, fontweight="bold")
        row_axes[0].text(0.5, 1.12, _pretty_name(model_name), transform=row_axes[0].transAxes, fontsize=font_size, ha="center")

        cbar_state = fig.colorbar(ims[1], ax=[row_axes[0], row_axes[1]], fraction=0.03, pad=0.01)
        cbar_state.ax.tick_params(labelsize=font_size - 1)
        cbar_state.set_label("u", fontsize=font_size)

        cbar_id_err = fig.colorbar(ims[2], ax=[row_axes[2]], fraction=0.05, pad=0.01)
        cbar_id_err.ax.tick_params(labelsize=font_size - 1)

        cbar_od_state = fig.colorbar(ims[4], ax=[row_axes[3], row_axes[4]], fraction=0.03, pad=0.01)
        cbar_od_state.ax.tick_params(labelsize=font_size - 1)
        cbar_od_state.set_label("u", fontsize=font_size)

        cbar_od_err = fig.colorbar(ims[5], ax=[row_axes[5]], fraction=0.05, pad=0.01)
        cbar_od_err.ax.tick_params(labelsize=font_size - 1)

    fig.tight_layout()
    return fig

def plot_old_style_equation_heatmap(
    bundle: Dict,
    equation: str,
    panel_labels: Tuple[str, str],
    model_order: Optional[List[str]] = None,
    font_size: int = 9,
    max_models: int = 3,
) -> plt.Figure:
    """Render one equation with the original ID/OD heatmap layout and custom panel labels."""
    equation_key = equation.strip().lower()

    if equation_key == "reactiondiffusion2d":
        return plot_id_od_heatmaps_2d_lasttime(
            bundle=bundle,
            max_models=max_models,
            font_size=font_size,
            model_order=model_order,
            panel_labels=panel_labels,
        )

    return plot_id_od_heatmaps(
        bundle=bundle,
        max_models=max_models,
        font_size=font_size,
        model_order=model_order,
        panel_labels=panel_labels,
    )

