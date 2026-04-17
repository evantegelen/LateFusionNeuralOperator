from src.train import build_configs
from src.data.equation_datamodule import EquationModule
from src.models.base_lightning_model import PDELightningModule
from src.utils.evaluation.metrics import compute_all_metrics, rmse
from typing import Dict, List, Iterable, Tuple, Union
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import pytorch_lightning as pl
import torch


def load_model_and_datamodule(run_cfg: dict, use_best: bool = True):
    """Returns loaded model, datamodule, and trainer for a given run configuration."""

    equation = run_cfg["equation"]
    model_name = run_cfg["model"]
    training = run_cfg.get("training", {})

    resolved_model_name, model_config, datamodule_config = build_configs(equation, model_name)
    model_config.update(run_cfg.get("model_overrides", {}))

    ckpt_path = resolve_ckpt_path(run_cfg, use_best=use_best)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    dm = EquationModule(**datamodule_config)
    dm.setup(stage=None)

    model = PDELightningModule.load_from_checkpoint(
        str(ckpt_path),
        model_name=resolved_model_name,
        model_config=model_config,
        learning_rate=float(training.get("learning_rate", 1e-3)),
        log_hyperparameters=False,
        strict=False,
    )

    trainer = pl.Trainer(
        accelerator=training.get("accelerator", "auto"),
        devices=training.get("devices", 1),
        logger=False,
        enable_checkpointing=False,
    )
    return model, dm, trainer

def resolve_ckpt_path(run_cfg: dict, use_best: bool = True) -> Path:
    """Returns the path to the checkpoint to load for a given run configuration."""
    run_dir = Path(run_cfg.get("output_dir", "outputs")) / run_cfg["experiment_name"] / run_cfg["name"]
    ckpt_dir = run_dir / "checkpoints"
    if use_best:
        best_ckpts = list(ckpt_dir.glob("epoch=*.ckpt"))
        if best_ckpts:
            return sorted(best_ckpts)[-1]
    return ckpt_dir / "last.ckpt"

def evaluate_run_validation(run_cfg: dict) -> Dict:
    model, dm, trainer = load_model_and_datamodule(run_cfg)

    val_out = trainer.validate(model=model, dataloaders=dm.val_dataloader(), verbose=False)
    val_error = val_out[0].get("val_loss", None)
    return {
        "run_name": run_cfg["name"],
        "model": run_cfg["model"],
        "val_error": val_error,
        "seed": run_cfg.get("seed", None),
    }

def combination_key(run_name: str) -> str:
    return re.sub(r"_seed\d+$", "", run_name)

def get_eval_loaders(dm, domains: Iterable[str]) -> List[Tuple[str, str, torch.utils.data.DataLoader]]:
    """Return list of (domain, loader_name, dataloader)."""
    out = []
    domains = set(domains)

    if "id" in domains:
        out.append(("id", "test_id", dm.test_dataloader()))
    if "od" in domains:
        out.append(("od", "test_od", dm.get_test_od_dataloaders()))
    if not out:
        raise ValueError("No valid domains requested. Use 'id', 'od', or both.")
    return out

def evaluate_one_loader(model, loader, device: torch.device) -> Dict[str, float]:
    """
    Evaluate only the first batch of a dataloader.
    Computes RMSE after filtering invalid trajectories.
    """
    with torch.no_grad():
        try:
            batch = next(iter(loader))
        except StopIteration:
            raise ValueError("Loader is empty.")

        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

        true_states = batch["state"][:, 1:, ...]
        pred_states = model.roll_out_predictions(batch)
        
        rmse_calc = rmse(pred_states,true_states).item()
        
    return {"test_rmse": rmse_calc}

def evaluate_run_test(
    run_cfg: dict,
    domains: Union[str, Iterable[str]] = ("id", "od"),
) -> Union[Dict, List[Dict]]:
    """
    Evaluate one run on one or multiple domains/loaders.

    Args:
        run_cfg: run configuration
        domains: "id", "od", or iterable like ("id", "od")

    Returns:
        - Dict if exactly one loader is evaluated (backward-compatible)
        - List[Dict] if multiple loaders are evaluated
    """
    if isinstance(domains, str):
        domains = [domains]

    model, dm, trainer = load_model_and_datamodule(run_cfg)
    dm.batch_size_test = 500

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    rows: List[Dict] = []
    for domain, loader_name, loader in get_eval_loaders(dm, domains):
        stats = evaluate_one_loader(model, loader, device)
        rows.append({
            "run_name": run_cfg["name"],
            "model": run_cfg["model"],
            "seed": run_cfg.get("seed", None),
            "domain": domain,
            "loader": loader_name,
            **stats,
        })

    return rows[0] if len(rows) == 1 else rows

def _to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

def collect_best_run_predictions(
    best_runs_by_model: Dict[str, dict],
    plot_id_idx: int = 2,
    plot_od_idx: int = 3,
    batch_size_test: int = 10,
    device: Optional[torch.device] = None,
) -> Dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    predictions_id, predictions_od = {}, {}
    true_solution_id, true_solution_od = None, None
    t_coord, x_coord, param_id, param_od = None, None, None, None

    for model_name, run_cfg in best_runs_by_model.items():
        model, dm, _ = load_model_and_datamodule(run_cfg)
        dm.batch_size_test = batch_size_test

        id_loader = dm.test_dataloader()
        od_loader = dm.get_test_od_dataloaders()
        model = model.to(device).eval()

        with torch.no_grad():
            # ID
            batch_id = _to_device(next(iter(id_loader)), device)
            pred_id = model.roll_out_predictions(batch_id)
            true_id = batch_id["state"][:, 1:, ...]

            predictions_id[model_name] = pred_id[plot_id_idx].detach().cpu().numpy()
            if true_solution_id is None:
                true_solution_id = true_id[plot_id_idx].detach().cpu().numpy()
                if "t" in batch_id:
                    t_coord = batch_id["t"].detach().cpu().numpy()
                if "x" in batch_id:
                    x_coord = batch_id["x"].detach().cpu().numpy()
                if "parameter" in batch_id:
                    param_id = batch_id["parameter"][plot_id_idx].detach().cpu().numpy()

            # OD
            batch_od = _to_device(next(iter(od_loader)), device)
            pred_od = model.roll_out_predictions(batch_od)
            true_od = batch_od["state"][:, 1:, ...]

            predictions_od[model_name] = pred_od[plot_od_idx].detach().cpu().numpy()
            if true_solution_od is None:
                true_solution_od = true_od[plot_od_idx].detach().cpu().numpy()
                if "parameter" in batch_od:
                    param_od = batch_od["parameter"][plot_od_idx].detach().cpu().numpy()

    return {
        "predictions_dict_id": predictions_id,
        "predictions_dict_od": predictions_od,
        "true_solution_id": true_solution_id,
        "true_solution_od": true_solution_od,
        "t_coord": t_coord,
        "x_coord": x_coord,
        "param_id": param_id,
        "param_od": param_od,
    }

def evaluate_one_loader_all_metrics(
    model,
    loader,
    device: torch.device,
) -> Dict[str, float]:
    all_preds = []
    all_trues = []

    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            true_full = batch["state"]                      # [B, T, ...]
            pred_states = model.roll_out_predictions(batch) # [B, T-1, ...]

            all_preds.append(pred_states)
            all_trues.append(true_full[:, 1:, ...])

    pred_all = torch.cat(all_preds, dim=0)
    true_all = torch.cat(all_trues, dim=0)

    metrics = compute_all_metrics(pred_all, true_all)
    return {k: float(v.detach().cpu().item() if torch.is_tensor(v) else v) for k, v in metrics.items()}

def evaluate_run_test_all_metrics(
    run_cfg: dict,
    domains: Union[str, Iterable[str]] = ("id", "od"),
    use_best: bool = True,
) -> Union[Dict, List[Dict]]:
    if isinstance(domains, str):
        domains = [domains]

    model, dm, _ = load_model_and_datamodule(run_cfg, use_best=use_best)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    rows: List[Dict] = []
    for domain, loader_name, loader in get_eval_loaders(dm, domains):
        metric_vals = evaluate_one_loader_all_metrics(model, loader, device)
        rows.append(
            {
                "run_name": run_cfg["name"],
                "model": run_cfg["model"],
                "seed": run_cfg.get("seed"),
                "domain": domain,
                "loader": loader_name,
                **metric_vals,
            }
        )

    return rows[0] if len(rows) == 1 else rows

def evaluate_run_test_traj_rmse(
    run_cfg: dict,
    domains: Union[str, Iterable[str]] = ("id", "od"),
    use_best: bool = True,
    od_loader_names: Optional[Iterable[str]] = None,
    param_key: str = "parameter",
) -> List[Dict]:
    """
    Per-trajectory RMSE for all test batches of each selected dataloader.
    Returns one row per trajectory.
    """
    if isinstance(domains, str):
        domains = [domains]
    od_loader_names = set(od_loader_names or [])

    model, dm, _ = load_model_and_datamodule(run_cfg, use_best=use_best)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    rows: List[Dict] = []
    loaders = get_eval_loaders(dm, domains)

    with torch.no_grad():
        for domain, loader_name, loader in loaders:
            if domain == "od" and od_loader_names and loader_name not in od_loader_names:
                continue

            for batch_idx, batch in enumerate(loader):
                batch = _to_device(batch, device)

                pred = model.roll_out_predictions(batch)      # [B, T-1, ...]
                true = batch["state"][:, 1:, ...]             # [B, T-1, ...]

                # RMSE per trajectory
                reduce_dims = tuple(range(1, pred.ndim))
                traj_rmse = torch.sqrt(torch.mean((pred - true) ** 2, dim=reduce_dims))  # [B]

                # Parameter per trajectory
                if param_key not in batch:
                    raise KeyError(f"Batch missing '{param_key}'. Available keys: {list(batch.keys())}")

                p = batch[param_key]
                if p.ndim > 1:
                    p = p.reshape(p.shape[0], -1)[:, 0]
                p = p.detach().cpu().numpy()

                rmse_np = traj_rmse.detach().cpu().numpy()

                for i in range(len(rmse_np)):
                    rows.append(
                        {
                            "run_name": run_cfg["name"],
                            "model": run_cfg["model"],
                            "seed": run_cfg.get("seed", None),
                            "domain": domain,
                            "loader": loader_name,
                            "batch_idx": batch_idx,
                            "traj_idx_in_batch": i,
                            "param_value": float(p[i]),
                            "traj_rmse": float(rmse_np[i]),
                        }
                    )

    return rows

def collect_traj_rmse_for_runs(
    run_cfgs: Iterable[dict],
    domains: Union[str, Iterable[str]] = ("id", "od"),
    use_best: bool = True,
    od_loader_names: Optional[Iterable[str]] = None,
    param_key: str = "parameter",
) -> pd.DataFrame:
    all_rows: List[Dict] = []
    for run_cfg in run_cfgs:
        all_rows.extend(
            evaluate_run_test_traj_rmse(
                run_cfg=run_cfg,
                domains=domains,
                use_best=use_best,
                od_loader_names=od_loader_names,
                param_key=param_key,
            )
        )
    return pd.DataFrame(all_rows)

def summary_metrics_to_latex_table(summary: pd.DataFrame, model_order: Optional[List[str]] = None, fontsize: int = 8) -> str:
    """
    Convert metrics summary table to LaTeX format.
    
    Input: summary table from groupby aggregation with columns like "metric_mean" and "metric_std"
    Output: LaTeX table with metrics as rows and model-domain combinations as columns.
    
    Args:
        summary: DataFrame with model and loader columns, and metrics with _mean/_std suffix
        model_order: Optional list specifying order of models (default: ["fno", "cape_fno", "late_fusion"])
        fontsize: Font size for LaTeX table
    
    Returns:
        str: LaTeX table code
    """
    if model_order is None:
        model_order = ["fno", "cape_fno", "late_fusion"]
    
    model_display = {
        "fno": "FNO",
        "cape_fno": "CAPE-FNO",
        "late_fusion": "Late Fusion",
    }
    
    # Helper function to format mean ± std
    def fmt_metric(mean: float, std: float) -> str:
        """Format as mean ± std with 3 significant figures."""
        s_mean = f"{mean:.2e}"
        s_std = f"{std:.2e}"
        s_mean = re.sub(r"e([+-])0*(\d+)", r"e\1\2", s_mean)
        s_std = re.sub(r"e([+-])0*(\d+)", r"e\1\2", s_std)
        return f"{s_mean} $\\pm$ {s_std}"
    
    # Make a working copy
    df = summary.copy()
    
    # Ensure model and loader columns exist
    if "model" not in df.columns or "loader" not in df.columns:
        raise ValueError("Summary must have 'model' and 'loader' columns")
    
    # Normalize model names and extract loader type
    df["model_norm"] = df["model"].str.lower().str.strip().str.replace("-", "_", regex=False)
    df["loader_type"] = df["loader"].str.lower().apply(lambda x: "id" if "id" in x else "od")
    
    # Extract metric names (all columns ending in _mean or _std)
    metric_names = sorted(set(
        c.replace("_mean", "").replace("_std", "") 
        for c in df.columns 
        if c.endswith(("_mean", "_std"))
    ))
    
    print(f"DEBUG: Found {len(metric_names)} metrics: {metric_names}")
    print(f"DEBUG: Models in data: {df['model_norm'].unique().tolist()}")
    print(f"DEBUG: Loaders in data: {df['loader_type'].unique().tolist()}")
    
    # Build result data
    result_data = []
    
    for metric in metric_names:
        row_data = {"Metric": metric}
        found_any = False
        
        # First all ID models
        for model_name in model_order:
            col_name = f"{model_display[model_name]} ID"
            
            mask = (df["model_norm"] == model_name) & (df["loader_type"] == "id")
            matching_rows = df[mask]
            
            mean_col = f"{metric}_mean"
            std_col = f"{metric}_std"
            
            if not matching_rows.empty and mean_col in df.columns and std_col in df.columns:
                try:
                    mean_val = matching_rows[mean_col].iloc[0]
                    std_val = matching_rows[std_col].iloc[0]
                    row_data[col_name] = fmt_metric(mean_val, std_val)
                    found_any = True
                except Exception as e:
                    row_data[col_name] = "—"
            else:
                row_data[col_name] = "—"
        
        # Then all OD models
        for model_name in model_order:
            col_name = f"{model_display[model_name]} OD"
            
            mask = (df["model_norm"] == model_name) & (df["loader_type"] == "od")
            matching_rows = df[mask]
            
            mean_col = f"{metric}_mean"
            std_col = f"{metric}_std"
            
            if not matching_rows.empty and mean_col in df.columns and std_col in df.columns:
                try:
                    mean_val = matching_rows[mean_col].iloc[0]
                    std_val = matching_rows[std_col].iloc[0]
                    row_data[col_name] = fmt_metric(mean_val, std_val)
                    found_any = True
                except Exception as e:
                    row_data[col_name] = "—"
            else:
                row_data[col_name] = "—"
        
        if found_any:
            result_data.append(row_data)
    
    if not result_data:
        print("WARNING: No data found! Printing summary columns and sample:")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Data:\n{df.head()}")
        return "ERROR: No matching data found"
    
    result_df = pd.DataFrame(result_data)
    
    # Build column list in the new order
    cols_to_use = ["Metric"]
    for model_name in model_order:
        cols_to_use.append(f"{model_display[model_name]} ID")
    for model_name in model_order:
        cols_to_use.append(f"{model_display[model_name]} OD")
    
    # Only keep columns that exist
    cols_to_use = [c for c in cols_to_use if c in result_df.columns]
    result_df = result_df[cols_to_use]
    
    latex_code = result_df.to_latex(
        index=False,
        escape=False,
        column_format="l" + "c" * (len(cols_to_use) - 1),
        caption="Metrics summary for all models and domains"
    )
    
    return latex_code
 