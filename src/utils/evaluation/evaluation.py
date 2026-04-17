import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd
import pytorch_lightning as pl
import torch

from src.data.equation_datamodule import EquationModule
from src.models.base_lightning_model import PDELightningModule
from src.train import build_configs
from src.utils.evaluation.metrics import compute_all_metrics, rmse


def resolve_ckpt_path(run_cfg: dict, use_best: bool = True) -> Path:
    run_dir = Path(run_cfg.get("output_dir", "outputs")) / run_cfg["experiment_name"] / run_cfg["name"]
    ckpt_dir = run_dir / "checkpoints"
    if use_best:
        best_ckpts = sorted(ckpt_dir.glob("epoch=*.ckpt"))
        if best_ckpts:
            return best_ckpts[-1]
    return ckpt_dir / "last.ckpt"

def load_model_and_datamodule(run_cfg: dict, use_best: bool = True):
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

def combination_key(run_name: str) -> str:
    return re.sub(r"_seed\d+$", "", run_name)

def _to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

def get_eval_loaders(
    dm,
    domains: Union[str, Iterable[str]] = ("id", "od"),
) -> List[Tuple[str, str, torch.utils.data.DataLoader]]:
    if isinstance(domains, str):
        domains = [domains]

    domains = {str(d).strip().lower() for d in domains}
    out: List[Tuple[str, str, torch.utils.data.DataLoader]] = []

    if "id" in domains:
        out.append(("id", "test_id", dm.test_dataloader()))

    if "od" in domains:
        out.append(("od", "test_od", dm.get_test_od_dataloaders()))
    if not out:
        raise ValueError("No valid domains requested. Use id, od, or both.")
    return out

def evaluate_run_validation(run_cfg: dict, use_best: bool = True) -> Dict:
    seed = int(run_cfg.get("seed", 0))
    pl.seed_everything(seed, workers=True)

    model, dm, trainer = load_model_and_datamodule(run_cfg, use_best=use_best)
    val_out = trainer.validate(model=model, dataloaders=dm.val_dataloader(), verbose=False)
    val_error = val_out[0].get("val_loss", float("nan")) if val_out else float("nan")

    return {
        "run_name": run_cfg["name"],
        "model": run_cfg["model"],
        "seed": seed,
        "val_error": float(val_error),
    }

def evaluate_one_loader_rmse(model, loader, device: torch.device) -> Dict[str, float]:
    all_preds = []
    all_trues = []

    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            pred = model.roll_out_predictions(batch)
            true = batch["state"][:, 1:, ...]
            all_preds.append(pred)
            all_trues.append(true)

    if not all_preds:
        return {"test_rmse": float("nan")}

    pred_all = torch.cat(all_preds, dim=0)
    true_all = torch.cat(all_trues, dim=0)
    return {"test_rmse": float(rmse(pred_all, true_all).detach().cpu().item())}

def evaluate_run_test(
    run_cfg: dict,
    domains: Union[str, Iterable[str]] = ("id", "od"),
    use_best: bool = True,
    batch_size_test: int = 500,
) -> List[Dict]:
    seed = int(run_cfg.get("seed", 0))
    pl.seed_everything(seed, workers=True)

    model, dm, _ = load_model_and_datamodule(run_cfg, use_best=use_best)
    if hasattr(dm, "batch_size_test"):
        dm.batch_size_test = int(batch_size_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    rows: List[Dict] = []
    for domain, loader_name, loader in get_eval_loaders(dm, domains):
        stats = evaluate_one_loader_rmse(model, loader, device)
        rows.append(
            {
                "run_name": run_cfg["name"],
                "model": run_cfg["model"],
                "seed": seed,
                "domain": domain,
                "loader": loader_name,
                **stats,
            }
        )
    return rows

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
            pred = model.roll_out_predictions(batch)
            true = batch["state"][:, 1:, ...]
            all_preds.append(pred)
            all_trues.append(true)

    if not all_preds:
        return {
            "RMSE": float("nan"),
            "nRMSE": float("nan"),
            "Max": float("nan"),
            "Boundary": float("nan"),
            "Conserved": float("nan"),
            "Fourier": float("nan"),
        }

    pred_all = torch.cat(all_preds, dim=0)
    true_all = torch.cat(all_trues, dim=0)
    metrics = compute_all_metrics(pred_all, true_all)

    return {
        k: float(v.detach().cpu().item() if torch.is_tensor(v) else v)
        for k, v in metrics.items()
    }

def evaluate_run_test_all_metrics(
    run_cfg: dict,
    domains: Union[str, Iterable[str]] = ("id", "od"),
    use_best: bool = True,
    batch_size_test: int = 500,
) -> List[Dict]:
    seed = int(run_cfg.get("seed", 0))
    pl.seed_everything(seed, workers=True)

    model, dm, _ = load_model_and_datamodule(run_cfg, use_best=use_best)
    if hasattr(dm, "batch_size_test"):
        dm.batch_size_test = int(batch_size_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    rows: List[Dict] = []
    for domain, loader_name, loader in get_eval_loaders(dm, domains):
        metric_vals = evaluate_one_loader_all_metrics(model, loader, device)
        rows.append(
            {
                "run_name": run_cfg["name"],
                "model": run_cfg["model"],
                "seed": seed,
                "domain": domain,
                "loader": loader_name,
                **metric_vals,
            }
        )
    return rows

def evaluate_run_test_traj_rmse(
    run_cfg: dict,
    domains: Union[str, Iterable[str]] = ("id", "od"),
    use_best: bool = True,
    param_key: str = "parameter",
    param_index: int = 0,
    batch_size_test: int = 500,
) -> List[Dict]:
    seed = int(run_cfg.get("seed", 0))
    pl.seed_everything(seed, workers=True)

    model, dm, _ = load_model_and_datamodule(run_cfg, use_best=use_best)
    if hasattr(dm, "batch_size_test"):
        dm.batch_size_test = int(batch_size_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    rows: List[Dict] = []

    with torch.no_grad():
        for domain, loader_name, loader in get_eval_loaders(dm, domains):
            for batch_idx, batch in enumerate(loader):
                batch = _to_device(batch, device)
                pred = model.roll_out_predictions(batch)
                true = batch["state"][:, 1:, ...]

                reduce_dims = tuple(range(1, pred.ndim))
                traj_rmse = torch.sqrt(torch.mean((pred - true) ** 2, dim=reduce_dims))

                if param_key not in batch:
                    raise KeyError(f"Batch missing '{param_key}'. Available keys: {list(batch.keys())}")

                p = batch[param_key]
                if p.ndim > 1:
                    p = p.reshape(p.shape[0], -1)
                    if param_index < 0 or param_index >= p.shape[1]:
                        raise IndexError(
                            f"param_index={param_index} out of range for '{param_key}' with shape {tuple(batch[param_key].shape)}"
                        )
                    p = p[:, param_index]
                p = p.detach().cpu().numpy()
                rmse_np = traj_rmse.detach().cpu().numpy()

                for i in range(len(rmse_np)):
                    rows.append(
                        {
                            "run_name": run_cfg["name"],
                            "model": run_cfg["model"],
                            "seed": seed,
                            "domain": domain,
                            "loader": loader_name,
                            "batch_idx": batch_idx,
                            "traj_idx_in_batch": i,
                            "param_index": int(param_index),
                            "param_value": float(p[i]),
                            "traj_rmse": float(rmse_np[i]),
                        }
                    )

    return rows

def collect_traj_rmse_for_runs(
    run_cfgs: Iterable[dict],
    domains: Union[str, Iterable[str]] = ("id", "od"),
    use_best: bool = True,
    param_key: str = "parameter",
    param_index: int = 0,
    batch_size_test: int = 500,
) -> pd.DataFrame:
    all_rows: List[Dict] = []
    for run_cfg in run_cfgs:
        all_rows.extend(
            evaluate_run_test_traj_rmse(
                run_cfg=run_cfg,
                domains=domains,
                use_best=use_best,
                param_key=param_key,
                param_index=param_index,
                batch_size_test=batch_size_test,
            )
        )
    return pd.DataFrame(all_rows)

def select_best_combinations(val_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = val_df.copy()
    df["combination"] = df["run_name"].map(combination_key)

    combo_val = (
        df.groupby(["model", "combination"], as_index=False)
        .agg(
            val_error_mean=("val_error", "mean"),
            val_error_std=("val_error", "std"),
            num_seeds=("seed", "count"),
        )
        .sort_values(["model", "val_error_mean"])
        .reset_index(drop=True)
    )
    combo_val["val_error_std"] = combo_val["val_error_std"].fillna(0.0)

    best_idx = combo_val.groupby("model")["val_error_mean"].idxmin()
    best_combo = combo_val.loc[best_idx].sort_values("val_error_mean").reset_index(drop=True)

    return combo_val, best_combo

def collect_best_run_predictions(
    best_runs_by_model: Dict[str, dict],
    plot_id_idx: int = 2,
    plot_od_idx: int = 3,
    batch_size_test: int = 10,
    use_best: bool = True,
    device: Optional[torch.device] = None,
) -> Dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    predictions_id: Dict[str, object] = {}
    predictions_od: Dict[str, object] = {}
    true_solution_id = None
    true_solution_od = None
    t_coord = None
    x_coord = None
    param_id = None
    param_od = None

    for model_name, run_cfg in best_runs_by_model.items():
        model, dm, _ = load_model_and_datamodule(run_cfg, use_best=use_best)
        if hasattr(dm, "batch_size_test"):
            dm.batch_size_test = int(batch_size_test)

        id_loader = dm.test_dataloader()
        od_loader = dm.get_test_od_dataloaders()
        model = model.to(device).eval()

        with torch.no_grad():
            batch_id = _to_device(next(iter(id_loader)), device)
            pred_id = model.roll_out_predictions(batch_id)
            true_id = batch_id["state"][:, 1:, ...]

            predictions_id[model_name] = pred_id[plot_id_idx].detach().cpu().numpy()
            if true_solution_id is None:
                true_solution_id = true_id[plot_id_idx].detach().cpu().numpy()
                
                t_coord = batch_id["t"].detach().cpu().numpy()
                x_coord = batch_id["x"].detach().cpu().numpy()
                param_id = batch_id["parameter"][plot_id_idx].detach().cpu().numpy()

            batch_od = _to_device(next(iter(od_loader)), device)
            pred_od = model.roll_out_predictions(batch_od)
            true_od = batch_od["state"][:, 1:, ...]

            predictions_od[model_name] = pred_od[plot_od_idx].detach().cpu().numpy()
            if true_solution_od is None:
                true_solution_od = true_od[plot_od_idx].detach().cpu().numpy()
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