import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.utils.configs.training_config import build_run_plan, load_config
from src.utils.evaluation.evaluation import (
    collect_traj_rmse_for_runs,
    combination_key,
    evaluate_run_test,
    evaluate_run_test_all_metrics,
    evaluate_run_validation,
    select_best_combinations,
)

def _make_output_root(cfg: dict) -> Path:
    experiment_name = cfg.get("experiment_name", "experiment")
    return Path(cfg.get("output_dir", "outputs")) / experiment_name / "evaluation"

def _aggregate_test_summary(test_df: pd.DataFrame) -> pd.DataFrame:
    if test_df.empty:
        return pd.DataFrame(
            columns=[
                "model",
                "combination",
                "domain",
                "loader",
                "test_rmse_mean",
                "test_rmse_std",
                "num_seeds",
            ]
        )

    grouped = (
        test_df.groupby(["model", "combination", "domain", "loader"], as_index=False)
        .agg(
            test_rmse_mean=("test_rmse", "mean"),
            test_rmse_std=("test_rmse", "std"),
            num_seeds=("seed", "count"),
        )
        .sort_values(["model", "domain", "loader"])
        .reset_index(drop=True)
    )
    grouped["test_rmse_std"] = grouped["test_rmse_std"].fillna(0.0)
    return grouped

def _aggregate_test_summary_wide(summary_long: pd.DataFrame) -> pd.DataFrame:
    if summary_long.empty:
        return pd.DataFrame(columns=["model", "combination"])

    mean_wide = (
        summary_long.pivot_table(
            index=["model", "combination"],
            columns="loader",
            values="test_rmse_mean",
            aggfunc="first",
        )
        .add_suffix("_mean")
        .reset_index()
    )

    std_wide = (
        summary_long.pivot_table(
            index=["model", "combination"],
            columns="loader",
            values="test_rmse_std",
            aggfunc="first",
        )
        .add_suffix("_std")
        .reset_index()
    )

    out = mean_wide.merge(std_wide, on=["model", "combination"], how="outer")
    return out.sort_values(["model", "combination"]).reset_index(drop=True)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate benchmark runs with one unified pipeline.")
    parser.add_argument("--config", type=str, required=True, help="Path to benchmark or single-run config.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best",
        choices=["best", "last"],
        help="Which checkpoint to evaluate.",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["id", "od"],
        help="Domains to evaluate. Options: id od",
    )
    parser.add_argument(
        "--batch-size-test",
        type=int,
        default=500,
        help="Batch size for test dataloaders.",
    )
    parser.add_argument(
        "--save-all-metrics",
        action="store_true",
        help="Also compute and save all metrics from compute_all_metrics.",
    )
    parser.add_argument(
        "--save-traj-rmse",
        action="store_true",
        help="Also compute and save trajectory-level RMSE.",
    )
    parser.add_argument(
        "--traj-param-key",
        type=str,
        default="parameter",
        help="Parameter key used when saving trajectory-level RMSE.",
    )
    parser.add_argument(
        "--traj-param-index",
        type=int,
        default=0,
        help="Index inside the parameter tensor to save when it has multiple parameters.",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    use_best = args.checkpoint == "best"

    cfg = load_config(args.config)
    runs = build_run_plan(cfg)

    print(f"Loaded {len(runs)} runs from {args.config}")

    # 1) Validation for all runs
    val_rows: List[Dict] = []
    for i, run_cfg in enumerate(runs, start=1):
        print(f"[VAL {i}/{len(runs)}] {run_cfg['name']}")
        val_rows.append(evaluate_run_validation(run_cfg, use_best=use_best))

    val_df = pd.DataFrame(val_rows)
    val_df["combination"] = val_df["run_name"].map(combination_key)

    combo_val, best_combo = select_best_combinations(val_df)

    print("\nBest combination per model (from validation):")
    print(
        best_combo[
            ["model", "combination", "val_error_mean", "val_error_std", "num_seeds"]
        ].to_string(index=False)
    )

    # 2) Select runs belonging to best combination per model
    selected = val_df.merge(
        best_combo[["model", "combination"]],
        on=["model", "combination"],
        how="inner",
    )

    run_lookup = {r["name"]: r for r in runs}
    selected_runs = [run_lookup[name] for name in selected["run_name"].tolist()]

    # 3) RMSE test evaluation
    test_rows: List[Dict] = []
    for i, run_cfg in enumerate(selected_runs, start=1):
        print(f"[TEST {i}/{len(selected_runs)}] {run_cfg['name']}")
        rows = evaluate_run_test(
            run_cfg=run_cfg,
            domains=args.domains,
            use_best=use_best,
            batch_size_test=args.batch_size_test,
        )
        test_rows.extend(rows)

    test_df = pd.DataFrame(test_rows)
    test_df["combination"] = test_df["run_name"].map(combination_key)

    test_summary_long = _aggregate_test_summary(test_df)
    test_summary_wide = _aggregate_test_summary_wide(test_summary_long)

    # 4) Save outputs
    output_root = _make_output_root(cfg)
    output_root.mkdir(parents=True, exist_ok=True)

    val_runs_csv = output_root / "validation_per_run.csv"
    val_combo_csv = output_root / "validation_by_combination.csv"
    best_combo_csv = output_root / "best_combination_per_model.csv"
    selected_runs_csv = output_root / "selected_runs.csv"
    test_runs_csv = output_root / "test_per_selected_run.csv"
    test_summary_long_csv = output_root / "test_summary_by_loader.csv"
    test_summary_wide_csv = output_root / "test_summary_best_combinations.csv"
    metadata_json = output_root / "metadata.json"

    val_df.to_csv(val_runs_csv, index=False)
    combo_val.to_csv(val_combo_csv, index=False)
    best_combo.to_csv(best_combo_csv, index=False)
    selected.to_csv(selected_runs_csv, index=False)
    test_df.to_csv(test_runs_csv, index=False)
    test_summary_long.to_csv(test_summary_long_csv, index=False)
    test_summary_wide.to_csv(test_summary_wide_csv, index=False)

    # Optional all metrics
    if args.save_all_metrics:
        all_metric_rows: List[Dict] = []
        for i, run_cfg in enumerate(selected_runs, start=1):
            print(f"[ALL-METRICS {i}/{len(selected_runs)}] {run_cfg['name']}")
            rows = evaluate_run_test_all_metrics(
                run_cfg=run_cfg,
                domains=args.domains,
                use_best=use_best,
                batch_size_test=args.batch_size_test,
            )
            all_metric_rows.extend(rows)

        all_metrics_df = pd.DataFrame(all_metric_rows)
        all_metrics_df["combination"] = all_metrics_df["run_name"].map(combination_key)
        all_metrics_df.to_csv(output_root / "test_all_metrics_per_selected_run.csv", index=False)

    # Optional trajectory rmse
    if args.save_traj_rmse:
        traj_df = collect_traj_rmse_for_runs(
            run_cfgs=selected_runs,
            domains=args.domains,
            use_best=use_best,
            param_key=args.traj_param_key,
            param_index=args.traj_param_index,
            batch_size_test=args.batch_size_test,
        )
        traj_df["combination"] = traj_df["run_name"].map(combination_key)
        traj_df.to_csv(output_root / "traj_rmse_per_trajectory.csv", index=False)

    metadata = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "domains": args.domains,
        "batch_size_test": args.batch_size_test,
        "num_runs_total": len(runs),
        "num_runs_selected": len(selected_runs),
        "saved_all_metrics": bool(args.save_all_metrics),
        "saved_traj_rmse": bool(args.save_traj_rmse),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nSaved files:")
    print(f"- {val_runs_csv}")
    print(f"- {val_combo_csv}")
    print(f"- {best_combo_csv}")
    print(f"- {selected_runs_csv}")
    print(f"- {test_runs_csv}")
    print(f"- {test_summary_long_csv}")
    print(f"- {test_summary_wide_csv}")
    if args.save_all_metrics:
        print(f"- {output_root / 'test_all_metrics_per_selected_run.csv'}")
    if args.save_traj_rmse:
        print(f"- {output_root / 'traj_rmse_per_trajectory.csv'}")
    print(f"- {metadata_json}")

if __name__ == "__main__":
    main()