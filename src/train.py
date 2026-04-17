import comet_ml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse
import os
from pathlib import Path
import json
from src.models.base_lightning_model import PDELightningModule, create_comet_logger
from src.data.equation_datamodule import EquationModule
from src.utils.configs.training_config import load_config, build_run_plan, build_configs

def train_from_run(run_cfg: dict):
    equation = run_cfg["equation"]
    model_name = run_cfg["model"]
    seed = int(run_cfg.get("seed", 0))
    training = run_cfg.get("training", {})

    pl.seed_everything(seed, workers=True)

    resolved_model_name, model_config, datamodule_config = build_configs(equation, model_name)
    model_config.update(run_cfg.get("model_overrides", {}))
    datamodule_config.update(run_cfg.get("data_overrides", {}))

    learning_rate = float(training.get("learning_rate", 1e-3))
    max_epochs = int(training.get("max_epochs", 100))
    devices = training.get("devices", 1)
    accelerator = training.get("accelerator", "auto")
    log_every_n_steps = int(training.get("log_every_n_steps", 10))

    run_name = run_cfg["name"]
    run_dir = Path(run_cfg.get("output_dir", "outputs")) / run_cfg["experiment_name"] / run_name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    resolved = {
        "equation": equation,
        "model": model_name,
        "resolved_model_name": resolved_model_name,
        "seed": seed,
        "model_config": model_config,
        "datamodule_config": datamodule_config,
        "training": training,
        "run_name": run_name,
    }
    (run_dir / "resolved_config.json").write_text(json.dumps(resolved, indent=2), encoding="utf-8")

    dm = EquationModule(**datamodule_config)
    model = PDELightningModule(
        model_name=resolved_model_name,
        model_config=model_config,
        learning_rate=learning_rate,
        log_hyperparameters=True,
    )

    logger = create_comet_logger(
        project_name=os.getenv("COMET_PROJECT_NAME"),
        model_name=model_name,
        equation_name=equation,
        experiment_name=run_name,
        config=resolved,
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        save_last=True,
        save_top_k=1,          # keep the best checkpoint as well as the last
        monitor="val_loss",    # adjust to the metric your model logs (e.g. "val_loss" or "val/metric")
        mode="min",            # "min" for loss, "max" for accuracy-like metrics
        every_n_epochs=1,
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=logger,
        devices=devices,
        accelerator=accelerator,
        log_every_n_steps=log_every_n_steps,
        default_root_dir=str(run_dir),
        callbacks=[checkpoint_cb],
    )

    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm, ckpt_path="last")
    print(f"Training completed: {run_name}")

    #Save run info for easy reference
    run_info = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "last_checkpoint": str(ckpt_dir / "last.ckpt"),
        "comet_experiment_key": logger.experiment.get_key(),
    }
    (run_dir / "run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")

    trainer.logger.finalize("success")
    logger.experiment.end()

def train_model(
    equation: str,
    model_name: str,
    seed: int = 0,
    max_epochs: int = 100,
    learning_rate: float = 1e-3,
    output_dir: str = "outputs",
    devices: int = 1,
    accelerator: str = "auto",
    log_every_n_steps: int = 10,
):
    """Train a single model using the same pipeline as config-based runs."""
    run_cfg = {
        "name": f"{equation}_{model_name}_seed{seed}",
        "experiment_name": f"{equation}_single",
        "equation": equation,
        "seed": int(seed),
        "model": model_name,
        "model_overrides": {},
        "training": {
            "max_epochs": int(max_epochs),
            "learning_rate": float(learning_rate),
            "devices": devices,
            "accelerator": accelerator,
            "log_every_n_steps": int(log_every_n_steps),
        },
        "output_dir": output_dir,
    }

    train_from_run(run_cfg)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to yaml/json config")
    parser.add_argument("--equation", type=str, required=False)
    parser.add_argument("--model", type=str, required=False)

    # single-run CLI overrides (used only when --config is not provided)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--log_every_n_steps", type=int, default=10)

    args = parser.parse_args()

    if args.config:
        cfg = load_config(args.config)
        runs = build_run_plan(cfg)
        # quick override check
        print("First run model_overrides:", runs[0].get("model_overrides", {}))
        print("First run data_overrides:", runs[0].get("data_overrides", {}))

        for run_cfg in runs:
            train_from_run(run_cfg)
        return

    if not args.equation or not args.model:
        parser.error("Provide --config OR both --equation and --model.")

    train_model(
        equation=args.equation,
        model_name=args.model,
        seed=args.seed,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        devices=args.devices,
        accelerator=args.accelerator,
        log_every_n_steps=args.log_every_n_steps,
    )

if __name__ == "__main__":
    main()