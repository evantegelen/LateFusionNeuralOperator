# Late Fusion Neural Operators

This repository contains the code accompanying the paper:

**Late Fusion Neural Operators for Extrapolation Across Parameter Space in Partial Differential Equations**.


## Project Structure

- `src/`: training, evaluation, models, and utilities
- `configs/`: benchmark and ablation configuration files
- `data/`: raw and processed datasets
- `notebooks/`: visualisation, interpretation, and figure generation
- `outputs/`: saved checkpoints, run configs, and evaluation outputs

## How To Train A Model

Training entrypoint: `src/train.py`

### Option 1: Train from a config file (recommended)

```bash
python -m src.train --config configs/advection_benchmark.yaml
```

You can similarly use other config files, for example:

- `configs/burgers_benchmark.yaml`
- `configs/reactiondiffusion_benchmark.yaml`
- `configs/reactiondiffusion2d_benchmark.yaml`
- `configs/advection_latefusion.yaml`
- `configs/advection_abblation.yaml`

### Option 2: Single run from command line arguments

```bash
python -m src.train --equation advection --model late_fusion_fno1d --seed 0 --max_epochs 100
```

Outputs are written under `outputs/<experiment_name>/<run_name>/`, including checkpoints and resolved run config.

## How To Evaluate A Model

Evaluation entrypoint: `src/evaluate.py`

```bash
python -m src.evaluate --config configs/advection_benchmark.yaml --checkpoint best
```

Useful optional flags:

- `--domains id od`
- `--batch-size-test 500`
- `--save-all-metrics`
- `--save-traj-rmse`

Evaluation outputs are written under:

- `outputs/<experiment_name>/evaluation/`

## Visualisation And Figure Generation

Visualisation and figure generation code is provided in `notebooks/`.

Examples include:

- `notebooks/evaluation_results.ipynb`
- `notebooks/heatmap_figure.ipynb`
- `notebooks/interpretation.ipynb`
- `notebooks/interpretation_2d.ipynb`
- `notebooks/parametererror_figures.ipynb`

## Running CAPE-FNO Benchmarks (Optional)

If you also want to run CAPE-FNO benchmarks:

1. Install CAPE locally: https://github.com/nec-research/CAPE-ML4Sci
2. In CAPE `models/PrmEmb/PrmEmb.py` (or `models/PrmEmbd/PrmEmbd.py`, depending on CAPE version), edit the block around lines 338-340:
   - remove `sigmoid` and `log`
   - use `x_p` as direct input to `gelu`
   - this avoids issues with negative data values
3. Add CAPE location to your local `.env` file:

```env
CAPE_ML4SCI_PATH=C:/path/to/CAPE-ML4Sci
```
