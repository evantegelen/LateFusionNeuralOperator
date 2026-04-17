import pandas as pd
from typing import Dict, List, Optional


def _normalize_benchmark_key(value: object) -> str:
    """Normalize benchmark names like 'Reaction-Diffusion 1D' -> 'reactiondiffusion1d'."""
    text = str(value).strip().lower()
    # Keep only alphanumeric characters so spaces, hyphens, and underscores don't matter.
    return "".join(ch for ch in text if ch.isalnum())


def _normalize_pm_latex(value: object) -> str:
    """Convert any plus-minus variant to LaTeX '$\\pm$' in table cell strings."""
    text = str(value)
    text = text.replace("$\\pm$", "+-")
    text = text.replace("\\pm", "+-")
    text = text.replace("±", "+-")
    text = text.replace("+-", "$\\pm$")
    return text

def rmse_summary_to_grouped_latex(
    rmse_summary: pd.DataFrame,
    benchmark_order: Optional[List[str]] = None,
    model_order: Optional[List[str]] = None,
    benchmark_display: Optional[Dict[str, str]] = None,
    model_display: Optional[Dict[str, str]] = None,
    caption: str = (
        "RMSE comparison for all test cases evaluated on in-domain and out-domain settings. "
        "Lower is better and best performing model is indicated in bold. Results are reported as mean $\\pm$ standard deviation."
    ),
    label: str = "tab:rmse_all",
) -> str:
    """
    Convert a combined RMSE summary DataFrame into a grouped LaTeX table.

    Expected input columns:
    - Benchmark
    - Model
    - In-domain
    - Out-domain

    The values in In-domain and Out-domain should already be formatted strings such as
    "4.53e-1 +- 9.88e-2".
    """
    if benchmark_order is None:
        benchmark_order = [
            "advection",
            "burgers",
            "reactiondiffusion1d",
            "reactiondiffusion2d",
        ]

    if model_order is None:
        model_order = ["fno", "cape_fno", "late_fusion"]

    if benchmark_display is None:
        benchmark_display = {
            "advection": "1D Advection",
            "burgers": "1D Burgers",
            "reactiondiffusion1d": "1D Reaction-Diffusion",
            "reactiondiffusion2d": "2D Reaction-Diffusion",
        }

    if model_display is None:
        model_display = {
            "fno": "FNO",
            "cape_fno": "CAPE-FNO",
            "late_fusion": "Late Fusion",
        }

    required = {"Benchmark", "Model", "In-domain", "Out-domain"}
    missing = required.difference(rmse_summary.columns)
    if missing:
        raise ValueError(f"RMSE summary is missing required columns: {sorted(missing)}")

    df = rmse_summary.copy()
    df["benchmark_key"] = df["Benchmark"].map(_normalize_benchmark_key)
    df["model_key"] = df["Model"].astype(str).str.strip().str.lower().str.replace("-", "_", regex=False)
    norm_benchmark_order = [_normalize_benchmark_key(x) for x in benchmark_order]
    df = df[df["benchmark_key"].isin(norm_benchmark_order) & df["model_key"].isin(model_order)].copy()

    def _value_to_float(text: object) -> float:
        if not isinstance(text, str):
            return float("nan")
        cleaned = text.replace(" ", "")
        cleaned = cleaned.replace("$\\pm$", "+-").replace("\\pm", "+-").replace("±", "+-")
        left = cleaned.split("+-", 1)[0]
        try:
            return float(left)
        except ValueError:
            return float("nan")

    lines: List[str] = []
    row_end = r"\\"
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\begin{tabular}{lcc}")
    lines.append(r"\toprule")
    lines.append("Model & In-domain & Out-domain " + row_end)
    lines.append(r"\midrule")

    for benchmark_key in norm_benchmark_order:
        subset = df[df["benchmark_key"] == benchmark_key].copy()
        if subset.empty:
            continue

        display_lookup = {
            _normalize_benchmark_key(k): v for k, v in benchmark_display.items()
        }
        bench_name = display_lookup.get(benchmark_key, benchmark_key)
        in_values = subset["In-domain"].map(_value_to_float)
        out_values = subset["Out-domain"].map(_value_to_float)
        in_min = in_values.min(skipna=True)
        out_min = out_values.min(skipna=True)

        lines.append(rf"\multicolumn{{3}}{{l}}{{\textbf{{{bench_name}}}}} " + row_end)
        lines.append(r"\midrule")

        for model_key in model_order:
            row = subset[subset["model_key"] == model_key]
            if row.empty:
                continue

            model_name = model_display.get(model_key, model_key)
            in_text = _normalize_pm_latex(row["In-domain"].iloc[0])
            out_text = _normalize_pm_latex(row["Out-domain"].iloc[0])

            in_fmt = f"\\textbf{{{in_text}}}" if _value_to_float(in_text) == in_min else in_text
            out_fmt = f"\\textbf{{{out_text}}}" if _value_to_float(out_text) == out_min else out_text

            lines.append(f"{model_name} & {in_fmt} & {out_fmt} " + row_end)

        lines.append(r"\midrule")

    if lines and lines[-1] == r"\midrule":
        lines.pop()

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def all_metrics_summary_to_grouped_latex(
    metrics_summary: pd.DataFrame,
    benchmark_order: Optional[List[str]] = None,
    model_order: Optional[List[str]] = None,
    benchmark_display: Optional[Dict[str, str]] = None,
    model_display: Optional[Dict[str, str]] = None,
    metric_order: Optional[List[str]] = None,
    caption: str = "Metrics summary for all models and domains",
    label: str = "tab:metrics_all",
) -> str:
    """
    Convert an all-metrics summary DataFrame into the grouped sidewaystable layout.

    Expected input columns:
    - Benchmark
    - Metric
    - FNO ID, CAPE-FNO ID, Late Fusion ID
    - FNO OD, CAPE-FNO OD, Late Fusion OD
    """
    if benchmark_order is None:
        benchmark_order = [
            "advection",
            "burgers",
            "reactiondiffusion1d",
            "reactiondiffusion2d",
        ]

    if model_order is None:
        model_order = ["fno", "cape_fno", "late_fusion"]

    if benchmark_display is None:
        benchmark_display = {
            "advection": "1D Advection",
            "burgers": "1D Burgers",
            "reactiondiffusion1d": "1D Reaction-diffusion",
            "reactiondiffusion2d": "2D Reaction-diffusion",
        }

    if model_display is None:
        model_display = {
            "fno": "FNO",
            "cape_fno": "CAPE-FNO",
            "late_fusion": "Late Fusion",
        }

    if metric_order is None:
        metric_order = ["Boundary", "Conserved", "Fourier", "Max", "RMSE", "nRMSE"]

    required = {"Benchmark", "Metric"}
    for model_key in model_order:
        required.add(f"{model_display[model_key]} ID")
        required.add(f"{model_display[model_key]} OD")

    missing = required.difference(metrics_summary.columns)
    if missing:
        raise ValueError(f"Metrics summary is missing required columns: {sorted(missing)}")

    df = metrics_summary.copy()
    df["benchmark_key"] = df["Benchmark"].map(_normalize_benchmark_key)
    df["metric_key"] = df["Metric"].astype(str).str.strip()
    norm_benchmark_order = [_normalize_benchmark_key(x) for x in benchmark_order]
    df = df[df["benchmark_key"].isin(norm_benchmark_order)].copy()

    def _value_to_float(text: object) -> float:
        if not isinstance(text, str):
            return float("nan")
        cleaned = text.replace(" ", "")
        cleaned = cleaned.replace("$\\pm$", "+-").replace("\\pm", "+-").replace("±", "+-")
        left = cleaned.split("+-", 1)[0]
        try:
            return float(left)
        except ValueError:
            return float("nan")

    row_end = r"\\"
    lines: List[str] = []
    lines.append(r"\begin{sidewaystable*}")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{|l|ccc|ccc|}")
    lines.append(r"\toprule")
    lines.append(r"& \multicolumn{3}{|c|}{In-domain} & \multicolumn{3}{c|}{Out-domain} " + row_end)
    lines.append(r"& \multicolumn{3}{|c|}{} & \multicolumn{3}{c|}{} " + row_end)
    lines.append(
        "Metric & "
        + " & ".join(model_display[m] for m in model_order)
        + " & "
        + " & ".join(model_display[m] for m in model_order)
        + " "
        + row_end
    )
    lines.append(r"\midrule")

    display_lookup = {
        _normalize_benchmark_key(k): v for k, v in benchmark_display.items()
    }

    for benchmark_key in norm_benchmark_order:
        subset = df[df["benchmark_key"] == benchmark_key].copy()
        if subset.empty:
            continue

        bench_name = display_lookup.get(benchmark_key, benchmark_key)
        lines.append(rf"\multicolumn{{3}}{{l}}{{\textbf{{{bench_name}}}}} " + row_end)
        lines.append(r"\midrule")

        metric_lookup = {m: subset[subset["metric_key"] == m] for m in metric_order}
        ordered_metrics = [m for m in metric_order if not metric_lookup[m].empty]
        extra_metrics = [m for m in subset["metric_key"].tolist() if m not in metric_order]
        ordered_metrics.extend(extra_metrics)

        for metric in ordered_metrics:
            row = metric_lookup.get(metric, pd.DataFrame())
            if row.empty:
                row = subset[subset["metric_key"] == metric]
            if row.empty:
                continue

            id_cells = []
            od_cells = []
            for model_key in model_order:
                id_col = f"{model_display[model_key]} ID"
                od_col = f"{model_display[model_key]} OD"
                id_cells.append(_normalize_pm_latex(row[id_col].iloc[0]))
                od_cells.append(_normalize_pm_latex(row[od_col].iloc[0]))

            id_values = [_value_to_float(value) for value in id_cells]
            od_values = [_value_to_float(value) for value in od_cells]
            id_min = min(id_values)
            od_min = min(od_values)

            row_text = [metric]
            for value in id_cells:
                if _value_to_float(value) == id_min:
                    row_text.append(f"\\textbf{{{value}}}")
                else:
                    row_text.append(value)
            for value in od_cells:
                if _value_to_float(value) == od_min:
                    row_text.append(f"\\textbf{{{value}}}")
                else:
                    row_text.append(value)
            lines.append(" & ".join(row_text) + " " + row_end)

        lines.append(r"\midrule")

    if lines and lines[-1] == r"\midrule":
        lines.pop()

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{sidewaystable*}")

    return "\n".join(lines)
