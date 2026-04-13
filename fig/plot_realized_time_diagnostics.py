from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = Path(__file__).resolve().parent


def resolve_log_path(result_dir: str | Path) -> Path | None:
    result_dir = Path(result_dir)
    candidates = [result_dir / "train.log", result_dir / "training_log.txt"]
    for path in candidates:
        if path.exists() and path.stat().st_size > 0:
            return path
    for path in candidates:
        if path.exists():
            return path
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot realized-time diagnostics for AtomWorld-Twins macro world model"
    )
    parser.add_argument(
        "--model-eval",
        type=str,
        default=str(ROOT / "dreamer4-main" / "results" / "dreamer_macro_edit_v26_realized_qtrain" / "eval_time_alignment_realized_final.json"),
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(ROOT / "dreamer4-main" / "results" / "dreamer_macro_edit_v26_realized_qtrain"),
    )
    parser.add_argument("--title-name", type=str, default="AtomWorld-Twins")
    parser.add_argument("--model-label", type=str, default="AtomWorld-Twins")
    parser.add_argument(
        "--output",
        type=str,
        default=str(FIG_DIR / "macro_edit_realized_time.png"),
    )
    return parser.parse_args()


def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_sample_array(samples: list[dict], keys: list[str], fallback: float | None = None) -> np.ndarray:
    values = []
    for sample in samples:
        value = None
        for key in keys:
            if key in sample:
                value = sample[key]
                break
        if value is None:
            value = fallback
        values.append(np.nan if value is None else float(value))
    return np.asarray(values, dtype=np.float64)


def extract_realized_arrays(eval_data: dict) -> dict[str, np.ndarray]:
    samples = eval_data.get("all_samples", [])
    pred_tau = _load_sample_array(samples, ["predicted_tau"])
    pred_realized_tau = _load_sample_array(
        samples,
        ["predicted_realized_tau", "predicted_tau_realized", "predicted_tau"],
    )
    pred_realized_tau_log_mu = _load_sample_array(samples, ["predicted_realized_tau_log_mu"], fallback=None)
    pred_realized_tau_log_sigma = _load_sample_array(samples, ["predicted_realized_tau_log_sigma"], fallback=None)
    true_tau_real = _load_sample_array(samples, ["traditional_kmc_realized_tau"])
    true_tau_exp = _load_sample_array(samples, ["traditional_kmc_expected_tau"])
    return {
        "pred_tau": pred_tau,
        "pred_realized_tau": pred_realized_tau,
        "pred_realized_tau_log_mu": pred_realized_tau_log_mu,
        "pred_realized_tau_log_sigma": pred_realized_tau_log_sigma,
        "true_tau_real": true_tau_real,
        "true_tau_exp": true_tau_exp,
    }


def parse_realized_val_log(result_dir: str | Path) -> dict[str, list[float]]:
    metric_keys = [
        "tau_log_mae",
        "real_tau_nll",
        "real_tau_log_mae",
        "real_tau_cov68",
        "real_tau_pit_ks",
    ]
    log_path = resolve_log_path(result_dir)
    if log_path is None:
        return {"epochs": [], **{key: [] for key in metric_keys}}
    epoch_to_metrics: dict[int, dict[str, float]] = {}
    current_epoch: int | None = None
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            epoch_match = re.search(r"\[Epoch\s+(\d+)/(\d+)\]", line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
            if ">>> VAL" not in line or current_epoch is None:
                continue
            metrics: dict[str, float] = {}
            for key in metric_keys:
                match = re.search(rf"{re.escape(key)}=([\d.e+-]+)", line)
                if match:
                    metrics[key] = float(match.group(1))
            epoch_to_metrics[current_epoch] = metrics
    epochs = sorted(epoch_to_metrics)
    result: dict[str, list[float]] = {"epochs": epochs}
    for key in metric_keys:
        result[key] = [epoch_to_metrics[epoch].get(key, np.nan) for epoch in epochs]
    return result


def _annotate_center(ax: plt.Axes, title: str, lines: list[str]) -> None:
    ax.set_title(title, fontsize=12)
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.6"},
    )


def _finite_mask(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones_like(arrays[0], dtype=bool)
    for array in arrays:
        mask &= np.isfinite(array)
    return mask


def _latest_finite(values: list[float]) -> float | None:
    for value in reversed(values):
        if np.isfinite(value):
            return float(value)
    return None


def _compute_pit(true_tau_real: np.ndarray, log_mu: np.ndarray, log_sigma: np.ndarray) -> np.ndarray:
    sigma = np.exp(np.clip(log_sigma, -12.0, 12.0))
    sigma = np.clip(sigma, 1e-6, None)
    z = (np.log(np.clip(true_tau_real, 1e-12, None)) - log_mu) / (sigma * math.sqrt(2.0))
    erf_values = np.vectorize(math.erf)(z)
    return np.clip(0.5 * (1.0 + erf_values), 0.0, 1.0)


def main() -> None:
    args = parse_args()
    model_eval = load_json(args.model_eval)
    arrays = extract_realized_arrays(model_eval)
    val_metrics = parse_realized_val_log(args.model_dir)

    realized_point = model_eval.get("tau_realized", {})
    realized_distribution = model_eval.get(
        "tau_realized_distribution",
        {"available": False, "reason": "missing_from_eval_json"},
    )
    time_heads = model_eval.get("time_heads", {})
    point_source = realized_point.get("prediction_source", time_heads.get("realized_tau_source", "unknown"))

    point_mask = _finite_mask(arrays["true_tau_real"], arrays["pred_realized_tau"])
    point_mask &= (arrays["true_tau_real"] > 0.0) & (arrays["pred_realized_tau"] > 0.0)
    dist_mask = point_mask.copy()
    dist_mask &= _finite_mask(arrays["pred_realized_tau_log_mu"], arrays["pred_realized_tau_log_sigma"])

    point_true = arrays["true_tau_real"][point_mask]
    point_pred = arrays["pred_realized_tau"][point_mask]
    residual = np.log10(np.clip(point_pred, 1e-12, None) / np.clip(point_true, 1e-12, None)) if point_true.size else np.asarray([], dtype=np.float64)

    distribution_available = bool(realized_distribution.get("available", False)) and bool(np.any(dist_mask))
    pit = _compute_pit(
        arrays["true_tau_real"][dist_mask],
        arrays["pred_realized_tau_log_mu"][dist_mask],
        arrays["pred_realized_tau_log_sigma"][dist_mask],
    ) if distribution_available else np.asarray([], dtype=np.float64)

    model_color = "tab:green"
    kmc_color = "black"

    fig = plt.figure(figsize=(18, 20))
    fig.suptitle(
        f"{args.title_name}: Realized Waiting Time Diagnostics\n"
        "Teacher baseline = traditional KMC realized tau over the same 400 val segments",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    gs = fig.add_gridspec(3, 2, hspace=0.36, wspace=0.26, top=0.93, bottom=0.05)

    ax1 = fig.add_subplot(gs[0, 0])
    if point_true.size:
        lo = min(point_true.min(), point_pred.min()) * 0.5
        hi = max(point_true.max(), point_pred.max()) * 2.0
        ax1.scatter(point_true, point_pred, alpha=0.35, s=16, c=model_color, edgecolors="none")
        ax1.plot([lo, hi], [lo, hi], "--", color=kmc_color, linewidth=2, alpha=0.7, label="y=x")
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.set_xlim(lo, hi)
        ax1.set_ylim(lo, hi)
        ax1.set_xlabel("Traditional KMC: realized tau", fontsize=10)
        ax1.set_ylabel(f"{args.model_label}: predicted realized tau", fontsize=10)
        ax1.set_title(
            f"Point Reference vs Teacher Realized Time\n"
            f"log_MAE={realized_point.get('log_mae', float('nan')):.3f}, "
            f"log_corr={realized_point.get('log_corr', float('nan')):.3f}, "
            f"scale={realized_point.get('scale_ratio', float('nan')):.2f}, "
            f"source={point_source}",
            fontsize=12,
        )
        ax1.legend(fontsize=9, loc="upper left")
        ax1.set_aspect("equal")
    else:
        _annotate_center(ax1, "Point Reference vs Teacher Realized Time", ["No valid realized-time samples found"])

    ax2 = fig.add_subplot(gs[0, 1])
    if residual.size:
        within_2x = float(np.mean(np.abs(residual) < np.log10(2.0)) * 100.0)
        within_5x = float(np.mean(np.abs(residual) < np.log10(5.0)) * 100.0)
        median_ratio = float(np.median(residual))
        ax2.scatter(point_true, residual, alpha=0.3, s=13, c=model_color, edgecolors="none")
        ax2.axhline(0.0, color=kmc_color, linewidth=2, linestyle="--", label="Perfect")
        ax2.axhline(np.log10(2.0), color="0.5", linewidth=0.8, linestyle=":", alpha=0.4)
        ax2.axhline(-np.log10(2.0), color="0.5", linewidth=0.8, linestyle=":", alpha=0.4)
        ax2.axhline(median_ratio, color=model_color, linewidth=1.5, linestyle="-", label=f"Median={10 ** median_ratio:.2f}x")
        ax2.set_xscale("log")
        ax2.set_xlabel("Traditional KMC: realized tau", fontsize=10)
        ax2.set_ylabel("log10(pred / teacher)", fontsize=10)
        ax2.set_ylim(-5, 2)
        ax2.set_title(
            f"Realized-Time Residual\nWithin 2x: {within_2x:.0f}%  |  Within 5x: {within_5x:.0f}%  |  source={point_source}",
            fontsize=12,
        )
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.2)
    else:
        _annotate_center(ax2, "Realized-Time Residual", ["No valid realized-time residuals found"])

    ax3 = fig.add_subplot(gs[1, 0])
    if residual.size:
        bins = np.linspace(min(residual.min(), -5.0), max(residual.max(), 2.0), 70)
        ax3.hist(residual, bins=bins, alpha=0.65, color=model_color, density=True, label=f"std={np.std(residual):.2f}")
        ax3.axvline(0.0, color=kmc_color, linewidth=2, linestyle="--", label="Perfect")
        ax3.axvline(np.mean(residual), color=model_color, linewidth=1.5, linestyle="-", label=f"mean={10 ** np.mean(residual):.2f}x")
        ax3.set_xlabel("log10(pred realized tau / teacher realized tau)", fontsize=10)
        ax3.set_ylabel("Density", fontsize=10)
        ax3.set_title("Realized-Time Error Distribution", fontsize=12)
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.2)
    else:
        _annotate_center(ax3, "Realized-Time Error Distribution", ["Distribution unavailable"])

    ax4 = fig.add_subplot(gs[1, 1])
    if distribution_available:
        ideal = np.asarray([0.68, 0.95], dtype=np.float64)
        observed = np.asarray(
            [
                realized_distribution.get("coverage_68", np.nan),
                realized_distribution.get("coverage_95", np.nan),
            ],
            dtype=np.float64,
        )
        x = np.arange(2)
        width = 0.32
        bars_ideal = ax4.bar(x - width / 2, ideal, width=width, color="0.75", alpha=0.85, label="Ideal")
        bars_obs = ax4.bar(x + width / 2, observed, width=width, color=model_color, alpha=0.85, label=args.model_label)
        for bars in [bars_ideal, bars_obs]:
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width() / 2, height + 0.015, f"{height:.2f}", ha="center", va="bottom", fontsize=9)
        summary_lines = [
            f"NLL = {realized_distribution.get('nll', float('nan')):.3f}",
            f"PIT mean = {realized_distribution.get('pit_mean', float('nan')):.3f}",
            f"PIT var = {realized_distribution.get('pit_var', float('nan')):.3f}",
            f"PIT KS = {realized_distribution.get('pit_ks', float('nan')):.3f}",
            f"mean log sigma = {realized_distribution.get('mean_log_sigma', float('nan')):.3f}",
            f"source = {realized_distribution.get('prediction_source', time_heads.get('realized_tau_source', 'unknown'))}",
        ]
        ax4.text(
            0.98,
            0.98,
            "\n".join(summary_lines),
            transform=ax4.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88, "edgecolor": "0.6"},
        )
        ax4.set_xticks(x)
        ax4.set_xticklabels(["68% interval", "95% interval"], fontsize=10)
        ax4.set_ylim(0.0, 1.15)
        ax4.set_ylabel("Coverage", fontsize=10)
        ax4.set_title("Interval Calibration Summary", fontsize=12)
        ax4.legend(fontsize=9, loc="lower left")
        ax4.grid(True, alpha=0.2, axis="y")
    else:
        _annotate_center(
            ax4,
            "Interval Calibration Summary",
            [
                "Realized-time distribution unavailable",
                f"reason: {realized_distribution.get('reason', 'unknown')}",
                f"source: {time_heads.get('realized_tau_source', 'unknown')}",
            ],
        )

    ax5 = fig.add_subplot(gs[2, 0])
    if distribution_available and pit.size:
        bins = np.linspace(0.0, 1.0, 11)
        ax5.hist(pit, bins=bins, density=True, color=model_color, alpha=0.65, edgecolor="white")
        ax5.axhline(1.0, color=kmc_color, linewidth=2, linestyle="--", label="Uniform")
        ax5.set_xlabel("PIT", fontsize=10)
        ax5.set_ylabel("Density", fontsize=10)
        ax5.set_xlim(0.0, 1.0)
        ax5.set_title(
            f"PIT Histogram\nmean={realized_distribution.get('pit_mean', float('nan')):.3f}, "
            f"var={realized_distribution.get('pit_var', float('nan')):.3f}, "
            f"KS={realized_distribution.get('pit_ks', float('nan')):.3f}",
            fontsize=12,
        )
        ax5.legend(fontsize=9)
        ax5.grid(True, alpha=0.2)
    else:
        _annotate_center(
            ax5,
            "PIT Histogram",
            [
                "No realized-time distribution available yet",
                "This panel will populate once the new head is trained",
            ],
        )

    ax6 = fig.add_subplot(gs[2, 1])
    epochs = np.asarray(val_metrics.get("epochs", []), dtype=np.int32)
    real_tau_nll = np.asarray(val_metrics.get("real_tau_nll", []), dtype=np.float64)
    real_tau_log_mae = np.asarray(val_metrics.get("real_tau_log_mae", []), dtype=np.float64)
    tau_log_mae = np.asarray(val_metrics.get("tau_log_mae", []), dtype=np.float64)
    has_realized_val_curve = epochs.size > 0 and (
        np.any(np.isfinite(real_tau_nll)) or np.any(np.isfinite(real_tau_log_mae))
    )
    has_expected_val_curve = epochs.size > 0 and np.any(np.isfinite(tau_log_mae))
    if has_realized_val_curve:
        ax6b = ax6.twinx()
        legend_handles = []
        legend_labels = []
        if np.any(np.isfinite(real_tau_nll)):
            line = ax6.plot(epochs, real_tau_nll, "o-", color=model_color, markersize=4, label="real_tau_nll")
            legend_handles.extend(line)
            legend_labels.append("real_tau_nll")
        if np.any(np.isfinite(real_tau_log_mae)):
            line = ax6b.plot(epochs, real_tau_log_mae, "s-", color="tab:blue", markersize=4, label="real_tau_log_mae")
            legend_handles.extend(line)
            legend_labels.append("real_tau_log_mae")
        if np.any(np.isfinite(tau_log_mae)):
            line = ax6b.plot(epochs, tau_log_mae, "^-", color="tab:orange", markersize=4, label="tau_log_mae")
            legend_handles.extend(line)
            legend_labels.append("tau_log_mae")
        latest_cov68 = _latest_finite(val_metrics.get("real_tau_cov68", []))
        latest_pit_ks = _latest_finite(val_metrics.get("real_tau_pit_ks", []))
        extra_lines = []
        if latest_cov68 is not None:
            extra_lines.append(f"latest cov68 = {latest_cov68:.3f}")
        if latest_pit_ks is not None:
            extra_lines.append(f"latest pit_ks = {latest_pit_ks:.3f}")
        if extra_lines:
            ax6.text(
                0.98,
                0.98,
                "\n".join(extra_lines),
                transform=ax6.transAxes,
                ha="right",
                va="top",
                fontsize=10,
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88, "edgecolor": "0.6"},
            )
        ax6.set_xlabel("Epoch", fontsize=10)
        ax6.set_ylabel("real_tau_nll", fontsize=10, color=model_color)
        ax6b.set_ylabel("log-MAE", fontsize=10)
        ax6.set_title("Validation Curves: Expected vs Realized Time", fontsize=12)
        ax6.grid(True, alpha=0.2)
        ax6.legend(legend_handles, legend_labels, fontsize=9, loc="upper left")
    elif has_expected_val_curve:
        ax6.plot(epochs, tau_log_mae, "^-", color="tab:orange", markersize=4, label="tau_log_mae")
        ax6.text(
            0.98,
            0.98,
            "Realized validation metrics are not in this log yet\nshowing tau_exp baseline only",
            transform=ax6.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88, "edgecolor": "0.6"},
        )
        ax6.set_xlabel("Epoch", fontsize=10)
        ax6.set_ylabel("tau_exp log-MAE", fontsize=10)
        ax6.set_title("Validation Curve: Expected-Time Baseline Only", fontsize=12)
        ax6.legend(fontsize=9, loc="upper left")
        ax6.grid(True, alpha=0.2)
    else:
        _annotate_center(
            ax6,
            "Validation Curves: Expected vs Realized Time",
            [
                "Validation realized-time metrics are not in the log yet",
                "This panel fills after the first VAL step with the new head",
            ],
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved realized-time diagnostics to", output_path)


if __name__ == "__main__":
    main()