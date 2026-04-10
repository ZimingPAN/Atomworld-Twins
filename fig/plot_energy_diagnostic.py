"""
Energy diagnostic: is ΔE prediction weakness systematic?

Compares v22 eval on val split (400 samples) vs train split (2000 samples).
Produces: energy_diagnostic.png
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = Path(__file__).resolve().parent

# ── Load eval data ──

def load_eval(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)

val_path = ROOT / "dreamer4-main" / "results" / "macro_dreamer_kmc_baseline" / "eval_full_samples.json"
train_path = ROOT / "dreamer4-main" / "results" / "macro_dreamer_kmc_baseline" / "eval_train_split.json"

val_eval = load_eval(val_path)

# Train split may not exist yet
try:
    train_eval = load_eval(train_path)
    has_train = True
except FileNotFoundError:
    train_eval = None
    has_train = False
    print("WARNING: train split eval not found, plotting val-only diagnostic")

def extract_energy_arrays(eval_data: dict):
    samples = eval_data["all_samples"]
    pred_reward = np.array([s["predicted_reward_sum"] for s in samples])
    true_reward = np.array([s["traditional_kmc_reward_sum"] for s in samples])
    pred_de = np.array([s["predicted_delta_e"] for s in samples])
    true_de = np.array([s["traditional_kmc_delta_e"] for s in samples])
    pred_tau = np.array([s["predicted_tau"] for s in samples])
    true_tau = np.array([s["traditional_kmc_expected_tau"] for s in samples])
    return pred_reward, true_reward, pred_de, true_de, pred_tau, true_tau

val_arrays = extract_energy_arrays(val_eval)
if has_train:
    train_arrays = extract_energy_arrays(train_eval)

# ── Compute conditional metrics ──

def conditional_metrics(pred_reward, true_reward, true_de):
    zero_mask = np.abs(true_de) < 1e-6
    nonzero_mask = ~zero_mask
    n_zero = zero_mask.sum()
    n_nonzero = nonzero_mask.sum()

    metrics = {
        "n_total": len(true_de),
        "n_zero_de": int(n_zero),
        "n_nonzero_de": int(n_nonzero),
        "zero_de_frac": n_zero / len(true_de),
    }

    if n_zero > 0:
        metrics["zero_de_pred_reward_mean"] = float(np.mean(np.abs(pred_reward[zero_mask])))
        metrics["zero_de_pred_reward_std"] = float(np.std(pred_reward[zero_mask]))
        metrics["zero_de_true_reward_mean"] = float(np.mean(np.abs(true_reward[zero_mask])))
    if n_nonzero > 0:
        nz_pred = pred_reward[nonzero_mask]
        nz_true = true_reward[nonzero_mask]
        metrics["nonzero_de_pred_mean"] = float(np.mean(nz_pred))
        metrics["nonzero_de_true_mean"] = float(np.mean(nz_true))
        metrics["nonzero_de_ratio"] = float(np.mean(nz_pred) / np.mean(nz_true)) if np.mean(nz_true) != 0 else 0
        metrics["nonzero_de_corr"] = float(np.corrcoef(nz_pred, nz_true)[0, 1])
        metrics["nonzero_de_mae"] = float(np.mean(np.abs(nz_pred - nz_true)))

    return metrics

val_metrics = conditional_metrics(*val_arrays[:3])
print("\n=== VAL Split Conditional Metrics ===")
for k, v in val_metrics.items():
    print(f"  {k}: {v}")

if has_train:
    train_metrics = conditional_metrics(*train_arrays[:3])
    print("\n=== TRAIN Split Conditional Metrics ===")
    for k, v in train_metrics.items():
        print(f"  {k}: {v}")

# ═══════════════════════════════════════════════════════════
# FIGURE
# ═══════════════════════════════════════════════════════════

n_rows = 3 if has_train else 2
fig = plt.figure(figsize=(16, 6 * n_rows))
fig.suptitle(
    "Energy (ΔE) Prediction Diagnostic — v22 (macro_dreamer_kmc_baseline)\n"
    "Is the energy underestimate systematic?",
    fontsize=14, fontweight="bold", y=0.99,
)

gs = fig.add_gridspec(n_rows, 2, hspace=0.4, wspace=0.3,
                       top=0.93, bottom=0.05, left=0.08, right=0.95)

# ── Row 1: Val scatter, colored by zero/nonzero dE ──

ax1 = fig.add_subplot(gs[0, 0])
pred_r, true_r, pred_de, true_de, pred_tau, true_tau = val_arrays
zero_mask = np.abs(true_de) < 1e-6
nonzero_mask = ~zero_mask

ax1.scatter(true_r[zero_mask], pred_r[zero_mask],
            alpha=0.5, s=20, c="red", label=f"zero-dE ({zero_mask.sum()})", zorder=3)
ax1.scatter(true_r[nonzero_mask], pred_r[nonzero_mask],
            alpha=0.5, s=20, c="tab:blue", label=f"nonzero-dE ({nonzero_mask.sum()})", zorder=3)
lims = [min(true_r.min(), pred_r.min()) - 0.1, max(true_r.max(), pred_r.max()) + 0.1]
ax1.plot(lims, lims, "k--", alpha=0.4, linewidth=1)
ax1.set_xlabel("True Reward (KMC)", fontsize=10)
ax1.set_ylabel("Predicted Reward", fontsize=10)
ax1.set_title("VAL: Pred vs True Reward\n(colored by zero/nonzero dE)", fontsize=11)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# ── Row 1 right: Conditional box plots ──

ax2 = fig.add_subplot(gs[0, 1])
zero_errors = pred_r[zero_mask] - true_r[zero_mask] if zero_mask.any() else np.array([])
nonzero_errors = pred_r[nonzero_mask] - true_r[nonzero_mask] if nonzero_mask.any() else np.array([])

data_box = []
labels_box = []
if len(zero_errors) > 0:
    data_box.append(zero_errors)
    labels_box.append(f"zero-dE\n(n={len(zero_errors)})")
if len(nonzero_errors) > 0:
    data_box.append(nonzero_errors)
    labels_box.append(f"nonzero-dE\n(n={len(nonzero_errors)})")

bp = ax2.boxplot(data_box, tick_labels=labels_box, patch_artist=True, showmeans=True,
                 meanprops=dict(marker='D', markerfacecolor='black', markersize=6))
colors = ["red", "tab:blue"]
for patch, c in zip(bp["boxes"], colors[:len(data_box)]):
    patch.set_facecolor(c)
    patch.set_alpha(0.3)

for i, d in enumerate(data_box):
    ax2.text(i+1, np.mean(d) + 0.02, f"mean={np.mean(d):.4f}\nstd={np.std(d):.4f}",
             ha="center", fontsize=9, fontweight="bold")

ax2.axhline(0, color="black", linewidth=1, linestyle="-", alpha=0.5)
ax2.set_ylabel("Prediction Error (pred - true)", fontsize=10)
ax2.set_title("VAL: Error by Zero/Nonzero dE", fontsize=11)
ax2.grid(True, alpha=0.3, axis="y")

# ── Row 2: Histograms of true vs predicted reward for each subset ──

ax3 = fig.add_subplot(gs[1, 0])
if nonzero_mask.any():
    bins = np.linspace(min(true_r[nonzero_mask].min(), pred_r[nonzero_mask].min()),
                       max(true_r[nonzero_mask].max(), pred_r[nonzero_mask].max()), 30)
    ax3.hist(true_r[nonzero_mask], bins=bins, alpha=0.5, color="tab:blue", label="True (KMC)")
    ax3.hist(pred_r[nonzero_mask], bins=bins, alpha=0.5, color="tab:orange", label="Predicted")
    ax3.axvline(np.mean(true_r[nonzero_mask]), color="tab:blue", linestyle="--", linewidth=2,
                label=f"True mean={np.mean(true_r[nonzero_mask]):.4f}")
    ax3.axvline(np.mean(pred_r[nonzero_mask]), color="tab:orange", linestyle="--", linewidth=2,
                label=f"Pred mean={np.mean(pred_r[nonzero_mask]):.4f}")
ax3.set_xlabel("Reward", fontsize=10)
ax3.set_ylabel("Count", fontsize=10)
ax3.set_title("Nonzero-dE Segments: Reward Distribution", fontsize=11)
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

ax4 = fig.add_subplot(gs[1, 1])
# Predicted reward on zero-dE segments (should be ~0 ideally)
if zero_mask.any():
    ax4.hist(pred_r[zero_mask], bins=30, alpha=0.6, color="red", label="Pred on zero-dE")
    ax4.axvline(0, color="black", linewidth=2, linestyle="-", alpha=0.5, label="Ideal = 0")
    ax4.axvline(np.mean(pred_r[zero_mask]), color="darkred", linewidth=2, linestyle="--",
                label=f"Mean = {np.mean(pred_r[zero_mask]):.4f}")
ax4.set_xlabel("Predicted Reward", fontsize=10)
ax4.set_ylabel("Count", fontsize=10)
ax4.set_title("Zero-dE Segments: Predicted Reward\n(should be ~0)", fontsize=11)
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

# ── Row 3 (if train data exists): Train vs Val comparison ──

if has_train:
    train_pred_r, train_true_r, train_pred_de, train_true_de, _, _ = train_arrays
    train_zero = np.abs(train_true_de) < 1e-6
    train_nonzero = ~train_zero

    ax5 = fig.add_subplot(gs[2, 0])
    ax5.scatter(train_true_r[train_zero], train_pred_r[train_zero],
                alpha=0.3, s=10, c="red", label=f"zero-dE ({train_zero.sum()})", zorder=3)
    ax5.scatter(train_true_r[train_nonzero], train_pred_r[train_nonzero],
                alpha=0.3, s=10, c="tab:blue", label=f"nonzero-dE ({train_nonzero.sum()})", zorder=3)
    lims_t = [min(train_true_r.min(), train_pred_r.min()) - 0.1,
              max(train_true_r.max(), train_pred_r.max()) + 0.1]
    ax5.plot(lims_t, lims_t, "k--", alpha=0.4, linewidth=1)
    ax5.set_xlabel("True Reward (KMC)", fontsize=10)
    ax5.set_ylabel("Predicted Reward", fontsize=10)
    ax5.set_title("TRAIN: Pred vs True Reward (2000 samples)\n(colored by zero/nonzero dE)", fontsize=11)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)

    # Comparison bar chart: train vs val metrics
    ax6 = fig.add_subplot(gs[2, 1])
    metric_names = ["zero-dE\npred mean", "nonzero-dE\nratio", "nonzero-dE\ncorr", "nonzero-dE\nMAE"]
    val_vals = [
        val_metrics.get("zero_de_pred_reward_mean", 0),
        val_metrics.get("nonzero_de_ratio", 0),
        val_metrics.get("nonzero_de_corr", 0),
        val_metrics.get("nonzero_de_mae", 0),
    ]
    train_vals = [
        train_metrics.get("zero_de_pred_reward_mean", 0),
        train_metrics.get("nonzero_de_ratio", 0),
        train_metrics.get("nonzero_de_corr", 0),
        train_metrics.get("nonzero_de_mae", 0),
    ]

    x = np.arange(len(metric_names))
    w = 0.35
    bars1 = ax6.bar(x - w/2, val_vals, w, label=f"Val ({val_metrics['n_total']})", alpha=0.8, color="tab:blue")
    bars2 = ax6.bar(x + w/2, train_vals, w, label=f"Train ({train_metrics['n_total']})", alpha=0.8, color="tab:orange")
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.3f}",
                     ha="center", va="bottom", fontsize=8)
    ax6.set_xticks(x)
    ax6.set_xticklabels(metric_names, fontsize=9)
    ax6.set_title("Train vs Val: Energy Prediction Metrics", fontsize=11)
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3, axis="y")

out_path = FIG_DIR / "energy_diagnostic.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out_path}")
plt.close(fig)
