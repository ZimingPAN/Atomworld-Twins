#!/usr/bin/env python3
"""Generate paper figures for the local NeurIPS draft.

This script is intentionally stored inside the local NeurIPS formatting folder.
The whole folder is not synced to the server.
"""

from __future__ import annotations

import json
import csv
import contextlib
import io
import math
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
import matplotlib.patheffects as path_effects
from matplotlib.collections import LineCollection
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.text import Text
from matplotlib.textpath import TextPath
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, NullLocator


def _find_repo_root(start: Path) -> Path:
    for parent in (start, *start.parents):
        if (parent / "dreamer4-main").is_dir():
            return parent
    return start.parents[1]


ROOT = _find_repo_root(Path(__file__).resolve())
OUT = Path(__file__).resolve().parent
FIG_DIR = OUT / "fig"
FIGTEST_DIR = OUT / "figtest"
OFFICIAL_FIGURES = {
    "fig1_key_evolution_budget_randomness.pdf",
    "fig2_main_results.pdf",
    "fig3_closed_loop_results.pdf",
    "fig4_speedup_benchmark.pdf",
}
RESULT = ROOT / "dreamer4-main" / "results" / "dreamer_macro_edit_v26_realized_qtrain"
V52_RESULT = ROOT / "dreamer4-main" / "results" / "dreamer_macro_edit_v52_planner_selected_duration_covered_aug_1024"
MULTICASE_FIG3_TEMP_SWEEP = V52_RESULT / "multicase_fig3_temp_sweep_250_500_s32.json"
FIG4_CASE_GRID = V52_RESULT / "benchmark_fig4_case_grid_t663_773_cu2_l20_60_k8_s32_r2.json"
FIG4_CASE_GRID_END_TO_END = V52_RESULT / "benchmark_fig4_case_grid_cu005_end_to_end_t663_773_l20_60_k8_s8_r2.json"
FIG4_CASE_GRID_END_TO_END_LARGE = V52_RESULT / "benchmark_fig4_case_grid_cu005_end_to_end_t263_373_l50_400_k8_s1_r1_checkpointed.json"
FIG4_CASE_GRID_END_TO_END_SELECTED_NO100 = V52_RESULT / "benchmark_fig4_case_grid_cu005_end_to_end_t263_373_l200_600_k8_s1_r1_selected_preview_no100.json"
FIG2_TEMPERATURE_CASES = V52_RESULT / "fig2_temperature_cases_t263_373_s64_paired.json"
FIG2_HIGH_TEMPERATURE_CASES = V52_RESULT / "fig2_temperature_cases_t663_773_s64_paired.json"
CLOSED_LOOP = ROOT / "dreamer4-main" / "results" / "neurips_closed_loop_onpolicy_matrix"
FIXEDK_MATRIX = ROOT / "dreamer4-main" / "results" / "neurips_fixedk_matrix"
MULTIK_CONSTRAINTS = ROOT / "dreamer4-main" / "results" / "neurips_closed_loop_multik_constraints"
MULTIK_DURATION_SEEDS = ROOT / "dreamer4-main" / "results" / "neurips_multik_duration_seed_stability"
FIG3_CU_TEMP_ABLATION = ROOT / "dreamer4-main" / "results" / "neurips_fig3_cu_temp_ablation"
SEGMENT_CACHE = ROOT / "dreamer4-main" / "results" / "kmc_teacher_dreamer_macro_wm" / "segments.pt"
MULTIK_LONG = ROOT / "dreamer4-main" / "results" / "dreamer_macro_edit_v32_noaug_warm_1024" / "eval_long_trajectory_500.json"
REMOVED_CRITICAL_BACKBONE_SOURCE = FIGTEST_DIR / "fig1_atomworld_mirror_powerpoint_recreation.pdf"
REMOVED_CRITICAL_BACKBONE_CROP_LOWER_Y = 246
REMOVED_CRITICAL_BACKBONE_CROP_HEIGHT = 294

BLUE = "#2f67ad"
LIGHT_BLUE = "#e9f1fb"
TEAL = "#0f7482"
LIGHT_TEAL = "#e6f4f4"
ORANGE = "#bd6a21"
LIGHT_ORANGE = "#f7ead8"
GREEN = "#2d8b57"
LIGHT_GREEN = "#e7f3ec"
PURPLE = "#6655c8"
LIGHT_PURPLE = "#eeeafd"
RED = "#c34b4b"
GREY = "#7d8796"
LIGHT_GREY = "#eef1f5"
DARK = "#1f2933"
SOFT_BLUE = "#aecce4"
SOFT_BLUE_DARK = "#83b7cf"
SOFT_PINK = "#efbcb8"
SOFT_PINK_DARK = "#deb4b5"
SOFT_GREEN = "#cce4c2"
SOFT_GREEN_DARK = "#bbd4bf"
SOFT_YELLOW = "#eae0ab"
SOFT_YELLOW_DARK = "#dcd2a1"
SOFT_NEUTRAL = "#e9e5e6"
SOFT_EDGE = "#6f7f8b"
TEMP_CASE_COLORS = ["#aecce4", "#efbcb8", "#cce4c2", "#eae0ab"]
TEMP_CASE_DARK = ["#83b7cf", "#deb4b5", "#7fab87", "#c5ac45"]
FIG2_PANEL_A_COLORS = [SOFT_BLUE, SOFT_GREEN]
FIG2_PANEL_C_COLORS = [SOFT_BLUE, SOFT_PINK]
FIG2_PANEL_C_ACCENT = SOFT_PINK_DARK
REF_YELLOW = "#FEDF91"
REF_BLUE = "#A7C6E6"
REF_ORANGE = "#EAA350"
REF_PURPLE = "#8984BE"
FIG1_PREVIEW_FONT = "Arial"
FIG2_FONT = "Arial"
FIG2_FONT_RC = {
    "font.family": FIG2_FONT,
    "mathtext.fontset": "custom",
    "mathtext.rm": FIG2_FONT,
    "mathtext.it": f"{FIG2_FONT}:italic",
    "mathtext.bf": f"{FIG2_FONT}:bold",
}


def setup():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    FIGTEST_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 180,
        }
    )


def _figure_output_path(filename: str) -> Path:
    folder = FIG_DIR if filename in OFFICIAL_FIGURES else FIGTEST_DIR
    return folder / filename


def _cube_size_label(edge: int) -> str:
    return f"{edge}\N{SUPERSCRIPT THREE}"


def clean_axis(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def _emphasize_axis_artists(
    ax,
    *,
    font_scale: float = 1.14,
    title_scale: float = 1.02,
    line_scale: float = 1.32,
    marker_area_scale: float = 1.95,
    marker_size_scale: float = 1.28,
):
    title_ids = {id(ax.title), id(ax._left_title), id(ax._right_title)}
    for text in ax.findobj(match=Text):
        scale = title_scale if id(text) in title_ids else font_scale
        text.set_fontsize(text.get_fontsize() * scale)
    for line in ax.lines:
        line.set_linewidth(line.get_linewidth() * line_scale)
        if line.get_marker() not in (None, "", "None", "none", " "):
            line.set_markersize(line.get_markersize() * marker_size_scale)
    for collection in ax.collections:
        sizes = collection.get_sizes()
        if len(sizes):
            collection.set_sizes(sizes * marker_area_scale)
        linewidths = collection.get_linewidths()
        if len(linewidths):
            collection.set_linewidths(linewidths * line_scale)
    for patch in ax.patches:
        patch.set_linewidth(patch.get_linewidth() * line_scale)


def box(ax, xy, wh, text, fc, ec, lw=1.0, size=8.5, weight="normal", radius=0.02):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        color=DARK,
        fontsize=size,
        fontweight=weight,
        linespacing=1.05,
    )
    return patch


def arrow(ax, start, end, color=DARK, lw=1.1, ms=10, style="-|>", rad=0.0, alpha=1.0):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=ms,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        alpha=alpha,
    )
    ax.add_patch(patch)
    return patch


def chip(ax, x, y, text, fc, ec, w=0.11):
    box(ax, (x, y), (w, 0.055), text, fc, ec, lw=0.8, size=7.1, radius=0.012)


def draw_lattice(ax, cx, cy, scale=1.0, sparse=False, highlight=None, spurious=False):
    dx, dy = 0.026 * scale, 0.027 * scale
    rows, cols = 5, 6
    colors = []
    for r in range(rows):
        for c in range(cols):
            colors.append("#dce4ed")
    cu_idx = [2, 9, 17, 24]
    vac_idx = [14]
    for i in cu_idx:
        colors[i] = "#d59b2a"
    for i in vac_idx:
        colors[i] = "#f7f7f7"
    if sparse:
        for i in [15, 20]:
            colors[i] = "#f7f7f7" if i == 15 else "#d59b2a"
    if spurious:
        for i in [4, 7, 21, 28]:
            colors[i] = "#d59b2a"
    pts = []
    for r in range(rows):
        for c in range(cols):
            x = cx + (c - (cols - 1) / 2) * dx + (r % 2) * dx * 0.35
            y = cy + ((rows - 1) / 2 - r) * dy
            pts.append((x, y))
    for i, (x, y) in enumerate(pts):
        ax.add_patch(Circle((x, y), 0.0075 * scale, fc=colors[i], ec="#8b98a8", lw=0.65))
    for i in highlight or []:
        x, y = pts[i]
        ax.add_patch(Circle((x, y), 0.011 * scale, fc="none", ec=RED, lw=0.9))


def _load_segment_cache():
    payload = torch.load(SEGMENT_CACHE, map_location="cpu", weights_only=False)
    samples = []
    for split in ("train", "val"):
        samples.extend(payload[split])
    return payload, samples


def _cache_reward_horizon_arrays():
    payload, samples = _load_segment_cache()
    rewards = np.asarray([float(s["reward_sum"]) for s in samples], dtype=np.float64)
    horizons = np.asarray([int(s.get("horizon_k", payload["signature"].get("segment_k", 4))) for s in samples], dtype=np.int64)
    positive = rewards[rewards > 1e-6]
    decisive_threshold = float(np.quantile(positive, 0.95)) if positive.size else float("inf")
    return payload, rewards, horizons, decisive_threshold


def _load_multik_long():
    with MULTIK_LONG.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    segments = payload["segments"]
    ks = np.asarray([int(s["segment_k"]) for s in segments], dtype=np.int64)
    rewards = np.asarray([float(s["traditional_kmc_reward_sum"]) for s in segments], dtype=np.float64)
    pred_rewards = np.asarray([float(s["predicted_reward_sum"]) for s in segments], dtype=np.float64)
    tau_true = np.asarray([float(s["traditional_kmc_expected_tau"]) for s in segments], dtype=np.float64)
    tau_pred = np.asarray([float(s["predicted_expected_tau"]) for s in segments], dtype=np.float64)
    event_edges = np.concatenate([[0], np.cumsum(ks)])
    return payload, ks, rewards, pred_rewards, tau_true, tau_pred, event_edges


def _draw_backbone_inset(ax):
    clean_axis(ax)
    xs = np.linspace(0.12, 0.92, 17)
    ys = 0.70 + 0.035 * np.sin(np.linspace(0, 4 * np.pi, len(xs)))
    ax.plot(xs, ys, color="#c7ced9", lw=0.9, zorder=1)
    for x, y in zip(xs, ys):
        ax.add_patch(Circle((x, y), 0.011, fc="#c7ced9", ec="white", lw=0.35, zorder=2))
    key_idx = [0, 5, 10, 16]
    bx = [xs[i] for i in key_idx]
    by = [0.26, 0.30, 0.23, 0.28]
    for idx in key_idx:
        ax.plot([xs[idx], xs[idx]], [ys[idx] - 0.03, 0.38], color="#d4dbe6", lw=0.65, ls="--")
    for i in range(len(bx) - 1):
        arrow(ax, (bx[i] + 0.045, by[i]), (bx[i + 1] - 0.045, by[i + 1]), color=TEAL, lw=1.6, ms=9)
    for j, (x, y) in enumerate(zip(bx, by)):
        ax.add_patch(Ellipse((x, y), 0.070, 0.058, fc=LIGHT_TEAL, ec=TEAL, lw=1.1))
        ax.text(x, y, rf"$X_{{\tau_{j}}}$", ha="center", va="center", fontsize=7.2, color=TEAL, weight="bold")
    ax.text(0.51, 0.88, "micro-event replay", ha="center", fontsize=7.4, color=GREY, weight="bold")
    ax.text(0.51, 0.08, "macro backbone", ha="center", fontsize=7.4, color=TEAL, weight="bold")


def removed_critical_backbone_figure():
    if not REMOVED_CRITICAL_BACKBONE_SOURCE.exists():
        raise FileNotFoundError(f"Missing removed critical-backbone source: {REMOVED_CRITICAL_BACKBONE_SOURCE}")
    target = _figure_output_path("removed_fig1_critical_backbone.pdf")
    gs = shutil.which("gs")
    if gs is None:
        raise RuntimeError("Ghostscript `gs` is required to crop the removed critical-backbone figure.")
    subprocess.run(
        [
            gs,
            "-q",
            "-o",
            str(target),
            "-sDEVICE=pdfwrite",
            "-dCompatibilityLevel=1.5",
            "-dPDFSETTINGS=/prepress",
            "-dDEVICEWIDTHPOINTS=960",
            f"-dDEVICEHEIGHTPOINTS={REMOVED_CRITICAL_BACKBONE_CROP_HEIGHT}",
            "-dFIXEDMEDIA",
            "-c",
            f"<</PageOffset [0 -{REMOVED_CRITICAL_BACKBONE_CROP_LOWER_Y}]>> setpagedevice",
            "-f",
            str(REMOVED_CRITICAL_BACKBONE_SOURCE),
        ],
        check=True,
    )


def fig2():
    fig, ax = plt.subplots(figsize=(6.8, 2.9))
    clean_axis(ax)

    ax.text(0.15, 0.92, "teacher supervision", ha="center", fontsize=8.2, weight="bold", color=ORANGE)
    ax.text(0.51, 0.92, "latent macro world model", ha="center", fontsize=8.2, weight="bold", color=BLUE)
    ax.text(0.85, 0.92, "macro outputs", ha="center", fontsize=8.2, weight="bold", color=GREEN)

    box(ax, (0.05, 0.64), (0.18, 0.14), "Teacher\nsimulator", LIGHT_ORANGE, ORANGE, lw=1.0, size=8.7, weight="bold")
    ax.text(0.14, 0.56, r"$X_t \rightarrow X_{t+1}\rightarrow \cdots \rightarrow X_{t+k}$",
            ha="center", va="center", fontsize=8.0, color=ORANGE)
    for x in np.linspace(0.08, 0.20, 5):
        ax.add_patch(Circle((x, 0.51), 0.008, fc=ORANGE, ec=ORANGE, alpha=0.75))

    table_x, table_y, table_w, table_h = 0.045, 0.215, 0.185, 0.205
    ax.add_patch(
        FancyBboxPatch(
            (table_x, table_y),
            table_w,
            table_h,
            boxstyle="round,pad=0.006,rounding_size=0.010",
            facecolor="#fffaf3",
            edgecolor=ORANGE,
            linewidth=0.75,
        )
    )
    for frac in (1 / 3, 2 / 3):
        ax.plot([table_x, table_x + table_w], [table_y + frac * table_h, table_y + frac * table_h],
                color="#d99a62", lw=0.45)
    ax.plot([table_x + table_w / 2, table_x + table_w / 2], [table_y, table_y + table_h],
            color="#d99a62", lw=0.45)
    labels = [
        (r"$X_{t+k}$", r"$\Delta X$"),
        (r"$\mathcal{C}_k$", r"$\tau_{\rm exp}$"),
        (r"$\tau_{\rm real}$", r"$s_{\rm path}$"),
    ]
    row_centers = [table_y + table_h * (5 / 6), table_y + table_h * 0.5, table_y + table_h * (1 / 6)]
    col_centers = [table_x + table_w * 0.25, table_x + table_w * 0.75]
    for row_y, row in zip(row_centers, labels):
        for col_x, text in zip(col_centers, row):
            ax.text(col_x, row_y, text, ha="center", va="center", fontsize=7.2, color=DARK, weight="bold")

    box(ax, (0.32, 0.68), (0.13, 0.12), "encoder\n" + r"$X_t \mapsto z_t$", LIGHT_BLUE, BLUE, size=7.9, weight="bold")
    box(ax, (0.32, 0.47), (0.13, 0.12), "path latent\n" + r"$h_t$", LIGHT_PURPLE, PURPLE, size=7.9, weight="bold")
    box(ax, (0.52, 0.57), (0.16, 0.15), "macro\ntransition\n" + r"$(z_t,h_t)\mapsto \hat z_{t+k}$",
        LIGHT_BLUE, BLUE, size=7.5, weight="bold")
    box(ax, (0.49, 0.33), (0.12, 0.12), "edit head\n" + r"$\Delta\hat X$", LIGHT_GREEN, GREEN, size=7.6, weight="bold")
    box(ax, (0.64, 0.33), (0.14, 0.12), "duration head\n" + r"$\hat\tau$", LIGHT_GREEN, GREEN, size=7.4, weight="bold")
    box(ax, (0.83, 0.59), (0.14, 0.18), r"$\Delta\hat X_{t:t+k}$" + "\n" + r"$\hat\tau_{t:t+k}$" + "\n" + r"$\hat X_{t+k}$",
        LIGHT_GREEN, GREEN, size=8.1, weight="bold")
    box(ax, (0.815, 0.23), (0.16, 0.11), "macro-step rollout\nalong backbone", "#e5f6f8", TEAL, size=7.0, weight="bold")

    arrow(ax, (0.23, 0.71), (0.32, 0.74), ORANGE, lw=1.1, ms=9)
    arrow(ax, (table_x + table_w, table_y + table_h * 0.58), (0.32, 0.53), ORANGE, lw=1.1, ms=9, style="-|>", rad=0.12, alpha=0.85)
    arrow(ax, (0.45, 0.74), (0.52, 0.65), BLUE, lw=1.2, ms=9)
    arrow(ax, (0.45, 0.53), (0.52, 0.62), PURPLE, lw=1.2, ms=9)
    arrow(ax, (0.60, 0.57), (0.55, 0.45), GREEN, lw=1.1, ms=9)
    arrow(ax, (0.62, 0.57), (0.70, 0.45), GREEN, lw=1.1, ms=9)
    arrow(ax, (0.68, 0.65), (0.83, 0.68), BLUE, lw=1.2, ms=9)
    arrow(ax, (0.61, 0.39), (0.83, 0.62), GREEN, lw=1.0, ms=8, rad=0.22)
    arrow(ax, (0.78, 0.39), (0.85, 0.59), GREEN, lw=1.0, ms=8, rad=-0.18)
    arrow(ax, (0.90, 0.59), (0.90, 0.34), TEAL, lw=1.2, ms=9)

    ax.plot([0.305, 0.805, 0.805, 0.305, 0.305], [0.18, 0.18, 0.84, 0.84, 0.18],
            color="#d7ddec", lw=0.7, ls="--")
    ax.text(0.555, 0.16, "posterior for training; prior for rollout", ha="center", va="top", fontsize=7.2, color=GREY)
    ax.text(0.50, 0.04, "event-driven atomic simulation is one teacher instantiation",
            ha="center", va="center", fontsize=7.3, color=GREY)

    fig.savefig(_figure_output_path("fig2_atomworld_overview.pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def fig3():
    fig, ax = plt.subplots(figsize=(3.45, 2.45))
    clean_axis(ax)

    panels = [
        ("current state", r"$X_t$", 0.25, 0.72),
        ("dense reconstruction", "unconstrained", 0.75, 0.72),
        ("projected sparse edit", r"$\Delta\hat X$", 0.25, 0.24),
        ("reachable support", r"$\mathcal{C}_k(X_t)$", 0.75, 0.24),
    ]
    for title, sub, x, y in panels:
        ax.text(x, y + 0.205, title, ha="center", va="bottom", fontsize=6.8, weight="bold", color=DARK)
        ax.text(x, y + 0.158, sub, ha="center", va="bottom", fontsize=6.8, color=DARK)
        ax.add_patch(FancyBboxPatch((x - 0.145, y - 0.095), 0.29, 0.19,
                                    boxstyle="round,pad=0.012,rounding_size=0.012",
                                    fc="#fbfcfe", ec="#d4dbe6", lw=0.8))

    draw_lattice(ax, 0.25, 0.72, scale=1.08)
    draw_lattice(ax, 0.75, 0.72, scale=1.08, spurious=True, highlight=[4, 7, 21, 28])
    draw_lattice(ax, 0.25, 0.24, scale=1.08, sparse=True, highlight=[15, 20])
    ax.add_patch(Ellipse((0.25, 0.24), 0.11, 0.095, fc="none", ec=GREEN, lw=1.1))
    draw_lattice(ax, 0.75, 0.24, scale=1.08)
    ax.add_patch(Ellipse((0.75, 0.24), 0.165, 0.135, fc="none", ec=TEAL, lw=1.1, ls="--"))

    arrow(ax, (0.405, 0.72), (0.595, 0.72), GREY, lw=0.9, ms=8)
    arrow(ax, (0.595, 0.24), (0.405, 0.24), GREY, lw=0.9, ms=8)

    ax.text(0.75, 0.535, "spurious distant edits", ha="center", va="center", fontsize=6.4, color=RED, weight="bold")
    ax.text(0.25, 0.065, "sparse + conserving", ha="center", va="center", fontsize=6.4, color=GREEN, weight="bold")
    ax.text(0.75, 0.065, "local reachability", ha="center", va="center", fontsize=6.4, color=TEAL, weight="bold")

    ax.set_ylim(0.0, 0.98)
    fig.savefig(_figure_output_path("fig3_reachability_sparse_edit.pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def fig4():
    fig, ax = plt.subplots(figsize=(6.8, 2.7))
    clean_axis(ax)

    xs = np.linspace(0.08, 0.48, 5)
    y = 0.68
    for i, x in enumerate(xs):
        ax.add_patch(Ellipse((x, y), 0.055, 0.064, fc=LIGHT_ORANGE, ec=ORANGE, lw=1.1))
        ax.text(x, y, rf"$X_{{t+{i}}}$" if i else r"$X_t$", ha="center", va="center", fontsize=8.2, color=ORANGE, weight="bold")
        if i < len(xs) - 1:
            arrow(ax, (x + 0.032, y), (xs[i + 1] - 0.032, y), ORANGE, lw=1.3, ms=9)
            ax.text((x + xs[i + 1]) / 2, 0.82, rf"$R_{i}$", ha="center", fontsize=8.2, color="#5d6878")
            ax.text((x + xs[i + 1]) / 2, 0.54, rf"$\Delta t_{i}$",
                    ha="center", fontsize=6.8, color="#5d6878")
            ax.text((x + xs[i + 1]) / 2, 0.49, rf"$\sim{{\rm Exp}}(R_{i})$",
                    ha="center", fontsize=6.6, color="#5d6878")
    ax.text(0.28, 0.91, r"$R_j = R(X_{t+j})$", ha="center", fontsize=7.8, color=GREY)
    ax.text(0.28, 0.40, "teacher micro-event path", ha="center", fontsize=7.4, color=GREY)

    box(ax, (0.60, 0.65), (0.17, 0.18), "expected\n" + r"$\tau_{\rm exp}=\sum 1/R_j$",
        LIGHT_ORANGE, ORANGE, size=7.7, weight="bold")
    box(ax, (0.80, 0.65), (0.17, 0.18), "realized\n" + r"$\tau_{\rm real}=\sum\Delta t_j$",
        "white", ORANGE, size=7.7, weight="bold")
    arrow(ax, (0.51, 0.69), (0.60, 0.74), ORANGE, lw=1.1, ms=8)
    realized_path = MplPath(
        [(0.51, 0.62), (0.60, 0.585), (0.72, 0.585), (0.80, 0.68)],
        [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4],
    )
    ax.add_patch(FancyArrowPatch(path=realized_path, arrowstyle="-|>", mutation_scale=8,
                                 linewidth=1.0, color=ORANGE, alpha=0.9))

    box(ax, (0.60, 0.21), (0.17, 0.16), "main head\n" + r"$\log\tau_{\rm exp}$",
        LIGHT_BLUE, BLUE, size=7.7, weight="bold")
    box(ax, (0.80, 0.21), (0.17, 0.16), "aux head\n" + r"${\rm LogNormal}$",
        LIGHT_PURPLE, PURPLE, size=7.7, weight="bold")
    arrow(ax, (0.685, 0.65), (0.685, 0.37), BLUE, lw=1.0, ms=8)
    arrow(ax, (0.885, 0.65), (0.885, 0.37), PURPLE, lw=1.0, ms=8)

    ax.text(0.50, 0.04, "time-aware macro world modeling keeps state transition and CTMC clock coupled",
            ha="center", va="bottom", fontsize=7.4, color=GREY)
    fig.savefig(_figure_output_path("fig4_continuous_time_duration.pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _panel_sparse_edit_metrics(ax, metrics: dict, *, title: str = "(a) Physically valid sparse edits"):
    vals_a = np.array(
        [
            1.0 - metrics["reachability_violation_rate"],
            metrics["changed_type_acc"],
        ],
        dtype=float,
    )
    labs_a = ["reachable\nedits", "changed-site\ntype acc."]
    colors_a = FIG2_PANEL_A_COLORS
    x_a = np.arange(len(vals_a))
    ax.bar(x_a, vals_a, color=colors_a, edgecolor=SOFT_EDGE, linewidth=0.55, alpha=0.98, width=0.58)
    for i, v in enumerate(vals_a):
        ax.text(i, min(v + 0.035, 1.05), f"{100*v:.1f}%", ha="center", va="bottom", fontsize=7.2)
    ax.set_xticks(x_a, labs_a)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("score")
    ax.set_title(title, loc="left", pad=5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#d9dee8", lw=0.5, alpha=0.8)


def _panel_sparse_edit_trajectory(
    ax,
    long: dict,
    metrics: dict,
    *,
    title: str = "(a) Physically valid sparse edits",
):
    segments = long.get("segments", [])
    if not segments:
        _panel_sparse_edit_metrics(ax, metrics, title=title)
        return
    teacher_counts = np.asarray([float(seg["traditional_changed_site_count"]) for seg in segments], dtype=float)
    model_counts = np.asarray([float(seg["projected_changed_count"]) for seg in segments], dtype=float)
    violations = np.asarray([float(bool(seg["reachability_violation"])) for seg in segments], dtype=float)
    x = np.arange(len(segments) + 1, dtype=float)
    teacher_path = np.concatenate([[0.0], np.cumsum(teacher_counts)])
    model_path = np.concatenate([[0.0], np.cumsum(model_counts)])
    count_ratio = model_path[-1] / teacher_path[-1] if teacher_path[-1] > 0 else np.nan
    reachable = 1.0 - violations.mean() if len(violations) else 1.0

    ax.plot(x, teacher_path, color=SOFT_BLUE_DARK, lw=1.55, label="teacher edits")
    ax.plot(x, model_path, color=SOFT_GREEN_DARK, lw=1.55, label="model edits")
    ax.scatter([x[-1]], [teacher_path[-1]], color=SOFT_BLUE_DARK, s=14, zorder=3)
    ax.scatter([x[-1]], [model_path[-1]], color=SOFT_GREEN_DARK, s=14, zorder=3)
    ax.set_xlabel("Macro Segment ID")
    ax.set_ylabel("Cumulative Changed Sites")
    ax.set_title(title, loc="left", pad=5)
    ax.text(
        0.97,
        0.06,
        f"reachable={100*reachable:.1f}%\ntype acc.={100*metrics['changed_type_acc']:.1f}%\ncount ratio={count_ratio:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.1,
        color=GREY,
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#d9dee8", lw=0.6),
    )
    ax.set_xlim(0, x[-1])
    ax.set_ylim(0, max(teacher_path.max(), model_path.max()) * 1.10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, which="major", color="#d9dee8", lw=0.5, alpha=0.8)
    ax.legend(frameon=False, fontsize=5.8, loc="upper left", handlelength=1.2, borderaxespad=0.2)


def _style_reference_line_axis(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.1)
    ax.grid(axis="y", which="major", color="black", linestyle=(0, (1.0, 3.0)), linewidth=0.82, alpha=0.72)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="both", which="major", direction="out", length=5.4, width=1.0, labelsize=8.9, colors="black")
    ax.tick_params(axis="both", which="minor", direction="out", length=3.8, width=0.9, colors="black")
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.set_axisbelow(True)
    ax.set_facecolor("white")
    ax.title.set_fontsize(9.4)
    ax.title.set_fontweight("bold")
    ax.xaxis.label.set_fontsize(9.8)
    ax.yaxis.label.set_fontsize(9.8)


def _style_reference_boxed_axis(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)
    ax.grid(axis="y", which="major", color="black", linestyle=(0, (1.0, 3.0)), linewidth=0.65, alpha=0.58)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="both", which="major", direction="out", length=4.6, width=0.95, colors="black")
    ax.tick_params(axis="both", which="minor", direction="out", length=3.0, width=0.8, colors="black")
    ax.set_axisbelow(True)


def _style_arial_axis_text(ax):
    text_artists = [
        ax.xaxis.label,
        ax.yaxis.label,
        ax.title,
        ax._left_title,
        ax._right_title,
        ax.xaxis.get_offset_text(),
        ax.yaxis.get_offset_text(),
        *ax.get_xticklabels(),
        *ax.get_yticklabels(),
        *ax.texts,
    ]
    legend = ax.get_legend()
    if legend is not None:
        text_artists.extend(legend.get_texts())
        if legend.get_title() is not None:
            text_artists.append(legend.get_title())
    for artist in text_artists:
        artist.set_fontfamily(FIG1_PREVIEW_FONT)
    for label in (ax.xaxis.label, ax.yaxis.label):
        label.set_fontweight("normal")


def _center_axis_title(ax, *, pad: float = 7.0):
    title_text = ax.get_title(loc="left") or ax.get_title(loc="center") or ax.get_title(loc="right")
    title_artist = ax._left_title if ax.get_title(loc="left") else ax.title
    fontsize = title_artist.get_fontsize()
    fontweight = title_artist.get_fontweight()
    ax.set_title("", loc="left")
    ax.set_title("", loc="right")
    ax.set_title(title_text, loc="center", pad=pad)
    ax.title.set_fontsize(fontsize)
    ax.title.set_fontweight(fontweight)


def _aligned_panel_title_specs(axes):
    specs = []
    for ax in axes:
        title_text = ax.get_title(loc="left") or ax.get_title(loc="center") or ax.get_title(loc="right")
        title_artist = ax._left_title if ax.get_title(loc="left") else ax.title
        specs.append((title_text, title_artist.get_fontsize(), title_artist.get_fontweight()))
        ax.set_title("", loc="left")
        ax.set_title("", loc="center")
        ax.set_title("", loc="right")
    return specs


def _draw_aligned_panel_titles(fig, axes, specs, *, gap: float = 0.014):
    title_y = min(0.985, max(ax.get_position().y1 for ax in axes) + gap)
    for ax, (title_text, fontsize, fontweight) in zip(axes, specs):
        bbox = ax.get_position()
        fig.text(
            0.5 * (bbox.x0 + bbox.x1),
            title_y,
            title_text,
            ha="center",
            va="bottom",
            fontsize=fontsize,
            fontweight=fontweight,
            fontfamily=FIG1_PREVIEW_FONT,
        )


def _fig1_preview_font_rc() -> dict[str, object]:
    return {
        "font.family": FIG1_PREVIEW_FONT,
        "font.sans-serif": [FIG1_PREVIEW_FONT],
        "mathtext.fontset": "custom",
        "mathtext.rm": FIG1_PREVIEW_FONT,
        "mathtext.it": f"{FIG1_PREVIEW_FONT}:italic",
        "mathtext.bf": f"{FIG1_PREVIEW_FONT}:bold",
    }


def _reference_line_marker_every(x: np.ndarray) -> int:
    return max(1, int(math.ceil(len(x) / 11)))


def _panel_sparse_edit_trajectory_reference_style(
    ax,
    long: dict,
    metrics: dict,
    *,
    title: str = "(a) Physically valid sparse edits",
):
    segments = long.get("segments", [])
    if not segments:
        _panel_sparse_edit_metrics(ax, metrics, title=title)
        _style_reference_line_axis(ax)
        return
    teacher_counts = np.asarray([float(seg["traditional_changed_site_count"]) for seg in segments], dtype=float)
    model_counts = np.asarray([float(seg["projected_changed_count"]) for seg in segments], dtype=float)
    x = np.arange(len(segments) + 1, dtype=float)
    teacher_path = np.concatenate([[0.0], np.cumsum(teacher_counts)])
    model_path = np.concatenate([[0.0], np.cumsum(model_counts)])
    markevery = _reference_line_marker_every(x)

    ax.plot(
        x,
        teacher_path,
        color=REF_BLUE,
        lw=2.05,
        marker="o",
        markersize=4.9,
        markerfacecolor=REF_BLUE,
        markeredgecolor=REF_BLUE,
        markevery=markevery,
        label="teacher",
    )
    ax.plot(
        x,
        model_path,
        color=REF_PURPLE,
        lw=2.05,
        marker="o",
        markersize=4.9,
        markerfacecolor=REF_PURPLE,
        markeredgecolor=REF_PURPLE,
        markevery=markevery,
        label="model",
    )
    ax.set_xlim(0, x[-1])
    ax.set_ylim(0, max(teacher_path.max(), model_path.max()) * 1.08)
    ax.set_xticks([0, 100, 200])
    ax.set_xlabel("Macro Segment ID")
    ax.set_ylabel("Cumulative Changed Sites")
    ax.set_title(title, loc="left", pad=7)
    _style_reference_line_axis(ax)
    ax.legend(
        frameon=False,
        fontsize=7.4,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.05),
        ncol=2,
        handlelength=1.2,
        columnspacing=0.75,
        handletextpad=0.30,
        borderaxespad=0.0,
    )


def _panel_time_alignment(ax, align: dict, *, title: str = "(b) Single-segment time alignment", color_by_k: bool = False):
    samples = align["all_samples"]
    true_tau = np.array([s["traditional_kmc_expected_tau"] for s in samples], dtype=float)
    pred_tau = np.array([s["predicted_expected_tau"] for s in samples], dtype=float)
    if color_by_k:
        ks = np.array([int(s.get("segment_k", align.get("segment_k", 0))) for s in samples], dtype=int)
        k_colors = {2: SOFT_BLUE_DARK, 4: SOFT_GREEN_DARK, 8: SOFT_PINK_DARK}
        for k in sorted(set(ks.tolist())):
            mask = ks == k
            ax.scatter(true_tau[mask], pred_tau[mask], s=10, color=k_colors.get(k, PURPLE),
                       alpha=0.56, edgecolors="none", label=fr"$k={k}$")
        ax.legend(frameon=False, fontsize=6.4, loc="lower right", handletextpad=0.2, borderaxespad=0.2)
    else:
        ax.scatter(true_tau, pred_tau, s=9, color=SOFT_BLUE_DARK, alpha=0.54, edgecolors="none")
    mn = min(true_tau.min(), pred_tau.min()) * 0.7
    mx = max(true_tau.max(), pred_tau.max()) * 1.25
    ax.plot([mn, mx], [mn, mx], color="#8793a0", lw=1.0, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(mn, mx)
    ax.set_ylim(mn, mx)
    ax.set_xlabel(r"traditional KMC $\tau_{\rm exp}$")
    ax.set_ylabel(r"predicted $\hat\tau_{\rm exp}$")
    ax.set_title(title, loc="left", pad=5)
    ax.text(
        0.04,
        0.95,
        f"n={len(samples)}\nlog-MAE={align['tau_expected']['log_mae']:.3f}\nlog-corr={align['tau_expected']['log_corr']:.3f}\nscale={align['tau_expected']['scale_ratio']:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.0,
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#d9dee8", lw=0.7),
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, which="major", color="#d9dee8", lw=0.5, alpha=0.8)


def _panel_cumulative_time(ax, long: dict, *, title: str = "(c) Long-trajectory cumulative time"):
    arr = long["arrays"]
    kmc_cum = np.array(arr["traditional_kmc_expected_tau_cumsum"], dtype=float)
    pred_cum = np.array(arr["predicted_expected_tau_cumsum"], dtype=float)
    final_ratio = pred_cum[-1] / kmc_cum[-1]
    x = np.arange(len(kmc_cum) + 1, dtype=float)
    kmc_path = np.concatenate([[0.0], kmc_cum])
    pred_path = np.concatenate([[0.0], pred_cum])
    ax.plot(
        x,
        kmc_path,
        color=SOFT_BLUE_DARK,
        lw=1.55,
        label="trad. KMC",
    )
    ax.plot(
        x,
        pred_path,
        color=SOFT_PINK_DARK,
        lw=1.55,
        label="world model",
    )
    ax.scatter([x[-1]], [kmc_path[-1]], color=SOFT_BLUE_DARK, s=14, zorder=3)
    ax.scatter([x[-1]], [pred_path[-1]], color=SOFT_PINK_DARK, s=14, zorder=3)
    ax.set_xlabel("Macro Segment ID")
    ax.set_ylabel(r"Cumulative $\tau_{\rm exp}$")
    ax.set_title(title, loc="left", pad=5)
    tau = long.get("tau_expected", {})
    if "log_mae" in tau and "log_corr" in tau:
        ax.text(
            0.97,
            0.06,
            f"final ratio={final_ratio:.3f}\nlog-MAE={tau['log_mae']:.3f}\nlog-corr={tau['log_corr']:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.2,
            color=DARK,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=FIG2_PANEL_C_ACCENT, lw=0.65),
        )
    ax.set_xlim(0, x[-1])
    ax.set_ylim(0, max(kmc_path.max(), pred_path.max()) * 1.16)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, which="major", color="#d9dee8", lw=0.5, alpha=0.8)
    ax.legend(frameon=False, fontsize=5.9, loc="upper left", handlelength=1.2, borderaxespad=0.2)


def _panel_cumulative_time_reference_style(ax, long: dict, *, title: str = "(c) Long-trajectory cumulative time"):
    arr = long["arrays"]
    kmc_cum = np.array(arr["traditional_kmc_expected_tau_cumsum"], dtype=float)
    pred_cum = np.array(arr["predicted_expected_tau_cumsum"], dtype=float)
    x = np.arange(len(kmc_cum) + 1, dtype=float)
    kmc_path = np.concatenate([[0.0], kmc_cum])
    pred_path = np.concatenate([[0.0], pred_cum])
    markevery = _reference_line_marker_every(x)

    ax.plot(
        x,
        kmc_path,
        color=REF_YELLOW,
        lw=2.05,
        marker="o",
        markersize=4.9,
        markerfacecolor=REF_YELLOW,
        markeredgecolor=REF_YELLOW,
        markevery=markevery,
        label="KMC",
    )
    ax.plot(
        x,
        pred_path,
        color=REF_ORANGE,
        lw=2.05,
        marker="o",
        markersize=4.9,
        markerfacecolor=REF_ORANGE,
        markeredgecolor=REF_ORANGE,
        markevery=markevery,
        label="model",
    )
    ax.set_xlim(0, x[-1])
    ax.set_ylim(0, max(kmc_path.max(), pred_path.max()) * 1.12)
    ax.set_xticks([0, 100, 200])
    ax.set_xlabel("Macro Segment ID")
    ax.set_ylabel(r"Cumulative $\tau_{\rm exp}$")
    ax.set_title(title, loc="left", pad=7)
    _style_reference_line_axis(ax)
    ax.legend(
        frameon=False,
        fontsize=7.4,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.05),
        ncol=2,
        handlelength=1.2,
        columnspacing=0.75,
        handletextpad=0.30,
        borderaxespad=0.0,
    )


def _panel_realized_time_calibration(ax, align: dict, *, title: str = "Auxiliary realized-time calibration"):
    dist = align["tau_realized_distribution"]
    nominal = np.array([0.68, 0.95])
    empirical = np.array([dist["coverage_68"], dist["coverage_95"]])
    labels = ["68%\ninterval", "95%\ninterval"]
    x4 = np.arange(len(labels))
    ax.bar(
        x4,
        empirical,
        color=[SOFT_BLUE, SOFT_PINK],
        edgecolor=SOFT_EDGE,
        linewidth=0.55,
        alpha=0.98,
        width=0.55,
        label="empirical",
    )
    ax.scatter(x4, nominal, color=DARK, marker="D", s=24, zorder=3, label="nominal")
    for i, (e, n) in enumerate(zip(empirical, nominal)):
        ax.plot([i - 0.26, i + 0.26], [n, n], color=DARK, lw=0.9, ls="--")
        ax.text(i, e + 0.035, f"{100*e:.1f}%", ha="center", va="bottom", fontsize=7.3)
    ax.set_xticks(x4, labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("coverage")
    ax.set_title(title, loc="left", pad=5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#d9dee8", lw=0.5, alpha=0.8)
    ax.legend(frameon=False, fontsize=6.6, loc="upper left")


def _fig_main_result_triplet(
    result_dir: Path,
    align_name: str,
    long_name: str,
    output_name: str,
    *,
    metrics_result_dir: Path | None = None,
    align_result_dir: Path | None = None,
    long_result_dir: Path | None = None,
    color_by_k: bool = False,
    emphasis_kwargs: dict | None = None,
):
    metrics_dir = metrics_result_dir if metrics_result_dir is not None else result_dir
    align_dir = align_result_dir if align_result_dir is not None else result_dir
    long_dir = long_result_dir if long_result_dir is not None else result_dir
    metrics = json.loads((metrics_dir / "metrics.json").read_text())["val"]
    align = json.loads((align_dir / align_name).read_text())
    long = json.loads((long_dir / long_name).read_text())

    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.25))
    _panel_sparse_edit_metrics(axes[0], metrics)
    _panel_time_alignment(axes[1], align, color_by_k=color_by_k)
    _panel_cumulative_time(axes[2], long)
    if output_name == "fig2_main_results.pdf" or emphasis_kwargs is not None:
        kwargs = emphasis_kwargs or {}
        for ax in axes:
            _emphasize_axis_artists(ax, **kwargs)
    fig.tight_layout(pad=0.8, w_pad=1.8)
    fig.savefig(_figure_output_path(output_name), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _case_values(rows: list[dict], metric_name: str, *, source: str = "metrics") -> np.ndarray:
    if source == "metrics":
        return np.asarray([float(row["metrics"][metric_name]) for row in rows], dtype=np.float64)
    if source == "collection_stats":
        return np.asarray([float(row["collection_stats"][metric_name]) for row in rows], dtype=np.float64)
    raise ValueError(f"Unknown case value source: {source}")


def _setup_case_axis(ax, x: np.ndarray, labels: list[str]) -> None:
    ax.set_xticks(x, labels)
    ax.tick_params(axis="x", labelsize=6.6, length=0)
    ax.grid(axis="y", color="#d9dee8", lw=0.48, alpha=0.82)
    ax.spines[["top", "right"]].set_visible(False)


def fig1_main_results(output_name: str = "fig2_main_results.pdf", *, emphasis_kwargs: dict | None = None):
    with plt.rc_context(_fig1_preview_font_rc()):
        _fig1_main_results_reference_style(output_name)


def fig1_main_results_ac_reference_preview():
    with plt.rc_context(_fig1_preview_font_rc()):
        _fig1_main_results_reference_style("fig1_main_results_ac_reference_preview.pdf")


def _fig1_main_results_reference_style(output_name: str):
    metrics = json.loads((RESULT / "metrics.json").read_text())["val"]
    long = json.loads((V52_RESULT / "eval_long_trajectory_500_noopstop.json").read_text())
    payload = _load_fig2_temperature_cases()
    if payload is None:
        _fig_main_result_triplet(
            RESULT,
            "eval_time_alignment_final_retry_b4.json",
            "eval_long_trajectory_500_noopstop.json",
            output_name,
            align_result_dir=V52_RESULT,
            long_result_dir=V52_RESULT,
            emphasis_kwargs={"font_scale": 1.16, "line_scale": 1.40, "marker_area_scale": 2.20},
        )
        return
    rows = sorted(payload["rows"], key=lambda row: float(row["case"]["temperature"]))

    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.65))
    _panel_sparse_edit_trajectory_reference_style(axes[0], long, metrics)
    _panel_time_alignment_temperature_cases(
        axes[1],
        rows,
        colors=[REF_YELLOW, REF_BLUE, REF_ORANGE, REF_PURPLE],
        show_summary=False,
        xlabel=r"Teacher $\tau_{\rm exp}$",
    )
    _panel_cumulative_time_reference_style(axes[2], long)
    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        axes[1].legend(
            handles,
            labels,
            frameon=False,
            fontsize=6.2,
            loc="lower right",
            handlelength=0.8,
            handletextpad=0.25,
            borderaxespad=0.2,
            labelspacing=0.15,
        )
    _emphasize_axis_artists(
        axes[1],
        font_scale=1.16,
        title_scale=1.02,
        line_scale=1.40,
        marker_area_scale=2.20,
        marker_size_scale=1.34,
    )
    _style_reference_boxed_axis(axes[1])
    axes[1].yaxis.set_minor_locator(NullLocator())
    axes[1].tick_params(axis="y", which="minor", length=0)
    for ax in axes:
        _center_axis_title(ax)
        _style_arial_axis_text(ax)
    panel_title_specs = _aligned_panel_title_specs(axes)
    fig.tight_layout(pad=0.75, w_pad=1.95, rect=(0.0, 0.0, 1.0, 0.89))
    _draw_aligned_panel_titles(fig, axes, panel_title_specs, gap=0.014)
    fig.savefig(_figure_output_path(output_name), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _load_fig2_temperature_cases() -> dict | None:
    if not FIG2_TEMPERATURE_CASES.exists():
        print(f"[fig2-temp-preview] missing {FIG2_TEMPERATURE_CASES}; skipping preview")
        return None
    return json.loads(FIG2_TEMPERATURE_CASES.read_text())


def _temperature_label(row: dict) -> str:
    case = row["case"]
    return f"{int(round(float(case['temperature'])))}K"


def _panel_sparse_edit_temperature_cases(ax, rows: list[dict], *, title: str = "(a) Physically valid sparse edits"):
    metrics = [
        ("reachable\nedits", [1.0 - float(row["metrics"]["reachability_violation_rate"]) for row in rows]),
        ("changed-site\ntype acc.", [float(row["metrics"]["changed_type_acc"]) for row in rows]),
    ]
    x = np.arange(len(metrics), dtype=float)
    n = len(rows)
    width = min(0.18, 0.72 / max(n, 1))
    offsets = (np.arange(n) - (n - 1) / 2.0) * width
    for idx, row in enumerate(rows):
        vals = [vals[idx] for _label, vals in metrics]
        color = TEMP_CASE_COLORS[idx % len(TEMP_CASE_COLORS)]
        ax.bar(
            x + offsets[idx],
            vals,
            width=width * 0.92,
            color=color,
            edgecolor=SOFT_EDGE,
            linewidth=0.5,
            alpha=0.98,
            label=_temperature_label(row),
        )
    for metric_idx, (_label, vals) in enumerate(metrics):
        for idx, val in enumerate(vals):
            ax.text(
                x[metric_idx] + offsets[idx],
                min(float(val) + 0.028, 1.075),
                f"{100*float(val):.0f}",
                ha="center",
                va="bottom",
                fontsize=5.5,
                rotation=90,
            )
    ax.set_xticks(x, [label for label, _vals in metrics])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("score")
    ax.set_title(title, loc="left", pad=5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#d9dee8", lw=0.5, alpha=0.8)


def _panel_time_alignment_temperature_cases(
    ax,
    rows: list[dict],
    *,
    title: str = "(b) Single-segment time alignment",
    colors: list[str] | None = None,
    show_summary: bool = True,
    xlabel: str = r"Traditional KMC $\tau_{\rm exp}$",
):
    palette = colors or TEMP_CASE_DARK
    all_true = []
    all_pred = []
    for idx, row in enumerate(rows):
        samples = row.get("scatter_samples", [])
        true_tau = np.asarray([float(s["traditional_kmc_expected_tau"]) for s in samples], dtype=float)
        pred_tau = np.asarray([float(s["predicted_expected_tau"]) for s in samples], dtype=float)
        if len(true_tau) == 0:
            continue
        all_true.append(true_tau)
        all_pred.append(pred_tau)
        ax.scatter(
            true_tau,
            pred_tau,
            s=9,
            color=palette[idx % len(palette)],
            alpha=0.55,
            edgecolors="none",
            label=_temperature_label(row),
        )
    if not all_true:
        ax.text(0.5, 0.5, "no samples", transform=ax.transAxes, ha="center", va="center")
        return
    true_all = np.concatenate(all_true)
    pred_all = np.concatenate(all_pred)
    mn = min(true_all.min(), pred_all.min()) * 0.70
    mx = max(true_all.max(), pred_all.max()) * 1.25
    ax.plot([mn, mx], [mn, mx], color="#8793a0", lw=1.0, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(mn, mx)
    ax.set_ylim(mn, mx)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Predicted $\hat\tau_{\rm exp}$")
    ax.set_title(title, loc="left", pad=5)
    if show_summary:
        ax.text(
            0.04,
            0.95,
            f"n={len(true_all)}\n4 temperature cases",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.9,
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#d9dee8", lw=0.7),
        )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, which="major", color="#d9dee8", lw=0.5, alpha=0.8)


def _paired_cumulative_tau(row: dict) -> tuple[float, float, int]:
    samples = row.get("scatter_samples", [])
    true_sum = float(sum(float(s["traditional_kmc_expected_tau"]) for s in samples))
    pred_sum = float(sum(float(s["predicted_expected_tau"]) for s in samples))
    return true_sum, pred_sum, int(len(samples))


def _panel_cumulative_time_temperature_cases(ax, rows: list[dict], *, title: str = "(c) Cumulative paired-segment time"):
    cumulative = [_paired_cumulative_tau(row) for row in rows]
    metrics = [
        (
            "traditional\nKMC",
            [item[0] for item in cumulative],
        ),
        (
            "world\nmodel",
            [item[1] for item in cumulative],
        ),
    ]
    x = np.arange(len(metrics), dtype=float)
    n = len(rows)
    width = min(0.18, 0.72 / max(n, 1))
    offsets = (np.arange(n) - (n - 1) / 2.0) * width
    for idx, row in enumerate(rows):
        vals = [vals[idx] for _label, vals in metrics]
        ax.bar(
            x + offsets[idx],
            vals,
            width=width * 0.92,
            color=TEMP_CASE_COLORS[idx % len(TEMP_CASE_COLORS)],
            edgecolor=SOFT_EDGE,
            linewidth=0.5,
            alpha=0.98,
        )
    vals_all = np.asarray([val for _label, vals in metrics for val in vals], dtype=float)
    positive = vals_all[vals_all > 0]
    use_log = bool(positive.size and positive.max() / max(positive.min(), 1e-12) > 80.0)
    if use_log:
        ax.set_yscale("log")
        ax.set_ylim(max(positive.min() * 0.45, 1e-12), positive.max() * 2.6)
    else:
        ax.set_ylim(0, max(vals_all.max() * 1.28, 1e-12))
    for idx, row in enumerate(rows):
        true_sum, pred_sum, completed = cumulative[idx]
        ratio = pred_sum / true_sum if true_sum > 1e-12 else None
        y = pred_sum
        if y <= 0:
            continue
        ax.text(
            x[1] + offsets[idx],
            y * (1.16 if use_log else 1.035),
            f"{ratio:.1f}x" if ratio is not None else f"n={completed}",
            ha="center",
            va="bottom",
            fontsize=5.5,
            rotation=90,
        )
    ax.set_xticks(x, [label for label, _vals in metrics])
    ax.set_ylabel(r"final cumulative $\tau_{\rm exp}$")
    ax.set_title(title, loc="left", pad=5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#d9dee8", lw=0.5, alpha=0.8)


def fig2_main_results_temperature_cases_preview():
    payload = _load_fig2_temperature_cases()
    if payload is None:
        return
    rows = sorted(payload["rows"], key=lambda row: float(row["case"]["temperature"]))
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.35))
    _panel_sparse_edit_temperature_cases(axes[0], rows)
    _panel_time_alignment_temperature_cases(axes[1], rows)
    _panel_cumulative_time_temperature_cases(axes[2], rows)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=6.8,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.52, 1.035),
        handlelength=1.0,
        columnspacing=1.0,
    )
    fig.tight_layout(pad=0.8, w_pad=1.25)
    fig.subplots_adjust(top=0.78)
    pdf_paths = [
        _figure_output_path("fig2_main_results_temperature_cases_preview.pdf"),
        _figure_output_path("fig2_main_results_temperature_cases_t263_373_preview.pdf"),
    ]
    png_paths = [
        _figure_output_path("fig2_main_results_temperature_cases_preview.png"),
        _figure_output_path("fig2_main_results_temperature_cases_t263_373_preview.png"),
    ]
    for pdf_path in pdf_paths:
        fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    for png_path in png_paths:
        fig.savefig(png_path, dpi=280, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def fig8_realized_time_calibration():
    align = json.loads((RESULT / "eval_time_alignment_realized_final.json").read_text())
    fig, ax = plt.subplots(1, 1, figsize=(2.85, 2.25))
    _panel_realized_time_calibration(ax, align)
    fig.tight_layout(pad=0.8)
    fig.savefig(_figure_output_path("fig8_realized_time_calibration.pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _fig4_lattice_size_speedup():
    scaling_path = V52_RESULT / "benchmark_lattice_size_speedup_l20_60_k8_s64.json"
    data = json.loads(scaling_path.read_text())
    rows = data["rows"]
    x = np.arange(len(rows), dtype=float)
    labels = []
    akmc = []
    akmc_min = []
    akmc_max = []
    atomworld = []
    atomworld_min = []
    atomworld_max = []
    speedup = []
    for row in rows:
        edge = row.get("lattice_edge")
        if edge is None:
            shape = row["lattice_shape"]
            labels.append(f"{int(np.prod(shape))}")
        else:
            labels.append(_cube_size_label(edge))
        akmc_stats = row["akmc_runtime_s"]
        atom_stats = row["atomworld_runtime_s"]
        akmc.append(float(akmc_stats["mean"]))
        akmc_min.append(float(akmc_stats["min"]))
        akmc_max.append(float(akmc_stats["max"]))
        atomworld.append(float(atom_stats["mean"]))
        atomworld_min.append(float(atom_stats["min"]))
        atomworld_max.append(float(atom_stats["max"]))
        speedup.append(float(row["speedup_akmc_over_atomworld"]))

    akmc = np.asarray(akmc, dtype=np.float64)
    akmc_min = np.asarray(akmc_min, dtype=np.float64)
    akmc_max = np.asarray(akmc_max, dtype=np.float64)
    atomworld = np.asarray(atomworld, dtype=np.float64)
    atomworld_min = np.asarray(atomworld_min, dtype=np.float64)
    atomworld_max = np.asarray(atomworld_max, dtype=np.float64)
    speedup = np.asarray(speedup, dtype=np.float64)

    fig, ax1 = plt.subplots(1, 1, figsize=(5.8, 2.65))

    bar_w = 0.33
    ax1.bar(x - bar_w / 2, akmc, width=bar_w, color="#aecce4", edgecolor=DARK,
            linewidth=0.6, alpha=0.95, label="AKMC runtime")
    ax1.bar(x + bar_w / 2, atomworld, width=bar_w, color="#efbcb8", edgecolor=DARK,
            linewidth=0.6, alpha=0.95, label="AtomWorld runtime")
    cap_half = 0.045
    for xpos, ymin, ymax in zip(x - bar_w / 2, akmc_min, akmc_max):
        if ymax > ymin:
            ax1.vlines(xpos, ymin, ymax, color=DARK, lw=0.75, zorder=4)
            ax1.hlines(ymax, xpos - cap_half, xpos + cap_half, color=DARK, lw=0.75, zorder=4)
    for xpos, ymin, ymax in zip(x + bar_w / 2, atomworld_min, atomworld_max):
        if ymax > ymin:
            ax1.vlines(xpos, ymin, ymax, color=DARK, lw=0.75, zorder=4)
            ax1.hlines(ymax, xpos - cap_half, xpos + cap_half, color=DARK, lw=0.75, zorder=4)
    ax1.set_yscale("log")
    ax1.set_ylim(max(0.004, float(atomworld_min.min()) * 0.45), float(akmc_max.max()) * 2.8)
    ax1.set_ylabel("RunTime (s)", fontsize=9.2, fontweight="bold")
    ax1.set_xticks(x, labels)
    ax1.set_xlabel("Cubic lattice size (L\N{SUPERSCRIPT THREE})", fontsize=9.0, fontweight="bold")
    ax1.tick_params(axis="x", labelsize=7.2, rotation=0)
    ax1.tick_params(axis="y", labelsize=7.2)
    ax1.grid(axis="y", which="major", linestyle="--", color="#aeb5c1", alpha=0.85, linewidth=0.5)
    ax1.spines[["top"]].set_visible(False)

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        speedup,
        color="#8f6dc0",
        linewidth=1.8,
        marker="v",
        markersize=6.0,
        markerfacecolor="white",
        markeredgecolor="#7351b4",
        markeredgewidth=1.1,
        label="Speedup",
        zorder=5,
    )
    for xi, val in zip(x, speedup):
        label = ax2.text(
            xi + 0.06,
            val * 1.035,
            f"{val:.1f}x",
            ha="left",
            va="bottom",
            fontsize=7.1,
            color=DARK,
            fontweight="bold",
            zorder=20,
        )
        label.set_path_effects([path_effects.withStroke(linewidth=2.0, foreground="white")])
    ax2.set_ylim(0, float(speedup.max()) * 1.22)
    ax2.set_ylabel("Speed Up", fontsize=9.2, fontweight="bold", rotation=270, labelpad=13)
    ax2.tick_params(axis="y", labelsize=7.2)
    ax2.spines[["top"]].set_visible(False)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="upper left",
        fontsize=7.2,
        frameon=True,
        borderpad=0.3,
        handlelength=1.35,
    )
    legend.get_frame().set_edgecolor(DARK)
    legend.get_frame().set_linewidth(0.55)

    fig.tight_layout(pad=0.75)
    fig.savefig(_figure_output_path("fig4_speedup_benchmark.pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _fmt_float(value: float) -> str:
    text = f"{value:g}"
    return text


def _fmt_speedup_label(value: float) -> str:
    if value < 100:
        return f"{value:.1f}x"
    return f"{value:.0f}x"


def _fig4_case_grid_speedup(
    cu_density: float,
    output_name: str,
    *,
    data_path: Path = FIG4_CASE_GRID,
    atomworld_label: str = "AtomWorld-Mirror",
):
    data = json.loads(data_path.read_text())
    rows = [
        row
        for row in data["rows"]
        if abs(float(row["cu_density"]) - float(cu_density)) < 1e-9
    ]
    rows.sort(key=lambda row: (int(row["lattice_edge"]), int(row["temperature_code"])))
    x = np.arange(len(rows), dtype=float)
    lattice_keys = ["a", "b", "c", "d", "e"]
    lattice_values = sorted({int(row["lattice_edge"]) for row in rows})
    lattice_code = {edge: lattice_keys[idx] for idx, edge in enumerate(lattice_values)}
    temperature_values = sorted({int(row["temperature"]) for row in rows})
    temperature_code = {temperature: idx + 1 for idx, temperature in enumerate(temperature_values)}
    labels = [
        f"{lattice_code[int(row['lattice_edge'])]}{temperature_code[int(row['temperature'])]}"
        for row in rows
    ]
    akmc = np.asarray([float(row["akmc_runtime_s"]["mean"]) for row in rows], dtype=np.float64)
    atomworld = np.asarray([float(row["atomworld_runtime_s"]["mean"]) for row in rows], dtype=np.float64)
    akmc_min = np.asarray([float(row["akmc_runtime_s"]["min"]) for row in rows], dtype=np.float64)
    akmc_max = np.asarray([float(row["akmc_runtime_s"]["max"]) for row in rows], dtype=np.float64)
    atom_min = np.asarray([float(row["atomworld_runtime_s"]["min"]) for row in rows], dtype=np.float64)
    atom_max = np.asarray([float(row["atomworld_runtime_s"]["max"]) for row in rows], dtype=np.float64)
    speedup = np.asarray([float(row["speedup_akmc_over_atomworld"]) for row in rows], dtype=np.float64)

    codebook = data.get("codebook", {})
    temp_book = {
        temperature_code[temperature]: temperature
        for temperature in temperature_values
    }
    lattice_book = {
        lattice_code[edge]: _cube_size_label(edge)
        for edge in lattice_values
    }
    temp_text = "Temperature: " + "   ".join(
        f"{key}={_fmt_float(float(value))}K" for key, value in temp_book.items()
    )
    lattice_text = "Cubic lattice size: " + "   ".join(
        f"{key}={value}" for key, value in lattice_book.items()
    )

    fig, ax1 = plt.subplots(1, 1, figsize=(6.9, 2.95))
    bar_w = 0.34
    ax1.bar(
        x - bar_w / 2,
        akmc,
        width=bar_w,
        color=SOFT_BLUE,
        edgecolor=DARK,
        linewidth=0.42,
        alpha=0.96,
        label="Teacher Replay",
        zorder=2,
    )
    ax1.bar(
        x + bar_w / 2,
        atomworld,
        width=bar_w,
        color=SOFT_PINK,
        edgecolor=DARK,
        linewidth=0.42,
        alpha=0.96,
        label=atomworld_label,
        zorder=2,
    )
    cap_half = 0.035
    for xpos, ymin, ymax in zip(x - bar_w / 2, akmc_min, akmc_max):
        if ymax > ymin:
            ax1.vlines(xpos, ymin, ymax, color=DARK, lw=0.52, zorder=4)
            ax1.hlines(ymax, xpos - cap_half, xpos + cap_half, color=DARK, lw=0.52, zorder=4)
    for xpos, ymin, ymax in zip(x + bar_w / 2, atom_min, atom_max):
        if ymax > ymin:
            ax1.vlines(xpos, ymin, ymax, color=DARK, lw=0.52, zorder=4)
            ax1.hlines(ymax, xpos - cap_half, xpos + cap_half, color=DARK, lw=0.52, zorder=4)

    ax1.set_yscale("log")
    ax1.set_ylim(max(0.01, float(atom_min.min()) * 0.42), float(akmc_max.max()) * 2.5)
    ax1.set_ylabel("RunTime (s)", fontsize=8.6, fontweight="bold")
    ax1.set_xlim(-0.65, len(rows) - 0.35)
    ax1.set_xticks(x, labels)
    ax1.tick_params(axis="x", labelsize=6.25, rotation=0, length=0, pad=2.0)
    ax1.tick_params(axis="y", labelsize=6.8)
    ax1.grid(axis="y", which="major", linestyle="--", color="#aeb5c1", alpha=0.78, linewidth=0.42)
    ax1.spines[["top"]].set_visible(False)
    group_size = len(temp_book) if temp_book else 4
    for boundary in np.arange(group_size - 0.5, len(rows) - 0.5, group_size):
        ax1.axvline(boundary, color="#d7dce4", lw=0.42, zorder=1)

    ax2 = ax1.twinx()
    speed_color = "#7fab87"
    speed_edge = "#4f8063"
    for start in range(0, len(rows), group_size):
        group_x = x[start:start + group_size]
        group_speedup = speedup[start:start + group_size]
        ax2.plot(
            group_x,
            group_speedup,
            color=speed_color,
            linewidth=1.45,
            marker="v",
            markersize=3.9,
            markerfacecolor="white",
            markeredgecolor=speed_edge,
            markeredgewidth=0.9,
            label="Speedup (KMC/AtomWorld-Mirror)" if start == 0 else None,
            zorder=8,
        )
        for xi, value in zip(group_x, group_speedup):
            label = ax2.text(
                xi,
                float(value) * 1.08,
                _fmt_speedup_label(float(value)),
                ha="center",
                va="bottom",
                fontsize=4.6,
                color=DARK,
                fontweight="bold",
                zorder=20,
            )
            label.set_path_effects([path_effects.withStroke(linewidth=1.35, foreground="white")])
    ax2.set_yscale("log")
    speed_low = max(0.05, float(speedup.min()) * 0.55)
    speed_high = max(float(speedup.max()) * 2.1, speed_low * 1.5)
    ax2.set_ylim(speed_low, speed_high)
    ax2.set_ylabel("Speed Up", fontsize=8.6, fontweight="bold", rotation=270, labelpad=12)
    ax2.tick_params(axis="y", labelsize=6.8)
    ax2.spines[["top"]].set_visible(False)

    ax1.text(
        0.5,
        1.18,
        temp_text,
        transform=ax1.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.65,
        fontweight="bold",
        color=DARK,
    )
    ax1.text(
        0.5,
        1.08,
        lattice_text,
        transform=ax1.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.3,
        fontweight="bold",
        color=DARK,
    )

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=3,
        fontsize=6.25,
        frameon=False,
        handlelength=1.5,
        columnspacing=0.85,
    )
    for handle in getattr(legend, "legend_handles", getattr(legend, "legendHandles", [])):
        if hasattr(handle, "set_linewidth"):
            handle.set_linewidth(1.4)

    fig.subplots_adjust(left=0.075, right=0.92, top=0.73, bottom=0.19)
    fig.savefig(_figure_output_path(output_name), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _fig4_case_grid_speedup_broken_axis(
    cu_density: float,
    output_name: str,
    *,
    data_path: Path = FIG4_CASE_GRID,
    atomworld_label: str = "AtomWorld-Mirror",
):
    data = json.loads(data_path.read_text())
    rows = [
        row
        for row in data["rows"]
        if abs(float(row["cu_density"]) - float(cu_density)) < 1e-9
    ]
    rows.sort(key=lambda row: (int(row["lattice_edge"]), int(row["temperature_code"])))
    x = np.arange(len(rows), dtype=float)
    lattice_keys = list("abcdef")
    lattice_values = sorted({int(row["lattice_edge"]) for row in rows})
    lattice_code = {edge: lattice_keys[idx] for idx, edge in enumerate(lattice_values)}
    temperature_values = sorted({int(row["temperature"]) for row in rows})
    temperature_code = {temperature: idx + 1 for idx, temperature in enumerate(temperature_values)}
    labels = [
        f"{lattice_code[int(row['lattice_edge'])]}{temperature_code[int(row['temperature'])]}"
        for row in rows
    ]
    akmc = np.asarray([float(row["akmc_runtime_s"]["mean"]) for row in rows], dtype=np.float64)
    atomworld = np.asarray([float(row["atomworld_runtime_s"]["mean"]) for row in rows], dtype=np.float64)
    akmc_min = np.asarray([float(row["akmc_runtime_s"]["min"]) for row in rows], dtype=np.float64)
    akmc_max = np.asarray([float(row["akmc_runtime_s"]["max"]) for row in rows], dtype=np.float64)
    atom_min = np.asarray([float(row["atomworld_runtime_s"]["min"]) for row in rows], dtype=np.float64)
    atom_max = np.asarray([float(row["atomworld_runtime_s"]["max"]) for row in rows], dtype=np.float64)
    speedup = np.asarray([float(row["speedup_akmc_over_atomworld"]) for row in rows], dtype=np.float64)

    temp_text = "Temperature: " + "   ".join(
        f"{temperature_code[temperature]}={_fmt_float(float(temperature))}K"
        for temperature in temperature_values
    )
    lattice_text = "Cubic lattice size: " + "   ".join(
        f"{lattice_code[edge]}={_cube_size_label(edge)}" for edge in lattice_values
    )

    fig = plt.figure(figsize=(6.9, 3.35))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.08, 1.0], hspace=0.055)
    ax_top = fig.add_subplot(gs[0])
    ax_bottom = fig.add_subplot(gs[1], sharex=ax_top)
    axes = (ax_top, ax_bottom)

    kmc_color = REF_ORANGE
    atom_color = REF_BLUE
    bar_w = 0.34
    for ax in axes:
        ax.bar(
            x - bar_w / 2,
            akmc,
            width=bar_w,
            color=kmc_color,
            edgecolor="#111111",
            linewidth=0.54,
            hatch="\\\\",
            alpha=0.98,
            label="Teacher Replay",
            zorder=3,
        )
        ax.bar(
            x + bar_w / 2,
            atomworld,
            width=bar_w,
            color=atom_color,
            edgecolor="#111111",
            linewidth=0.54,
            hatch="//",
            alpha=0.96,
            label=atomworld_label,
            zorder=3,
        )
        cap_half = 0.035
        for xpos, ymin, ymax in zip(x - bar_w / 2, akmc_min, akmc_max):
            if ymax > ymin:
                ax.vlines(xpos, ymin, ymax, color="#111111", lw=0.52, zorder=5)
                ax.hlines(ymax, xpos - cap_half, xpos + cap_half, color="#111111", lw=0.52, zorder=5)
        for xpos, ymin, ymax in zip(x + bar_w / 2, atom_min, atom_max):
            if ymax > ymin:
                ax.vlines(xpos, ymin, ymax, color="#111111", lw=0.52, zorder=5)
                ax.hlines(ymax, xpos - cap_half, xpos + cap_half, color="#111111", lw=0.52, zorder=5)
        ax.grid(
            axis="y",
            which="major",
            linestyle=(0, (1.0, 3.0)),
            color="#111111",
            alpha=0.55,
            linewidth=0.72,
        )
        ax.set_axisbelow(True)
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(axis="y", which="major", direction="out", length=4.8, width=0.95, labelsize=7.0)
        ax.tick_params(axis="y", which="minor", direction="out", length=2.9, width=0.78)
        ax.tick_params(axis="x", which="major", direction="out", length=4.5, width=0.9)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#111111")
            spine.set_linewidth(1.02)

    bottom_ymax = 700.0
    top_ymin = 900.0
    top_ymax = float(np.ceil(float(akmc_max.max()) * 1.08 / 100.0) * 100.0)
    ax_bottom.set_ylim(0, bottom_ymax)
    ax_top.set_ylim(top_ymin, top_ymax)
    ax_bottom.set_yticks([0, 200, 400, 600])
    ax_top.set_yticks([tick for tick in [1000, 2000, 3000] if top_ymin <= tick <= top_ymax])
    ax_bottom.set_xlim(-0.65, len(rows) - 0.35)
    ax_bottom.set_xticks(x, labels)
    ax_bottom.tick_params(axis="x", labelsize=7.0, pad=2.0)
    ax_top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    ax_top.spines["bottom"].set_visible(False)
    ax_bottom.spines["top"].set_visible(False)

    speed_ax = ax_bottom.twinx()
    speed_color = REF_PURPLE
    speed_edge = "#5f5a94"
    group_size = len(temperature_values) if temperature_values else 4
    for start in range(0, len(rows), group_size):
        group_x = x[start:start + group_size]
        group_speedup = speedup[start:start + group_size]
        speed_ax.plot(
            group_x,
            group_speedup,
            color=speed_color,
            linewidth=1.55,
            marker="v",
            markersize=4.1,
            markerfacecolor="none",
            markeredgecolor=speed_edge,
            markeredgewidth=0.9,
            label="Speedup (KMC/AtomWorld-Mirror)" if start == 0 else None,
            zorder=9,
        )
        for xi, value in zip(group_x, group_speedup):
            label = speed_ax.text(
                xi,
                float(value) + 5.0,
                _fmt_speedup_label(float(value)),
                ha="center",
                va="bottom",
                fontsize=7.2,
                color=DARK,
                fontweight="bold",
                zorder=20,
            )
            label.set_path_effects([path_effects.withStroke(linewidth=1.35, foreground="white")])
    speed_ax.set_ylim(0, max(160.0, float(speedup.max()) * 1.18))
    speed_ax.set_yticks([50, 100, 150])
    speed_ax.set_ylabel("Speed Up", fontsize=8.6, fontweight="bold", rotation=270, labelpad=12)
    speed_ax.tick_params(axis="y", which="major", direction="out", length=4.8, width=0.95, labelsize=7.0)
    speed_ax.spines["top"].set_visible(False)
    speed_ax.spines["left"].set_visible(False)
    speed_ax.spines["right"].set_color("#111111")
    speed_ax.spines["right"].set_linewidth(1.02)
    ax_bottom.spines["right"].set_visible(False)

    diag = 0.016
    slash_kwargs = dict(color="#111111", clip_on=False, lw=1.15, solid_capstyle="butt")
    ax_top.plot((-diag, +diag), (-diag, +diag), transform=ax_top.transAxes, **slash_kwargs)
    ax_top.plot((1 - diag, 1 + diag), (-diag, +diag), transform=ax_top.transAxes, **slash_kwargs)
    ax_bottom.plot((-diag, +diag), (1 - diag, 1 + diag), transform=ax_bottom.transAxes, **slash_kwargs)
    ax_bottom.plot((1 - diag, 1 + diag), (1 - diag, 1 + diag), transform=ax_bottom.transAxes, **slash_kwargs)

    fig.text(0.060, 0.49, "RunTime (s)", ha="center", va="center", rotation=90, fontsize=9.2, fontweight="bold")
    fig.text(0.5, 0.955, temp_text, ha="center", va="top", fontsize=7.6, fontweight="bold", color=DARK)
    fig.text(0.5, 0.925, lattice_text, ha="center", va="top", fontsize=7.3, fontweight="bold", color=DARK)

    handles1, labels1 = ax_top.get_legend_handles_labels()
    handles2, labels2 = speed_ax.get_legend_handles_labels()
    seen = set()
    handles = []
    legend_labels = []
    for handle, label in zip(handles1 + handles2, labels1 + labels2):
        if label and label not in seen:
            handles.append(handle)
            legend_labels.append(label)
            seen.add(label)
    legend = ax_top.legend(
        handles,
        legend_labels,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        ncol=3,
        fontsize=7.15,
        frameon=False,
        handlelength=1.45,
        columnspacing=0.85,
        borderaxespad=0.0,
    )
    for handle in getattr(legend, "legend_handles", getattr(legend, "legendHandles", [])):
        if hasattr(handle, "set_linewidth"):
            handle.set_linewidth(1.4)

    for text in fig.findobj(match=Text):
        text.set_fontfamily("Arial")
        text.set_fontweight("normal")

    fig.subplots_adjust(left=0.125, right=0.925, top=0.88, bottom=0.14)
    fig.savefig(_figure_output_path(output_name), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _fig4_case_grid_speedup_broken_axis_lowbars(
    cu_density: float,
    output_name: str,
    *,
    data_path: Path = FIG4_CASE_GRID,
    atomworld_label: str = "AtomWorld-Mirror",
):
    data = json.loads(data_path.read_text())
    rows = [
        row
        for row in data["rows"]
        if abs(float(row["cu_density"]) - float(cu_density)) < 1e-9
    ]
    rows.sort(key=lambda row: (int(row["lattice_edge"]), int(row["temperature_code"])))
    x = np.arange(len(rows), dtype=float)
    lattice_keys = list("abcdef")
    lattice_values = sorted({int(row["lattice_edge"]) for row in rows})
    lattice_code = {edge: lattice_keys[idx] for idx, edge in enumerate(lattice_values)}
    temperature_values = sorted({int(row["temperature"]) for row in rows})
    temperature_code = {temperature: idx + 1 for idx, temperature in enumerate(temperature_values)}
    labels = [
        f"{lattice_code[int(row['lattice_edge'])]}{temperature_code[int(row['temperature'])]}"
        for row in rows
    ]
    akmc = np.asarray([float(row["akmc_runtime_s"]["mean"]) for row in rows], dtype=np.float64)
    atomworld = np.asarray([float(row["atomworld_runtime_s"]["mean"]) for row in rows], dtype=np.float64)
    akmc_min = np.asarray([float(row["akmc_runtime_s"]["min"]) for row in rows], dtype=np.float64)
    akmc_max = np.asarray([float(row["akmc_runtime_s"]["max"]) for row in rows], dtype=np.float64)
    atom_min = np.asarray([float(row["atomworld_runtime_s"]["min"]) for row in rows], dtype=np.float64)
    atom_max = np.asarray([float(row["atomworld_runtime_s"]["max"]) for row in rows], dtype=np.float64)
    speedup = np.asarray([float(row["speedup_akmc_over_atomworld"]) for row in rows], dtype=np.float64)

    temp_text = "Temperature: " + "   ".join(
        f"{temperature_code[temperature]}={_fmt_float(float(temperature))}K"
        for temperature in temperature_values
    )
    lattice_text = "Cubic lattice size: " + "   ".join(
        f"{lattice_code[edge]}={_cube_size_label(edge)}" for edge in lattice_values
    )

    fig = plt.figure(figsize=(8.4, 3.55))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.95, 1.05, 1.05], hspace=0.055)
    ax_high = fig.add_subplot(gs[0])
    ax_mid = fig.add_subplot(gs[1], sharex=ax_high)
    ax_low = fig.add_subplot(gs[2], sharex=ax_high)
    axes = (ax_high, ax_mid, ax_low)

    kmc_color = REF_ORANGE
    atom_color = REF_BLUE
    bar_w = 0.34
    for ax in axes:
        ax.bar(
            x - bar_w / 2,
            akmc,
            width=bar_w,
            color=kmc_color,
            edgecolor="#111111",
            linewidth=0.54,
            hatch="\\\\",
            alpha=0.98,
            label="Teacher Replay",
            zorder=3,
        )
        ax.bar(
            x + bar_w / 2,
            atomworld,
            width=bar_w,
            color=atom_color,
            edgecolor="#111111",
            linewidth=0.54,
            hatch="//",
            alpha=0.96,
            label=atomworld_label,
            zorder=3,
        )
        cap_half = 0.035
        for xpos, ymin, ymax in zip(x - bar_w / 2, akmc_min, akmc_max):
            if ymax > ymin:
                ax.vlines(xpos, ymin, ymax, color="#111111", lw=0.52, zorder=5)
                ax.hlines(ymax, xpos - cap_half, xpos + cap_half, color="#111111", lw=0.52, zorder=5)
        for xpos, ymin, ymax in zip(x + bar_w / 2, atom_min, atom_max):
            if ymax > ymin:
                ax.vlines(xpos, ymin, ymax, color="#111111", lw=0.52, zorder=5)
                ax.hlines(ymax, xpos - cap_half, xpos + cap_half, color="#111111", lw=0.52, zorder=5)
        ax.grid(
            axis="y",
            which="major",
            linestyle=(0, (1.0, 3.0)),
            color="#111111",
            alpha=0.55,
            linewidth=0.72,
        )
        ax.set_axisbelow(True)
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(axis="y", which="major", direction="out", length=4.8, width=0.95, labelsize=8.3)
        ax.tick_params(axis="y", which="minor", direction="out", length=2.9, width=0.78)
        ax.tick_params(axis="x", which="major", direction="out", length=4.5, width=0.9)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#111111")
            spine.set_linewidth(1.02)

    high_ymax = float(np.ceil(float(akmc_max.max()) * 1.08 / 100.0) * 100.0)
    ax_low.set_ylim(0, 45)
    ax_mid.set_ylim(70, 700)
    ax_high.set_ylim(900, high_ymax)
    ax_low.set_yticks([0, 15, 30, 45])
    ax_mid.set_yticks([100, 300, 500, 700])
    ax_high.set_yticks([tick for tick in [1000, 2000, 3000] if 900 <= tick <= high_ymax])
    ax_low.set_xlim(-0.65, len(rows) - 0.35)
    ax_low.set_xticks(x, labels)
    ax_low.tick_params(axis="x", labelsize=8.4, pad=2.0)
    ax_high.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_mid.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    ax_high.spines["bottom"].set_visible(False)
    ax_mid.spines["top"].set_visible(False)
    ax_mid.spines["bottom"].set_visible(False)
    ax_low.spines["top"].set_visible(False)

    speed_ax = ax_mid.twinx()
    speed_color = REF_PURPLE
    speed_edge = "#5f5a94"
    group_size = len(temperature_values) if temperature_values else 4
    for start in range(0, len(rows), group_size):
        group_x = x[start:start + group_size]
        group_speedup = speedup[start:start + group_size]
        speed_ax.plot(
            group_x,
            group_speedup,
            color=speed_color,
            linewidth=1.55,
            marker="v",
            markersize=4.1,
            markerfacecolor="none",
            markeredgecolor=speed_edge,
            markeredgewidth=0.9,
            label="Speedup (KMC/AtomWorld-Mirror)" if start == 0 else None,
            zorder=9,
        )
        for xi, value in zip(group_x, group_speedup):
            label = speed_ax.text(
                xi,
                float(value) + 5.0,
                _fmt_speedup_label(float(value)),
                ha="center",
                va="bottom",
                fontsize=10.0,
                color=DARK,
                fontweight="bold",
                zorder=20,
            )
            label.set_path_effects([path_effects.withStroke(linewidth=1.85, foreground="white")])
    speed_ax.set_ylim(0, max(160.0, float(speedup.max()) * 1.18))
    speed_ax.set_yticks([50, 100, 150])
    speed_ax.set_ylabel("Speed Up", fontsize=10.2, fontweight="bold", rotation=270, labelpad=12)
    speed_ax.tick_params(axis="y", which="major", direction="out", length=4.8, width=0.95, labelsize=8.3)
    speed_ax.spines["top"].set_visible(False)
    speed_ax.spines["bottom"].set_visible(False)
    speed_ax.spines["left"].set_visible(False)
    speed_ax.spines["right"].set_color("#111111")
    speed_ax.spines["right"].set_linewidth(1.02)
    ax_mid.spines["right"].set_visible(False)

    diag = 0.016
    slash_kwargs = dict(color="#111111", clip_on=False, lw=1.15, solid_capstyle="butt")
    ax_high.plot((-diag, +diag), (-diag, +diag), transform=ax_high.transAxes, **slash_kwargs)
    ax_high.plot((1 - diag, 1 + diag), (-diag, +diag), transform=ax_high.transAxes, **slash_kwargs)
    ax_mid.plot((-diag, +diag), (1 - diag, 1 + diag), transform=ax_mid.transAxes, **slash_kwargs)
    ax_mid.plot((1 - diag, 1 + diag), (1 - diag, 1 + diag), transform=ax_mid.transAxes, **slash_kwargs)
    ax_mid.plot((-diag, +diag), (-diag, +diag), transform=ax_mid.transAxes, **slash_kwargs)
    ax_mid.plot((1 - diag, 1 + diag), (-diag, +diag), transform=ax_mid.transAxes, **slash_kwargs)
    ax_low.plot((-diag, +diag), (1 - diag, 1 + diag), transform=ax_low.transAxes, **slash_kwargs)
    ax_low.plot((1 - diag, 1 + diag), (1 - diag, 1 + diag), transform=ax_low.transAxes, **slash_kwargs)

    fig.text(0.060, 0.49, "RunTime (s)", ha="center", va="center", rotation=90, fontsize=10.4, fontweight="bold")
    fig.text(0.5, 0.955, temp_text, ha="center", va="top", fontsize=9.0, fontweight="bold", color=DARK)
    fig.text(0.5, 0.920, lattice_text, ha="center", va="top", fontsize=8.7, fontweight="bold", color=DARK)

    handles1, labels1 = ax_high.get_legend_handles_labels()
    handles2, labels2 = speed_ax.get_legend_handles_labels()
    seen = set()
    handles = []
    legend_labels = []
    for handle, label in zip(handles1 + handles2, labels1 + labels2):
        if label and label not in seen:
            handles.append(handle)
            legend_labels.append(label)
            seen.add(label)
    legend = ax_high.legend(
        handles,
        legend_labels,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        ncol=3,
        fontsize=8.2,
        frameon=False,
        handlelength=1.55,
        columnspacing=1.05,
        borderaxespad=0.0,
    )
    for handle in getattr(legend, "legend_handles", getattr(legend, "legendHandles", [])):
        if hasattr(handle, "set_linewidth"):
            handle.set_linewidth(1.4)

    for text in fig.findobj(match=Text):
        text.set_fontfamily("Arial")
        text.set_fontweight("normal")

    fig.subplots_adjust(left=0.105, right=0.925, top=0.860, bottom=0.145)
    fig.savefig(_figure_output_path(output_name), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def fig3_speedup_benchmark():
    _fig4_case_grid_speedup_broken_axis_lowbars(
        0.005,
        "fig4_speedup_benchmark.pdf",
        data_path=FIG4_CASE_GRID_END_TO_END_SELECTED_NO100,
        atomworld_label="AtomWorld-Mirror",
    )


def fig4_case_grid_figtest_candidates():
    _fig4_case_grid_speedup(0.005, "fig4_speedup_case_grid_cu005.pdf")
    if FIG4_CASE_GRID_END_TO_END.exists():
        _fig4_case_grid_speedup(
            0.005,
            "fig4_speedup_case_grid_cu005_end_to_end_timing.pdf",
            data_path=FIG4_CASE_GRID_END_TO_END,
            atomworld_label="AtomWorld-Mirror",
        )
    if FIG4_CASE_GRID_END_TO_END_LARGE.exists():
        _fig4_case_grid_speedup(
            0.005,
            "fig4_speedup_case_grid_cu005_end_to_end_large_t263_373_l50_400.pdf",
            data_path=FIG4_CASE_GRID_END_TO_END_LARGE,
            atomworld_label="AtomWorld-Mirror",
        )


def _closed(name: str) -> dict:
    return json.loads((CLOSED_LOOP / name / "eval_closed_loop.json").read_text())


def _short_name(name: str) -> str:
    return (
        name.replace("strict_", "")
        .replace("abl_", "")
        .replace("_seed0", "")
        .replace("baseline_", "")
        .replace("_", "\n")
    )


def _fixedk_full_rows() -> list[dict[str, str]]:
    path = FIXEDK_MATRIX / "summary.csv"
    rows = list(csv.DictReader(path.open()))
    full = [row for row in rows if row.get("name", "").startswith("full_seed")]
    return sorted(full, key=lambda row: int(row.get("seed", 0)))


def _fixedk_rows_by_name() -> dict[str, dict[str, str]]:
    path = FIXEDK_MATRIX / "summary.csv"
    return {row["name"]: row for row in csv.DictReader(path.open())}


def _float_or_nan(value: object) -> float:
    if value in (None, ""):
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _fig3_time_stress_log_mae() -> float:
    """K-only pseudo-clock stress diagnostic calibrated at the 293K case."""
    low_payload = json.loads(FIG2_TEMPERATURE_CASES.read_text(encoding="utf-8"))
    high_payload = json.loads(FIG2_HIGH_TEMPERATURE_CASES.read_text(encoding="utf-8"))
    calibration_rows = [
        item
        for row in low_payload["rows"]
        if abs(float(row["case"]["temperature"]) - 293.0) < 1e-6
        for item in row["scatter_samples"]
    ]
    if not calibration_rows:
        return float("nan")
    alpha = math.exp(
        float(
            np.mean(
                [
                    math.log(
                        max(float(item["traditional_kmc_expected_tau"]), 1e-12)
                        / max(float(item["segment_k"]), 1.0)
                    )
                    for item in calibration_rows
                ]
            )
        )
    )
    stress_rows = [item for row in high_payload["rows"] for item in row["scatter_samples"]]
    if not stress_rows:
        return float("nan")
    true_tau = np.asarray([float(item["traditional_kmc_expected_tau"]) for item in stress_rows], dtype=np.float64)
    pred_tau = np.asarray([alpha * max(float(item["segment_k"]), 1.0) for item in stress_rows], dtype=np.float64)
    return float(
        np.mean(np.abs(np.log(np.clip(pred_tau, 1e-12, None)) - np.log(np.clip(true_tau, 1e-12, None))))
    )


FIG3_CU_VALUES = [0.005, 0.0134]
FIG3_TEMP_VALUES = [263.0, 293.0, 333.0, 373.0]
FIG3_ROW_SPECS = [
    ("full", "Full"),
    ("no_reachability", "No Local\nReachability"),
    ("no_inventory", "No Inventory\nConservation"),
    ("no_continuous_time", "No Continuous-Time\nConsistency"),
    ("no_duration", "No duration\nloss"),
    ("no_tau_exp", r"No $\tau_{\rm exp}$" + "\nloss"),
    ("no_projection", "No projection\nloss"),
    ("no_prior", "No prior\nrollout"),
    ("no_edit_context", "No edit\ncontext"),
]
FIG3_TRANSPOSED_COLUMN_LABELS = [
    ("full", "Full"),
    ("no_reachability", "No\nReach."),
    ("no_inventory", "No\nInv."),
    ("no_continuous_time", "No\nCT"),
    ("no_duration", "No\ndur."),
    ("no_tau_exp", r"No $\tau_{\rm exp}$"),
    ("no_projection", "No\nproj."),
    ("no_prior", "No\nprior"),
    ("no_edit_context", "No\nedit ctx"),
]
FIG3_METRIC_SPECS = [
    ("energy", "Energy\nerr."),
    ("reach", "Reach.\nviol."),
    ("inventory", "Inv.\nviol."),
    ("time", r"$\tau$\nMAE"),
    ("edit", "Edit\namp."),
]
FIG3_BLUE_HEATMAP = mcolors.LinearSegmentedColormap.from_list(
    "fig3_paper_blues",
    ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#08519c", "#08306b"],
)


def _fig3_tick_label(label: str) -> str:
    return label.replace("\\n", "\n")


def _fig3_cell_fontsize(fig: plt.Figure, ax: plt.Axes, label: str, x_count: int, base_size: float) -> float:
    cell_width_pt = fig.get_size_inches()[0] * 72.0 * ax.get_position().width / max(float(x_count), 1.0)
    max_width_pt = cell_width_pt * 0.78
    font_prop = FontProperties(family=FIG2_FONT)
    size = base_size
    while size > 2.35:
        width_pt = TextPath((0, 0), label, size=size, prop=font_prop).get_extents().width
        if width_pt <= max_width_pt:
            return size
        size -= 0.1
    return 2.35


def _fig3_cu_code(cu_density: float) -> str:
    return f"Cu{str(float(cu_density)).replace('.', 'p')}"


def _fig3_case_code(cu_density: float, temperature: float) -> str:
    return f"{_fig3_cu_code(cu_density)}_T{int(round(float(temperature)))}"


def _fig3_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fig3_temp_row(payload: dict, temperature: float) -> dict:
    rows = payload.get("rows", [])
    if not rows:
        return {}
    return min(rows, key=lambda row: abs(float(row["case"]["temperature"]) - float(temperature)))


def _fig3_k_clock_log_mae(row: dict) -> float:
    samples = row.get("scatter_samples", [])
    if not samples:
        return float("nan")
    true_tau = np.asarray([float(item["traditional_kmc_expected_tau"]) for item in samples], dtype=np.float64)
    pred_tau = np.asarray([max(float(item.get("segment_k", 1.0)), 1.0) for item in samples], dtype=np.float64)
    return float(np.mean(np.abs(np.log(np.clip(pred_tau, 1e-12, None)) - np.log(np.clip(true_tau, 1e-12, None)))))


def _fig3_result_bundle_ready() -> bool:
    closed_names = ["full", "no_reachability", "no_inventory", "no_continuous_time"]
    paired_names = ["full", "no_duration", "no_tau_exp", "no_projection", "no_prior", "no_edit_context"]
    for cu_density in FIG3_CU_VALUES:
        for temperature in FIG3_TEMP_VALUES:
            case_code = _fig3_case_code(cu_density, temperature)
            for name in closed_names:
                if not (FIG3_CU_TEMP_ABLATION / "closed_loop" / case_code / name / "eval_closed_loop.json").exists():
                    return False
        cu_code = _fig3_cu_code(cu_density)
        for name in paired_names:
            if not (FIG3_CU_TEMP_ABLATION / "paired" / cu_code / f"{name}.json").exists():
                return False
    return True


def _fig3_ratio(value: float, baseline: float, *, floor: float = 1e-4) -> float:
    if not np.isfinite(value):
        return float("nan")
    return float(value / max(float(baseline), floor))


def _fig3_ratio_label(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    if value >= 100:
        return f"{value:.0f}x"
    if value >= 10:
        return f"{value:.1f}x"
    return f"{value:.1f}x"


def _fig3_raw_label(metric: str, value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    if metric == "energy":
        return f"{value:.2f}" if abs(value) < 10 else f"{value:.1f}"
    if metric == "reach":
        return f"{value:.2f}"
    if metric == "inventory":
        return f"{value:.1f}"
    if metric == "time":
        return f"{value:.2f}" if abs(value) < 10 else f"{value:.1f}"
    if metric == "edit":
        return f"{value:.4f}"
    return f"{value:.2f}"


def _fig3_case_matrix(cu_density: float, temperature: float) -> dict:
    case_code = _fig3_case_code(cu_density, temperature)
    closed = {
        name: _fig3_json(FIG3_CU_TEMP_ABLATION / "closed_loop" / case_code / name / "eval_closed_loop.json")
        for name in ["full", "no_reachability", "no_inventory", "no_continuous_time"]
    }
    cu_code = _fig3_cu_code(cu_density)
    paired_payloads = {
        name: _fig3_json(FIG3_CU_TEMP_ABLATION / "paired" / cu_code / f"{name}.json")
        for name in ["full", "no_duration", "no_tau_exp", "no_projection", "no_prior", "no_edit_context"]
    }
    paired_rows = {name: _fig3_temp_row(payload, temperature) for name, payload in paired_payloads.items()}

    full_closed = closed["full"]
    full_pair = paired_rows["full"]
    full_energy = float(full_closed["physical_consistency"]["energy_abs_error_mean"])
    full_edit_error = 1.0 - float(full_pair["metrics"]["changed_type_acc"])
    full_time = float(full_closed["tau_expected"]["log_mae"])
    time_stress = _fig3_k_clock_log_mae(full_pair)

    values = np.full((len(FIG3_ROW_SPECS), len(FIG3_METRIC_SPECS)), np.nan, dtype=np.float64)
    severity_values = np.full_like(values, np.nan)
    labels: list[list[str]] = [["n/a" for _metric in FIG3_METRIC_SPECS] for _row in FIG3_ROW_SPECS]
    row_index = {key: idx for idx, (key, _label) in enumerate(FIG3_ROW_SPECS)}
    metric_index = {key: idx for idx, (key, _label) in enumerate(FIG3_METRIC_SPECS)}

    def set_cell(row_key: str, metric_key: str, value: float, label: str, severity_value: float | None = None) -> None:
        row_idx = row_index[row_key]
        metric_idx = metric_index[metric_key]
        values[row_idx, metric_idx] = float(value)
        severity_values[row_idx, metric_idx] = float(value if severity_value is None else severity_value)
        labels[row_idx][metric_idx] = label

    def set_raw(row_key: str, metric_key: str, value: float) -> None:
        set_cell(row_key, metric_key, value, _fig3_raw_label(metric_key, value))

    def set_multiplier(row_key: str, metric_key: str, value: float, baseline: float) -> None:
        ratio = _fig3_ratio(value, baseline)
        set_cell(row_key, metric_key, value, _fig3_ratio_label(ratio), ratio)

    for metric_key, value in [
        ("energy", full_energy),
        ("reach", float(full_closed["physical_consistency"]["reachability_violation_rate_mean"])),
        ("inventory", float(full_closed["physical_consistency"]["inventory_violation_l1_mean"])),
        ("time", full_time),
        ("edit", full_edit_error),
    ]:
        severity = 1.0 if metric_key in ("energy", "edit") else value
        set_cell("full", metric_key, value, _fig3_raw_label(metric_key, value), severity)

    for row_key, closed_key in [
        ("no_reachability", "no_reachability"),
        ("no_inventory", "no_inventory"),
        ("no_continuous_time", "no_continuous_time"),
    ]:
        row = closed[closed_key]
        phys = row["physical_consistency"]
        set_multiplier(row_key, "energy", float(phys["energy_abs_error_mean"]), full_energy)
        set_raw(row_key, "reach", float(phys["reachability_violation_rate_mean"]))
        set_raw(row_key, "inventory", float(phys["inventory_violation_l1_mean"]))
        if row_key == "no_continuous_time":
            set_raw(row_key, "time", time_stress)
        else:
            set_raw(row_key, "time", float(row["tau_expected"]["log_mae"]))

    for row_key, paired_key in [
        ("no_duration", "no_duration"),
        ("no_tau_exp", "no_tau_exp"),
        ("no_projection", "no_projection"),
        ("no_prior", "no_prior"),
        ("no_edit_context", "no_edit_context"),
    ]:
        row = paired_rows[paired_key]
        metrics = row["metrics"]
        paired_energy = float(metrics["reward_mae"]) / 10.0
        paired_edit_error = 1.0 - float(metrics["changed_type_acc"])
        set_multiplier(row_key, "energy", paired_energy, full_energy)
        set_raw(row_key, "reach", float(metrics["reachability_violation_rate"]))
        set_raw(row_key, "inventory", 0.0)
        if row_key in ("no_duration", "no_tau_exp"):
            set_raw(row_key, "time", time_stress)
        else:
            set_raw(row_key, "time", float(metrics["tau_log_mae"]))
        set_multiplier(row_key, "edit", paired_edit_error, full_edit_error)

    return {
        "cu_density": float(cu_density),
        "temperature": float(temperature),
        "values": values,
        "severity_values": severity_values,
        "labels": labels,
    }


def _fig3_draw_cu_temp_matrix(
    cases: list[dict],
    *,
    transposed: bool = False,
    output_name: str = "fig3_closed_loop_results.pdf",
    output_png_name: str | None = None,
) -> None:
    with plt.rc_context(FIG2_FONT_RC):
        _fig3_draw_cu_temp_matrix_body(
            cases,
            transposed=transposed,
            output_name=output_name,
            output_png_name=output_png_name,
        )


def _fig3_draw_cu_temp_matrix_body(
    cases: list[dict],
    *,
    transposed: bool = False,
    output_name: str = "fig3_closed_loop_results.pdf",
    output_png_name: str | None = None,
) -> None:
    all_severity = np.stack([case["severity_values"] for case in cases], axis=0)
    normalized = np.zeros_like(all_severity)
    for metric_idx, (metric_key, _metric_label) in enumerate(FIG3_METRIC_SPECS):
        data = all_severity[:, :, metric_idx]
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            continue
        safe = np.where(np.isfinite(data), data, 0.0)
        if metric_key in ("energy", "inventory", "edit"):
            safe = np.log1p(np.maximum(safe, 0.0))
            denom = max(float(np.nanmax(safe)), 1e-12)
            normalized[:, :, metric_idx] = safe / denom
        else:
            denom = max(float(np.nanmax(finite)), 1e-12)
            normalized[:, :, metric_idx] = np.where(np.isfinite(data), data / denom, 0.0)

    if transposed:
        fig, axes = plt.subplots(2, 4, figsize=(7.35, 3.45))
        fig.subplots_adjust(left=0.082, right=0.996, top=0.94, bottom=0.14, wspace=0.11, hspace=0.34)
    else:
        fig, axes = plt.subplots(2, 4, figsize=(7.35, 5.05))
        fig.subplots_adjust(left=0.145, right=0.995, top=0.965, bottom=0.07, wspace=0.12, hspace=0.22)
    for idx, case in enumerate(cases):
        ax = axes.flat[idx]
        panel = normalized[idx].T if transposed else normalized[idx]
        ax.imshow(panel, cmap=FIG3_BLUE_HEATMAP, vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_title(
            f"({chr(ord('a') + idx)}) Cu={case['cu_density']:.4g}, {int(round(case['temperature']))}K",
            loc="left",
            pad=3,
            fontsize=5.8,
            weight="bold",
        )
        if transposed:
            ax.set_xticks(
                np.arange(len(FIG3_TRANSPOSED_COLUMN_LABELS)),
                [_fig3_tick_label(label) for _key, label in FIG3_TRANSPOSED_COLUMN_LABELS],
            )
            if idx % 4 == 0:
                ax.set_yticks(
                    np.arange(len(FIG3_METRIC_SPECS)),
                    [_fig3_tick_label(label) for _key, label in FIG3_METRIC_SPECS],
                )
                ax.tick_params(axis="y", labelsize=4.7, length=0, pad=1.1)
            else:
                ax.set_yticks(np.arange(len(FIG3_METRIC_SPECS)), [""] * len(FIG3_METRIC_SPECS))
                ax.tick_params(axis="y", length=0)
            ax.tick_params(axis="x", labelsize=3.9, length=0, pad=1.2, rotation=42)
        else:
            ax.set_xticks(
                np.arange(len(FIG3_METRIC_SPECS)),
                [_fig3_tick_label(label) for _key, label in FIG3_METRIC_SPECS],
            )
            if idx % 4 == 0:
                ax.set_yticks(
                    np.arange(len(FIG3_ROW_SPECS)),
                    [_fig3_tick_label(label) for _key, label in FIG3_ROW_SPECS],
                )
                ax.tick_params(axis="y", labelsize=4.8, length=0, pad=1.2)
            else:
                ax.set_yticks(np.arange(len(FIG3_ROW_SPECS)), [""] * len(FIG3_ROW_SPECS))
                ax.tick_params(axis="y", length=0)
            ax.tick_params(axis="x", labelsize=4.7, length=0, pad=1.6, rotation=28)
        for tick in ax.get_xticklabels():
            tick.set_ha("right")
            tick.set_rotation_mode("anchor")
        for tick in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
            tick.set_fontname(FIG2_FONT)
        x_count = len(FIG3_ROW_SPECS) if transposed else len(FIG3_METRIC_SPECS)
        y_count = len(FIG3_METRIC_SPECS) if transposed else len(FIG3_ROW_SPECS)
        ax.set_xticks(np.arange(-0.5, x_count, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, y_count, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.78)
        ax.tick_params(which="minor", bottom=False, left=False)
        for y_idx in range(y_count):
            for x_idx in range(x_count):
                row_idx = x_idx if transposed else y_idx
                metric_idx = y_idx if transposed else x_idx
                label = case["labels"][row_idx][metric_idx]
                if label == "n/a":
                    label_size = _fig3_cell_fontsize(fig, ax, "n/a", x_count, 3.7)
                    ax.text(
                        x_idx,
                        y_idx,
                        "n/a",
                        ha="center",
                        va="center",
                        fontsize=label_size,
                        color="#8a939f",
                        fontname=FIG2_FONT,
                        clip_on=True,
                    )
                    continue
                severity = normalized[idx, row_idx, metric_idx]
                color = "white" if severity > 0.52 else "#222222"
                weight = "normal"
                base_label_size = 3.75 if transposed else 4.35
                label_size = _fig3_cell_fontsize(fig, ax, label, x_count, base_label_size)
                ax.text(
                    x_idx,
                    y_idx,
                    label,
                    ha="center",
                    va="center",
                    fontsize=label_size,
                    color=color,
                    weight=weight,
                    fontname=FIG2_FONT,
                    clip_on=True,
                )
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.savefig(_figure_output_path(output_name), bbox_inches="tight", pad_inches=0.018)
    if output_png_name is not None:
        fig.savefig(_figure_output_path(output_png_name), dpi=280, bbox_inches="tight", pad_inches=0.018)
    plt.close(fig)


def _fig3_closed_loop_results_cu_temp() -> None:
    cases = [
        _fig3_case_matrix(cu_density, temperature)
        for cu_density in FIG3_CU_VALUES
        for temperature in FIG3_TEMP_VALUES
    ]
    _fig3_draw_cu_temp_matrix(cases, transposed=True)


def fig2_closed_loop_results():
    if _fig3_result_bundle_ready():
        _fig3_closed_loop_results_cu_temp()
        return
    print(f"[fig3] missing complete {FIG3_CU_TEMP_ABLATION}; using legacy ablation matrix")
    _fig3_closed_loop_results_legacy()


def fig2_closed_loop_results_transposed_preview():
    if not _fig3_result_bundle_ready():
        print(f"[fig2-transposed-preview] missing complete {FIG3_CU_TEMP_ABLATION}; skipping preview")
        return
    cases = [
        _fig3_case_matrix(cu_density, temperature)
        for cu_density in FIG3_CU_VALUES
        for temperature in FIG3_TEMP_VALUES
    ]
    _fig3_draw_cu_temp_matrix(
        cases,
        transposed=True,
        output_name="fig2_closed_loop_results_transposed_preview.pdf",
        output_png_name="fig2_closed_loop_results_transposed_preview.png",
    )


def _fig3_closed_loop_results_legacy():
    constraint_rows = {row["name"]: row for row in csv.DictReader((MULTIK_CONSTRAINTS / "summary.csv").open())}
    closed_rows = {row["name"]: row for row in csv.DictReader((CLOSED_LOOP / "summary.csv").open())}
    paired_rows = _fixedk_rows_by_name()
    full_edit_error = 1.0 - _float_or_nan(paired_rows["full_seed0"]["changed_type_acc"])
    time_stress_log_mae = _fig3_time_stress_log_mae()

    def from_summary(
        row: dict[str, str],
        *,
        raw_edit_error: float | None = None,
        time_stress: float | None = None,
    ) -> dict[str, float | str | None]:
        completed = _float_or_nan(row.get("completed"))
        requested = _float_or_nan(row.get("requested"))
        scale = _float_or_nan(row.get("tau_scale_ratio"))
        return {
            "completed": completed,
            "requested": requested,
            "energy": _float_or_nan(row.get("energy_abs_error_mean")),
            "reach": _float_or_nan(row.get("reachability_violation_rate_mean")),
            "inventory": _float_or_nan(row.get("inventory_violation_l1_mean")),
            "tau_scale_err": abs(scale - 1.0) if np.isfinite(scale) else float("nan"),
            "tau_log_mae": _float_or_nan(row.get("tau_log_mae")),
            "time_stress": time_stress,
            "raw_edit_error": raw_edit_error,
        }

    def raw_edit_error(name: str) -> float:
        return 1.0 - _float_or_nan(paired_rows[name]["changed_type_acc"])

    columns = [
        (
            "Full",
            from_summary(constraint_rows["full_seed0"], raw_edit_error=full_edit_error),
        ),
        (
            "No Local\nReachability",
            from_summary(constraint_rows["no_reachability_seed0"]),
        ),
        (
            "No Inventory\nConservation",
            from_summary(constraint_rows["no_inventory_seed0"]),
        ),
        (
            "No Continuous-Time\nConsistency",
            from_summary(constraint_rows["no_continuous_time_seed0"], time_stress=time_stress_log_mae),
        ),
        (
            "No duration\nloss",
            from_summary(
                closed_rows["abl_no_duration_seed0"],
                raw_edit_error=raw_edit_error("abl_no_duration_seed0"),
                time_stress=time_stress_log_mae,
            ),
        ),
        (
            r"No $\tau_{\rm exp}$" + "\nloss",
            from_summary(
                closed_rows["abl_no_tau_exp_seed0"],
                raw_edit_error=raw_edit_error("abl_no_tau_exp_seed0"),
                time_stress=time_stress_log_mae,
            ),
        ),
        (
            "No projection\nloss",
            from_summary(closed_rows["abl_no_proj_loss_seed0"], raw_edit_error=raw_edit_error("abl_no_proj_loss_seed0")),
        ),
        (
            "No prior\nrollout",
            from_summary(closed_rows["abl_no_prior_rollout_seed0"], raw_edit_error=raw_edit_error("abl_no_prior_rollout_seed0")),
        ),
        (
            "No edit\ncontext",
            from_summary(closed_rows["abl_no_future_candidate_aug_seed0"], raw_edit_error=raw_edit_error("abl_no_future_candidate_aug_seed0")),
        ),
    ]

    metric_specs = [
        ("energy", "Energy\nerror"),
        ("reach", "Reachability\nviolation"),
        ("inventory", "Inventory\nviolation"),
        ("time_stress", "Time stress\nlog-MAE"),
        ("raw_edit_error", "Raw edit\nerror amp."),
    ]

    values = np.full((len(metric_specs), len(columns)), np.nan, dtype=float)
    labels: list[list[str]] = [["" for _ in columns] for _ in metric_specs]
    for col_idx, (_label, data) in enumerate(columns):
        values[0, col_idx] = float(data["energy"])
        values[1, col_idx] = float(data["reach"])
        values[2, col_idx] = float(data["inventory"])
        override_time_stress = data["time_stress"]
        values[3, col_idx] = (
            float(override_time_stress)
            if override_time_stress is not None and np.isfinite(float(override_time_stress))
            else float(data["tau_log_mae"])
        )
        raw_edit = data["raw_edit_error"]
        if raw_edit is not None and np.isfinite(float(raw_edit)):
            values[4, col_idx] = float(raw_edit)

    def raw_label(metric_idx: int, col_idx: int) -> str:
        if metric_idx == 0:
            value = values[0, col_idx]
            return f"{value:.2f}" if value < 10 else f"{value:.1f}"
        if metric_idx == 1:
            return f"{values[1, col_idx]:.2f}"
        if metric_idx == 2:
            return f"{values[2, col_idx]:.1f}"
        if metric_idx == 3:
            return f"{values[3, col_idx]:.2f}" if values[3, col_idx] < 10 else f"{values[3, col_idx]:.1f}"
        value = values[4, col_idx]
        return "n/a" if not np.isfinite(value) else f"{value:.4f}"

    def multiplier_label(value: float, full_value: float) -> str:
        if not np.isfinite(value):
            return "n/a"
        if abs(full_value) < 1e-12:
            return "1.0x" if abs(value) < 1e-12 else "inf"
        ratio = value / full_value
        if ratio >= 100:
            return f"{ratio:.0f}x"
        if ratio >= 10:
            return f"{ratio:.1f}x"
        return f"{ratio:.1f}x"

    for metric_idx in range(len(metric_specs)):
        full_value = values[metric_idx, 0]
        for col_idx in range(len(columns)):
            if col_idx == 0:
                labels[metric_idx][col_idx] = raw_label(metric_idx, col_idx)
            elif metric_idx in (1, 2, 3):
                labels[metric_idx][col_idx] = raw_label(metric_idx, col_idx)
            else:
                labels[metric_idx][col_idx] = multiplier_label(values[metric_idx, col_idx], full_value)

    severity = np.zeros_like(values)
    for row_idx in range(values.shape[0]):
        row = values[row_idx]
        finite = row[np.isfinite(row)]
        if len(finite) == 0:
            continue
        if row_idx == 4:
            baseline = max(float(values[row_idx, 0]), 1e-12)
            ratios = np.where(np.isfinite(row), row / baseline, 0.0)
            transformed = np.log1p(ratios)
            denom = max(float(np.nanmax(transformed)), 1e-12)
            severity[row_idx] = transformed / denom
        elif row_idx in (0, 2):
            transformed = np.log1p(np.where(np.isfinite(row), row, 0.0))
            denom = max(float(np.nanmax(transformed)), 1e-12)
            severity[row_idx] = transformed / denom
        elif row_idx == 1:
            denom = max(float(np.nanmax(finite)), 1e-12)
            severity[row_idx] = np.where(np.isfinite(row), row / denom, 0.0)
        else:
            denom = max(float(np.nanmax(finite)), 1e-12)
            severity[row_idx] = np.where(np.isfinite(row), row / denom, 0.0)

    display_severity = severity.T
    display_labels = [[labels[row_idx][col_idx] for row_idx in range(len(metric_specs))] for col_idx in range(len(columns))]

    fig, ax = plt.subplots(1, 1, figsize=(7.15, 3.45))
    ax.imshow(display_severity, cmap=FIG3_BLUE_HEATMAP, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(metric_specs)), [label for _, label in metric_specs])
    ax.set_yticks(np.arange(len(columns)), [label for label, _data in columns])
    ax.tick_params(axis="x", labelsize=6.75, length=0, pad=5)
    ax.tick_params(axis="y", labelsize=6.45, length=0)
    ax.set_title("Ablation matrix", loc="left", pad=8, fontsize=8.4, weight="bold")
    ax.set_xticks(np.arange(-0.5, len(metric_specs), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(columns), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.15)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i in range(len(columns)):
        for j in range(len(metric_specs)):
            if display_labels[i][j] == "n/a":
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=6.45, color="#8a939f")
                continue
            color = "white" if display_severity[i, j] > 0.52 else "#222222"
            weight = "normal"
            ax.text(j, i, display_labels[i][j], ha="center", va="center", fontsize=6.7, color=color, weight=weight)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.subplots_adjust(left=0.245, right=0.995, top=0.88, bottom=0.18)
    fig.savefig(_figure_output_path("fig3_closed_loop_results.pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def fig4_multik_duration_seed_stability():
    rows = _fixedk_full_rows()
    seeds = np.asarray([int(row["seed"]) for row in rows], dtype=int)
    ratios = np.asarray([float(row["long_expected_time_ratio"]) for row in rows], dtype=float)
    log_mae = np.asarray([float(row["long_tau_log_mae"]) for row in rows], dtype=float)
    mean = float(np.mean(ratios))
    std = float(np.std(ratios, ddof=1)) if len(ratios) > 1 else 0.0

    fig, ax = plt.subplots(1, 1, figsize=(5.8, 2.45))
    ax.axhspan(mean - std, mean + std, color=SOFT_BLUE, alpha=0.28, lw=0)
    ax.axhline(1.0, color=DARK, lw=0.85, ls="--", alpha=0.9, label="KMC reference")
    ax.axhline(mean, color=SOFT_BLUE_DARK, lw=1.25, label=f"mean {mean:.3f}")
    scatter = ax.scatter(
        seeds,
        ratios,
        s=38,
        c=log_mae,
        cmap="viridis_r",
        edgecolor=DARK,
        linewidth=0.45,
        zorder=4,
    )
    for seed, ratio in zip(seeds, ratios):
        ax.text(seed, ratio + 0.010, str(seed), ha="center", va="bottom", fontsize=6.8, color=DARK)
    ax.set_xlim(float(np.min(seeds)) - 0.55, float(np.max(seeds)) + 0.55)
    ymin = min(0.94, float(np.min(ratios)) - 0.035)
    ymax = max(1.07, float(np.max(ratios)) + 0.045)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(seeds)
    ax.set_xlabel("evaluation seed")
    ax.set_ylabel(r"cumulative $\tau_{\rm exp}$ ratio")
    ax.set_title("Duration calibration", loc="left", pad=6, fontsize=8.5, weight="bold")
    ax.text(
        0.995,
        0.97,
        f"mean={mean:.3f}\nstd={std:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.2,
        color=DARK,
        bbox={"facecolor": "white", "edgecolor": "#d7dce4", "alpha": 0.92, "pad": 2.0},
    )
    cbar = fig.colorbar(scatter, ax=ax, pad=0.012, fraction=0.05)
    cbar.set_label(r"$\tau_{\rm exp}$ log-MAE", fontsize=7.0)
    cbar.ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", color="#d9dee8", lw=0.5, alpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left", fontsize=6.6, frameon=False, handlelength=1.6)
    fig.subplots_adjust(left=0.12, right=0.93, top=0.86, bottom=0.20)
    fig.savefig(_figure_output_path("fig4_multik_duration_seed_stability.pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _draw_fig5_base_axes(fig: plt.Figure, axes: list[plt.Axes]) -> None:
    fixedk_full = _fixedk_full_rows()
    fixedk_rows = _fixedk_rows_by_name()
    strict_names = [
        "baseline_no_change_seed0",
        "strict_full_seed0",
        "strict_no_continuous_time_seed0",
        "strict_no_reachability_seed0",
        "strict_no_constraints_seed0",
    ]
    strict = [_closed(name) for name in strict_names]
    ablation_names = [
        "full_seed0",
        "abl_no_duration_seed0",
        "abl_no_tau_exp_seed0",
        "abl_no_proj_loss_seed0",
        "abl_no_prior_rollout_seed0",
        "abl_no_future_candidate_aug_seed0",
    ]
    ablations = [fixedk_rows[name] for name in ablation_names]

    ax1, ax2, ax3 = axes
    fig5_blue = "#aecce4"
    fig5_blue_dark = "#83b7cf"
    fig5_pink = "#efbcb8"
    fig5_pink_dark = "#deb4b5"
    fig5_green = "#cce4c2"
    fig5_green_dark = "#bbd4bf"
    fig5_yellow = "#eae0ab"
    fig5_yellow_dark = "#dcd2a1"
    fig5_neutral = "#e9e5e6"
    fig5_edge = "#6f7f8b"

    ratios = np.array([float(item["long_expected_time_ratio"]) for item in fixedk_full], dtype=float)
    log_mae = np.array([float(item["long_tau_log_mae"]) for item in fixedk_full], dtype=float)
    seed_labels = [f"seed {int(item['seed'])}" for item in fixedk_full]
    x = np.arange(len(seed_labels))
    ax1.bar(x, ratios, color=fig5_blue, edgecolor=fig5_edge, linewidth=0.55, alpha=0.98, width=0.58)
    ax1.axhline(1.0, color=DARK, lw=0.9, ls="--")
    for i, (r, m) in enumerate(zip(ratios, log_mae)):
        ax1.text(i, r + 0.025, f"{r:.3f}x", ha="center", va="bottom", fontsize=7.2)
        ax1.text(i, 0.075, f"MAE\n{m:.2f}", ha="center", va="bottom", fontsize=6.6, color=DARK, weight="bold")
    ax1.set_xticks(x, seed_labels)
    ax1.set_ylim(0, 1.22)
    ax1.tick_params(axis="both", labelsize=6.8)
    ax1.set_ylabel(r"cumulative $\tau_{\rm exp}$ ratio", labelpad=1.5)
    ax1.set_title("(a) Duration calibration", loc="left", pad=5, fontsize=7.8)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.grid(axis="y", color="#d9dee8", lw=0.5, alpha=0.8)

    vals = np.array([item["physical_consistency"]["energy_abs_error_mean"] for item in strict], dtype=float)
    labs = ["copy\nstate", "full", "CTMC\ntime\nbase.", "no\nreach.", "no\nconst."]
    colors = [fig5_neutral, fig5_blue, fig5_green, fig5_pink, fig5_pink_dark]
    ax2.bar(np.arange(len(vals)), vals, color=colors, edgecolor=fig5_edge, linewidth=0.55, alpha=0.98, width=0.58)
    ymax = max(vals) * 1.20
    for i, v in enumerate(vals):
        label = f"{v:.2f}" if v < 10 else f"{v:.1f}"
        ax2.text(i, v + max(vals) * 0.018, label, ha="center", va="bottom", fontsize=6.6)
    ax2.set_xticks(np.arange(len(vals)), labs)
    ax2.set_ylim(0, ymax)
    ax2.tick_params(axis="both", labelsize=6.7)
    ax2.set_ylabel("mean energy error", labelpad=1.5)
    ax2.set_title("(b) Physical-constraint ablations", loc="left", pad=5, fontsize=7.8)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="y", color="#d9dee8", lw=0.5, alpha=0.8)

    acc = np.array([float(item["projected_changed_type_acc"]) for item in ablations], dtype=float)
    err = 1.0 - acc
    vals = err / max(err[0], 1e-12)
    labs = ["full", "no\ndur.", "no\n$\\tau_{exp}$", "no proj.\nloss", "no prior\nroll.", "no edit\nctx."]
    ax3.bar(
        np.arange(len(vals)),
        vals,
        color=[fig5_blue, fig5_green, fig5_green_dark, fig5_yellow, fig5_yellow_dark, fig5_pink],
        edgecolor=fig5_edge,
        linewidth=0.55,
        alpha=0.98,
        width=0.58,
    )
    for i, v in enumerate(vals):
        ax3.text(i, v + max(vals) * 0.025, f"{v:.1f}x", ha="center", va="bottom", fontsize=6.8)
    ax3.axhline(1.0, color=fig5_blue_dark, lw=0.8, ls="--", alpha=0.85)
    ax3.set_xticks(np.arange(len(vals)), labs)
    ax3.set_ylim(0, max(vals) * 1.16)
    ax3.set_ylabel("projected edit error / full", labelpad=1.5)
    ax3.tick_params(axis="both", labelsize=6.5)
    ax3.set_title("(c) Sparse-edit error amplification", loc="left", pad=5, fontsize=7.8)
    ax3.spines[["top", "right"]].set_visible(False)
    ax3.grid(axis="y", color="#d9dee8", lw=0.5, alpha=0.8)


def _temperature_sweep_by_temperature(payload: dict) -> list[dict[str, float]]:
    grouped: dict[float, list[dict]] = {}
    for row in payload.get("rows", []):
        temp = float(row["case"]["temperature"])
        grouped.setdefault(temp, []).append(row)
    if not grouped:
        raise ValueError("Temperature-sweep payload has no rows.")

    summaries: list[dict[str, float]] = []
    for temp, rows in sorted(grouped.items()):
        projected = np.asarray(
            [float(row["metrics"]["projected_changed_type_acc"]) for row in rows],
            dtype=np.float64,
        )
        tau_scale = np.asarray(
            [float(row["metrics"]["tau_scale_ratio"]) for row in rows],
            dtype=np.float64,
        )
        coverage = np.asarray(
            [float(row["collection_stats"]["coverage"]) for row in rows],
            dtype=np.float64,
        )
        reachability_violation = np.asarray(
            [float(row["metrics"]["reachability_violation_rate"]) for row in rows],
            dtype=np.float64,
        )
        summaries.append(
            {
                "temperature": float(temp),
                "projected_mean": float(projected.mean()),
                "projected_min": float(projected.min()),
                "projected_max": float(projected.max()),
                "tau_scale_mean": float(tau_scale.mean()),
                "tau_scale_min": float(tau_scale.min()),
                "tau_scale_max": float(tau_scale.max()),
                "coverage_mean": float(coverage.mean()),
                "reachability_violation_max": float(reachability_violation.max()),
                "n_cases": float(len(rows)),
            }
        )
    return summaries


def _style_temperature_strip(ax: plt.Axes, color: str, label: str) -> None:
    ax.set_xlim(235, 515)
    ax.set_xticks([250, 500])
    ax.set_xlabel("T(K)", fontsize=5.4, labelpad=0.8)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_ylabel(label, color=color, fontsize=5.6, labelpad=1.8)
    ax.tick_params(axis="x", labelsize=5.2, pad=1.0, colors=GREY, width=0.45, length=2.0)
    ax.tick_params(axis="y", labelsize=5.2, pad=1.0, colors=color, width=0.45, length=2.0)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_color("#cfd6df")
    ax.spines["right"].set_color(color)
    ax.spines["right"].set_linewidth(0.65)
    ax.grid(axis="y", color="#e5e9f0", lw=0.4, alpha=0.9)


def fig3_closed_loop_dualaxis_preview():
    payload = json.loads(MULTICASE_FIG3_TEMP_SWEEP.read_text())
    temp_rows = _temperature_sweep_by_temperature(payload)

    fig = plt.figure(figsize=(8.15, 2.62))
    outer = fig.add_gridspec(1, 3, width_ratios=[1.45, 1.86, 2.55], wspace=0.36)
    group_specs = [
        outer[0].subgridspec(1, 2, width_ratios=[1.03, 0.36], wspace=0.06),
        outer[1].subgridspec(1, 2, width_ratios=[1.42, 0.38], wspace=0.06),
        outer[2].subgridspec(1, 2, width_ratios=[2.02, 0.45], wspace=0.06),
    ]
    axes = [fig.add_subplot(spec[0, 0]) for spec in group_specs]
    strip_axes = [fig.add_subplot(spec[0, 1]) for spec in group_specs]
    _draw_fig5_base_axes(fig, axes)

    temps = np.asarray([row["temperature"] for row in temp_rows], dtype=float)
    projected = np.asarray([row["projected_mean"] for row in temp_rows], dtype=float)
    projected_min = np.asarray([row["projected_min"] for row in temp_rows], dtype=float)
    projected_max = np.asarray([row["projected_max"] for row in temp_rows], dtype=float)
    tau_scale = np.asarray([row["tau_scale_mean"] for row in temp_rows], dtype=float)
    reachability = np.asarray([row["reachability_violation_max"] for row in temp_rows], dtype=float)

    ax_tau, ax_reach, ax_edit = strip_axes
    ax_tau.plot(
        temps,
        tau_scale,
        color=SOFT_PINK_DARK,
        linewidth=1.15,
        marker="D",
        markersize=2.8,
        markerfacecolor="white",
        markeredgewidth=0.7,
    )
    ax_tau.axhline(1.0, color="#8793a0", lw=0.55, ls="--", alpha=0.75)
    ax_tau.set_ylim(0.55, max(7.35, float(tau_scale.max()) * 1.04))
    _style_temperature_strip(ax_tau, SOFT_PINK_DARK, r"$\tau$ scale")

    ax_reach.plot(
        temps,
        reachability * 100.0,
        color=SOFT_PINK_DARK,
        linewidth=1.1,
        marker="v",
        markersize=2.9,
        markerfacecolor=SOFT_PINK,
        markeredgewidth=0.5,
        markeredgecolor=SOFT_EDGE,
    )
    ax_reach.set_ylim(0.0, max(7.0, float(reachability.max() * 100.0) * 1.2))
    _style_temperature_strip(ax_reach, SOFT_PINK_DARK, "viol. %")

    ax_edit.fill_between(temps, projected_min, projected_max, color=SOFT_BLUE, alpha=0.26, linewidth=0)
    ax_edit.plot(
        temps,
        projected,
        color=SOFT_BLUE_DARK,
        linewidth=1.15,
        marker="s",
        markersize=2.8,
        markerfacecolor=SOFT_BLUE,
        markeredgewidth=0.5,
        markeredgecolor=SOFT_EDGE,
    )
    ax_edit.set_ylim(0.15, max(0.72, float(projected_max.max()) * 1.08))
    _style_temperature_strip(ax_edit, SOFT_BLUE_DARK, "edit acc.")

    fig.subplots_adjust(left=0.063, right=0.985, top=0.84, bottom=0.28)
    fig.savefig(_figure_output_path("fig3_closed_loop_results_dualaxis_preview.pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(_figure_output_path("fig3_closed_loop_results_dualaxis_preview.png"), dpi=280, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _plot_snapshot(ax, snap: dict, key: str, title: str, box: float = 80.0):
    cu = np.asarray(snap[key]["cu"], dtype=float)
    vac = np.asarray(snap[key]["vacancies"], dtype=float)
    ax.scatter(cu[:, 0], cu[:, 1], cu[:, 2], s=1.7, c=BLUE if key == "model" else GREY, alpha=0.26, depthshade=False)
    if vac.size:
        ax.scatter(vac[:, 0], vac[:, 1], vac[:, 2], s=20, c=RED, marker="x", linewidths=0.9, depthshade=False)
    ax.set_xlim(0, box)
    ax.set_ylim(0, box)
    ax.set_zlim(0, box)
    ax.view_init(elev=18, azim=-58)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_title(title, pad=1.0, fontsize=7.2)
    ax.set_box_aspect((1, 1, 0.82))
    ax.grid(False)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.set_edgecolor("#d9dee8")
        axis.pane.set_facecolor((1, 1, 1, 0.0))


def _periodic_delta(points: np.ndarray, center: np.ndarray, box_dims: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    box = np.asarray(box_dims, dtype=np.float64)
    delta = points - center
    return (delta + 0.5 * box) % box - 0.5 * box


def _pair_vacancy_moves(
    *,
    positions: np.ndarray,
    current_types: np.ndarray,
    target_types: np.ndarray,
    candidate_mask: np.ndarray,
    box_dims: np.ndarray,
) -> list[tuple[int, int]]:
    valid = candidate_mask > 0
    teacher_changed = valid & (target_types != current_types)
    vac_start = np.flatnonzero(teacher_changed & (current_types == 2) & (target_types != 2))
    vac_end = np.flatnonzero(teacher_changed & (current_types != 2) & (target_types == 2))
    unused = set(int(idx) for idx in vac_end.tolist())
    pairs: list[tuple[int, int]] = []
    for src in vac_start.tolist():
        atom_type = int(target_types[src])
        typed = [idx for idx in unused if int(current_types[idx]) == atom_type]
        pool = typed if typed else list(unused)
        if not pool:
            break
        dst = min(
            pool,
            key=lambda idx: float(np.linalg.norm(_periodic_delta(positions[idx], positions[src], box_dims))),
        )
        pairs.append((int(src), int(dst)))
        unused.discard(int(dst))
    return pairs


def _load_fig8_sparse_edit_example(sample_index: int = 376) -> dict[str, object] | None:
    ckpt_path = ROOT / "dreamer4-main" / "results" / "kmc_teacher_dreamer_macro_wm" / "final_model.pt"
    cache_path = ROOT / "dreamer4-main" / "results" / "kmc_teacher_dreamer_macro_wm" / "segments.pt"
    if not ckpt_path.exists() or not cache_path.exists():
        return None
    for path in [ROOT / "dreamer4-main", ROOT / "kmcteacher_backend", ROOT / "LightZero-main"]:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    import train_dreamer_macro_edit as mod
    import eval_macro_time_alignment as eval_time

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    with contextlib.redirect_stdout(io.StringIO()):
        model = eval_time._build_model(ckpt, "cpu")
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    raw_sample = payload["val"][sample_index]
    sample = mod.MacroSegmentSample(**raw_sample)
    tensors = mod._batch_to_device([sample], "cpu")
    with torch.no_grad():
        global_latent = model.encode_global(tensors["start_obs"])
        prior_mu, prior_logvar = model.prior_stats(
            global_latent,
            tensors["global_summary"],
            tensors["horizon_k"],
        )
        path_latent = model.sample_path_latent(prior_mu, prior_logvar, deterministic=True)
        next_pred = model.predict_next_global(global_latent, path_latent, tensors["horizon_k"])
        site_latent, patch_latent = model.encode_patch(
            positions=tensors["candidate_positions"],
            nearest_vacancy_offset=tensors["nearest_vacancy_offset"],
            reach_depth=tensors["reach_depth"],
            is_start_vacancy=tensors["is_start_vacancy"],
            type_ids=tensors["current_types"],
            node_mask=tensors["candidate_mask"],
            global_summary=tensors["global_summary"],
            box_dims=tensors["box_dims"],
        )
        change_logits, type_logits = model.decode_edit(
            site_latent=site_latent,
            patch_latent=patch_latent,
            predicted_next_global=next_pred,
            path_latent=path_latent,
            horizon_k=tensors["horizon_k"],
            current_types=tensors["current_types"],
        )
        projected_types, _projected_changed_mask, transport_cost, violation = mod.project_types_by_inventory(
            current_types=tensors["current_types"],
            change_logits=change_logits,
            type_logits=type_logits,
            node_mask=tensors["candidate_mask"],
            positions=tensors["candidate_positions"],
            box_dims=tensors["box_dims"],
            horizon_k=tensors["horizon_k"],
            max_changed_sites=2 * tensors["horizon_k"],
        )

    positions = np.asarray(sample.candidate_positions, dtype=np.float64)
    current_types = np.asarray(sample.current_types, dtype=np.int64)
    target_types = np.asarray(sample.target_types, dtype=np.int64)
    candidate_mask = np.asarray(sample.candidate_mask, dtype=np.float64)
    projected_np = projected_types[0].cpu().numpy().astype(np.int64)
    valid = candidate_mask > 0
    teacher_changed = valid & (target_types != current_types)
    model_changed = valid & (projected_np != current_types)
    strict_correct = teacher_changed & model_changed & (projected_np == target_types)
    overlap = teacher_changed & model_changed
    precision = float(overlap.sum() / max(model_changed.sum(), 1))
    recall = float(overlap.sum() / max(teacher_changed.sum(), 1))
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-9)
    changed_type_acc = (
        float((projected_np[teacher_changed] == target_types[teacher_changed]).mean())
        if teacher_changed.any()
        else 1.0
    )
    pairs = _pair_vacancy_moves(
        positions=positions,
        current_types=current_types,
        target_types=target_types,
        candidate_mask=candidate_mask,
        box_dims=np.asarray(sample.box_dims, dtype=np.float64),
    )
    return {
        "sample_index": sample_index,
        "positions": positions,
        "current_types": current_types,
        "target_types": target_types,
        "projected_types": projected_np,
        "candidate_mask": candidate_mask,
        "box_dims": np.asarray(sample.box_dims, dtype=np.float64),
        "teacher_changed": teacher_changed,
        "model_changed": model_changed,
        "strict_correct": strict_correct,
        "pairs": pairs,
        "horizon_k": int(sample.horizon_k),
        "reward_sum": float(sample.reward_sum),
        "transport_cost": float(transport_cost.item()),
        "violation": float(violation.item()),
        "change_f1": float(f1),
        "changed_type_acc": changed_type_acc,
    }


def _prepare_fig8_local_cloud(example: dict[str, object]) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    positions = example["positions"]
    candidate_mask = example["candidate_mask"] > 0
    box_dims = example["box_dims"]
    pairs = example["pairs"]
    cloud_points = []
    cloud_indices = []
    arrows = []
    cluster_gap = 6.0
    for cluster_idx, (src, dst) in enumerate(pairs):
        center = positions[src]
        local_all = _periodic_delta(positions, center, box_dims)
        dist_src = np.linalg.norm(local_all, axis=1)
        dist_dst = np.linalg.norm(_periodic_delta(positions, positions[dst], box_dims), axis=1)
        keep = candidate_mask & ((dist_src <= 2.1) | (dist_dst <= 2.1))
        keep[src] = True
        keep[dst] = True
        local = local_all[keep].copy()
        local[:, 0] += cluster_idx * cluster_gap
        indices = np.flatnonzero(keep)
        cloud_points.append(local)
        cloud_indices.append(indices)
        start = _periodic_delta(positions[src], center, box_dims)
        end = _periodic_delta(positions[dst], center, box_dims)
        start[0] += cluster_idx * cluster_gap
        end[0] += cluster_idx * cluster_gap
        arrows.append((start, end))
    return np.vstack(cloud_points), np.concatenate(cloud_indices), arrows


def _style_fig8_3d_axis(ax, title: str, xlim: tuple[float, float]):
    ax.set_title(title, pad=0.5, fontsize=8.2, fontweight="bold")
    ax.view_init(elev=18, azim=-54)
    ax.set_xlim(*xlim)
    ax.set_ylim(-2.8, 2.8)
    ax.set_zlim(-2.8, 2.8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_box_aspect((2.45, 1.0, 0.90))
    ax.grid(False)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.set_edgecolor("#dfe4ec")
        axis.pane.set_facecolor((1, 1, 1, 0.0))


def _scatter_type_cloud(ax, points: np.ndarray, indices: np.ndarray, stage_types: np.ndarray, highlight_mask: np.ndarray | None = None):
    type_style = {
        0: ("#b7c0cc", "o", 15, 0.45, "Fe"),
        1: (BLUE, "o", 24, 0.75, "Cu"),
        2: (RED, "X", 34, 0.88, "Vacancy"),
    }
    for type_id, (color, marker, size, alpha, _label) in type_style.items():
        mask = stage_types[indices] == type_id
        if mask.any():
            ax.scatter(
                points[mask, 0],
                points[mask, 1],
                points[mask, 2],
                s=size,
                c=color,
                marker=marker,
                alpha=alpha,
                edgecolors="white" if marker != "X" else DARK,
                linewidths=0.25 if marker != "X" else 0.35,
                depthshade=False,
            )
    if highlight_mask is not None:
        mask = highlight_mask[indices]
        if mask.any():
            ax.scatter(
                points[mask, 0],
                points[mask, 1],
                points[mask, 2],
                s=74,
                facecolors="none",
                edgecolors=DARK,
                linewidths=0.95,
                depthshade=False,
            )


def fig7_ctmc_semimarkov_timeline():
    _payload, ks, rewards, _pred_rewards, tau, _tau_pred, event_edges = _load_multik_long()
    n_show = 18
    ks = ks[:n_show]
    rewards = rewards[:n_show]
    tau = tau[:n_show]
    starts = event_edges[:n_show]
    ends = event_edges[1 : n_show + 1]
    total_micro = int(ends[-1])

    max_abs = max(float(np.max(np.abs(rewards))) if len(rewards) else 1.0, 1e-12)
    reward_norm = rewards / max_abs
    tau_norm = tau / max(float(np.max(tau)) if len(tau) else 1.0, 1e-12)

    fig = plt.figure(figsize=(7.05, 3.18))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.55, 0.95, 1.15], hspace=0.20)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax2 = fig.add_subplot(gs[2], sharex=ax0)

    micro_ticks = [((x, 0.25), (x, 0.78)) for x in range(total_micro + 1)]
    lc = LineCollection(micro_ticks, colors="#dce7eb", linewidths=0.62, alpha=0.95)
    ax0.add_collection(lc)
    for x in starts:
        ax0.plot([x, x], [0.10, 0.93], color=SOFT_GREEN_DARK, lw=0.95, alpha=0.62)
    ax0.set_xlim(0, total_micro)
    ax0.set_ylim(0, 1)
    ax0.set_yticks([])
    ax0.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax0.text(-2.5, 0.50, "CTMC\nmicro events", ha="right", va="center", color=GREY, fontsize=7.5, weight="bold")
    ax0.spines[["top", "right", "left", "bottom"]].set_visible(False)

    ax1.set_ylim(0, 1)
    ax1.set_yticks([])
    ax1.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax1.text(-2.5, 0.50, "multi-k\nmacro steps", ha="right", va="center", color=SOFT_GREEN_DARK, fontsize=7.5, weight="bold")
    lane_y = {2: 0.22, 4: 0.50, 8: 0.78}
    k_palette = {2: SOFT_BLUE, 4: SOFT_GREEN, 8: SOFT_PINK}
    k_text_palette = {2: SOFT_BLUE_DARK, 4: SOFT_GREEN_DARK, 8: SOFT_PINK_DARK}
    for i, (a, b, k) in enumerate(zip(starts, ends, ks)):
        k = int(k)
        color = k_palette.get(k, SOFT_YELLOW)
        y = lane_y.get(k, 0.50)
        ax1.add_patch(Rectangle((float(a), y - 0.085), float(b - a), 0.17,
                                fc=color, ec=SOFT_EDGE, lw=0.6, alpha=0.98))
    for k in [2, 4, 8]:
        ax1.text(total_micro + 1.5, lane_y[k], rf"$k={k}$", ha="left", va="center",
                 fontsize=6.8, color=k_text_palette[k], weight="bold")
    ax1.spines[["top", "right", "left", "bottom"]].set_visible(False)

    bar_colors = np.where(rewards > 1e-9, SOFT_PINK, np.where(rewards < -1e-9, SOFT_GREEN, SOFT_NEUTRAL))
    ax2.axhline(0, color="#9aa6b7", lw=0.7)
    ax2.bar(
        ends - ks / 2,
        reward_norm,
        width=np.maximum(ks * 0.68, 1.1),
        color=bar_colors,
        edgecolor=SOFT_EDGE,
        linewidth=0.35,
        alpha=0.98,
    )
    duration_y = tau_norm * 0.42 - 0.92
    ax2.plot(ends, duration_y, color=SOFT_BLUE_DARK, lw=1.45, marker="o", ms=2.8)
    ax2.set_ylim(-1.05, 1.05)
    ax2.set_xlabel("cumulative teacher micro-event index")
    ax2.set_ylabel("reward /\nduration", rotation=0, ha="right", va="center", labelpad=27, color=GREY)
    ax2.set_yticks([-1, 0, 1])
    ax2.text(
        0.985,
        0.88,
        "teacher reward",
        transform=ax2.transAxes,
        ha="right",
        va="top",
        fontsize=6.8,
        color=SOFT_PINK_DARK,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.2},
    )
    ax2.text(
        0.985,
        0.18,
        r"duration $\tau_{\rm exp}$",
        transform=ax2.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.8,
        color=SOFT_BLUE_DARK,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.2},
    )
    ax2.grid(axis="y", color="#d9dee8", lw=0.45, alpha=0.85)
    ax2.spines[["top", "right"]].set_visible(False)
    fig.suptitle("CTMC micro-events to multi-k Semi-Markov macro steps", x=0.52, y=0.99, fontsize=9.2)
    fig.subplots_adjust(left=0.17, right=0.90, top=0.86, bottom=0.17, hspace=0.23)
    fig.savefig(_figure_output_path("fig7_ctmc_semimarkov_timeline.pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    setup()
    fig2()
    fig3()
    fig1_main_results()
    fig2_main_results_temperature_cases_preview()
    fig2_closed_loop_results()
    fig3_speedup_benchmark()
    fig7_ctmc_semimarkov_timeline()
    fig8_realized_time_calibration()


if __name__ == "__main__":
    main()
