#!/usr/bin/env python3
"""Generate an appendix visualization for AtomWorld-Mirror.

The output mirrors the whole-box / zoomed-box layout requested for the
supplementary material while preserving a figtest preview copy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree


BASE_DIR = Path(__file__).resolve().parent
FIG_OUT = BASE_DIR / "fig"
FIGTEST_OUT = BASE_DIR / "figtest"
OFFICIAL_PDF_OUT = FIG_OUT / "fig5_cu_cluster_evolution.pdf"
PNG_OUT = FIGTEST_OUT / "atomworld_mirror_appendix_visualization_preview.png"
PDF_OUT = FIGTEST_OUT / "atomworld_mirror_appendix_visualization_preview.pdf"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_SNAPSHOT_SOURCE = (
    PROJECT_ROOT
    / "dreamer4-main"
    / "results"
    / "visualization_teacher"
    / "teacher_50000step_fullcu_initial_final_seed0.snapshots.json"
)


def _cluster_sizes(points: np.ndarray, box: float, radius: float = 2.01) -> np.ndarray:
    """Return true Cu cluster sizes using periodic BCC 1NN/2NN connectivity."""
    if len(points) == 0:
        return np.zeros((0,), dtype=np.float32)
    # Coordinates are stored on the doubled BCC lattice. The first two BCC
    # shells have squared distances 3 and 4, so r=2.01 connects 1NN and 2NN Cu.
    tree = cKDTree(np.asarray(points, dtype=np.float64), boxsize=float(box))
    pairs = tree.query_pairs(r=radius)
    parent = np.arange(points.shape[0], dtype=np.int32)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in pairs:
        union(int(a), int(b))

    roots = np.asarray([find(i) for i in range(points.shape[0])], dtype=np.int32)
    unique, counts = np.unique(roots, return_counts=True)
    lookup = dict(zip(unique.tolist(), counts.tolist()))
    return np.asarray([lookup[int(root)] for root in roots], dtype=np.float32)


def _load_real_teacher_states(source: Path = REAL_SNAPSHOT_SOURCE) -> tuple[np.ndarray, np.ndarray, float]:
    if not source.exists():
        raise FileNotFoundError(f"real teacher snapshot source is missing: {source}")
    payload = json.loads(source.read_text())
    snapshots = payload.get("snapshots", [])
    if len(snapshots) < 2:
        raise ValueError(f"need at least two snapshots in {source}")
    initial = np.asarray(snapshots[0]["teacher"]["cu"], dtype=np.float64)
    final = np.asarray(snapshots[-1]["teacher"]["cu"], dtype=np.float64)
    if initial.ndim != 2 or final.ndim != 2 or initial.shape[1] != 3 or final.shape[1] != 3:
        raise ValueError(f"invalid Cu coordinate arrays in {source}")
    initial_total = int(snapshots[0]["teacher"].get("cu_total", initial.shape[0]))
    final_total = int(snapshots[-1]["teacher"].get("cu_total", final.shape[0]))
    if initial_total != initial.shape[0] or final_total != final.shape[0]:
        raise ValueError(
            "snapshot file does not contain the full Cu coordinate set; "
            f"saved {initial.shape[0]}/{initial_total} and {final.shape[0]}/{final_total}"
        )
    box = float(max(initial.max(initial=0.0), final.max(initial=0.0)) + 1.0)
    return initial, final, box


def _partial_window(initial: np.ndarray, final: np.ndarray, box: float) -> tuple[np.ndarray, float]:
    side = box * 0.85
    initial_set = {tuple(map(int, pos)) for pos in np.rint(initial).astype(np.int32).tolist()}
    final_set = {tuple(map(int, pos)) for pos in np.rint(final).astype(np.int32).tolist()}
    changed = np.asarray(sorted(initial_set ^ final_set), dtype=np.float64).reshape(-1, 3)
    if len(changed):
        center = changed.mean(axis=0)
    else:
        tree = cKDTree(final)
        neighbors = tree.query_ball_point(final, r=max(side / 4.2, 1.0))
        counts = np.asarray([len(items) for items in neighbors], dtype=np.int32)
        center = final[int(np.argmax(counts))] if len(final) else np.full(3, box / 2.0)
    low = np.clip(center - side / 2.0, 0.0, box - side)
    return low, side


def _crop(points: np.ndarray, low: np.ndarray, side: float) -> np.ndarray:
    return points[_crop_mask(points, low, side)] - low


def _crop_mask(points: np.ndarray, low: np.ndarray, side: float) -> np.ndarray:
    high = low + side
    return np.all((points >= low) & (points <= high), axis=1)


def _cube_vertices_3d() -> np.ndarray:
    """Return real cube corners in the edge order used by the reference frame."""
    return np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )


def _perspective_project(points: np.ndarray) -> np.ndarray:
    """Project 3D cube coordinates with one calibrated perspective camera."""
    pts = np.asarray(points, dtype=np.float64)
    camera = np.array([1.85, -2.90, 1.72], dtype=np.float64)
    target = np.array([0.50, 0.50, 0.50], dtype=np.float64)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    forward = target - camera
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    rel = pts - camera
    cam_x = rel @ right
    cam_y = rel @ up
    cam_z = np.maximum(rel @ forward, 1e-6)
    focal = 1.0
    return np.column_stack([focal * cam_x / cam_z, focal * cam_y / cam_z])


def _projected_cube_transform() -> tuple[np.ndarray, float]:
    raw_vertices = _perspective_project(_cube_vertices_3d())
    raw_min = raw_vertices.min(axis=0)
    raw_max = raw_vertices.max(axis=0)
    center = (raw_min + raw_max) / 2.0
    span = raw_max - raw_min
    scale = 0.88 / float(max(span[0], span[1]))
    return center, scale


def _project_unit_points(unit_points: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(unit_points, dtype=np.float64), 0.0, 1.0)
    center, scale = _projected_cube_transform()
    return (_perspective_project(p) - center) * scale + np.array([0.50, 0.50])


def _reference_cube_vertices() -> np.ndarray:
    """2D perspective projection of a real 3D cube."""
    return _project_unit_points(_cube_vertices_3d())


def _project_points_to_reference_cube(points: np.ndarray, side: float, vertices: np.ndarray | None = None) -> np.ndarray:
    unit = np.asarray(points, dtype=np.float64) / max(float(side), 1e-12)
    xy = _project_unit_points(unit)
    if vertices is None:
        return xy
    base = _reference_cube_vertices()
    base_min = base.min(axis=0)
    base_span = base.max(axis=0) - base_min
    verts_min = vertices.min(axis=0)
    verts_span = vertices.max(axis=0) - verts_min
    return (xy - base_min) / np.maximum(base_span, 1e-12) * verts_span + verts_min


def _draw_reference_cube(ax, vertices: np.ndarray | None = None, *, color="#111111", lw=0.72, alpha=0.72) -> None:
    verts = _reference_cube_vertices() if vertices is None else vertices
    edges = [
        (0, 1),
        (1, 5),
        (5, 4),
        (4, 0),
        (3, 2),
        (2, 6),
        (6, 7),
        (7, 3),
        (0, 3),
        (1, 2),
        (5, 6),
        (4, 7),
    ]
    for i, j in edges:
        ax.plot(
            [verts[i, 0], verts[j, 0]],
            [verts[i, 1], verts[j, 1]],
            color=color,
            linewidth=lw,
            alpha=alpha,
            solid_capstyle="round",
            zorder=5,
        )


def _style_projected_axis(ax, title: str) -> None:
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.set_title(title, fontsize=10.0, pad=2.5, fontfamily="Times New Roman")


def _scatter_projected_state(
    ax,
    points: np.ndarray,
    scores: np.ndarray,
    norm,
    cmap,
    *,
    side: float,
    partial: bool = False,
) -> None:
    if len(points) == 0:
        _draw_reference_cube(ax)
        return
    unit = np.asarray(points, dtype=np.float64) / max(float(side), 1e-12)
    xy = _project_unit_points(unit)
    depth = unit[:, 1]
    order = np.argsort(-depth)
    marker_size = 4.0 if partial else 3.2
    marker_alpha = 0.86
    ax.scatter(
        xy[order, 0],
        xy[order, 1],
        c=np.asarray(scores)[order],
        cmap=cmap,
        norm=norm,
        s=marker_size,
        alpha=marker_alpha,
        linewidths=0,
        edgecolors="none",
        antialiaseds=False,
        rasterized=False,
        zorder=3,
    )
    _draw_reference_cube(ax)


def _draw_mini_cube(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    mini_vertices = _reference_cube_vertices() * 0.68 + np.array([0.13, 0.18])
    _draw_reference_cube(ax, mini_vertices, color="#777777", lw=0.75, alpha=0.48)
    # Small front-face square marking the zoomed region in the style of the reference.
    square_unit = np.asarray(
        [
            [0.02, 0.0, 0.02],
            [0.15, 0.0, 0.02],
            [0.15, 0.0, 0.14],
            [0.02, 0.0, 0.14],
            [0.02, 0.0, 0.02],
        ],
        dtype=np.float64,
    )
    square_xy = _project_unit_points(square_unit) * 0.68 + np.array([0.13, 0.18])
    ax.plot(square_xy[:, 0], square_xy[:, 1], color="#777777", linewidth=1.15, zorder=6)
    ax.text(0.5, 0.02, "Enlarge scale", ha="center", va="top", fontsize=8.3, fontfamily="Times New Roman")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=REAL_SNAPSHOT_SOURCE)
    parser.add_argument("--png-out", type=Path, default=PNG_OUT)
    parser.add_argument("--pdf-out", type=Path, default=PDF_OUT)
    parser.add_argument("--official-pdf-out", type=Path, default=OFFICIAL_PDF_OUT)
    args = parser.parse_args()

    png_out = args.png_out
    pdf_out = args.pdf_out
    official_pdf_out = args.official_pdf_out
    for out_path in [png_out, pdf_out, official_pdf_out]:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    FIGTEST_OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    initial, final, box = _load_real_teacher_states(args.source)
    low, side = _partial_window(initial, final, box)
    initial_zoom_mask = _crop_mask(initial, low, side)
    final_zoom_mask = _crop_mask(final, low, side)
    initial_zoom = initial[initial_zoom_mask] - low
    final_zoom = final[final_zoom_mask] - low

    initial_scores = _cluster_sizes(initial, box=box)
    final_scores = _cluster_sizes(final, box=box)
    initial_zoom_scores = initial_scores[initial_zoom_mask]
    final_zoom_scores = final_scores[final_zoom_mask]
    vmax = max(1.0, float(initial_scores.max(initial=1.0)), float(final_scores.max(initial=1.0)))
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "atomworld_cluster_size",
        ["#cf3a2a", "#c77ab8", "#2f6fae"],
    )
    norm = mcolors.Normalize(vmin=1.0, vmax=vmax)

    fig = plt.figure(figsize=(6.05, 5.67), facecolor="white")
    ax_i = fig.add_axes([0.015, 0.545, 0.345, 0.385])
    ax_f = fig.add_axes([0.385, 0.545, 0.345, 0.385])
    ax_iz = fig.add_axes([0.015, 0.065, 0.345, 0.385])
    ax_fz = fig.add_axes([0.385, 0.065, 0.345, 0.385])
    ax_cb = fig.add_axes([0.825, 0.535, 0.035, 0.245])
    ax_mini = fig.add_axes([0.765, 0.205, 0.185, 0.205])

    _style_projected_axis(ax_i, "Initial (whole box)")
    _style_projected_axis(ax_f, "Final (whole box)")
    _style_projected_axis(ax_iz, "Initial (partial box)")
    _style_projected_axis(ax_fz, "Final (partial box)")

    _scatter_projected_state(ax_i, initial, initial_scores, norm, cmap, side=box)
    _scatter_projected_state(ax_f, final, final_scores, norm, cmap, side=box)
    _scatter_projected_state(ax_iz, initial_zoom, initial_zoom_scores, norm, cmap, side=side, partial=True)
    _scatter_projected_state(ax_fz, final_zoom, final_zoom_scores, norm, cmap, side=side, partial=True)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=ax_cb, orientation="vertical")
    cb.outline.set_visible(False)
    cb.set_ticks([])
    ax_cb.invert_yaxis()
    fig.text(0.843, 0.846, "$C_1$\n(single Cu atom)", ha="center", va="top", fontsize=8.1, fontfamily="Times New Roman")
    fig.text(0.843, 0.505, "$C_{max}$", ha="center", va="top", fontsize=8.1, fontfamily="Times New Roman")
    ax_cb.tick_params(length=0)

    _draw_mini_cube(ax_mini)

    fig.savefig(png_out, dpi=300)
    fig.savefig(pdf_out)
    fig.savefig(official_pdf_out)
    print(f"source={args.source}")
    print(f"true_cluster_radius=2.01 BCC_1NN_2NN Cmax={vmax:g}")
    print(official_pdf_out)
    print(png_out)
    print(pdf_out)


if __name__ == "__main__":
    main()
