from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

import eval_macro_long_trajectory as long_eval
import train_dreamer_macro_edit as mod

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover - optional dependency fallback
    linear_sum_assignment = None

try:
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover - optional dependency fallback
    cKDTree = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Closed-loop autonomous AtomWorld-Mirror macro rollout evaluation"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--duration_checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rollout_segments", type=int, default=200)
    parser.add_argument("--max_episode_steps_override", type=int, default=None)
    parser.add_argument("--temperature_override", type=float, default=None)
    parser.add_argument("--cu_density_override", type=float, default=None)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--print_segments", type=int, default=5)
    parser.add_argument("--progress_every", type=int, default=25)
    parser.add_argument(
        "--reference_mode",
        type=str,
        default="independent_teacher",
        choices=["independent_teacher", "on_policy_teacher_probe"],
        help=(
            "independent_teacher compares the autonomous model trajectory against one independently "
            "evolving KMC path from the same initial state. on_policy_teacher_probe keeps the model "
            "state closed-loop, but evaluates each predicted macro transition with a KMC probe started "
            "from the model's current state."
        ),
    )
    parser.add_argument("--planner_segment_ks", type=int, nargs="+", default=None)
    parser.add_argument("--min_projected_changed_sites", type=int, default=2)
    parser.add_argument(
        "--constraint_mode",
        type=str,
        default="full",
        choices=[
            "full",
            "no_inventory",
            "no_reachability",
            "no_continuous_time",
            "no_constraints",
            "no_change",
        ],
        help=(
            "Closed-loop diagnostic mode. full uses reachable candidates, inventory projection, and learned "
            "duration. no_inventory disables inventory projection. no_reachability evaluates on a global "
            "unrestricted candidate support. no_continuous_time uses the CTMC baseline duration instead of the "
            "learned time head. no_constraints combines unrestricted support, raw edits, and baseline duration. "
            "no_change is a copy-state baseline."
        ),
    )
    parser.add_argument(
        "--duration_source",
        type=str,
        default="model",
        choices=["model", "baseline", "blend"],
        help="Duration source for reported closed-loop time; overridden by no_continuous_time/no_constraints/no_change.",
    )
    parser.add_argument("--duration_blend_alpha", type=float, default=1.0)
    parser.add_argument("--duration_log_offset", type=float, default=0.0)
    parser.add_argument(
        "--planner_score_mode",
        type=str,
        default="energy_per_tau",
        choices=["energy_per_tau", "energy_per_sqrt_tau", "energy"],
    )
    parser.add_argument("--planner_tau_residual_penalty", type=float, default=0.0)
    parser.add_argument("--planner_k_penalty_power", type=float, default=0.0)
    parser.add_argument("--raw_changed_budget_multiplier", type=float, default=2.0)
    parser.add_argument(
        "--inventory_stress_mode",
        type=str,
        default="none",
        choices=["none", "vacancy_bias"],
        help=(
            "Extra no-inventory stress mode for ablation diagnostics. vacancy_bias keeps the reachable "
            "candidate support but forces selected non-vacancy sites to vacancy before applying the raw edit."
        ),
    )
    parser.add_argument("--global_candidate_seed_vacancies", type=int, default=8)
    parser.add_argument("--global_candidate_cu_fraction", type=float, default=0.5)
    parser.add_argument("--allow_teacher_noop_segments", action="store_true")
    parser.add_argument("--save_snapshots", action="store_true")
    parser.add_argument("--snapshot_every", type=int, default=25)
    parser.add_argument("--snapshot_max_cu", type=int, default=1200)
    parser.add_argument("--save_edit_trace", action="store_true")
    return parser.parse_args()


def _as_pos_tuple(pos: Any, box: np.ndarray) -> tuple[int, int, int]:
    arr = np.asarray(pos, dtype=np.int64).reshape(3)
    arr = arr % box.astype(np.int64)
    return tuple(int(v) for v in arr.tolist())


def _periodic_offset(src: np.ndarray, dst: np.ndarray, box: np.ndarray) -> np.ndarray:
    delta = src - dst
    return delta - np.round(delta / box) * box


def _periodic_distance_matrix(src: np.ndarray, dst: np.ndarray, box: np.ndarray) -> np.ndarray:
    if src.size == 0 or dst.size == 0:
        return np.empty((src.shape[0], dst.shape[0]), dtype=np.float64)
    delta = src[:, None, :] - dst[None, :, :]
    delta = delta - np.round(delta / box[None, None, :]) * box[None, None, :]
    return np.linalg.norm(delta, axis=-1)


def _match_mean_distance(src: np.ndarray, dst: np.ndarray, box: np.ndarray, missing_penalty: float | None = None) -> float:
    src = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    if src.shape[0] == 0 and dst.shape[0] == 0:
        return 0.0
    if src.shape[0] == 0 or dst.shape[0] == 0:
        penalty = float(missing_penalty if missing_penalty is not None else np.linalg.norm(box) / 2.0)
        return penalty
    dist = _periodic_distance_matrix(src, dst, box)
    if src.shape[0] == dst.shape[0] and linear_sum_assignment is not None:
        row, col = linear_sum_assignment(dist)
        return float(dist[row, col].mean())
    nearest_src = dist.min(axis=1)
    nearest_dst = dist.min(axis=0)
    penalty = float(missing_penalty if missing_penalty is not None else np.linalg.norm(box) / 2.0)
    missing = abs(src.shape[0] - dst.shape[0])
    denom = max(max(src.shape[0], dst.shape[0]), 1)
    total = float(nearest_src.sum() + nearest_dst.sum()) / 2.0 + missing * penalty
    return float(total / denom)


def _chamfer_mean_distance(src: np.ndarray, dst: np.ndarray, box: np.ndarray) -> float:
    src = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    if src.shape[0] == 0 and dst.shape[0] == 0:
        return 0.0
    if src.shape[0] == 0 or dst.shape[0] == 0:
        return float(np.linalg.norm(box) / 2.0)
    if cKDTree is not None:
        src_tree = cKDTree(src % box, boxsize=box)
        dst_tree = cKDTree(dst % box, boxsize=box)
        src_to_dst, _ = dst_tree.query(src % box, k=1)
        dst_to_src, _ = src_tree.query(dst % box, k=1)
        return float(0.5 * (np.mean(src_to_dst) + np.mean(dst_to_src)))
    dist = _periodic_distance_matrix(src, dst, box)
    return float(0.5 * (dist.min(axis=1).mean() + dist.min(axis=0).mean()))


def _topology_metrics(cu_pos: np.ndarray, box: np.ndarray, radius: float = 4.0) -> dict[str, float]:
    cu_pos = np.asarray(cu_pos, dtype=np.float64).reshape(-1, 3) % box
    n = int(cu_pos.shape[0])
    if n == 0:
        return {
            "cu_neighbor_edges": 0.0,
            "cu_mean_degree": 0.0,
            "cu_isolated_fraction": 0.0,
            "cu_largest_cluster_fraction": 0.0,
        }
    if cKDTree is not None:
        tree = cKDTree(cu_pos, boxsize=box)
        neighbors = tree.query_ball_point(cu_pos, r=float(radius))
        pairs = tree.query_pairs(r=float(radius))
        counts = np.asarray([len(items) - 1 for items in neighbors], dtype=np.float64)
    else:
        dist = _periodic_distance_matrix(cu_pos, cu_pos, box)
        adjacency = (dist <= float(radius)) & (dist > 0)
        pairs = set(zip(*np.where(np.triu(adjacency, k=1))))
        counts = adjacency.sum(axis=1).astype(np.float64)

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in pairs:
        union(int(a), int(b))
    cluster_sizes: dict[int, int] = {}
    for idx in range(n):
        root = find(idx)
        cluster_sizes[root] = cluster_sizes.get(root, 0) + 1
    largest = max(cluster_sizes.values()) if cluster_sizes else 0
    return {
        "cu_neighbor_edges": float(len(pairs)),
        "cu_mean_degree": float(counts.mean()) if counts.size else 0.0,
        "cu_isolated_fraction": float(np.mean(counts == 0)) if counts.size else 0.0,
        "cu_largest_cluster_fraction": float(largest / max(n, 1)),
    }


def _environment_state_arrays(env: mod.MacroKMCEnv) -> tuple[np.ndarray, np.ndarray]:
    return (
        env.env.get_vacancy_array().astype(np.int32).reshape(-1, 3),
        env.env.get_cu_array().astype(np.int32).reshape(-1, 3),
    )


def _state_metrics(
    *,
    model_env: mod.MacroKMCEnv,
    teacher_env: mod.MacroKMCEnv,
    initial_vacancy_count: int,
    initial_cu_count: int,
) -> dict[str, float]:
    box = np.asarray(model_env.env.dims, dtype=np.float64)
    model_vac, model_cu = _environment_state_arrays(model_env)
    teacher_vac, teacher_cu = _environment_state_arrays(teacher_env)
    model_energy = float(model_env.env.calculate_system_energy())
    teacher_energy = float(teacher_env.env.calculate_system_energy())
    model_top = _topology_metrics(model_cu, box)
    teacher_top = _topology_metrics(teacher_cu, box)
    topology_abs = {
        f"{key}_abs_error": float(abs(model_top[key] - teacher_top[key]))
        for key in model_top.keys()
    }
    return {
        "model_energy": model_energy,
        "teacher_energy": teacher_energy,
        "energy_abs_error": float(abs(model_energy - teacher_energy)),
        "vacancy_count": float(model_vac.shape[0]),
        "cu_count": float(model_cu.shape[0]),
        "inventory_violation_l1": float(
            abs(model_vac.shape[0] - initial_vacancy_count) + abs(model_cu.shape[0] - initial_cu_count)
        ),
        "vacancy_matching_mean_distance": _match_mean_distance(model_vac, teacher_vac, box),
        "cu_chamfer_mean_distance": _chamfer_mean_distance(model_cu, teacher_cu, box),
        **{f"model_{key}": value for key, value in model_top.items()},
        **{f"teacher_{key}": value for key, value in teacher_top.items()},
        **topology_abs,
    }


def _state_snapshot(env: mod.MacroKMCEnv, *, max_cu: int, rng: np.random.Generator) -> dict[str, Any]:
    vac, cu = _environment_state_arrays(env)
    if cu.shape[0] > int(max_cu):
        idx = rng.choice(cu.shape[0], size=int(max_cu), replace=False)
        cu_out = cu[np.sort(idx)]
    else:
        cu_out = cu
    return {
        "vacancies": vac.astype(int).tolist(),
        "cu": cu_out.astype(int).tolist(),
        "cu_total": int(cu.shape[0]),
        "energy": float(env.env.calculate_system_energy()),
    }


def _refresh_env_after_direct_state_set(env: mod.MacroKMCEnv) -> None:
    lattice = env.env
    lattice._rebuild_global_lin_cache()
    if hasattr(lattice, "_calculate_vacancy_local_environments_sparse"):
        (
            lattice.nn1_types,
            lattice.nn2_types,
            lattice.nn1_nn1_types,
            lattice.nn1_nn2_types,
        ) = lattice._calculate_vacancy_local_environments_sparse()
    if hasattr(lattice, "_init_topk_system"):
        lattice._init_topk_system()
    lattice.diffusion_rates = None
    lattice._ensure_diffusion_rates()
    lattice.energy_last = float(lattice.calculate_system_energy())


def _set_env_sets(
    env: mod.MacroKMCEnv,
    *,
    vac_set: set[tuple[int, int, int]],
    cu_set: set[tuple[int, int, int]],
) -> None:
    overlap = vac_set & cu_set
    if overlap:
        cu_set = set(cu_set)
        cu_set.difference_update(overlap)
    lattice = env.env
    lattice.vac_pos_set = set(sorted(vac_set))
    lattice.cu_pos_set = set(sorted(cu_set))
    lattice.V_nums = int(len(lattice.vac_pos_set))
    lattice.Cu_nums = int(len(lattice.cu_pos_set))
    vac_arr = np.asarray(sorted(lattice.vac_pos_set), dtype=np.int32).reshape(-1, 3)
    cu_arr = np.asarray(sorted(lattice.cu_pos_set), dtype=np.int32).reshape(-1, 3)
    lattice.cu_pos = cu_arr
    lattice.v_pos_to_id = {tuple(map(int, pos)): idx for idx, pos in enumerate(vac_arr.tolist())}
    lattice.v_pos_of_id = {idx: tuple(map(int, pos)) for idx, pos in enumerate(vac_arr.tolist())}
    lattice.cu_pos_of_id = {idx + lattice.V_nums: tuple(map(int, pos)) for idx, pos in enumerate(cu_arr.tolist())}
    lattice._build_cu_pos_index()
    if hasattr(lattice, "pure_local_cu_pos"):
        lattice.pure_local_cu_pos = cu_arr
    if hasattr(lattice, "local_cu_pos"):
        lattice.local_cu_pos = cu_arr
    _refresh_env_after_direct_state_set(env)


def _apply_candidate_to_env(env: mod.MacroKMCEnv, candidate: dict[str, Any], *, predicted_tau: float) -> dict[str, float]:
    box = np.asarray(env.env.dims, dtype=np.int32)
    start_energy = float(env.env.calculate_system_energy())
    start_vac, start_cu = _environment_state_arrays(env)
    vac_set, cu_set = mod._positions_to_type_lookup(start_vac, start_cu)
    positions = np.asarray(candidate["candidate_positions"], dtype=np.int64).reshape(-1, 3)
    current_types = np.asarray(candidate["current_types"], dtype=np.int64).reshape(-1)
    new_types = np.asarray(candidate["selected_types"], dtype=np.int64).reshape(-1)
    mask = np.asarray(candidate["candidate_mask"], dtype=np.float32).reshape(-1) > 0
    changed_positions = 0
    for pos_arr, cur_type, new_type, valid in zip(positions, current_types, new_types, mask):
        if not valid or int(cur_type) == int(new_type):
            continue
        pos = _as_pos_tuple(pos_arr, box)
        vac_set.discard(pos)
        cu_set.discard(pos)
        if int(new_type) == mod.V_TYPE:
            vac_set.add(pos)
        elif int(new_type) == mod.CU_TYPE:
            cu_set.add(pos)
        changed_positions += 1
    _set_env_sets(env, vac_set=vac_set, cu_set=cu_set)
    end_energy = float(env.env.calculate_system_energy())
    env.timestep += int(candidate["segment_k"])
    env.env.time += float(predicted_tau)
    env.env.time_history.append(float(env.env.time))
    env.env.energy_history.append(end_energy)
    return {
        "model_energy_before": start_energy,
        "model_energy_after": end_energy,
        "model_delta_e": float(start_energy - end_energy),
        "applied_changed_count": float(changed_positions),
    }


def _candidate_edit_trace_row(
    *,
    segment_idx: int,
    candidate: dict[str, Any],
    applied: dict[str, float],
) -> dict[str, Any]:
    if "candidate_positions" not in candidate:
        return {
            "index": int(segment_idx),
            "segment_k": int(candidate.get("segment_k", 0)),
            "applied_changed_count": float(applied.get("applied_changed_count", 0.0)),
            "positions": [],
            "current_types": [],
            "selected_types": [],
        }
    positions = np.asarray(candidate["candidate_positions"], dtype=np.int64).reshape(-1, 3)
    current_types = np.asarray(candidate["current_types"], dtype=np.int64).reshape(-1)
    selected_types = np.asarray(candidate["selected_types"], dtype=np.int64).reshape(-1)
    mask = np.asarray(candidate["candidate_mask"], dtype=np.float32).reshape(-1) > 0
    changed = mask & (current_types != selected_types)
    changed_positions = positions[changed]
    changed_current = current_types[changed]
    changed_selected = selected_types[changed]
    return {
        "index": int(segment_idx),
        "segment_k": int(candidate["segment_k"]),
        "candidate_mode": str(candidate.get("candidate_mode", "")),
        "edit_mode": str(candidate.get("edit_mode", "")),
        "predicted_expected_tau": float(candidate.get("predicted_expected_tau", 0.0)),
        "applied_changed_count": float(applied.get("applied_changed_count", 0.0)),
        "positions": changed_positions.astype(int).tolist(),
        "current_types": changed_current.astype(int).tolist(),
        "selected_types": changed_selected.astype(int).tolist(),
    }


def _sample_global_candidate_positions(
    *,
    env: mod.MacroKMCEnv,
    max_seed_vacancies: int,
    max_candidate_sites: int,
    cu_fraction: float,
    rng: np.random.Generator,
) -> tuple[list[tuple[int, int, int]], dict[tuple[int, int, int], int], np.ndarray]:
    box = np.asarray(env.env.dims, dtype=np.int32)
    vacancies = env.env.get_vacancy_array().astype(np.int32).reshape(-1, 3)
    cu = env.env.get_cu_array().astype(np.int32).reshape(-1, 3)
    vac_set, cu_set = mod._positions_to_type_lookup(vacancies, cu)
    positions: list[tuple[int, int, int]] = []
    for pos in vacancies[: max(1, int(max_seed_vacancies))]:
        positions.append(_as_pos_tuple(pos, box))

    remaining = max(0, int(max_candidate_sites) - len(positions))
    cu_take = min(cu.shape[0], int(round(remaining * float(np.clip(cu_fraction, 0.0, 1.0)))))
    if cu_take > 0:
        cu_idx = rng.choice(cu.shape[0], size=cu_take, replace=False)
        for pos in cu[cu_idx]:
            positions.append(_as_pos_tuple(pos, box))

    remaining = max(0, int(max_candidate_sites) - len(positions))
    if remaining > 0:
        coords = np.asarray(env.env.coords, dtype=np.int32).reshape(-1, 3)
        attempts = 0
        seen = set(positions)
        while remaining > 0 and attempts < max(remaining * 50, 200):
            pos = _as_pos_tuple(coords[int(rng.integers(0, coords.shape[0]))], box)
            attempts += 1
            if pos in seen or pos in vac_set or pos in cu_set:
                continue
            seen.add(pos)
            positions.append(pos)
            remaining -= 1
    positions = list(dict.fromkeys(positions))[: int(max_candidate_sites)]
    seeds = vacancies[: max(1, min(int(max_seed_vacancies), len(vacancies)))]
    depth_map = {pos: 0 if pos in vac_set else 10**6 for pos in positions}
    return positions, depth_map, seeds


def _build_closed_loop_tensors(
    *,
    env: mod.MacroKMCEnv,
    max_seed_vacancies: int,
    max_candidate_sites: int,
    horizon_k: int,
    device: str,
    candidate_mode: str,
    rng: np.random.Generator,
    global_candidate_cu_fraction: float,
) -> dict[str, torch.Tensor] | None:
    if candidate_mode == "global_random":
        candidate_positions, depth_map, seeds = _sample_global_candidate_positions(
            env=env,
            max_seed_vacancies=max_seed_vacancies,
            max_candidate_sites=max_candidate_sites,
            cu_fraction=global_candidate_cu_fraction,
            rng=rng,
        )
    else:
        candidate_positions, depth_map, seeds = mod._build_candidate_positions(
            env,
            horizon_k,
            max_seed_vacancies=max_seed_vacancies,
            max_candidate_sites=max_candidate_sites,
        )
    if not candidate_positions:
        return None

    start_vacancies = env.env.get_vacancy_array().astype(np.int32)
    start_cu = env.env.get_cu_array().astype(np.int32)
    start_vac_set, start_cu_set = mod._positions_to_type_lookup(start_vacancies, start_cu)
    positions, nearest_offsets, reach_depth, is_start_vacancy, current_types, _, _ = mod._build_patch_features(
        candidate_positions=candidate_positions,
        depth_map=depth_map,
        seeds=seeds,
        start_vac_set=start_vac_set,
        start_cu_set=start_cu_set,
        end_vac_set=start_vac_set,
        end_cu_set=start_cu_set,
        max_candidate_sites=max_candidate_sites,
        box=np.asarray(env.env.dims, dtype=np.int32),
        horizon_k=horizon_k,
    )
    candidate_mask = np.zeros((max_candidate_sites,), dtype=np.float32)
    candidate_mask[: len(candidate_positions)] = 1.0
    return {
        "start_obs": torch.tensor(env.obs()[None, :], dtype=torch.float32, device=device),
        "global_summary": torch.tensor(mod._global_summary(env)[None, :], dtype=torch.float32, device=device),
        "candidate_positions": torch.tensor(positions[None, ...], dtype=torch.float32, device=device),
        "nearest_vacancy_offset": torch.tensor(nearest_offsets[None, ...], dtype=torch.float32, device=device),
        "reach_depth": torch.tensor(reach_depth[None, ...], dtype=torch.float32, device=device),
        "is_start_vacancy": torch.tensor(is_start_vacancy[None, ...], dtype=torch.float32, device=device),
        "current_types": torch.tensor(current_types[None, ...], dtype=torch.long, device=device),
        "candidate_mask": torch.tensor(candidate_mask[None, ...], dtype=torch.float32, device=device),
        "box_dims": torch.tensor(np.asarray(env.env.dims, dtype=np.float32)[None, :], dtype=torch.float32, device=device),
        "horizon_k": torch.tensor([horizon_k], dtype=torch.long, device=device),
    }


def _raw_selected_types(
    *,
    current_types: torch.Tensor,
    type_logits: torch.Tensor,
    change_logits: torch.Tensor,
    candidate_mask: torch.Tensor,
    horizon_k: torch.Tensor,
    raw_changed_budget_multiplier: float,
    inventory_stress_mode: str = "none",
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = candidate_mask > 0
    budget = int(max(1, round(float(raw_changed_budget_multiplier) * int(horizon_k.item()))))
    if inventory_stress_mode == "vacancy_bias":
        atom_valid = valid & (current_types != mod.V_TYPE)
        score = torch.sigmoid(change_logits).masked_fill(~atom_valid, -1.0)
        selected = torch.zeros_like(valid, dtype=torch.bool)
        topk = min(budget, int(torch.sum(atom_valid).item()))
        if topk > 0:
            idx = torch.topk(score[0], k=topk).indices
            selected[0, idx] = True
        selected_types = current_types.clone()
        selected_types[selected] = int(mod.V_TYPE)
        return selected_types, selected

    raw_types = torch.argmax(type_logits, dim=-1)
    would_change = (raw_types != current_types) & valid
    score = torch.sigmoid(change_logits).masked_fill(~would_change, -1.0)
    selected = torch.zeros_like(valid, dtype=torch.bool)
    valid_scores = score[0]
    topk = min(budget, int(torch.sum(would_change).item()))
    if topk > 0:
        idx = torch.topk(valid_scores, k=topk).indices
        selected[0, idx] = True
    selected_types = current_types.clone()
    selected_types[selected] = raw_types[selected]
    return selected_types, selected


def _predict_closed_loop_candidate(
    *,
    model: mod.MacroDreamerEditModel,
    duration_model: mod.MacroDreamerEditModel | None,
    env: mod.MacroKMCEnv,
    horizon_k: int,
    max_seed_vacancies: int,
    max_candidate_sites: int,
    reward_scale: float,
    device: str,
    candidate_mode: str,
    edit_mode: str,
    duration_source: str,
    duration_blend_alpha: float,
    duration_log_offset: float,
    planner_score_mode: str,
    planner_tau_residual_penalty: float,
    planner_k_penalty_power: float,
    raw_changed_budget_multiplier: float,
    inventory_stress_mode: str,
    rng: np.random.Generator,
    global_candidate_cu_fraction: float,
    reward_prediction_source: str,
) -> dict[str, Any] | None:
    tensors = _build_closed_loop_tensors(
        env=env,
        max_seed_vacancies=max_seed_vacancies,
        max_candidate_sites=max_candidate_sites,
        horizon_k=horizon_k,
        device=device,
        candidate_mode=candidate_mode,
        rng=rng,
        global_candidate_cu_fraction=global_candidate_cu_fraction,
    )
    if tensors is None:
        return None
    global_latent = model.encode_global(tensors["start_obs"])
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
    prior_mu, prior_logvar = model.prior_stats(global_latent, tensors["global_summary"], tensors["horizon_k"])
    path_latent = model.sample_path_latent(prior_mu, prior_logvar, deterministic=True)
    next_pred = model.predict_next_global(global_latent, path_latent, tensors["horizon_k"])
    change_logits, raw_type_logits = model.decode_edit(
        site_latent=site_latent,
        patch_latent=patch_latent,
        predicted_next_global=next_pred,
        path_latent=path_latent,
        horizon_k=tensors["horizon_k"],
        current_types=tensors["current_types"],
    )
    projected_types, projected_changed_mask, transport_cost, projected_reachability_violation = mod.project_types_by_inventory(
        current_types=tensors["current_types"],
        change_logits=change_logits,
        type_logits=raw_type_logits,
        node_mask=tensors["candidate_mask"],
        positions=tensors["candidate_positions"],
        box_dims=tensors["box_dims"],
        horizon_k=tensors["horizon_k"],
        max_changed_sites=2 * tensors["horizon_k"],
    )
    if edit_mode == "projected":
        selected_types = projected_types
        selected_changed_mask = projected_changed_mask > 0
        reachability_violation = float(projected_reachability_violation.item())
    else:
        selected_types, selected_changed_mask = _raw_selected_types(
            current_types=tensors["current_types"],
            type_logits=raw_type_logits,
            change_logits=change_logits,
            candidate_mask=tensors["candidate_mask"],
            horizon_k=tensors["horizon_k"],
            raw_changed_budget_multiplier=raw_changed_budget_multiplier,
            inventory_stress_mode=inventory_stress_mode,
        )
        changed_valid = selected_changed_mask & (tensors["candidate_mask"] > 0)
        if int(changed_valid.sum().item()) > 0:
            reach_depth = tensors["reach_depth"][changed_valid]
            reachability_violation = float(torch.mean((reach_depth > 1.0).to(torch.float32)).item())
        else:
            reachability_violation = 0.0

    inventory_delta = []
    for site_type in [mod.FE_TYPE, mod.CU_TYPE, mod.V_TYPE]:
        before = torch.sum(((tensors["current_types"] == site_type) & (tensors["candidate_mask"] > 0)).to(torch.float32))
        after = torch.sum(((selected_types == site_type) & (tensors["candidate_mask"] > 0)).to(torch.float32))
        inventory_delta.append(float((after - before).item()))
    inventory_delta_l1 = float(sum(abs(value) for value in inventory_delta))

    reward_patch_latent = patch_latent
    reward_change_logits = change_logits
    reward_type_logits = raw_type_logits
    if reward_prediction_source == "projected":
        _, reward_patch_latent = model.encode_patch(
            positions=tensors["candidate_positions"],
            nearest_vacancy_offset=tensors["nearest_vacancy_offset"],
            reach_depth=tensors["reach_depth"],
            is_start_vacancy=tensors["is_start_vacancy"],
            type_ids=selected_types,
            node_mask=tensors["candidate_mask"],
            global_summary=tensors["global_summary"],
            box_dims=tensors["box_dims"],
        )
        reward_change_logits, reward_type_logits = mod.projected_edit_logits_from_types(
            current_types=tensors["current_types"],
            projected_types=selected_types,
            candidate_mask=tensors["candidate_mask"],
        )
    primary_outputs = mod._predict_reward_and_duration_outputs(
        model,
        global_latent,
        next_pred,
        path_latent,
        tensors["global_summary"],
        tensors["horizon_k"],
        patch_latent=reward_patch_latent,
        change_logits=reward_change_logits,
        type_logits=reward_type_logits,
        current_types=tensors["current_types"],
        candidate_mask=tensors["candidate_mask"],
    )
    duration_outputs = primary_outputs
    if duration_model is not None and duration_model is not model:
        duration_global_latent = duration_model.encode_global(tensors["start_obs"])
        duration_site_latent, duration_patch_latent = duration_model.encode_patch(
            positions=tensors["candidate_positions"],
            nearest_vacancy_offset=tensors["nearest_vacancy_offset"],
            reach_depth=tensors["reach_depth"],
            is_start_vacancy=tensors["is_start_vacancy"],
            type_ids=tensors["current_types"],
            node_mask=tensors["candidate_mask"],
            global_summary=tensors["global_summary"],
            box_dims=tensors["box_dims"],
        )
        duration_prior_mu, duration_prior_logvar = duration_model.prior_stats(
            duration_global_latent,
            tensors["global_summary"],
            tensors["horizon_k"],
        )
        duration_path_latent = duration_model.sample_path_latent(duration_prior_mu, duration_prior_logvar, deterministic=True)
        duration_next_pred = duration_model.predict_next_global(duration_global_latent, duration_path_latent, tensors["horizon_k"])
        duration_change_logits, duration_type_logits = duration_model.decode_edit(
            site_latent=duration_site_latent,
            patch_latent=duration_patch_latent,
            predicted_next_global=duration_next_pred,
            path_latent=duration_path_latent,
            horizon_k=tensors["horizon_k"],
            current_types=tensors["current_types"],
        )
        duration_outputs = mod._predict_reward_and_duration_outputs(
            duration_model,
            duration_global_latent,
            duration_next_pred,
            duration_path_latent,
            tensors["global_summary"],
            tensors["horizon_k"],
            patch_latent=duration_patch_latent,
            change_logits=duration_change_logits,
            type_logits=duration_type_logits,
            current_types=tensors["current_types"],
            candidate_mask=tensors["candidate_mask"],
        )

    pred_reward_raw = float(primary_outputs["reward"].item())
    pred_reward_gate_prob = float(torch.sigmoid(primary_outputs["gate_logit"]).item())
    pred_reward = float((primary_outputs["reward"] * torch.sigmoid(primary_outputs["gate_logit"])).item())
    model_expected_tau = float(torch.exp(duration_outputs["expected_tau_mu"]).item())
    model_realized_tau = float(torch.exp(duration_outputs["realized_tau_mu"]).item())
    baseline_log_tau = mod.macro_duration_baseline_log_tau(tensors["global_summary"], tensors["horizon_k"])
    baseline_expected_tau = float(torch.exp(baseline_log_tau).item())
    pred_expected_tau = long_eval._duration_from_source(
        model_expected_tau=model_expected_tau,
        baseline_expected_tau=baseline_expected_tau,
        source=duration_source,
        blend_alpha=duration_blend_alpha,
        duration_log_offset=duration_log_offset,
    )
    selection_score, tau_for_score = long_eval._compute_selection_score(
        pred_reward_sum=pred_reward,
        reward_scale=reward_scale,
        model_expected_tau=model_expected_tau,
        baseline_expected_tau=baseline_expected_tau,
        horizon_k=int(horizon_k),
        planner_tau_source=duration_source,
        planner_score_mode=planner_score_mode,
        planner_tau_residual_penalty=planner_tau_residual_penalty,
        planner_k_penalty_power=planner_k_penalty_power,
        planner_tau_blend_alpha=duration_blend_alpha,
        planner_tau_log_offset=duration_log_offset,
    )
    if edit_mode == "projected" and reachability_violation > 0.0:
        selection_score = -float("inf")
    return {
        "segment_k": int(horizon_k),
        "candidate_mode": candidate_mode,
        "edit_mode": edit_mode,
        "predicted_reward_sum": pred_reward,
        "predicted_delta_e": float(pred_reward / max(reward_scale, 1e-12)),
        "predicted_reward_raw": pred_reward_raw,
        "predicted_reward_gate_prob": pred_reward_gate_prob,
        "predicted_expected_tau": pred_expected_tau,
        "predicted_realized_tau": model_realized_tau if duration_source == "model" else pred_expected_tau,
        "model_expected_tau": model_expected_tau,
        "model_realized_tau": model_realized_tau,
        "baseline_expected_tau": baseline_expected_tau,
        "planner_tau_for_score": tau_for_score,
        "selection_score": float(selection_score),
        "reachability_violation": reachability_violation,
        "projected_reachability_violation": float(projected_reachability_violation.item()),
        "inventory_delta_fe_cu_v": inventory_delta,
        "candidate_inventory_delta_l1": inventory_delta_l1,
        "projected_changed_count": float(selected_changed_mask.sum().item()),
        "transport_cost": float(transport_cost.item()),
        "candidate_positions": tensors["candidate_positions"][0].detach().cpu().numpy().astype(np.int32).tolist(),
        "candidate_mask": tensors["candidate_mask"][0].detach().cpu().numpy().astype(np.float32).tolist(),
        "current_types": tensors["current_types"][0].detach().cpu().numpy().astype(np.int64).tolist(),
        "selected_types": selected_types[0].detach().cpu().numpy().astype(np.int64).tolist(),
    }


def _choose_closed_loop_candidate(candidates: list[dict[str, Any]], *, min_projected_changed_sites: int) -> dict[str, Any] | None:
    if not candidates:
        return None
    legal = [
        item
        for item in candidates
        if np.isfinite(float(item.get("selection_score", -float("inf"))))
        and float(item.get("projected_changed_count", 0.0)) >= float(min_projected_changed_sites)
    ]
    if not legal:
        return None
    return max(legal, key=lambda item: float(item.get("selection_score", -float("inf"))))


def _constraint_settings(args: argparse.Namespace) -> tuple[str, str, str]:
    if args.constraint_mode == "full":
        return "reachable", "projected", args.duration_source
    if args.constraint_mode == "no_inventory":
        return "reachable", "raw", args.duration_source
    if args.constraint_mode == "no_reachability":
        return "global_random", "raw", args.duration_source
    if args.constraint_mode == "no_continuous_time":
        return "reachable", "projected", "baseline"
    if args.constraint_mode == "no_constraints":
        return "global_random", "raw", "baseline"
    if args.constraint_mode == "no_change":
        return "reachable", "none", "baseline"
    raise ValueError(f"Unknown constraint mode: {args.constraint_mode}")


def _no_change_candidate(env: mod.MacroKMCEnv, horizon_k: int, duration_source: str) -> dict[str, Any]:
    summary = mod._global_summary(env)
    baseline_expected_tau = float(np.exp(np.log(max(int(horizon_k), 1)) - float(summary[10])))
    return {
        "segment_k": int(horizon_k),
        "candidate_mode": "none",
        "edit_mode": "none",
        "predicted_reward_sum": 0.0,
        "predicted_delta_e": 0.0,
        "predicted_reward_raw": 0.0,
        "predicted_reward_gate_prob": 0.0,
        "predicted_expected_tau": baseline_expected_tau,
        "predicted_realized_tau": baseline_expected_tau,
        "model_expected_tau": baseline_expected_tau,
        "model_realized_tau": baseline_expected_tau,
        "baseline_expected_tau": baseline_expected_tau,
        "planner_tau_for_score": baseline_expected_tau,
        "selection_score": 0.0,
        "reachability_violation": 0.0,
        "projected_reachability_violation": 0.0,
        "inventory_delta_fe_cu_v": [0.0, 0.0, 0.0],
        "candidate_inventory_delta_l1": 0.0,
        "projected_changed_count": 0.0,
        "transport_cost": 0.0,
        "duration_source": duration_source,
    }


def _apply_no_change_to_env(env: mod.MacroKMCEnv, candidate: dict[str, Any]) -> dict[str, float]:
    energy = float(env.env.calculate_system_energy())
    env.timestep += int(candidate["segment_k"])
    env.env.time += float(candidate["predicted_expected_tau"])
    env.env.time_history.append(float(env.env.time))
    env.env.energy_history.append(energy)
    return {
        "model_energy_before": energy,
        "model_energy_after": energy,
        "model_delta_e": 0.0,
        "applied_changed_count": 0.0,
    }


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else 0.0


def _histogram_int(values: list[int]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        key = str(int(value))
        out[key] = out.get(key, 0) + 1
    return out


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    snapshot_rng = np.random.default_rng(args.seed + 991)

    checkpoint_path = Path(args.checkpoint)
    ckpt = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    ckpt_args = ckpt["args"]
    reward_scale = float(ckpt_args.get("reward_scale", 1.0))
    reward_prediction_source = str(ckpt_args.get("reward_prediction_source", "raw"))
    ckpt_segment_ks = long_eval._segment_ks_from_ckpt_args(ckpt_args)
    if args.planner_segment_ks:
        horizon_choices = sorted({int(k) for k in args.planner_segment_ks})
    elif len(ckpt_segment_ks) > 1:
        horizon_choices = ckpt_segment_ks
    else:
        horizon_choices = [int(ckpt_args["segment_k"])]
    planner_enabled = len(horizon_choices) > 1
    max_seed_vacancies = int(ckpt_args["max_seed_vacancies"])
    max_candidate_sites = int(ckpt_args["max_candidate_sites"])
    candidate_mode, edit_mode, effective_duration_source = _constraint_settings(args)

    model = None
    duration_model = None
    if args.constraint_mode != "no_change":
        model = long_eval._build_model(ckpt, args.device)
        if args.duration_checkpoint:
            duration_ckpt = torch.load(Path(args.duration_checkpoint), map_location=args.device, weights_only=False)
            duration_model = long_eval._build_model(duration_ckpt, args.device)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    env_cfg = long_eval._build_env_cfg(ckpt_args, max_episode_steps_override=args.max_episode_steps_override)
    if args.temperature_override is not None:
        env_cfg["temperature"] = float(args.temperature_override)
    if args.cu_density_override is not None:
        env_cfg["cu_density"] = float(args.cu_density_override)
    teacher_env = mod.MacroKMCEnv(env_cfg)
    teacher_env.reset()
    model_env = copy.deepcopy(teacher_env)
    last_reference_env = copy.deepcopy(teacher_env)
    initial_vac, initial_cu = _environment_state_arrays(model_env)
    initial_vacancy_count = int(initial_vac.shape[0])
    initial_cu_count = int(initial_cu.shape[0])

    segments: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    edit_trace: list[dict[str, Any]] = []
    pred_tau_exp: list[float] = []
    true_tau_exp: list[float] = []
    pred_reward_sum: list[float] = []
    true_reward_sum: list[float] = []
    model_delta_e: list[float] = []
    chosen_ks: list[int] = []
    reachability_violations: list[float] = []
    inventory_delta_l1: list[float] = []
    state_metric_rows: list[dict[str, float]] = []
    prediction_wall_times: list[float] = []
    apply_wall_times: list[float] = []
    teacher_wall_times: list[float] = []
    stop_reason = "completed"
    stop_segment: dict[str, Any] | None = None
    skipped_noop = 0
    skipped_terminal = 0

    if args.save_snapshots:
        snapshots.append(
            {
                "index": 0,
                "model": _state_snapshot(model_env, max_cu=args.snapshot_max_cu, rng=snapshot_rng),
                "teacher": _state_snapshot(last_reference_env, max_cu=args.snapshot_max_cu, rng=snapshot_rng),
            }
        )

    with torch.no_grad():
        for segment_idx in range(int(args.rollout_segments)):
            predict_t0 = time.perf_counter()
            if args.constraint_mode == "no_change":
                candidates = [_no_change_candidate(model_env, int(horizon_choices[0]), effective_duration_source)]
            else:
                assert model is not None
                candidates = [
                    item
                    for item in (
                        _predict_closed_loop_candidate(
                            model=model,
                            duration_model=duration_model,
                            env=model_env,
                            horizon_k=item_k,
                            max_seed_vacancies=max_seed_vacancies,
                            max_candidate_sites=max_candidate_sites,
                            reward_scale=reward_scale,
                            device=args.device,
                            candidate_mode=candidate_mode,
                            edit_mode=edit_mode,
                            duration_source=effective_duration_source,
                            duration_blend_alpha=args.duration_blend_alpha,
                            duration_log_offset=args.duration_log_offset,
                            planner_score_mode=args.planner_score_mode,
                            planner_tau_residual_penalty=args.planner_tau_residual_penalty,
                            planner_k_penalty_power=args.planner_k_penalty_power,
                            raw_changed_budget_multiplier=args.raw_changed_budget_multiplier,
                            inventory_stress_mode=args.inventory_stress_mode,
                            rng=rng,
                            global_candidate_cu_fraction=args.global_candidate_cu_fraction,
                            reward_prediction_source=reward_prediction_source,
                        )
                        for item_k in horizon_choices
                    )
                    if item is not None
                ]
            effective_min = 0 if args.constraint_mode == "no_change" else int(args.min_projected_changed_sites)
            selected = _choose_closed_loop_candidate(candidates, min_projected_changed_sites=effective_min)
            prediction_wall_times.append(float(time.perf_counter() - predict_t0))
            if selected is None:
                stop_reason = "no_legal_closed_loop_candidate"
                stop_segment = {"index": segment_idx, "planner_candidates": _json_safe_candidates(candidates)}
                break
            selected_k = int(selected["segment_k"])

            if args.reference_mode == "on_policy_teacher_probe":
                reference_env = copy.deepcopy(model_env)
            else:
                reference_env = teacher_env
            teacher_t0 = time.perf_counter()
            teacher_segment = long_eval._collect_teacher_segment(reference_env, horizon_k=selected_k, rng=rng)
            teacher_wall_times.append(float(time.perf_counter() - teacher_t0))
            if teacher_segment is None:
                skipped_terminal += 1
                stop_reason = "teacher_terminal_or_action_missing"
                stop_segment = {
                    "index": segment_idx,
                    "segment_k": selected_k,
                    "selected": _json_safe_candidate(selected),
                }
                break
            if bool(teacher_segment.get("is_noop", False)) and not args.allow_teacher_noop_segments:
                skipped_noop += 1
                stop_reason = "noop_teacher_segment"
                stop_segment = {
                    "index": segment_idx,
                    "segment_k": selected_k,
                    "selected": _json_safe_candidate(selected),
                    "traditional_kmc_reward_sum": float(teacher_segment["reward_sum"]),
                    "traditional_kmc_expected_tau": float(teacher_segment["tau_exp"]),
                    "traditional_changed_site_count": int(teacher_segment.get("changed_site_count", 0)),
                }
                break

            apply_t0 = time.perf_counter()
            if args.constraint_mode == "no_change":
                applied = _apply_no_change_to_env(model_env, selected)
            else:
                applied = _apply_candidate_to_env(
                    model_env,
                    selected,
                    predicted_tau=float(selected["predicted_expected_tau"]),
                )
            apply_wall_times.append(float(time.perf_counter() - apply_t0))
            if args.save_edit_trace:
                edit_trace.append(
                    _candidate_edit_trace_row(
                        segment_idx=segment_idx,
                        candidate=selected,
                        applied=applied,
                    )
                )

            if args.reference_mode == "independent_teacher":
                last_reference_env = reference_env
            else:
                last_reference_env = reference_env

            state_metrics = _state_metrics(
                model_env=model_env,
                teacher_env=last_reference_env,
                initial_vacancy_count=initial_vacancy_count,
                initial_cu_count=initial_cu_count,
            )
            state_metric_rows.append(state_metrics)
            pred_tau_exp.append(float(selected["predicted_expected_tau"]))
            true_tau_exp.append(float(teacher_segment["tau_exp"]))
            pred_reward_sum.append(float(selected["predicted_reward_sum"]))
            true_reward_sum.append(float(teacher_segment["reward_sum"]))
            model_delta_e.append(float(applied["model_delta_e"]))
            chosen_ks.append(selected_k)
            reachability_violations.append(float(selected["reachability_violation"]))
            inventory_delta_l1.append(float(selected.get("candidate_inventory_delta_l1", 0.0)))

            row = {
                "index": int(segment_idx),
                "segment_k": int(selected_k),
                "constraint_mode": args.constraint_mode,
                "candidate_mode": str(selected["candidate_mode"]),
                "edit_mode": str(selected["edit_mode"]),
                "selection_score": float(selected["selection_score"]),
                "predicted_reward_sum": float(selected["predicted_reward_sum"]),
                "traditional_kmc_reward_sum": float(teacher_segment["reward_sum"]),
                "predicted_delta_e": float(selected["predicted_delta_e"]),
                "model_applied_delta_e": float(applied["model_delta_e"]),
                "traditional_kmc_delta_e": float(teacher_segment["reward_sum"] / reward_scale),
                "predicted_expected_tau": float(selected["predicted_expected_tau"]),
                "traditional_kmc_expected_tau": float(teacher_segment["tau_exp"]),
                "model_expected_tau": float(selected["model_expected_tau"]),
                "baseline_expected_tau": float(selected["baseline_expected_tau"]),
                "reachability_violation": float(selected["reachability_violation"]),
                "candidate_inventory_delta_l1": float(selected.get("candidate_inventory_delta_l1", 0.0)),
                "projected_changed_count": float(selected["projected_changed_count"]),
                "applied_changed_count": float(applied["applied_changed_count"]),
                "traditional_changed_site_count": int(teacher_segment.get("changed_site_count", 0)),
                "prediction_wall_time_sec": float(prediction_wall_times[-1]),
                "apply_wall_time_sec": float(apply_wall_times[-1]),
                "teacher_wall_time_sec": float(teacher_wall_times[-1]),
                **state_metrics,
            }
            segments.append(row)

            if args.save_snapshots and (
                (segment_idx + 1) % max(int(args.snapshot_every), 1) == 0
                or segment_idx + 1 == int(args.rollout_segments)
            ):
                snapshots.append(
                    {
                        "index": int(segment_idx + 1),
                        "model": _state_snapshot(model_env, max_cu=args.snapshot_max_cu, rng=snapshot_rng),
                        "teacher": _state_snapshot(last_reference_env, max_cu=args.snapshot_max_cu, rng=snapshot_rng),
                    }
                )

            if args.progress_every > 0 and (segment_idx + 1) % int(args.progress_every) == 0:
                print(
                    json.dumps(
                        {
                            "closed_loop_progress": {
                                "segments": int(segment_idx + 1),
                                "constraint_mode": args.constraint_mode,
                                "chosen_k_histogram": _histogram_int(chosen_ks),
                                "tau_ratio": float(np.sum(pred_tau_exp) / max(np.sum(true_tau_exp), 1e-12)),
                                "energy_abs_error": float(state_metrics["energy_abs_error"]),
                                "cu_chamfer": float(state_metrics["cu_chamfer_mean_distance"]),
                            }
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    pred_tau_exp_np = np.asarray(pred_tau_exp, dtype=np.float64)
    true_tau_exp_np = np.asarray(true_tau_exp, dtype=np.float64)
    pred_reward_sum_np = np.asarray(pred_reward_sum, dtype=np.float64)
    true_reward_sum_np = np.asarray(true_reward_sum, dtype=np.float64)
    model_delta_e_np = np.asarray(model_delta_e, dtype=np.float64)
    teacher_delta_e_np = true_reward_sum_np / max(reward_scale, 1e-12)
    energy_abs_errors = [row["energy_abs_error"] for row in state_metric_rows]
    cu_chamfer = [row["cu_chamfer_mean_distance"] for row in state_metric_rows]
    vac_dist = [row["vacancy_matching_mean_distance"] for row in state_metric_rows]
    topology_errors = {
        key: [row[key] for row in state_metric_rows]
        for key in [
            "cu_neighbor_edges_abs_error",
            "cu_mean_degree_abs_error",
            "cu_isolated_fraction_abs_error",
            "cu_largest_cluster_fraction_abs_error",
        ]
    }
    completed = int(len(segments))
    cumulative = {
        "predicted_expected_time_final": float(pred_tau_exp_np.sum()) if completed else 0.0,
        "traditional_kmc_expected_time_final": float(true_tau_exp_np.sum()) if completed else 0.0,
        "expected_time_ratio": float(pred_tau_exp_np.sum() / true_tau_exp_np.sum()) if completed and true_tau_exp_np.sum() > 1e-12 else None,
        "predicted_reward_sum_final": float(pred_reward_sum_np.sum()) if completed else 0.0,
        "traditional_kmc_reward_sum_final": float(true_reward_sum_np.sum()) if completed else 0.0,
        "model_applied_delta_e_final": float(model_delta_e_np.sum()) if completed else 0.0,
        "traditional_kmc_delta_e_final": float(teacher_delta_e_np.sum()) if completed else 0.0,
        "model_delta_e_mae": float(np.mean(np.abs(model_delta_e_np - teacher_delta_e_np))) if completed else 0.0,
        "cumulative_model_delta_e_mae": float(
            np.mean(np.abs(np.cumsum(model_delta_e_np) - np.cumsum(teacher_delta_e_np)))
        )
        if completed
        else 0.0,
    }
    summary = {
        "mode": "closed_loop_autonomous_macro_rollout",
        "reference_mode": args.reference_mode,
        "constraint_mode": args.constraint_mode,
        "candidate_mode": candidate_mode,
        "edit_mode": edit_mode,
        "duration_source": effective_duration_source,
        "checkpoint": str(checkpoint_path),
        "duration_checkpoint": str(Path(args.duration_checkpoint)) if args.duration_checkpoint else None,
        "temperature": float(env_cfg["temperature"]),
        "cu_density": float(env_cfg["cu_density"]),
        "seed": int(args.seed),
        "segment_ks": horizon_choices,
        "planner_enabled": planner_enabled,
        "requested_rollout_segments": int(args.rollout_segments),
        "completed_rollout_segments": completed,
        "stop_reason": stop_reason,
        "stop_segment": stop_segment,
        "skipped_noop": int(skipped_noop),
        "skipped_terminal": int(skipped_terminal),
        "allow_teacher_noop_segments": bool(args.allow_teacher_noop_segments),
        "chosen_k_histogram": _histogram_int(chosen_ks),
        "macro_efficiency": {
            "macro_steps": completed,
            "teacher_micro_events_replaced": int(np.sum(chosen_ks)) if chosen_ks else 0,
            "compression_ratio_micro_events_per_macro_step": float(np.mean(chosen_ks)) if chosen_ks else 0.0,
            "prediction_wall_time_mean_sec": _mean(prediction_wall_times),
            "apply_wall_time_mean_sec": _mean(apply_wall_times),
            "teacher_segment_wall_time_mean_sec": _mean(teacher_wall_times),
            "teacher_over_model_prediction_speedup": (
                float(_mean(teacher_wall_times) / max(_mean(prediction_wall_times) + _mean(apply_wall_times), 1e-12))
                if completed
                else None
            ),
            "cuda_max_memory_allocated_mb": (
                float(torch.cuda.max_memory_allocated() / (1024**2))
                if torch.cuda.is_available() and str(args.device).startswith("cuda")
                else None
            ),
        },
        "reward_sum": long_eval._compute_metrics(pred_reward_sum_np, true_reward_sum_np) if completed else {},
        "reward_diagnostics": mod._compute_reward_diagnostics(pred_reward_sum_np, true_reward_sum_np) if completed else {},
        "tau_expected": (
            {
                **long_eval._compute_metrics(pred_tau_exp_np, true_tau_exp_np),
                **long_eval._compute_log_metrics(pred_tau_exp_np, true_tau_exp_np),
            }
            if completed
            else {}
        ),
        "physical_consistency": {
            "energy_abs_error_mean": _mean(energy_abs_errors),
            "energy_abs_error_final": float(energy_abs_errors[-1]) if energy_abs_errors else 0.0,
            "vacancy_matching_mean_distance_mean": _mean(vac_dist),
            "cu_chamfer_mean_distance_mean": _mean(cu_chamfer),
            "inventory_violation_l1_mean": _mean([row["inventory_violation_l1"] for row in state_metric_rows]),
            "candidate_inventory_delta_l1_mean": _mean(inventory_delta_l1),
            "reachability_violation_rate_mean": _mean(reachability_violations),
            **{f"{key}_mean": _mean(values) for key, values in topology_errors.items()},
        },
        "cumulative": cumulative,
        "arrays": {
            "predicted_expected_tau_cumsum": np.cumsum(pred_tau_exp_np).tolist(),
            "traditional_kmc_expected_tau_cumsum": np.cumsum(true_tau_exp_np).tolist(),
            "model_delta_e_cumsum": np.cumsum(model_delta_e_np).tolist(),
            "traditional_kmc_delta_e_cumsum": np.cumsum(teacher_delta_e_np).tolist(),
            "energy_abs_error": energy_abs_errors,
            "cu_chamfer_mean_distance": cu_chamfer,
            "vacancy_matching_mean_distance": vac_dist,
        },
        "segment_preview": segments[: max(int(args.print_segments), 0)],
        "segments": segments,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary["snapshot_path"] = None
    summary["edit_trace_path"] = None
    if args.save_snapshots:
        snapshot_path = output_path.with_suffix(".snapshots.json")
        snapshot_path.write_text(json.dumps({"snapshots": snapshots}, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["snapshot_path"] = str(snapshot_path)
    if args.save_edit_trace:
        edit_trace_path = output_path.with_suffix(".edit_trace.json")
        edit_trace_path.write_text(json.dumps({"edit_trace": edit_trace}, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["edit_trace_path"] = str(edit_trace_path)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print("AtomWorld-Mirror Closed-loop Autonomous Rollout")
    print(
        f"completed={completed}/{args.rollout_segments}, stop_reason={stop_reason}, "
        f"constraint_mode={args.constraint_mode}, segment_ks={horizon_choices}"
    )
    if completed:
        print(
            "Expected-time: "
            f"log_mae={summary['tau_expected']['log_mae']:.4f}, "
            f"scale={summary['tau_expected']['scale_ratio']:.4f}, "
            f"ratio={summary['cumulative']['expected_time_ratio']:.4f}"
        )
        print(
            "Physical consistency: "
            f"energy_err_mean={summary['physical_consistency']['energy_abs_error_mean']:.4f}, "
            f"cu_chamfer_mean={summary['physical_consistency']['cu_chamfer_mean_distance_mean']:.4f}, "
            f"reach_viol={summary['physical_consistency']['reachability_violation_rate_mean']:.4f}, "
            f"inventory_l1={summary['physical_consistency']['inventory_violation_l1_mean']:.4f}"
        )
        print(
            "Efficiency: "
            f"compression={summary['macro_efficiency']['compression_ratio_micro_events_per_macro_step']:.2f}x, "
            f"teacher/model_wall={summary['macro_efficiency']['teacher_over_model_prediction_speedup']:.3f}"
        )
    print(f"Saved summary to {output_path}")


def _json_safe_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    drop_keys = {"candidate_positions", "candidate_mask", "current_types", "selected_types"}
    return {key: value for key, value in candidate.items() if key not in drop_keys}


def _json_safe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_json_safe_candidate(candidate) for candidate in candidates]


if __name__ == "__main__":
    main()
