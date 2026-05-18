from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

import train_dreamer_macro_edit as train_mod


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only teacher action-edge support diagnostic for macro-edit checkpoints."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--segment_ks", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--samples_per_k", type=int, default=6)
    parser.add_argument("--topk_budgets", type=int, nargs="+", default=[32, 64, 96, 128, 256])
    parser.add_argument("--sources", nargs="+", default=["change", "proposal", "action_support"])
    parser.add_argument(
        "--neighbor_expand_anchor_sources",
        nargs="*",
        default=[],
        help="Optional sources whose top-k anchors are expanded by 1NN candidate neighbors for edge-completion upper bounds.",
    )
    parser.add_argument(
        "--neighbor_expand_caps",
        type=int,
        nargs="*",
        default=[96, 128, 160, 256],
        help="Caps for neighbor-expanded support sets. Use 0 to keep the uncapped expanded set.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lattice_size", type=int, nargs=3, default=[40, 40, 40])
    parser.add_argument("--cu_density", type=float, default=0.0134)
    parser.add_argument("--v_density", type=float, default=0.0002)
    parser.add_argument("--max_episode_steps", type=int, default=1024)
    parser.add_argument("--max_vacancies", type=int, default=32)
    parser.add_argument("--max_defects", type=int, default=64)
    parser.add_argument("--max_shells", type=int, default=16)
    parser.add_argument("--stats_dim", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--reward_scale", type=float, default=10.0)
    parser.add_argument("--neighbor_order", default="2NN")
    parser.add_argument("--max_seed_vacancies", type=int, default=32)
    parser.add_argument("--max_candidate_sites", type=int, default=2048)
    parser.add_argument("--segment_boundary_mode", choices=["fixed_k", "adaptive_key_event"], default="adaptive_key_event")
    parser.add_argument("--adaptive_min_k", type=int, default=8)
    parser.add_argument("--adaptive_candidate_horizon_source", choices=["nominal", "actual"], default="actual")
    parser.add_argument("--adaptive_key_moving_types", type=int, nargs="*", default=[train_mod.CU_TYPE])
    parser.add_argument("--adaptive_min_touched_sites", type=int, default=0)
    parser.add_argument("--adaptive_abs_delta_e_threshold", type=float, default=0.0)
    parser.add_argument("--adaptive_cumulative_abs_delta_e_threshold", type=float, default=0.0)
    return parser.parse_args()


def _env_cfg(args: argparse.Namespace) -> dict[str, object]:
    return {
        "lattice_size": tuple(args.lattice_size),
        "max_episode_steps": int(args.max_episode_steps),
        "max_vacancies": int(args.max_vacancies),
        "max_defects": int(args.max_defects),
        "max_shells": int(args.max_shells),
        "stats_dim": int(args.stats_dim),
        "temperature": float(args.temperature),
        "reward_scale": float(args.reward_scale),
        "cu_density": float(args.cu_density),
        "v_density": float(args.v_density),
        "rlkmc_topk": 16,
        "neighbor_order": str(args.neighbor_order),
    }


def _boundary_config(args: argparse.Namespace) -> train_mod.AdaptiveBoundaryConfig:
    return train_mod.AdaptiveBoundaryConfig(
        mode=args.segment_boundary_mode,
        min_k=max(1, int(args.adaptive_min_k)),
        candidate_horizon_source=args.adaptive_candidate_horizon_source,
        key_moving_types=tuple(int(item) for item in (args.adaptive_key_moving_types or [])),
        min_touched_sites=max(0, int(args.adaptive_min_touched_sites)),
        abs_delta_e_threshold=max(0.0, float(args.adaptive_abs_delta_e_threshold)),
        cumulative_abs_delta_e_threshold=max(0.0, float(args.adaptive_cumulative_abs_delta_e_threshold)),
    )


def _position_to_index(
    candidate_positions: np.ndarray,
    candidate_mask: np.ndarray,
) -> dict[tuple[int, int, int], int]:
    lookup: dict[tuple[int, int, int], int] = {}
    for idx in np.flatnonzero(candidate_mask > 0).tolist():
        pos = tuple(int(x) for x in candidate_positions[idx].astype(np.int32).tolist())
        lookup[pos] = int(idx)
    return lookup


def _mask_from_positions(
    lookup: dict[tuple[int, int, int], int],
    max_candidate_sites: int,
    positions: Iterable[tuple[int, int, int]],
) -> np.ndarray:
    mask = np.zeros((max_candidate_sites,), dtype=np.float32)
    for pos in positions:
        idx = lookup.get(tuple(int(x) for x in pos))
        if idx is not None:
            mask[idx] = 1.0
    return mask


def _topk_indices(logits: np.ndarray, valid_mask: np.ndarray, budget: int) -> set[int]:
    valid_idx = np.flatnonzero(valid_mask > 0)
    if valid_idx.size == 0 or budget <= 0:
        return set()
    k = min(int(budget), int(valid_idx.size))
    order = np.argsort(logits[valid_idx])[-k:]
    return {int(valid_idx[i]) for i in order.tolist()}


def _ranked_subset(indices: set[int], logits: np.ndarray, limit: int) -> set[int]:
    if limit <= 0 or len(indices) <= limit:
        return set(indices)
    ranked = sorted(indices, key=lambda idx: float(logits[int(idx)]), reverse=True)
    return {int(idx) for idx in ranked[:limit]}


def _neighbor_expanded_indices(
    *,
    selected: set[int],
    candidate_positions: np.ndarray,
    candidate_mask: np.ndarray,
    lookup: dict[tuple[int, int, int], int],
    nn1: np.ndarray,
    box: np.ndarray,
    logits: np.ndarray,
    cap: int,
) -> set[int]:
    expanded = set(selected)
    valid = set(int(idx) for idx in np.flatnonzero(candidate_mask > 0).tolist())
    for idx in selected:
        if int(idx) not in valid:
            continue
        pos = tuple(int(x) for x in candidate_positions[int(idx)].astype(np.int32).tolist())
        for nxt in train_mod._one_hop_neighbors(pos, nn1, box):
            nxt_idx = lookup.get(nxt)
            if nxt_idx is not None and int(nxt_idx) in valid:
                expanded.add(int(nxt_idx))
    return _ranked_subset(expanded, logits, int(cap)) if int(cap) > 0 else expanded


def _prf(selected: set[int], target: set[int]) -> dict[str, float]:
    overlap = selected & target
    precision = float(len(overlap) / max(len(selected), 1))
    recall = float(len(overlap) / max(len(target), 1))
    f1 = float(2.0 * precision * recall / max(precision + recall, 1e-12))
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "overlap_count": float(len(overlap)),
        "target_count": float(len(target)),
        "selected_count": float(len(selected)),
    }


def _edge_metrics(
    selected: set[int],
    edge_pairs: list[tuple[int | None, int | None]],
) -> dict[str, float]:
    valid_pairs = [(old, new) for old, new in edge_pairs if old is not None and new is not None]
    if not valid_pairs:
        return {
            "edge_pair_recall": 0.0,
            "edge_either_recall": 0.0,
            "first_edge_pair_hit": 0.0,
            "valid_edge_count": 0.0,
        }
    pair_hits = sum(1 for old, new in valid_pairs if old in selected and new in selected)
    either_hits = sum(1 for old, new in valid_pairs if old in selected or new in selected)
    first_old, first_new = valid_pairs[0]
    return {
        "edge_pair_recall": float(pair_hits / len(valid_pairs)),
        "edge_either_recall": float(either_hits / len(valid_pairs)),
        "first_edge_pair_hit": float(first_old in selected and first_new in selected),
        "valid_edge_count": float(len(valid_pairs)),
    }


@torch.no_grad()
def _decode_logits(
    model: torch.nn.Module,
    tensors: dict[str, torch.Tensor],
) -> dict[str, np.ndarray]:
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
    change_logits, _ = model.decode_edit(
        site_latent=site_latent,
        patch_latent=patch_latent,
        predicted_next_global=next_pred,
        path_latent=path_latent,
        horizon_k=tensors["horizon_k"],
        current_types=tensors["current_types"],
    )
    proposal_logits = model.decode_proposal(
        site_latent=site_latent,
        patch_latent=patch_latent,
        predicted_next_global=next_pred,
        path_latent=path_latent,
        horizon_k=tensors["horizon_k"],
        current_types=tensors["current_types"],
    )
    if hasattr(model, "decode_action_support"):
        action_support_logits = model.decode_action_support(
            site_latent=site_latent,
            patch_latent=patch_latent,
            predicted_next_global=next_pred,
            path_latent=path_latent,
            horizon_k=tensors["horizon_k"],
            current_types=tensors["current_types"],
        )
    else:
        action_support_logits = proposal_logits
    if hasattr(model, "decode_action_source_support"):
        action_source_logits = model.decode_action_source_support(
            site_latent=site_latent,
            patch_latent=patch_latent,
            predicted_next_global=next_pred,
            path_latent=path_latent,
            horizon_k=tensors["horizon_k"],
            current_types=tensors["current_types"],
        )
    else:
        action_source_logits = action_support_logits
    if hasattr(model, "decode_action_destination_support"):
        action_destination_logits = model.decode_action_destination_support(
            site_latent=site_latent,
            patch_latent=patch_latent,
            predicted_next_global=next_pred,
            path_latent=path_latent,
            horizon_k=tensors["horizon_k"],
            current_types=tensors["current_types"],
        )
    else:
        action_destination_logits = action_support_logits
    if hasattr(train_mod, "combine_action_endpoint_logits"):
        action_endpoint_logits = train_mod.combine_action_endpoint_logits(action_source_logits, action_destination_logits)
    else:
        action_endpoint_logits = torch.maximum(action_source_logits, action_destination_logits)
    return {
        "change": change_logits[0].detach().cpu().numpy(),
        "proposal": proposal_logits[0].detach().cpu().numpy(),
        "action_support": action_support_logits[0].detach().cpu().numpy(),
        "action_source": action_source_logits[0].detach().cpu().numpy(),
        "action_destination": action_destination_logits[0].detach().cpu().numpy(),
        "action_endpoint": action_endpoint_logits[0].detach().cpu().numpy(),
    }


def _build_tensors(
    *,
    env: train_mod.MacroKMCEnv,
    candidate_positions: list[tuple[int, int, int]],
    depth_map: dict[tuple[int, int, int], int],
    seeds: np.ndarray,
    horizon_k: int,
    max_candidate_sites: int,
    device: str,
) -> tuple[dict[str, torch.Tensor], dict[str, np.ndarray]]:
    start_vacancies = env.env.get_vacancy_array().astype(np.int32)
    start_cu = env.env.get_cu_array().astype(np.int32)
    start_vac_set, start_cu_set = train_mod._positions_to_type_lookup(start_vacancies, start_cu)
    positions, nearest_offsets, reach_depth, is_start_vacancy, current_types, _, _ = train_mod._build_patch_features(
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
    tensors = {
        "start_obs": torch.tensor(env.obs()[None, :], dtype=torch.float32, device=device),
        "global_summary": torch.tensor(train_mod._global_summary(env)[None, :], dtype=torch.float32, device=device),
        "candidate_positions": torch.tensor(positions[None, ...], dtype=torch.float32, device=device),
        "nearest_vacancy_offset": torch.tensor(nearest_offsets[None, ...], dtype=torch.float32, device=device),
        "reach_depth": torch.tensor(reach_depth[None, ...], dtype=torch.float32, device=device),
        "is_start_vacancy": torch.tensor(is_start_vacancy[None, ...], dtype=torch.float32, device=device),
        "current_types": torch.tensor(current_types[None, ...], dtype=torch.long, device=device),
        "candidate_mask": torch.tensor(candidate_mask[None, ...], dtype=torch.float32, device=device),
        "box_dims": torch.tensor(np.asarray(env.env.dims, dtype=np.float32)[None, :], dtype=torch.float32, device=device),
        "horizon_k": torch.tensor([horizon_k], dtype=torch.long, device=device),
    }
    arrays = {
        "positions": positions,
        "candidate_mask": candidate_mask,
    }
    return tensors, arrays


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _metric_summary(rows: list[dict[str, object]]) -> dict[str, float]:
    return {
        "changed_f1": _safe_mean([float(row["changed"]["f1"]) for row in rows]),
        "changed_recall": _safe_mean([float(row["changed"]["recall"]) for row in rows]),
        "touched_f1": _safe_mean([float(row["touched"]["f1"]) for row in rows]),
        "touched_recall": _safe_mean([float(row["touched"]["recall"]) for row in rows]),
        "old_recall": _safe_mean([float(row["old"]["recall"]) for row in rows]),
        "new_recall": _safe_mean([float(row["new"]["recall"]) for row in rows]),
        "edge_pair_recall": _safe_mean([float(row["edge"]["edge_pair_recall"]) for row in rows]),
        "edge_either_recall": _safe_mean([float(row["edge"]["edge_either_recall"]) for row in rows]),
        "first_edge_pair_hit": _safe_mean([float(row["edge"]["first_edge_pair_hit"]) for row in rows]),
        "selected_count": _safe_mean([float(row["changed"]["selected_count"]) for row in rows]),
    }


def _summarize(samples: list[dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {
        "num_samples": len(samples),
        "coverage": float(np.mean([sample["candidate_teacher_changed_recall"] for sample in samples])) if samples else 0.0,
        "teacher_changed_count_mean": _safe_mean([float(sample["teacher_changed_count"]) for sample in samples]),
        "teacher_touched_count_mean": _safe_mean([float(sample["teacher_touched_count"]) for sample in samples]),
        "teacher_edge_count_mean": _safe_mean([float(sample["teacher_edge_count"]) for sample in samples]),
        "by_source_budget": {},
    }
    all_keys = sorted({key for sample in samples for key in sample["metrics"]})
    for key in all_keys:
        rows = [sample["metrics"][key] for sample in samples if key in sample["metrics"]]
        if rows:
            summary["by_source_budget"][key] = _metric_summary(rows)
    return summary


def main() -> None:
    args = _parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model, ckpt_args = train_mod._build_planner_model_from_checkpoint(args.checkpoint, str(device))
    model.eval()
    env_cfg = _env_cfg(args)
    boundary_config = _boundary_config(args)
    neighbor_expand_sources = {str(item) for item in (args.neighbor_expand_anchor_sources or [])}
    neighbor_expand_caps = [int(item) for item in (args.neighbor_expand_caps or [])]
    samples: list[dict[str, object]] = []
    for horizon_idx, horizon_k in enumerate(args.segment_ks):
        env = train_mod.MacroKMCEnv(copy.deepcopy(env_cfg))
        rng = np.random.default_rng(int(args.seed) + 1009 * horizon_idx)
        obs = env.reset()
        attempts = 0
        while sum(1 for sample in samples if sample["nominal_horizon_k"] == int(horizon_k)) < int(args.samples_per_k):
            attempts += 1
            if attempts > int(args.samples_per_k) * 50:
                break
            adapter = train_mod._build_natural_teacher_adapter(
                backend="kmc",
                env=env,
                teacher_mode="kmc",
                neural_teacher=None,
                neural_teacher_device=str(device),
                neural_teacher_temperature=1.0,
                neural_teacher_epsilon=0.0,
            )
            start_vac_set, start_cu_set = train_mod._positions_to_type_lookup(
                adapter.vacancy_positions(),
                adapter.cu_positions(),
            )
            candidate_positions, depth_map, seeds = adapter.build_candidate_positions(
                horizon_k=int(horizon_k),
                max_seed_vacancies=int(args.max_seed_vacancies),
                max_candidate_sites=int(args.max_candidate_sites),
            )
            if not candidate_positions:
                env = train_mod.MacroKMCEnv(copy.deepcopy(env_cfg))
                obs = env.reset()
                continue
            tensors, arrays = _build_tensors(
                env=env,
                candidate_positions=candidate_positions,
                depth_map=depth_map,
                seeds=seeds,
                horizon_k=int(horizon_k),
                max_candidate_sites=int(args.max_candidate_sites),
                device=str(device),
            )
            logits = _decode_logits(model, tensors)
            next_obs, done, path_infos, tau_exp, tau_real, reward_sum, touched_positions, realized_horizon_k, boundary_hit = (
                train_mod._rollout_teacher_path(
                    adapter=adapter,
                    rng=rng,
                    max_horizon_k=int(horizon_k),
                    boundary_config=boundary_config,
                )
            )
            if done or int(realized_horizon_k) <= 0:
                env = train_mod.MacroKMCEnv(copy.deepcopy(env_cfg))
                obs = env.reset()
                continue
            end_vac_set, end_cu_set = train_mod._positions_to_type_lookup(
                adapter.vacancy_positions(),
                adapter.cu_positions(),
            )
            changed_positions = train_mod._changed_positions_between(start_vac_set, start_cu_set, end_vac_set, end_cu_set)
            lookup = _position_to_index(arrays["positions"], arrays["candidate_mask"])
            nn1 = adapter.nn1_offsets()
            box = np.asarray(env.env.dims, dtype=np.int32)
            old_positions = {tuple(int(x) for x in info["old_pos"].tolist()) for info in path_infos}
            new_positions = {tuple(int(x) for x in info["new_pos"].tolist()) for info in path_infos}
            edge_pairs = [
                (
                    lookup.get(tuple(int(x) for x in info["old_pos"].tolist())),
                    lookup.get(tuple(int(x) for x in info["new_pos"].tolist())),
                )
                for info in path_infos
            ]
            masks = {
                "changed": set(np.flatnonzero(_mask_from_positions(lookup, int(args.max_candidate_sites), changed_positions) > 0).tolist()),
                "touched": set(np.flatnonzero(_mask_from_positions(lookup, int(args.max_candidate_sites), touched_positions) > 0).tolist()),
                "old": set(np.flatnonzero(_mask_from_positions(lookup, int(args.max_candidate_sites), old_positions) > 0).tolist()),
                "new": set(np.flatnonzero(_mask_from_positions(lookup, int(args.max_candidate_sites), new_positions) > 0).tolist()),
            }
            metrics: dict[str, object] = {}
            for source in args.sources:
                if source not in logits:
                    continue
                for budget in args.topk_budgets:
                    selected = _topk_indices(logits[source], arrays["candidate_mask"], int(budget))
                    metrics[f"{source}_topk{budget}"] = {
                        "changed": _prf(selected, masks["changed"]),
                        "touched": _prf(selected, masks["touched"]),
                        "old": _prf(selected, masks["old"]),
                        "new": _prf(selected, masks["new"]),
                        "edge": _edge_metrics(selected, edge_pairs),
                    }
                    if source in neighbor_expand_sources:
                        for cap in neighbor_expand_caps:
                            expanded = _neighbor_expanded_indices(
                                selected=selected,
                                candidate_positions=arrays["positions"],
                                candidate_mask=arrays["candidate_mask"],
                                lookup=lookup,
                                nn1=nn1,
                                box=box,
                                logits=logits[source],
                                cap=int(cap),
                            )
                            cap_name = "uncap" if int(cap) <= 0 else str(int(cap))
                            metrics[f"{source}_topk{budget}_nn1cap{cap_name}"] = {
                                "changed": _prf(expanded, masks["changed"]),
                                "touched": _prf(expanded, masks["touched"]),
                                "old": _prf(expanded, masks["old"]),
                                "new": _prf(expanded, masks["new"]),
                                "edge": _edge_metrics(expanded, edge_pairs),
                            }
            samples.append(
                {
                    "nominal_horizon_k": int(horizon_k),
                    "realized_horizon_k": int(realized_horizon_k),
                    "boundary_hit": bool(boundary_hit),
                    "tau_exp": float(tau_exp),
                    "tau_real": float(tau_real),
                    "reward_sum": float(reward_sum),
                    "teacher_changed_count": float(len(changed_positions)),
                    "teacher_touched_count": float(len(touched_positions)),
                    "teacher_old_count": float(len(old_positions)),
                    "teacher_new_count": float(len(new_positions)),
                    "teacher_edge_count": float(len(edge_pairs)),
                    "candidate_count": float(arrays["candidate_mask"].sum()),
                    "candidate_teacher_changed_recall": float(len(masks["changed"]) / max(len(changed_positions), 1)),
                    "candidate_teacher_touched_recall": float(len(masks["touched"]) / max(len(touched_positions), 1)),
                    "metrics": metrics,
                }
            )
            obs = next_obs
    payload = {
        "checkpoint": args.checkpoint,
        "checkpoint_args": {
            "segment_ks": ckpt_args.get("segment_ks"),
            "max_candidate_sites": ckpt_args.get("max_candidate_sites"),
        },
        "segment_ks": [int(k) for k in args.segment_ks],
        "samples_per_k": int(args.samples_per_k),
        "topk_budgets": [int(k) for k in args.topk_budgets],
        "sources": list(args.sources),
        "neighbor_expand_anchor_sources": sorted(neighbor_expand_sources),
        "neighbor_expand_caps": neighbor_expand_caps,
        "summary": _summarize(samples),
        "samples": samples,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
