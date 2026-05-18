from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import eval_macro_long_trajectory as long_eval
import train_dreamer_macro_edit as train_mod
from dreamer4 import macro_edit as mod


def _load_split_samples(cache_path: Path) -> dict[str, list[train_mod.MacroSegmentSample]]:
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    if isinstance(payload, list):
        return {"all": [train_mod.MacroSegmentSample(**item) for item in payload]}
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported cache payload type: {type(payload)!r}")
    splits: dict[str, list[train_mod.MacroSegmentSample]] = {}
    for split in ("train", "val"):
        items = payload.get(split)
        if items is None:
            continue
        splits[split] = [train_mod.MacroSegmentSample(**item) for item in items]
    if not splits:
        raise ValueError(f"Cache {cache_path} has no train/val split")
    return splits


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def _prf_from_masks(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> dict[str, float]:
    pred_b = (pred > 0) & (valid > 0)
    target_b = (target > 0) & (valid > 0)
    tp = float((pred_b & target_b).sum().item())
    fp = float((pred_b & (~target_b)).sum().item())
    fn = float(((~pred_b) & target_b).sum().item())
    pred_count = tp + fp
    target_count = tp + fn
    precision = _safe_div(tp, pred_count)
    recall = _safe_div(tp, target_count)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    if target_count == 0 and pred_count == 0:
        precision = recall = f1 = 1.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pred_count": pred_count,
        "target_count": target_count,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "target_positive": float(target_count > 0),
    }


def _topk_like_target(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    pred = torch.zeros_like(target, dtype=torch.bool)
    for batch_idx in range(target.shape[0]):
        valid_idx = torch.nonzero(valid[batch_idx] > 0, as_tuple=False).squeeze(-1)
        if valid_idx.numel() == 0:
            continue
        target_count = int(((target[batch_idx] > 0) & (valid[batch_idx] > 0)).sum().item())
        if target_count <= 0:
            continue
        k = min(target_count, int(valid_idx.numel()))
        local = torch.topk(logits[batch_idx, valid_idx], k=k).indices
        pred[batch_idx, valid_idx[local]] = True
    return pred


def _topk_budget(logits: torch.Tensor, valid: torch.Tensor, budget: int) -> torch.Tensor:
    pred = torch.zeros_like(valid, dtype=torch.bool)
    if budget <= 0:
        return pred
    for batch_idx in range(valid.shape[0]):
        valid_idx = torch.nonzero(valid[batch_idx] > 0, as_tuple=False).squeeze(-1)
        if valid_idx.numel() == 0:
            continue
        k = min(int(budget), int(valid_idx.numel()))
        local = torch.topk(logits[batch_idx, valid_idx], k=k).indices
        pred[batch_idx, valid_idx[local]] = True
    return pred


def _append_metrics(
    store: dict[str, list[dict[str, float]]],
    key: str,
    metrics: dict[str, float],
) -> None:
    store.setdefault(key, []).append(metrics)


def _summarize_metric_list(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    keys = sorted({key for item in items for key in item})
    return {key: float(np.mean([float(item.get(key, 0.0)) for item in items])) for key in keys}


def _decode_batch(
    model: mod.MacroDreamerEditModel,
    tensors: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
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
    prior_mu, prior_logvar = model.prior_stats(
        global_latent,
        tensors["global_summary"],
        tensors["horizon_k"],
    )
    path_latent = model.sample_path_latent(prior_mu, prior_logvar, deterministic=True)
    predicted_next_global = model.predict_next_global(global_latent, path_latent, tensors["horizon_k"])
    change_logits, type_logits = model.decode_edit(
        site_latent=site_latent,
        patch_latent=patch_latent,
        predicted_next_global=predicted_next_global,
        path_latent=path_latent,
        horizon_k=tensors["horizon_k"],
        current_types=tensors["current_types"],
    )
    proposal_logits = model.decode_proposal(
        site_latent=site_latent,
        patch_latent=patch_latent,
        predicted_next_global=predicted_next_global,
        path_latent=path_latent,
        horizon_k=tensors["horizon_k"],
        current_types=tensors["current_types"],
    )
    action_support_logits = model.decode_action_support(
        site_latent=site_latent,
        patch_latent=patch_latent,
        predicted_next_global=predicted_next_global,
        path_latent=path_latent,
        horizon_k=tensors["horizon_k"],
        current_types=tensors["current_types"],
    )
    action_source_logits = model.decode_action_source_support(
        site_latent=site_latent,
        patch_latent=patch_latent,
        predicted_next_global=predicted_next_global,
        path_latent=path_latent,
        horizon_k=tensors["horizon_k"],
        current_types=tensors["current_types"],
    )
    action_destination_logits = model.decode_action_destination_support(
        site_latent=site_latent,
        patch_latent=patch_latent,
        predicted_next_global=predicted_next_global,
        path_latent=path_latent,
        horizon_k=tensors["horizon_k"],
        current_types=tensors["current_types"],
    )
    action_endpoint_logits = train_mod.combine_action_endpoint_logits(action_source_logits, action_destination_logits)
    sequence_logits, sequence_mask = long_eval._sequence_rollout_projection_logits(
        model=model,
        site_latent=site_latent,
        patch_latent=patch_latent,
        predicted_next_global=predicted_next_global,
        path_latent=path_latent,
        horizon_k=tensors["horizon_k"],
        current_types=tensors["current_types"],
        change_logits=change_logits,
        proposal_logits=proposal_logits,
        action_support_logits=action_support_logits,
        action_source_logits=action_source_logits,
        action_destination_logits=action_destination_logits,
        candidate_positions=tensors["candidate_positions"],
        candidate_mask=tensors["candidate_mask"],
        box_dims=tensors["box_dims"],
        nn1_offsets=np.asarray(train_mod.BCC_NN1_OFFSETS, dtype=np.int64),
        anchor_source="action_endpoint",
        anchor_budget=32,
        destinations_per_anchor=4,
        blend_alpha=0.5,
        support_blend_alpha=0.0,
        multiobjective_type_weight=0.15,
        multiobjective_order_weight=0.10,
        gate_to_rollout=False,
    )
    vacancy_logits, vacancy_type_logits, vacancy_mask = long_eval._action_edge_pair_vacancy_displacement_logits(
        model=model,
        site_latent=site_latent,
        patch_latent=patch_latent,
        predicted_next_global=predicted_next_global,
        path_latent=path_latent,
        horizon_k=tensors["horizon_k"],
        current_types=tensors["current_types"],
        raw_type_logits=type_logits,
        change_logits=change_logits,
        proposal_logits=proposal_logits,
        action_support_logits=action_support_logits,
        action_source_logits=action_source_logits,
        action_destination_logits=action_destination_logits,
        candidate_positions=tensors["candidate_positions"],
        candidate_mask=tensors["candidate_mask"],
        box_dims=tensors["box_dims"],
        nn1_offsets=np.asarray(train_mod.BCC_NN1_OFFSETS, dtype=np.int64),
        anchor_source="action_endpoint",
        anchor_budget=32,
        destinations_per_anchor=4,
        blend_alpha=0.5,
        support_blend_alpha=0.0,
        multiobjective_type_weight=0.15,
        multiobjective_order_weight=0.10,
    )
    return {
        "change": change_logits,
        "proposal": proposal_logits,
        "action_support": action_support_logits,
        "action_source": action_source_logits,
        "action_destination": action_destination_logits,
        "action_endpoint": action_endpoint_logits,
        "sequence_rollout": sequence_logits,
        "two_stage_vacancy": vacancy_logits,
        "sequence_rollout_mask": sequence_mask.float(),
        "two_stage_vacancy_mask": vacancy_mask.float(),
        "two_stage_vacancy_type_logits": vacancy_type_logits,
    }


def _samplewise_metrics(
    *,
    sources: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    valid: torch.Tensor,
    budgets: tuple[int, ...],
) -> dict[str, list[dict[str, float]]]:
    metric_store: dict[str, list[dict[str, float]]] = {}
    logit_sources = {key: value for key, value in sources.items() if not key.endswith("_mask") and key != "two_stage_vacancy_type_logits"}
    mask_sources = {key: value for key, value in sources.items() if key.endswith("_mask")}
    for target_name, target in targets.items():
        for source_name, logits in logit_sources.items():
            pred = _topk_like_target(logits, target, valid)
            for batch_idx in range(target.shape[0]):
                _append_metrics(
                    metric_store,
                    f"{source_name}_top_target_count__vs__{target_name}",
                    _prf_from_masks(pred[batch_idx], target[batch_idx], valid[batch_idx]),
                )
            for budget in budgets:
                pred_budget = _topk_budget(logits, valid, budget)
                for batch_idx in range(target.shape[0]):
                    _append_metrics(
                        metric_store,
                        f"{source_name}_top{budget}__vs__{target_name}",
                        _prf_from_masks(pred_budget[batch_idx], target[batch_idx], valid[batch_idx]),
                    )
        for source_name, mask in mask_sources.items():
            for batch_idx in range(target.shape[0]):
                _append_metrics(
                    metric_store,
                    f"{source_name}__vs__{target_name}",
                    _prf_from_masks(mask[batch_idx], target[batch_idx], valid[batch_idx]),
                )
    return metric_store


def _merge_metric_stores(
    dest: dict[str, list[dict[str, float]]],
    src: dict[str, list[dict[str, float]]],
) -> None:
    for key, values in src.items():
        dest.setdefault(key, []).extend(values)


def _tensor_mean_count(mask: torch.Tensor, valid: torch.Tensor) -> float:
    counts = ((mask > 0) & (valid > 0)).sum(dim=1).detach().cpu().numpy()
    return float(np.mean(counts)) if counts.size else 0.0


def _run_split(
    *,
    model: mod.MacroDreamerEditModel,
    samples: list[train_mod.MacroSegmentSample],
    device: str,
    batch_size: int,
    budgets: tuple[int, ...],
) -> dict[str, Any]:
    metric_store: dict[str, list[dict[str, float]]] = {}
    target_counts: dict[str, list[float]] = {}
    pair_stats: dict[str, list[float]] = {
        "edge_pair_count": [],
        "edge_pair_support_count": [],
        "sequence_step_count": [],
        "rollout_changed_count": [],
    }
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch = samples[start : start + batch_size]
            tensors = train_mod._batch_to_device(batch, device)
            decoded = _decode_batch(model, tensors)
            valid = tensors["candidate_mask"]
            targets = {
                "changed": tensors["changed_mask"].float(),
                "vacancy_displacement": train_mod._vacancy_displacement_target_from_tensors(tensors),
                "teacher_rollout": tensors["teacher_action_rollout_changed_mask"].float(),
                "action_endpoint": train_mod._proposal_target_from_tensors(tensors, "action_endpoint"),
                "touched": tensors["teacher_touched_mask"].float(),
            }
            _merge_metric_stores(
                metric_store,
                _samplewise_metrics(sources=decoded, targets=targets, valid=valid, budgets=budgets),
            )
            for name, target in targets.items():
                target_counts.setdefault(name, []).extend(
                    ((target > 0) & (valid > 0)).sum(dim=1).detach().cpu().numpy().astype(float).tolist()
                )
            pair_stats["edge_pair_count"].extend(
                (tensors["teacher_action_edge_pair_mask"] > 0).sum(dim=1).detach().cpu().numpy().astype(float).tolist()
            )
            pair_stats["edge_pair_support_count"].extend(
                (
                    (tensors["teacher_action_edge_pair_mask"] > 0)
                    & (tensors["teacher_action_edge_pair_support_mask"] > 0)
                )
                .sum(dim=1)
                .detach()
                .cpu()
                .numpy()
                .astype(float)
                .tolist()
            )
            pair_stats["sequence_step_count"].extend(
                (tensors["teacher_action_sequence_mask"] > 0).sum(dim=1).detach().cpu().numpy().astype(float).tolist()
            )
            pair_stats["rollout_changed_count"].extend(
                (
                    (tensors["teacher_action_rollout_changed_mask"] > 0)
                    & (tensors["candidate_mask"] > 0)
                )
                .sum(dim=1)
                .detach()
                .cpu()
                .numpy()
                .astype(float)
                .tolist()
            )
    summarized = {key: _summarize_metric_list(values) for key, values in sorted(metric_store.items())}
    target_summary = {
        key: {
            "mean": float(np.mean(values)) if values else 0.0,
            "min": float(np.min(values)) if values else 0.0,
            "max": float(np.max(values)) if values else 0.0,
        }
        for key, values in sorted(target_counts.items())
    }
    pair_summary = {
        key: float(np.mean(values)) if values else 0.0
        for key, values in sorted(pair_stats.items())
    }
    best_by_target: dict[str, list[dict[str, float | str]]] = {}
    for metric_name, metrics in summarized.items():
        if "__vs__" not in metric_name:
            continue
        source_name, target_name = metric_name.split("__vs__", 1)
        entry = {"source": source_name, **metrics}
        best_by_target.setdefault(target_name, []).append(entry)
    for target_name, entries in best_by_target.items():
        entries.sort(key=lambda item: (float(item.get("f1", 0.0)), float(item.get("recall", 0.0))), reverse=True)
        best_by_target[target_name] = entries[:8]
    return {
        "sample_count": len(samples),
        "target_counts": target_summary,
        "pair_stats": pair_summary,
        "metrics": summarized,
        "best_by_target": best_by_target,
        "note": "reward_improving_support is not a site-level cache target; closed-loop delta_e frontier remains the proxy.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only v97 sequence support precision diagnostic.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--budgets", type=str, default="32,64,96,128")
    args = parser.parse_args()

    budgets = tuple(int(item) for item in args.budgets.split(",") if item.strip())
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = long_eval._build_model(ckpt, args.device)
    splits = _load_split_samples(args.cache)
    summary: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "cache": str(args.cache),
        "budgets": list(budgets),
        "splits": {},
    }
    for split_name, samples in splits.items():
        summary["splits"][split_name] = _run_split(
            model=model,
            samples=samples,
            device=args.device,
            batch_size=max(int(args.batch_size), 1),
            budgets=budgets,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"wrote": str(args.out), "splits": list(summary["splits"])}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
