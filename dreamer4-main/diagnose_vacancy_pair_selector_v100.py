from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train_dreamer_macro_edit as train_mod


def _load_split_samples(cache_path: Path) -> tuple[dict[str, list[train_mod.MacroSegmentSample]], dict[str, Any]]:
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    if isinstance(payload, list):
        return {"all": [train_mod.MacroSegmentSample(**item) for item in payload]}, {}
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
    signature = payload.get("signature", {})
    return splits, signature if isinstance(signature, dict) else {}


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def _prf(pred: np.ndarray, target: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    pred_b = (np.asarray(pred) > 0.5) & (np.asarray(valid) > 0.5)
    target_b = (np.asarray(target) > 0.5) & (np.asarray(valid) > 0.5)
    tp = float(np.logical_and(pred_b, target_b).sum())
    fp = float(np.logical_and(pred_b, np.logical_not(target_b)).sum())
    fn = float(np.logical_and(np.logical_not(pred_b), target_b).sum())
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    if (tp + fp) == 0.0 and (tp + fn) == 0.0:
        precision = recall = f1 = 1.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pred_count": float(tp + fp),
        "target_count": float(tp + fn),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def _endpoint_mask(pair_indices: np.ndarray, pair_mask: np.ndarray, size: int) -> np.ndarray:
    mask = np.zeros((size,), dtype=np.float32)
    for pair, keep in zip(np.asarray(pair_indices, dtype=np.int64), np.asarray(pair_mask, dtype=np.float32)):
        if float(keep) <= 0.0:
            continue
        if pair.shape[0] != 2:
            continue
        source_idx = int(pair[0])
        dest_idx = int(pair[1])
        if 0 <= source_idx < size:
            mask[source_idx] = 1.0
        if 0 <= dest_idx < size:
            mask[dest_idx] = 1.0
    return mask


def _pair_set(pair_indices: np.ndarray, pair_mask: np.ndarray) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for pair, keep in zip(np.asarray(pair_indices, dtype=np.int64), np.asarray(pair_mask, dtype=np.float32)):
        if float(keep) <= 0.0 or pair.shape[0] != 2:
            continue
        source_idx = int(pair[0])
        dest_idx = int(pair[1])
        if source_idx >= 0 and dest_idx >= 0:
            pairs.add((source_idx, dest_idx))
    return pairs


def _vacancy_displacement_mask(sample: train_mod.MacroSegmentSample) -> np.ndarray:
    current = np.asarray(sample.current_types, dtype=np.int64)
    target = np.asarray(sample.target_types, dtype=np.int64)
    valid = np.asarray(sample.candidate_mask > 0, dtype=bool)
    source = (current == train_mod.V_TYPE) & np.isin(target, [train_mod.FE_TYPE, train_mod.CU_TYPE])
    dest = np.isin(current, [train_mod.FE_TYPE, train_mod.CU_TYPE]) & (target == train_mod.V_TYPE)
    return np.asarray((source | dest) & valid, dtype=np.float32)


def _vacancy_pairs_for_sample(sample: train_mod.MacroSegmentSample) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pair_indices = getattr(sample, "teacher_vacancy_pair_indices", None)
    pair_mask = getattr(sample, "teacher_vacancy_pair_mask", None)
    pair_moving_type = getattr(sample, "teacher_vacancy_pair_moving_type", None)
    pair_order = getattr(sample, "teacher_vacancy_pair_order", None)
    if pair_indices is not None and pair_mask is not None and float(np.asarray(pair_mask).sum()) > 0.0:
        if pair_moving_type is None:
            pair_moving_type = np.full_like(sample.candidate_mask, -1, dtype=np.int64)
        if pair_order is None:
            pair_order = np.zeros_like(sample.candidate_mask, dtype=np.float32)
        return (
            np.asarray(pair_indices, dtype=np.int64),
            np.asarray(pair_mask, dtype=np.float32),
            np.asarray(pair_moving_type, dtype=np.int64),
            np.asarray(pair_order, dtype=np.float32),
        )

    return train_mod._teacher_vacancy_displacement_pair_targets_from_sequence(
        candidate_mask=np.asarray(sample.candidate_mask, dtype=np.float32),
        current_types=np.asarray(sample.current_types, dtype=np.int64),
        target_types=np.asarray(sample.target_types, dtype=np.int64),
        action_sequence_indices=np.asarray(
            getattr(sample, "teacher_action_sequence_indices", np.full((sample.candidate_mask.shape[0], 2), -1)),
            dtype=np.int64,
        ),
        action_sequence_mask=np.asarray(
            getattr(sample, "teacher_action_sequence_mask", np.zeros_like(sample.candidate_mask)),
            dtype=np.float32,
        ),
        action_sequence_moving_type=np.asarray(
            getattr(sample, "teacher_action_sequence_moving_type", np.full_like(sample.candidate_mask, -1)),
            dtype=np.int64,
        ),
        action_sequence_order=np.asarray(
            getattr(sample, "teacher_action_sequence_order", np.zeros_like(sample.candidate_mask)),
            dtype=np.float32,
        ),
    )[:4]


def _sample_row(sample: train_mod.MacroSegmentSample, split: str, idx: int) -> dict[str, float | int | str]:
    valid = np.asarray(sample.candidate_mask, dtype=np.float32)
    changed = np.asarray(sample.changed_mask, dtype=np.float32)
    vacancy_target = _vacancy_displacement_mask(sample)
    pair_indices, pair_mask, pair_moving_type, pair_order = _vacancy_pairs_for_sample(sample)
    pair_endpoint = _endpoint_mask(pair_indices, pair_mask, int(valid.shape[0]))

    edge_indices = np.asarray(
        getattr(sample, "teacher_action_edge_pair_indices", np.full((sample.candidate_mask.shape[0], 2), -1)),
        dtype=np.int64,
    )
    edge_mask = np.asarray(
        getattr(sample, "teacher_action_edge_pair_mask", np.zeros_like(sample.candidate_mask)),
        dtype=np.float32,
    )
    sequence_indices = np.asarray(
        getattr(sample, "teacher_action_sequence_indices", edge_indices),
        dtype=np.int64,
    )
    sequence_mask = np.asarray(
        getattr(sample, "teacher_action_sequence_mask", edge_mask),
        dtype=np.float32,
    )
    edge_endpoint = _endpoint_mask(edge_indices, edge_mask, int(valid.shape[0]))
    sequence_endpoint = _endpoint_mask(sequence_indices, sequence_mask, int(valid.shape[0]))

    terminal_pairs = _pair_set(pair_indices, pair_mask)
    action_pairs = _pair_set(edge_indices, edge_mask)
    exact_hits = len(terminal_pairs & action_pairs)
    terminal_count = len(terminal_pairs)
    moving_types = [
        int(t)
        for t, keep in zip(np.asarray(pair_moving_type, dtype=np.int64), np.asarray(pair_mask, dtype=np.float32))
        if float(keep) > 0.0 and int(t) >= 0
    ]
    orders = [
        float(o)
        for o, keep in zip(np.asarray(pair_order, dtype=np.float32), np.asarray(pair_mask, dtype=np.float32))
        if float(keep) > 0.0
    ]

    pair_vs_changed = _prf(pair_endpoint, changed, valid)
    pair_vs_vacancy = _prf(pair_endpoint, vacancy_target, valid)
    sequence_vs_pair = _prf(sequence_endpoint, pair_endpoint, valid)
    edge_vs_pair = _prf(edge_endpoint, pair_endpoint, valid)
    return {
        "split": split,
        "sample_idx": int(idx),
        "horizon_k": int(sample.horizon_k),
        "candidate_count": float(valid.sum()),
        "changed_count": float(changed.sum()),
        "vacancy_displacement_count": float(vacancy_target.sum()),
        "terminal_vacancy_pair_count": float(terminal_count),
        "terminal_vacancy_pair_endpoint_count": float(pair_endpoint.sum()),
        "terminal_pair_exact_action_pair_recall": _safe_div(float(exact_hits), float(terminal_count)),
        "terminal_pair_fe_fraction": _safe_div(float(sum(t == train_mod.FE_TYPE for t in moving_types)), float(len(moving_types))),
        "terminal_pair_cu_fraction": _safe_div(float(sum(t == train_mod.CU_TYPE for t in moving_types)), float(len(moving_types))),
        "terminal_pair_order_mean": float(np.mean(orders)) if orders else 0.0,
        "pair_vs_changed_precision": pair_vs_changed["precision"],
        "pair_vs_changed_recall": pair_vs_changed["recall"],
        "pair_vs_changed_f1": pair_vs_changed["f1"],
        "pair_vs_vacancy_precision": pair_vs_vacancy["precision"],
        "pair_vs_vacancy_recall": pair_vs_vacancy["recall"],
        "pair_vs_vacancy_f1": pair_vs_vacancy["f1"],
        "sequence_endpoint_vs_pair_precision": sequence_vs_pair["precision"],
        "sequence_endpoint_vs_pair_recall": sequence_vs_pair["recall"],
        "sequence_endpoint_vs_pair_f1": sequence_vs_pair["f1"],
        "edge_endpoint_vs_pair_precision": edge_vs_pair["precision"],
        "edge_endpoint_vs_pair_recall": edge_vs_pair["recall"],
        "edge_endpoint_vs_pair_f1": edge_vs_pair["f1"],
    }


def _summarize(rows: list[dict[str, float | int | str]]) -> dict[str, float]:
    if not rows:
        return {}
    numeric_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if key != "sample_idx" and isinstance(value, (int, float))
        }
    )
    return {
        key: float(np.mean([float(row.get(key, 0.0)) for row in rows]))
        for key in numeric_keys
    }


def _interpret(splits: dict[str, dict[str, float]]) -> list[str]:
    notes: list[str] = []
    val = splits.get("val") or next(iter(splits.values()), {})
    pair_f1 = float(val.get("pair_vs_vacancy_f1", 0.0))
    exact_recall = float(val.get("terminal_pair_exact_action_pair_recall", 0.0))
    pair_count = float(val.get("terminal_vacancy_pair_count", 0.0))
    if pair_f1 >= 0.95:
        notes.append(
            "terminal vacancy-displacement pair target reconstructs the final vacancy-diff endpoints with high F1."
        )
    if exact_recall < 0.5 and pair_count > 0.0:
        notes.append(
            "terminal vacancy-displacement pairs are mostly not identical to single micro action edges; v100 needs a terminal pair selector, not another NN1 action-edge retune."
        )
    notes.append(
        "next trainable step should supervise pair-level vacancy source, atom destination, moving type and order/count, then combine terminal-support score with energy pair score in planner."
    )
    return notes


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only v100 terminal vacancy-pair selector target diagnostic.")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-rows", type=int, default=32)
    args = parser.parse_args()

    cache_path = Path(args.cache)
    splits, signature = _load_split_samples(cache_path)
    rows_by_split: dict[str, list[dict[str, float | int | str]]] = {}
    summaries: dict[str, dict[str, float]] = {}
    for split, samples in splits.items():
        rows = [_sample_row(sample, split, idx) for idx, sample in enumerate(samples)]
        rows_by_split[split] = rows
        summaries[split] = _summarize(rows)

    result = {
        "diagnostic": "v100_terminal_vacancy_pair_selector_target",
        "cache": str(cache_path),
        "signature": signature,
        "splits": summaries,
        "interpretation": _interpret(summaries),
        "rows_preview": {
            split: rows[: max(int(args.max_rows), 0)]
            for split, rows in rows_by_split.items()
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
