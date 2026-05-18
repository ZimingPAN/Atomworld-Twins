from __future__ import annotations

import argparse
import copy
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

import eval_macro_closed_loop_rollout as closed_loop
import eval_macro_long_trajectory as long_eval
import train_dreamer_macro_edit as mod


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast model-only closed-loop visual rollout")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--rollout_segments", type=int, default=5000)
    parser.add_argument("--max_episode_steps_override", type=int, default=None)
    parser.add_argument("--planner_segment_ks", type=int, nargs="*", default=None)
    parser.add_argument("--constraint_mode", default="full", choices=["full", "no_change"])
    parser.add_argument("--duration_source", default="model", choices=["model", "baseline", "blend"])
    parser.add_argument("--duration_blend_alpha", type=float, default=1.0)
    parser.add_argument("--duration_log_offset", type=float, default=0.0)
    parser.add_argument("--planner_score_mode", default="energy_per_tau", choices=["energy_per_tau", "energy_per_sqrt_tau", "energy"])
    parser.add_argument("--planner_tau_residual_penalty", type=float, default=0.0)
    parser.add_argument("--planner_k_penalty_power", type=float, default=0.0)
    parser.add_argument("--raw_changed_budget_multiplier", type=float, default=2.0)
    parser.add_argument("--global_candidate_cu_fraction", type=float, default=0.5)
    parser.add_argument("--min_projected_changed_sites", type=int, default=2)
    parser.add_argument("--output", required=True)
    parser.add_argument("--save_snapshots", action="store_true")
    parser.add_argument("--snapshot_every", type=int, default=500)
    parser.add_argument("--snapshot_max_cu", type=int, default=2500)
    parser.add_argument("--save_edit_trace", action="store_true")
    parser.add_argument("--progress_every", type=int, default=100)
    return parser.parse_args()


def _pair_counts(edit_trace: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in edit_trace:
        for cur_type, selected_type in zip(row.get("current_types", []), row.get("selected_types", [])):
            counts[f"{int(cur_type)}->{int(selected_type)}"] += 1
    return dict(sorted(counts.items()))


def _cu_edit_count(pair_counts: dict[str, int]) -> int:
    total = 0
    for key, value in pair_counts.items():
        cur_type, selected_type = key.split("->")
        if int(cur_type) == int(mod.CU_TYPE) or int(selected_type) == int(mod.CU_TYPE):
            total += int(value)
    return total


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

    candidate_mode, edit_mode, effective_duration_source = closed_loop._constraint_settings(args)
    model = None
    if args.constraint_mode != "no_change":
        model = long_eval._build_model(ckpt, args.device)
    max_seed_vacancies = int(ckpt_args["max_seed_vacancies"])
    max_candidate_sites = int(ckpt_args["max_candidate_sites"])

    env_cfg = long_eval._build_env_cfg(ckpt_args, max_episode_steps_override=args.max_episode_steps_override)
    base_env = mod.MacroKMCEnv(env_cfg)
    base_env.reset()
    model_env = copy.deepcopy(base_env)
    initial_vac, initial_cu = closed_loop._environment_state_arrays(model_env)
    initial_vacancy_count = int(initial_vac.shape[0])
    initial_cu_count = int(initial_cu.shape[0])

    segments: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    edit_trace: list[dict[str, Any]] = []
    chosen_ks: list[int] = []
    pred_tau_exp: list[float] = []
    pred_reward_sum: list[float] = []
    model_delta_e: list[float] = []
    reachability_violations: list[float] = []
    inventory_delta_l1: list[float] = []
    prediction_wall_times: list[float] = []
    apply_wall_times: list[float] = []
    stop_reason = "completed"
    stop_segment: dict[str, Any] | None = None

    if args.save_snapshots:
        snapshots.append(
            {
                "index": 0,
                "model": closed_loop._state_snapshot(model_env, max_cu=args.snapshot_max_cu, rng=snapshot_rng),
            }
        )

    with torch.no_grad():
        for segment_idx in range(int(args.rollout_segments)):
            predict_t0 = time.perf_counter()
            if args.constraint_mode == "no_change":
                candidates = [closed_loop._no_change_candidate(model_env, int(horizon_choices[0]), effective_duration_source)]
            else:
                assert model is not None
                candidates = [
                    item
                    for item in (
                        closed_loop._predict_closed_loop_candidate(
                            model=model,
                            duration_model=None,
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
                            rng=rng,
                            global_candidate_cu_fraction=args.global_candidate_cu_fraction,
                            reward_prediction_source=reward_prediction_source,
                        )
                        for item_k in horizon_choices
                    )
                    if item is not None
                ]
            selected = closed_loop._choose_closed_loop_candidate(
                candidates,
                min_projected_changed_sites=0 if args.constraint_mode == "no_change" else int(args.min_projected_changed_sites),
            )
            prediction_wall_times.append(float(time.perf_counter() - predict_t0))
            if selected is None:
                stop_reason = "no_legal_closed_loop_candidate"
                stop_segment = {"index": segment_idx, "planner_candidates": closed_loop._json_safe_candidates(candidates)}
                break

            apply_t0 = time.perf_counter()
            if args.constraint_mode == "no_change":
                applied = closed_loop._apply_no_change_to_env(model_env, selected)
            else:
                applied = closed_loop._apply_candidate_to_env(
                    model_env,
                    selected,
                    predicted_tau=float(selected["predicted_expected_tau"]),
                )
            apply_wall_times.append(float(time.perf_counter() - apply_t0))
            if args.save_edit_trace:
                edit_trace.append(
                    closed_loop._candidate_edit_trace_row(
                        segment_idx=segment_idx,
                        candidate=selected,
                        applied=applied,
                    )
                )

            chosen_ks.append(int(selected["segment_k"]))
            pred_tau_exp.append(float(selected["predicted_expected_tau"]))
            pred_reward_sum.append(float(selected["predicted_reward_sum"]))
            model_delta_e.append(float(applied["model_delta_e"]))
            reachability_violations.append(float(selected["reachability_violation"]))
            inventory_delta_l1.append(float(selected.get("candidate_inventory_delta_l1", 0.0)))

            vac, cu = closed_loop._environment_state_arrays(model_env)
            row = {
                "index": int(segment_idx),
                "segment_k": int(selected["segment_k"]),
                "candidate_mode": str(selected["candidate_mode"]),
                "edit_mode": str(selected["edit_mode"]),
                "selection_score": float(selected["selection_score"]),
                "predicted_reward_sum": float(selected["predicted_reward_sum"]),
                "predicted_delta_e": float(selected["predicted_delta_e"]),
                "model_applied_delta_e": float(applied["model_delta_e"]),
                "predicted_expected_tau": float(selected["predicted_expected_tau"]),
                "model_expected_tau": float(selected["model_expected_tau"]),
                "baseline_expected_tau": float(selected["baseline_expected_tau"]),
                "reachability_violation": float(selected["reachability_violation"]),
                "candidate_inventory_delta_l1": float(selected.get("candidate_inventory_delta_l1", 0.0)),
                "projected_changed_count": float(selected["projected_changed_count"]),
                "applied_changed_count": float(applied["applied_changed_count"]),
                "cu_total": int(cu.shape[0]),
                "vacancy_total": int(vac.shape[0]),
                "inventory_violation_l1": float(abs(int(cu.shape[0]) - initial_cu_count) + abs(int(vac.shape[0]) - initial_vacancy_count)),
                "prediction_wall_time_sec": float(prediction_wall_times[-1]),
                "apply_wall_time_sec": float(apply_wall_times[-1]),
            }
            segments.append(row)

            if args.save_snapshots and (
                (segment_idx + 1) % max(int(args.snapshot_every), 1) == 0
                or segment_idx + 1 == int(args.rollout_segments)
            ):
                snapshots.append(
                    {
                        "index": int(segment_idx + 1),
                        "model": closed_loop._state_snapshot(model_env, max_cu=args.snapshot_max_cu, rng=snapshot_rng),
                    }
                )

            if args.progress_every > 0 and (segment_idx + 1) % int(args.progress_every) == 0:
                pair_counts = _pair_counts(edit_trace)
                print(
                    json.dumps(
                        {
                            "model_only_progress": {
                                "segments": int(segment_idx + 1),
                                "chosen_k_histogram": closed_loop._histogram_int(chosen_ks),
                                "applied_edits": int(round(sum(row["applied_changed_count"] for row in segments))),
                                "cu_edits": int(_cu_edit_count(pair_counts)),
                                "pair_counts": pair_counts,
                            }
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    completed = int(len(segments))
    pair_counts = _pair_counts(edit_trace)
    summary = {
        "mode": "closed_loop_model_only_visual",
        "checkpoint": str(checkpoint_path),
        "seed": int(args.seed),
        "constraint_mode": args.constraint_mode,
        "candidate_mode": candidate_mode,
        "edit_mode": edit_mode,
        "duration_source": effective_duration_source,
        "segment_ks": horizon_choices,
        "requested_rollout_segments": int(args.rollout_segments),
        "completed_rollout_segments": completed,
        "stop_reason": stop_reason,
        "stop_segment": stop_segment,
        "chosen_k_histogram": closed_loop._histogram_int(chosen_ks),
        "type_pair_counts": pair_counts,
        "cu_edit_count": int(_cu_edit_count(pair_counts)),
        "cumulative": {
            "predicted_expected_time_final": float(np.sum(pred_tau_exp)) if pred_tau_exp else 0.0,
            "predicted_reward_sum_final": float(np.sum(pred_reward_sum)) if pred_reward_sum else 0.0,
            "model_applied_delta_e_final": float(np.sum(model_delta_e)) if model_delta_e else 0.0,
            "applied_sparse_edits": int(round(sum(row["applied_changed_count"] for row in segments))),
            "mean_prediction_wall_time_sec": closed_loop._mean(prediction_wall_times),
            "mean_apply_wall_time_sec": closed_loop._mean(apply_wall_times),
            "reachability_violation_rate_mean": closed_loop._mean(reachability_violations),
            "candidate_inventory_delta_l1_mean": closed_loop._mean(inventory_delta_l1),
        },
        "segment_preview": segments[:5],
        "segments": segments,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary["snapshot_path"] = None
    if args.save_snapshots:
        snapshot_path = output_path.with_suffix(".snapshots.json")
        snapshot_path.write_text(json.dumps({"snapshots": snapshots}, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["snapshot_path"] = str(snapshot_path)
    summary["edit_trace_path"] = None
    if args.save_edit_trace:
        edit_trace_path = output_path.with_suffix(".edit_trace.json")
        edit_trace_path.write_text(json.dumps({"edit_trace": edit_trace}, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["edit_trace_path"] = str(edit_trace_path)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print("AtomWorld-Mirror Model-only Closed-loop Visual Rollout")
    print(f"completed={completed}/{args.rollout_segments}, stop_reason={stop_reason}, segment_ks={horizon_choices}")
    print(f"type_pair_counts={pair_counts}, cu_edit_count={_cu_edit_count(pair_counts)}")
    print(f"Saved summary to {output_path}")


if __name__ == "__main__":
    main()
