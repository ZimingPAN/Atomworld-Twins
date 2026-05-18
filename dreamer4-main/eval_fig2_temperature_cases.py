#!/usr/bin/env python3
"""Evaluate Fig.2 temperature cases for AtomWorld-Mirror.

The script keeps the current Fig.2 protocol narrow: controlled planner-selected
macro-step validation plus a teacher-forced long trajectory. It only changes the
teacher environment temperature for each case.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

import eval_macro_long_trajectory as long_eval
import train_dreamer_macro_edit as mod


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temperatures", type=float, nargs="+", default=[663.0, 693.0, 733.0, 773.0])
    parser.add_argument("--temperature_codes", type=str, nargs="+", default=["1", "2", "3", "4"])
    parser.add_argument("--samples_per_temperature", type=int, default=96)
    parser.add_argument("--long_segments", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cu_density", type=float, default=None)
    parser.add_argument("--max_candidate_sites", type=int, default=None)
    parser.add_argument("--max_episode_steps_override", type=int, default=5000)
    parser.add_argument("--planner_segment_ks", type=int, nargs="+", default=None)
    parser.add_argument("--min_projected_changed_sites", type=int, default=2)
    parser.add_argument("--planner_score_mode", type=str, default="energy_per_tau",
                        choices=["energy_per_tau", "energy_per_sqrt_tau", "energy"])
    parser.add_argument("--duration_source", type=str, default="model", choices=["model", "baseline", "blend"])
    parser.add_argument("--duration_blend_alpha", type=float, default=1.0)
    parser.add_argument("--allow_teacher_noop_segments", action="store_true")
    parser.add_argument("--progress_every", type=int, default=50)
    return parser.parse_args()


def _segment_ks_from_ckpt_args(args: dict[str, object]) -> list[int]:
    if args.get("segment_ks"):
        return sorted({int(k) for k in args["segment_ks"]})
    return [int(args["segment_k"])]


def _summary_horizon_k(segment_ks: list[int]) -> int:
    return max(int(k) for k in segment_ks)


def _compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mae = float(np.mean(np.abs(pred - target))) if pred.size else 0.0
    rmse = float(np.sqrt(np.mean((pred - target) ** 2))) if pred.size else 0.0
    if pred.size > 1 and np.std(pred) > 0 and np.std(target) > 0:
        corr = float(np.corrcoef(pred, target)[0, 1])
    else:
        corr = 0.0
    return {"mae": mae, "rmse": rmse, "corr": corr}


def _compute_log_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    eps = 1e-12
    pred = np.clip(np.asarray(pred, dtype=np.float64), eps, None)
    target = np.clip(np.asarray(target, dtype=np.float64), eps, None)
    log_pred = np.log(pred)
    log_target = np.log(target)
    log_mae = float(np.mean(np.abs(log_pred - log_target))) if pred.size else 0.0
    log_rmse = float(np.sqrt(np.mean((log_pred - log_target) ** 2))) if pred.size else 0.0
    if pred.size > 1 and np.std(log_pred) > 0 and np.std(log_target) > 0:
        log_corr = float(np.corrcoef(log_pred, log_target)[0, 1])
    else:
        log_corr = 0.0
    scale_ratio = float(np.mean(pred / target)) if pred.size else 0.0
    return {
        "log_mae": log_mae,
        "log_rmse": log_rmse,
        "log_corr": log_corr,
        "scale_ratio": scale_ratio,
    }


def _env_cfg(
    ckpt_args: dict[str, object],
    *,
    temperature: float,
    cu_density: float | None,
    max_episode_steps: int | None,
) -> dict[str, object]:
    cfg = long_eval._build_env_cfg(ckpt_args, max_episode_steps_override=max_episode_steps)
    cfg["temperature"] = float(temperature)
    if cu_density is not None:
        cfg["cu_density"] = float(cu_density)
    return cfg


def _paired_rows(
    *,
    model: mod.MacroDreamerEditModel,
    samples: list[mod.MacroSegmentSample],
    batch_size: int,
    device: str,
    reward_prediction_source: str,
    temperature_code: str,
    temperature: float,
) -> tuple[list[dict[str, float | int | str]], dict[str, object]]:
    loader = mod._build_loader(samples, batch_size=batch_size, shuffle=False)
    rows: list[dict[str, float | int | str]] = []
    pred_tau: list[float] = []
    true_tau: list[float] = []
    sample_index = 0
    with torch.no_grad():
        for batch in loader:
            tensors = mod._batch_to_device(batch, device)
            global_latent = model.encode_global(tensors["start_obs"])
            prior_mu, prior_logvar = model.prior_stats(global_latent, tensors["global_summary"], tensors["horizon_k"])
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
            change_logits, raw_type_logits = model.decode_edit(
                site_latent=site_latent,
                patch_latent=patch_latent,
                predicted_next_global=next_pred,
                path_latent=path_latent,
                horizon_k=tensors["horizon_k"],
                current_types=tensors["current_types"],
            )
            duration_outputs = mod._predict_reward_and_duration_outputs(
                model,
                global_latent,
                next_pred,
                path_latent,
                tensors["global_summary"],
                tensors["horizon_k"],
                patch_latent=patch_latent,
                change_logits=change_logits,
                type_logits=raw_type_logits,
                current_types=tensors["current_types"],
                candidate_mask=tensors["candidate_mask"],
            )
            if reward_prediction_source == "projected":
                projected_types, _, _, _ = mod.project_types_by_inventory(
                    current_types=tensors["current_types"],
                    change_logits=change_logits,
                    type_logits=raw_type_logits,
                    node_mask=tensors["candidate_mask"],
                    positions=tensors["candidate_positions"],
                    box_dims=tensors["box_dims"],
                    horizon_k=tensors["horizon_k"],
                    max_changed_sites=2 * tensors["horizon_k"],
                )
                _, projected_patch_latent = model.encode_patch(
                    positions=tensors["candidate_positions"],
                    nearest_vacancy_offset=tensors["nearest_vacancy_offset"],
                    reach_depth=tensors["reach_depth"],
                    is_start_vacancy=tensors["is_start_vacancy"],
                    type_ids=projected_types,
                    node_mask=tensors["candidate_mask"],
                    global_summary=tensors["global_summary"],
                    box_dims=tensors["box_dims"],
                )
                projected_change_logits, projected_type_logits = mod.projected_edit_logits_from_types(
                    current_types=tensors["current_types"],
                    projected_types=projected_types,
                    candidate_mask=tensors["candidate_mask"],
                )
                duration_outputs = mod._predict_reward_and_duration_outputs(
                    model,
                    global_latent,
                    next_pred,
                    path_latent,
                    tensors["global_summary"],
                    tensors["horizon_k"],
                    patch_latent=projected_patch_latent,
                    change_logits=projected_change_logits,
                    type_logits=projected_type_logits,
                    current_types=tensors["current_types"],
                    candidate_mask=tensors["candidate_mask"],
                )
            batch_pred_tau = torch.exp(duration_outputs["expected_tau_mu"]).cpu().numpy()
            batch_true_tau = tensors["tau_exp"].cpu().numpy()
            pred_tau.extend(batch_pred_tau.tolist())
            true_tau.extend(batch_true_tau.tolist())
            batch_pred_real = torch.exp(duration_outputs["realized_tau_mu"]).cpu().numpy()
            batch_true_real = tensors["tau_real"].cpu().numpy()
            for local_idx in range(len(batch)):
                rows.append(
                    {
                        "sample_index": int(sample_index),
                        "temperature_code": str(temperature_code),
                        "temperature": float(temperature),
                        "segment_k": int(tensors["horizon_k"][local_idx].item()),
                        "traditional_kmc_expected_tau": float(batch_true_tau[local_idx]),
                        "predicted_expected_tau": float(batch_pred_tau[local_idx]),
                        "traditional_kmc_realized_tau": float(batch_true_real[local_idx]),
                        "predicted_realized_tau": float(batch_pred_real[local_idx]),
                        "traditional_reward_sum": float(tensors["reward_sum"][local_idx].item()),
                    }
                )
                sample_index += 1
    pred = np.asarray(pred_tau, dtype=np.float64)
    target = np.asarray(true_tau, dtype=np.float64)
    metrics = {**_compute_metrics(pred, target), **_compute_log_metrics(pred, target)}
    return rows, metrics


def _run_long_case(
    *,
    model: mod.MacroDreamerEditModel,
    ckpt_args: dict[str, object],
    env_cfg: dict[str, object],
    horizon_choices: list[int],
    rollout_segments: int,
    seed: int,
    device: str,
    max_candidate_sites: int,
    min_projected_changed_sites: int,
    duration_source: str,
    duration_blend_alpha: float,
    planner_score_mode: str,
    reward_prediction_source: str,
    allow_teacher_noop_segments: bool,
    progress_every: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    reward_scale = float(ckpt_args.get("reward_scale", 1.0))
    max_seed_vacancies = int(ckpt_args["max_seed_vacancies"])
    planner_enabled = len(horizon_choices) > 1
    env = mod.MacroKMCEnv(copy.deepcopy(env_cfg))
    env.reset()
    pred_tau_exp: list[float] = []
    true_tau_exp: list[float] = []
    pred_reward_sum: list[float] = []
    true_reward_sum: list[float] = []
    chosen_ks: list[int] = []
    segments: list[dict[str, object]] = []
    stop_reason = "completed"
    stop_segment: dict[str, object] | None = None
    skipped_noop = 0
    skipped_terminal = 0
    with torch.no_grad():
        for segment_idx in range(int(rollout_segments)):
            candidates = [
                item
                for item in (
                    long_eval._predict_candidate_for_horizon(
                        model=model,
                        duration_model=None,
                        env=env,
                        horizon_k=int(item_k),
                        max_seed_vacancies=max_seed_vacancies,
                        max_candidate_sites=max_candidate_sites,
                        reward_scale=reward_scale,
                        device=device,
                        duration_source=duration_source,
                        planner_tau_source=duration_source,
                        planner_score_mode=planner_score_mode,
                        duration_blend_alpha=duration_blend_alpha,
                        planner_tau_blend_alpha=duration_blend_alpha,
                        reward_prediction_source=reward_prediction_source,
                    )
                    for item_k in horizon_choices
                )
                if item is not None
            ]
            selected = long_eval._choose_planner_candidate(
                candidates,
                min_projected_changed_sites=int(min_projected_changed_sites) if planner_enabled else 0,
            )
            if selected is None:
                stop_reason = "no_legal_planner_candidate"
                stop_segment = {"index": segment_idx, "planner_candidates": candidates}
                break
            selected_k = int(selected["segment_k"])
            teacher_segment = long_eval._collect_teacher_segment(env, horizon_k=selected_k, rng=rng)
            if teacher_segment is None:
                skipped_terminal += 1
                stop_reason = "teacher_terminal_or_action_missing"
                stop_segment = {"index": segment_idx, "segment_k": selected_k, "selected": selected}
                break
            if bool(teacher_segment.get("is_noop", False)) and not allow_teacher_noop_segments:
                skipped_noop += 1
                stop_reason = "noop_teacher_segment"
                stop_segment = {
                    "index": segment_idx,
                    "segment_k": selected_k,
                    "selected": selected,
                    "traditional_kmc_expected_tau": float(teacher_segment["tau_exp"]),
                    "traditional_changed_site_count": int(teacher_segment.get("changed_site_count", 0)),
                }
                break
            pred_tau_exp.append(float(selected["predicted_expected_tau"]))
            true_tau_exp.append(float(teacher_segment["tau_exp"]))
            pred_reward_sum.append(float(selected["predicted_reward_sum"]))
            true_reward_sum.append(float(teacher_segment["reward_sum"]))
            chosen_ks.append(selected_k)
            segments.append(
                {
                    "index": int(segment_idx),
                    "segment_k": selected_k,
                    "predicted_expected_tau": float(selected["predicted_expected_tau"]),
                    "traditional_kmc_expected_tau": float(teacher_segment["tau_exp"]),
                    "predicted_reward_sum": float(selected["predicted_reward_sum"]),
                    "traditional_kmc_reward_sum": float(teacher_segment["reward_sum"]),
                    "reachability_violation": float(selected["reachability_violation"]),
                    "projected_changed_count": float(selected["projected_changed_count"]),
                    "traditional_changed_site_count": int(teacher_segment.get("changed_site_count", 0)),
                }
            )
            if progress_every > 0 and (segment_idx + 1) % int(progress_every) == 0:
                print(
                    json.dumps(
                        {
                            "long_progress": {
                                "segments": segment_idx + 1,
                                "predicted_tau": float(np.sum(pred_tau_exp)),
                                "traditional_tau": float(np.sum(true_tau_exp)),
                            }
                        }
                    ),
                    flush=True,
                )
    pred_tau = np.asarray(pred_tau_exp, dtype=np.float64)
    true_tau = np.asarray(true_tau_exp, dtype=np.float64)
    pred_reward = np.asarray(pred_reward_sum, dtype=np.float64)
    true_reward = np.asarray(true_reward_sum, dtype=np.float64)
    chosen = np.asarray(chosen_ks, dtype=np.int64)
    chosen_hist = {
        str(int(k)): int(np.sum(chosen == int(k)))
        for k in sorted(set(chosen_ks))
    }
    return {
        "requested_rollout_segments": int(rollout_segments),
        "completed_rollout_segments": int(len(segments)),
        "stop_reason": stop_reason,
        "stop_segment": stop_segment,
        "skipped_noop": int(skipped_noop),
        "skipped_terminal": int(skipped_terminal),
        "chosen_k_histogram": chosen_hist,
        "tau_expected": {**_compute_metrics(pred_tau, true_tau), **_compute_log_metrics(pred_tau, true_tau)}
        if len(segments) > 0 else {},
        "reward_sum": _compute_metrics(pred_reward, true_reward) if len(segments) > 0 else {},
        "cumulative": {
            "predicted_expected_time_final": float(pred_tau.sum()) if len(segments) > 0 else 0.0,
            "traditional_kmc_expected_time_final": float(true_tau.sum()) if len(segments) > 0 else 0.0,
            "expected_time_ratio": float(pred_tau.sum() / true_tau.sum())
            if len(segments) > 0 and true_tau.sum() > 1e-12 else None,
            "predicted_delta_e_final": float(np.sum(pred_reward / reward_scale)) if len(segments) > 0 else 0.0,
            "traditional_kmc_delta_e_final": float(np.sum(true_reward / reward_scale)) if len(segments) > 0 else 0.0,
        },
        "arrays": {
            "predicted_expected_tau_cumsum": np.cumsum(pred_tau).tolist(),
            "traditional_kmc_expected_tau_cumsum": np.cumsum(true_tau).tolist(),
        },
        "segments": segments,
    }


def main() -> None:
    args = parse_args()
    if len(args.temperature_codes) != len(args.temperatures):
        raise ValueError("--temperature_codes must have the same length as --temperatures")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    ckpt_args = ckpt["args"]
    model = long_eval._build_model(ckpt, args.device)
    model.eval()

    segment_ks = sorted({int(k) for k in (args.planner_segment_ks or _segment_ks_from_ckpt_args(ckpt_args))})
    summary_horizon = _summary_horizon_k(segment_ks)
    include_stepwise_path_summary = str(ckpt_args.get("teacher_path_summary_mode", "stepwise")) == "stepwise"
    reward_prediction_source = str(ckpt_args.get("reward_prediction_source", "raw"))
    max_candidate_sites = int(args.max_candidate_sites or ckpt_args["max_candidate_sites"])
    rows: list[dict[str, object]] = []
    for case_idx, (code, temperature) in enumerate(zip(args.temperature_codes, args.temperatures)):
        case_seed = int(args.seed) + case_idx * 1009
        cfg = _env_cfg(
            ckpt_args,
            temperature=float(temperature),
            cu_density=args.cu_density,
            max_episode_steps=int(args.max_episode_steps_override),
        )
        print(
            json.dumps(
                {
                    "case": code,
                    "temperature": float(temperature),
                    "paired_samples": int(args.samples_per_temperature),
                    "long_segments": int(args.long_segments),
                    "seed": int(case_seed),
                }
            ),
            flush=True,
        )
        samples, collection_stats = mod._collect_planner_selected_segments(
            env=mod.MacroKMCEnv(copy.deepcopy(cfg)),
            num_segments=int(args.samples_per_temperature),
            segment_ks=segment_ks,
            planner_model=model,
            planner_device=args.device,
            max_seed_vacancies=int(ckpt_args["max_seed_vacancies"]),
            max_candidate_sites=max_candidate_sites,
            rng=np.random.default_rng(case_seed),
            include_stepwise_path_summary=include_stepwise_path_summary,
            summary_horizon_k=summary_horizon,
            max_segments_per_rollout=int(ckpt_args.get("max_segments_per_rollout", 50)),
            min_projected_changed_sites=int(args.min_projected_changed_sites),
            duration_source=args.duration_source,
            planner_tau_source=args.duration_source,
            planner_score_mode=args.planner_score_mode,
            planner_tau_residual_penalty=0.0,
            planner_k_penalty_power=0.0,
            reward_prediction_source=reward_prediction_source,
            duration_blend_alpha=float(args.duration_blend_alpha),
            planner_tau_blend_alpha=float(args.duration_blend_alpha),
            allow_uncovered_reward_only=False,
            teacher_candidate_augmentation=True,
            teacher_candidate_neighbor_depth=int(ckpt_args.get("teacher_candidate_neighbor_depth", 1)),
            teacher_mode="kmc",
            neural_teacher=None,
            neural_teacher_device=args.device,
            neural_teacher_temperature=1.0,
            neural_teacher_epsilon=0.0,
            max_attempt_multiplier=50,
        )
        loader = mod._build_loader(samples, batch_size=int(args.batch_size), shuffle=False)
        metrics = mod._evaluate(
            model,
            loader,
            args.device,
            max_changed_sites=2 * summary_horizon,
            reward_prediction_source=reward_prediction_source,
        )
        scatter_rows, paired_tau = _paired_rows(
            model=model,
            samples=samples,
            batch_size=int(args.batch_size),
            device=args.device,
            reward_prediction_source=reward_prediction_source,
            temperature_code=str(code),
            temperature=float(temperature),
        )
        long_summary = _run_long_case(
            model=model,
            ckpt_args=ckpt_args,
            env_cfg=cfg,
            horizon_choices=segment_ks,
            rollout_segments=int(args.long_segments),
            seed=case_seed + 503,
            device=args.device,
            max_candidate_sites=max_candidate_sites,
            min_projected_changed_sites=int(args.min_projected_changed_sites),
            duration_source=args.duration_source,
            duration_blend_alpha=float(args.duration_blend_alpha),
            planner_score_mode=args.planner_score_mode,
            reward_prediction_source=reward_prediction_source,
            allow_teacher_noop_segments=bool(args.allow_teacher_noop_segments),
            progress_every=int(args.progress_every),
        )
        row = {
            "case": {
                "temperature_code": str(code),
                "temperature": float(temperature),
                "cu_density": float(cfg["cu_density"]),
                "lattice_size": [int(x) for x in cfg["lattice_size"]],
            },
            "seed": int(case_seed),
            "num_samples": int(len(samples)),
            "collection_stats": collection_stats,
            "metrics": metrics,
            "paired_tau": paired_tau,
            "scatter_samples": scatter_rows,
            "long": long_summary,
        }
        rows.append(row)
        print(
            json.dumps(
                {
                    "case_done": str(code),
                    "temperature": float(temperature),
                    "samples": int(len(samples)),
                    "coverage": float(collection_stats.get("coverage", 0.0)),
                    "reachable_edits": float(1.0 - metrics["reachability_violation_rate"]),
                    "changed_type_acc": float(metrics["changed_type_acc"]),
                    "tau_log_mae": float(paired_tau["log_mae"]),
                    "tau_log_corr": float(paired_tau["log_corr"]),
                    "tau_scale_ratio": float(paired_tau["scale_ratio"]),
                    "long_completed": int(long_summary["completed_rollout_segments"]),
                    "long_expected_time_ratio": long_summary["cumulative"]["expected_time_ratio"],
                    "long_stop_reason": long_summary["stop_reason"],
                }
            ),
            flush=True,
        )

    output = {
        "mode": "fig2_temperature_cases_controlled_validation",
        "checkpoint": str(args.checkpoint),
        "device": args.device,
        "seed": int(args.seed),
        "segment_ks": segment_ks,
        "summary_horizon_k": int(summary_horizon),
        "samples_per_temperature": int(args.samples_per_temperature),
        "long_segments": int(args.long_segments),
        "max_candidate_sites": int(max_candidate_sites),
        "min_projected_changed_sites": int(args.min_projected_changed_sites),
        "duration_source": args.duration_source,
        "duration_blend_alpha": float(args.duration_blend_alpha),
        "reward_prediction_source": reward_prediction_source,
        "codebook": {
            "temperature": {
                str(code): float(temp)
                for code, temp in zip(args.temperature_codes, args.temperatures)
            }
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[fig2-temp] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
