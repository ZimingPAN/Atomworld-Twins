from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import train_dreamer_macro_edit as mod


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate AtomWorld-Twins on a contiguous long teacher trajectory"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rollout_segments", type=int, default=500)
    parser.add_argument(
        "--max_episode_steps_override",
        type=int,
        default=None,
        help="Override the teacher env max_episode_steps so an older checkpoint can be evaluated on longer contiguous trajectories.",
    )
    parser.add_argument("--print_segments", type=int, default=5)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def _compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mae = float(np.mean(np.abs(pred - target)))
    rmse = float(np.sqrt(np.mean((pred - target) ** 2)))
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
    log_mae = float(np.mean(np.abs(log_pred - log_target)))
    log_rmse = float(np.sqrt(np.mean((log_pred - log_target) ** 2)))
    if pred.size > 1 and np.std(log_pred) > 0 and np.std(log_target) > 0:
        log_corr = float(np.corrcoef(log_pred, log_target)[0, 1])
    else:
        log_corr = 0.0
    scale_ratio = float(np.mean(pred / target))
    return {
        "log_mae": log_mae,
        "log_rmse": log_rmse,
        "log_corr": log_corr,
        "scale_ratio": scale_ratio,
    }


def _build_model(ckpt: dict[str, object], device: str) -> mod.MacroDreamerEditModel:
    args = ckpt["args"]
    include_stepwise_path_summary = args.get("teacher_path_summary_mode", "stepwise") == "stepwise"
    model = mod.MacroDreamerEditModel(
        max_vacancies=args["max_vacancies"],
        max_defects=args["max_defects"],
        max_shells=args["max_shells"],
        stats_dim=args["stats_dim"],
        lattice_size=tuple(args["lattice_size"]),
        neighbor_order=args["neighbor_order"],
        dim_latent=args["dim_latent"],
        graph_hidden_size=args["graph_hidden_size"],
        patch_hidden_size=args["patch_hidden_size"],
        patch_latent_dim=args["patch_latent_dim"],
        path_latent_dim=args["path_latent_dim"],
        global_summary_dim=16,
        teacher_path_summary_dim=mod.teacher_path_summary_dim(int(args["segment_k"]), include_stepwise_features=include_stepwise_path_summary),
        max_macro_k=max(int(args["segment_k"]), 16),
    ).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    model.realized_tau_head_loaded = not any(key.startswith("realized_duration_head.") for key in missing)
    if missing:
        print(f"Long-eval: missing keys initialized from scratch: {missing}")
    if unexpected:
        print(f"Long-eval: unexpected keys ignored: {unexpected}")
    model.eval()
    return model


def _build_env_cfg(ckpt_args: dict[str, object], max_episode_steps_override: int | None = None) -> dict[str, object]:
    return {
        "lattice_size": tuple(ckpt_args["lattice_size"]),
        "max_episode_steps": int(max_episode_steps_override if max_episode_steps_override is not None else ckpt_args["max_episode_steps"]),
        "max_vacancies": int(ckpt_args["max_vacancies"]),
        "max_defects": int(ckpt_args["max_defects"]),
        "max_shells": int(ckpt_args["max_shells"]),
        "stats_dim": int(ckpt_args["stats_dim"]),
        "temperature": float(ckpt_args["temperature"]),
        "reward_scale": float(ckpt_args["reward_scale"]),
        "cu_density": float(ckpt_args["cu_density"]),
        "v_density": float(ckpt_args["v_density"]),
        "rlkmc_topk": 16,
        "neighbor_order": ckpt_args["neighbor_order"],
    }


def _collect_teacher_segment(env: mod.MacroKMCEnv, horizon_k: int, rng: np.random.Generator) -> dict[str, object] | None:
    tau_exp = 0.0
    tau_real = 0.0
    reward_sum = 0.0
    done = False
    for _ in range(horizon_k):
        action = mod._sample_teacher_action(env, rng)
        if action is None:
            return None
        _next_obs, reward, done, info = env.step(action)
        tau_exp += float(info["expected_delta_t"])
        tau_real += float(info["delta_t"])
        reward_sum += float(reward)
        if done:
            return None
    return {
        "tau_exp": tau_exp,
        "tau_real": tau_real,
        "reward_sum": reward_sum,
    }


def _build_inference_tensors(
    *,
    env: mod.MacroKMCEnv,
    max_seed_vacancies: int,
    max_candidate_sites: int,
    horizon_k: int,
    device: str,
) -> dict[str, torch.Tensor] | None:
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
    positions, nearest_offsets, reach_depth, is_start_vacancy, current_types, _, _, = mod._build_patch_features(
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


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    checkpoint_path = Path(args.checkpoint)
    ckpt = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    ckpt_args = ckpt["args"]
    reward_scale = float(ckpt_args.get("reward_scale", 1.0))
    horizon_k = int(ckpt_args["segment_k"])
    max_seed_vacancies = int(ckpt_args["max_seed_vacancies"])
    max_candidate_sites = int(ckpt_args["max_candidate_sites"])

    model = _build_model(ckpt, args.device)
    env = mod.MacroKMCEnv(_build_env_cfg(ckpt_args, max_episode_steps_override=args.max_episode_steps_override))
    env.reset()

    pred_reward_sum = []
    pred_reward_raw = []
    pred_reward_gate = []
    true_reward_sum = []
    pred_tau_exp = []
    true_tau_exp = []
    pred_tau_real = []
    true_tau_real = []
    segments = []

    with torch.no_grad():
        for segment_idx in range(args.rollout_segments):
            tensors = _build_inference_tensors(
                env=env,
                max_seed_vacancies=max_seed_vacancies,
                max_candidate_sites=max_candidate_sites,
                horizon_k=horizon_k,
                device=args.device,
            )
            if tensors is None:
                break

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
            reward_raw = float(duration_outputs["reward"].item())
            reward_gate_prob = float(torch.sigmoid(duration_outputs["gate_logit"]).item())
            pred_reward = float((duration_outputs["reward"] * torch.sigmoid(duration_outputs["gate_logit"])).item())
            pred_expected_tau = float(torch.exp(duration_outputs["expected_tau_mu"]).item())
            pred_realized_tau_value = float(torch.exp(duration_outputs["realized_tau_mu"]).item())

            teacher_segment = _collect_teacher_segment(env, horizon_k=horizon_k, rng=rng)
            if teacher_segment is None:
                break

            pred_reward_sum.append(pred_reward)
            pred_reward_raw.append(reward_raw)
            pred_reward_gate.append(reward_gate_prob)
            true_reward_sum.append(float(teacher_segment["reward_sum"]))
            pred_tau_exp.append(pred_expected_tau)
            true_tau_exp.append(float(teacher_segment["tau_exp"]))
            pred_tau_real.append(pred_realized_tau_value)
            true_tau_real.append(float(teacher_segment["tau_real"]))
            segments.append(
                {
                    "index": segment_idx,
                    "segment_k": horizon_k,
                    "predicted_reward_sum": pred_reward,
                    "predicted_reward_raw": reward_raw,
                    "predicted_reward_gate_prob": reward_gate_prob,
                    "traditional_kmc_reward_sum": float(teacher_segment["reward_sum"]),
                    "predicted_expected_tau": pred_expected_tau,
                    "traditional_kmc_expected_tau": float(teacher_segment["tau_exp"]),
                    "predicted_realized_tau": pred_realized_tau_value,
                    "traditional_kmc_realized_tau": float(teacher_segment["tau_real"]),
                }
            )

    pred_reward_sum_np = np.asarray(pred_reward_sum, dtype=np.float64)
    true_reward_sum_np = np.asarray(true_reward_sum, dtype=np.float64)
    pred_tau_exp_np = np.asarray(pred_tau_exp, dtype=np.float64)
    true_tau_exp_np = np.asarray(true_tau_exp, dtype=np.float64)
    pred_tau_real_np = np.asarray(pred_tau_real, dtype=np.float64)
    true_tau_real_np = np.asarray(true_tau_real, dtype=np.float64)

    pred_delta_e_cumsum = np.cumsum(pred_reward_sum_np / reward_scale).tolist()
    true_delta_e_cumsum = np.cumsum(true_reward_sum_np / reward_scale).tolist()
    pred_tau_exp_cumsum = np.cumsum(pred_tau_exp_np).tolist()
    true_tau_exp_cumsum = np.cumsum(true_tau_exp_np).tolist()
    true_tau_real_cumsum = np.cumsum(true_tau_real_np).tolist()

    summary = {
        "mode": "teacher_forced_contiguous_long_trajectory",
        "checkpoint": str(checkpoint_path),
        "segment_k": horizon_k,
        "requested_rollout_segments": int(args.rollout_segments),
        "completed_rollout_segments": int(len(segments)),
        "teacher_env_max_episode_steps": int(
            args.max_episode_steps_override if args.max_episode_steps_override is not None else ckpt_args["max_episode_steps"]
        ),
        "teacher_source": "traditional_kmc_online_long_trajectory",
        "time_heads": {
            "expected_tau_head": True,
            "realized_tau_head_loaded": bool(getattr(model, "realized_tau_head_loaded", True)),
        },
        "reward_sum": _compute_metrics(pred_reward_sum_np, true_reward_sum_np) if len(segments) > 0 else {},
        "reward_diagnostics": mod._compute_reward_diagnostics(pred_reward_sum_np, true_reward_sum_np) if len(segments) > 0 else {},
        "tau_expected": (
            {**_compute_metrics(pred_tau_exp_np, true_tau_exp_np), **_compute_log_metrics(pred_tau_exp_np, true_tau_exp_np)}
            if len(segments) > 0
            else {}
        ),
        "tau_realized_reference": (
            {**_compute_metrics(pred_tau_real_np, true_tau_real_np), **_compute_log_metrics(pred_tau_real_np, true_tau_real_np)}
            if len(segments) > 0
            else {}
        ),
        "cumulative": {
            "predicted_reward_sum_final": float(pred_reward_sum_np.sum()) if len(segments) > 0 else 0.0,
            "traditional_kmc_reward_sum_final": float(true_reward_sum_np.sum()) if len(segments) > 0 else 0.0,
            "predicted_delta_e_final": float(np.sum(pred_reward_sum_np / reward_scale)) if len(segments) > 0 else 0.0,
            "traditional_kmc_delta_e_final": float(np.sum(true_reward_sum_np / reward_scale)) if len(segments) > 0 else 0.0,
            "predicted_expected_time_final": float(pred_tau_exp_np.sum()) if len(segments) > 0 else 0.0,
            "traditional_kmc_expected_time_final": float(true_tau_exp_np.sum()) if len(segments) > 0 else 0.0,
            "traditional_kmc_realized_time_final": float(true_tau_real_np.sum()) if len(segments) > 0 else 0.0,
            "delta_e_ratio": float(np.sum(pred_reward_sum_np) / np.sum(true_reward_sum_np)) if len(segments) > 0 and abs(np.sum(true_reward_sum_np)) > 1e-9 else None,
            "expected_time_ratio": float(np.sum(pred_tau_exp_np) / np.sum(true_tau_exp_np)) if len(segments) > 0 and np.sum(true_tau_exp_np) > 1e-12 else None,
            "cumulative_delta_e_mae": float(np.mean(np.abs(np.asarray(pred_delta_e_cumsum) - np.asarray(true_delta_e_cumsum)))) if len(segments) > 0 else 0.0,
        },
        "arrays": {
            "predicted_delta_e_cumsum": pred_delta_e_cumsum,
            "traditional_kmc_delta_e_cumsum": true_delta_e_cumsum,
            "predicted_expected_tau_cumsum": pred_tau_exp_cumsum,
            "traditional_kmc_expected_tau_cumsum": true_tau_exp_cumsum,
            "traditional_kmc_realized_tau_cumsum": true_tau_real_cumsum,
        },
        "segment_preview": segments[: max(args.print_segments, 0)],
        "segments": segments,
    }

    print("=" * 60)
    print("AtomWorld-Twins Long Trajectory Evaluation")
    print(f"completed_segments={summary['completed_rollout_segments']}, requested={summary['requested_rollout_segments']}, segment_k={horizon_k}")
    print("=" * 60)
    if len(segments) > 0:
        print(
            "Reward alignment: "
            f"mae={summary['reward_sum']['mae']:.6f}, rmse={summary['reward_sum']['rmse']:.6f}, corr={summary['reward_sum']['corr']:.4f}"
        )
        print(
            "Expected-time alignment: "
            f"log_mae={summary['tau_expected']['log_mae']:.4f}, log_corr={summary['tau_expected']['log_corr']:.4f}, "
            f"scale_ratio={summary['tau_expected']['scale_ratio']:.4f}"
        )
        print(
            "Cumulative long-horizon summary: "
            f"pred_dE={summary['cumulative']['predicted_delta_e_final']:.4f}, "
            f"teacher_dE={summary['cumulative']['traditional_kmc_delta_e_final']:.4f}, "
            f"pred_tau={summary['cumulative']['predicted_expected_time_final']:.4f}, "
            f"teacher_tau={summary['cumulative']['traditional_kmc_expected_time_final']:.4f}, "
            f"cum_dE_mae={summary['cumulative']['cumulative_delta_e_mae']:.4f}"
        )
        if summary["segment_preview"]:
            print("Segment preview:")
            for row in summary["segment_preview"]:
                print(json.dumps(row, ensure_ascii=False))
    else:
        print("No valid contiguous segments were collected.")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved summary to {output_path}")


if __name__ == "__main__":
    main()