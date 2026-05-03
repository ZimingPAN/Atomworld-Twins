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
    parser.add_argument("--duration_checkpoint", type=str, default=None,
                        help="Optional checkpoint used only for duration prediction/scoring. "
                             "The primary checkpoint still provides edit/reward predictions.")
    parser.add_argument("--planner_duration_checkpoint_source", type=str, default="duration", choices=["primary", "duration"],
                        help="When --duration_checkpoint is set, choose whether planner scoring uses the primary checkpoint's "
                             "duration estimate or the duration checkpoint's estimate. Reported duration still uses --duration_checkpoint.")
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
    parser.add_argument("--progress_every", type=int, default=50,
                        help="Print a compact progress line every N completed long-eval segments; set <=0 to disable.")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--planner_segment_ks", type=int, nargs="+", default=None,
                        help="Optional multi-k planning horizons. If omitted, a multi-k checkpoint uses its saved segment_ks.")
    parser.add_argument("--min_projected_changed_sites", type=int, default=2,
                        help="Reject planner candidates whose projected edit changes fewer sites than this value.")
    parser.add_argument("--duration_source", type=str, default="model", choices=["model", "baseline", "blend"],
                        help="Duration estimate used for planner scoring and reported prediction. "
                             "'baseline' uses the CTMC start-state baseline k/Gamma_tot(s_t) for diagnosis; "
                             "'blend' uses log-space interpolation between baseline and model.")
    parser.add_argument("--duration_blend_alpha", type=float, default=1.0,
                        help="For --duration_source blend, alpha in log_tau = (1-alpha)*baseline + alpha*model.")
    parser.add_argument("--duration_log_offset", type=float, default=0.0,
                        help="Global additive offset applied to log(model expected tau) before reporting/scoring durations.")
    parser.add_argument("--online_duration_calibration_segments", type=int, default=0,
                        help="Use the first N long-trajectory teacher segments to estimate a global log-duration offset, "
                             "then apply it to later segments. Overall metrics include warmup; post_calibration excludes it.")
    parser.add_argument("--planner_tau_source", type=str, default=None, choices=["model", "baseline", "blend"],
                        help="Duration estimate used only for planner scoring. Defaults to --duration_source.")
    parser.add_argument("--planner_tau_blend_alpha", type=float, default=None,
                        help="For --planner_tau_source blend, alpha in log_tau. Defaults to --duration_blend_alpha.")
    parser.add_argument("--planner_score_mode", type=str, default="energy_per_tau",
                        choices=["energy_per_tau", "energy_per_sqrt_tau", "energy"],
                        help="How to score legal multi-k candidates after reachability projection.")
    parser.add_argument("--planner_tau_residual_penalty", type=float, default=0.0,
                        help="Apply exp(-w * |log(model_tau / baseline_tau)|) to planner scores.")
    parser.add_argument("--planner_k_penalty_power", type=float, default=0.0,
                        help="Apply score /= k ** power to conservatively prefer shorter legal horizons.")
    parser.add_argument("--allow_teacher_noop_segments", action="store_true",
                        help="Keep teacher macro segments whose start/end lattice state is unchanged. "
                             "By default long eval stops at the first such segment, matching the no-op "
                             "filter used by macro segment training data.")
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


def _segment_ks_from_ckpt_args(args: dict[str, object]) -> list[int]:
    if args.get("segment_ks"):
        return sorted({int(k) for k in args["segment_ks"]})
    return [int(args["segment_k"])]


def _summary_horizon_k_from_ckpt_args(args: dict[str, object]) -> int:
    return max(_segment_ks_from_ckpt_args(args))


def _build_model(ckpt: dict[str, object], device: str) -> mod.MacroDreamerEditModel:
    args = ckpt["args"]
    include_stepwise_path_summary = args.get("teacher_path_summary_mode", "stepwise") == "stepwise"
    summary_horizon_k = _summary_horizon_k_from_ckpt_args(args)
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
        teacher_path_summary_dim=mod.teacher_path_summary_dim(summary_horizon_k, include_stepwise_features=include_stepwise_path_summary),
        max_macro_k=max(summary_horizon_k, 16),
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
    start_vacancies = env.env.get_vacancy_array().astype(np.int32)
    start_cu = env.env.get_cu_array().astype(np.int32)
    start_vac_set, start_cu_set = mod._positions_to_type_lookup(start_vacancies, start_cu)
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
    end_vacancies = env.env.get_vacancy_array().astype(np.int32)
    end_cu = env.env.get_cu_array().astype(np.int32)
    end_vac_set, end_cu_set = mod._positions_to_type_lookup(end_vacancies, end_cu)
    changed_positions = mod._changed_positions_between(start_vac_set, start_cu_set, end_vac_set, end_cu_set)
    return {
        "tau_exp": tau_exp,
        "tau_real": tau_real,
        "reward_sum": reward_sum,
        "changed_site_count": int(len(changed_positions)),
        "is_noop": bool(len(changed_positions) == 0),
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


def _choose_planner_candidate(candidates: list[dict[str, object]], *, min_projected_changed_sites: int = 2) -> dict[str, object] | None:
    if not candidates:
        return None
    legal = [
        item
        for item in candidates
        if float(item.get("reachability_violation", 1.0)) <= 0.0
        and float(item.get("projected_changed_count", 0.0)) >= float(min_projected_changed_sites)
    ]
    if not legal:
        return None
    return max(legal, key=lambda item: float(item.get("selection_score", -float("inf"))))


def _duration_from_source(
    *,
    model_expected_tau: float,
    baseline_expected_tau: float,
    source: str,
    blend_alpha: float = 1.0,
    duration_log_offset: float = 0.0,
) -> float:
    model_tau = max(float(model_expected_tau), 1e-12)
    baseline_tau = max(float(baseline_expected_tau), 1e-12)
    model_log_tau = float(np.log(model_tau) + float(duration_log_offset))
    if source == "model":
        return float(np.exp(model_log_tau))
    if source == "baseline":
        return baseline_tau
    if source == "blend":
        alpha = float(np.clip(blend_alpha, 0.0, 1.0))
        return float(np.exp((1.0 - alpha) * np.log(baseline_tau) + alpha * model_log_tau))
    raise ValueError(f"Unknown duration source: {source}")


def _estimate_duration_log_offset(
    *,
    base_log_offset: float,
    predicted_tau: list[float],
    target_tau: list[float],
) -> float:
    if not predicted_tau:
        return float(base_log_offset)
    pred = np.clip(np.asarray(predicted_tau, dtype=np.float64), 1e-12, None)
    target = np.clip(np.asarray(target_tau, dtype=np.float64), 1e-12, None)
    return float(base_log_offset + np.mean(np.log(target) - np.log(pred)))


def _compute_selection_score(
    *,
    pred_reward_sum: float,
    reward_scale: float,
    model_expected_tau: float,
    baseline_expected_tau: float,
    horizon_k: int,
    planner_tau_source: str = "model",
    planner_score_mode: str = "energy_per_tau",
    planner_tau_residual_penalty: float = 0.0,
    planner_k_penalty_power: float = 0.0,
    planner_tau_blend_alpha: float = 1.0,
    planner_tau_log_offset: float = 0.0,
) -> tuple[float, float]:
    tau_for_score = _duration_from_source(
        model_expected_tau=model_expected_tau,
        baseline_expected_tau=baseline_expected_tau,
        source=planner_tau_source,
        blend_alpha=planner_tau_blend_alpha,
        duration_log_offset=planner_tau_log_offset,
    )
    delta_e = float(pred_reward_sum) / max(float(reward_scale), 1e-12)

    if planner_score_mode == "energy":
        score = delta_e
    elif planner_score_mode == "energy_per_sqrt_tau":
        score = delta_e / float(np.sqrt(tau_for_score))
    else:
        score = delta_e / tau_for_score

    if planner_tau_residual_penalty > 0.0:
        model_tau = max(float(model_expected_tau), 1e-12)
        baseline_tau = max(float(baseline_expected_tau), 1e-12)
        residual = abs(float(np.log(model_tau / baseline_tau)))
        score *= float(np.exp(-float(planner_tau_residual_penalty) * residual))
    if planner_k_penalty_power > 0.0:
        score /= max(float(horizon_k), 1.0) ** float(planner_k_penalty_power)
    return float(score), float(tau_for_score)


def _predict_candidate_for_horizon(
    *,
    model: mod.MacroDreamerEditModel,
    duration_model: mod.MacroDreamerEditModel | None,
    env: mod.MacroKMCEnv,
    horizon_k: int,
    max_seed_vacancies: int,
    max_candidate_sites: int,
    reward_scale: float,
    device: str,
    duration_source: str = "model",
    planner_tau_source: str = "model",
    planner_score_mode: str = "energy_per_tau",
    planner_tau_residual_penalty: float = 0.0,
    planner_k_penalty_power: float = 0.0,
    duration_blend_alpha: float = 1.0,
    planner_tau_blend_alpha: float = 1.0,
    duration_log_offset: float = 0.0,
    planner_tau_log_offset: float = 0.0,
    planner_duration_checkpoint_source: str = "duration",
    reward_prediction_source: str = "raw",
) -> dict[str, object] | None:
    tensors = _build_inference_tensors(
        env=env,
        max_seed_vacancies=max_seed_vacancies,
        max_candidate_sites=max_candidate_sites,
        horizon_k=horizon_k,
        device=device,
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
    projected_types, projected_changed_mask, transport_cost, reachability_violation = mod.project_types_by_inventory(
        current_types=tensors["current_types"],
        change_logits=change_logits,
        type_logits=raw_type_logits,
        node_mask=tensors["candidate_mask"],
        positions=tensors["candidate_positions"],
        box_dims=tensors["box_dims"],
        horizon_k=tensors["horizon_k"],
        max_changed_sites=2 * tensors["horizon_k"],
    )
    reward_patch_latent = patch_latent
    reward_change_logits = change_logits
    reward_type_logits = raw_type_logits
    if reward_prediction_source == "projected":
        _, reward_patch_latent = model.encode_patch(
            positions=tensors["candidate_positions"],
            nearest_vacancy_offset=tensors["nearest_vacancy_offset"],
            reach_depth=tensors["reach_depth"],
            is_start_vacancy=tensors["is_start_vacancy"],
            type_ids=projected_types,
            node_mask=tensors["candidate_mask"],
            global_summary=tensors["global_summary"],
            box_dims=tensors["box_dims"],
        )
        reward_change_logits, reward_type_logits = mod.projected_edit_logits_from_types(
            current_types=tensors["current_types"],
            projected_types=projected_types,
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
        duration_path_latent = duration_model.sample_path_latent(
            duration_prior_mu,
            duration_prior_logvar,
            deterministic=True,
        )
        duration_next_pred = duration_model.predict_next_global(
            duration_global_latent,
            duration_path_latent,
            tensors["horizon_k"],
        )
        duration_change_logits, duration_type_logits = duration_model.decode_edit(
            site_latent=duration_site_latent,
            patch_latent=duration_patch_latent,
            predicted_next_global=duration_next_pred,
            path_latent=duration_path_latent,
            horizon_k=tensors["horizon_k"],
            current_types=tensors["current_types"],
        )
        duration_patch_latent_for_head = duration_patch_latent
        duration_change_logits_for_head = duration_change_logits
        duration_type_logits_for_head = duration_type_logits
        if reward_prediction_source == "projected":
            duration_projected_types, _, _, _ = mod.project_types_by_inventory(
                current_types=tensors["current_types"],
                change_logits=duration_change_logits,
                type_logits=duration_type_logits,
                node_mask=tensors["candidate_mask"],
                positions=tensors["candidate_positions"],
                box_dims=tensors["box_dims"],
                horizon_k=tensors["horizon_k"],
                max_changed_sites=2 * tensors["horizon_k"],
            )
            _, duration_patch_latent_for_head = duration_model.encode_patch(
                positions=tensors["candidate_positions"],
                nearest_vacancy_offset=tensors["nearest_vacancy_offset"],
                reach_depth=tensors["reach_depth"],
                is_start_vacancy=tensors["is_start_vacancy"],
                type_ids=duration_projected_types,
                node_mask=tensors["candidate_mask"],
                global_summary=tensors["global_summary"],
                box_dims=tensors["box_dims"],
            )
            duration_change_logits_for_head, duration_type_logits_for_head = mod.projected_edit_logits_from_types(
                current_types=tensors["current_types"],
                projected_types=duration_projected_types,
                candidate_mask=tensors["candidate_mask"],
            )
        duration_outputs = mod._predict_reward_and_duration_outputs(
            duration_model,
            duration_global_latent,
            duration_next_pred,
            duration_path_latent,
            tensors["global_summary"],
            tensors["horizon_k"],
            patch_latent=duration_patch_latent_for_head,
            change_logits=duration_change_logits_for_head,
            type_logits=duration_type_logits_for_head,
            current_types=tensors["current_types"],
            candidate_mask=tensors["candidate_mask"],
        )
    reward_raw = float(primary_outputs["reward"].item())
    reward_gate_prob = float(torch.sigmoid(primary_outputs["gate_logit"]).item())
    pred_reward = float((primary_outputs["reward"] * torch.sigmoid(primary_outputs["gate_logit"])).item())
    primary_expected_tau = float(torch.exp(primary_outputs["expected_tau_mu"]).item())
    model_expected_tau = float(torch.exp(duration_outputs["expected_tau_mu"]).item())
    model_realized_tau = float(torch.exp(duration_outputs["realized_tau_mu"]).item())
    baseline_log_tau = mod.macro_duration_baseline_log_tau(tensors["global_summary"], tensors["horizon_k"])
    baseline_expected_tau = float(torch.exp(baseline_log_tau).item())
    pred_expected_tau = _duration_from_source(
        model_expected_tau=model_expected_tau,
        baseline_expected_tau=baseline_expected_tau,
        source=duration_source,
        blend_alpha=duration_blend_alpha,
        duration_log_offset=duration_log_offset,
    )
    pred_realized_tau = model_realized_tau
    if duration_source in {"baseline", "blend"}:
        pred_realized_tau = pred_expected_tau
    violation = float(reachability_violation.item())
    changed_count = float(projected_changed_mask.sum().item())
    score_model_expected_tau = (
        primary_expected_tau
        if duration_model is not None
        and duration_model is not model
        and planner_duration_checkpoint_source == "primary"
        else model_expected_tau
    )
    selection_score, tau_for_score = _compute_selection_score(
        pred_reward_sum=pred_reward,
        reward_scale=reward_scale,
        model_expected_tau=score_model_expected_tau,
        baseline_expected_tau=baseline_expected_tau,
        horizon_k=int(horizon_k),
        planner_tau_source=planner_tau_source,
        planner_score_mode=planner_score_mode,
        planner_tau_residual_penalty=planner_tau_residual_penalty,
        planner_k_penalty_power=planner_k_penalty_power,
        planner_tau_blend_alpha=planner_tau_blend_alpha,
        planner_tau_log_offset=planner_tau_log_offset,
    )
    if violation > 0.0:
        selection_score = -float("inf")
    return {
        "segment_k": int(horizon_k),
        "predicted_reward_sum": pred_reward,
        "predicted_delta_e": float(pred_reward / reward_scale),
        "predicted_reward_raw": reward_raw,
        "predicted_reward_gate_prob": reward_gate_prob,
        "predicted_expected_tau": pred_expected_tau,
        "predicted_realized_tau": pred_realized_tau,
        "model_expected_tau": model_expected_tau,
        "primary_model_expected_tau": primary_expected_tau,
        "score_model_expected_tau": score_model_expected_tau,
        "model_realized_tau": model_realized_tau,
        "baseline_expected_tau": baseline_expected_tau,
        "planner_tau_for_score": tau_for_score,
        "duration_blend_alpha": float(duration_blend_alpha),
        "planner_tau_blend_alpha": float(planner_tau_blend_alpha),
        "duration_log_offset": float(duration_log_offset),
        "planner_tau_log_offset": float(planner_tau_log_offset),
        "reward_prediction_source": reward_prediction_source,
        "reachability_violation": violation,
        "projected_changed_count": changed_count,
        "transport_cost": float(transport_cost.item()),
        "selection_score": float(selection_score),
    }


def main() -> None:
    args = parse_args()
    planner_tau_source = args.planner_tau_source or args.duration_source
    planner_tau_blend_alpha = (
        float(args.duration_blend_alpha)
        if args.planner_tau_blend_alpha is None
        else float(args.planner_tau_blend_alpha)
    )
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    checkpoint_path = Path(args.checkpoint)
    ckpt = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    ckpt_args = ckpt["args"]
    reward_scale = float(ckpt_args.get("reward_scale", 1.0))
    reward_prediction_source = str(ckpt_args.get("reward_prediction_source", "raw"))
    ckpt_segment_ks = _segment_ks_from_ckpt_args(ckpt_args)
    if args.planner_segment_ks:
        horizon_choices = sorted({int(k) for k in args.planner_segment_ks})
    elif len(ckpt_segment_ks) > 1:
        horizon_choices = ckpt_segment_ks
    else:
        horizon_choices = [int(ckpt_args["segment_k"])]
    planner_enabled = len(horizon_choices) > 1
    horizon_k = int(horizon_choices[0]) if len(horizon_choices) == 1 else int(max(horizon_choices))
    max_seed_vacancies = int(ckpt_args["max_seed_vacancies"])
    max_candidate_sites = int(ckpt_args["max_candidate_sites"])

    model = _build_model(ckpt, args.device)
    duration_checkpoint_path = Path(args.duration_checkpoint) if args.duration_checkpoint else None
    duration_model = None
    if duration_checkpoint_path is not None:
        duration_ckpt = torch.load(duration_checkpoint_path, map_location=args.device, weights_only=False)
        duration_model = _build_model(duration_ckpt, args.device)
    # Model construction consumes torch RNG. Re-seed before creating the KMC env
    # so adding an auxiliary duration checkpoint cannot change the teacher path.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
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
    chosen_ks = []
    segments = []
    skipped_noop = 0
    skipped_terminal = 0
    stop_reason = "completed"
    stop_segment: dict[str, object] | None = None
    duration_log_offset = float(args.duration_log_offset)
    calibration_source_pred_tau: list[float] = []
    calibration_target_tau: list[float] = []
    calibration_active = int(args.online_duration_calibration_segments) > 0 and args.duration_source != "baseline"

    with torch.no_grad():
        for segment_idx in range(args.rollout_segments):
            candidates = [
                item
                for item in (
                    _predict_candidate_for_horizon(
                        model=model,
                        duration_model=duration_model,
                        env=env,
                        horizon_k=item_k,
                        max_seed_vacancies=max_seed_vacancies,
                        max_candidate_sites=max_candidate_sites,
                        reward_scale=reward_scale,
                        device=args.device,
                        duration_source=args.duration_source,
                        planner_tau_source=planner_tau_source,
                        planner_score_mode=args.planner_score_mode,
                        planner_tau_residual_penalty=args.planner_tau_residual_penalty,
                        planner_k_penalty_power=args.planner_k_penalty_power,
                        duration_blend_alpha=args.duration_blend_alpha,
                        planner_tau_blend_alpha=planner_tau_blend_alpha,
                        duration_log_offset=duration_log_offset,
                        planner_tau_log_offset=duration_log_offset,
                        planner_duration_checkpoint_source=args.planner_duration_checkpoint_source,
                        reward_prediction_source=reward_prediction_source,
                    )
                    for item_k in horizon_choices
                )
                if item is not None
            ]
            # The minimum projected-change filter is a planner-selection guard.
            # For a single fixed-k checkpoint there is no competing candidate to
            # choose among; rejecting the only candidate turns a valid
            # teacher-forced time evaluation into a zero-length rollout whenever
            # the projected edit is a no-op at the current teacher state.
            effective_min_projected_changed_sites = (
                int(args.min_projected_changed_sites) if planner_enabled else 0
            )
            selected = _choose_planner_candidate(
                candidates,
                min_projected_changed_sites=effective_min_projected_changed_sites,
            )
            if selected is None:
                stop_reason = "no_legal_planner_candidate"
                stop_segment = {
                    "index": segment_idx,
                    "planner_candidates": candidates,
                }
                break
            selected_k = int(selected["segment_k"])

            teacher_segment = _collect_teacher_segment(env, horizon_k=selected_k, rng=rng)
            if teacher_segment is None:
                skipped_terminal += 1
                stop_reason = "teacher_terminal_or_action_missing"
                stop_segment = {
                    "index": segment_idx,
                    "segment_k": selected_k,
                    "planner_candidates": candidates,
                    "selected": selected,
                }
                break
            if bool(teacher_segment.get("is_noop", False)) and not args.allow_teacher_noop_segments:
                skipped_noop += 1
                stop_reason = "noop_teacher_segment"
                stop_segment = {
                    "index": segment_idx,
                    "segment_k": selected_k,
                    "planner_candidates": candidates,
                    "selected": selected,
                    "traditional_kmc_reward_sum": float(teacher_segment["reward_sum"]),
                    "traditional_kmc_delta_e": float(teacher_segment["reward_sum"] / reward_scale),
                    "traditional_kmc_expected_tau": float(teacher_segment["tau_exp"]),
                    "traditional_kmc_realized_tau": float(teacher_segment["tau_real"]),
                    "traditional_changed_site_count": int(teacher_segment.get("changed_site_count", 0)),
                }
                break

            pred_reward_sum.append(float(selected["predicted_reward_sum"]))
            pred_reward_raw.append(float(selected["predicted_reward_raw"]))
            pred_reward_gate.append(float(selected["predicted_reward_gate_prob"]))
            true_reward_sum.append(float(teacher_segment["reward_sum"]))
            pred_tau_exp.append(float(selected["predicted_expected_tau"]))
            true_tau_exp.append(float(teacher_segment["tau_exp"]))
            pred_tau_real.append(float(selected["predicted_realized_tau"]))
            true_tau_real.append(float(teacher_segment["tau_real"]))
            chosen_ks.append(selected_k)
            segments.append(
                {
                    "index": segment_idx,
                    "segment_k": selected_k,
                    "planner_candidates": candidates,
                    "selection_score": float(selected["selection_score"]),
                    "predicted_reward_sum": float(selected["predicted_reward_sum"]),
                    "predicted_reward_raw": float(selected["predicted_reward_raw"]),
                    "predicted_reward_gate_prob": float(selected["predicted_reward_gate_prob"]),
                    "predicted_delta_e": float(selected["predicted_delta_e"]),
                    "traditional_kmc_reward_sum": float(teacher_segment["reward_sum"]),
                    "traditional_kmc_delta_e": float(teacher_segment["reward_sum"] / reward_scale),
                    "predicted_expected_tau": float(selected["predicted_expected_tau"]),
                    "traditional_kmc_expected_tau": float(teacher_segment["tau_exp"]),
                    "predicted_realized_tau": float(selected["predicted_realized_tau"]),
                    "traditional_kmc_realized_tau": float(teacher_segment["tau_real"]),
                    "model_expected_tau": float(selected["model_expected_tau"]),
                    "primary_model_expected_tau": float(selected["primary_model_expected_tau"]),
                    "score_model_expected_tau": float(selected["score_model_expected_tau"]),
                    "baseline_expected_tau": float(selected["baseline_expected_tau"]),
                    "planner_tau_for_score": float(selected["planner_tau_for_score"]),
                    "duration_log_offset": float(selected["duration_log_offset"]),
                    "planner_tau_log_offset": float(selected["planner_tau_log_offset"]),
                    "duration_calibration_observations": int(len(calibration_source_pred_tau)),
                    "reachability_violation": float(selected["reachability_violation"]),
                    "projected_changed_count": float(selected["projected_changed_count"]),
                    "traditional_changed_site_count": int(teacher_segment.get("changed_site_count", 0)),
                    "traditional_is_noop": bool(teacher_segment.get("is_noop", False)),
                }
            )
            if calibration_active and len(calibration_source_pred_tau) < int(args.online_duration_calibration_segments):
                calibration_pred_tau = _duration_from_source(
                    model_expected_tau=float(selected["model_expected_tau"]),
                    baseline_expected_tau=float(selected["baseline_expected_tau"]),
                    source=args.duration_source,
                    blend_alpha=args.duration_blend_alpha,
                    duration_log_offset=float(args.duration_log_offset),
                )
                calibration_source_pred_tau.append(float(calibration_pred_tau))
                calibration_target_tau.append(float(teacher_segment["tau_exp"]))
                if len(calibration_source_pred_tau) == int(args.online_duration_calibration_segments):
                    duration_log_offset = _estimate_duration_log_offset(
                        base_log_offset=float(args.duration_log_offset),
                        predicted_tau=calibration_source_pred_tau,
                        target_tau=calibration_target_tau,
                    )
            if args.progress_every > 0 and (segment_idx + 1) % int(args.progress_every) == 0:
                print(
                    json.dumps(
                        {
                            "long_eval_progress": {
                                "segments": int(segment_idx + 1),
                                "chosen_k_histogram": {
                                    str(int(item_k)): int(np.sum(np.asarray(chosen_ks, dtype=np.int64) == int(item_k)))
                                    for item_k in sorted(set(chosen_ks))
                                },
                                "predicted_delta_e": float(np.sum(np.asarray(pred_reward_sum, dtype=np.float64) / reward_scale)),
                                "traditional_delta_e": float(np.sum(np.asarray(true_reward_sum, dtype=np.float64) / reward_scale)),
                                "predicted_tau": float(np.sum(np.asarray(pred_tau_exp, dtype=np.float64))),
                                "traditional_tau": float(np.sum(np.asarray(true_tau_exp, dtype=np.float64))),
                            }
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    pred_reward_sum_np = np.asarray(pred_reward_sum, dtype=np.float64)
    true_reward_sum_np = np.asarray(true_reward_sum, dtype=np.float64)
    pred_tau_exp_np = np.asarray(pred_tau_exp, dtype=np.float64)
    true_tau_exp_np = np.asarray(true_tau_exp, dtype=np.float64)
    pred_tau_real_np = np.asarray(pred_tau_real, dtype=np.float64)
    true_tau_real_np = np.asarray(true_tau_real, dtype=np.float64)
    chosen_ks_np = np.asarray(chosen_ks, dtype=np.int64)

    pred_delta_e_cumsum = np.cumsum(pred_reward_sum_np / reward_scale).tolist()
    true_delta_e_cumsum = np.cumsum(true_reward_sum_np / reward_scale).tolist()
    pred_tau_exp_cumsum = np.cumsum(pred_tau_exp_np).tolist()
    true_tau_exp_cumsum = np.cumsum(true_tau_exp_np).tolist()
    true_tau_real_cumsum = np.cumsum(true_tau_real_np).tolist()

    by_k = {}
    chosen_k_histogram = {}
    for item_k in sorted(set(chosen_ks)):
        mask = chosen_ks_np == int(item_k)
        chosen_k_histogram[str(int(item_k))] = int(mask.sum())
        by_k[str(int(item_k))] = {
            "num_segments": int(mask.sum()),
            "reward_sum": _compute_metrics(pred_reward_sum_np[mask], true_reward_sum_np[mask]),
            "reward_diagnostics": mod._compute_reward_diagnostics(pred_reward_sum_np[mask], true_reward_sum_np[mask]),
            "tau_expected": {**_compute_metrics(pred_tau_exp_np[mask], true_tau_exp_np[mask]), **_compute_log_metrics(pred_tau_exp_np[mask], true_tau_exp_np[mask])},
            "tau_realized_reference": {**_compute_metrics(pred_tau_real_np[mask], true_tau_real_np[mask]), **_compute_log_metrics(pred_tau_real_np[mask], true_tau_real_np[mask])},
            "cumulative": {
                "predicted_delta_e_final": float(np.sum(pred_reward_sum_np[mask] / reward_scale)),
                "traditional_kmc_delta_e_final": float(np.sum(true_reward_sum_np[mask] / reward_scale)),
                "predicted_expected_time_final": float(np.sum(pred_tau_exp_np[mask])),
                "traditional_kmc_expected_time_final": float(np.sum(true_tau_exp_np[mask])),
            },
        }
    post_calibration = {}
    calibration_start = int(args.online_duration_calibration_segments) if calibration_active else 0
    if calibration_active and len(segments) > calibration_start:
        post_pred_reward = pred_reward_sum_np[calibration_start:]
        post_true_reward = true_reward_sum_np[calibration_start:]
        post_pred_tau = pred_tau_exp_np[calibration_start:]
        post_true_tau = true_tau_exp_np[calibration_start:]
        post_calibration = {
            "start_index": int(calibration_start),
            "num_segments": int(len(segments) - calibration_start),
            "duration_log_offset": float(duration_log_offset),
            "reward_sum": _compute_metrics(post_pred_reward, post_true_reward),
            "reward_diagnostics": mod._compute_reward_diagnostics(post_pred_reward, post_true_reward),
            "tau_expected": {**_compute_metrics(post_pred_tau, post_true_tau), **_compute_log_metrics(post_pred_tau, post_true_tau)},
            "cumulative": {
                "predicted_delta_e_final": float(np.sum(post_pred_reward / reward_scale)),
                "traditional_kmc_delta_e_final": float(np.sum(post_true_reward / reward_scale)),
                "predicted_expected_time_final": float(post_pred_tau.sum()),
                "traditional_kmc_expected_time_final": float(post_true_tau.sum()),
            },
        }

    summary = {
        "mode": "multi_k_planner_teacher_forced_contiguous_long_trajectory" if planner_enabled else "teacher_forced_contiguous_long_trajectory",
        "checkpoint": str(checkpoint_path),
        "duration_checkpoint": str(duration_checkpoint_path) if duration_checkpoint_path is not None else None,
        "planner_duration_checkpoint_source": args.planner_duration_checkpoint_source,
        "segment_k": horizon_k,
        "segment_ks": horizon_choices,
        "planner_enabled": planner_enabled,
        "min_projected_changed_sites": int(args.min_projected_changed_sites),
        "effective_min_projected_changed_sites": int(args.min_projected_changed_sites) if planner_enabled else 0,
        "duration_source": args.duration_source,
        "duration_blend_alpha": float(args.duration_blend_alpha),
        "duration_log_offset": float(args.duration_log_offset),
        "duration_log_offset_final": float(duration_log_offset),
        "online_duration_calibration_segments": int(args.online_duration_calibration_segments),
        "duration_calibration_samples": int(len(calibration_source_pred_tau)),
        "planner_tau_source": planner_tau_source,
        "planner_tau_blend_alpha": float(planner_tau_blend_alpha),
        "planner_score_mode": args.planner_score_mode,
        "planner_tau_residual_penalty": float(args.planner_tau_residual_penalty),
        "planner_k_penalty_power": float(args.planner_k_penalty_power),
        "reward_prediction_source": reward_prediction_source,
        "chosen_k_histogram": chosen_k_histogram,
        "requested_rollout_segments": int(args.rollout_segments),
        "completed_rollout_segments": int(len(segments)),
        "stop_reason": stop_reason,
        "stop_segment": stop_segment,
        "skipped_noop": int(skipped_noop),
        "skipped_terminal": int(skipped_terminal),
        "allow_teacher_noop_segments": bool(args.allow_teacher_noop_segments),
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
        "by_k": by_k,
        "post_calibration": post_calibration,
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
    print(f"completed_segments={summary['completed_rollout_segments']}, requested={summary['requested_rollout_segments']}, segment_ks={horizon_choices}, planner={planner_enabled}")
    print(
        f"stop_reason={summary['stop_reason']}, skipped_noop={summary['skipped_noop']}, "
        f"allow_teacher_noop_segments={summary['allow_teacher_noop_segments']}"
    )
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
        if summary["post_calibration"]:
            post_tau = summary["post_calibration"]["tau_expected"]
            print(
                "Post-calibration expected-time alignment: "
                f"start={summary['post_calibration']['start_index']}, "
                f"log_mae={post_tau['log_mae']:.4f}, log_corr={post_tau['log_corr']:.4f}, "
                f"scale_ratio={post_tau['scale_ratio']:.4f}"
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
