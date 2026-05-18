from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

import train_dreamer_macro_edit as mod


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate AtomWorld-Mirror on a contiguous long teacher trajectory"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--duration_checkpoint", type=str, default=None,
                        help="Optional checkpoint used only for duration prediction/scoring. "
                             "The primary checkpoint still provides edit/reward predictions.")
    parser.add_argument("--reward_checkpoint", type=str, default=None,
                        help="Optional checkpoint used only for reward/gate/no-op-risk prediction/scoring. "
                             "The primary checkpoint still provides edit/projection candidates.")
    parser.add_argument("--planner_duration_checkpoint_source", type=str, default="duration", choices=["primary", "duration"],
                        help="When --duration_checkpoint is set, choose whether planner scoring uses the primary checkpoint's "
                             "duration estimate or the duration checkpoint's estimate. Reported duration still uses --duration_checkpoint.")
    parser.add_argument("--aux_projected_types_source", type=str, default="aux", choices=["primary", "aux"],
                        help="When an auxiliary reward/duration checkpoint is used with projected reward source, choose whether "
                             "the auxiliary heads evaluate the primary edit checkpoint's projected types or their own projection.")
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
                        help="Override multi-k planning horizons. If omitted, a multi-k checkpoint uses its saved segment_ks.")
    parser.add_argument("--max_seed_vacancies_override", type=int, default=None,
                        help="Override checkpoint max_seed_vacancies for inference candidate construction.")
    parser.add_argument("--max_candidate_sites_override", type=int, default=None,
                        help="Override checkpoint max_candidate_sites for inference candidate construction.")
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
    parser.add_argument("--planner_noop_risk_penalty", type=float, default=0.0,
                        help="Apply an explicit no-op/terminal-risk penalty to planner scores when the checkpoint exposes noop_risk_logit.")
    parser.add_argument("--planner_projection_change_source", type=str, default="change",
                        choices=[
                            "change",
                            "proposal",
                            "action_support",
                            "action_source",
                            "action_destination",
                            "action_endpoint",
                            "action_edge_completion",
                            "action_edge_pair_completion",
                            "action_edge_pair_support_completion",
                            "action_edge_pair_blend_completion",
                            "action_edge_pair_multiobjective_completion",
                            "vacancy_pair_completion",
                            "vacancy_pair_energy_blend_completion",
                            "vacancy_pair_interaction_completion",
                            "vacancy_pair_interaction_energy_blend_completion",
                            "terminal_edit_decoupled",
                            "terminal_edit_inside_action_edge",
                            "terminal_edit_sequence_rollout",
                            "terminal_edit_inside_sequence_rollout",
                            "two_stage_vacancy_displacement",
                            "terminal_typed_diff",
                            "blend",
                        ],
                        help="Change logits used by projection during planner candidate construction. "
                             "'proposal' uses terminal support, 'action_support' uses the independent action-support head; "
                             "default preserves legacy behavior.")
    parser.add_argument("--planner_projection_change_blend_alpha", type=float, default=0.5,
                        help="For --planner_projection_change_source blend, alpha in logits=(1-alpha)*change + alpha*proposal.")
    parser.add_argument("--planner_projection_topk_source", type=str, default="none",
                        choices=[
                            "none",
                            "change",
                            "proposal",
                            "action_support",
                            "action_source",
                            "action_destination",
                            "action_endpoint",
                            "action_edge_completion",
                            "action_edge_pair_completion",
                            "action_edge_pair_support_completion",
                            "action_edge_pair_blend_completion",
                            "action_edge_pair_multiobjective_completion",
                            "vacancy_pair_completion",
                            "vacancy_pair_energy_blend_completion",
                            "vacancy_pair_interaction_completion",
                            "vacancy_pair_interaction_energy_blend_completion",
                            "terminal_edit_decoupled",
                            "terminal_edit_inside_action_edge",
                            "terminal_edit_sequence_rollout",
                            "terminal_edit_inside_sequence_rollout",
                            "two_stage_vacancy_displacement",
                            "terminal_typed_diff",
                            "blend",
                        ],
                        help="Eval-only support restriction for projection. When enabled, only the top-k sites from this "
                             "score source may change; sites outside the top-k are forced to copy for projection.")
    parser.add_argument("--planner_projection_topk_budget", type=int, default=0,
                        help="Number of valid candidate sites kept by --planner_projection_topk_source. Default 0 disables it.")
    parser.add_argument("--planner_edge_completion_anchor_source", type=str, default="action_source",
                        choices=["change", "proposal", "action_support", "action_source", "action_destination", "action_endpoint"],
                        help="For action_edge_completion, logits used to select old/vacancy endpoint anchors.")
    parser.add_argument("--planner_edge_completion_destination_source", type=str, default="action_destination",
                        choices=["change", "proposal", "action_support", "action_source", "action_destination", "action_endpoint"],
                        help="For action_edge_completion, logits used to rank NN1 destination endpoints around selected anchors.")
    parser.add_argument("--planner_edge_completion_anchor_budget", type=int, default=32,
                        help="For action_edge_completion, number of valid anchor sites selected before NN1 completion.")
    parser.add_argument("--planner_edge_completion_destinations_per_anchor", type=int, default=8,
                        help="For action_edge_completion, max NN1 destinations kept per selected anchor. 8 keeps every BCC NN1 edge.")
    parser.add_argument("--planner_edge_completion_global_pair_budget", type=int, default=0,
                        help="Eval-only pair-level compression cap for edge-pair/sequence/two-stage sources. "
                             "Default 0 disables it; negative uses the current horizon_k as the pair budget.")
    parser.add_argument("--planner_edge_completion_destination_scope", type=str, default="nn1",
                        choices=["nn1", "global_atom"],
                        help="Destination candidate scope for two_stage_vacancy_displacement. "
                             "nn1 preserves micro-action edge completion; global_atom evaluates terminal vacancy-pair selectors.")
    parser.add_argument("--planner_edge_completion_require_vacancy_atom_pair", action="store_true",
                        help="For pair-derived projection sources, keep only KMC-legal source-vacancy -> destination-atom NN1 pairs.")
    parser.add_argument("--planner_edge_pair_multiobjective_type_weight", type=float, default=0.15,
                        help="For action_edge_pair_multiobjective_completion, additive weight for predicted moving Cu type probability.")
    parser.add_argument("--planner_edge_pair_multiobjective_order_weight", type=float, default=0.10,
                        help="For action_edge_pair_multiobjective_completion, additive weight for early path-edge order probability.")
    parser.add_argument("--planner_proposal_score_weight", type=float, default=0.0,
                        help="Optional additive planner score term log1p(proposal_support_mass) * weight.")
    parser.add_argument("--planner_candidate_quality_score_weight", type=float, default=0.0,
                        help="Optional additive planner score term from the candidate-quality head sigmoid output. "
                             "Default 0 preserves existing behavior.")
    parser.add_argument("--planner_candidate_pareto_selector_spec", type=str, default=None,
                        help="Default-off v125 deployability hook. Load a v123 Pareto selector spec and build its "
                             "45 planner-visible features from live candidates without teacher labels. Default None disables it.")
    parser.add_argument("--planner_candidate_pareto_selector_mode", type=str, default="diagnostic",
                        choices=["diagnostic", "add", "replace"],
                        help="How to use --planner_candidate_pareto_selector_spec. diagnostic records feature/score parity "
                             "without changing selection; add adds weight*score to selection_score; replace chooses by selector score.")
    parser.add_argument("--planner_candidate_pareto_selector_policy", type=str, default="pareto_balanced",
                        choices=["pareto_balanced", "recall_floor_balanced"],
                        help="Default-off v131 policy hook for candidate-budget row selection. pareto_balanced preserves "
                             "the v125/v127 behavior. recall_floor_balanced first filters rows by predicted pair_recall.")
    parser.add_argument("--planner_candidate_pareto_recall_floor", type=float, default=None,
                        help="Pair-recall floor used by --planner_candidate_pareto_selector_policy recall_floor_balanced. "
                             "Default None reads policy.pair_recall_floor from the selector spec, falling back to 0.6.")
    parser.add_argument("--planner_candidate_pareto_min_budget", type=int, default=0,
                        help="Default-off v134 guard for recall-floor selector rows. When positive, candidate-budget "
                             "rows below this pair budget are not eligible for live selector picks.")
    parser.add_argument("--planner_candidate_pareto_live_score_scale_normalize", action="store_true",
                        help="Default-off v134 diagnostic hook. Normalize live pair-score curve features to the "
                             "selector-spec training scale before making Pareto/recall-floor predictions.")
    parser.add_argument("--planner_candidate_pareto_clip_probability_predictions", action="store_true",
                        help="Default-off v134 safety hook. Clip predicted F1/precision/recall targets to [0, 1] "
                             "before selector scoring and recall-floor filtering.")
    parser.add_argument("--planner_candidate_pareto_selector_weight", type=float, default=1.0,
                        help="Weight used when --planner_candidate_pareto_selector_mode=add.")
    parser.add_argument("--planner_candidate_pareto_pair_score_field", type=str, default="score",
                        choices=[
                            "score",
                            "vacancy_score",
                            "energy_score",
                            "source_score",
                            "destination_score",
                            "endpoint_sum_score",
                            "interaction_score",
                            "moving_type_score",
                            "order_early_score",
                            "interaction_residual",
                        ],
                        help="Factorized pair-score field used for the v125 pair-score curve features. "
                             "Default score matches the live pair ranking score.")
    parser.add_argument("--planner_candidate_pareto_apply_budget_to_projection", action="store_true",
                        help="Default-off v127 hook. After the live Pareto selector predicts a per-candidate "
                             "pair budget, rerun each candidate with that budget as the projection pair-list cap. "
                             "This tests support-count pruning separately from candidate ranking.")
    parser.add_argument("--planner_candidate_pareto_teacher_label_after_budget_projection", action="store_true",
                        help="Default-off v137 diagnostic hook. When a Pareto selector budget is applied to "
                             "projection, run teacher-overlap labeling only after the budgeted projection rerun. "
                             "Requires oracle mode add with zero weight so labels cannot alter selection.")
    parser.add_argument("--planner_teacher_overlap_oracle_mode", type=str, default="none",
                        choices=["none", "add", "replace"],
                        help="Eval-only oracle diagnostic. Probe each horizon candidate with the teacher and score candidates "
                             "by selected/projected-vs-teacher changed-site overlap. 'add' adds weight*f1 to the model score; "
                             "'replace' chooses by overlap F1 only. This is not a deployable planner.")
    parser.add_argument("--planner_teacher_overlap_oracle_weight", type=float, default=1.0,
                        help="Weight used when --planner_teacher_overlap_oracle_mode=add.")
    parser.add_argument("--planner_teacher_overlap_oracle_metric", type=str, default="overlap_f1",
                        choices=[
                            "overlap_f1",
                            "teacher_reward",
                            "teacher_reward_per_tau",
                            "teacher_reward_per_sqrt_tau",
                            "overlap_reward_norm",
                        ],
                        help="Eval-only oracle metric used with --planner_teacher_overlap_oracle_mode. "
                             "Default preserves the original teacher-overlap oracle. "
                             "The *_tau variants use the probed teacher segment duration; "
                             "overlap_reward_norm averages overlap F1 with per-state min-max normalized teacher reward.")
    parser.add_argument("--planner_candidate_joint_diagnostic", action="store_true",
                        help="Default-off candidate-level diagnostic for joint ranking design. Requires "
                             "--proposal_diagnostic and a non-none teacher overlap oracle mode; records compact "
                             "per-candidate site overlap, vacancy-pair overlap, teacher reward/tau/no-op, "
                             "model reward/tau/risk, support size, and offline oracle selector summaries.")
    parser.add_argument("--planner_candidate_joint_compact_candidates", action="store_true",
                        help="When candidate joint diagnostics are enabled, store compact planner-candidate records "
                             "instead of large pair lists. Default false preserves existing JSON output.")
    parser.add_argument("--planner_vacancy_pair_rank_diagnostic", action="store_true",
                        help="Default-off v107 diagnostic. When vacancy-pair projection is used, keep compact "
                             "rank data for every scored pair long enough to report where teacher terminal "
                             "vacancy pairs fall in the pair list. Requires teacher-overlap probing to attach labels.")
    parser.add_argument("--planner_vacancy_pair_rank_max_pairs", type=int, default=0,
                        help="Maximum ranked pairs stored per candidate for v107 diagnostics. Default 0 stores all "
                             "scored pairs for exact rank diagnostics; positive values store only top-N ranks.")
    parser.add_argument("--planner_vacancy_pair_factorized_diagnostic", action="store_true",
                        help="Default-off v111 diagnostic. When vacancy-pair projection is used, store factorized "
                             "per-pair scores: source endpoint, destination endpoint, vacancy pair, energy pair, "
                             "moving-type, order, interaction residual, and final score.")
    parser.add_argument("--planner_vacancy_pair_factorized_max_pairs", type=int, default=0,
                        help="Maximum ranked pairs stored with v111 factorized diagnostics. Default 0 stores all "
                             "scored pairs for exact decomposition; positive values store only top-N ranks.")
    parser.add_argument("--reward_edit_context_source", type=str, default=None, choices=["default", "none"],
                        help="Override checkpoint reward/tau edit-context source. 'none' keeps patch+k context but zeros edit-summary features.")
    parser.add_argument("--allow_teacher_noop_segments", action="store_true",
                        help="Keep teacher macro segments whose start/end lattice state is unchanged. "
                             "By default long eval stops at the first such segment, matching the no-op "
                             "filter used by macro segment training data.")
    parser.add_argument("--proposal_diagnostic", action="store_true",
                        help="Emit per-candidate projected support summaries and selected-vs-teacher site overlap. "
                             "This is eval-only instrumentation for proposal-head design; default output is unchanged.")
    parser.add_argument("--proposal_diagnostic_max_sites", type=int, default=256,
                        help="Maximum changed/top-probability sites stored per candidate when --proposal_diagnostic is enabled.")
    parser.add_argument("--proposal_diagnostic_store_candidate_positions", action="store_true",
                        help="When --proposal_diagnostic is enabled, also store all valid candidate positions so "
                             "teacher changed-site coverage by the candidate set can be audited. Default output is unchanged.")
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
    vacancy_origin: dict[tuple[int, int, int], tuple[int, int, int]] = {
        tuple(map(int, pos.tolist())): tuple(map(int, pos.tolist()))
        for pos in start_vacancies
    }
    vacancy_order: dict[tuple[int, int, int], float] = {pos: 0.0 for pos in vacancy_origin}
    tau_exp = 0.0
    tau_real = 0.0
    reward_sum = 0.0
    done = False
    for step_idx in range(horizon_k):
        action = mod._sample_teacher_action(env, rng)
        if action is None:
            return None
        _next_obs, reward, done, info = env.step(action)
        old_pos_raw = info.get("old_pos")
        new_pos_raw = info.get("new_pos")
        if old_pos_raw is not None and new_pos_raw is not None:
            old_key = tuple(map(int, np.asarray(old_pos_raw, dtype=np.int32).tolist()))
            new_key = tuple(map(int, np.asarray(new_pos_raw, dtype=np.int32).tolist()))
            origin = vacancy_origin.pop(old_key, old_key)
            vacancy_order.pop(old_key, None)
            vacancy_origin[new_key] = origin
            vacancy_order[new_key] = float(step_idx / max(horizon_k - 1, 1))
        tau_exp += float(info["expected_delta_t"])
        tau_real += float(info["delta_t"])
        reward_sum += float(reward)
        if done:
            return None
    end_vacancies = env.env.get_vacancy_array().astype(np.int32)
    end_cu = env.env.get_cu_array().astype(np.int32)
    end_vac_set, end_cu_set = mod._positions_to_type_lookup(end_vacancies, end_cu)
    changed_positions = mod._changed_positions_between(start_vac_set, start_cu_set, end_vac_set, end_cu_set)
    vacancy_pair_positions = []
    seen_pairs: set[tuple[tuple[int, int, int], tuple[int, int, int]]] = set()
    for final_vacancy, origin in vacancy_origin.items():
        if origin == final_vacancy:
            continue
        if origin not in start_vac_set or final_vacancy not in end_vac_set:
            continue
        origin_end_type = mod._type_from_lookup(origin, end_vac_set, end_cu_set)
        final_start_type = mod._type_from_lookup(final_vacancy, start_vac_set, start_cu_set)
        if origin_end_type not in (mod.FE_TYPE, mod.CU_TYPE) or final_start_type not in (mod.FE_TYPE, mod.CU_TYPE):
            continue
        key = (origin, final_vacancy)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        vacancy_pair_positions.append(
            {
                "source": [int(v) for v in origin],
                "destination": [int(v) for v in final_vacancy],
                "moving_type": int(origin_end_type),
                "order": float(vacancy_order.get(final_vacancy, 1.0)),
            }
        )
    return {
        "tau_exp": tau_exp,
        "tau_real": tau_real,
        "reward_sum": reward_sum,
        "changed_site_count": int(len(changed_positions)),
        "is_noop": bool(len(changed_positions) == 0),
        "changed_positions": [[int(a), int(b), int(c)] for a, b, c in sorted(changed_positions)],
        "vacancy_pair_positions": vacancy_pair_positions,
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


_PARETO_BUDGETS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 512]
_PARETO_TARGETS = ("teacher_reward_sum", "reward_norm", "site_f1", "pair_precision", "pair_f1", "endpoint_f1", "pair_recall")
_PARETO_PROBABILITY_TARGETS = ("site_f1", "pair_precision", "pair_f1", "endpoint_f1", "pair_recall")
_PARETO_PAIR_SCORE_FEATURE_INDICES = tuple(range(8, 29))


def _minmax_pair(values: list[float]) -> tuple[float, float]:
    finite = [float(value) for value in values if np.isfinite(float(value))]
    if not finite:
        return 0.0, 1.0
    lo = min(finite)
    hi = max(finite)
    if abs(hi - lo) < 1e-12:
        return lo, lo + 1.0
    return lo, hi


def _minmax_norm(value: float, lo: float, hi: float) -> float:
    return float((float(value) - float(lo)) / max(float(hi) - float(lo), 1e-12))


def _score_percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    idx = min(max(int(round(float(q) * (len(values) - 1))), 0), len(values) - 1)
    return float(sorted(values)[idx])


def _score_at_rank(desc_scores: list[float], rank: int) -> float:
    if not desc_scores:
        return 0.0
    idx = min(max(int(rank) - 1, 0), len(desc_scores) - 1)
    return float(desc_scores[idx])


def _candidate_pair_scores_for_pareto(
    candidate: dict[str, object],
    *,
    pair_score_field: str,
) -> list[float]:
    diagnostic = candidate.get("vacancy_pair_projection_diagnostic")
    if not isinstance(diagnostic, dict):
        return []
    pairs = diagnostic.get("factorized_pair_scores")
    if not isinstance(pairs, list):
        return []
    scores: list[float] = []
    for item in pairs:
        if not isinstance(item, dict):
            continue
        scores.append(_as_float(item.get(pair_score_field, item.get("score", 0.0))))
    return scores


def _candidate_family_flags(candidate: dict[str, object]) -> tuple[float, float, float]:
    source = str(candidate.get("planner_projection_change_source", ""))
    anchor = str(candidate.get("planner_edge_completion_anchor_source", ""))
    destination = str(candidate.get("planner_edge_completion_destination_source", ""))
    text = " ".join([source, anchor, destination])
    diagnostic = candidate.get("vacancy_pair_projection_diagnostic")
    return (
        1.0 if "vacancy" in text else 0.0,
        1.0 if "energy" in text else 0.0,
        1.0
        if isinstance(diagnostic, dict) and isinstance(diagnostic.get("factorized_pair_scores"), list)
        else 0.0,
    )


def _pareto_curve_features(
    candidate: dict[str, object],
    *,
    candidate_index: int,
    selected_by_preselector: bool,
    pair_scores: list[float],
) -> list[float]:
    desc = sorted([float(value) for value in pair_scores], reverse=True)
    if not desc:
        desc = [0.0]
    score_mean = float(np.mean(desc))
    score_std = float(np.std(desc))
    max_score = _score_at_rank(desc, 1)
    source_is_vacancy, source_is_energy, source_is_factorized = _candidate_family_flags(candidate)
    return [
        1.0,
        float(len(pair_scores)),
        float(int(candidate.get("segment_k", 0))),
        float(int(candidate_index)),
        1.0 if selected_by_preselector else 0.0,
        source_is_vacancy,
        source_is_energy,
        source_is_factorized,
        max_score,
        _score_at_rank(desc, 2),
        _score_at_rank(desc, 4),
        _score_at_rank(desc, 8),
        _score_at_rank(desc, 16),
        _score_at_rank(desc, 32),
        _score_at_rank(desc, 64),
        _score_at_rank(desc, 128),
        _score_percentile(desc, 0.50),
        _score_percentile(desc, 0.75),
        _score_percentile(desc, 0.90),
        _score_percentile(desc, 0.95),
        _score_percentile(desc, 0.99),
        score_mean,
        score_std,
        max_score - _score_at_rank(desc, 4),
        max_score - _score_at_rank(desc, 8),
        max_score - _score_at_rank(desc, 16),
        max_score - _score_at_rank(desc, 32),
        max_score - _score_at_rank(desc, 64),
        max_score - _score_at_rank(desc, 128),
    ]


def _pareto_metric_features(base_features: list[float], budget: int, pair_count: int) -> list[float]:
    budget_value = max(int(budget), 1)
    return list(base_features) + [
        float(np.log(float(budget_value))),
        float(budget_value),
        float(budget_value / max(int(pair_count), 1)),
        float(np.sqrt(float(budget_value))),
        1.0 / float(budget_value),
    ]


def _planner_visible_v104_features_for_pareto(
    candidates: list[dict[str, object]],
    *,
    selected_by_preselector: dict[int, bool],
) -> dict[int, list[float]]:
    pre_scores = [
        _as_float(candidate.get("pre_pareto_selection_score", candidate.get("selection_score", 0.0)))
        for candidate in candidates
    ]
    reward_values = [_as_float(candidate.get("predicted_reward_sum", 0.0)) for candidate in candidates]
    delta_values = [_as_float(candidate.get("predicted_delta_e", 0.0)) for candidate in candidates]
    tau_inv_values = [1.0 / max(_as_float(candidate.get("predicted_expected_tau", 0.0)), 1e-12) for candidate in candidates]
    noop_inv_values = [1.0 - _as_float(candidate.get("predicted_noop_risk_prob", 0.0)) for candidate in candidates]
    quality_values = [_as_float(candidate.get("candidate_quality_score", 0.0)) for candidate in candidates]
    projected_inv_values = [
        1.0 / (1.0 + max(_as_float(candidate.get("projected_changed_count", 0.0)), 0.0))
        for candidate in candidates
    ]
    pair_inv_values = [
        1.0
        / (
            1.0
            + max(
                _as_float(
                    (
                        candidate.get("vacancy_pair_projection_diagnostic", {})
                        if isinstance(candidate.get("vacancy_pair_projection_diagnostic"), dict)
                        else {}
                    ).get("selected_pair_count", candidate.get("planner_edge_completion_support_count", 0.0))
                ),
                0.0,
            )
        )
        for candidate in candidates
    ]
    density_inv_values = [1.0 - _as_float(candidate.get("proposal_support_density", 0.0)) for candidate in candidates]
    k_values = [_as_float(candidate.get("segment_k", 0.0)) for candidate in candidates]
    spans = {
        "pre": _minmax_pair(pre_scores),
        "reward": _minmax_pair(reward_values),
        "delta": _minmax_pair(delta_values),
        "tau": _minmax_pair(tau_inv_values),
        "noop": _minmax_pair(noop_inv_values),
        "quality": _minmax_pair(quality_values),
        "projected": _minmax_pair(projected_inv_values),
        "pair": _minmax_pair(pair_inv_values),
        "density": _minmax_pair(density_inv_values),
        "k": _minmax_pair(k_values),
    }
    features: dict[int, list[float]] = {}
    for idx, candidate in enumerate(candidates):
        features[id(candidate)] = [
            1.0,
            _minmax_norm(pre_scores[idx], *spans["pre"]),
            _minmax_norm(reward_values[idx], *spans["reward"]),
            _minmax_norm(delta_values[idx], *spans["delta"]),
            _minmax_norm(tau_inv_values[idx], *spans["tau"]),
            _minmax_norm(noop_inv_values[idx], *spans["noop"]),
            _minmax_norm(quality_values[idx], *spans["quality"]),
            _minmax_norm(projected_inv_values[idx], *spans["projected"]),
            _minmax_norm(pair_inv_values[idx], *spans["pair"]),
            _minmax_norm(density_inv_values[idx], *spans["density"]),
            _minmax_norm(k_values[idx], *spans["k"]),
        ]
    return features


def _load_pareto_selector_spec(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if int(data.get("feature_dim", 0)) != 45:
        raise ValueError(f"v125 Pareto selector expects feature_dim=45, got {data.get('feature_dim')}")
    models = data.get("models")
    if not isinstance(models, dict):
        raise ValueError("v125 Pareto selector spec is missing models")
    for target in _PARETO_TARGETS:
        model = models.get(target)
        if not isinstance(model, dict):
            raise ValueError(f"v125 Pareto selector spec is missing model target {target}")
        if len(model.get("mean", [])) != 45 or len(model.get("std", [])) != 45 or len(model.get("weights", [])) != 46:
            raise ValueError(f"v125 Pareto selector model {target} has incompatible dimensions")
    return data


def _predict_pareto_target(spec: dict[str, object], target: str, features: list[float]) -> float:
    model = (spec.get("models") or {}).get(target)
    if not isinstance(model, dict):
        return 0.0
    mean = [float(value) for value in model.get("mean", [])]
    std = [max(float(value), 1e-12) for value in model.get("std", [])]
    weights = [float(value) for value in model.get("weights", [])]
    if len(features) != len(mean) or len(weights) != len(mean) + 1:
        return 0.0
    normalized = [(float(value) - mean[idx]) / std[idx] for idx, value in enumerate(features)] + [1.0]
    raw = float(sum(weight * value for weight, value in zip(weights, normalized)))
    return raw * _as_float(model.get("target_std", 0.0)) + _as_float(model.get("target_mean", 0.0))


def _pareto_feature_reference_stats(selector_spec: dict[str, object]) -> tuple[list[float], list[float]]:
    models = selector_spec.get("models")
    if not isinstance(models, dict):
        return [0.0] * 45, [1.0] * 45
    for target in _PARETO_TARGETS:
        model = models.get(target)
        if not isinstance(model, dict):
            continue
        mean = [float(value) for value in model.get("mean", [])]
        std = [max(float(value), 1e-12) for value in model.get("std", [])]
        if len(mean) == 45 and len(std) == 45:
            return mean, std
    return [0.0] * 45, [1.0] * 45


def _normalize_live_pair_score_features(
    rows: list[dict[str, object]],
    *,
    selector_spec: dict[str, object],
) -> dict[str, object]:
    if not rows:
        return {"applied": False, "reason": "no_rows"}
    reference_mean, reference_std = _pareto_feature_reference_stats(selector_spec)
    row_features = [
        row.get("features")
        for row in rows
        if isinstance(row.get("features"), list) and len(row.get("features", [])) >= 45
    ]
    if not row_features:
        return {"applied": False, "reason": "no_features"}
    live_mean: dict[int, float] = {}
    live_std: dict[int, float] = {}
    for idx in _PARETO_PAIR_SCORE_FEATURE_INDICES:
        values = [float(features[idx]) for features in row_features if len(features) > idx]
        if not values:
            continue
        live_mean[idx] = float(np.mean(values))
        live_std[idx] = max(float(np.std(values)), 1e-12)
    for row in rows:
        features_raw = row.get("features")
        if not isinstance(features_raw, list):
            continue
        features = [float(value) for value in features_raw]
        for idx in _PARETO_PAIR_SCORE_FEATURE_INDICES:
            if idx >= len(features) or idx not in live_mean:
                continue
            features[idx] = float(
                reference_mean[idx] + (features[idx] - live_mean[idx]) * (reference_std[idx] / live_std[idx])
            )
        row["features"] = features
    tracked = {}
    for idx in (8, 21, 22):
        if idx in live_mean:
            tracked[str(idx)] = {
                "live_mean": float(live_mean[idx]),
                "live_std": float(live_std[idx]),
                "reference_mean": float(reference_mean[idx]),
                "reference_std": float(reference_std[idx]),
            }
    return {"applied": True, "feature_indices": list(_PARETO_PAIR_SCORE_FEATURE_INDICES), "tracked": tracked}


def _clip_pareto_probability_predictions(predictions: dict[str, float]) -> dict[str, float]:
    clipped = dict(predictions)
    for target in _PARETO_PROBABILITY_TARGETS:
        clipped[target] = float(np.clip(_as_float(clipped.get(target, 0.0)), 0.0, 1.0))
    return clipped


def _pareto_prediction_values(row: dict[str, object], stats: dict[str, tuple[float, float]]) -> dict[str, float]:
    preds = row.get("predictions", {})
    if not isinstance(preds, dict):
        preds = {}
    reward_raw = _minmax_norm(_as_float(preds.get("teacher_reward_sum", 0.0)), *stats["teacher_reward_sum"])
    reward_rel = _minmax_norm(_as_float(preds.get("reward_norm", 0.0)), *stats["reward_norm"])
    return {
        "reward": 0.5 * reward_raw + 0.5 * reward_rel,
        "site": _minmax_norm(_as_float(preds.get("site_f1", 0.0)), *stats["site_f1"]),
        "pair_precision": _minmax_norm(_as_float(preds.get("pair_precision", 0.0)), *stats["pair_precision"]),
        "pair_f1": _minmax_norm(_as_float(preds.get("pair_f1", 0.0)), *stats["pair_f1"]),
        "endpoint": _minmax_norm(_as_float(preds.get("endpoint_f1", 0.0)), *stats["endpoint_f1"]),
    }


def _pareto_selector_score(row: dict[str, object], stats: dict[str, tuple[float, float]]) -> float:
    values = _pareto_prediction_values(row, stats)
    return float(
        0.24 * values["reward"]
        + 0.22 * values["site"]
        + 0.22 * values["pair_f1"]
        + 0.18 * values["pair_precision"]
        + 0.14 * values["endpoint"]
    )


def _pareto_recall_floor_from_spec(selector_spec: dict[str, object], override: float | None) -> float:
    if override is not None:
        return float(override)
    policy = selector_spec.get("policy")
    if isinstance(policy, dict) and "pair_recall_floor" in policy:
        return float(policy.get("pair_recall_floor", 0.6))
    return 0.6


def _pareto_dominates(a: dict[str, float], b: dict[str, float]) -> bool:
    keys = ("reward", "site", "pair_precision", "pair_f1", "endpoint")
    return all(a[key] >= b[key] - 1e-12 for key in keys) and any(a[key] > b[key] + 1e-12 for key in keys)


def _pareto_front_rows(rows: list[dict[str, object]], stats: dict[str, tuple[float, float]]) -> list[dict[str, object]]:
    values = {id(row): _pareto_prediction_values(row, stats) for row in rows}
    front = [
        row
        for row in rows
        if not any(_pareto_dominates(values[id(other)], values[id(row)]) for other in rows if other is not row)
    ]
    return front or rows


def _pareto_policy_pool(
    rows: list[dict[str, object]],
    *,
    stats: dict[str, tuple[float, float]],
    selector_policy: str,
    recall_floor: float,
    min_budget: int = 0,
) -> list[dict[str, object]]:
    min_budget = max(int(min_budget), 0)
    if selector_policy == "recall_floor_balanced":
        eligible = [
            row
            for row in rows
            if _as_float(row.get("predictions", {}).get("pair_recall", 0.0)) >= float(recall_floor)
            and (min_budget <= 0 or int(row.get("budget", 0)) >= min_budget)
        ]
        if eligible:
            return eligible
        if min_budget > 0:
            budget_guarded = [row for row in rows if int(row.get("budget", 0)) >= min_budget]
            if budget_guarded:
                return budget_guarded
        return rows
    front = _pareto_front_rows(rows, stats)
    if min_budget > 0:
        guarded_front = [row for row in front if int(row.get("budget", 0)) >= min_budget]
        if guarded_front:
            return guarded_front
        budget_guarded = [row for row in rows if int(row.get("budget", 0)) >= min_budget]
        if budget_guarded:
            return budget_guarded
    return front


def _pick_pareto_policy_row(
    rows: list[dict[str, object]],
    *,
    stats: dict[str, tuple[float, float]],
    selector_policy: str,
    recall_floor: float,
    min_budget: int = 0,
) -> dict[str, object]:
    pool = _pareto_policy_pool(
        rows,
        stats=stats,
        selector_policy=selector_policy,
        recall_floor=recall_floor,
        min_budget=min_budget,
    )
    return max(
        pool,
        key=lambda row: (
            _as_float(row.get("pareto_selector_score", 0.0)),
            -_as_float(row.get("selected_pair_count", 0.0)),
        ),
    )


def _apply_candidate_pareto_selector(
    candidates: list[dict[str, object]],
    *,
    selector_spec: dict[str, object] | None,
    mode: str,
    weight: float,
    pair_score_field: str,
    selector_policy: str = "pareto_balanced",
    recall_floor: float | None = None,
    min_budget: int = 0,
    live_score_scale_normalize: bool = False,
    clip_probability_predictions: bool = False,
    min_projected_changed_sites: int,
) -> dict[str, object]:
    stats: dict[str, object] = {
        "enabled": bool(selector_spec is not None),
        "mode": mode,
        "selector_policy": selector_policy,
        "min_budget": max(int(min_budget), 0),
        "live_score_scale_normalize": bool(live_score_scale_normalize),
        "clip_probability_predictions": bool(clip_probability_predictions),
        "candidate_count": int(len(candidates)),
        "legal_candidate_count": 0,
        "feature_row_count": 0,
        "missing_pair_score_candidate_count": 0,
        "applied": False,
    }
    if selector_spec is None or not candidates:
        return stats
    effective_recall_floor = _pareto_recall_floor_from_spec(selector_spec, recall_floor)
    stats["recall_floor"] = float(effective_recall_floor)
    for candidate in candidates:
        candidate["pre_pareto_selection_score"] = _as_float(candidate.get("selection_score", -float("inf")))
    preselected = _choose_planner_candidate(candidates, min_projected_changed_sites=min_projected_changed_sites)
    legal = [
        candidate
        for candidate in candidates
        if _as_float(candidate.get("reachability_violation", 1.0)) <= 0.0
        and _as_float(candidate.get("projected_changed_count", 0.0)) >= float(min_projected_changed_sites)
    ]
    stats["legal_candidate_count"] = int(len(legal))
    if not legal:
        return stats
    selected_by_preselector = {id(candidate): bool(candidate is preselected) for candidate in legal}
    v104_features = _planner_visible_v104_features_for_pareto(legal, selected_by_preselector=selected_by_preselector)
    rows: list[dict[str, object]] = []
    for candidate_index, candidate in enumerate(legal):
        pair_scores = _candidate_pair_scores_for_pareto(candidate, pair_score_field=pair_score_field)
        if not pair_scores:
            stats["missing_pair_score_candidate_count"] = int(stats["missing_pair_score_candidate_count"]) + 1
            continue
        base = _pareto_curve_features(
            candidate,
            candidate_index=candidate_index,
            selected_by_preselector=selected_by_preselector.get(id(candidate), False),
            pair_scores=pair_scores,
        )
        extra = v104_features.get(id(candidate), [0.0] * 11)
        for budget in _PARETO_BUDGETS:
            features = _pareto_metric_features(base, int(budget), len(pair_scores)) + extra
            rows.append(
                {
                    "candidate": candidate,
                    "candidate_index": int(candidate_index),
                    "budget": int(budget),
                    "features": features,
                    "selected_pair_count": float(min(int(budget), len(pair_scores))),
                }
            )
    stats["feature_row_count"] = int(len(rows))
    if not rows:
        return stats
    if live_score_scale_normalize:
        stats["score_scale_normalization"] = _normalize_live_pair_score_features(rows, selector_spec=selector_spec)
    else:
        stats["score_scale_normalization"] = {"applied": False}
    for row in rows:
        features = row.get("features", [])
        predictions = {
            target: _predict_pareto_target(selector_spec, target, features if isinstance(features, list) else [])
            for target in _PARETO_TARGETS
        }
        if clip_probability_predictions:
            predictions = _clip_pareto_probability_predictions(predictions)
        row["predictions"] = predictions
    prediction_stats = {
        target: _minmax_pair([
            _as_float(row.get("predictions", {}).get(target, 0.0))
            for row in rows
            if isinstance(row.get("predictions"), dict)
        ])
        for target in _PARETO_TARGETS
    }
    for row in rows:
        row["pareto_selector_score"] = _pareto_selector_score(row, prediction_stats)
    best_by_candidate: dict[int, dict[str, object]] = {}
    for row in rows:
        candidate = row.get("candidate")
        if not isinstance(candidate, dict):
            continue
        key = id(candidate)
        current = best_by_candidate.get(key)
        candidate_rows = [item for item in rows if item.get("candidate") is candidate]
        picked_for_candidate = _pick_pareto_policy_row(
            candidate_rows,
            stats=prediction_stats,
            selector_policy=selector_policy,
            recall_floor=effective_recall_floor,
            min_budget=max(int(min_budget), 0),
        )
        if current is None or picked_for_candidate is row:
            best_by_candidate[key] = picked_for_candidate
    for row in best_by_candidate.values():
        candidate = row.get("candidate")
        if not isinstance(candidate, dict):
            continue
        score = _as_float(row.get("pareto_selector_score", 0.0))
        candidate["planner_candidate_pareto_selector"] = {
            "mode": mode,
            "selector_policy": selector_policy,
            "score": float(score),
            "budget": int(row["budget"]),
            "candidate_index": int(row["candidate_index"]),
            "predictions": row["predictions"],
            "feature_dim": int(len(row["features"])),
            "pair_score_field": pair_score_field,
            "recall_floor": float(effective_recall_floor),
            "min_budget": max(int(min_budget), 0),
            "min_budget_passed": int(row.get("budget", 0)) >= max(int(min_budget), 0),
            "live_score_scale_normalize": bool(live_score_scale_normalize),
            "clip_probability_predictions": bool(clip_probability_predictions),
            "pair_recall_floor_passed": (
                _as_float(row.get("predictions", {}).get("pair_recall", 0.0)) >= float(effective_recall_floor)
            ),
            "budget_applied_to_projection": False,
            "teacher_label_fields_used": False,
        }
        if mode == "replace":
            candidate["selection_score"] = float(score)
        elif mode == "add":
            candidate["selection_score"] = _as_float(candidate.get("selection_score", -float("inf"))) + float(weight) * score
    if mode in {"replace", "add"}:
        stats["applied"] = True
    picked = _pick_pareto_policy_row(
        rows,
        stats=prediction_stats,
        selector_policy=selector_policy,
        recall_floor=effective_recall_floor,
        min_budget=max(int(min_budget), 0),
    )
    picked_score = _as_float(picked.get("pareto_selector_score", 0.0))
    picked_candidate = picked["candidate"]
    stats.update(
        {
            "selected_candidate_index": int(picked["candidate_index"]),
            "selected_budget": int(picked["budget"]),
            "selected_score": float(picked_score),
            "selected_prediction_teacher_reward_sum": _as_float(picked["predictions"].get("teacher_reward_sum", 0.0)),
            "selected_prediction_site_f1": _as_float(picked["predictions"].get("site_f1", 0.0)),
            "selected_prediction_pair_precision": _as_float(picked["predictions"].get("pair_precision", 0.0)),
            "selected_prediction_pair_recall": _as_float(picked["predictions"].get("pair_recall", 0.0)),
            "selected_prediction_pair_f1": _as_float(picked["predictions"].get("pair_f1", 0.0)),
            "selected_prediction_endpoint_f1": _as_float(picked["predictions"].get("endpoint_f1", 0.0)),
            "selected_pair_recall_floor_passed": (
                _as_float(picked["predictions"].get("pair_recall", 0.0)) >= float(effective_recall_floor)
            ),
            "selected_min_budget_passed": int(picked.get("budget", 0)) >= max(int(min_budget), 0),
            "budget_applied_to_projection": False,
        }
    )
    return stats


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
    planner_noop_risk_penalty: float = 0.0,
    noop_risk_prob: float = 0.0,
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
    if planner_noop_risk_penalty > 0.0:
        risk = float(np.clip(noop_risk_prob, 0.0, 1.0))
        score -= float(planner_noop_risk_penalty) * risk * max(abs(float(score)), 1.0)
    return float(score), float(tau_for_score)


def _projection_logits_from_source(
    *,
    change_logits: torch.Tensor,
    proposal_logits: torch.Tensor,
    action_support_logits: torch.Tensor | None = None,
    action_source_logits: torch.Tensor | None = None,
    action_destination_logits: torch.Tensor | None = None,
    source: str,
    blend_alpha: float,
) -> torch.Tensor:
    if source == "proposal":
        return proposal_logits
    if source == "action_support":
        return action_support_logits if action_support_logits is not None else proposal_logits
    if source == "action_source":
        return action_source_logits if action_source_logits is not None else (
            action_support_logits if action_support_logits is not None else proposal_logits
        )
    if source == "action_destination":
        return action_destination_logits if action_destination_logits is not None else (
            action_support_logits if action_support_logits is not None else proposal_logits
        )
    if source == "action_endpoint":
        if action_source_logits is not None and action_destination_logits is not None:
            return mod.combine_action_endpoint_logits(action_source_logits, action_destination_logits)
        return action_support_logits if action_support_logits is not None else proposal_logits
    if source == "blend":
        alpha = float(np.clip(blend_alpha, 0.0, 1.0))
        return (1.0 - alpha) * change_logits + alpha * proposal_logits
    return change_logits


def _periodic_neighbor_position(pos: np.ndarray, offset: np.ndarray, box: np.ndarray) -> tuple[int, int, int]:
    return tuple(((pos.astype(np.int64) + offset.astype(np.int64)) % box.astype(np.int64)).tolist())


def _resolve_global_pair_budget(global_pair_budget: int, horizon_k: torch.Tensor, batch_idx: int) -> int:
    budget = int(global_pair_budget)
    if budget < 0:
        return max(int(horizon_k[batch_idx].item()), 0)
    return max(budget, 0)


def _is_valid_vacancy_atom_pair(
    current_types: torch.Tensor,
    batch_idx: int,
    source_idx: int,
    dest_idx: int,
    *,
    require_vacancy_atom_pair: bool,
) -> bool:
    if not bool(require_vacancy_atom_pair):
        return True
    source_type = int(current_types[batch_idx, source_idx].item())
    dest_type = int(current_types[batch_idx, dest_idx].item())
    return source_type == mod.V_TYPE and dest_type in (mod.FE_TYPE, mod.CU_TYPE)


def _action_edge_completion_logits(
    *,
    change_logits: torch.Tensor,
    proposal_logits: torch.Tensor,
    action_support_logits: torch.Tensor | None,
    action_source_logits: torch.Tensor | None,
    action_destination_logits: torch.Tensor | None,
    candidate_positions: torch.Tensor,
    candidate_mask: torch.Tensor,
    box_dims: torch.Tensor,
    nn1_offsets: np.ndarray,
    anchor_source: str,
    destination_source: str,
    anchor_budget: int,
    destinations_per_anchor: int,
    blend_alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    anchor_logits = _projection_logits_from_source(
        change_logits=change_logits,
        proposal_logits=proposal_logits,
        action_support_logits=action_support_logits,
        action_source_logits=action_source_logits,
        action_destination_logits=action_destination_logits,
        source=anchor_source,
        blend_alpha=blend_alpha,
    )
    destination_logits = _projection_logits_from_source(
        change_logits=change_logits,
        proposal_logits=proposal_logits,
        action_support_logits=action_support_logits,
        action_source_logits=action_source_logits,
        action_destination_logits=action_destination_logits,
        source=destination_source,
        blend_alpha=blend_alpha,
    )
    completion_logits = torch.full_like(change_logits, -20.0)
    completion_mask = torch.zeros_like(candidate_mask, dtype=torch.bool)
    anchor_count = max(int(anchor_budget), 0)
    per_anchor = max(int(destinations_per_anchor), 0)
    if anchor_count <= 0:
        return completion_logits, completion_mask

    nn1 = np.asarray(nn1_offsets, dtype=np.int64)
    for batch_idx in range(candidate_mask.shape[0]):
        valid_idx = torch.nonzero(candidate_mask[batch_idx] > 0, as_tuple=False).squeeze(-1)
        if valid_idx.numel() == 0:
            continue
        sample_anchor_count = min(anchor_count, int(valid_idx.numel()))
        anchor_local = torch.topk(anchor_logits[batch_idx, valid_idx], k=sample_anchor_count).indices
        anchor_idx = valid_idx[anchor_local]
        positions_np = torch.round(candidate_positions[batch_idx]).detach().cpu().numpy().astype(np.int64)
        box_np = torch.round(box_dims[batch_idx]).detach().cpu().numpy().astype(np.int64)
        lookup = {
            tuple(int(x) for x in positions_np[int(idx)].tolist()): int(idx)
            for idx in valid_idx.detach().cpu().numpy().tolist()
        }
        for anchor in anchor_idx.detach().cpu().numpy().tolist():
            anchor = int(anchor)
            completion_mask[batch_idx, anchor] = True
            completion_logits[batch_idx, anchor] = torch.maximum(
                completion_logits[batch_idx, anchor],
                anchor_logits[batch_idx, anchor],
            )
            anchor_pos = positions_np[anchor]
            dest_indices: list[int] = []
            for offset in nn1:
                dest_idx = lookup.get(_periodic_neighbor_position(anchor_pos, offset, box_np))
                if dest_idx is not None:
                    dest_indices.append(int(dest_idx))
            if per_anchor > 0 and len(dest_indices) > per_anchor:
                dest_tensor = torch.tensor(dest_indices, dtype=torch.long, device=destination_logits.device)
                keep_local = torch.topk(destination_logits[batch_idx, dest_tensor], k=per_anchor).indices
                dest_indices = [int(dest_tensor[int(i)].item()) for i in keep_local.detach().cpu().tolist()]
            for dest_idx in dest_indices:
                completion_mask[batch_idx, dest_idx] = True
                completion_logits[batch_idx, dest_idx] = torch.maximum(
                    completion_logits[batch_idx, dest_idx],
                    destination_logits[batch_idx, dest_idx],
                )
    return completion_logits, completion_mask


def _action_edge_pair_completion_logits(
    *,
    model: mod.MacroDreamerEditModel,
    site_latent: torch.Tensor,
    patch_latent: torch.Tensor,
    predicted_next_global: torch.Tensor,
    path_latent: torch.Tensor,
    horizon_k: torch.Tensor,
    current_types: torch.Tensor,
    change_logits: torch.Tensor,
    proposal_logits: torch.Tensor,
    action_support_logits: torch.Tensor | None,
    action_source_logits: torch.Tensor | None,
    action_destination_logits: torch.Tensor | None,
    candidate_positions: torch.Tensor,
    candidate_mask: torch.Tensor,
    box_dims: torch.Tensor,
    nn1_offsets: np.ndarray,
    anchor_source: str,
    anchor_budget: int,
    destinations_per_anchor: int,
    global_pair_budget: int,
    destination_scope: str,
    blend_alpha: float,
    score_source: str = "energy",
    support_blend_alpha: float = 0.5,
    multiobjective_type_weight: float = 0.15,
    multiobjective_order_weight: float = 0.10,
    require_vacancy_atom_pair: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    anchor_logits = _projection_logits_from_source(
        change_logits=change_logits,
        proposal_logits=proposal_logits,
        action_support_logits=action_support_logits,
        action_source_logits=action_source_logits,
        action_destination_logits=action_destination_logits,
        source=anchor_source,
        blend_alpha=blend_alpha,
    )
    completion_logits = torch.full_like(change_logits, -20.0)
    completion_mask = torch.zeros_like(candidate_mask, dtype=torch.bool)
    if not hasattr(model, "decode_action_edge_pairs"):
        return completion_logits, completion_mask
    anchor_count = max(int(anchor_budget), 0)
    per_anchor = max(int(destinations_per_anchor), 0)
    if anchor_count <= 0 or per_anchor <= 0:
        return completion_logits, completion_mask

    nn1 = np.asarray(nn1_offsets, dtype=np.int64)
    for batch_idx in range(candidate_mask.shape[0]):
        valid_idx = torch.nonzero(candidate_mask[batch_idx] > 0, as_tuple=False).squeeze(-1)
        if valid_idx.numel() == 0:
            continue
        sample_anchor_count = min(anchor_count, int(valid_idx.numel()))
        anchor_local = torch.topk(anchor_logits[batch_idx, valid_idx], k=sample_anchor_count).indices
        anchor_idx = valid_idx[anchor_local]
        positions_np = torch.round(candidate_positions[batch_idx]).detach().cpu().numpy().astype(np.int64)
        box_np = torch.round(box_dims[batch_idx]).detach().cpu().numpy().astype(np.int64)
        lookup = {
            tuple(int(x) for x in positions_np[int(idx)].tolist()): int(idx)
            for idx in valid_idx.detach().cpu().numpy().tolist()
        }
        pair_entries: list[tuple[int, int, int]] = []
        use_global_atom_destinations = str(destination_scope).strip().lower() == "global_atom"
        global_destinations: list[int] = []
        if use_global_atom_destinations:
            for dest_idx in valid_idx.detach().cpu().numpy().tolist():
                dest_idx = int(dest_idx)
                dest_type = int(current_types[batch_idx, dest_idx].item())
                if dest_type in (mod.FE_TYPE, mod.CU_TYPE):
                    global_destinations.append(dest_idx)
        for anchor in anchor_idx.detach().cpu().numpy().tolist():
            anchor = int(anchor)
            if use_global_atom_destinations:
                destinations = global_destinations
            else:
                anchor_pos = positions_np[anchor]
                destinations = []
                for offset in nn1:
                    dest_idx = lookup.get(_periodic_neighbor_position(anchor_pos, offset, box_np))
                    if dest_idx is not None:
                        destinations.append(int(dest_idx))
            for dest_idx in destinations:
                if int(dest_idx) == anchor:
                    continue
                if not _is_valid_vacancy_atom_pair(
                    current_types,
                    batch_idx,
                    anchor,
                    int(dest_idx),
                    require_vacancy_atom_pair=require_vacancy_atom_pair,
                ):
                    continue
                pair_entries.append((anchor, int(dest_idx), len(pair_entries)))
        if not pair_entries:
            continue
        pair_tensor = torch.tensor(
            [[anchor, dest] for anchor, dest, _ in pair_entries],
            dtype=torch.long,
            device=change_logits.device,
        ).unsqueeze(0)
        energy_pair_scores = model.decode_action_edge_pairs(
            site_latent=site_latent[batch_idx : batch_idx + 1],
            patch_latent=patch_latent[batch_idx : batch_idx + 1],
            predicted_next_global=predicted_next_global[batch_idx : batch_idx + 1],
            path_latent=path_latent[batch_idx : batch_idx + 1],
            horizon_k=horizon_k[batch_idx : batch_idx + 1],
            current_types=current_types[batch_idx : batch_idx + 1],
            edge_pair_indices=pair_tensor,
        )[0]
        support_pair_scores = torch.zeros_like(energy_pair_scores)
        if score_source in {"support", "blend", "multiobjective"} and hasattr(model, "decode_action_edge_pair_support"):
            support_pair_scores = model.decode_action_edge_pair_support(
                site_latent=site_latent[batch_idx : batch_idx + 1],
                patch_latent=patch_latent[batch_idx : batch_idx + 1],
                predicted_next_global=predicted_next_global[batch_idx : batch_idx + 1],
                path_latent=path_latent[batch_idx : batch_idx + 1],
                horizon_k=horizon_k[batch_idx : batch_idx + 1],
                current_types=current_types[batch_idx : batch_idx + 1],
                edge_pair_indices=pair_tensor,
            )[0]
        pair_scores = energy_pair_scores
        if score_source == "multiobjective":
            alpha = float(np.clip(support_blend_alpha, 0.0, 1.0))
            pair_scores = (1.0 - alpha) * energy_pair_scores + alpha * support_pair_scores
            if hasattr(model, "decode_action_edge_pair_moving_type"):
                moving_type_logits = model.decode_action_edge_pair_moving_type(
                    site_latent=site_latent[batch_idx : batch_idx + 1],
                    patch_latent=patch_latent[batch_idx : batch_idx + 1],
                    predicted_next_global=predicted_next_global[batch_idx : batch_idx + 1],
                    path_latent=path_latent[batch_idx : batch_idx + 1],
                    horizon_k=horizon_k[batch_idx : batch_idx + 1],
                    current_types=current_types[batch_idx : batch_idx + 1],
                    edge_pair_indices=pair_tensor,
                )[0]
                moving_type_score = torch.softmax(moving_type_logits, dim=-1)[:, mod.CU_TYPE]
                pair_scores = pair_scores + float(multiobjective_type_weight) * moving_type_score
            if hasattr(model, "decode_action_edge_pair_order"):
                order_logits = model.decode_action_edge_pair_order(
                    site_latent=site_latent[batch_idx : batch_idx + 1],
                    patch_latent=patch_latent[batch_idx : batch_idx + 1],
                    predicted_next_global=predicted_next_global[batch_idx : batch_idx + 1],
                    path_latent=path_latent[batch_idx : batch_idx + 1],
                    horizon_k=horizon_k[batch_idx : batch_idx + 1],
                    current_types=current_types[batch_idx : batch_idx + 1],
                    edge_pair_indices=pair_tensor,
                )[0]
                early_order_score = 1.0 - torch.sigmoid(order_logits)
                pair_scores = pair_scores + float(multiobjective_order_weight) * early_order_score
        elif score_source in {"support", "blend"}:
            if score_source == "support":
                pair_scores = support_pair_scores
            else:
                alpha = float(np.clip(support_blend_alpha, 0.0, 1.0))
                pair_scores = (1.0 - alpha) * energy_pair_scores + alpha * support_pair_scores
        by_anchor: dict[int, list[tuple[int, int]]] = {}
        for anchor, dest, pair_idx in pair_entries:
            by_anchor.setdefault(anchor, []).append((dest, pair_idx))
        selected_pairs: list[tuple[int, int, int, float]] = []
        for _anchor, dest_pairs in by_anchor.items():
            if len(dest_pairs) > per_anchor:
                score_idx = torch.tensor([pair_idx for _, pair_idx in dest_pairs], dtype=torch.long, device=change_logits.device)
                keep_local = torch.topk(pair_scores[score_idx], k=per_anchor).indices.detach().cpu().tolist()
                dest_pairs = [dest_pairs[int(i)] for i in keep_local]
            for dest_idx, pair_idx in dest_pairs:
                source_idx = int(pair_entries[int(pair_idx)][0])
                selected_pairs.append((source_idx, int(dest_idx), int(pair_idx), float(pair_scores[int(pair_idx)].item())))
        pair_budget = _resolve_global_pair_budget(global_pair_budget, horizon_k, batch_idx)
        if pair_budget > 0 and len(selected_pairs) > pair_budget:
            selected_pairs = sorted(selected_pairs, key=lambda item: item[3], reverse=True)[:pair_budget]
        for source_idx, dest_idx, pair_idx, _score in selected_pairs:
            completion_mask[batch_idx, source_idx] = True
            completion_logits[batch_idx, source_idx] = torch.maximum(
                completion_logits[batch_idx, source_idx],
                pair_scores[int(pair_idx)],
            )
            completion_mask[batch_idx, dest_idx] = True
            completion_logits[batch_idx, dest_idx] = torch.maximum(
                completion_logits[batch_idx, dest_idx],
                pair_scores[int(pair_idx)],
            )
    return completion_logits, completion_mask


def _edge_pair_completion_score_source(source: str) -> str:
    if source == "action_edge_pair_support_completion":
        return "support"
    if source == "action_edge_pair_blend_completion":
        return "blend"
    if source == "action_edge_pair_multiobjective_completion":
        return "multiobjective"
    return "energy"


def _action_edge_pair_vacancy_displacement_logits(
    *,
    model: mod.MacroDreamerEditModel,
    site_latent: torch.Tensor,
    patch_latent: torch.Tensor,
    predicted_next_global: torch.Tensor,
    path_latent: torch.Tensor,
    horizon_k: torch.Tensor,
    current_types: torch.Tensor,
    raw_type_logits: torch.Tensor,
    change_logits: torch.Tensor,
    proposal_logits: torch.Tensor,
    action_support_logits: torch.Tensor | None,
    action_source_logits: torch.Tensor | None,
    action_destination_logits: torch.Tensor | None,
    candidate_positions: torch.Tensor,
    candidate_mask: torch.Tensor,
    box_dims: torch.Tensor,
    nn1_offsets: np.ndarray,
    anchor_source: str,
    anchor_budget: int,
    destinations_per_anchor: int,
    global_pair_budget: int,
    blend_alpha: float,
    support_blend_alpha: float,
    multiobjective_type_weight: float,
    multiobjective_order_weight: float,
    require_vacancy_atom_pair: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    anchor_logits = _projection_logits_from_source(
        change_logits=change_logits,
        proposal_logits=proposal_logits,
        action_support_logits=action_support_logits,
        action_source_logits=action_source_logits,
        action_destination_logits=action_destination_logits,
        source=anchor_source,
        blend_alpha=blend_alpha,
    )
    displacement_logits = torch.full_like(change_logits, -20.0)
    displacement_type_logits = torch.full_like(raw_type_logits, -20.0)
    displacement_type_logits.scatter_(2, current_types.unsqueeze(-1), 20.0)
    displacement_mask = torch.zeros_like(candidate_mask, dtype=torch.bool)
    if not hasattr(model, "decode_action_edge_pairs"):
        return displacement_logits, displacement_type_logits, displacement_mask

    anchor_count = max(int(anchor_budget), 0)
    per_anchor = max(int(destinations_per_anchor), 0)
    if anchor_count <= 0 or per_anchor <= 0:
        return displacement_logits, displacement_type_logits, displacement_mask

    nn1 = np.asarray(nn1_offsets, dtype=np.int64)
    for batch_idx in range(candidate_mask.shape[0]):
        valid_idx = torch.nonzero(candidate_mask[batch_idx] > 0, as_tuple=False).squeeze(-1)
        if valid_idx.numel() == 0:
            continue
        sample_anchor_count = min(anchor_count, int(valid_idx.numel()))
        anchor_local = torch.topk(anchor_logits[batch_idx, valid_idx], k=sample_anchor_count).indices
        anchor_idx = valid_idx[anchor_local]
        positions_np = torch.round(candidate_positions[batch_idx]).detach().cpu().numpy().astype(np.int64)
        box_np = torch.round(box_dims[batch_idx]).detach().cpu().numpy().astype(np.int64)
        lookup = {
            tuple(int(x) for x in positions_np[int(idx)].tolist()): int(idx)
            for idx in valid_idx.detach().cpu().numpy().tolist()
        }
        pair_entries: list[tuple[int, int, int]] = []
        for anchor in anchor_idx.detach().cpu().numpy().tolist():
            anchor = int(anchor)
            anchor_pos = positions_np[anchor]
            for offset in nn1:
                dest_idx = lookup.get(_periodic_neighbor_position(anchor_pos, offset, box_np))
                if dest_idx is not None:
                    if not _is_valid_vacancy_atom_pair(
                        current_types,
                        batch_idx,
                        anchor,
                        int(dest_idx),
                        require_vacancy_atom_pair=require_vacancy_atom_pair,
                    ):
                        continue
                    pair_entries.append((anchor, int(dest_idx), len(pair_entries)))
        if not pair_entries:
            continue

        pair_tensor = torch.tensor(
            [[anchor, dest] for anchor, dest, _ in pair_entries],
            dtype=torch.long,
            device=change_logits.device,
        ).unsqueeze(0)
        energy_pair_scores = model.decode_action_edge_pairs(
            site_latent=site_latent[batch_idx : batch_idx + 1],
            patch_latent=patch_latent[batch_idx : batch_idx + 1],
            predicted_next_global=predicted_next_global[batch_idx : batch_idx + 1],
            path_latent=path_latent[batch_idx : batch_idx + 1],
            horizon_k=horizon_k[batch_idx : batch_idx + 1],
            current_types=current_types[batch_idx : batch_idx + 1],
            edge_pair_indices=pair_tensor,
        )[0]
        support_pair_scores = torch.zeros_like(energy_pair_scores)
        if hasattr(model, "decode_action_edge_pair_support"):
            support_pair_scores = model.decode_action_edge_pair_support(
                site_latent=site_latent[batch_idx : batch_idx + 1],
                patch_latent=patch_latent[batch_idx : batch_idx + 1],
                predicted_next_global=predicted_next_global[batch_idx : batch_idx + 1],
                path_latent=path_latent[batch_idx : batch_idx + 1],
                horizon_k=horizon_k[batch_idx : batch_idx + 1],
                current_types=current_types[batch_idx : batch_idx + 1],
                edge_pair_indices=pair_tensor,
            )[0]
        alpha = float(np.clip(support_blend_alpha, 0.0, 1.0))
        pair_scores = (1.0 - alpha) * energy_pair_scores + alpha * support_pair_scores
        moving_type_pred = torch.full((pair_scores.shape[0],), mod.CU_TYPE, dtype=torch.long, device=pair_scores.device)
        if hasattr(model, "decode_action_edge_pair_moving_type"):
            moving_type_logits = model.decode_action_edge_pair_moving_type(
                site_latent=site_latent[batch_idx : batch_idx + 1],
                patch_latent=patch_latent[batch_idx : batch_idx + 1],
                predicted_next_global=predicted_next_global[batch_idx : batch_idx + 1],
                path_latent=path_latent[batch_idx : batch_idx + 1],
                horizon_k=horizon_k[batch_idx : batch_idx + 1],
                current_types=current_types[batch_idx : batch_idx + 1],
                edge_pair_indices=pair_tensor,
            )[0]
            moving_type_score = torch.softmax(moving_type_logits, dim=-1)[:, mod.CU_TYPE]
            pair_scores = pair_scores + float(multiobjective_type_weight) * moving_type_score
            moving_type_pred = torch.argmax(moving_type_logits, dim=-1)
        if hasattr(model, "decode_action_edge_pair_order"):
            order_logits = model.decode_action_edge_pair_order(
                site_latent=site_latent[batch_idx : batch_idx + 1],
                patch_latent=patch_latent[batch_idx : batch_idx + 1],
                predicted_next_global=predicted_next_global[batch_idx : batch_idx + 1],
                path_latent=path_latent[batch_idx : batch_idx + 1],
                horizon_k=horizon_k[batch_idx : batch_idx + 1],
                current_types=current_types[batch_idx : batch_idx + 1],
                edge_pair_indices=pair_tensor,
            )[0]
            pair_scores = pair_scores + float(multiobjective_order_weight) * (1.0 - torch.sigmoid(order_logits))

        by_anchor: dict[int, list[tuple[int, int]]] = {}
        for anchor, dest, pair_idx in pair_entries:
            by_anchor.setdefault(anchor, []).append((dest, pair_idx))
        selected_pairs: list[tuple[int, int, int, float]] = []
        for _anchor, dest_pairs in by_anchor.items():
            if len(dest_pairs) > per_anchor:
                score_idx = torch.tensor([pair_idx for _, pair_idx in dest_pairs], dtype=torch.long, device=change_logits.device)
                keep_local = torch.topk(pair_scores[score_idx], k=per_anchor).indices.detach().cpu().tolist()
                dest_pairs = [dest_pairs[int(i)] for i in keep_local]
            for dest_idx, pair_idx in dest_pairs:
                source_idx = int(pair_entries[int(pair_idx)][0])
                selected_pairs.append((source_idx, int(dest_idx), int(pair_idx), float(pair_scores[int(pair_idx)].item())))
        pair_budget = _resolve_global_pair_budget(global_pair_budget, horizon_k, batch_idx)
        if pair_budget > 0 and len(selected_pairs) > pair_budget:
            selected_pairs = sorted(selected_pairs, key=lambda item: item[3], reverse=True)[:pair_budget]
        for source_idx, dest_idx, pair_idx, _score in selected_pairs:
            source_score = pair_scores[int(pair_idx)]
            displacement_mask[batch_idx, source_idx] = True
            displacement_mask[batch_idx, dest_idx] = True
            displacement_logits[batch_idx, source_idx] = torch.maximum(
                displacement_logits[batch_idx, source_idx],
                source_score,
            )
            displacement_logits[batch_idx, dest_idx] = torch.maximum(
                displacement_logits[batch_idx, dest_idx],
                source_score,
            )
            dest_current_type = int(current_types[batch_idx, dest_idx].item())
            if dest_current_type in (mod.FE_TYPE, mod.CU_TYPE):
                moving_type = dest_current_type
            else:
                moving_type = int(moving_type_pred[int(pair_idx)].item())
                if moving_type not in (mod.FE_TYPE, mod.CU_TYPE):
                    moving_type = mod.CU_TYPE
            displacement_type_logits[batch_idx, source_idx, :] = -20.0
            displacement_type_logits[batch_idx, source_idx, moving_type] = 20.0
            displacement_type_logits[batch_idx, dest_idx, :] = -20.0
            displacement_type_logits[batch_idx, dest_idx, mod.V_TYPE] = 20.0
    return displacement_logits, displacement_type_logits, displacement_mask


def _vacancy_pair_completion_logits(
    *,
    model: mod.MacroDreamerEditModel,
    site_latent: torch.Tensor,
    patch_latent: torch.Tensor,
    predicted_next_global: torch.Tensor,
    path_latent: torch.Tensor,
    horizon_k: torch.Tensor,
    current_types: torch.Tensor,
    candidate_positions: torch.Tensor,
    raw_type_logits: torch.Tensor,
    change_logits: torch.Tensor,
    proposal_logits: torch.Tensor,
    action_support_logits: torch.Tensor | None,
    action_source_logits: torch.Tensor | None,
    action_destination_logits: torch.Tensor | None,
    candidate_mask: torch.Tensor,
    anchor_source: str,
    destination_source: str,
    anchor_budget: int,
    destinations_per_anchor: int,
    global_pair_budget: int,
    blend_alpha: float,
    energy_blend: bool = False,
    use_interaction_score: bool = False,
    multiobjective_type_weight: float = 0.15,
    multiobjective_order_weight: float = 0.10,
    diagnostics: list[dict[str, object]] | None = None,
    rank_diagnostic: bool = False,
    rank_diagnostic_max_pairs: int = 0,
    factorized_diagnostic: bool = False,
    factorized_diagnostic_max_pairs: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    source_logits = _projection_logits_from_source(
        change_logits=change_logits,
        proposal_logits=proposal_logits,
        action_support_logits=action_support_logits,
        action_source_logits=action_source_logits,
        action_destination_logits=action_destination_logits,
        source=anchor_source,
        blend_alpha=blend_alpha,
    )
    destination_logits = _projection_logits_from_source(
        change_logits=change_logits,
        proposal_logits=proposal_logits,
        action_support_logits=action_support_logits,
        action_source_logits=action_source_logits,
        action_destination_logits=action_destination_logits,
        source=destination_source,
        blend_alpha=blend_alpha,
    )
    displacement_logits = torch.full_like(change_logits, -20.0)
    displacement_type_logits = torch.full_like(raw_type_logits, -20.0)
    displacement_type_logits.scatter_(2, current_types.unsqueeze(-1), 20.0)
    displacement_mask = torch.zeros_like(candidate_mask, dtype=torch.bool)
    if not hasattr(model, "decode_vacancy_pairs"):
        return displacement_logits, displacement_type_logits, displacement_mask

    source_budget = max(int(anchor_budget), 0)
    dest_budget = max(int(destinations_per_anchor), 0)
    if source_budget <= 0 or dest_budget <= 0:
        return displacement_logits, displacement_type_logits, displacement_mask

    for batch_idx in range(candidate_mask.shape[0]):
        valid = candidate_mask[batch_idx] > 0
        source_candidates = torch.nonzero(
            valid & (current_types[batch_idx] == mod.V_TYPE),
            as_tuple=False,
        ).squeeze(-1)
        destination_candidates = torch.nonzero(
            valid
            & (
                (current_types[batch_idx] == mod.FE_TYPE)
                | (current_types[batch_idx] == mod.CU_TYPE)
            ),
            as_tuple=False,
        ).squeeze(-1)
        if source_candidates.numel() == 0 or destination_candidates.numel() == 0:
            continue
        source_keep = min(source_budget, int(source_candidates.numel()))
        destination_keep = min(max(dest_budget * source_keep, dest_budget), int(destination_candidates.numel()))
        source_idx = source_candidates[
            torch.topk(source_logits[batch_idx, source_candidates], k=source_keep).indices
        ]
        dest_idx = destination_candidates[
            torch.topk(destination_logits[batch_idx, destination_candidates], k=destination_keep).indices
        ]
        pair_entries = [
            (int(source.item()), int(dest.item()))
            for source in source_idx
            for dest in dest_idx
            if int(source.item()) != int(dest.item())
        ]
        if not pair_entries:
            continue
        pair_tensor = torch.tensor(pair_entries, dtype=torch.long, device=change_logits.device).unsqueeze(0)
        vacancy_pair_scores = model.decode_vacancy_pairs(
            site_latent=site_latent[batch_idx : batch_idx + 1],
            patch_latent=patch_latent[batch_idx : batch_idx + 1],
            predicted_next_global=predicted_next_global[batch_idx : batch_idx + 1],
            path_latent=path_latent[batch_idx : batch_idx + 1],
            horizon_k=horizon_k[batch_idx : batch_idx + 1],
            current_types=current_types[batch_idx : batch_idx + 1],
            edge_pair_indices=pair_tensor,
        )[0]
        pair_scores = vacancy_pair_scores
        interaction_pair_scores_for_diag: torch.Tensor | None = None
        if use_interaction_score and hasattr(model, "decode_vacancy_pair_interaction"):
            interaction_pair_scores_for_diag = model.decode_vacancy_pair_interaction(
                site_latent=site_latent[batch_idx : batch_idx + 1],
                patch_latent=patch_latent[batch_idx : batch_idx + 1],
                predicted_next_global=predicted_next_global[batch_idx : batch_idx + 1],
                path_latent=path_latent[batch_idx : batch_idx + 1],
                horizon_k=horizon_k[batch_idx : batch_idx + 1],
                current_types=current_types[batch_idx : batch_idx + 1],
                edge_pair_indices=pair_tensor,
            )[0]
            pair_scores = interaction_pair_scores_for_diag
        source_endpoint_scores = source_logits[batch_idx]
        destination_endpoint_scores = destination_logits[batch_idx]
        energy_pair_scores_for_diag: torch.Tensor | None = None
        moving_type_score_for_diag: torch.Tensor | None = None
        order_early_score_for_diag: torch.Tensor | None = None
        if (energy_blend or factorized_diagnostic) and hasattr(model, "decode_action_edge_pairs"):
            energy_pair_scores_for_diag = model.decode_action_edge_pairs(
                site_latent=site_latent[batch_idx : batch_idx + 1],
                patch_latent=patch_latent[batch_idx : batch_idx + 1],
                predicted_next_global=predicted_next_global[batch_idx : batch_idx + 1],
                path_latent=path_latent[batch_idx : batch_idx + 1],
                horizon_k=horizon_k[batch_idx : batch_idx + 1],
                current_types=current_types[batch_idx : batch_idx + 1],
                edge_pair_indices=pair_tensor,
            )[0]
            if energy_blend:
                alpha = float(np.clip(blend_alpha, 0.0, 1.0))
                support_pair_scores = (
                    interaction_pair_scores_for_diag
                    if interaction_pair_scores_for_diag is not None
                    else vacancy_pair_scores
                )
                pair_scores = (1.0 - alpha) * energy_pair_scores_for_diag + alpha * support_pair_scores
        moving_type_pred = current_types[batch_idx, pair_tensor[0, :, 1]].clone()
        if hasattr(model, "decode_vacancy_pair_moving_type"):
            moving_type_logits = model.decode_vacancy_pair_moving_type(
                site_latent=site_latent[batch_idx : batch_idx + 1],
                patch_latent=patch_latent[batch_idx : batch_idx + 1],
                predicted_next_global=predicted_next_global[batch_idx : batch_idx + 1],
                path_latent=path_latent[batch_idx : batch_idx + 1],
                horizon_k=horizon_k[batch_idx : batch_idx + 1],
                current_types=current_types[batch_idx : batch_idx + 1],
                edge_pair_indices=pair_tensor,
            )[0]
            moving_type_score = torch.softmax(moving_type_logits, dim=-1)[:, mod.CU_TYPE]
            moving_type_score_for_diag = moving_type_score
            pair_scores = pair_scores + float(multiobjective_type_weight) * moving_type_score
            moving_type_pred = torch.argmax(moving_type_logits, dim=-1)
        if hasattr(model, "decode_vacancy_pair_order"):
            order_logits = model.decode_vacancy_pair_order(
                site_latent=site_latent[batch_idx : batch_idx + 1],
                patch_latent=patch_latent[batch_idx : batch_idx + 1],
                predicted_next_global=predicted_next_global[batch_idx : batch_idx + 1],
                path_latent=path_latent[batch_idx : batch_idx + 1],
                horizon_k=horizon_k[batch_idx : batch_idx + 1],
                current_types=current_types[batch_idx : batch_idx + 1],
                edge_pair_indices=pair_tensor,
            )[0]
            order_early_score_for_diag = 1.0 - torch.sigmoid(order_logits)
            pair_scores = pair_scores + float(multiobjective_order_weight) * order_early_score_for_diag

        selected_pairs: list[tuple[int, int, int, float]] = [
            (int(source), int(dest), pair_idx, float(pair_scores[pair_idx].item()))
            for pair_idx, (source, dest) in enumerate(pair_entries)
        ]
        ranked_pairs = sorted(selected_pairs, key=lambda item: item[3], reverse=True)
        pair_budget = _resolve_global_pair_budget(global_pair_budget, horizon_k, batch_idx)
        if pair_budget > 0 and len(selected_pairs) > pair_budget:
            selected_pairs = sorted(selected_pairs, key=lambda item: item[3], reverse=True)[:pair_budget]
        diagnostic_pair_keys: set[tuple[int, int, int]] = set()
        if diagnostics is not None:
            diagnostic_selected = selected_pairs
            if len(diagnostic_selected) > 128:
                diagnostic_selected = sorted(diagnostic_selected, key=lambda item: item[3], reverse=True)[:128]
            diagnostic_pair_keys = {
                (int(source_idx), int(dest_idx), int(pair_idx))
                for source_idx, dest_idx, pair_idx, _score in diagnostic_selected
            }
        ranked_pair_payload: list[dict[str, object]] = []
        if diagnostics is not None and rank_diagnostic:
            max_ranked = int(rank_diagnostic_max_pairs)
            ranked_for_payload = ranked_pairs if max_ranked <= 0 else ranked_pairs[:max_ranked]
            for rank, (source_idx, dest_idx, pair_idx, score_value) in enumerate(ranked_for_payload, start=1):
                moving_type = int(moving_type_pred[int(pair_idx)].item())
                if moving_type not in (mod.FE_TYPE, mod.CU_TYPE):
                    moving_type = int(current_types[batch_idx, dest_idx].item())
                ranked_pair_payload.append(
                    {
                        "rank": int(rank),
                        "source_position": [
                            int(round(float(v)))
                            for v in candidate_positions[batch_idx, source_idx].detach().cpu().tolist()
                        ],
                        "destination_position": [
                            int(round(float(v)))
                            for v in candidate_positions[batch_idx, dest_idx].detach().cpu().tolist()
                        ],
                        "moving_type": int(moving_type),
                        "score": float(score_value),
                    }
                )
        factorized_pair_payload: list[dict[str, object]] = []
        if diagnostics is not None and factorized_diagnostic:
            max_factorized = int(factorized_diagnostic_max_pairs)
            factorized_for_payload = ranked_pairs if max_factorized <= 0 else ranked_pairs[:max_factorized]
            for rank, (source_idx, dest_idx, pair_idx, score_value) in enumerate(factorized_for_payload, start=1):
                moving_type = int(moving_type_pred[int(pair_idx)].item())
                if moving_type not in (mod.FE_TYPE, mod.CU_TYPE):
                    moving_type = int(current_types[batch_idx, dest_idx].item())
                source_score = float(source_endpoint_scores[int(source_idx)].detach().cpu().item())
                destination_score = float(destination_endpoint_scores[int(dest_idx)].detach().cpu().item())
                vacancy_score = float(vacancy_pair_scores[int(pair_idx)].detach().cpu().item())
                energy_score = (
                    float(energy_pair_scores_for_diag[int(pair_idx)].detach().cpu().item())
                    if energy_pair_scores_for_diag is not None
                    else 0.0
                )
                interaction_score = (
                    float(interaction_pair_scores_for_diag[int(pair_idx)].detach().cpu().item())
                    if interaction_pair_scores_for_diag is not None
                    else 0.0
                )
                moving_type_score = (
                    float(moving_type_score_for_diag[int(pair_idx)].detach().cpu().item())
                    if moving_type_score_for_diag is not None
                    else 0.0
                )
                order_early_score = (
                    float(order_early_score_for_diag[int(pair_idx)].detach().cpu().item())
                    if order_early_score_for_diag is not None
                    else 0.0
                )
                endpoint_sum_score = float(source_score + destination_score)
                factorized_pair_payload.append(
                    {
                        "rank": int(rank),
                        "source_position": [
                            int(round(float(v)))
                            for v in candidate_positions[batch_idx, source_idx].detach().cpu().tolist()
                        ],
                        "destination_position": [
                            int(round(float(v)))
                            for v in candidate_positions[batch_idx, dest_idx].detach().cpu().tolist()
                        ],
                        "moving_type": int(moving_type),
                        "score": float(score_value),
                        "source_score": source_score,
                        "destination_score": destination_score,
                        "endpoint_sum_score": endpoint_sum_score,
                        "vacancy_score": vacancy_score,
                        "energy_score": energy_score,
                        "interaction_score": interaction_score,
                        "moving_type_score": moving_type_score,
                        "order_early_score": order_early_score,
                        "interaction_residual": float(vacancy_score - endpoint_sum_score),
                    }
                )
        diagnostic_pairs = []
        for source_idx, dest_idx, pair_idx, _score in selected_pairs:
            score = pair_scores[int(pair_idx)]
            displacement_mask[batch_idx, source_idx] = True
            displacement_mask[batch_idx, dest_idx] = True
            displacement_logits[batch_idx, source_idx] = torch.maximum(displacement_logits[batch_idx, source_idx], score)
            displacement_logits[batch_idx, dest_idx] = torch.maximum(displacement_logits[batch_idx, dest_idx], score)
            moving_type = int(moving_type_pred[int(pair_idx)].item())
            if moving_type not in (mod.FE_TYPE, mod.CU_TYPE):
                moving_type = int(current_types[batch_idx, dest_idx].item())
            if moving_type not in (mod.FE_TYPE, mod.CU_TYPE):
                moving_type = mod.CU_TYPE
            displacement_type_logits[batch_idx, source_idx, :] = -20.0
            displacement_type_logits[batch_idx, source_idx, moving_type] = 20.0
            displacement_type_logits[batch_idx, dest_idx, :] = -20.0
            displacement_type_logits[batch_idx, dest_idx, mod.V_TYPE] = 20.0
            if diagnostics is not None and (int(source_idx), int(dest_idx), int(pair_idx)) in diagnostic_pair_keys:
                diagnostic_pairs.append(
                    {
                        "source_index": int(source_idx),
                        "destination_index": int(dest_idx),
                        "source_position": [
                            int(round(float(v)))
                            for v in candidate_positions[batch_idx, source_idx].detach().cpu().tolist()
                        ],
                        "destination_position": [
                            int(round(float(v)))
                            for v in candidate_positions[batch_idx, dest_idx].detach().cpu().tolist()
                        ],
                        "moving_type": int(moving_type),
                        "score": float(score.detach().cpu().item()),
                    }
                )
        if diagnostics is not None:
            diagnostic_payload = {
                "batch_index": int(batch_idx),
                "candidate_pair_count": int(len(pair_entries)),
                "selected_pair_count": int(len(selected_pairs)),
                "selected_pairs": diagnostic_pairs,
            }
            if rank_diagnostic:
                diagnostic_payload.update(
                    {
                        "ranked_pair_score_count": int(len(ranked_pairs)),
                        "ranked_pair_scores": ranked_pair_payload,
                    }
                )
            if factorized_diagnostic:
                diagnostic_payload.update(
                    {
                        "factorized_pair_score_count": int(len(ranked_pairs)),
                        "factorized_pair_scores": factorized_pair_payload,
                    }
                )
            diagnostics.append(diagnostic_payload)
    return displacement_logits, displacement_type_logits, displacement_mask


def _typed_diff_change_logits(type_logits: torch.Tensor, current_types: torch.Tensor) -> torch.Tensor:
    current_logits = type_logits.gather(-1, current_types.unsqueeze(-1)).squeeze(-1)
    noncopy_logits = type_logits.masked_fill(
        torch.nn.functional.one_hot(current_types, num_classes=mod.NUM_SITE_TYPES).bool(),
        -1.0e4,
    )
    return torch.logsumexp(noncopy_logits, dim=-1) - current_logits


def _terminal_typed_diff_projection_logits(
    *,
    model: mod.MacroDreamerEditModel,
    site_latent: torch.Tensor,
    patch_latent: torch.Tensor,
    predicted_next_global: torch.Tensor,
    path_latent: torch.Tensor,
    horizon_k: torch.Tensor,
    current_types: torch.Tensor,
    action_context_logits: torch.Tensor,
    raw_type_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not hasattr(model, "decode_terminal_typed_diff"):
        return _typed_diff_change_logits(raw_type_logits, current_types), raw_type_logits
    typed_logits = model.decode_terminal_typed_diff(
        site_latent=site_latent,
        patch_latent=patch_latent,
        predicted_next_global=predicted_next_global,
        path_latent=path_latent,
        horizon_k=horizon_k,
        current_types=current_types,
        action_sequence_logits=action_context_logits,
    )
    return _typed_diff_change_logits(typed_logits, current_types), typed_logits


def _terminal_decoupled_projection_logits(
    *,
    model: mod.MacroDreamerEditModel,
    site_latent: torch.Tensor,
    patch_latent: torch.Tensor,
    predicted_next_global: torch.Tensor,
    path_latent: torch.Tensor,
    horizon_k: torch.Tensor,
    current_types: torch.Tensor,
    change_logits: torch.Tensor,
    proposal_logits: torch.Tensor,
    action_support_logits: torch.Tensor | None,
    action_source_logits: torch.Tensor | None,
    action_destination_logits: torch.Tensor | None,
    candidate_positions: torch.Tensor,
    candidate_mask: torch.Tensor,
    box_dims: torch.Tensor,
    nn1_offsets: np.ndarray,
    anchor_source: str,
    anchor_budget: int,
    destinations_per_anchor: int,
    global_pair_budget: int,
    blend_alpha: float,
    support_blend_alpha: float,
    multiobjective_type_weight: float,
    multiobjective_order_weight: float,
    gate_to_action_edge: bool,
    require_vacancy_atom_pair: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    action_sequence_logits, action_sequence_mask = _action_edge_pair_completion_logits(
        model=model,
        site_latent=site_latent,
        patch_latent=patch_latent,
        predicted_next_global=predicted_next_global,
        path_latent=path_latent,
        horizon_k=horizon_k,
        current_types=current_types,
        change_logits=change_logits,
        proposal_logits=proposal_logits,
        action_support_logits=action_support_logits,
        action_source_logits=action_source_logits,
        action_destination_logits=action_destination_logits,
        candidate_positions=candidate_positions,
        candidate_mask=candidate_mask,
        box_dims=box_dims,
        nn1_offsets=nn1_offsets,
        anchor_source=anchor_source,
        anchor_budget=anchor_budget,
        destinations_per_anchor=destinations_per_anchor,
        global_pair_budget=global_pair_budget,
        blend_alpha=blend_alpha,
        score_source="multiobjective",
        support_blend_alpha=support_blend_alpha,
        multiobjective_type_weight=multiobjective_type_weight,
        multiobjective_order_weight=multiobjective_order_weight,
        require_vacancy_atom_pair=require_vacancy_atom_pair,
    )
    if hasattr(model, "decode_terminal_edit_support"):
        terminal_logits = model.decode_terminal_edit_support(
            site_latent=site_latent,
            patch_latent=patch_latent,
            predicted_next_global=predicted_next_global,
            path_latent=path_latent,
            horizon_k=horizon_k,
            current_types=current_types,
            action_sequence_logits=action_sequence_logits,
        )
    else:
        terminal_logits = proposal_logits
    if gate_to_action_edge:
        terminal_logits = torch.where(action_sequence_mask, terminal_logits, torch.full_like(terminal_logits, -20.0))
    return terminal_logits, action_sequence_mask


def _sequence_rollout_projection_logits(
    *,
    model: mod.MacroDreamerEditModel,
    site_latent: torch.Tensor,
    patch_latent: torch.Tensor,
    predicted_next_global: torch.Tensor,
    path_latent: torch.Tensor,
    horizon_k: torch.Tensor,
    current_types: torch.Tensor,
    change_logits: torch.Tensor,
    proposal_logits: torch.Tensor,
    action_support_logits: torch.Tensor | None,
    action_source_logits: torch.Tensor | None,
    action_destination_logits: torch.Tensor | None,
    candidate_positions: torch.Tensor,
    candidate_mask: torch.Tensor,
    box_dims: torch.Tensor,
    nn1_offsets: np.ndarray,
    anchor_source: str,
    anchor_budget: int,
    destinations_per_anchor: int,
    global_pair_budget: int,
    blend_alpha: float,
    support_blend_alpha: float,
    multiobjective_type_weight: float,
    multiobjective_order_weight: float,
    gate_to_rollout: bool,
    require_vacancy_atom_pair: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    anchor_logits = _projection_logits_from_source(
        change_logits=change_logits,
        proposal_logits=proposal_logits,
        action_support_logits=action_support_logits,
        action_source_logits=action_source_logits,
        action_destination_logits=action_destination_logits,
        source=anchor_source,
        blend_alpha=blend_alpha,
    )
    rollout_logits = torch.full_like(change_logits, -6.0)
    rollout_mask = torch.zeros_like(candidate_mask, dtype=torch.bool)
    action_mask = torch.zeros_like(candidate_mask, dtype=torch.bool)
    if not hasattr(model, "decode_action_edge_pairs"):
        rollout_logits = torch.where(candidate_mask > 0, rollout_logits, torch.full_like(rollout_logits, -20.0))
        return rollout_logits, rollout_mask
    anchor_count = max(int(anchor_budget), 0)
    per_anchor = max(int(destinations_per_anchor), 0)
    if anchor_count <= 0 or per_anchor <= 0:
        rollout_logits = torch.where(candidate_mask > 0, rollout_logits, torch.full_like(rollout_logits, -20.0))
        return rollout_logits, rollout_mask

    nn1 = np.asarray(nn1_offsets, dtype=np.int64)
    for batch_idx in range(candidate_mask.shape[0]):
        valid_idx = torch.nonzero(candidate_mask[batch_idx] > 0, as_tuple=False).squeeze(-1)
        if valid_idx.numel() == 0:
            continue
        sample_anchor_count = min(anchor_count, int(valid_idx.numel()))
        anchor_local = torch.topk(anchor_logits[batch_idx, valid_idx], k=sample_anchor_count).indices
        anchor_idx = valid_idx[anchor_local]
        positions_np = torch.round(candidate_positions[batch_idx]).detach().cpu().numpy().astype(np.int64)
        box_np = torch.round(box_dims[batch_idx]).detach().cpu().numpy().astype(np.int64)
        lookup = {
            tuple(int(x) for x in positions_np[int(idx)].tolist()): int(idx)
            for idx in valid_idx.detach().cpu().numpy().tolist()
        }
        pair_entries: list[tuple[int, int, int]] = []
        for anchor in anchor_idx.detach().cpu().numpy().tolist():
            anchor = int(anchor)
            anchor_pos = positions_np[anchor]
            for offset in nn1:
                dest_idx = lookup.get(_periodic_neighbor_position(anchor_pos, offset, box_np))
                if dest_idx is not None:
                    if not _is_valid_vacancy_atom_pair(
                        current_types,
                        batch_idx,
                        anchor,
                        int(dest_idx),
                        require_vacancy_atom_pair=require_vacancy_atom_pair,
                    ):
                        continue
                    pair_entries.append((anchor, int(dest_idx), len(pair_entries)))
        if not pair_entries:
            continue
        pair_tensor = torch.tensor(
            [[anchor, dest] for anchor, dest, _ in pair_entries],
            dtype=torch.long,
            device=change_logits.device,
        ).unsqueeze(0)
        energy_pair_scores = model.decode_action_edge_pairs(
            site_latent=site_latent[batch_idx : batch_idx + 1],
            patch_latent=patch_latent[batch_idx : batch_idx + 1],
            predicted_next_global=predicted_next_global[batch_idx : batch_idx + 1],
            path_latent=path_latent[batch_idx : batch_idx + 1],
            horizon_k=horizon_k[batch_idx : batch_idx + 1],
            current_types=current_types[batch_idx : batch_idx + 1],
            edge_pair_indices=pair_tensor,
        )[0]
        support_pair_scores = torch.zeros_like(energy_pair_scores)
        if hasattr(model, "decode_action_edge_pair_support"):
            support_pair_scores = model.decode_action_edge_pair_support(
                site_latent=site_latent[batch_idx : batch_idx + 1],
                patch_latent=patch_latent[batch_idx : batch_idx + 1],
                predicted_next_global=predicted_next_global[batch_idx : batch_idx + 1],
                path_latent=path_latent[batch_idx : batch_idx + 1],
                horizon_k=horizon_k[batch_idx : batch_idx + 1],
                current_types=current_types[batch_idx : batch_idx + 1],
                edge_pair_indices=pair_tensor,
            )[0]
        alpha = float(np.clip(support_blend_alpha, 0.0, 1.0))
        pair_scores = (1.0 - alpha) * energy_pair_scores + alpha * support_pair_scores
        moving_type_pred = torch.full((pair_scores.shape[0],), mod.CU_TYPE, dtype=torch.long, device=pair_scores.device)
        if hasattr(model, "decode_action_edge_pair_moving_type"):
            moving_type_logits = model.decode_action_edge_pair_moving_type(
                site_latent=site_latent[batch_idx : batch_idx + 1],
                patch_latent=patch_latent[batch_idx : batch_idx + 1],
                predicted_next_global=predicted_next_global[batch_idx : batch_idx + 1],
                path_latent=path_latent[batch_idx : batch_idx + 1],
                horizon_k=horizon_k[batch_idx : batch_idx + 1],
                current_types=current_types[batch_idx : batch_idx + 1],
                edge_pair_indices=pair_tensor,
            )[0]
            moving_type_score = torch.softmax(moving_type_logits, dim=-1)[:, mod.CU_TYPE]
            pair_scores = pair_scores + float(multiobjective_type_weight) * moving_type_score
            moving_type_pred = torch.argmax(moving_type_logits, dim=-1)
        order_score = torch.zeros_like(pair_scores)
        if hasattr(model, "decode_action_edge_pair_order"):
            order_logits = model.decode_action_edge_pair_order(
                site_latent=site_latent[batch_idx : batch_idx + 1],
                patch_latent=patch_latent[batch_idx : batch_idx + 1],
                predicted_next_global=predicted_next_global[batch_idx : batch_idx + 1],
                path_latent=path_latent[batch_idx : batch_idx + 1],
                horizon_k=horizon_k[batch_idx : batch_idx + 1],
                current_types=current_types[batch_idx : batch_idx + 1],
                edge_pair_indices=pair_tensor,
            )[0]
            order_score = torch.sigmoid(order_logits)
            pair_scores = pair_scores + float(multiobjective_order_weight) * (1.0 - order_score)
        by_anchor: dict[int, list[tuple[int, int]]] = {}
        for anchor, dest, pair_idx in pair_entries:
            by_anchor.setdefault(anchor, []).append((dest, pair_idx))
        selected_pairs: list[tuple[int, int, int, float, float]] = []
        for anchor, dest_pairs in by_anchor.items():
            if len(dest_pairs) > per_anchor:
                score_idx = torch.tensor([pair_idx for _, pair_idx in dest_pairs], dtype=torch.long, device=change_logits.device)
                keep_local = torch.topk(pair_scores[score_idx], k=per_anchor).indices.detach().cpu().tolist()
                dest_pairs = [dest_pairs[int(i)] for i in keep_local]
            for dest_idx, pair_idx in dest_pairs:
                selected_pairs.append(
                    (
                        int(anchor),
                        int(dest_idx),
                        int(moving_type_pred[int(pair_idx)].item()),
                        float(order_score[int(pair_idx)].item()),
                        float(pair_scores[int(pair_idx)].item()),
                    )
                )
        pair_budget = _resolve_global_pair_budget(global_pair_budget, horizon_k, batch_idx)
        if pair_budget > 0 and len(selected_pairs) > pair_budget:
            selected_pairs = sorted(selected_pairs, key=lambda item: item[4], reverse=True)[:pair_budget]
        selected_pairs.sort(key=lambda item: (item[3], -item[4]))
        state = current_types[batch_idx].detach().cpu().numpy().astype(np.int64).copy()
        initial = state.copy()
        valid_np = (candidate_mask[batch_idx].detach().cpu().numpy() > 0.0)
        for source_idx, dest_idx, moving_type, _order, _score in selected_pairs:
            action_mask[batch_idx, int(source_idx)] = True
            action_mask[batch_idx, int(dest_idx)] = True
            moved = int(moving_type)
            if moved not in (mod.FE_TYPE, mod.CU_TYPE):
                moved = int(state[dest_idx])
                if moved == mod.V_TYPE:
                    moved = mod.FE_TYPE
            # KMC action pair orientation is source vacancy -> destination atom.
            state[source_idx] = moved
            state[dest_idx] = mod.V_TYPE
        changed_np = (state != initial) & valid_np
        if np.any(changed_np):
            changed_idx = torch.tensor(np.flatnonzero(changed_np), dtype=torch.long, device=change_logits.device)
            rollout_mask[batch_idx, changed_idx] = True
            rollout_logits[batch_idx, changed_idx] = 6.0
    if gate_to_rollout:
        rollout_logits = torch.where(rollout_mask, rollout_logits, torch.full_like(rollout_logits, -20.0))
    rollout_logits = torch.where(candidate_mask > 0, rollout_logits, torch.full_like(rollout_logits, -20.0))
    if hasattr(model, "decode_terminal_edit_support"):
        terminal_logits = model.decode_terminal_edit_support(
            site_latent=site_latent,
            patch_latent=patch_latent,
            predicted_next_global=predicted_next_global,
            path_latent=path_latent,
            horizon_k=horizon_k,
            current_types=current_types,
            action_sequence_logits=rollout_logits,
        )
    else:
        terminal_logits = rollout_logits
    if gate_to_rollout:
        terminal_logits = torch.where(rollout_mask, terminal_logits, torch.full_like(terminal_logits, -20.0))
    return terminal_logits, action_mask | rollout_mask


def _apply_projection_topk_support(
    *,
    projection_change_logits: torch.Tensor,
    projection_type_logits: torch.Tensor,
    ranking_logits: torch.Tensor,
    candidate_mask: torch.Tensor,
    current_types: torch.Tensor,
    topk_budget: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    budget = int(topk_budget)
    topk_mask = torch.zeros_like(candidate_mask, dtype=torch.bool)
    if budget <= 0:
        return projection_change_logits, projection_type_logits, topk_mask
    restricted_change_logits = torch.full_like(projection_change_logits, -20.0)
    for batch_idx in range(candidate_mask.shape[0]):
        valid_idx = torch.nonzero(candidate_mask[batch_idx] > 0, as_tuple=False).squeeze(-1)
        if valid_idx.numel() == 0:
            continue
        sample_budget = min(budget, int(valid_idx.numel()))
        top_local = torch.topk(ranking_logits[batch_idx, valid_idx], k=sample_budget).indices
        top_idx = valid_idx[top_local]
        topk_mask[batch_idx, top_idx] = True
        restricted_change_logits[batch_idx, top_idx] = projection_change_logits[batch_idx, top_idx]
    copy_logits = torch.full_like(projection_type_logits, -20.0)
    copy_logits.scatter_(2, current_types.unsqueeze(-1), 20.0)
    outside_topk = (candidate_mask > 0) & (~topk_mask)
    restricted_type_logits = torch.where(outside_topk.unsqueeze(-1), copy_logits, projection_type_logits)
    return restricted_change_logits, restricted_type_logits, topk_mask


def _predict_candidate_for_horizon(
    *,
    model: mod.MacroDreamerEditModel,
    reward_model: mod.MacroDreamerEditModel | None,
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
    planner_noop_risk_penalty: float = 0.0,
    duration_blend_alpha: float = 1.0,
    planner_tau_blend_alpha: float = 1.0,
    duration_log_offset: float = 0.0,
    planner_tau_log_offset: float = 0.0,
    planner_duration_checkpoint_source: str = "duration",
    reward_prediction_source: str = "raw",
    reward_edit_context_source: str = "default",
    aux_projected_types_source: str = "aux",
    proposal_diagnostic: bool = False,
    proposal_diagnostic_max_sites: int = 256,
    proposal_diagnostic_store_candidate_positions: bool = False,
    planner_projection_change_source: str = "change",
    planner_projection_change_blend_alpha: float = 0.5,
    planner_projection_topk_source: str = "none",
    planner_projection_topk_budget: int = 0,
    planner_edge_completion_anchor_source: str = "action_source",
    planner_edge_completion_destination_source: str = "action_destination",
    planner_edge_completion_anchor_budget: int = 32,
    planner_edge_completion_destinations_per_anchor: int = 8,
    planner_edge_completion_global_pair_budget: int = 0,
    planner_edge_completion_destination_scope: str = "nn1",
    planner_edge_completion_require_vacancy_atom_pair: bool = False,
    planner_edge_pair_multiobjective_type_weight: float = 0.15,
    planner_edge_pair_multiobjective_order_weight: float = 0.10,
    planner_proposal_score_weight: float = 0.0,
    planner_candidate_quality_score_weight: float = 0.0,
    planner_vacancy_pair_rank_diagnostic: bool = False,
    planner_vacancy_pair_rank_max_pairs: int = 0,
    planner_vacancy_pair_factorized_diagnostic: bool = False,
    planner_vacancy_pair_factorized_max_pairs: int = 0,
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
    proposal_logits = model.decode_proposal(
        site_latent=site_latent,
        patch_latent=patch_latent,
        predicted_next_global=next_pred,
        path_latent=path_latent,
        horizon_k=tensors["horizon_k"],
        current_types=tensors["current_types"],
    )
    action_support_logits = (
        model.decode_action_support(
            site_latent=site_latent,
            patch_latent=patch_latent,
            predicted_next_global=next_pred,
            path_latent=path_latent,
            horizon_k=tensors["horizon_k"],
            current_types=tensors["current_types"],
        )
        if hasattr(model, "decode_action_support")
        else proposal_logits
    )
    action_source_logits = (
        model.decode_action_source_support(
            site_latent=site_latent,
            patch_latent=patch_latent,
            predicted_next_global=next_pred,
            path_latent=path_latent,
            horizon_k=tensors["horizon_k"],
            current_types=tensors["current_types"],
        )
        if hasattr(model, "decode_action_source_support")
        else action_support_logits
    )
    action_destination_logits = (
        model.decode_action_destination_support(
            site_latent=site_latent,
            patch_latent=patch_latent,
            predicted_next_global=next_pred,
            path_latent=path_latent,
            horizon_k=tensors["horizon_k"],
            current_types=tensors["current_types"],
        )
        if hasattr(model, "decode_action_destination_support")
        else action_support_logits
    )
    if hasattr(model, "decode_candidate_quality"):
        candidate_quality_logit = model.decode_candidate_quality(
            site_latent=site_latent,
            patch_latent=patch_latent,
            predicted_next_global=next_pred,
            path_latent=path_latent,
            horizon_k=tensors["horizon_k"],
            current_types=tensors["current_types"],
            candidate_mask=tensors["candidate_mask"],
        )
        candidate_quality_score = float(torch.sigmoid(candidate_quality_logit).item())
    else:
        candidate_quality_score = 0.0
    projection_type_logits = raw_type_logits
    edge_completion_mask = torch.zeros_like(tensors["candidate_mask"], dtype=torch.bool)
    vacancy_pair_projection_diagnostics: list[dict[str, object]] = []
    if planner_projection_change_source == "action_edge_completion":
        projection_change_logits, edge_completion_mask = _action_edge_completion_logits(
            change_logits=change_logits,
            proposal_logits=proposal_logits,
            action_support_logits=action_support_logits,
            action_source_logits=action_source_logits,
            action_destination_logits=action_destination_logits,
            candidate_mask=tensors["candidate_mask"],
            box_dims=tensors["box_dims"],
            nn1_offsets=np.asarray(env.env.NN1, dtype=np.int64),
            anchor_source=planner_edge_completion_anchor_source,
            destination_source=planner_edge_completion_destination_source,
            anchor_budget=int(planner_edge_completion_anchor_budget),
            destinations_per_anchor=int(planner_edge_completion_destinations_per_anchor),
            blend_alpha=float(planner_projection_change_blend_alpha),
        )
    elif planner_projection_change_source in {
        "action_edge_pair_completion",
        "action_edge_pair_support_completion",
        "action_edge_pair_blend_completion",
        "action_edge_pair_multiobjective_completion",
    }:
        projection_change_logits, edge_completion_mask = _action_edge_pair_completion_logits(
            model=model,
            site_latent=site_latent,
            patch_latent=patch_latent,
            predicted_next_global=next_pred,
            path_latent=path_latent,
            horizon_k=tensors["horizon_k"],
            current_types=tensors["current_types"],
            change_logits=change_logits,
            proposal_logits=proposal_logits,
            action_support_logits=action_support_logits,
            action_source_logits=action_source_logits,
            action_destination_logits=action_destination_logits,
            candidate_mask=tensors["candidate_mask"],
            box_dims=tensors["box_dims"],
            nn1_offsets=np.asarray(env.env.NN1, dtype=np.int64),
            anchor_source=planner_edge_completion_anchor_source,
            anchor_budget=int(planner_edge_completion_anchor_budget),
            destinations_per_anchor=int(planner_edge_completion_destinations_per_anchor),
            global_pair_budget=int(planner_edge_completion_global_pair_budget),
            destination_scope=planner_edge_completion_destination_scope,
            blend_alpha=float(planner_projection_change_blend_alpha),
            score_source=_edge_pair_completion_score_source(planner_projection_change_source),
            support_blend_alpha=float(planner_projection_change_blend_alpha),
            multiobjective_type_weight=float(planner_edge_pair_multiobjective_type_weight),
            multiobjective_order_weight=float(planner_edge_pair_multiobjective_order_weight),
            require_vacancy_atom_pair=bool(planner_edge_completion_require_vacancy_atom_pair),
        )
    elif planner_projection_change_source in {"terminal_edit_decoupled", "terminal_edit_inside_action_edge"}:
        projection_change_logits, edge_completion_mask = _terminal_decoupled_projection_logits(
            model=model,
            site_latent=site_latent,
            patch_latent=patch_latent,
            predicted_next_global=next_pred,
            path_latent=path_latent,
            horizon_k=tensors["horizon_k"],
            current_types=tensors["current_types"],
            change_logits=change_logits,
            proposal_logits=proposal_logits,
            action_support_logits=action_support_logits,
            action_source_logits=action_source_logits,
            action_destination_logits=action_destination_logits,
            candidate_mask=tensors["candidate_mask"],
            box_dims=tensors["box_dims"],
            nn1_offsets=np.asarray(env.env.NN1, dtype=np.int64),
            anchor_source=planner_edge_completion_anchor_source,
            anchor_budget=int(planner_edge_completion_anchor_budget),
            destinations_per_anchor=int(planner_edge_completion_destinations_per_anchor),
            global_pair_budget=int(planner_edge_completion_global_pair_budget),
            blend_alpha=float(planner_projection_change_blend_alpha),
            support_blend_alpha=float(planner_projection_change_blend_alpha),
            multiobjective_type_weight=float(planner_edge_pair_multiobjective_type_weight),
            multiobjective_order_weight=float(planner_edge_pair_multiobjective_order_weight),
            gate_to_action_edge=planner_projection_change_source == "terminal_edit_inside_action_edge",
            require_vacancy_atom_pair=bool(planner_edge_completion_require_vacancy_atom_pair),
        )
    elif planner_projection_change_source in {"terminal_edit_sequence_rollout", "terminal_edit_inside_sequence_rollout"}:
        projection_change_logits, edge_completion_mask = _sequence_rollout_projection_logits(
            model=model,
            site_latent=site_latent,
            patch_latent=patch_latent,
            predicted_next_global=next_pred,
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
            nn1_offsets=np.asarray(env.env.NN1, dtype=np.int64),
            anchor_source=planner_edge_completion_anchor_source,
            anchor_budget=int(planner_edge_completion_anchor_budget),
            destinations_per_anchor=int(planner_edge_completion_destinations_per_anchor),
            global_pair_budget=int(planner_edge_completion_global_pair_budget),
            blend_alpha=float(planner_projection_change_blend_alpha),
            support_blend_alpha=float(planner_projection_change_blend_alpha),
            multiobjective_type_weight=float(planner_edge_pair_multiobjective_type_weight),
            multiobjective_order_weight=float(planner_edge_pair_multiobjective_order_weight),
            gate_to_rollout=planner_projection_change_source == "terminal_edit_inside_sequence_rollout",
            require_vacancy_atom_pair=bool(planner_edge_completion_require_vacancy_atom_pair),
        )
    elif planner_projection_change_source == "two_stage_vacancy_displacement":
        projection_change_logits, projection_type_logits, edge_completion_mask = _action_edge_pair_vacancy_displacement_logits(
            model=model,
            site_latent=site_latent,
            patch_latent=patch_latent,
            predicted_next_global=next_pred,
            path_latent=path_latent,
            horizon_k=tensors["horizon_k"],
            current_types=tensors["current_types"],
            candidate_positions=tensors["candidate_positions"],
            raw_type_logits=raw_type_logits,
            change_logits=change_logits,
            proposal_logits=proposal_logits,
            action_support_logits=action_support_logits,
            action_source_logits=action_source_logits,
            action_destination_logits=action_destination_logits,
            candidate_mask=tensors["candidate_mask"],
            box_dims=tensors["box_dims"],
            nn1_offsets=np.asarray(env.env.NN1, dtype=np.int64),
            anchor_source=planner_edge_completion_anchor_source,
            anchor_budget=int(planner_edge_completion_anchor_budget),
            destinations_per_anchor=int(planner_edge_completion_destinations_per_anchor),
            global_pair_budget=int(planner_edge_completion_global_pair_budget),
            blend_alpha=float(planner_projection_change_blend_alpha),
            support_blend_alpha=float(planner_projection_change_blend_alpha),
            multiobjective_type_weight=float(planner_edge_pair_multiobjective_type_weight),
            multiobjective_order_weight=float(planner_edge_pair_multiobjective_order_weight),
            require_vacancy_atom_pair=bool(planner_edge_completion_require_vacancy_atom_pair),
        )
    elif planner_projection_change_source in {
        "vacancy_pair_completion",
        "vacancy_pair_energy_blend_completion",
        "vacancy_pair_interaction_completion",
        "vacancy_pair_interaction_energy_blend_completion",
    }:
        projection_change_logits, projection_type_logits, edge_completion_mask = _vacancy_pair_completion_logits(
            model=model,
            site_latent=site_latent,
            patch_latent=patch_latent,
            predicted_next_global=next_pred,
            path_latent=path_latent,
            horizon_k=tensors["horizon_k"],
            current_types=tensors["current_types"],
            raw_type_logits=raw_type_logits,
            change_logits=change_logits,
            proposal_logits=proposal_logits,
            action_support_logits=action_support_logits,
            action_source_logits=action_source_logits,
            action_destination_logits=action_destination_logits,
            candidate_positions=tensors["candidate_positions"],
            candidate_mask=tensors["candidate_mask"],
            anchor_source=planner_edge_completion_anchor_source,
            destination_source=planner_edge_completion_destination_source,
            anchor_budget=int(planner_edge_completion_anchor_budget),
            destinations_per_anchor=int(planner_edge_completion_destinations_per_anchor),
            global_pair_budget=int(planner_edge_completion_global_pair_budget),
            blend_alpha=float(planner_projection_change_blend_alpha),
            energy_blend=planner_projection_change_source
            in {"vacancy_pair_energy_blend_completion", "vacancy_pair_interaction_energy_blend_completion"},
            use_interaction_score=planner_projection_change_source
            in {"vacancy_pair_interaction_completion", "vacancy_pair_interaction_energy_blend_completion"},
            multiobjective_type_weight=float(planner_edge_pair_multiobjective_type_weight),
            multiobjective_order_weight=float(planner_edge_pair_multiobjective_order_weight),
            diagnostics=vacancy_pair_projection_diagnostics,
            rank_diagnostic=bool(planner_vacancy_pair_rank_diagnostic),
            rank_diagnostic_max_pairs=int(planner_vacancy_pair_rank_max_pairs),
            factorized_diagnostic=bool(planner_vacancy_pair_factorized_diagnostic),
            factorized_diagnostic_max_pairs=int(planner_vacancy_pair_factorized_max_pairs),
        )
    elif planner_projection_change_source == "terminal_typed_diff":
        terminal_action_context_logits = _projection_logits_from_source(
            change_logits=change_logits,
            proposal_logits=proposal_logits,
            action_support_logits=action_support_logits,
            action_source_logits=action_source_logits,
            action_destination_logits=action_destination_logits,
            source="action_endpoint",
            blend_alpha=float(planner_projection_change_blend_alpha),
        )
        projection_change_logits, projection_type_logits = _terminal_typed_diff_projection_logits(
            model=model,
            site_latent=site_latent,
            patch_latent=patch_latent,
            predicted_next_global=next_pred,
            path_latent=path_latent,
            horizon_k=tensors["horizon_k"],
            current_types=tensors["current_types"],
            action_context_logits=terminal_action_context_logits,
            raw_type_logits=raw_type_logits,
        )
    else:
        projection_change_logits = _projection_logits_from_source(
            change_logits=change_logits,
            proposal_logits=proposal_logits,
            action_support_logits=action_support_logits,
            action_source_logits=action_source_logits,
            action_destination_logits=action_destination_logits,
            source=planner_projection_change_source,
            blend_alpha=float(planner_projection_change_blend_alpha),
        )
    projection_topk_mask = torch.zeros_like(tensors["candidate_mask"], dtype=torch.bool)
    if planner_projection_topk_source != "none" and int(planner_projection_topk_budget) > 0:
        if planner_projection_topk_source == "action_edge_completion":
            ranking_logits, topk_edge_completion_mask = _action_edge_completion_logits(
                change_logits=change_logits,
                proposal_logits=proposal_logits,
                action_support_logits=action_support_logits,
                action_source_logits=action_source_logits,
                action_destination_logits=action_destination_logits,
                candidate_positions=tensors["candidate_positions"],
                candidate_mask=tensors["candidate_mask"],
                box_dims=tensors["box_dims"],
                nn1_offsets=np.asarray(env.env.NN1, dtype=np.int64),
                anchor_source=planner_edge_completion_anchor_source,
                destination_source=planner_edge_completion_destination_source,
                anchor_budget=int(planner_edge_completion_anchor_budget),
                destinations_per_anchor=int(planner_edge_completion_destinations_per_anchor),
                blend_alpha=float(planner_projection_change_blend_alpha),
            )
            edge_completion_mask = edge_completion_mask | topk_edge_completion_mask
        elif planner_projection_topk_source in {
            "action_edge_pair_completion",
            "action_edge_pair_support_completion",
            "action_edge_pair_blend_completion",
            "action_edge_pair_multiobjective_completion",
        }:
            ranking_logits, topk_edge_completion_mask = _action_edge_pair_completion_logits(
                model=model,
                site_latent=site_latent,
                patch_latent=patch_latent,
                predicted_next_global=next_pred,
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
                nn1_offsets=np.asarray(env.env.NN1, dtype=np.int64),
                anchor_source=planner_edge_completion_anchor_source,
                anchor_budget=int(planner_edge_completion_anchor_budget),
                destinations_per_anchor=int(planner_edge_completion_destinations_per_anchor),
                global_pair_budget=int(planner_edge_completion_global_pair_budget),
                destination_scope=planner_edge_completion_destination_scope,
                blend_alpha=float(planner_projection_change_blend_alpha),
                score_source=_edge_pair_completion_score_source(planner_projection_topk_source),
                support_blend_alpha=float(planner_projection_change_blend_alpha),
                multiobjective_type_weight=float(planner_edge_pair_multiobjective_type_weight),
                multiobjective_order_weight=float(planner_edge_pair_multiobjective_order_weight),
                require_vacancy_atom_pair=bool(planner_edge_completion_require_vacancy_atom_pair),
            )
            edge_completion_mask = edge_completion_mask | topk_edge_completion_mask
        elif planner_projection_topk_source in {"terminal_edit_decoupled", "terminal_edit_inside_action_edge"}:
            ranking_logits, topk_edge_completion_mask = _terminal_decoupled_projection_logits(
                model=model,
                site_latent=site_latent,
                patch_latent=patch_latent,
                predicted_next_global=next_pred,
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
                nn1_offsets=np.asarray(env.env.NN1, dtype=np.int64),
                anchor_source=planner_edge_completion_anchor_source,
                anchor_budget=int(planner_edge_completion_anchor_budget),
                destinations_per_anchor=int(planner_edge_completion_destinations_per_anchor),
                global_pair_budget=int(planner_edge_completion_global_pair_budget),
                blend_alpha=float(planner_projection_change_blend_alpha),
                support_blend_alpha=float(planner_projection_change_blend_alpha),
                multiobjective_type_weight=float(planner_edge_pair_multiobjective_type_weight),
                multiobjective_order_weight=float(planner_edge_pair_multiobjective_order_weight),
                gate_to_action_edge=planner_projection_topk_source == "terminal_edit_inside_action_edge",
                require_vacancy_atom_pair=bool(planner_edge_completion_require_vacancy_atom_pair),
            )
            edge_completion_mask = edge_completion_mask | topk_edge_completion_mask
        elif planner_projection_topk_source in {"terminal_edit_sequence_rollout", "terminal_edit_inside_sequence_rollout"}:
            ranking_logits, topk_edge_completion_mask = _sequence_rollout_projection_logits(
                model=model,
                site_latent=site_latent,
                patch_latent=patch_latent,
                predicted_next_global=next_pred,
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
                nn1_offsets=np.asarray(env.env.NN1, dtype=np.int64),
                anchor_source=planner_edge_completion_anchor_source,
                anchor_budget=int(planner_edge_completion_anchor_budget),
                destinations_per_anchor=int(planner_edge_completion_destinations_per_anchor),
                global_pair_budget=int(planner_edge_completion_global_pair_budget),
                blend_alpha=float(planner_projection_change_blend_alpha),
                support_blend_alpha=float(planner_projection_change_blend_alpha),
                multiobjective_type_weight=float(planner_edge_pair_multiobjective_type_weight),
                multiobjective_order_weight=float(planner_edge_pair_multiobjective_order_weight),
                gate_to_rollout=planner_projection_topk_source == "terminal_edit_inside_sequence_rollout",
                require_vacancy_atom_pair=bool(planner_edge_completion_require_vacancy_atom_pair),
            )
            edge_completion_mask = edge_completion_mask | topk_edge_completion_mask
        elif planner_projection_topk_source == "two_stage_vacancy_displacement":
            ranking_logits, _ranking_type_logits, topk_edge_completion_mask = _action_edge_pair_vacancy_displacement_logits(
                model=model,
                site_latent=site_latent,
                patch_latent=patch_latent,
                predicted_next_global=next_pred,
                path_latent=path_latent,
                horizon_k=tensors["horizon_k"],
                current_types=tensors["current_types"],
                candidate_positions=tensors["candidate_positions"],
                raw_type_logits=raw_type_logits,
                change_logits=change_logits,
                proposal_logits=proposal_logits,
                action_support_logits=action_support_logits,
                action_source_logits=action_source_logits,
                action_destination_logits=action_destination_logits,
                candidate_mask=tensors["candidate_mask"],
                box_dims=tensors["box_dims"],
                nn1_offsets=np.asarray(env.env.NN1, dtype=np.int64),
                anchor_source=planner_edge_completion_anchor_source,
                anchor_budget=int(planner_edge_completion_anchor_budget),
                destinations_per_anchor=int(planner_edge_completion_destinations_per_anchor),
                global_pair_budget=int(planner_edge_completion_global_pair_budget),
                blend_alpha=float(planner_projection_change_blend_alpha),
                support_blend_alpha=float(planner_projection_change_blend_alpha),
                multiobjective_type_weight=float(planner_edge_pair_multiobjective_type_weight),
                multiobjective_order_weight=float(planner_edge_pair_multiobjective_order_weight),
                require_vacancy_atom_pair=bool(planner_edge_completion_require_vacancy_atom_pair),
            )
            edge_completion_mask = edge_completion_mask | topk_edge_completion_mask
        elif planner_projection_topk_source in {
            "vacancy_pair_completion",
            "vacancy_pair_energy_blend_completion",
            "vacancy_pair_interaction_completion",
            "vacancy_pair_interaction_energy_blend_completion",
        }:
            ranking_logits, _ranking_type_logits, topk_edge_completion_mask = _vacancy_pair_completion_logits(
                model=model,
                site_latent=site_latent,
                patch_latent=patch_latent,
                predicted_next_global=next_pred,
                path_latent=path_latent,
                horizon_k=tensors["horizon_k"],
                current_types=tensors["current_types"],
                raw_type_logits=raw_type_logits,
                change_logits=change_logits,
                proposal_logits=proposal_logits,
                action_support_logits=action_support_logits,
                action_source_logits=action_source_logits,
                action_destination_logits=action_destination_logits,
                candidate_positions=tensors["candidate_positions"],
                candidate_mask=tensors["candidate_mask"],
                anchor_source=planner_edge_completion_anchor_source,
                destination_source=planner_edge_completion_destination_source,
                anchor_budget=int(planner_edge_completion_anchor_budget),
                destinations_per_anchor=int(planner_edge_completion_destinations_per_anchor),
                global_pair_budget=int(planner_edge_completion_global_pair_budget),
                blend_alpha=float(planner_projection_change_blend_alpha),
                energy_blend=planner_projection_topk_source
                in {"vacancy_pair_energy_blend_completion", "vacancy_pair_interaction_energy_blend_completion"},
                use_interaction_score=planner_projection_topk_source
                in {"vacancy_pair_interaction_completion", "vacancy_pair_interaction_energy_blend_completion"},
                multiobjective_type_weight=float(planner_edge_pair_multiobjective_type_weight),
                multiobjective_order_weight=float(planner_edge_pair_multiobjective_order_weight),
            )
            edge_completion_mask = edge_completion_mask | topk_edge_completion_mask
        elif planner_projection_topk_source == "terminal_typed_diff":
            terminal_action_context_logits = _projection_logits_from_source(
                change_logits=change_logits,
                proposal_logits=proposal_logits,
                action_support_logits=action_support_logits,
                action_source_logits=action_source_logits,
                action_destination_logits=action_destination_logits,
                source="action_endpoint",
                blend_alpha=float(planner_projection_change_blend_alpha),
            )
            ranking_logits, _ranking_type_logits = _terminal_typed_diff_projection_logits(
                model=model,
                site_latent=site_latent,
                patch_latent=patch_latent,
                predicted_next_global=next_pred,
                path_latent=path_latent,
                horizon_k=tensors["horizon_k"],
                current_types=tensors["current_types"],
                action_context_logits=terminal_action_context_logits,
                raw_type_logits=raw_type_logits,
            )
        else:
            ranking_logits = _projection_logits_from_source(
                change_logits=change_logits,
                proposal_logits=proposal_logits,
                action_support_logits=action_support_logits,
                action_source_logits=action_source_logits,
                action_destination_logits=action_destination_logits,
                source=planner_projection_topk_source,
                blend_alpha=float(planner_projection_change_blend_alpha),
            )
        projection_change_logits, projection_type_logits, projection_topk_mask = _apply_projection_topk_support(
            projection_change_logits=projection_change_logits,
            projection_type_logits=projection_type_logits,
            ranking_logits=ranking_logits,
            candidate_mask=tensors["candidate_mask"],
            current_types=tensors["current_types"],
            topk_budget=int(planner_projection_topk_budget),
        )
    projected_types, projected_changed_mask, transport_cost, reachability_violation = mod.project_types_by_inventory(
        current_types=tensors["current_types"],
        change_logits=projection_change_logits,
        type_logits=projection_type_logits,
        node_mask=tensors["candidate_mask"],
        positions=tensors["candidate_positions"],
        box_dims=tensors["box_dims"],
        horizon_k=tensors["horizon_k"],
        max_changed_sites=2 * tensors["horizon_k"],
    )
    reward_patch_latent = patch_latent
    reward_change_logits = change_logits
    reward_type_logits = raw_type_logits
    reward_global_latent = global_latent
    reward_path_latent = path_latent
    reward_next_pred = next_pred
    reward_head_model = model if reward_model is None else reward_model
    if reward_model is not None and reward_model is not model:
        reward_global_latent = reward_model.encode_global(tensors["start_obs"])
        reward_site_latent, reward_patch_latent = reward_model.encode_patch(
            positions=tensors["candidate_positions"],
            nearest_vacancy_offset=tensors["nearest_vacancy_offset"],
            reach_depth=tensors["reach_depth"],
            is_start_vacancy=tensors["is_start_vacancy"],
            type_ids=tensors["current_types"],
            node_mask=tensors["candidate_mask"],
            global_summary=tensors["global_summary"],
            box_dims=tensors["box_dims"],
        )
        reward_prior_mu, reward_prior_logvar = reward_model.prior_stats(
            reward_global_latent,
            tensors["global_summary"],
            tensors["horizon_k"],
        )
        reward_path_latent = reward_model.sample_path_latent(
            reward_prior_mu,
            reward_prior_logvar,
            deterministic=True,
        )
        reward_next_pred = reward_model.predict_next_global(
            reward_global_latent,
            reward_path_latent,
            tensors["horizon_k"],
        )
        reward_change_logits, reward_type_logits = reward_model.decode_edit(
            site_latent=reward_site_latent,
            patch_latent=reward_patch_latent,
            predicted_next_global=reward_next_pred,
            path_latent=reward_path_latent,
            horizon_k=tensors["horizon_k"],
            current_types=tensors["current_types"],
        )
    if reward_prediction_source == "projected":
        reward_projected_types = projected_types
        if (
            reward_model is not None
            and reward_model is not model
            and aux_projected_types_source == "aux"
        ):
            reward_projected_types, _, _, _ = mod.project_types_by_inventory(
                current_types=tensors["current_types"],
                change_logits=reward_change_logits,
                type_logits=reward_type_logits,
                node_mask=tensors["candidate_mask"],
                positions=tensors["candidate_positions"],
                box_dims=tensors["box_dims"],
                horizon_k=tensors["horizon_k"],
                max_changed_sites=2 * tensors["horizon_k"],
            )
        _, reward_patch_latent = model.encode_patch(
            positions=tensors["candidate_positions"],
            nearest_vacancy_offset=tensors["nearest_vacancy_offset"],
            reach_depth=tensors["reach_depth"],
            is_start_vacancy=tensors["is_start_vacancy"],
            type_ids=reward_projected_types,
            node_mask=tensors["candidate_mask"],
            global_summary=tensors["global_summary"],
            box_dims=tensors["box_dims"],
        )
        reward_change_logits, reward_type_logits = mod.projected_edit_logits_from_types(
            current_types=tensors["current_types"],
            projected_types=reward_projected_types,
            candidate_mask=tensors["candidate_mask"],
        )
        if reward_model is not None and reward_model is not model:
            _, reward_patch_latent = reward_model.encode_patch(
                positions=tensors["candidate_positions"],
                nearest_vacancy_offset=tensors["nearest_vacancy_offset"],
                reach_depth=tensors["reach_depth"],
                is_start_vacancy=tensors["is_start_vacancy"],
                type_ids=reward_projected_types,
                node_mask=tensors["candidate_mask"],
                global_summary=tensors["global_summary"],
                box_dims=tensors["box_dims"],
            )
    reward_change_logits, reward_type_logits = mod._select_reward_edit_context(
        reward_edit_context_source,
        reward_change_logits,
        reward_type_logits,
    )
    primary_outputs = mod._predict_reward_and_duration_outputs(
        reward_head_model,
        reward_global_latent,
        reward_next_pred,
        reward_path_latent,
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
            if aux_projected_types_source == "primary":
                duration_projected_types = projected_types
            else:
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
        duration_change_logits_for_head, duration_type_logits_for_head = mod._select_reward_edit_context(
            reward_edit_context_source,
            duration_change_logits_for_head,
            duration_type_logits_for_head,
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
    noop_risk_prob = float(torch.sigmoid(primary_outputs.get("noop_risk_logit", torch.zeros_like(primary_outputs["reward"]))).item())
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
    proposal_prob = torch.sigmoid(proposal_logits) * tensors["candidate_mask"]
    proposal_support_mass = float(proposal_prob.sum().item())
    proposal_support_density = float(proposal_support_mass / max(float(tensors["candidate_mask"].sum().item()), 1.0))
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
        planner_noop_risk_penalty=planner_noop_risk_penalty,
        noop_risk_prob=noop_risk_prob,
        planner_tau_blend_alpha=planner_tau_blend_alpha,
        planner_tau_log_offset=planner_tau_log_offset,
    )
    if violation > 0.0:
        selection_score = -float("inf")
    elif float(planner_proposal_score_weight) != 0.0:
        selection_score += float(planner_proposal_score_weight) * float(np.log1p(max(proposal_support_mass, 0.0)))
    if violation <= 0.0 and float(planner_candidate_quality_score_weight) != 0.0:
        selection_score += float(planner_candidate_quality_score_weight) * candidate_quality_score
    result = {
        "segment_k": int(horizon_k),
        "predicted_reward_sum": pred_reward,
        "predicted_delta_e": float(pred_reward / reward_scale),
        "predicted_reward_raw": reward_raw,
        "predicted_reward_gate_prob": reward_gate_prob,
        "predicted_noop_risk_prob": noop_risk_prob,
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
        "reward_edit_context_source": reward_edit_context_source,
        "reachability_violation": violation,
        "projected_changed_count": changed_count,
        "proposal_support_mass": proposal_support_mass,
        "proposal_support_density": proposal_support_density,
        "candidate_quality_score": candidate_quality_score,
        "planner_projection_change_source": planner_projection_change_source,
        "planner_projection_topk_source": planner_projection_topk_source,
        "planner_projection_topk_budget": int(planner_projection_topk_budget),
        "planner_projection_topk_count": int(projection_topk_mask.sum().item()),
        "planner_edge_completion_anchor_source": planner_edge_completion_anchor_source,
        "planner_edge_completion_destination_source": planner_edge_completion_destination_source,
        "planner_edge_completion_anchor_budget": int(planner_edge_completion_anchor_budget),
        "planner_edge_completion_destinations_per_anchor": int(planner_edge_completion_destinations_per_anchor),
        "planner_edge_completion_global_pair_budget": int(planner_edge_completion_global_pair_budget),
        "planner_edge_completion_require_vacancy_atom_pair": bool(planner_edge_completion_require_vacancy_atom_pair),
        "planner_edge_completion_support_count": int(edge_completion_mask.sum().item()),
        "transport_cost": float(transport_cost.item()),
        "selection_score": float(selection_score),
    }
    if vacancy_pair_projection_diagnostics:
        result["vacancy_pair_projection_diagnostic"] = vacancy_pair_projection_diagnostics[0]
    if proposal_diagnostic:
        max_sites = max(int(proposal_diagnostic_max_sites), 0)
        mask_np = tensors["candidate_mask"][0].detach().cpu().numpy().astype(bool)
        positions_np = tensors["candidate_positions"][0].detach().cpu().numpy()
        current_np = tensors["current_types"][0].detach().cpu().numpy()
        projected_np = projected_types[0].detach().cpu().numpy()
        changed_np = projected_changed_mask[0].detach().cpu().numpy().astype(bool) & mask_np
        changed_indices = np.flatnonzero(changed_np)
        if max_sites > 0:
            changed_indices = changed_indices[:max_sites]
        change_prob_np = torch.sigmoid(projection_change_logits[0]).detach().cpu().numpy()
        proposal_prob_np = torch.sigmoid(proposal_logits[0]).detach().cpu().numpy()
        valid_indices = np.flatnonzero(mask_np)
        top_budget = min(max_sites, valid_indices.size)
        if top_budget > 0:
            order = valid_indices[np.argsort(change_prob_np[valid_indices])[::-1][:top_budget]]
        else:
            order = np.asarray([], dtype=np.int64)
        result["proposal_diagnostic"] = {
            "candidate_count": int(mask_np.sum()),
            "projected_changed_indices": [int(i) for i in changed_indices.tolist()],
            "projected_changed_positions": [
                [int(round(x)) for x in positions_np[int(i)].tolist()]
                for i in changed_indices.tolist()
            ],
            "projected_changed_current_types": [int(current_np[int(i)]) for i in changed_indices.tolist()],
            "projected_changed_target_types": [int(projected_np[int(i)]) for i in changed_indices.tolist()],
            "top_change_probability_indices": [int(i) for i in order.tolist()],
            "top_change_probability_positions": [
                [int(round(x)) for x in positions_np[int(i)].tolist()]
                for i in order.tolist()
            ],
            "top_change_probability_values": [float(change_prob_np[int(i)]) for i in order.tolist()],
            "change_probability_mass": float(change_prob_np[mask_np].sum()),
            "proposal_probability_mass": float(proposal_prob_np[mask_np].sum()),
        }
        if proposal_diagnostic_store_candidate_positions:
            result["proposal_diagnostic"]["candidate_positions"] = [
                [int(round(x)) for x in positions_np[int(i)].tolist()]
                for i in valid_indices.tolist()
            ]
    return result


def _proposal_overlap_summary(selected: dict[str, object], teacher_segment: dict[str, object]) -> dict[str, object]:
    candidate_summary = selected.get("proposal_diagnostic")
    if not isinstance(candidate_summary, dict):
        return {}
    projected_positions = {
        tuple(int(v) for v in item)
        for item in candidate_summary.get("projected_changed_positions", [])
    }
    teacher_positions = {
        tuple(int(v) for v in item)
        for item in teacher_segment.get("changed_positions", [])
    }
    overlap = projected_positions & teacher_positions
    precision = len(overlap) / max(len(projected_positions), 1)
    recall = len(overlap) / max(len(teacher_positions), 1)
    if precision + recall > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    candidate_positions = {
        tuple(int(v) for v in item)
        for item in candidate_summary.get("candidate_positions", [])
    }
    candidate_teacher_overlap = candidate_positions & teacher_positions
    teacher_missing_from_candidate = teacher_positions - candidate_positions
    summary = {
        "teacher_changed_positions": [list(item) for item in sorted(teacher_positions)],
        "overlap_positions": [list(item) for item in sorted(overlap)],
        "projected_changed_count": int(len(projected_positions)),
        "teacher_changed_count": int(len(teacher_positions)),
        "overlap_count": int(len(overlap)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
    if candidate_positions:
        summary.update(
            {
                "candidate_position_count": int(len(candidate_positions)),
                "candidate_teacher_overlap_count": int(len(candidate_teacher_overlap)),
                "candidate_teacher_precision": float(
                    len(candidate_teacher_overlap) / max(len(candidate_positions), 1)
                ),
                "candidate_teacher_recall": float(
                    len(candidate_teacher_overlap) / max(len(teacher_positions), 1)
                ),
                "teacher_missing_from_candidate_count": int(len(teacher_missing_from_candidate)),
                "teacher_missing_from_candidate_positions": [
                    list(item) for item in sorted(teacher_missing_from_candidate)
                ],
            }
        )
    return summary


def _vacancy_pair_overlap_summary(selected: dict[str, object], teacher_segment: dict[str, object]) -> dict[str, object]:
    diagnostic = selected.get("vacancy_pair_projection_diagnostic")
    if not isinstance(diagnostic, dict):
        return {}
    selected_pairs_raw = diagnostic.get("selected_pairs", [])
    if not isinstance(selected_pairs_raw, list):
        return {}
    selected_pairs: dict[tuple[tuple[int, int, int], tuple[int, int, int]], int] = {}
    for item in selected_pairs_raw:
        if not isinstance(item, dict):
            continue
        source = item.get("source_position")
        destination = item.get("destination_position")
        if not (
            isinstance(source, (list, tuple))
            and isinstance(destination, (list, tuple))
            and len(source) == 3
            and len(destination) == 3
        ):
            continue
        key = (
            tuple(int(v) for v in source),
            tuple(int(v) for v in destination),
        )
        selected_pairs[key] = int(item.get("moving_type", -1))
    teacher_pairs_raw = teacher_segment.get("vacancy_pair_positions", [])
    if not isinstance(teacher_pairs_raw, list):
        teacher_pairs_raw = []
    teacher_pairs: dict[tuple[tuple[int, int, int], tuple[int, int, int]], int] = {}
    for item in teacher_pairs_raw:
        if not isinstance(item, dict):
            continue
        source = item.get("source")
        destination = item.get("destination")
        if not (
            isinstance(source, (list, tuple))
            and isinstance(destination, (list, tuple))
            and len(source) == 3
            and len(destination) == 3
        ):
            continue
        key = (
            tuple(int(v) for v in source),
            tuple(int(v) for v in destination),
        )
        teacher_pairs[key] = int(item.get("moving_type", -1))
    selected_set = set(selected_pairs)
    teacher_set = set(teacher_pairs)
    overlap = selected_set & teacher_set
    precision = len(overlap) / max(len(selected_set), 1)
    recall = len(overlap) / max(len(teacher_set), 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    typed_matches = sum(1 for pair in overlap if int(selected_pairs[pair]) == int(teacher_pairs[pair]))
    return {
        "selected_pair_count": int(len(selected_set)),
        "teacher_pair_count": int(len(teacher_set)),
        "overlap_pair_count": int(len(overlap)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "typed_endpoint_accuracy": float(typed_matches / max(len(overlap), 1)),
        "overlap_pairs": [
            {"source": list(source), "destination": list(destination)}
            for source, destination in sorted(overlap)
        ],
    }


def _vacancy_pair_rank_summary(selected: dict[str, object], teacher_segment: dict[str, object]) -> dict[str, object]:
    diagnostic = selected.get("vacancy_pair_projection_diagnostic")
    if not isinstance(diagnostic, dict):
        return {}
    ranked_pairs_raw = diagnostic.get("ranked_pair_scores", [])
    if not isinstance(ranked_pairs_raw, list) or not ranked_pairs_raw:
        return {}
    ranked_lookup: dict[tuple[tuple[int, int, int], tuple[int, int, int]], dict[str, object]] = {}
    ranked_items: list[dict[str, object]] = []
    for item in ranked_pairs_raw:
        if not isinstance(item, dict):
            continue
        source = item.get("source_position")
        destination = item.get("destination_position")
        if not (
            isinstance(source, (list, tuple))
            and isinstance(destination, (list, tuple))
            and len(source) == 3
            and len(destination) == 3
        ):
            continue
        key = (
            tuple(int(v) for v in source),
            tuple(int(v) for v in destination),
        )
        ranked = {
            "key": key,
            "rank": int(item.get("rank", len(ranked_lookup) + 1)),
            "score": _as_float(item.get("score", 0.0)),
            "moving_type": int(item.get("moving_type", -1)),
        }
        ranked_lookup[key] = ranked
        ranked_items.append(ranked)
    teacher_pairs_raw = teacher_segment.get("vacancy_pair_positions", [])
    if not isinstance(teacher_pairs_raw, list):
        teacher_pairs_raw = []
    teacher_pairs: dict[tuple[tuple[int, int, int], tuple[int, int, int]], int] = {}
    for item in teacher_pairs_raw:
        if not isinstance(item, dict):
            continue
        source = item.get("source")
        destination = item.get("destination")
        if not (
            isinstance(source, (list, tuple))
            and isinstance(destination, (list, tuple))
            and len(source) == 3
            and len(destination) == 3
        ):
            continue
        key = (
            tuple(int(v) for v in source),
            tuple(int(v) for v in destination),
        )
        teacher_pairs[key] = int(item.get("moving_type", -1))
    teacher_count = len(teacher_pairs)
    if teacher_count == 0:
        return {
            "ranked_pair_count": int(diagnostic.get("ranked_pair_score_count", len(ranked_lookup))),
            "teacher_pair_count": 0,
            "teacher_pair_found_count": 0,
            "teacher_pair_found_recall": 0.0,
        }
    ranks: list[int] = []
    scores: list[float] = []
    typed_matches = 0
    for pair, teacher_type in teacher_pairs.items():
        ranked = ranked_lookup.get(pair)
        if ranked is None:
            continue
        ranks.append(int(ranked["rank"]))
        scores.append(float(ranked["score"]))
        if int(ranked["moving_type"]) == int(teacher_type):
            typed_matches += 1
    ranked_pair_count = int(diagnostic.get("ranked_pair_score_count", len(ranked_lookup)))
    rank_array = np.asarray(ranks, dtype=np.float64)
    teacher_pair_set = set(teacher_pairs)
    teacher_source_set = {pair[0] for pair in teacher_pairs}
    teacher_destination_set = {pair[1] for pair in teacher_pairs}
    topk_false_positive_rate: dict[str, float] = {}
    topk_true_pair_count: dict[str, int] = {}
    topk_source_hard_negative_count: dict[str, int] = {}
    topk_destination_hard_negative_count: dict[str, int] = {}
    topk_source_destination_unpaired_count: dict[str, int] = {}
    topk_type_mismatch_count: dict[str, int] = {}
    topk_true_score_mean: dict[str, float] = {}
    topk_false_score_mean: dict[str, float] = {}
    for k in (8, 16, 32, 64, 128, 256, 512, 1024):
        top = [item for item in ranked_items if int(item.get("rank", ranked_pair_count + 1)) <= k]
        true_scores: list[float] = []
        false_scores: list[float] = []
        source_hard = 0
        destination_hard = 0
        source_destination_unpaired = 0
        type_mismatch = 0
        true_count = 0
        for item in top:
            pair = item.get("key")
            if not isinstance(pair, tuple):
                continue
            score = _as_float(item.get("score", 0.0))
            is_true_pair = pair in teacher_pair_set
            if is_true_pair:
                true_count += 1
                true_scores.append(score)
                if int(item.get("moving_type", -1)) != int(teacher_pairs[pair]):
                    type_mismatch += 1
            else:
                false_scores.append(score)
                source_match = pair[0] in teacher_source_set
                destination_match = pair[1] in teacher_destination_set
                if source_match:
                    source_hard += 1
                if destination_match:
                    destination_hard += 1
                if source_match and destination_match:
                    source_destination_unpaired += 1
        total = max(len(top), 1)
        key = str(k)
        topk_false_positive_rate[key] = float((len(top) - true_count) / total)
        topk_true_pair_count[key] = int(true_count)
        topk_source_hard_negative_count[key] = int(source_hard)
        topk_destination_hard_negative_count[key] = int(destination_hard)
        topk_source_destination_unpaired_count[key] = int(source_destination_unpaired)
        topk_type_mismatch_count[key] = int(type_mismatch)
        topk_true_score_mean[key] = float(np.mean(true_scores)) if true_scores else 0.0
        topk_false_score_mean[key] = float(np.mean(false_scores)) if false_scores else 0.0
    recall_at = {
        str(k): float(np.mean(rank_array <= k)) if rank_array.size else 0.0
        for k in (8, 16, 32, 64, 128, 256, 512, 1024)
    }
    return {
        "ranked_pair_count": int(ranked_pair_count),
        "teacher_pair_count": int(teacher_count),
        "teacher_pair_found_count": int(len(ranks)),
        "teacher_pair_found_recall": float(len(ranks) / max(teacher_count, 1)),
        "teacher_pair_rank_mean": float(np.mean(rank_array)) if rank_array.size else 0.0,
        "teacher_pair_rank_median": float(np.median(rank_array)) if rank_array.size else 0.0,
        "teacher_pair_rank_best": float(np.min(rank_array)) if rank_array.size else 0.0,
        "teacher_pair_rank_worst": float(np.max(rank_array)) if rank_array.size else 0.0,
        "teacher_pair_rank_percentile_mean": float(np.mean(rank_array / max(ranked_pair_count, 1))) if rank_array.size else 0.0,
        "teacher_pair_mrr": float(np.mean(1.0 / rank_array)) if rank_array.size else 0.0,
        "teacher_pair_score_mean": float(np.mean(scores)) if scores else 0.0,
        "teacher_pair_typed_rank_accuracy": float(typed_matches / max(len(ranks), 1)),
        "teacher_pair_recall_at_rank": recall_at,
        "topk_false_positive_rate": topk_false_positive_rate,
        "topk_true_pair_count": topk_true_pair_count,
        "topk_source_hard_negative_count": topk_source_hard_negative_count,
        "topk_destination_hard_negative_count": topk_destination_hard_negative_count,
        "topk_source_destination_unpaired_count": topk_source_destination_unpaired_count,
        "topk_type_mismatch_count": topk_type_mismatch_count,
        "topk_true_score_mean": topk_true_score_mean,
        "topk_false_score_mean": topk_false_score_mean,
    }


def _annotate_factorized_vacancy_pairs(candidate: dict[str, object], teacher_segment: dict[str, object]) -> None:
    diagnostic = candidate.get("vacancy_pair_projection_diagnostic")
    if not isinstance(diagnostic, dict):
        return
    factorized_scores = diagnostic.get("factorized_pair_scores")
    if not isinstance(factorized_scores, list) or not factorized_scores:
        return
    teacher_pairs_raw = teacher_segment.get("vacancy_pair_positions", [])
    if not isinstance(teacher_pairs_raw, list):
        teacher_pairs_raw = []
    teacher_pairs: dict[tuple[tuple[int, int, int], tuple[int, int, int]], int] = {}
    for item in teacher_pairs_raw:
        if not isinstance(item, dict):
            continue
        source = item.get("source")
        destination = item.get("destination")
        if not (
            isinstance(source, (list, tuple))
            and isinstance(destination, (list, tuple))
            and len(source) == 3
            and len(destination) == 3
        ):
            continue
        teacher_pairs[(tuple(int(v) for v in source), tuple(int(v) for v in destination))] = int(
            item.get("moving_type", -1)
        )
    teacher_sources = {pair[0] for pair in teacher_pairs}
    teacher_destinations = {pair[1] for pair in teacher_pairs}
    for item in factorized_scores:
        if not isinstance(item, dict):
            continue
        source = item.get("source_position")
        destination = item.get("destination_position")
        if not (
            isinstance(source, (list, tuple))
            and isinstance(destination, (list, tuple))
            and len(source) == 3
            and len(destination) == 3
        ):
            continue
        source_key = tuple(int(v) for v in source)
        destination_key = tuple(int(v) for v in destination)
        pair = (source_key, destination_key)
        source_match = source_key in teacher_sources
        destination_match = destination_key in teacher_destinations
        teacher_type = teacher_pairs.get(pair)
        if teacher_type is not None:
            label = "true_pair"
        elif source_match and destination_match:
            label = "source_destination_unpaired"
        elif source_match:
            label = "same_source_wrong_destination"
        elif destination_match:
            label = "same_destination_wrong_source"
        else:
            label = "false_pair"
        item["pair_label"] = label
        item["is_true_pair"] = bool(teacher_type is not None)
        item["is_teacher_source"] = bool(source_match)
        item["is_teacher_destination"] = bool(destination_match)
        item["teacher_moving_type"] = int(teacher_type) if teacher_type is not None else -1
        item["moving_type_matches_teacher"] = bool(
            teacher_type is not None and int(item.get("moving_type", -2)) == int(teacher_type)
        )


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _validate_pareto_teacher_label_diagnostic(args: argparse.Namespace) -> None:
    label_after_budget = bool(
        getattr(args, "planner_candidate_pareto_teacher_label_after_budget_projection", False)
    )
    if not label_after_budget:
        return
    if not getattr(args, "planner_candidate_pareto_selector_spec", None):
        raise ValueError(
            "--planner_candidate_pareto_teacher_label_after_budget_projection requires "
            "--planner_candidate_pareto_selector_spec."
        )
    if not bool(getattr(args, "planner_candidate_pareto_apply_budget_to_projection", False)):
        raise ValueError(
            "--planner_candidate_pareto_teacher_label_after_budget_projection requires "
            "--planner_candidate_pareto_apply_budget_to_projection so labels describe budgeted projections."
        )
    if getattr(args, "planner_teacher_overlap_oracle_mode", "none") != "add":
        raise ValueError(
            "--planner_candidate_pareto_teacher_label_after_budget_projection requires "
            "--planner_teacher_overlap_oracle_mode add."
        )
    if abs(_as_float(getattr(args, "planner_teacher_overlap_oracle_weight", 0.0))) > 1e-12:
        raise ValueError(
            "--planner_candidate_pareto_teacher_label_after_budget_projection requires "
            "--planner_teacher_overlap_oracle_weight 0.0 so teacher labels cannot alter selection."
        )


def _count_efficiency(projected_count: float, teacher_count: float) -> float:
    projected = max(float(projected_count), 0.0)
    teacher = max(float(teacher_count), 0.0)
    if projected <= 0.0 or teacher <= 0.0:
        return 0.0
    return float(min(projected / teacher, teacher / projected))


def _candidate_joint_diagnostic_record(
    candidate: dict[str, object],
    site_overlap: dict[str, float],
    vacancy_pair_overlap: dict[str, object],
    teacher_segment: dict[str, object],
    vacancy_pair_rank: dict[str, object] | None = None,
) -> dict[str, object]:
    teacher_reward = _as_float(teacher_segment.get("reward_sum"))
    teacher_tau = max(_as_float(teacher_segment.get("tau_exp")), 1e-12)
    projected_count = _as_float(
        site_overlap.get("projected_changed_count", candidate.get("projected_changed_count", 0.0))
    )
    teacher_changed_count = _as_float(
        site_overlap.get("teacher_changed_count", teacher_segment.get("changed_site_count", 0.0))
    )
    vacancy_selected_count = _as_float(
        vacancy_pair_overlap.get(
            "selected_pair_count",
            (candidate.get("vacancy_pair_projection_diagnostic") or {}).get("selected_pair_count", 0.0)
            if isinstance(candidate.get("vacancy_pair_projection_diagnostic"), dict)
            else 0.0,
        )
    )
    vacancy_teacher_count = _as_float(vacancy_pair_overlap.get("teacher_pair_count", 0.0))
    rank_summary = vacancy_pair_rank or {}
    rank_recall_at = rank_summary.get("teacher_pair_recall_at_rank", {})
    if not isinstance(rank_recall_at, dict):
        rank_recall_at = {}
    record = {
        "segment_k": int(candidate.get("segment_k", 0)),
        "pre_oracle_selection_score": _as_float(
            candidate.get("pre_oracle_selection_score", candidate.get("selection_score", 0.0))
        ),
        "selection_score": _as_float(candidate.get("selection_score", 0.0)),
        "candidate_quality_score": _as_float(candidate.get("candidate_quality_score", 0.0)),
        "model_reward_sum": _as_float(candidate.get("predicted_reward_sum", 0.0)),
        "model_delta_e": _as_float(candidate.get("predicted_delta_e", 0.0)),
        "model_expected_tau": _as_float(candidate.get("predicted_expected_tau", 0.0)),
        "model_tau_for_score": _as_float(candidate.get("planner_tau_for_score", 0.0)),
        "model_noop_risk": _as_float(candidate.get("predicted_noop_risk_prob", 0.0)),
        "teacher_reward_sum": teacher_reward,
        "teacher_tau_exp": teacher_tau,
        "teacher_reward_per_tau": float(teacher_reward / teacher_tau),
        "teacher_reward_per_sqrt_tau": float(teacher_reward / np.sqrt(teacher_tau)),
        "teacher_is_noop": bool(teacher_segment.get("is_noop", False)),
        "site_precision": _as_float(site_overlap.get("precision", 0.0)),
        "site_recall": _as_float(site_overlap.get("recall", 0.0)),
        "site_f1": _as_float(site_overlap.get("f1", 0.0)),
        "site_overlap_count": _as_float(site_overlap.get("overlap_count", 0.0)),
        "projected_changed_count": projected_count,
        "teacher_changed_count": teacher_changed_count,
        "site_count_efficiency": _count_efficiency(projected_count, teacher_changed_count),
        "vacancy_pair_precision": _as_float(vacancy_pair_overlap.get("precision", 0.0)),
        "vacancy_pair_recall": _as_float(vacancy_pair_overlap.get("recall", 0.0)),
        "vacancy_pair_f1": _as_float(vacancy_pair_overlap.get("f1", 0.0)),
        "vacancy_pair_overlap_count": _as_float(vacancy_pair_overlap.get("overlap_pair_count", 0.0)),
        "vacancy_pair_selected_count": vacancy_selected_count,
        "vacancy_pair_teacher_count": vacancy_teacher_count,
        "vacancy_pair_count_efficiency": _count_efficiency(vacancy_selected_count, vacancy_teacher_count),
        "vacancy_pair_typed_endpoint_accuracy": _as_float(
            vacancy_pair_overlap.get("typed_endpoint_accuracy", 0.0)
        ),
        "planner_edge_completion_support_count": _as_float(
            candidate.get("planner_edge_completion_support_count", 0.0)
        ),
        "proposal_support_mass": _as_float(candidate.get("proposal_support_mass", 0.0)),
        "proposal_support_density": _as_float(candidate.get("proposal_support_density", 0.0)),
    }
    if rank_summary:
        record.update(
            {
                "vacancy_pair_ranked_pair_count": _as_float(rank_summary.get("ranked_pair_count", 0.0)),
                "vacancy_pair_rank_teacher_pair_count": _as_float(rank_summary.get("teacher_pair_count", 0.0)),
                "vacancy_pair_rank_found_count": _as_float(rank_summary.get("teacher_pair_found_count", 0.0)),
                "vacancy_pair_rank_found_recall": _as_float(rank_summary.get("teacher_pair_found_recall", 0.0)),
                "vacancy_pair_rank_mean": _as_float(rank_summary.get("teacher_pair_rank_mean", 0.0)),
                "vacancy_pair_rank_median": _as_float(rank_summary.get("teacher_pair_rank_median", 0.0)),
                "vacancy_pair_rank_best": _as_float(rank_summary.get("teacher_pair_rank_best", 0.0)),
                "vacancy_pair_rank_worst": _as_float(rank_summary.get("teacher_pair_rank_worst", 0.0)),
                "vacancy_pair_rank_percentile_mean": _as_float(
                    rank_summary.get("teacher_pair_rank_percentile_mean", 0.0)
                ),
                "vacancy_pair_rank_score_mean": _as_float(rank_summary.get("teacher_pair_score_mean", 0.0)),
                "vacancy_pair_rank_typed_accuracy": _as_float(
                    rank_summary.get("teacher_pair_typed_rank_accuracy", 0.0)
                ),
            }
        )
        for k in (8, 16, 32, 64, 128, 256, 512, 1024):
            record[f"vacancy_pair_rank_recall_at_{k}"] = _as_float(rank_recall_at.get(str(k), 0.0))
    return record


def _candidate_joint_group_summary(records: list[dict[str, object]]) -> dict[str, float]:
    if not records:
        return {}
    fields = [
        "site_precision",
        "site_recall",
        "site_f1",
        "site_count_efficiency",
        "vacancy_pair_precision",
        "vacancy_pair_recall",
        "vacancy_pair_f1",
        "vacancy_pair_count_efficiency",
        "vacancy_pair_typed_endpoint_accuracy",
        "teacher_reward_sum",
        "teacher_tau_exp",
        "teacher_reward_per_tau",
        "teacher_reward_per_sqrt_tau",
        "model_reward_sum",
        "model_expected_tau",
        "model_noop_risk",
        "projected_changed_count",
        "teacher_changed_count",
        "vacancy_pair_selected_count",
        "vacancy_pair_teacher_count",
        "planner_edge_completion_support_count",
        "proposal_support_mass",
        "proposal_support_density",
        "vacancy_pair_ranked_pair_count",
        "vacancy_pair_rank_teacher_pair_count",
        "vacancy_pair_rank_found_count",
        "vacancy_pair_rank_found_recall",
        "vacancy_pair_rank_mean",
        "vacancy_pair_rank_median",
        "vacancy_pair_rank_best",
        "vacancy_pair_rank_worst",
        "vacancy_pair_rank_percentile_mean",
        "vacancy_pair_rank_score_mean",
        "vacancy_pair_rank_typed_accuracy",
        "vacancy_pair_rank_recall_at_8",
        "vacancy_pair_rank_recall_at_16",
        "vacancy_pair_rank_recall_at_32",
        "vacancy_pair_rank_recall_at_64",
        "vacancy_pair_rank_recall_at_128",
        "vacancy_pair_rank_recall_at_256",
        "vacancy_pair_rank_recall_at_512",
        "vacancy_pair_rank_recall_at_1024",
    ]
    return {
        f"avg_{field}": float(np.mean([_as_float(record.get(field, 0.0)) for record in records]))
        for field in fields
    }


def _candidate_joint_selector_score(
    records: list[dict[str, object]],
    record: dict[str, object],
    selector: str,
) -> float:
    if selector == "pre_oracle_selection_score":
        return _as_float(record.get("pre_oracle_selection_score", -float("inf")))
    if selector == "site_f1":
        return _as_float(record.get("site_f1", 0.0))
    if selector == "vacancy_pair_f1":
        return _as_float(record.get("vacancy_pair_f1", 0.0))
    if selector == "teacher_reward_sum":
        return _as_float(record.get("teacher_reward_sum", 0.0))
    if selector == "teacher_reward_per_sqrt_tau":
        return _as_float(record.get("teacher_reward_per_sqrt_tau", 0.0))
    if selector == "joint_reward_site_vacancy":
        rewards = [_as_float(item.get("teacher_reward_sum", 0.0)) for item in records]
        reward_min = min(rewards) if rewards else 0.0
        reward_span = max((max(rewards) - reward_min) if rewards else 0.0, 1e-12)
        reward_norm = (_as_float(record.get("teacher_reward_sum", 0.0)) - reward_min) / reward_span
        return float(
            0.40 * reward_norm
            + 0.25 * _as_float(record.get("site_f1", 0.0))
            + 0.25 * _as_float(record.get("vacancy_pair_f1", 0.0))
            + 0.05 * _as_float(record.get("site_count_efficiency", 0.0))
            + 0.05 * _as_float(record.get("vacancy_pair_count_efficiency", 0.0))
        )
    return _as_float(record.get(selector, 0.0))


def _candidate_joint_diagnostic_summary(segments: list[dict[str, object]]) -> dict[str, object]:
    per_segment: list[dict[str, object]] = []
    all_records: list[dict[str, object]] = []
    selected_records: list[dict[str, object]] = []
    selector_records: dict[str, list[dict[str, object]]] = {
        "pre_oracle_selection_score": [],
        "site_f1": [],
        "vacancy_pair_f1": [],
        "teacher_reward_sum": [],
        "teacher_reward_per_sqrt_tau": [],
        "joint_reward_site_vacancy": [],
    }
    for segment in segments:
        records = [
            item.get("candidate_joint_diagnostic", {})
            for item in segment.get("planner_candidates", [])
            if isinstance(item, dict) and isinstance(item.get("candidate_joint_diagnostic"), dict)
        ]
        records = [item for item in records if item]
        if not records:
            continue
        all_records.extend(records)
        selected = [
            item for item in records
            if bool(item.get("selected_by_planner", False))
        ]
        selected_records.extend(selected)
        selector_picks: dict[str, dict[str, object]] = {}
        for selector in selector_records:
            pick = max(records, key=lambda item, name=selector: _candidate_joint_selector_score(records, item, name))
            selector_records[selector].append(pick)
            selector_picks[selector] = {
                "segment_k": int(pick.get("segment_k", 0)),
                "site_f1": _as_float(pick.get("site_f1", 0.0)),
                "vacancy_pair_f1": _as_float(pick.get("vacancy_pair_f1", 0.0)),
                "teacher_reward_sum": _as_float(pick.get("teacher_reward_sum", 0.0)),
                "teacher_tau_exp": _as_float(pick.get("teacher_tau_exp", 0.0)),
                "projected_changed_count": _as_float(pick.get("projected_changed_count", 0.0)),
                "vacancy_pair_selected_count": _as_float(pick.get("vacancy_pair_selected_count", 0.0)),
            }
        per_segment.append(
            {
                "index": int(segment.get("index", len(per_segment))),
                "candidate_count": int(len(records)),
                "selected": {
                    "segment_k": int(selected[0].get("segment_k", segment.get("segment_k", 0))) if selected else int(segment.get("segment_k", 0)),
                    "site_f1": _as_float(selected[0].get("site_f1", 0.0)) if selected else 0.0,
                    "vacancy_pair_f1": _as_float(selected[0].get("vacancy_pair_f1", 0.0)) if selected else 0.0,
                    "teacher_reward_sum": _as_float(selected[0].get("teacher_reward_sum", 0.0)) if selected else 0.0,
                    "teacher_tau_exp": _as_float(selected[0].get("teacher_tau_exp", 0.0)) if selected else 0.0,
                },
                "selector_picks": selector_picks,
            }
        )
    return {
        "candidate_count": int(len(all_records)),
        "segment_count": int(len(per_segment)),
        "all_candidates": _candidate_joint_group_summary(all_records),
        "selected_by_planner": _candidate_joint_group_summary(selected_records),
        "selector_upper_bounds": {
            selector: _candidate_joint_group_summary(records)
            for selector, records in selector_records.items()
        },
        "segment_preview": per_segment[:20],
    }


def _planner_candidates_for_output(
    candidates: list[dict[str, object]],
    *,
    compact: bool = False,
) -> list[dict[str, object]]:
    if not compact:
        return candidates
    compacted: list[dict[str, object]] = []
    for candidate in candidates:
        item: dict[str, object] = {
            "segment_k": int(candidate.get("segment_k", 0)),
            "selection_score": _as_float(candidate.get("selection_score", 0.0)),
            "pre_oracle_selection_score": _as_float(
                candidate.get("pre_oracle_selection_score", candidate.get("selection_score", 0.0))
            ),
            "predicted_reward_sum": _as_float(candidate.get("predicted_reward_sum", 0.0)),
            "predicted_delta_e": _as_float(candidate.get("predicted_delta_e", 0.0)),
            "predicted_expected_tau": _as_float(candidate.get("predicted_expected_tau", 0.0)),
            "predicted_noop_risk_prob": _as_float(candidate.get("predicted_noop_risk_prob", 0.0)),
            "projected_changed_count": _as_float(candidate.get("projected_changed_count", 0.0)),
            "planner_edge_completion_support_count": _as_float(
                candidate.get("planner_edge_completion_support_count", 0.0)
            ),
            "proposal_support_mass": _as_float(candidate.get("proposal_support_mass", 0.0)),
            "proposal_support_density": _as_float(candidate.get("proposal_support_density", 0.0)),
            "candidate_quality_score": _as_float(candidate.get("candidate_quality_score", 0.0)),
            "reachability_violation": _as_float(candidate.get("reachability_violation", 0.0)),
        }
        diagnostic = candidate.get("candidate_joint_diagnostic")
        if isinstance(diagnostic, dict):
            item["candidate_joint_diagnostic"] = diagnostic
        oracle = candidate.get("teacher_overlap_oracle")
        if isinstance(oracle, dict):
            item["teacher_overlap_oracle"] = {
                key: oracle.get(key)
                for key in [
                    "precision",
                    "recall",
                    "f1",
                    "teacher_reward_sum",
                    "teacher_tau_exp",
                    "teacher_reward_per_tau",
                    "teacher_reward_per_sqrt_tau",
                    "teacher_is_noop",
                    "metric",
                    "metric_value",
                ]
                if key in oracle
            }
            vacancy_pair_overlap = oracle.get("vacancy_pair_overlap")
            if isinstance(vacancy_pair_overlap, dict):
                item["teacher_overlap_oracle"]["vacancy_pair_overlap"] = {
                    key: vacancy_pair_overlap.get(key)
                    for key in [
                        "selected_pair_count",
                        "teacher_pair_count",
                        "overlap_pair_count",
                        "precision",
                        "recall",
                        "f1",
                        "typed_endpoint_accuracy",
                    ]
                    if key in vacancy_pair_overlap
                }
            vacancy_pair_rank = oracle.get("vacancy_pair_rank")
            if isinstance(vacancy_pair_rank, dict):
                item["teacher_overlap_oracle"]["vacancy_pair_rank"] = {
                    key: vacancy_pair_rank.get(key)
                    for key in [
                        "ranked_pair_count",
                        "teacher_pair_count",
                        "teacher_pair_found_count",
                        "teacher_pair_found_recall",
                        "teacher_pair_rank_mean",
                        "teacher_pair_rank_median",
                        "teacher_pair_rank_best",
                        "teacher_pair_rank_worst",
                        "teacher_pair_rank_percentile_mean",
                        "teacher_pair_mrr",
                        "teacher_pair_score_mean",
                        "teacher_pair_typed_rank_accuracy",
                        "teacher_pair_recall_at_rank",
                        "topk_false_positive_rate",
                        "topk_true_pair_count",
                        "topk_source_hard_negative_count",
                        "topk_destination_hard_negative_count",
                        "topk_source_destination_unpaired_count",
                        "topk_type_mismatch_count",
                        "topk_true_score_mean",
                        "topk_false_score_mean",
                    ]
                    if key in vacancy_pair_rank
                }
        vacancy_pair_projection = candidate.get("vacancy_pair_projection_diagnostic")
        if isinstance(vacancy_pair_projection, dict):
            item["vacancy_pair_projection_diagnostic"] = {
                key: vacancy_pair_projection.get(key)
                for key in ["candidate_pair_count", "selected_pair_count"]
                if key in vacancy_pair_projection
            }
            if "factorized_pair_score_count" in vacancy_pair_projection:
                item["vacancy_pair_projection_diagnostic"]["factorized_pair_score_count"] = (
                    vacancy_pair_projection.get("factorized_pair_score_count")
                )
            factorized_pair_scores = vacancy_pair_projection.get("factorized_pair_scores")
            if isinstance(factorized_pair_scores, list):
                item["vacancy_pair_projection_diagnostic"]["factorized_pair_scores"] = factorized_pair_scores
        pareto_selector = candidate.get("planner_candidate_pareto_selector")
        if isinstance(pareto_selector, dict):
            item["planner_candidate_pareto_selector"] = {
                key: pareto_selector.get(key)
                for key in [
                    "mode",
                    "selector_policy",
                    "score",
                    "budget",
                    "candidate_index",
                    "feature_dim",
                    "pair_score_field",
                    "recall_floor",
                    "min_budget",
                    "min_budget_passed",
                    "live_score_scale_normalize",
                    "clip_probability_predictions",
                    "pair_recall_floor_passed",
                    "budget_applied_to_projection",
                    "projection_rerun_pair_budget",
                    "teacher_label_fields_used",
                ]
                if key in pareto_selector
            }
            predictions = pareto_selector.get("predictions")
            if isinstance(predictions, dict):
                item["planner_candidate_pareto_selector"]["predictions"] = predictions
        compacted.append(item)
    return compacted


def _site_overlap_from_position_lists(
    projected_positions_raw: object,
    teacher_positions_raw: object,
) -> dict[str, float]:
    projected_positions = {
        tuple(int(x) for x in item)
        for item in (projected_positions_raw or [])
        if isinstance(item, (list, tuple)) and len(item) == 3
    }
    teacher_positions = {
        tuple(int(x) for x in item)
        for item in (teacher_positions_raw or [])
        if isinstance(item, (list, tuple)) and len(item) == 3
    }
    overlap = projected_positions & teacher_positions
    precision = len(overlap) / max(len(projected_positions), 1)
    recall = len(overlap) / max(len(teacher_positions), 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "overlap_count": float(len(overlap)),
        "projected_changed_count": float(len(projected_positions)),
        "teacher_changed_count": float(len(teacher_positions)),
    }


def _apply_teacher_overlap_oracle(
    *,
    env: mod.MacroKMCEnv,
    rng: np.random.Generator,
    candidates: list[dict[str, object]],
    mode: str,
    weight: float,
    metric: str,
    candidate_joint_diagnostic: bool = False,
) -> dict[str, float]:
    stats = {
        "probe_count": 0.0,
        "probe_failures": 0.0,
        "f1_sum": 0.0,
        "precision_sum": 0.0,
        "recall_sum": 0.0,
        "metric_sum": 0.0,
    }
    if mode == "none" or not candidates:
        return stats
    rng_state = copy.deepcopy(rng.bit_generator.state)
    valid_oracle_records: list[dict[str, object]] = []
    for candidate in candidates:
        candidate["pre_oracle_selection_score"] = float(candidate.get("selection_score", -float("inf")))
        try:
            probe_env = copy.deepcopy(env)
            probe_rng = np.random.default_rng()
            probe_rng.bit_generator.state = copy.deepcopy(rng_state)
            teacher_segment = _collect_teacher_segment(
                probe_env,
                horizon_k=int(candidate["segment_k"]),
                rng=probe_rng,
            )
            if teacher_segment is None:
                stats["probe_failures"] += 1.0
                continue
            teacher_reward = float(teacher_segment.get("reward_sum", 0.0))
            teacher_tau = max(float(teacher_segment.get("tau_exp", 0.0)), 1e-12)
            projected_positions = candidate.get("projected_changed_positions", [])
            if not projected_positions and isinstance(candidate.get("proposal_diagnostic"), dict):
                projected_positions = candidate["proposal_diagnostic"].get("projected_changed_positions", [])
            overlap = _site_overlap_from_position_lists(
                projected_positions,
                teacher_segment.get("changed_positions", []),
            )
            vacancy_pair_overlap = _vacancy_pair_overlap_summary(candidate, teacher_segment)
            vacancy_pair_rank = _vacancy_pair_rank_summary(candidate, teacher_segment)
            _annotate_factorized_vacancy_pairs(candidate, teacher_segment)
            candidate["teacher_overlap_oracle"] = {
                **overlap,
                "teacher_reward_sum": teacher_reward,
                "teacher_tau_exp": teacher_tau,
                "teacher_reward_per_tau": float(teacher_reward / teacher_tau),
                "teacher_reward_per_sqrt_tau": float(teacher_reward / np.sqrt(teacher_tau)),
                "teacher_is_noop": bool(teacher_segment.get("is_noop", False)),
            }
            if vacancy_pair_overlap:
                candidate["teacher_overlap_oracle"]["vacancy_pair_overlap"] = vacancy_pair_overlap
            if vacancy_pair_rank:
                candidate["teacher_overlap_oracle"]["vacancy_pair_rank"] = vacancy_pair_rank
            if candidate_joint_diagnostic:
                candidate["candidate_joint_diagnostic"] = _candidate_joint_diagnostic_record(
                    candidate,
                    overlap,
                    vacancy_pair_overlap,
                    teacher_segment,
                    vacancy_pair_rank,
                )
            candidate["teacher_overlap_oracle_f1"] = float(overlap["f1"])
            valid_oracle_records.append(
                {
                    "candidate": candidate,
                    "overlap_f1": float(overlap["f1"]),
                    "teacher_reward": teacher_reward,
                    "teacher_reward_per_tau": float(teacher_reward / teacher_tau),
                    "teacher_reward_per_sqrt_tau": float(teacher_reward / np.sqrt(teacher_tau)),
                }
            )
            stats["probe_count"] += 1.0
            stats["f1_sum"] += float(overlap["f1"])
            stats["precision_sum"] += float(overlap["precision"])
            stats["recall_sum"] += float(overlap["recall"])
        except Exception:
            stats["probe_failures"] += 1.0
            candidate["selection_score"] = float(candidate.get("pre_oracle_selection_score", candidate.get("selection_score", -float("inf"))))
            continue
    if valid_oracle_records:
        reward_values = [float(record["teacher_reward"]) for record in valid_oracle_records]
        reward_min = min(reward_values)
        reward_max = max(reward_values)
        reward_span = max(reward_max - reward_min, 1e-12)
        for record in valid_oracle_records:
            candidate = record["candidate"]
            if metric == "overlap_reward_norm":
                reward_norm = (float(record["teacher_reward"]) - reward_min) / reward_span
                metric_value = 0.5 * float(record["overlap_f1"]) + 0.5 * float(reward_norm)
            else:
                metric_value = float(record.get(metric, record["overlap_f1"]))
            candidate["teacher_overlap_oracle"]["metric"] = metric
            candidate["teacher_overlap_oracle"]["metric_value"] = float(metric_value)
            candidate["teacher_overlap_oracle_metric_value"] = float(metric_value)
            if mode == "replace":
                candidate["selection_score"] = float(metric_value)
            elif mode == "add":
                candidate["selection_score"] = float(candidate["pre_oracle_selection_score"]) + float(weight) * float(metric_value)
            stats["metric_sum"] += float(metric_value)
    rng.bit_generator.state = copy.deepcopy(rng_state)
    return stats


def main() -> None:
    args = parse_args()
    _validate_pareto_teacher_label_diagnostic(args)
    pareto_label_after_budget = bool(args.planner_candidate_pareto_teacher_label_after_budget_projection)
    if args.planner_candidate_joint_diagnostic:
        if args.planner_teacher_overlap_oracle_mode == "none":
            raise ValueError(
                "--planner_candidate_joint_diagnostic requires a non-none "
                "--planner_teacher_overlap_oracle_mode so teacher-probed candidate labels exist."
            )
        if not args.proposal_diagnostic:
            raise ValueError(
                "--planner_candidate_joint_diagnostic requires --proposal_diagnostic so projected "
                "changed-site positions are available for site-overlap labels."
            )
    if args.planner_vacancy_pair_rank_diagnostic and args.planner_teacher_overlap_oracle_mode == "none":
        raise ValueError(
            "--planner_vacancy_pair_rank_diagnostic requires a non-none "
            "--planner_teacher_overlap_oracle_mode so teacher vacancy-pair labels exist."
        )
    if (
        args.planner_vacancy_pair_factorized_diagnostic
        and args.planner_teacher_overlap_oracle_mode == "none"
        and not args.planner_candidate_pareto_selector_spec
    ):
        raise ValueError(
            "--planner_vacancy_pair_factorized_diagnostic requires a non-none "
            "--planner_teacher_overlap_oracle_mode so teacher vacancy-pair labels exist."
        )
    if args.planner_candidate_pareto_selector_spec:
        if args.planner_teacher_overlap_oracle_mode != "none" and not pareto_label_after_budget:
            raise ValueError(
                "--planner_candidate_pareto_selector_spec is a deployable-selector preflight and "
                "must not be combined with teacher-overlap oracle modes."
            )
        if not args.planner_vacancy_pair_factorized_diagnostic:
            raise ValueError(
                "--planner_candidate_pareto_selector_spec requires --planner_vacancy_pair_factorized_diagnostic "
                "so live factorized pair scores are available without teacher labels."
            )
    elif args.planner_candidate_pareto_apply_budget_to_projection:
        raise ValueError(
            "--planner_candidate_pareto_apply_budget_to_projection requires "
            "--planner_candidate_pareto_selector_spec so a live budget can be predicted."
        )
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
    reward_edit_context_source = str(args.reward_edit_context_source or ckpt_args.get("reward_edit_context_source", "default"))
    ckpt_segment_ks = _segment_ks_from_ckpt_args(ckpt_args)
    if args.planner_segment_ks:
        horizon_choices = sorted({int(k) for k in args.planner_segment_ks})
    elif len(ckpt_segment_ks) > 1:
        horizon_choices = ckpt_segment_ks
    else:
        horizon_choices = [int(ckpt_args["segment_k"])]
    planner_enabled = len(horizon_choices) > 1
    horizon_k = int(horizon_choices[0]) if len(horizon_choices) == 1 else int(max(horizon_choices))
    max_seed_vacancies = int(
        ckpt_args["max_seed_vacancies"]
        if args.max_seed_vacancies_override is None
        else args.max_seed_vacancies_override
    )
    max_candidate_sites = int(
        ckpt_args["max_candidate_sites"]
        if args.max_candidate_sites_override is None
        else args.max_candidate_sites_override
    )

    model = _build_model(ckpt, args.device)
    reward_checkpoint_path = Path(args.reward_checkpoint) if args.reward_checkpoint else None
    reward_model = None
    if reward_checkpoint_path is not None:
        reward_ckpt = torch.load(reward_checkpoint_path, map_location=args.device, weights_only=False)
        reward_model = _build_model(reward_ckpt, args.device)
    duration_checkpoint_path = Path(args.duration_checkpoint) if args.duration_checkpoint else None
    duration_model = None
    if duration_checkpoint_path is not None:
        duration_ckpt = torch.load(duration_checkpoint_path, map_location=args.device, weights_only=False)
        duration_model = _build_model(duration_ckpt, args.device)
    pareto_selector_spec = (
        _load_pareto_selector_spec(Path(args.planner_candidate_pareto_selector_spec))
        if args.planner_candidate_pareto_selector_spec
        else None
    )
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
    oracle_probe_count = 0.0
    oracle_probe_failures = 0.0
    oracle_f1_sum = 0.0
    oracle_precision_sum = 0.0
    oracle_recall_sum = 0.0
    oracle_metric_sum = 0.0
    oracle_selected_f1_sum = 0.0
    oracle_selected_metric_sum = 0.0
    pareto_selector_group_count = 0
    pareto_selector_applied_count = 0
    pareto_selector_feature_rows = 0
    pareto_selector_missing_pair_score_candidates = 0
    pareto_selector_selected_score_sum = 0.0
    pareto_selector_selected_budget_sum = 0.0
    pareto_selector_selected_pair_recall_sum = 0.0
    pareto_selector_selected_floor_pass_count = 0
    pareto_selector_selected_budget_histogram: Counter[int] = Counter()
    pareto_selector_budget_projection_applied_count = 0
    pareto_selector_budget_projection_failed_count = 0

    with torch.no_grad():
        for segment_idx in range(args.rollout_segments):
            def predict_candidate_for_current_state(
                item_k: int,
                *,
                global_pair_budget_override: int | None = None,
            ) -> dict[str, object] | None:
                return _predict_candidate_for_horizon(
                    model=model,
                    reward_model=reward_model,
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
                    planner_noop_risk_penalty=args.planner_noop_risk_penalty,
                    duration_blend_alpha=args.duration_blend_alpha,
                    planner_tau_blend_alpha=planner_tau_blend_alpha,
                    duration_log_offset=duration_log_offset,
                    planner_tau_log_offset=duration_log_offset,
                    planner_duration_checkpoint_source=args.planner_duration_checkpoint_source,
                    reward_prediction_source=reward_prediction_source,
                    reward_edit_context_source=reward_edit_context_source,
                    aux_projected_types_source=args.aux_projected_types_source,
                    proposal_diagnostic=bool(args.proposal_diagnostic),
                    proposal_diagnostic_max_sites=int(args.proposal_diagnostic_max_sites),
                    proposal_diagnostic_store_candidate_positions=bool(
                        args.proposal_diagnostic_store_candidate_positions
                    ),
                    planner_projection_change_source=args.planner_projection_change_source,
                    planner_projection_change_blend_alpha=float(args.planner_projection_change_blend_alpha),
                    planner_projection_topk_source=args.planner_projection_topk_source,
                    planner_projection_topk_budget=int(args.planner_projection_topk_budget),
                    planner_edge_completion_anchor_source=args.planner_edge_completion_anchor_source,
                    planner_edge_completion_destination_source=args.planner_edge_completion_destination_source,
                    planner_edge_completion_anchor_budget=int(args.planner_edge_completion_anchor_budget),
                    planner_edge_completion_destinations_per_anchor=int(
                        args.planner_edge_completion_destinations_per_anchor
                    ),
                    planner_edge_completion_global_pair_budget=(
                        int(args.planner_edge_completion_global_pair_budget)
                        if global_pair_budget_override is None
                        else int(global_pair_budget_override)
                    ),
                    planner_edge_completion_destination_scope=args.planner_edge_completion_destination_scope,
                    planner_edge_completion_require_vacancy_atom_pair=bool(
                        args.planner_edge_completion_require_vacancy_atom_pair
                    ),
                    planner_edge_pair_multiobjective_type_weight=float(
                        args.planner_edge_pair_multiobjective_type_weight
                    ),
                    planner_edge_pair_multiobjective_order_weight=float(
                        args.planner_edge_pair_multiobjective_order_weight
                    ),
                    planner_proposal_score_weight=float(args.planner_proposal_score_weight),
                    planner_candidate_quality_score_weight=float(args.planner_candidate_quality_score_weight),
                    planner_vacancy_pair_rank_diagnostic=bool(args.planner_vacancy_pair_rank_diagnostic),
                    planner_vacancy_pair_rank_max_pairs=int(args.planner_vacancy_pair_rank_max_pairs),
                    planner_vacancy_pair_factorized_diagnostic=bool(
                        args.planner_vacancy_pair_factorized_diagnostic
                    ),
                    planner_vacancy_pair_factorized_max_pairs=int(
                        args.planner_vacancy_pair_factorized_max_pairs
                    ),
                )

            candidates = [
                item
                for item in (predict_candidate_for_current_state(item_k) for item_k in horizon_choices)
                if item is not None
            ]
            if args.planner_teacher_overlap_oracle_mode != "none" and not pareto_label_after_budget:
                oracle_stats = _apply_teacher_overlap_oracle(
                    env=env,
                    rng=rng,
                    candidates=candidates,
                    mode=args.planner_teacher_overlap_oracle_mode,
                    weight=float(args.planner_teacher_overlap_oracle_weight),
                    metric=args.planner_teacher_overlap_oracle_metric,
                    candidate_joint_diagnostic=bool(args.planner_candidate_joint_diagnostic),
                )
                oracle_probe_count += float(oracle_stats["probe_count"])
                oracle_probe_failures += float(oracle_stats["probe_failures"])
                oracle_f1_sum += float(oracle_stats["f1_sum"])
                oracle_precision_sum += float(oracle_stats["precision_sum"])
                oracle_recall_sum += float(oracle_stats["recall_sum"])
                oracle_metric_sum += float(oracle_stats.get("metric_sum", 0.0))
            # The minimum projected-change filter is a planner-selection guard.
            # For a legacy single-horizon checkpoint there is no competing candidate to
            # choose among; rejecting the only candidate turns a valid
            # teacher-forced time evaluation into a zero-length rollout whenever
            # the projected edit is a no-op at the current teacher state.
            effective_min_projected_changed_sites = (
                int(args.min_projected_changed_sites) if planner_enabled else 0
            )
            pareto_stats = _apply_candidate_pareto_selector(
                candidates,
                selector_spec=pareto_selector_spec,
                mode=args.planner_candidate_pareto_selector_mode,
                weight=float(args.planner_candidate_pareto_selector_weight),
                pair_score_field=args.planner_candidate_pareto_pair_score_field,
                selector_policy=args.planner_candidate_pareto_selector_policy,
                recall_floor=args.planner_candidate_pareto_recall_floor,
                min_budget=int(args.planner_candidate_pareto_min_budget),
                live_score_scale_normalize=bool(args.planner_candidate_pareto_live_score_scale_normalize),
                clip_probability_predictions=bool(args.planner_candidate_pareto_clip_probability_predictions),
                min_projected_changed_sites=effective_min_projected_changed_sites,
            )
            if bool(pareto_stats.get("enabled", False)):
                pareto_selector_group_count += 1
                pareto_selector_applied_count += int(bool(pareto_stats.get("applied", False)))
                pareto_selector_feature_rows += int(pareto_stats.get("feature_row_count", 0))
                pareto_selector_missing_pair_score_candidates += int(
                    pareto_stats.get("missing_pair_score_candidate_count", 0)
                )
                pareto_selector_selected_score_sum += _as_float(pareto_stats.get("selected_score", 0.0))
                pareto_selector_selected_budget_sum += _as_float(pareto_stats.get("selected_budget", 0.0))
                pareto_selector_selected_pair_recall_sum += _as_float(
                    pareto_stats.get("selected_prediction_pair_recall", 0.0)
                )
                pareto_selector_selected_floor_pass_count += int(
                    bool(pareto_stats.get("selected_pair_recall_floor_passed", False))
                )
                pareto_selector_selected_budget_histogram[
                    int(pareto_stats.get("selected_budget", 0) or 0)
                ] += 1
            if (
                bool(args.planner_candidate_pareto_apply_budget_to_projection)
                and bool(pareto_stats.get("enabled", False))
            ):
                reprojected_candidates: list[dict[str, object]] = []
                for candidate in candidates:
                    selector = candidate.get("planner_candidate_pareto_selector")
                    if not isinstance(selector, dict):
                        reprojected_candidates.append(candidate)
                        continue
                    budget = int(selector.get("budget", 0) or 0)
                    if budget <= 0:
                        reprojected_candidates.append(candidate)
                        continue
                    reprojected = predict_candidate_for_current_state(
                        int(candidate.get("segment_k", 0)),
                        global_pair_budget_override=budget,
                    )
                    if reprojected is None:
                        pareto_selector_budget_projection_failed_count += 1
                        reprojected_candidates.append(candidate)
                        continue
                    selector_for_reprojected = dict(selector)
                    selector_for_reprojected["budget_applied_to_projection"] = True
                    selector_for_reprojected["projection_rerun_pair_budget"] = int(budget)
                    reprojected["planner_candidate_pareto_selector"] = selector_for_reprojected
                    reprojected["pre_pareto_selection_score"] = candidate.get(
                        "pre_pareto_selection_score",
                        candidate.get("selection_score", reprojected.get("selection_score", -float("inf"))),
                    )
                    if args.planner_candidate_pareto_selector_mode == "replace":
                        reprojected["selection_score"] = float(selector_for_reprojected.get("score", 0.0))
                    elif args.planner_candidate_pareto_selector_mode == "add":
                        reprojected["selection_score"] = _as_float(
                            reprojected.get("selection_score", -float("inf"))
                        ) + float(args.planner_candidate_pareto_selector_weight) * _as_float(
                            selector_for_reprojected.get("score", 0.0)
                        )
                    pareto_selector_budget_projection_applied_count += 1
                    reprojected_candidates.append(reprojected)
                candidates = reprojected_candidates
            if args.planner_teacher_overlap_oracle_mode != "none" and pareto_label_after_budget:
                oracle_stats = _apply_teacher_overlap_oracle(
                    env=env,
                    rng=rng,
                    candidates=candidates,
                    mode=args.planner_teacher_overlap_oracle_mode,
                    weight=float(args.planner_teacher_overlap_oracle_weight),
                    metric=args.planner_teacher_overlap_oracle_metric,
                    candidate_joint_diagnostic=bool(args.planner_candidate_joint_diagnostic),
                )
                oracle_probe_count += float(oracle_stats["probe_count"])
                oracle_probe_failures += float(oracle_stats["probe_failures"])
                oracle_f1_sum += float(oracle_stats["f1_sum"])
                oracle_precision_sum += float(oracle_stats["precision_sum"])
                oracle_recall_sum += float(oracle_stats["recall_sum"])
                oracle_metric_sum += float(oracle_stats.get("metric_sum", 0.0))
            selected = _choose_planner_candidate(
                candidates,
                min_projected_changed_sites=effective_min_projected_changed_sites,
            )
            if selected is None:
                stop_reason = "no_legal_planner_candidate"
                stop_segment = {
                    "index": segment_idx,
                    "planner_candidates": _planner_candidates_for_output(
                        candidates,
                        compact=bool(args.planner_candidate_joint_compact_candidates),
                    ),
                }
                break
            if args.planner_candidate_joint_diagnostic:
                for candidate in candidates:
                    diagnostic = candidate.get("candidate_joint_diagnostic")
                    if isinstance(diagnostic, dict):
                        diagnostic["selected_by_planner"] = bool(candidate is selected)
            selected_k = int(selected["segment_k"])
            if args.planner_teacher_overlap_oracle_mode != "none":
                selected_oracle = selected.get("teacher_overlap_oracle", {})
                if isinstance(selected_oracle, dict):
                    oracle_selected_f1_sum += float(selected_oracle.get("f1", 0.0))
                    oracle_selected_metric_sum += float(selected_oracle.get("metric_value", 0.0))

            teacher_segment = _collect_teacher_segment(env, horizon_k=selected_k, rng=rng)
            if teacher_segment is None:
                skipped_terminal += 1
                stop_reason = "teacher_terminal_or_action_missing"
                stop_segment = {
                    "index": segment_idx,
                    "segment_k": selected_k,
                    "planner_candidates": _planner_candidates_for_output(
                        candidates,
                        compact=bool(args.planner_candidate_joint_compact_candidates),
                    ),
                    "selected": selected,
                }
                break
            if bool(teacher_segment.get("is_noop", False)) and not args.allow_teacher_noop_segments:
                skipped_noop += 1
                stop_reason = "noop_teacher_segment"
                stop_segment = {
                    "index": segment_idx,
                    "segment_k": selected_k,
                    "planner_candidates": _planner_candidates_for_output(
                        candidates,
                        compact=bool(args.planner_candidate_joint_compact_candidates),
                    ),
                    "selected": selected,
                    "traditional_kmc_reward_sum": float(teacher_segment["reward_sum"]),
                    "traditional_kmc_delta_e": float(teacher_segment["reward_sum"] / reward_scale),
                    "traditional_kmc_expected_tau": float(teacher_segment["tau_exp"]),
                    "traditional_kmc_realized_tau": float(teacher_segment["tau_real"]),
                    "traditional_changed_site_count": int(teacher_segment.get("changed_site_count", 0)),
                }
                if args.proposal_diagnostic:
                    stop_segment["proposal_overlap"] = _proposal_overlap_summary(selected, teacher_segment)
                vacancy_pair_overlap = _vacancy_pair_overlap_summary(selected, teacher_segment)
                if vacancy_pair_overlap:
                    stop_segment["vacancy_pair_overlap"] = vacancy_pair_overlap
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
                    "planner_candidates": _planner_candidates_for_output(
                        candidates,
                        compact=bool(args.planner_candidate_joint_compact_candidates),
                    ),
                    "selection_score": float(selected["selection_score"]),
                    "predicted_reward_sum": float(selected["predicted_reward_sum"]),
                    "predicted_reward_raw": float(selected["predicted_reward_raw"]),
                    "predicted_reward_gate_prob": float(selected["predicted_reward_gate_prob"]),
                    "predicted_noop_risk_prob": float(selected.get("predicted_noop_risk_prob", 0.0)),
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
                    "planner_edge_completion_support_count": int(
                        selected.get("planner_edge_completion_support_count", 0)
                    ),
                    "traditional_changed_site_count": int(teacher_segment.get("changed_site_count", 0)),
                    "traditional_is_noop": bool(teacher_segment.get("is_noop", False)),
                    "proposal_overlap": _proposal_overlap_summary(selected, teacher_segment)
                    if args.proposal_diagnostic
                    else {},
                    "vacancy_pair_overlap": _vacancy_pair_overlap_summary(selected, teacher_segment),
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

    candidate_joint_segments = list(segments)
    if isinstance(stop_segment, dict):
        candidate_joint_segments.append(stop_segment)

    summary = {
        "mode": "multi_k_planner_teacher_forced_contiguous_long_trajectory" if planner_enabled else "teacher_forced_contiguous_long_trajectory",
        "checkpoint": str(checkpoint_path),
        "duration_checkpoint": str(duration_checkpoint_path) if duration_checkpoint_path is not None else None,
        "reward_checkpoint": str(reward_checkpoint_path) if reward_checkpoint_path is not None else None,
        "aux_projected_types_source": args.aux_projected_types_source,
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
        "planner_noop_risk_penalty": float(args.planner_noop_risk_penalty),
        "planner_projection_change_source": args.planner_projection_change_source,
        "planner_projection_change_blend_alpha": float(args.planner_projection_change_blend_alpha),
        "planner_projection_topk_source": args.planner_projection_topk_source,
        "planner_projection_topk_budget": int(args.planner_projection_topk_budget),
        "planner_edge_completion_anchor_source": args.planner_edge_completion_anchor_source,
        "planner_edge_completion_destination_source": args.planner_edge_completion_destination_source,
        "planner_edge_completion_anchor_budget": int(args.planner_edge_completion_anchor_budget),
        "planner_edge_completion_destinations_per_anchor": int(args.planner_edge_completion_destinations_per_anchor),
        "planner_edge_completion_global_pair_budget": int(args.planner_edge_completion_global_pair_budget),
        "planner_edge_completion_destination_scope": args.planner_edge_completion_destination_scope,
        "planner_edge_completion_require_vacancy_atom_pair": bool(args.planner_edge_completion_require_vacancy_atom_pair),
        "planner_edge_pair_multiobjective_type_weight": float(args.planner_edge_pair_multiobjective_type_weight),
        "planner_edge_pair_multiobjective_order_weight": float(args.planner_edge_pair_multiobjective_order_weight),
        "planner_proposal_score_weight": float(args.planner_proposal_score_weight),
        "planner_candidate_quality_score_weight": float(args.planner_candidate_quality_score_weight),
        "planner_candidate_pareto_selector_spec": str(args.planner_candidate_pareto_selector_spec or ""),
        "planner_candidate_pareto_selector_mode": args.planner_candidate_pareto_selector_mode,
        "planner_candidate_pareto_selector_policy": args.planner_candidate_pareto_selector_policy,
        "planner_candidate_pareto_recall_floor": (
            None
            if args.planner_candidate_pareto_recall_floor is None
            else float(args.planner_candidate_pareto_recall_floor)
        ),
        "planner_candidate_pareto_min_budget": int(args.planner_candidate_pareto_min_budget),
        "planner_candidate_pareto_live_score_scale_normalize": bool(
            args.planner_candidate_pareto_live_score_scale_normalize
        ),
        "planner_candidate_pareto_clip_probability_predictions": bool(
            args.planner_candidate_pareto_clip_probability_predictions
        ),
        "planner_candidate_pareto_selector_weight": float(args.planner_candidate_pareto_selector_weight),
        "planner_candidate_pareto_pair_score_field": args.planner_candidate_pareto_pair_score_field,
        "planner_candidate_pareto_apply_budget_to_projection": bool(
            args.planner_candidate_pareto_apply_budget_to_projection
        ),
        "planner_candidate_pareto_teacher_label_after_budget_projection": bool(
            args.planner_candidate_pareto_teacher_label_after_budget_projection
        ),
        "planner_candidate_pareto_selector": {
            "enabled": bool(pareto_selector_spec is not None),
            "selector_policy": args.planner_candidate_pareto_selector_policy,
            "min_budget": int(args.planner_candidate_pareto_min_budget),
            "live_score_scale_normalize": bool(args.planner_candidate_pareto_live_score_scale_normalize),
            "clip_probability_predictions": bool(args.planner_candidate_pareto_clip_probability_predictions),
            "group_count": int(pareto_selector_group_count),
            "applied_count": int(pareto_selector_applied_count),
            "budget_projection_rerun_count": int(pareto_selector_budget_projection_applied_count),
            "budget_projection_rerun_failed_count": int(pareto_selector_budget_projection_failed_count),
            "feature_row_count": int(pareto_selector_feature_rows),
            "missing_pair_score_candidate_count": int(pareto_selector_missing_pair_score_candidates),
            "avg_selected_score": float(
                pareto_selector_selected_score_sum / max(float(pareto_selector_group_count), 1.0)
            ),
            "avg_selected_budget": float(
                pareto_selector_selected_budget_sum / max(float(pareto_selector_group_count), 1.0)
            ),
            "selected_budget_histogram": {
                str(key): int(value) for key, value in sorted(pareto_selector_selected_budget_histogram.items())
            },
            "avg_selected_prediction_pair_recall": float(
                pareto_selector_selected_pair_recall_sum / max(float(pareto_selector_group_count), 1.0)
            ),
            "selected_pair_recall_floor_pass_count": int(pareto_selector_selected_floor_pass_count),
            "budget_applied_to_projection": bool(args.planner_candidate_pareto_apply_budget_to_projection),
            "teacher_label_fields_used": bool(
                args.planner_candidate_pareto_teacher_label_after_budget_projection
            ),
        },
        "planner_teacher_overlap_oracle_mode": args.planner_teacher_overlap_oracle_mode,
        "planner_teacher_overlap_oracle_weight": float(args.planner_teacher_overlap_oracle_weight),
        "planner_teacher_overlap_oracle_metric": args.planner_teacher_overlap_oracle_metric,
        "planner_candidate_joint_diagnostic": bool(args.planner_candidate_joint_diagnostic),
        "planner_candidate_joint_compact_candidates": bool(args.planner_candidate_joint_compact_candidates),
        "planner_vacancy_pair_rank_diagnostic": bool(args.planner_vacancy_pair_rank_diagnostic),
        "planner_vacancy_pair_rank_max_pairs": int(args.planner_vacancy_pair_rank_max_pairs),
        "planner_vacancy_pair_factorized_diagnostic": bool(args.planner_vacancy_pair_factorized_diagnostic),
        "planner_vacancy_pair_factorized_max_pairs": int(args.planner_vacancy_pair_factorized_max_pairs),
        "candidate_joint_diagnostic": (
            _candidate_joint_diagnostic_summary(candidate_joint_segments)
            if args.planner_candidate_joint_diagnostic
            else {}
        ),
        "teacher_overlap_oracle": {
            "probe_count": float(oracle_probe_count),
            "probe_failures": float(oracle_probe_failures),
            "avg_precision": float(oracle_precision_sum / max(oracle_probe_count, 1.0)),
            "avg_recall": float(oracle_recall_sum / max(oracle_probe_count, 1.0)),
            "avg_f1": float(oracle_f1_sum / max(oracle_probe_count, 1.0)),
            "avg_metric_value": float(oracle_metric_sum / max(oracle_probe_count, 1.0)),
            "selected_avg_f1": float(oracle_selected_f1_sum / max(len(segments), 1)),
            "selected_avg_metric_value": float(oracle_selected_metric_sum / max(len(segments), 1)),
        },
        "reward_prediction_source": reward_prediction_source,
        "reward_edit_context_source": reward_edit_context_source,
        "proposal_diagnostic": bool(args.proposal_diagnostic),
        "proposal_diagnostic_max_sites": int(args.proposal_diagnostic_max_sites),
        "proposal_diagnostic_store_candidate_positions": bool(args.proposal_diagnostic_store_candidate_positions),
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
    print("AtomWorld-Mirror Long Trajectory Evaluation")
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
