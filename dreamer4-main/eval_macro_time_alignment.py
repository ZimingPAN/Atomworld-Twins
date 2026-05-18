from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import train_dreamer_macro_edit as mod


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate macro-edit Dreamer time alignment against traditional KMC teacher segments"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--cache", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--print_samples", type=int, default=5)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--save_all_samples", action="store_true",
                        help="Save all per-sample predictions (not just preview) to output JSON")
    parser.add_argument("--duration_source", type=str, default="model", choices=["model", "baseline", "blend"],
                        help="Expected tau source for reported paired metrics. 'blend' interpolates log tau between CTMC baseline and model.")
    parser.add_argument("--duration_blend_alpha", type=float, default=1.0,
                        help="For --duration_source blend, alpha in log_tau = (1-alpha)*baseline + alpha*model.")
    parser.add_argument("--reward_edit_context_source", type=str, default=None, choices=["default", "none"],
                        help="Override checkpoint reward/tau edit-context source. 'none' keeps patch+k context but zeros edit-summary features.")
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
        print(f"Eval: missing keys initialized from scratch: {missing}")
    if unexpected:
        print(f"Eval: unexpected keys ignored: {unexpected}")
    model.eval()
    return model


def _load_samples(
    cache_path: Path,
    split: str,
    limit: int,
    expected_segment_k: int | None = None,
    expected_segment_ks: list[int] | None = None,
    expected_summary_horizon_k: int | None = None,
) -> tuple[list[mod.MacroSegmentSample], dict[str, object], dict[str, object]]:
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    signature = payload.get("signature")
    if not isinstance(signature, dict):
        raise ValueError("Dataset cache is missing signature metadata; refusing to run time alignment without segment_k validation")
    if expected_segment_ks is None:
        if expected_segment_k is None:
            raise ValueError("expected_segment_k or expected_segment_ks is required")
        expected_segment_ks = [int(expected_segment_k)]
    expected_segment_ks = sorted({int(k) for k in expected_segment_ks})
    cache_segment_ks = signature.get("segment_ks")
    if cache_segment_ks is None:
        cache_segment_k = signature.get("segment_k")
        cache_segment_ks = [int(cache_segment_k)] if cache_segment_k is not None else []
    cache_segment_ks = sorted({int(k) for k in cache_segment_ks})
    if cache_segment_ks != expected_segment_ks:
        raise ValueError(
            f"Dataset cache segment_ks={cache_segment_ks} does not match checkpoint segment_ks={expected_segment_ks}"
        )
    if expected_summary_horizon_k is not None and signature.get("summary_horizon_k") is not None:
        cache_summary_horizon_k = int(signature["summary_horizon_k"])
        if cache_summary_horizon_k != int(expected_summary_horizon_k):
            raise ValueError(
                f"Dataset cache summary_horizon_k={cache_summary_horizon_k} does not match checkpoint summary_horizon_k={int(expected_summary_horizon_k)}"
            )
    samples = [mod.MacroSegmentSample(**item) for item in payload[split]]
    boundary_mode = str(signature.get("segment_boundary_mode", "fixed_k"))
    if boundary_mode == "adaptive_key_event":
        max_horizon = int(signature.get("summary_horizon_k") or expected_summary_horizon_k or max(expected_segment_ks))
        min_horizon = 1
        mismatched_sample = next(
            (sample for sample in samples if int(sample.horizon_k) < min_horizon or int(sample.horizon_k) > max_horizon),
            None,
        )
        if mismatched_sample is not None:
            raise ValueError(
                f"Found adaptive sample with horizon_k={int(mismatched_sample.horizon_k)} in cache split {split}, "
                f"expected {min_horizon} <= horizon_k <= {max_horizon}"
            )
    else:
        expected_set = set(expected_segment_ks)
        mismatched_sample = next((sample for sample in samples if int(sample.horizon_k) not in expected_set), None)
        if mismatched_sample is not None:
            raise ValueError(
                f"Found sample with horizon_k={int(mismatched_sample.horizon_k)} in cache split {split}, expected one of {expected_segment_ks}"
            )
    if limit > 0:
        samples = samples[:limit]
    return samples, payload.get("stats", {}), signature


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    cache_path = Path(args.cache)

    ckpt = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    model = _build_model(ckpt, args.device)
    reward_scale = float(ckpt["args"].get("reward_scale", 1.0))
    reward_prediction_source = str(ckpt["args"].get("reward_prediction_source", "raw"))
    reward_edit_context_source = str(args.reward_edit_context_source or ckpt["args"].get("reward_edit_context_source", "default"))
    segment_ks = _segment_ks_from_ckpt_args(ckpt["args"])
    segment_k = int(segment_ks[0]) if len(segment_ks) == 1 else int(max(segment_ks))

    samples, dataset_stats, cache_signature = _load_samples(
        cache_path,
        args.split,
        args.limit,
        expected_segment_ks=segment_ks,
        expected_summary_horizon_k=_summary_horizon_k_from_ckpt_args(ckpt["args"]),
    )
    loader = mod._build_loader(samples, batch_size=args.batch_size, shuffle=False)

    pred_reward_sum = []
    true_reward_sum = []
    pred_noop_risk = []
    true_noop_risk = []
    pred_tau = []
    pred_realized_tau = []
    pred_realized_tau_mu = []
    pred_realized_tau_log_sigma = []
    true_tau_exp = []
    true_tau_real = []
    sample_horizon_ks = []
    sample_rows = []
    sample_index = 0
    realized_tau_source = "realized_duration_head" if getattr(model, "realized_tau_head_loaded", True) else "expected_duration_head_fallback"

    with torch.no_grad():
        for batch in loader:
            tensors = mod._batch_to_device(batch, args.device)
            global_latent = model.encode_global(tensors["start_obs"])
            prior_mu, prior_logvar = model.prior_stats(
                global_latent,
                tensors["global_summary"],
                tensors["horizon_k"],
            )
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
            reward_change_logits, reward_type_logits = mod._select_reward_edit_context(
                reward_edit_context_source,
                change_logits,
                raw_type_logits,
            )
            duration_outputs = mod._predict_reward_and_duration_outputs(
                model,
                global_latent,
                next_pred,
                path_latent,
                tensors["global_summary"],
                tensors["horizon_k"],
                patch_latent=patch_latent,
                change_logits=reward_change_logits,
                type_logits=reward_type_logits,
                current_types=tensors["current_types"],
                candidate_mask=tensors["candidate_mask"],
            )
            if reward_prediction_source == "projected":
                projected_types, _projected_changed_mask, _transport_cost, _violation = mod.project_types_by_inventory(
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
                projected_change_logits, projected_type_logits = mod._select_reward_edit_context(
                    reward_edit_context_source,
                    projected_change_logits,
                    projected_type_logits,
                )
                projected_duration_outputs = mod._predict_reward_and_duration_outputs(
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
                duration_outputs = projected_duration_outputs
            reward_hat = duration_outputs["reward"]
            tau_mu = duration_outputs["expected_tau_mu"]
            tau_log_sigma = duration_outputs["expected_tau_log_sigma"]
            baseline_tau_mu = mod.macro_duration_baseline_log_tau(tensors["global_summary"], tensors["horizon_k"])
            if args.duration_source == "baseline":
                reported_tau_mu = baseline_tau_mu
            elif args.duration_source == "blend":
                alpha = float(np.clip(args.duration_blend_alpha, 0.0, 1.0))
                reported_tau_mu = (1.0 - alpha) * baseline_tau_mu + alpha * tau_mu
            else:
                reported_tau_mu = tau_mu
            if realized_tau_source == "realized_duration_head":
                realized_tau_mu = duration_outputs["realized_tau_mu"]
                realized_tau_log_sigma = duration_outputs["realized_tau_log_sigma"]
            else:
                realized_tau_mu = tau_mu
                realized_tau_log_sigma = tau_log_sigma
            gate_logit = duration_outputs["gate_logit"]
            noop_risk_logit = duration_outputs.get("noop_risk_logit", torch.zeros_like(reward_hat))
            gated_reward = reward_hat * torch.sigmoid(gate_logit)
            batch_pred_reward = gated_reward.detach().cpu().numpy()
            batch_pred_reward_raw = reward_hat.detach().cpu().numpy()
            batch_pred_reward_gate = torch.sigmoid(gate_logit).detach().cpu().numpy()
            batch_pred_noop_risk = torch.sigmoid(noop_risk_logit).detach().cpu().numpy()
            batch_pred_tau = torch.exp(reported_tau_mu).detach().cpu().numpy()
            batch_model_tau = torch.exp(tau_mu).detach().cpu().numpy()
            batch_baseline_tau = torch.exp(baseline_tau_mu).detach().cpu().numpy()
            batch_pred_realized_tau = torch.exp(realized_tau_mu).detach().cpu().numpy()
            batch_pred_realized_tau_mu = realized_tau_mu.detach().cpu().numpy()
            batch_pred_realized_tau_log_sigma = realized_tau_log_sigma.detach().cpu().numpy()

            for sample, item_pred_reward, item_pred_reward_raw, item_pred_reward_gate, item_pred_noop_risk, item_pred_tau, item_model_tau, item_baseline_tau, item_pred_realized_tau, item_pred_realized_tau_mu, item_pred_realized_tau_log_sigma in zip(
                batch,
                batch_pred_reward,
                batch_pred_reward_raw,
                batch_pred_reward_gate,
                batch_pred_noop_risk,
                batch_pred_tau,
                batch_model_tau,
                batch_baseline_tau,
                batch_pred_realized_tau,
                batch_pred_realized_tau_mu,
                batch_pred_realized_tau_log_sigma,
            ):
                pred_reward_sum.append(float(item_pred_reward))
                true_reward_sum.append(float(sample.reward_sum))
                pred_noop_risk.append(float(item_pred_noop_risk))
                true_noop_risk.append(float(np.sum(sample.changed_mask) <= 0.0))
                pred_tau.append(float(item_pred_tau))
                pred_realized_tau.append(float(item_pred_realized_tau))
                pred_realized_tau_mu.append(float(item_pred_realized_tau_mu))
                pred_realized_tau_log_sigma.append(float(item_pred_realized_tau_log_sigma))
                true_tau_exp.append(float(sample.tau_exp))
                true_tau_real.append(float(sample.tau_real))
                sample_horizon_ks.append(int(sample.horizon_k))
                sample_rows.append(
                    {
                        "index": sample_index,
                        "segment_k": int(sample.horizon_k),
                        "traditional_kmc_reward_sum": float(sample.reward_sum),
                        "traditional_kmc_delta_e": float(sample.reward_sum / reward_scale),
                        "traditional_kmc_expected_tau": float(sample.tau_exp),
                        "traditional_kmc_realized_tau": float(sample.tau_real),
                        "predicted_reward_sum": float(item_pred_reward),
                        "predicted_reward_raw": float(item_pred_reward_raw),
                        "predicted_reward_gate_prob": float(item_pred_reward_gate),
                        "predicted_noop_risk_prob": float(item_pred_noop_risk),
                        "traditional_noop_target": bool(np.sum(sample.changed_mask) <= 0.0),
                        "predicted_delta_e": float(item_pred_reward / reward_scale),
                        "predicted_tau": float(item_pred_tau),
                        "predicted_expected_tau": float(item_pred_tau),
                        "model_expected_tau": float(item_model_tau),
                        "baseline_expected_tau": float(item_baseline_tau),
                        "predicted_realized_tau": float(item_pred_realized_tau),
                        "predicted_realized_tau_log_mu": float(item_pred_realized_tau_mu),
                        "predicted_realized_tau_log_sigma": float(item_pred_realized_tau_log_sigma),
                    }
                )
                sample_index += 1

    pred_reward_sum_np = np.asarray(pred_reward_sum, dtype=np.float64)
    true_reward_sum_np = np.asarray(true_reward_sum, dtype=np.float64)
    pred_noop_risk_np = np.asarray(pred_noop_risk, dtype=np.float64)
    true_noop_risk_np = np.asarray(true_noop_risk, dtype=np.float64)
    pred_delta_e_np = pred_reward_sum_np / reward_scale
    true_delta_e_np = true_reward_sum_np / reward_scale
    pred_tau_np = np.asarray(pred_tau, dtype=np.float64)
    pred_realized_tau_np = np.asarray(pred_realized_tau, dtype=np.float64)
    pred_realized_tau_mu_np = np.asarray(pred_realized_tau_mu, dtype=np.float64)
    pred_realized_tau_log_sigma_np = np.asarray(pred_realized_tau_log_sigma, dtype=np.float64)
    pred_realized_tau_distribution_mean_np = np.exp(
        np.clip(
            pred_realized_tau_mu_np + 0.5 * np.exp(2.0 * pred_realized_tau_log_sigma_np),
            a_min=-60.0,
            a_max=60.0,
        )
    )
    true_tau_exp_np = np.asarray(true_tau_exp, dtype=np.float64)
    true_tau_real_np = np.asarray(true_tau_real, dtype=np.float64)
    sample_horizon_ks_np = np.asarray(sample_horizon_ks, dtype=np.int64)

    if realized_tau_source == "realized_duration_head":
        realized_distribution_summary = {
            "available": True,
            "prediction_source": realized_tau_source,
            **mod._compute_lognormal_distribution_metrics(
                pred_realized_tau_mu_np,
                pred_realized_tau_log_sigma_np,
                true_tau_real_np,
            ),
            "traditional_mean": float(np.mean(true_tau_real_np)),
            "predicted_median_mean": float(np.mean(pred_realized_tau_np)),
            "predicted_mean": float(np.mean(pred_realized_tau_distribution_mean_np)),
        }
    else:
        realized_distribution_summary = {
            "available": False,
            "prediction_source": realized_tau_source,
            "reason": "checkpoint_missing_realized_duration_head",
        }

    by_k = {}
    for horizon_k in sorted(set(sample_horizon_ks)):
        mask = sample_horizon_ks_np == int(horizon_k)
        by_k[str(int(horizon_k))] = {
            "num_samples": int(mask.sum()),
            "reward_sum": _compute_metrics(pred_reward_sum_np[mask], true_reward_sum_np[mask]),
            "reward_diagnostics": mod._compute_reward_diagnostics(pred_reward_sum_np[mask], true_reward_sum_np[mask]),
            "delta_e": _compute_metrics(pred_delta_e_np[mask], true_delta_e_np[mask]),
            "tau_expected": {
                **_compute_metrics(pred_tau_np[mask], true_tau_exp_np[mask]),
                **_compute_log_metrics(pred_tau_np[mask], true_tau_exp_np[mask]),
                "traditional_mean": float(np.mean(true_tau_exp_np[mask])),
                "predicted_mean": float(np.mean(pred_tau_np[mask])),
            },
            "tau_realized": {
                **_compute_metrics(pred_realized_tau_np[mask], true_tau_real_np[mask]),
                **_compute_log_metrics(pred_realized_tau_np[mask], true_tau_real_np[mask]),
                "traditional_mean": float(np.mean(true_tau_real_np[mask])),
                "predicted_median": float(np.mean(pred_realized_tau_np[mask])),
            },
        }

    summary = {
        "checkpoint": str(checkpoint_path),
        "cache": str(cache_path),
        "split": args.split,
        "num_samples": int(len(samples)),
        "segment_k": segment_k,
        "segment_ks": segment_ks,
        "reward_prediction_source": reward_prediction_source,
        "reward_edit_context_source": reward_edit_context_source,
        "cache_signature": cache_signature,
        "dataset_stats": dataset_stats.get(args.split, {}),
        "teacher_source": f"{cache_signature.get('natural_teacher_backend', 'kmc')}_segment_cache",
        "time_heads": {
            "expected_tau_head": True,
            "realized_tau_head_loaded": bool(getattr(model, "realized_tau_head_loaded", True)),
            "realized_tau_source": realized_tau_source,
            "duration_source": args.duration_source,
            "duration_blend_alpha": float(args.duration_blend_alpha),
        },
        "reward_sum": _compute_metrics(pred_reward_sum_np, true_reward_sum_np),
        "reward_diagnostics": mod._compute_reward_diagnostics(pred_reward_sum_np, true_reward_sum_np),
        "noop_risk": {
            "target_frac": float(np.mean(true_noop_risk_np)) if true_noop_risk_np.size else 0.0,
            "pred_mean": float(np.mean(pred_noop_risk_np)) if pred_noop_risk_np.size else 0.0,
            "noop_pred_mean": (
                float(np.mean(pred_noop_risk_np[true_noop_risk_np > 0.5]))
                if np.any(true_noop_risk_np > 0.5)
                else 0.0
            ),
            "nonnoop_pred_mean": (
                float(np.mean(pred_noop_risk_np[true_noop_risk_np <= 0.5]))
                if np.any(true_noop_risk_np <= 0.5)
                else 0.0
            ),
        },
        "delta_e": _compute_metrics(pred_delta_e_np, true_delta_e_np),
        "tau_expected": {
            **_compute_metrics(pred_tau_np, true_tau_exp_np),
            **_compute_log_metrics(pred_tau_np, true_tau_exp_np),
            "traditional_mean": float(np.mean(true_tau_exp_np)),
            "predicted_mean": float(np.mean(pred_tau_np)),
        },
        "tau_realized": {
            **_compute_metrics(pred_realized_tau_np, true_tau_real_np),
            **_compute_log_metrics(pred_realized_tau_np, true_tau_real_np),
            "traditional_mean": float(np.mean(true_tau_real_np)),
            "predicted_median": float(np.mean(pred_realized_tau_np)),
            "predicted_mean": float(np.mean(pred_realized_tau_distribution_mean_np)),
            "prediction_source": realized_tau_source,
        },
        "tau_realized_distribution": realized_distribution_summary,
        "traditional_energy": {
            "reward_sum_mean": float(np.mean(true_reward_sum_np)),
            "delta_e_mean": float(np.mean(true_delta_e_np)),
        },
        "predicted_energy": {
            "reward_sum_mean": float(np.mean(pred_reward_sum_np)),
            "delta_e_mean": float(np.mean(pred_delta_e_np)),
        },
        "by_k": by_k,
        "sample_preview": sample_rows[: max(args.print_samples, 0)],
    }
    if args.save_all_samples:
        summary["all_samples"] = sample_rows

    print("=" * 60)
    print("Macro-Edit Dreamer vs Traditional KMC Teacher")
    print(f"samples={len(samples)}, split={args.split}, segment_ks={segment_ks}")
    print("=" * 60)
    print(
        "Traditional KMC energy/time means: "
        f"reward_sum={summary['traditional_energy']['reward_sum_mean']:.6f}, "
        f"delta_E={summary['traditional_energy']['delta_e_mean']:.6f}, "
        f"E[tau]={summary['tau_expected']['traditional_mean']:.6e}, "
        f"real_tau={summary['tau_realized']['traditional_mean']:.6e}"
    )
    print(
        "Model prediction means: "
        f"reward_sum={summary['predicted_energy']['reward_sum_mean']:.6f}, "
        f"delta_E={summary['predicted_energy']['delta_e_mean']:.6f}, "
        f"pred_tau={summary['tau_expected']['predicted_mean']:.6e}"
    )
    print(
        "Reward alignment: "
        f"mae={summary['reward_sum']['mae']:.6f}, rmse={summary['reward_sum']['rmse']:.6f}, corr={summary['reward_sum']['corr']:.4f}"
    )
    print(
        "Expected-time alignment: "
        f"mae={summary['tau_expected']['mae']:.6e}, log_mae={summary['tau_expected']['log_mae']:.4f}, "
        f"log_corr={summary['tau_expected']['log_corr']:.4f}, scale_ratio={summary['tau_expected']['scale_ratio']:.2f}"
    )
    print(
        "Realized-time reference: "
        f"mae={summary['tau_realized']['mae']:.6e}, log_mae={summary['tau_realized']['log_mae']:.4f}, "
        f"log_corr={summary['tau_realized']['log_corr']:.4f}, scale_ratio={summary['tau_realized']['scale_ratio']:.2f}"
    )
    if summary["tau_realized_distribution"].get("available"):
        print(
            "Realized-time distribution: "
            f"nll={summary['tau_realized_distribution']['nll']:.4f}, "
            f"coverage68={summary['tau_realized_distribution']['coverage_68']:.4f}, "
            f"coverage95={summary['tau_realized_distribution']['coverage_95']:.4f}, "
            f"pit_ks={summary['tau_realized_distribution']['pit_ks']:.4f}"
        )
    else:
        print(
            "Realized-time distribution: unavailable "
            f"({summary['tau_realized_distribution'].get('reason', 'unknown reason')})"
        )
    if summary["sample_preview"]:
        print("Sample preview:")
        for row in summary["sample_preview"]:
            print(json.dumps(row, ensure_ascii=False))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved summary to {output_path}")


if __name__ == "__main__":
    main()
