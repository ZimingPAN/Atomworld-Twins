from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
KMC_BACKEND = ROOT / "kmcteacher_backend"
PRIVATE_RLKMC = ROOT / "RLKMC-MASSIVE-main"
DREAMER = ROOT / "dreamer4-main"
for path in [str(ROOT), str(KMC_BACKEND), str(DREAMER)]:
    if path not in sys.path:
        sys.path.insert(0, path)

if PRIVATE_RLKMC.is_dir() and str(PRIVATE_RLKMC) not in sys.path:
    sys.path.insert(0, str(PRIVATE_RLKMC))

import train_dreamer_macro_edit as macro_mod

if PRIVATE_RLKMC.is_dir():
    from train_ppo_standalone import KMCEnvWrapper, PPOGNNAgent
else:
    KMCEnvWrapper = Any
    PPOGNNAgent = Any


def _require_private_rlkmc_checkout() -> None:
    if PRIVATE_RLKMC.is_dir():
        return
    raise RuntimeError(
        "Historical PPO evaluation requires a local private RLKMC-MASSIVE-main checkout; "
        "the public repository only ships the minimal kmcteacher_backend teacher subset."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SwarmThinkers PPO on macro KMC teacher segments"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--cache", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--print_samples", type=int, default=5)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--save_all_samples",
        action="store_true",
        help="Save all paired PPO-vs-KMC samples to the output JSON",
    )
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


def _load_samples(
    cache_path: Path,
    split: str,
    limit: int,
) -> tuple[list[macro_mod.MacroSegmentSample], dict[str, object], dict[str, object]]:
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    samples = [macro_mod.MacroSegmentSample(**item) for item in payload[split]]
    if limit > 0:
        samples = samples[:limit]
    return samples, payload.get("stats", {}), payload.get("signature", {})


def _restore_env_state(env_wrapper: KMCEnvWrapper, sample: macro_mod.MacroSegmentSample) -> None:
    if env_wrapper.env is None:
        env_wrapper.reset()
    env = env_wrapper.env
    assert env is not None

    start_vacancies = np.asarray(sample.start_vacancy_positions, dtype=np.int32)
    start_cu = np.asarray(sample.start_cu_positions, dtype=np.int32)

    env.V_nums = int(start_vacancies.shape[0])
    env.Cu_nums = int(start_cu.shape[0])
    env.args.lattice_v_nums = int(env.V_nums)
    env.args.lattice_cu_nums = int(env.Cu_nums)
    env.num_agents = int(env.V_nums)

    env.vac_pos_set = {tuple(map(int, pos)) for pos in start_vacancies.tolist()}
    env.cu_pos_set = {tuple(map(int, pos)) for pos in start_cu.tolist()}
    env.cu_pos = start_cu.copy()

    env.v_pos_to_id = {
        tuple(map(int, start_vacancies[idx].tolist())): idx
        for idx in range(start_vacancies.shape[0])
    }
    env.v_pos_of_id = {
        idx: tuple(map(int, start_vacancies[idx].tolist()))
        for idx in range(start_vacancies.shape[0])
    }
    env.cu_pos_of_id = {
        idx + env.V_nums: tuple(map(int, start_cu[idx].tolist()))
        for idx in range(start_cu.shape[0])
    }
    env._init_vacancy_mappings()
    env._build_cu_pos_index()
    env._rebuild_global_lin_cache()

    if getattr(env.args, "compute_global_static_env_reset", True):
        env.nn1_types, env.nn2_types, env.nn1_nn1_types, env.nn1_nn2_types = (
            env._calculate_vacancy_local_environments_sparse()
        )

    env.diffusion_rates = env.calculate_diffusion_rate()
    env.time = 0.0
    env.energy_history = []
    env.time_history = []
    env.energy_last = env.calculate_system_energy()
    env_wrapper.timestep = 0


def _build_env_cfg(config: dict[str, object]) -> dict[str, object]:
    return {
        "lattice_size": tuple(config["lattice_size"]),
        "cu_density": float(config["cu_density"]),
        "v_density": float(config["v_density"]),
        "max_episode_steps": int(config["max_episode_steps"]),
        "temperature": float(config["temperature"]),
        "reward_scale": float(config["reward_scale"]),
        "neighbor_order": str(config["neighbor_order"]),
        "max_vacancies": int(config["max_vacancies"]),
        "max_defects": int(config["max_defects"]),
        "max_shells": int(config["max_shells"]),
        "stats_dim": 10,
    }


def _build_agent(config: dict[str, object], checkpoint_path: Path, device: torch.device) -> PPOGNNAgent:
    _require_private_rlkmc_checkout()
    action_space_size = int(config["max_vacancies"]) * 8
    agent = PPOGNNAgent(
        max_vacancies=int(config["max_vacancies"]),
        max_defects=int(config["max_defects"]),
        max_shells=int(config["max_shells"]),
        stats_dim=10,
        lattice_size=tuple(config["lattice_size"]),
        neighbor_order=str(config["neighbor_order"]),
        action_space_size=action_space_size,
        graph_hidden_size=int(config.get("graph_hidden_size", 32)),
        latent_dim=int(config.get("latent_dim", 16)),
    ).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    agent.load_state_dict(state_dict)
    agent.eval()
    return agent


def main() -> None:
    args = parse_args()
    _require_private_rlkmc_checkout()
    checkpoint_path = Path(args.checkpoint)
    config_path = Path(args.config)
    cache_path = Path(args.cache)

    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    device = torch.device(args.device)
    agent = _build_agent(config, checkpoint_path, device)
    env_cfg = _build_env_cfg(config)
    env_wrapper = KMCEnvWrapper(env_cfg)

    samples, dataset_stats, cache_signature = _load_samples(cache_path, args.split, args.limit)

    reward_scale = float(config["reward_scale"])
    pred_reward_sum: list[float] = []
    pred_delta_e: list[float] = []
    pred_tau_exp: list[float] = []
    pred_tau_real: list[float] = []
    true_reward_sum: list[float] = []
    true_delta_e: list[float] = []
    true_tau_exp: list[float] = []
    true_tau_real: list[float] = []
    sample_rows: list[dict[str, object]] = []

    with torch.no_grad():
        for index, sample in enumerate(samples):
            _restore_env_state(env_wrapper, sample)

            obs = env_wrapper._obs()
            mask = env_wrapper._mask()
            sample_pred_reward = 0.0
            sample_pred_delta_e = 0.0
            sample_pred_tau_exp = 0.0
            sample_pred_tau_real = 0.0
            actions: list[int] = []

            for _ in range(int(sample.horizon_k)):
                env = env_wrapper.env
                assert env is not None
                env._ensure_diffusion_rates()
                flat_rates = [
                    rate
                    for vac_rates in env.diffusion_rates
                    for rate in vac_rates
                    if rate > 0
                ]
                total_rate = float(np.sum(flat_rates)) if flat_rates else 0.0
                sample_pred_tau_exp += 1.0 / total_rate if total_rate > 0 else 0.0

                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
                action, _, _, _ = agent.get_action_and_value(obs_t, mask_t, deterministic=True)
                action_i = int(action.item())
                actions.append(action_i)

                obs, mask, reward, _done, info = env_wrapper.step(action_i)
                sample_pred_reward += float(reward)
                sample_pred_delta_e += float(info["delta_E"])
                sample_pred_tau_real += float(info["delta_t"])

            pred_reward_sum.append(sample_pred_reward)
            pred_delta_e.append(sample_pred_delta_e)
            pred_tau_exp.append(sample_pred_tau_exp)
            pred_tau_real.append(sample_pred_tau_real)
            true_reward_sum.append(float(sample.reward_sum))
            true_delta_e.append(float(sample.reward_sum / reward_scale))
            true_tau_exp.append(float(sample.tau_exp))
            true_tau_real.append(float(sample.tau_real))

            sample_rows.append(
                {
                    "index": index,
                    "segment_k": int(sample.horizon_k),
                    "traditional_kmc_reward_sum": float(sample.reward_sum),
                    "traditional_kmc_delta_e": float(sample.reward_sum / reward_scale),
                    "traditional_kmc_expected_tau": float(sample.tau_exp),
                    "traditional_kmc_realized_tau": float(sample.tau_real),
                    "predicted_reward_sum": float(sample_pred_reward),
                    "predicted_delta_e": float(sample_pred_delta_e),
                    "predicted_tau": float(sample_pred_tau_exp),
                    "predicted_tau_realized": float(sample_pred_tau_real),
                    "actions": actions,
                }
            )

    pred_reward_sum_np = np.asarray(pred_reward_sum, dtype=np.float64)
    pred_delta_e_np = np.asarray(pred_delta_e, dtype=np.float64)
    pred_tau_exp_np = np.asarray(pred_tau_exp, dtype=np.float64)
    pred_tau_real_np = np.asarray(pred_tau_real, dtype=np.float64)
    true_reward_sum_np = np.asarray(true_reward_sum, dtype=np.float64)
    true_delta_e_np = np.asarray(true_delta_e, dtype=np.float64)
    true_tau_exp_np = np.asarray(true_tau_exp, dtype=np.float64)
    true_tau_real_np = np.asarray(true_tau_real, dtype=np.float64)

    summary = {
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "cache": str(cache_path),
        "split": args.split,
        "num_samples": int(len(samples)),
        "segment_k": int(samples[0].horizon_k) if samples else 0,
        "cache_signature": cache_signature,
        "dataset_stats": dataset_stats.get(args.split, {}),
        "teacher_source": "traditional_kmc_segment_cache",
        "model_type": "swarmthinkers_ppo",
        "reward_sum": _compute_metrics(pred_reward_sum_np, true_reward_sum_np),
        "delta_e": _compute_metrics(pred_delta_e_np, true_delta_e_np),
        "tau_expected": {
            **_compute_metrics(pred_tau_exp_np, true_tau_exp_np),
            **_compute_log_metrics(pred_tau_exp_np, true_tau_exp_np),
            "traditional_mean": float(np.mean(true_tau_exp_np)),
            "predicted_mean": float(np.mean(pred_tau_exp_np)),
        },
        "tau_realized": {
            **_compute_metrics(pred_tau_real_np, true_tau_real_np),
            **_compute_log_metrics(pred_tau_real_np, true_tau_real_np),
            "traditional_mean": float(np.mean(true_tau_real_np)),
            "predicted_mean": float(np.mean(pred_tau_real_np)),
        },
        "traditional_energy": {
            "reward_sum_mean": float(np.mean(true_reward_sum_np)),
            "delta_e_mean": float(np.mean(true_delta_e_np)),
        },
        "predicted_energy": {
            "reward_sum_mean": float(np.mean(pred_reward_sum_np)),
            "delta_e_mean": float(np.mean(pred_delta_e_np)),
        },
        "sample_preview": sample_rows[: max(args.print_samples, 0)],
    }
    if args.save_all_samples:
        summary["all_samples"] = sample_rows

    print("=" * 60)
    print("SwarmThinkers PPO vs Traditional KMC Teacher Segments")
    print(f"samples={len(samples)}, split={args.split}, segment_k={summary['segment_k']}")
    print("=" * 60)
    print(
        "Traditional KMC energy/time means: "
        f"reward_sum={summary['traditional_energy']['reward_sum_mean']:.6f}, "
        f"delta_E={summary['traditional_energy']['delta_e_mean']:.6f}, "
        f"E[tau]={summary['tau_expected']['traditional_mean']:.6e}, "
        f"real_tau={summary['tau_realized']['traditional_mean']:.6e}"
    )
    print(
        "PPO segment means: "
        f"reward_sum={summary['predicted_energy']['reward_sum_mean']:.6f}, "
        f"delta_E={summary['predicted_energy']['delta_e_mean']:.6f}, "
        f"pred_E[tau]={summary['tau_expected']['predicted_mean']:.6e}, "
        f"pred_real_tau={summary['tau_realized']['predicted_mean']:.6e}"
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