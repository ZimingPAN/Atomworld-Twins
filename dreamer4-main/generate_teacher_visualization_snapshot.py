#!/usr/bin/env python3
"""Generate full-Cu initial/final teacher snapshots for appendix visualization."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[0]
PROJECT_ROOT = ROOT.parent
RLKMC = PROJECT_ROOT / "kmcteacher_backend"
LIGHTZERO = PROJECT_ROOT / "LightZero-main"
for path in [str(ROOT), str(RLKMC), str(LIGHTZERO)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from train_dreamer_macro_edit import CU_TYPE, MacroKMCEnv, _sample_teacher_action


def _sorted_coords(array_like) -> np.ndarray:
    arr = np.asarray(array_like, dtype=np.int32).reshape(-1, 3)
    if len(arr) == 0:
        return arr
    order = np.lexsort((arr[:, 2], arr[:, 1], arr[:, 0]))
    return arr[order]


def _coord_set(array_like) -> set[tuple[int, int, int]]:
    return {tuple(map(int, row)) for row in np.asarray(array_like, dtype=np.int32).reshape(-1, 3).tolist()}


def _positive_total_rate(env: MacroKMCEnv) -> float:
    env.env._ensure_diffusion_rates()
    return float(
        sum(
            float(rate)
            for vac_rates in env.env.diffusion_rates
            for rate in vac_rates
            if float(rate) > 0.0
        )
    )


def generate_snapshot(args: argparse.Namespace) -> Path:
    cfg = {
        "lattice_size": list(args.lattice_size),
        "cu_density": float(args.cu_density),
        "v_density": float(args.v_density),
        "max_episode_steps": int(args.steps) + 1,
        "max_vacancies": int(args.max_vacancies),
        "max_defects": int(args.max_defects),
        "max_shells": int(args.max_shells),
        "neighbor_order": str(args.neighbor_order),
        "reward_scale": float(args.reward_scale),
        "temperature": float(args.temperature),
        "stats_dim": int(args.stats_dim),
        "rlkmc_topk": int(args.rlk_mc_topk),
    }
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    started = time.time()
    env = MacroKMCEnv(cfg)
    env.reset()
    initial_cu = _sorted_coords(env.env.get_cu_array())
    initial_vac = _sorted_coords(env.env.get_vacancy_array())
    initial_energy = float(env.env.calculate_system_energy())

    tau_exp_sum = 0.0
    tau_real_sum = 0.0
    cu_exchange_events = 0
    completed = 0
    for _ in range(int(args.steps)):
        action = _sample_teacher_action(env, rng)
        if action is None:
            break
        total_rate = _positive_total_rate(env)
        expected_delta_t = 1.0 / total_rate if total_rate > 0.0 else 0.0
        delta_t = -np.log(np.random.rand()) / total_rate if total_rate > 0.0 else 0.0
        _, _, _, _, moving_type = env.env._decode_action(int(action))
        env.env.step_fast(int(action), env.timestep)
        env.env.time += delta_t
        env.env.time_history.append(env.env.time)
        env.timestep += 1
        tau_exp_sum += float(expected_delta_t)
        tau_real_sum += float(delta_t)
        if int(moving_type) == int(CU_TYPE):
            cu_exchange_events += 1
        completed += 1
        if completed % int(args.progress_every) == 0:
            print(
                f"progress steps={completed}/{args.steps} "
                f"cu_exchange_events={cu_exchange_events} elapsed={time.time() - started:.1f}s",
                flush=True,
            )
        if env.timestep >= env.max_steps:
            break

    final_cu = _sorted_coords(env.env.get_cu_array())
    final_vac = _sorted_coords(env.env.get_vacancy_array())
    final_energy = float(env.env.calculate_system_energy())
    initial_set = _coord_set(initial_cu)
    final_set = _coord_set(final_cu)
    removed = sorted(initial_set - final_set)
    added = sorted(final_set - initial_set)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "source": "continuous_teacher_kmc",
            "seed": int(args.seed),
            "micro_steps_requested": int(args.steps),
            "micro_steps_completed": int(completed),
            "cu_density": cfg["cu_density"],
            "v_density": cfg["v_density"],
            "lattice_size": cfg["lattice_size"],
            "temperature": cfg["temperature"],
            "neighbor_order": cfg["neighbor_order"],
            "tau_exp_sum": float(tau_exp_sum),
            "tau_real_sum": float(tau_real_sum),
            "cu_exchange_events": int(cu_exchange_events),
            "initial_cu": int(len(initial_cu)),
            "final_cu": int(len(final_cu)),
            "symmetric_difference": int(len(initial_set ^ final_set)),
            "removed_cu_sites": int(len(removed)),
            "added_cu_sites": int(len(added)),
            "elapsed_seconds": float(time.time() - started),
        },
        "snapshots": [
            {
                "index": 0,
                "teacher": {
                    "vacancies": initial_vac.tolist(),
                    "cu": initial_cu.tolist(),
                    "cu_total": int(len(initial_cu)),
                    "energy": initial_energy,
                },
            },
            {
                "index": int(completed),
                "teacher": {
                    "vacancies": final_vac.tolist(),
                    "cu": final_cu.tolist(),
                    "cu_total": int(len(final_cu)),
                    "energy": final_energy,
                },
            },
        ],
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["metadata"], indent=2), flush=True)
    print(out, flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lattice_size", type=int, nargs=3, default=[40, 40, 40])
    parser.add_argument("--cu_density", type=float, default=0.0134)
    parser.add_argument("--v_density", type=float, default=0.0002)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--neighbor_order", type=str, default="2NN")
    parser.add_argument("--reward_scale", type=float, default=10.0)
    parser.add_argument("--max_vacancies", type=int, default=32)
    parser.add_argument("--max_defects", type=int, default=64)
    parser.add_argument("--max_shells", type=int, default=16)
    parser.add_argument("--stats_dim", type=int, default=10)
    parser.add_argument("--rlk_mc_topk", type=int, default=16)
    parser.add_argument("--progress_every", type=int, default=1000)
    generate_snapshot(parser.parse_args())


if __name__ == "__main__":
    main()
