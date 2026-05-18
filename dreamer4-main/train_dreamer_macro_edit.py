from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[0]
RLKMC = ROOT.parent / "kmcteacher_backend"
LIGHTZERO = ROOT.parent / "LightZero-main"
for path in [str(ROOT), str(RLKMC), str(LIGHTZERO)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from RL4KMC.envs.kmc import KMCEnv
from RL4KMC.parser.parser import get_config
from RL4KMC.world_models import DefectGraphObservationShape, build_defect_graph_observation
from dreamer4.macro_edit import (
    MacroDreamerEditModel,
    NUM_SITE_TYPES,
    kl_divergence_diag_gaussian,
    lognormal_nll,
    macro_duration_baseline_log_tau,
    project_types_by_inventory,
    combine_action_endpoint_logits,
    projected_edit_logits_from_types,
    teacher_path_summary_dim,
)


FE_TYPE = 0
CU_TYPE = 1
V_TYPE = 2
BCC_NN1_OFFSETS = (
    (1, 1, 1),
    (-1, -1, -1),
    (1, 1, -1),
    (-1, -1, 1),
    (1, -1, 1),
    (-1, 1, -1),
    (-1, 1, 1),
    (1, -1, -1),
)


@dataclass
class MacroSegmentSample:
    start_obs: np.ndarray
    next_obs: np.ndarray
    start_vacancy_positions: np.ndarray
    start_cu_positions: np.ndarray
    global_summary: np.ndarray
    teacher_path_summary: np.ndarray
    candidate_positions: np.ndarray
    nearest_vacancy_offset: np.ndarray
    reach_depth: np.ndarray
    is_start_vacancy: np.ndarray
    current_types: np.ndarray
    target_types: np.ndarray
    candidate_mask: np.ndarray
    changed_mask: np.ndarray
    tau_exp: float
    tau_real: float
    reward_sum: float
    horizon_k: int
    box_dims: np.ndarray
    planner_projected_changed_mask: np.ndarray | None = None
    planner_teacher_overlap_f1: float = 0.0
    planner_candidate_teacher_changed_mask: np.ndarray | None = None
    planner_candidate_false_positive_mask: np.ndarray | None = None
    planner_candidate_quality_target: float = 0.0
    planner_candidate_quality_available: float = 0.0
    teacher_touched_mask: np.ndarray | None = None
    teacher_action_source_mask: np.ndarray | None = None
    teacher_action_destination_mask: np.ndarray | None = None
    teacher_action_edge_pair_indices: np.ndarray | None = None
    teacher_action_edge_pair_mask: np.ndarray | None = None
    teacher_action_edge_pair_support_mask: np.ndarray | None = None
    teacher_action_edge_pair_moving_type: np.ndarray | None = None
    teacher_action_edge_pair_order: np.ndarray | None = None
    teacher_vacancy_pair_indices: np.ndarray | None = None
    teacher_vacancy_pair_mask: np.ndarray | None = None
    teacher_vacancy_pair_moving_type: np.ndarray | None = None
    teacher_vacancy_pair_order: np.ndarray | None = None
    teacher_action_sequence_indices: np.ndarray | None = None
    teacher_action_sequence_mask: np.ndarray | None = None
    teacher_action_sequence_moving_type: np.ndarray | None = None
    teacher_action_sequence_order: np.ndarray | None = None
    teacher_action_rollout_changed_mask: np.ndarray | None = None


@dataclass(frozen=True)
class AdaptiveBoundaryConfig:
    mode: str = "fixed_k"
    min_k: int = 1
    candidate_horizon_source: str = "nominal"
    key_moving_types: tuple[int, ...] = (CU_TYPE,)
    min_touched_sites: int = 0
    abs_delta_e_threshold: float = 0.0
    cumulative_abs_delta_e_threshold: float = 0.0


class MacroSegmentDataset(Dataset):
    def __init__(self, samples: list[MacroSegmentSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> MacroSegmentSample:
        return self.samples[idx]


def _normalize_segment_ks(segment_k: int, segment_ks: Optional[list[int]] = None) -> list[int]:
    raw = segment_ks if segment_ks else [segment_k]
    normalized = sorted({int(k) for k in raw})
    if not normalized or any(k <= 0 for k in normalized):
        raise ValueError(f"segment horizons must be positive integers, got {raw}")
    return normalized


def _segment_ks_from_args(args: argparse.Namespace) -> list[int]:
    return _normalize_segment_ks(int(args.segment_k), getattr(args, "segment_ks", None))


def _segment_ks_from_ckpt_args(ckpt_args: dict[str, object]) -> list[int]:
    if ckpt_args.get("segment_ks"):
        return _normalize_segment_ks(int(ckpt_args.get("segment_k", 4)), list(ckpt_args["segment_ks"]))
    return _normalize_segment_ks(int(ckpt_args.get("segment_k", 4)), None)


def _summary_horizon_k_from_segment_ks(segment_ks: list[int]) -> int:
    return max(segment_ks)


def _split_segments_per_k(args: argparse.Namespace, split: str, segment_ks: list[int]) -> int:
    if split == "train":
        explicit = getattr(args, "train_segments_per_k", None)
        fallback = int(args.train_segments)
    elif split == "val":
        explicit = getattr(args, "val_segments_per_k", None)
        fallback = int(args.val_segments)
    else:
        raise ValueError(f"unknown split: {split}")
    if explicit is not None:
        return int(explicit)
    return fallback if len(segment_ks) == 1 else fallback


def _split_total_segments(args: argparse.Namespace, split: str, segment_ks: list[int]) -> int:
    return int(_split_segments_per_k(args, split, segment_ks) * len(segment_ks))


def _path_fingerprint(path: Optional[str]) -> dict[str, object] | None:
    if not path:
        return None
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        return {"path": str(path), "exists": False}
    stat = ckpt_path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _build_args(cfg: dict):
    parser = get_config()
    args = parser.parse_known_args([])[0]
    total = int(np.prod(cfg["lattice_size"]) * 2)
    args.lattice_size = list(cfg["lattice_size"])
    args.temperature = cfg.get("temperature", 300.0)
    args.reward_scale = cfg.get("reward_scale", 1.0)
    args.topk = cfg.get("rlkmc_topk", 16)
    args.device = "cpu"
    args.cu_density = cfg["cu_density"]
    args.v_density = cfg["v_density"]
    args.lattice_cu_nums = int(round(cfg["cu_density"] * total))
    args.lattice_v_nums = max(int(round(cfg["v_density"] * total)), 1)
    args.compute_global_static_env_reset = True
    args.skip_stats = True
    args.skip_global_diffusion_reset = False
    args.max_ssa_rounds = cfg["max_episode_steps"]
    args.neighbor_order = cfg.get("neighbor_order", "2NN")
    return args


class MacroKMCEnv:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.shape = DefectGraphObservationShape(
            max_vacancies=cfg["max_vacancies"],
            max_defects=cfg["max_defects"],
            max_shells=cfg["max_shells"],
            node_feat_dim=4,
            stats_dim=cfg.get("stats_dim", 10),
        )
        self.env = KMCEnv(_build_args(cfg))
        self.timestep = 0
        self.max_steps = cfg["max_episode_steps"]

    def reset(self) -> np.ndarray:
        self.env.reset()
        self.timestep = 0
        return self.obs()

    def current_total_rate(self) -> float:
        self.env._ensure_diffusion_rates()
        flat = [rate for vac_rates in self.env.diffusion_rates for rate in vac_rates if rate > 0]
        return float(np.sum(flat)) if flat else 0.0

    def obs(self) -> np.ndarray:
        share_obs = np.zeros(self.shape.stats_dim, dtype=np.float32)
        share_obs[0] = self.cfg.get("temperature", 300.0) / 1000.0
        share_obs[1] = self.cfg.get("cu_density", 0.0134)
        share_obs[2] = self.cfg.get("v_density", 0.0002)
        return build_defect_graph_observation(self.env, shape=self.shape, share_obs=share_obs).astype(np.float32)

    def action_mask(self) -> np.ndarray:
        self.env._ensure_diffusion_rates()
        masks = []
        for vac_rates in self.env.diffusion_rates[: self.shape.max_vacancies]:
            masks.extend([1.0 if rate > 0 else 0.0 for rate in vac_rates])
        masks.extend([0.0] * max(0, self.shape.max_vacancies * 8 - len(masks)))
        return np.asarray(masks[: self.shape.max_vacancies * 8], dtype=np.float32)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        total_rate = self.current_total_rate()
        expected_delta_t = 1.0 / total_rate if total_rate > 0 else 0.0
        vac_idx, dir_idx, old_pos, new_pos, moving_type = self.env._decode_action(int(action))
        self.env.step_fast(int(action), self.timestep)
        delta_t = -np.log(np.random.rand()) / total_rate if total_rate > 0 else 0.0
        self.env.time += delta_t
        self.env.time_history.append(self.env.time)
        energy_after = self.env.calculate_system_energy()
        delta_E = self.env.energy_last - energy_after
        reward = float(delta_E * self.env.args.reward_scale)
        self.env.energy_last = energy_after
        self.env.energy_history.append(energy_after)
        self.timestep += 1
        done = self.timestep >= self.max_steps
        return self.obs(), reward, done, {
            "delta_t": float(delta_t),
            "expected_delta_t": float(expected_delta_t),
            "total_rate": float(total_rate),
            "delta_E": float(delta_E),
            "dir_idx": int(dir_idx),
            "vac_idx": int(vac_idx),
            "moving_type": int(moving_type),
            "old_pos": np.asarray(old_pos, dtype=np.int32),
            "new_pos": np.asarray(new_pos, dtype=np.int32),
        }


def _periodic_offset(src: np.ndarray, dst: np.ndarray, box: np.ndarray) -> np.ndarray:
    delta = src - dst
    return delta - np.round(delta / box) * box


def _positions_to_type_lookup(vacancies: np.ndarray, cu_atoms: np.ndarray) -> tuple[set[tuple[int, int, int]], set[tuple[int, int, int]]]:
    vac_set = {tuple(map(int, pos)) for pos in vacancies.tolist()}
    cu_set = {tuple(map(int, pos)) for pos in cu_atoms.tolist()}
    return vac_set, cu_set


def _changed_positions_between(
    start_vac_set: set[tuple[int, int, int]],
    start_cu_set: set[tuple[int, int, int]],
    end_vac_set: set[tuple[int, int, int]],
    end_cu_set: set[tuple[int, int, int]],
) -> set[tuple[int, int, int]]:
    positions = start_vac_set | start_cu_set | end_vac_set | end_cu_set
    return {
        pos
        for pos in positions
        if _type_from_lookup(pos, start_vac_set, start_cu_set) != _type_from_lookup(pos, end_vac_set, end_cu_set)
    }


def _type_from_lookup(pos: tuple[int, int, int], vac_set: set[tuple[int, int, int]], cu_set: set[tuple[int, int, int]]) -> int:
    if pos in vac_set:
        return V_TYPE
    if pos in cu_set:
        return CU_TYPE
    return FE_TYPE


def _one_hop_neighbors(pos: tuple[int, int, int], nn1: np.ndarray, box: np.ndarray) -> list[tuple[int, int, int]]:
    base = np.asarray(pos, dtype=np.int32)
    out = []
    for step in nn1:
        nxt = tuple(((base + step) % box).tolist())
        out.append(nxt)
    return out


def _vacancy_rate_sums(env: MacroKMCEnv) -> np.ndarray:
    env.env._ensure_diffusion_rates()
    rate_sums = [float(np.sum([rate for rate in vac_rates if rate > 0])) for vac_rates in env.env.diffusion_rates]
    return np.asarray(rate_sums, dtype=np.float32)


def _sample_teacher_action(env: MacroKMCEnv, rng: np.random.Generator) -> Optional[int]:
    env.env._ensure_diffusion_rates()
    actions: list[int] = []
    rates: list[float] = []
    for vac_idx, vac_rates in enumerate(env.env.diffusion_rates):
        for dir_idx, rate in enumerate(vac_rates):
            if rate > 0:
                actions.append(vac_idx * 8 + dir_idx)
                rates.append(float(rate))
    if not actions:
        return None
    probs = np.asarray(rates, dtype=np.float64)
    probs = probs / probs.sum()
    return int(actions[int(rng.choice(len(actions), p=probs))])


def _load_neural_teacher(model_path: str, env_cfg: dict, device: str = "cpu"):
    """Load a DreamerKMCAgent as neural teacher for segment collection."""
    from train_dreamer_standalone import DreamerKMCAgent
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    feature_flags = {
        "use_topology_head": any(k.startswith("topology_head.") for k in state_dict),
        "use_shortcut_forcing": "horizon_embed.weight" in state_dict,
    }
    agent = DreamerKMCAgent(
        dim_latent=16,
        max_vacancies=env_cfg["max_vacancies"],
        max_defects=env_cfg["max_defects"],
        max_shells=env_cfg["max_shells"],
        stats_dim=env_cfg["stats_dim"],
        lattice_size=env_cfg["lattice_size"],
        neighbor_order=env_cfg["neighbor_order"],
        action_space_size=env_cfg["max_vacancies"] * 8,
        graph_hidden_size=32,
        **feature_flags,
    ).to(device)
    agent.load_state_dict(state_dict)
    agent.eval()
    print(f"Loaded neural teacher from {model_path} (topology_head={feature_flags['use_topology_head']}, "
          f"shortcut_forcing={feature_flags['use_shortcut_forcing']})")
    return agent


def _sample_neural_teacher_action(
    env: MacroKMCEnv,
    agent: torch.nn.Module,
    device: str = "cpu",
    temperature: float = 1.0,
) -> Optional[int]:
    """Sample action from a neural teacher (DreamerKMCAgent) with temperature softmax."""
    obs = env.obs()
    mask = env.action_mask()
    with torch.no_grad():
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        mask_t = torch.tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
        latent = agent.encode(obs_t)
        logits = agent.forward_policy(latent, mask_t)
        # Temperature-scaled softmax sampling instead of argmax
        probs = torch.softmax(logits[0] / max(temperature, 1e-6), dim=-1)
        action = int(torch.multinomial(probs, 1).item())
    if mask[action] < 0.5:
        return None
    return action


class NaturalTeacherAdapter:
    """Minimal simulator-teacher interface for macro segment collection."""

    backend_name = "base"

    def vacancy_positions(self) -> np.ndarray:
        raise NotImplementedError

    def cu_positions(self) -> np.ndarray:
        raise NotImplementedError

    def box_dims(self) -> np.ndarray:
        raise NotImplementedError

    def nn1_offsets(self) -> np.ndarray:
        raise NotImplementedError

    def global_summary(self) -> np.ndarray:
        raise NotImplementedError

    def build_candidate_positions(
        self,
        *,
        horizon_k: int,
        max_seed_vacancies: int,
        max_candidate_sites: int,
    ) -> tuple[list[tuple[int, int, int]], dict[tuple[int, int, int], int], np.ndarray]:
        raise NotImplementedError

    def step_once(self, rng: np.random.Generator) -> tuple[np.ndarray, float, bool, dict] | None:
        raise NotImplementedError


class KMCNaturalTeacherAdapter(NaturalTeacherAdapter):
    backend_name = "kmc"

    def __init__(
        self,
        *,
        env: MacroKMCEnv,
        teacher_mode: str,
        neural_teacher: Optional[torch.nn.Module],
        neural_teacher_device: str,
        neural_teacher_temperature: float,
        neural_teacher_epsilon: float,
    ) -> None:
        self.env = env
        self.teacher_mode = teacher_mode
        self.neural_teacher = neural_teacher
        self.neural_teacher_device = neural_teacher_device
        self.neural_teacher_temperature = float(neural_teacher_temperature)
        self.neural_teacher_epsilon = float(neural_teacher_epsilon)

    def _sample_action(self, rng: np.random.Generator) -> Optional[int]:
        if self.teacher_mode == "neural" and self.neural_teacher is not None:
            if self.neural_teacher_epsilon > 0 and rng.random() < self.neural_teacher_epsilon:
                return _sample_teacher_action(self.env, rng)
            return _sample_neural_teacher_action(
                self.env,
                self.neural_teacher,
                self.neural_teacher_device,
                temperature=self.neural_teacher_temperature,
            )
        return _sample_teacher_action(self.env, rng)

    def vacancy_positions(self) -> np.ndarray:
        return self.env.env.get_vacancy_array().astype(np.int32)

    def cu_positions(self) -> np.ndarray:
        return self.env.env.get_cu_array().astype(np.int32)

    def box_dims(self) -> np.ndarray:
        return np.asarray(self.env.env.dims, dtype=np.int32)

    def nn1_offsets(self) -> np.ndarray:
        return np.asarray(self.env.env.NN1, dtype=np.int32)

    def global_summary(self) -> np.ndarray:
        return _global_summary(self.env)

    def build_candidate_positions(
        self,
        *,
        horizon_k: int,
        max_seed_vacancies: int,
        max_candidate_sites: int,
    ) -> tuple[list[tuple[int, int, int]], dict[tuple[int, int, int], int], np.ndarray]:
        return _build_candidate_positions(
            self.env,
            horizon_k,
            max_seed_vacancies=max_seed_vacancies,
            max_candidate_sites=max_candidate_sites,
        )

    def step_once(self, rng: np.random.Generator) -> tuple[np.ndarray, float, bool, dict] | None:
        action = self._sample_action(rng)
        if action is None:
            return None
        return self.env.step(action)


def _build_natural_teacher_adapter(
    *,
    backend: str,
    env: MacroKMCEnv,
    teacher_mode: str,
    neural_teacher: Optional[torch.nn.Module],
    neural_teacher_device: str,
    neural_teacher_temperature: float,
    neural_teacher_epsilon: float,
) -> NaturalTeacherAdapter:
    if backend != "kmc":
        raise NotImplementedError(
            f"natural_teacher_backend={backend!r} is not implemented yet; "
            "add a NaturalTeacherAdapter that exposes state, support, path summary, and physical duration"
        )
    return KMCNaturalTeacherAdapter(
        env=env,
        teacher_mode=teacher_mode,
        neural_teacher=neural_teacher,
        neural_teacher_device=neural_teacher_device,
        neural_teacher_temperature=neural_teacher_temperature,
        neural_teacher_epsilon=neural_teacher_epsilon,
    )


def _adaptive_boundary_reached(
    *,
    config: AdaptiveBoundaryConfig,
    step_count: int,
    latest_info: dict,
    touched_positions: set[tuple[int, int, int]],
    cumulative_abs_delta_e: float,
) -> bool:
    if config.mode != "adaptive_key_event":
        return False
    if step_count < max(int(config.min_k), 1):
        return False
    moving_type = int(latest_info.get("moving_type", -1))
    if config.key_moving_types and moving_type in set(config.key_moving_types):
        return True
    if int(config.min_touched_sites) > 0 and len(touched_positions) >= int(config.min_touched_sites):
        return True
    if float(config.abs_delta_e_threshold) > 0.0 and abs(float(latest_info.get("delta_E", 0.0))) >= float(config.abs_delta_e_threshold):
        return True
    if (
        float(config.cumulative_abs_delta_e_threshold) > 0.0
        and float(cumulative_abs_delta_e) >= float(config.cumulative_abs_delta_e_threshold)
    ):
        return True
    return False


def _rollout_teacher_path(
    *,
    adapter: NaturalTeacherAdapter,
    rng: np.random.Generator,
    max_horizon_k: int,
    boundary_config: AdaptiveBoundaryConfig,
) -> tuple[np.ndarray, bool, list[dict], float, float, float, set[tuple[int, int, int]], int, bool]:
    tau_exp = 0.0
    tau_real = 0.0
    reward_sum = 0.0
    touched_positions: set[tuple[int, int, int]] = set()
    path_infos: list[dict] = []
    done = False
    boundary_hit = False
    next_obs = np.zeros((0,), dtype=np.float32)
    cumulative_abs_delta_e = 0.0
    for _ in range(max_horizon_k):
        step_result = adapter.step_once(rng)
        if step_result is None:
            done = True
            break
        next_obs, reward, done, info = step_result
        tau_exp += float(info["expected_delta_t"])
        tau_real += float(info["delta_t"])
        reward_sum += float(reward)
        cumulative_abs_delta_e += abs(float(info.get("delta_E", 0.0)))
        path_infos.append(info)
        touched_positions.add(tuple(map(int, info["old_pos"].tolist())))
        touched_positions.add(tuple(map(int, info["new_pos"].tolist())))
        step_count = len(path_infos)
        if done:
            break
        if _adaptive_boundary_reached(
            config=boundary_config,
            step_count=step_count,
            latest_info=info,
            touched_positions=touched_positions,
            cumulative_abs_delta_e=cumulative_abs_delta_e,
        ):
            boundary_hit = True
            break
    return next_obs, done, path_infos, tau_exp, tau_real, reward_sum, touched_positions, len(path_infos), boundary_hit


def _global_summary(env: MacroKMCEnv) -> np.ndarray:
    stats = env.env.get_system_stats().astype(np.float32)
    env.env._ensure_diffusion_rates()
    positive_rates = np.asarray([rate for vac_rates in env.env.diffusion_rates for rate in vac_rates if rate > 0], dtype=np.float32)
    rate_sums = _vacancy_rate_sums(env)
    top_rates = np.sort(positive_rates)[-8:] if positive_rates.size > 0 else np.zeros((0,), dtype=np.float32)
    summary = np.zeros((16,), dtype=np.float32)
    summary[: min(10, stats.size)] = stats[:10]
    total_rate = float(positive_rates.sum()) if positive_rates.size > 0 else 0.0
    summary[10] = math.log(total_rate + 1e-12)
    summary[11] = math.log(float(top_rates.mean()) + 1e-12) if top_rates.size > 0 else -27.0
    summary[12] = math.log(float(top_rates.max()) + 1e-12) if top_rates.size > 0 else -27.0
    summary[13] = math.log(float(top_rates.std()) + 1e-12) if top_rates.size > 1 else -27.0
    summary[14] = float((positive_rates.size / max(env.shape.max_vacancies * 8, 1)))
    summary[15] = float((rate_sums > 0).mean()) if rate_sums.size > 0 else 0.0
    return summary
def _teacher_path_summary(
    path_infos: list[dict],
    max_candidate_sites: int,
    horizon_k: int,
    *,
    include_stepwise_features: bool = True,
    summary_horizon_k: Optional[int] = None,
) -> np.ndarray:
    summary_horizon = int(summary_horizon_k if summary_horizon_k is not None else horizon_k)
    if horizon_k > summary_horizon:
        raise ValueError(f"horizon_k={horizon_k} exceeds summary_horizon_k={summary_horizon}")
    direction_hist = np.zeros((8,), dtype=np.float32)
    moving_hist = np.zeros((NUM_SITE_TYPES,), dtype=np.float32)
    log_rates = []
    delta_es = []
    step_log_expected_dt = np.full((summary_horizon,), -27.0, dtype=np.float32)
    step_delta_es = np.zeros((summary_horizon,), dtype=np.float32)
    touched = set()
    vacancy_ids = set()
    for step_idx, info in enumerate(path_infos[:horizon_k]):
        direction_hist[int(info["dir_idx"])] += 1.0
        moving_type = int(info["moving_type"])
        if 0 <= moving_type < NUM_SITE_TYPES:
            moving_hist[moving_type] += 1.0
        log_rates.append(math.log(float(info["total_rate"]) + 1e-12))
        delta_es.append(float(info["delta_E"]))
        step_log_expected_dt[step_idx] = math.log(float(info["expected_delta_t"]) + 1e-12)
        step_delta_es[step_idx] = float(info["delta_E"])
        touched.add(tuple(map(int, info["old_pos"].tolist())))
        touched.add(tuple(map(int, info["new_pos"].tolist())))
        vacancy_ids.add(int(info["vac_idx"]))
    if path_infos:
        direction_hist /= len(path_infos)
        moving_hist /= len(path_infos)
    summary = np.zeros((teacher_path_summary_dim(summary_horizon, include_stepwise_features=include_stepwise_features),), dtype=np.float32)
    summary[:8] = direction_hist
    summary[8:11] = moving_hist
    summary[11] = float(np.mean(log_rates)) if log_rates else -27.0
    summary[12] = float(np.std(log_rates)) if len(log_rates) > 1 else 0.0
    summary[13] = float(np.mean(delta_es)) if delta_es else 0.0
    summary[14] = float(np.mean([de > 0 for de in delta_es])) if delta_es else 0.0
    summary[15] = float(len(touched) / max(max_candidate_sites, 1))
    summary[16] = float(len(vacancy_ids) / max(len(path_infos), 1)) if path_infos else 0.0
    summary[17] = float(len(path_infos) / max(horizon_k, 1))
    if include_stepwise_features:
        summary[18 : 18 + summary_horizon] = step_log_expected_dt
        summary[18 + summary_horizon : 18 + 2 * summary_horizon] = step_delta_es
    return summary


def _select_seed_vacancies(env: MacroKMCEnv, max_seed_vacancies: int) -> np.ndarray:
    vacancy_positions = env.env.get_vacancy_array().astype(np.int32)
    if vacancy_positions.size == 0:
        return np.empty((0, 3), dtype=np.int32)
    rate_sums = _vacancy_rate_sums(env)
    order = np.argsort(rate_sums)[::-1][: max(1, min(max_seed_vacancies, len(vacancy_positions)))]
    return vacancy_positions[order]


def _build_candidate_positions(env: MacroKMCEnv, horizon_k: int, max_seed_vacancies: int, max_candidate_sites: int) -> tuple[list[tuple[int, int, int]], dict[tuple[int, int, int], int], np.ndarray]:
    box = np.asarray(env.env.dims, dtype=np.int32)
    nn1 = np.asarray(env.env.NN1, dtype=np.int32)
    seeds = _select_seed_vacancies(env, max_seed_vacancies)
    if seeds.size == 0 or max_candidate_sites <= 0:
        return [], {}, seeds
    depth_map: dict[tuple[int, int, int], int] = {}
    frontier = {tuple(map(int, pos.tolist())) for pos in seeds}
    for pos in frontier:
        depth_map[pos] = 0
    if len(depth_map) >= max_candidate_sites:
        frontier = set()
    for depth in range(1, horizon_k + 1):
        next_frontier: set[tuple[int, int, int]] = set()
        for pos in frontier:
            for nxt in _one_hop_neighbors(pos, nn1, box):
                if nxt not in depth_map:
                    depth_map[nxt] = depth
                    next_frontier.add(nxt)
        frontier = next_frontier
        if not frontier or len(depth_map) >= max_candidate_sites:
            break

    def rank_key(pos: tuple[int, int, int]) -> tuple[int, float]:
        pos_arr = np.asarray(pos, dtype=np.float32)
        min_dist = min(np.linalg.norm(_periodic_offset(pos_arr, seed.astype(np.float32), box.astype(np.float32))) for seed in seeds)
        return depth_map[pos], float(min_dist)

    ranked = sorted(depth_map.keys(), key=rank_key)
    return ranked[:max_candidate_sites], depth_map, seeds


def _augment_candidate_positions_with_teacher_path(
    *,
    candidate_positions: list[tuple[int, int, int]],
    depth_map: dict[tuple[int, int, int], int],
    seeds: np.ndarray,
    touched_positions: set[tuple[int, int, int]],
    box: np.ndarray,
    nn1: np.ndarray,
    horizon_k: int,
    max_candidate_sites: int,
    teacher_neighbor_depth: int,
) -> tuple[list[tuple[int, int, int]], dict[tuple[int, int, int], int], np.ndarray]:
    if not touched_positions:
        return candidate_positions, depth_map, seeds

    merged_depth = dict(depth_map)
    frontier = set(touched_positions)
    for pos in frontier:
        merged_depth[pos] = min(merged_depth.get(pos, horizon_k + teacher_neighbor_depth), 0)
    for depth in range(1, max(int(teacher_neighbor_depth), 0) + 1):
        next_frontier: set[tuple[int, int, int]] = set()
        for pos in frontier:
            for nxt in _one_hop_neighbors(pos, nn1, box):
                if nxt not in merged_depth or depth < merged_depth[nxt]:
                    merged_depth[nxt] = depth
                    next_frontier.add(nxt)
        frontier = next_frontier
        if not frontier:
            break

    touched_seed_array = np.asarray(sorted(touched_positions), dtype=np.int32)
    all_seeds = np.concatenate([seeds, touched_seed_array], axis=0) if seeds.size > 0 else touched_seed_array

    def rank_key(pos: tuple[int, int, int]) -> tuple[int, int, float]:
        pos_arr = np.asarray(pos, dtype=np.float32)
        min_dist = min(
            np.linalg.norm(_periodic_offset(pos_arr, seed.astype(np.float32), box.astype(np.float32)))
            for seed in all_seeds
        )
        return (0 if pos in touched_positions else 1, merged_depth.get(pos, horizon_k + teacher_neighbor_depth + 1), float(min_dist))

    ranked = sorted(merged_depth.keys(), key=rank_key)
    return ranked[:max_candidate_sites], merged_depth, all_seeds


def _build_patch_features(
    *,
    candidate_positions: list[tuple[int, int, int]],
    depth_map: dict[tuple[int, int, int], int],
    seeds: np.ndarray,
    start_vac_set: set[tuple[int, int, int]],
    start_cu_set: set[tuple[int, int, int]],
    end_vac_set: set[tuple[int, int, int]],
    end_cu_set: set[tuple[int, int, int]],
    max_candidate_sites: int,
    box: np.ndarray,
    horizon_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    positions = np.zeros((max_candidate_sites, 3), dtype=np.float32)
    nearest_offsets = np.zeros((max_candidate_sites, 3), dtype=np.float32)
    reach_depth = np.zeros((max_candidate_sites,), dtype=np.float32)
    is_start_vacancy = np.zeros((max_candidate_sites,), dtype=np.float32)
    current_types = np.zeros((max_candidate_sites,), dtype=np.int64)
    target_types = np.zeros((max_candidate_sites,), dtype=np.int64)
    mask = np.zeros((max_candidate_sites,), dtype=np.float32)
    for idx, pos in enumerate(candidate_positions[:max_candidate_sites]):
        pos_arr = np.asarray(pos, dtype=np.float32)
        positions[idx] = pos_arr
        if len(seeds) > 0:
            offsets = [_periodic_offset(pos_arr, seed.astype(np.float32), box.astype(np.float32)) for seed in seeds]
            nearest = min(offsets, key=lambda item: float(np.linalg.norm(item)))
            nearest_offsets[idx] = nearest.astype(np.float32)
        reach_depth[idx] = float(depth_map.get(pos, horizon_k)) / max(horizon_k, 1)
        is_start_vacancy[idx] = 1.0 if pos in start_vac_set else 0.0
        current_types[idx] = _type_from_lookup(pos, start_vac_set, start_cu_set)
        target_types[idx] = _type_from_lookup(pos, end_vac_set, end_cu_set)
        mask[idx] = 1.0
    changed_mask = (current_types != target_types).astype(np.float32) * mask
    return positions, nearest_offsets, reach_depth, is_start_vacancy, current_types, target_types, changed_mask


def _candidate_mask_from_position_set(
    *,
    candidate_positions: np.ndarray,
    candidate_mask: np.ndarray,
    positions: set[tuple[int, int, int]],
) -> np.ndarray:
    support = np.zeros_like(candidate_mask, dtype=np.float32)
    if not positions:
        return support
    for idx in np.flatnonzero(candidate_mask > 0).tolist():
        pos = tuple(map(int, candidate_positions[idx].astype(np.int32).tolist()))
        if pos in positions:
            support[idx] = 1.0
    return support


def _teacher_action_endpoint_sets(
    path_infos: list[dict],
) -> tuple[set[tuple[int, int, int]], set[tuple[int, int, int]]]:
    source_positions: set[tuple[int, int, int]] = set()
    destination_positions: set[tuple[int, int, int]] = set()
    for info in path_infos:
        old_pos = info.get("old_pos")
        new_pos = info.get("new_pos")
        if old_pos is not None:
            source_positions.add(tuple(map(int, np.asarray(old_pos, dtype=np.int32).tolist())))
        if new_pos is not None:
            destination_positions.add(tuple(map(int, np.asarray(new_pos, dtype=np.int32).tolist())))
    return source_positions, destination_positions


def _teacher_action_edge_pair_targets(
    *,
    candidate_positions: np.ndarray,
    candidate_mask: np.ndarray,
    changed_mask: np.ndarray,
    path_infos: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    max_pairs = int(candidate_mask.shape[0])
    edge_indices = np.full((max_pairs, 2), -1, dtype=np.int64)
    edge_mask = np.zeros((max_pairs,), dtype=np.float32)
    edge_support_mask = np.zeros((max_pairs,), dtype=np.float32)
    edge_moving_type = np.full((max_pairs,), -1, dtype=np.int64)
    edge_order = np.zeros((max_pairs,), dtype=np.float32)
    lookup = {
        tuple(map(int, candidate_positions[idx].astype(np.int32).tolist())): int(idx)
        for idx in np.flatnonzero(candidate_mask > 0).tolist()
    }
    seen: set[tuple[int, int]] = set()
    total_pairs = 0
    covered_pairs = 0
    write_idx = 0
    denom = max(len(path_infos) - 1, 1)
    for path_idx, info in enumerate(path_infos):
        old_pos = info.get("old_pos")
        new_pos = info.get("new_pos")
        if old_pos is None or new_pos is None:
            continue
        total_pairs += 1
        old_key = tuple(map(int, np.asarray(old_pos, dtype=np.int32).tolist()))
        new_key = tuple(map(int, np.asarray(new_pos, dtype=np.int32).tolist()))
        old_idx = lookup.get(old_key)
        new_idx = lookup.get(new_key)
        if old_idx is None or new_idx is None:
            continue
        covered_pairs += 1
        pair = (int(old_idx), int(new_idx))
        if pair in seen:
            continue
        seen.add(pair)
        if write_idx >= max_pairs:
            continue
        edge_indices[write_idx] = np.asarray(pair, dtype=np.int64)
        edge_mask[write_idx] = 1.0
        edge_support_mask[write_idx] = float(
            float(changed_mask[int(old_idx)]) > 0.5 or float(changed_mask[int(new_idx)]) > 0.5
        )
        edge_moving_type[write_idx] = int(info.get("moving_type", -1))
        edge_order[write_idx] = float(path_idx / denom)
        write_idx += 1
    return edge_indices, edge_mask, edge_support_mask, edge_moving_type, edge_order, total_pairs, covered_pairs


def _teacher_action_sequence_targets(
    *,
    candidate_positions: np.ndarray,
    candidate_mask: np.ndarray,
    current_types: np.ndarray,
    path_infos: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    max_steps = int(candidate_mask.shape[0])
    sequence_indices = np.full((max_steps, 2), -1, dtype=np.int64)
    sequence_mask = np.zeros((max_steps,), dtype=np.float32)
    sequence_moving_type = np.full((max_steps,), -1, dtype=np.int64)
    sequence_order = np.zeros((max_steps,), dtype=np.float32)
    rollout_types = np.asarray(current_types, dtype=np.int64).copy()
    lookup = {
        tuple(map(int, candidate_positions[idx].astype(np.int32).tolist())): int(idx)
        for idx in np.flatnonzero(candidate_mask > 0).tolist()
    }
    total_steps = 0
    covered_steps = 0
    write_idx = 0
    denom = max(len(path_infos) - 1, 1)
    for path_idx, info in enumerate(path_infos):
        old_pos = info.get("old_pos")
        new_pos = info.get("new_pos")
        if old_pos is None or new_pos is None:
            continue
        total_steps += 1
        old_key = tuple(map(int, np.asarray(old_pos, dtype=np.int32).tolist()))
        new_key = tuple(map(int, np.asarray(new_pos, dtype=np.int32).tolist()))
        old_idx = lookup.get(old_key)
        new_idx = lookup.get(new_key)
        if old_idx is None or new_idx is None:
            continue
        covered_steps += 1
        moving_type = int(info.get("moving_type", int(rollout_types[int(new_idx)])))
        if write_idx < max_steps:
            sequence_indices[write_idx] = np.asarray([old_idx, new_idx], dtype=np.int64)
            sequence_mask[write_idx] = 1.0
            sequence_moving_type[write_idx] = moving_type
            sequence_order[write_idx] = float(path_idx / denom)
            write_idx += 1
        # KMC actions are decoded as old vacancy -> neighboring moving atom.
        # The atom at new_idx fills old_idx; new_idx becomes the vacancy.
        rollout_types[int(old_idx)] = moving_type
        rollout_types[int(new_idx)] = V_TYPE
    valid = np.asarray(candidate_mask > 0, dtype=bool)
    rollout_changed_mask = ((rollout_types != np.asarray(current_types, dtype=np.int64)) & valid).astype(np.float32)
    return (
        sequence_indices,
        sequence_mask,
        sequence_moving_type,
        sequence_order,
        rollout_changed_mask,
        total_steps,
        covered_steps,
    )


def _teacher_vacancy_displacement_pair_targets_from_sequence(
    *,
    candidate_mask: np.ndarray,
    current_types: np.ndarray,
    target_types: np.ndarray,
    action_sequence_indices: np.ndarray,
    action_sequence_mask: np.ndarray,
    action_sequence_moving_type: np.ndarray,
    action_sequence_order: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    max_pairs = int(candidate_mask.shape[0])
    pair_indices = np.full((max_pairs, 2), -1, dtype=np.int64)
    pair_mask = np.zeros((max_pairs,), dtype=np.float32)
    pair_moving_type = np.full((max_pairs,), -1, dtype=np.int64)
    pair_order = np.zeros((max_pairs,), dtype=np.float32)
    valid = np.asarray(candidate_mask > 0, dtype=bool)
    current_arr = np.asarray(current_types, dtype=np.int64)
    target_arr = np.asarray(target_types, dtype=np.int64)

    vacancy_origin: dict[int, int] = {
        int(idx): int(idx)
        for idx in np.flatnonzero(valid).tolist()
        if int(current_arr[int(idx)]) == V_TYPE
    }
    vacancy_order: dict[int, float] = {idx: 0.0 for idx in vacancy_origin}
    step_count = 0
    covered_steps = 0
    for step_idx, pair in enumerate(np.asarray(action_sequence_indices, dtype=np.int64)):
        if step_idx >= int(action_sequence_mask.shape[0]) or float(action_sequence_mask[step_idx]) <= 0.0:
            continue
        step_count += 1
        if pair.shape[0] != 2:
            continue
        source_idx = int(pair[0])
        dest_idx = int(pair[1])
        if (
            source_idx < 0
            or dest_idx < 0
            or source_idx >= max_pairs
            or dest_idx >= max_pairs
            or not valid[source_idx]
            or not valid[dest_idx]
        ):
            continue
        covered_steps += 1
        origin_idx = vacancy_origin.pop(source_idx, None)
        vacancy_order.pop(source_idx, None)
        if origin_idx is None:
            origin_idx = source_idx
        vacancy_origin[dest_idx] = int(origin_idx)
        vacancy_order[dest_idx] = float(action_sequence_order[step_idx])

    seen: set[tuple[int, int]] = set()
    total_pairs = 0
    covered_pairs = 0
    write_idx = 0
    for final_vacancy_idx, origin_idx in vacancy_origin.items():
        origin_idx = int(origin_idx)
        final_vacancy_idx = int(final_vacancy_idx)
        if origin_idx == final_vacancy_idx:
            continue
        if (
            origin_idx < 0
            or final_vacancy_idx < 0
            or origin_idx >= max_pairs
            or final_vacancy_idx >= max_pairs
            or not valid[origin_idx]
            or not valid[final_vacancy_idx]
        ):
            continue
        # Terminal vacancy displacement is a state-diff target:
        # source starts as vacancy and is filled by an atom, destination starts
        # as an atom and ends as vacancy. This is different from any single
        # microscopic NN1 action edge when the vacancy moves over multiple hops.
        if int(current_arr[origin_idx]) != V_TYPE:
            continue
        if int(target_arr[origin_idx]) not in (FE_TYPE, CU_TYPE):
            continue
        if int(current_arr[final_vacancy_idx]) not in (FE_TYPE, CU_TYPE):
            continue
        if int(target_arr[final_vacancy_idx]) != V_TYPE:
            continue
        total_pairs += 1
        pair_key = (origin_idx, final_vacancy_idx)
        if pair_key in seen:
            continue
        covered_pairs += 1
        seen.add(pair_key)
        if write_idx >= max_pairs:
            continue
        pair_indices[write_idx] = np.asarray(pair_key, dtype=np.int64)
        pair_mask[write_idx] = 1.0
        pair_moving_type[write_idx] = int(target_arr[origin_idx])
        pair_order[write_idx] = float(vacancy_order.get(final_vacancy_idx, 1.0))
        write_idx += 1
    return pair_indices, pair_mask, pair_moving_type, pair_order, total_pairs, covered_pairs


def _collect_segments(
    *,
    env: MacroKMCEnv,
    num_segments: int,
    horizon_k: int,
    max_seed_vacancies: int,
    max_candidate_sites: int,
    rng: np.random.Generator,
    max_attempt_multiplier: int = 20,
    include_stepwise_path_summary: bool = True,
    summary_horizon_k: Optional[int] = None,
    max_segments_per_rollout: int = 50,
    teacher_candidate_neighbor_depth: int = 1,
    teacher_candidate_augmentation: bool = True,
    teacher_mode: str = "kmc",
    neural_teacher: Optional[torch.nn.Module] = None,
    neural_teacher_device: str = "cpu",
    neural_teacher_temperature: float = 1.0,
    neural_teacher_epsilon: float = 0.0,
    natural_teacher_backend: str = "kmc",
    adaptive_boundary_config: Optional[AdaptiveBoundaryConfig] = None,
    include_noop_segments: bool = False,
    keep_after_noop_segments: bool = False,
) -> tuple[list[MacroSegmentSample], dict[str, float]]:
    def restart_env(current_env: MacroKMCEnv) -> tuple[MacroKMCEnv, np.ndarray]:
        new_env = MacroKMCEnv(copy.deepcopy(current_env.cfg))
        return new_env, new_env.reset()

    samples: list[MacroSegmentSample] = []
    stats = {
        "attempts": 0,
        "skipped_uncovered": 0,
        "skipped_terminal": 0,
        "skipped_noop": 0,
        "included_noop": 0,
        "candidate_size_sum": 0.0,
        "teacher_touched_count_sum": 0.0,
        "teacher_touched_mask_count_sum": 0.0,
        "teacher_touched_recall_sum": 0.0,
        "teacher_action_source_mask_count_sum": 0.0,
        "teacher_action_destination_mask_count_sum": 0.0,
        "teacher_action_source_recall_sum": 0.0,
        "teacher_action_destination_recall_sum": 0.0,
        "teacher_action_edge_pair_count_sum": 0.0,
        "teacher_action_edge_pair_covered_sum": 0.0,
        "teacher_action_edge_pair_unique_count_sum": 0.0,
        "teacher_action_edge_pair_support_count_sum": 0.0,
        "teacher_vacancy_pair_count_sum": 0.0,
        "teacher_vacancy_pair_covered_sum": 0.0,
        "teacher_vacancy_pair_unique_count_sum": 0.0,
        "teacher_action_sequence_step_count_sum": 0.0,
        "teacher_action_sequence_covered_sum": 0.0,
        "teacher_action_rollout_changed_count_sum": 0.0,
        "teacher_action_rollout_changed_f1_sum": 0.0,
        "adaptive_boundary_hits": 0,
        "adaptive_truncated_at_max": 0,
        "realized_horizon_sum": 0.0,
    }
    progress_every = max(50, num_segments // 10)
    max_stall_attempts = 16
    obs = env.reset()
    segments_since_reset = 0
    stall_attempts = 0
    attempts_limit = num_segments * max_attempt_multiplier
    while len(samples) < num_segments and stats["attempts"] < attempts_limit:
        stats["attempts"] += 1
        start_obs = obs.copy()
        adapter = _build_natural_teacher_adapter(
            backend=natural_teacher_backend,
            env=env,
            teacher_mode=teacher_mode,
            neural_teacher=neural_teacher,
            neural_teacher_device=neural_teacher_device,
            neural_teacher_temperature=neural_teacher_temperature,
            neural_teacher_epsilon=neural_teacher_epsilon,
        )
        start_vacancies = adapter.vacancy_positions()
        start_cu = adapter.cu_positions()
        start_vac_set, start_cu_set = _positions_to_type_lookup(start_vacancies, start_cu)
        candidate_positions, depth_map, seeds = adapter.build_candidate_positions(
            horizon_k=horizon_k,
            max_seed_vacancies=max_seed_vacancies,
            max_candidate_sites=max_candidate_sites,
        )
        if not candidate_positions:
            env, obs = restart_env(env)
            segments_since_reset = 0
            stall_attempts = 0
            continue

        global_summary = adapter.global_summary()
        boundary_config = adaptive_boundary_config or AdaptiveBoundaryConfig()
        next_obs, done, path_infos, tau_exp, tau_real, reward_sum, touched_positions, realized_horizon_k, boundary_hit = (
            _rollout_teacher_path(
                adapter=adapter,
                rng=rng,
                max_horizon_k=horizon_k,
                boundary_config=boundary_config,
            )
        )
        if done:
            stats["skipped_terminal"] += 1
            env, obs = restart_env(env)
            segments_since_reset = 0
            stall_attempts = 0
            continue
        if realized_horizon_k <= 0:
            stats["skipped_terminal"] += 1
            env, obs = restart_env(env)
            segments_since_reset = 0
            stall_attempts = 0
            continue
        if boundary_config.mode == "adaptive_key_event":
            if boundary_hit:
                stats["adaptive_boundary_hits"] += 1
            elif realized_horizon_k >= horizon_k:
                stats["adaptive_truncated_at_max"] += 1
        stats["realized_horizon_sum"] += float(realized_horizon_k)
        if boundary_config.mode == "adaptive_key_event" and boundary_config.candidate_horizon_source == "actual":
            candidate_positions = [
                pos for pos in candidate_positions if int(depth_map.get(pos, horizon_k + 1)) <= int(realized_horizon_k)
            ]
            depth_map = {pos: depth for pos, depth in depth_map.items() if int(depth) <= int(realized_horizon_k)}
        if teacher_candidate_augmentation:
            candidate_positions, depth_map, seeds = _augment_candidate_positions_with_teacher_path(
                candidate_positions=candidate_positions,
                depth_map=depth_map,
                seeds=seeds,
                touched_positions=touched_positions,
                box=adapter.box_dims(),
                nn1=adapter.nn1_offsets(),
                horizon_k=realized_horizon_k,
                max_candidate_sites=max_candidate_sites,
                teacher_neighbor_depth=teacher_candidate_neighbor_depth,
            )
        end_vacancies = adapter.vacancy_positions()
        end_cu = adapter.cu_positions()
        end_vac_set, end_cu_set = _positions_to_type_lookup(end_vacancies, end_cu)
        changed_positions = _changed_positions_between(start_vac_set, start_cu_set, end_vac_set, end_cu_set)
        if not changed_positions and not include_noop_segments:
            stats["skipped_noop"] += 1
            env, obs = restart_env(env)
            segments_since_reset = 0
            stall_attempts = 0
            continue
        candidate_set = set(candidate_positions)
        if not changed_positions.issubset(candidate_set):
            stats["skipped_uncovered"] += 1
            stall_attempts += 1
            if stall_attempts >= max_stall_attempts:
                env, obs = restart_env(env)
                segments_since_reset = 0
                stall_attempts = 0
            else:
                obs = next_obs
            continue

        positions, nearest_offsets, reach_depth, is_start_vacancy, current_types, target_types, changed_mask = _build_patch_features(
            candidate_positions=candidate_positions,
            depth_map=depth_map,
            seeds=seeds,
            start_vac_set=start_vac_set,
            start_cu_set=start_cu_set,
            end_vac_set=end_vac_set,
            end_cu_set=end_cu_set,
            max_candidate_sites=max_candidate_sites,
            box=adapter.box_dims(),
            horizon_k=realized_horizon_k,
        )
        is_noop_sample = float(changed_mask.sum()) <= 0.0
        if is_noop_sample and not include_noop_segments:
            stats["skipped_noop"] += 1
            env, obs = restart_env(env)
            segments_since_reset = 0
            stall_attempts = 0
            continue
        teacher_summary = _teacher_path_summary(
            path_infos,
            max_candidate_sites=max_candidate_sites,
            horizon_k=realized_horizon_k,
            include_stepwise_features=include_stepwise_path_summary,
            summary_horizon_k=summary_horizon_k,
        )
        mask = np.zeros((max_candidate_sites,), dtype=np.float32)
        mask[: len(candidate_positions)] = 1.0
        teacher_touched_mask = _candidate_mask_from_position_set(
            candidate_positions=positions,
            candidate_mask=mask,
            positions=touched_positions,
        )
        action_source_positions, action_destination_positions = _teacher_action_endpoint_sets(path_infos)
        teacher_action_source_mask = _candidate_mask_from_position_set(
            candidate_positions=positions,
            candidate_mask=mask,
            positions=action_source_positions,
        )
        teacher_action_destination_mask = _candidate_mask_from_position_set(
            candidate_positions=positions,
            candidate_mask=mask,
            positions=action_destination_positions,
        )
        (
            teacher_action_edge_pair_indices,
            teacher_action_edge_pair_mask,
            teacher_action_edge_pair_support_mask,
            teacher_action_edge_pair_moving_type,
            teacher_action_edge_pair_order,
            edge_pair_count,
            edge_pair_covered,
        ) = (
            _teacher_action_edge_pair_targets(
                candidate_positions=positions,
                candidate_mask=mask,
                changed_mask=changed_mask,
                path_infos=path_infos,
            )
        )
        (
            teacher_action_sequence_indices,
            teacher_action_sequence_mask,
            teacher_action_sequence_moving_type,
            teacher_action_sequence_order,
            teacher_action_rollout_changed_mask,
            action_sequence_step_count,
            action_sequence_covered,
        ) = _teacher_action_sequence_targets(
            candidate_positions=positions,
            candidate_mask=mask,
            current_types=current_types,
            path_infos=path_infos,
        )
        (
            teacher_vacancy_pair_indices,
            teacher_vacancy_pair_mask,
            teacher_vacancy_pair_moving_type,
            teacher_vacancy_pair_order,
            vacancy_pair_count,
            vacancy_pair_covered,
        ) = _teacher_vacancy_displacement_pair_targets_from_sequence(
            candidate_mask=mask,
            current_types=current_types,
            target_types=target_types,
            action_sequence_indices=teacher_action_sequence_indices,
            action_sequence_mask=teacher_action_sequence_mask,
            action_sequence_moving_type=teacher_action_sequence_moving_type,
            action_sequence_order=teacher_action_sequence_order,
        )
        rollout_hits = float(np.logical_and(teacher_action_rollout_changed_mask > 0.5, changed_mask > 0.5).sum())
        rollout_precision = rollout_hits / max(float(teacher_action_rollout_changed_mask.sum()), 1.0)
        rollout_recall = rollout_hits / max(float(changed_mask.sum()), 1.0)
        rollout_f1 = 2.0 * rollout_precision * rollout_recall / max(rollout_precision + rollout_recall, 1e-12)
        stats["teacher_touched_count_sum"] += float(len(touched_positions))
        stats["teacher_touched_mask_count_sum"] += float(teacher_touched_mask.sum())
        stats["teacher_touched_recall_sum"] += float(
            teacher_touched_mask.sum() / max(len(touched_positions), 1)
        )
        stats["teacher_action_source_mask_count_sum"] += float(teacher_action_source_mask.sum())
        stats["teacher_action_destination_mask_count_sum"] += float(teacher_action_destination_mask.sum())
        stats["teacher_action_source_recall_sum"] += float(
            teacher_action_source_mask.sum() / max(len(action_source_positions), 1)
        )
        stats["teacher_action_destination_recall_sum"] += float(
            teacher_action_destination_mask.sum() / max(len(action_destination_positions), 1)
        )
        stats["teacher_action_edge_pair_count_sum"] += float(edge_pair_count)
        stats["teacher_action_edge_pair_covered_sum"] += float(edge_pair_covered)
        stats["teacher_action_edge_pair_unique_count_sum"] += float(teacher_action_edge_pair_mask.sum())
        stats["teacher_action_edge_pair_support_count_sum"] += float(teacher_action_edge_pair_support_mask.sum())
        stats["teacher_vacancy_pair_count_sum"] += float(vacancy_pair_count)
        stats["teacher_vacancy_pair_covered_sum"] += float(vacancy_pair_covered)
        stats["teacher_vacancy_pair_unique_count_sum"] += float(teacher_vacancy_pair_mask.sum())
        stats["teacher_action_sequence_step_count_sum"] += float(action_sequence_step_count)
        stats["teacher_action_sequence_covered_sum"] += float(action_sequence_covered)
        stats["teacher_action_rollout_changed_count_sum"] += float(teacher_action_rollout_changed_mask.sum())
        stats["teacher_action_rollout_changed_f1_sum"] += float(rollout_f1)
        stats["candidate_size_sum"] += float(mask.sum())
        if is_noop_sample:
            stats["included_noop"] += 1
        samples.append(
            MacroSegmentSample(
                start_obs=start_obs,
                next_obs=next_obs.copy(),
                start_vacancy_positions=start_vacancies.copy(),
                start_cu_positions=start_cu.copy(),
                global_summary=global_summary,
                teacher_path_summary=teacher_summary,
                candidate_positions=positions,
                nearest_vacancy_offset=nearest_offsets,
                reach_depth=reach_depth,
                is_start_vacancy=is_start_vacancy,
                current_types=current_types,
                target_types=target_types,
                candidate_mask=mask,
                changed_mask=changed_mask,
                tau_exp=float(tau_exp),
                tau_real=float(tau_real),
                reward_sum=float(reward_sum),
                horizon_k=int(realized_horizon_k),
                box_dims=adapter.box_dims().astype(np.float32),
                teacher_touched_mask=teacher_touched_mask,
                teacher_action_source_mask=teacher_action_source_mask,
                teacher_action_destination_mask=teacher_action_destination_mask,
                teacher_action_edge_pair_indices=teacher_action_edge_pair_indices,
                teacher_action_edge_pair_mask=teacher_action_edge_pair_mask,
                teacher_action_edge_pair_support_mask=teacher_action_edge_pair_support_mask,
                teacher_action_edge_pair_moving_type=teacher_action_edge_pair_moving_type,
                teacher_action_edge_pair_order=teacher_action_edge_pair_order,
                teacher_vacancy_pair_indices=teacher_vacancy_pair_indices,
                teacher_vacancy_pair_mask=teacher_vacancy_pair_mask,
                teacher_vacancy_pair_moving_type=teacher_vacancy_pair_moving_type,
                teacher_vacancy_pair_order=teacher_vacancy_pair_order,
                teacher_action_sequence_indices=teacher_action_sequence_indices,
                teacher_action_sequence_mask=teacher_action_sequence_mask,
                teacher_action_sequence_moving_type=teacher_action_sequence_moving_type,
                teacher_action_sequence_order=teacher_action_sequence_order,
                teacher_action_rollout_changed_mask=teacher_action_rollout_changed_mask,
            )
        )
        if len(samples) % progress_every == 0 or len(samples) == num_segments:
            coverage = float(len(samples) / max(stats["attempts"], 1))
            print(
                json.dumps(
                    {
                        "collect_progress": {
                            "samples": len(samples),
                            "target": num_segments,
                            "attempts": stats["attempts"],
                            "coverage": coverage,
                        }
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        stall_attempts = 0
        segments_since_reset += 1
        if is_noop_sample and not keep_after_noop_segments:
            env, obs = restart_env(env)
            segments_since_reset = 0
        elif max_segments_per_rollout > 0 and segments_since_reset >= max_segments_per_rollout:
            env, obs = restart_env(env)
            segments_since_reset = 0
        else:
            obs = next_obs
    denom = max(len(samples), 1)
    stats["coverage"] = float(len(samples) / max(stats["attempts"], 1))
    stats["avg_candidate_size"] = float(stats["candidate_size_sum"] / denom)
    stats["avg_teacher_touched_count"] = float(stats["teacher_touched_count_sum"] / denom)
    stats["avg_teacher_touched_mask_count"] = float(stats["teacher_touched_mask_count_sum"] / denom)
    stats["avg_teacher_touched_recall"] = float(stats["teacher_touched_recall_sum"] / denom)
    stats["avg_teacher_action_source_mask_count"] = float(stats["teacher_action_source_mask_count_sum"] / denom)
    stats["avg_teacher_action_destination_mask_count"] = float(stats["teacher_action_destination_mask_count_sum"] / denom)
    stats["avg_teacher_action_source_recall"] = float(stats["teacher_action_source_recall_sum"] / denom)
    stats["avg_teacher_action_destination_recall"] = float(stats["teacher_action_destination_recall_sum"] / denom)
    stats["avg_teacher_action_edge_pair_count"] = float(stats["teacher_action_edge_pair_count_sum"] / denom)
    stats["avg_teacher_action_edge_pair_covered"] = float(stats["teacher_action_edge_pair_covered_sum"] / denom)
    stats["avg_teacher_action_edge_pair_unique_count"] = float(
        stats["teacher_action_edge_pair_unique_count_sum"] / denom
    )
    stats["avg_teacher_action_edge_pair_support_count"] = float(
        stats["teacher_action_edge_pair_support_count_sum"] / denom
    )
    stats["avg_teacher_vacancy_pair_count"] = float(stats["teacher_vacancy_pair_count_sum"] / denom)
    stats["avg_teacher_vacancy_pair_covered"] = float(stats["teacher_vacancy_pair_covered_sum"] / denom)
    stats["avg_teacher_vacancy_pair_unique_count"] = float(
        stats["teacher_vacancy_pair_unique_count_sum"] / denom
    )
    stats["avg_teacher_action_sequence_step_count"] = float(
        stats["teacher_action_sequence_step_count_sum"] / denom
    )
    stats["avg_teacher_action_sequence_covered"] = float(
        stats["teacher_action_sequence_covered_sum"] / denom
    )
    stats["avg_teacher_action_rollout_changed_count"] = float(
        stats["teacher_action_rollout_changed_count_sum"] / denom
    )
    stats["avg_teacher_action_rollout_changed_f1"] = float(
        stats["teacher_action_rollout_changed_f1_sum"] / denom
    )
    stats["avg_teacher_action_edge_pair_recall"] = float(
        stats["teacher_action_edge_pair_covered_sum"] / max(stats["teacher_action_edge_pair_count_sum"], 1.0)
    )
    stats["avg_teacher_vacancy_pair_recall"] = float(
        stats["teacher_vacancy_pair_covered_sum"] / max(stats["teacher_vacancy_pair_count_sum"], 1.0)
    )
    stats["avg_teacher_action_sequence_recall"] = float(
        stats["teacher_action_sequence_covered_sum"] / max(stats["teacher_action_sequence_step_count_sum"], 1.0)
    )
    stats["avg_realized_horizon"] = float(stats["realized_horizon_sum"] / denom)
    return samples, stats


def _aggregate_collection_stats(samples_by_k: dict[int, list[MacroSegmentSample]], stats_by_k: dict[int, dict[str, float]]) -> dict[str, object]:
    total_samples = sum(len(samples) for samples in samples_by_k.values())
    aggregate: dict[str, object] = {"by_k": {}}
    numeric_keys = [
        "attempts",
        "skipped_uncovered",
        "skipped_terminal",
        "skipped_noop",
        "included_noop",
        "candidate_size_sum",
        "teacher_touched_count_sum",
        "teacher_touched_mask_count_sum",
        "teacher_touched_recall_sum",
        "teacher_action_source_mask_count_sum",
        "teacher_action_destination_mask_count_sum",
        "teacher_action_source_recall_sum",
        "teacher_action_destination_recall_sum",
        "teacher_action_edge_pair_count_sum",
        "teacher_action_edge_pair_covered_sum",
        "teacher_action_edge_pair_unique_count_sum",
        "teacher_action_edge_pair_support_count_sum",
        "teacher_vacancy_pair_count_sum",
        "teacher_vacancy_pair_covered_sum",
        "teacher_vacancy_pair_unique_count_sum",
        "teacher_action_sequence_step_count_sum",
        "teacher_action_sequence_covered_sum",
        "teacher_action_rollout_changed_count_sum",
        "teacher_action_rollout_changed_f1_sum",
        "adaptive_boundary_hits",
        "adaptive_truncated_at_max",
        "realized_horizon_sum",
    ]
    for key in numeric_keys:
        aggregate[key] = float(sum(float(stats_by_k[k].get(key, 0.0)) for k in stats_by_k))
    attempts = float(aggregate.get("attempts", 0.0))
    aggregate["samples"] = int(total_samples)
    aggregate["coverage"] = float(total_samples / max(attempts, 1.0))
    aggregate["avg_candidate_size"] = float(
        sum(float(stats_by_k[k].get("candidate_size_sum", 0.0)) for k in stats_by_k) / max(total_samples, 1)
    )
    aggregate["avg_teacher_touched_count"] = float(
        sum(float(stats_by_k[k].get("teacher_touched_count_sum", 0.0)) for k in stats_by_k) / max(total_samples, 1)
    )
    aggregate["avg_teacher_touched_mask_count"] = float(
        sum(float(stats_by_k[k].get("teacher_touched_mask_count_sum", 0.0)) for k in stats_by_k) / max(total_samples, 1)
    )
    aggregate["avg_teacher_touched_recall"] = float(
        sum(float(stats_by_k[k].get("teacher_touched_recall_sum", 0.0)) for k in stats_by_k) / max(total_samples, 1)
    )
    aggregate["avg_teacher_action_source_mask_count"] = float(
        sum(float(stats_by_k[k].get("teacher_action_source_mask_count_sum", 0.0)) for k in stats_by_k)
        / max(total_samples, 1)
    )
    aggregate["avg_teacher_action_destination_mask_count"] = float(
        sum(float(stats_by_k[k].get("teacher_action_destination_mask_count_sum", 0.0)) for k in stats_by_k)
        / max(total_samples, 1)
    )
    aggregate["avg_teacher_action_source_recall"] = float(
        sum(float(stats_by_k[k].get("teacher_action_source_recall_sum", 0.0)) for k in stats_by_k)
        / max(total_samples, 1)
    )
    aggregate["avg_teacher_action_destination_recall"] = float(
        sum(float(stats_by_k[k].get("teacher_action_destination_recall_sum", 0.0)) for k in stats_by_k)
        / max(total_samples, 1)
    )
    aggregate["avg_teacher_action_edge_pair_count"] = float(
        sum(float(stats_by_k[k].get("teacher_action_edge_pair_count_sum", 0.0)) for k in stats_by_k)
        / max(total_samples, 1)
    )
    aggregate["avg_teacher_action_edge_pair_covered"] = float(
        sum(float(stats_by_k[k].get("teacher_action_edge_pair_covered_sum", 0.0)) for k in stats_by_k)
        / max(total_samples, 1)
    )
    aggregate["avg_teacher_action_edge_pair_unique_count"] = float(
        sum(float(stats_by_k[k].get("teacher_action_edge_pair_unique_count_sum", 0.0)) for k in stats_by_k)
        / max(total_samples, 1)
    )
    aggregate["avg_teacher_action_edge_pair_support_count"] = float(
        sum(float(stats_by_k[k].get("teacher_action_edge_pair_support_count_sum", 0.0)) for k in stats_by_k)
        / max(total_samples, 1)
    )
    aggregate["avg_teacher_vacancy_pair_count"] = float(
        sum(float(stats_by_k[k].get("teacher_vacancy_pair_count_sum", 0.0)) for k in stats_by_k)
        / max(total_samples, 1)
    )
    aggregate["avg_teacher_vacancy_pair_covered"] = float(
        sum(float(stats_by_k[k].get("teacher_vacancy_pair_covered_sum", 0.0)) for k in stats_by_k)
        / max(total_samples, 1)
    )
    aggregate["avg_teacher_vacancy_pair_unique_count"] = float(
        sum(float(stats_by_k[k].get("teacher_vacancy_pair_unique_count_sum", 0.0)) for k in stats_by_k)
        / max(total_samples, 1)
    )
    aggregate["avg_teacher_action_sequence_step_count"] = float(
        sum(float(stats_by_k[k].get("teacher_action_sequence_step_count_sum", 0.0)) for k in stats_by_k)
        / max(total_samples, 1)
    )
    aggregate["avg_teacher_action_sequence_covered"] = float(
        sum(float(stats_by_k[k].get("teacher_action_sequence_covered_sum", 0.0)) for k in stats_by_k)
        / max(total_samples, 1)
    )
    aggregate["avg_teacher_action_rollout_changed_count"] = float(
        sum(float(stats_by_k[k].get("teacher_action_rollout_changed_count_sum", 0.0)) for k in stats_by_k)
        / max(total_samples, 1)
    )
    aggregate["avg_teacher_action_rollout_changed_f1"] = float(
        sum(float(stats_by_k[k].get("teacher_action_rollout_changed_f1_sum", 0.0)) for k in stats_by_k)
        / max(total_samples, 1)
    )
    edge_total = sum(float(stats_by_k[k].get("teacher_action_edge_pair_count_sum", 0.0)) for k in stats_by_k)
    edge_covered = sum(float(stats_by_k[k].get("teacher_action_edge_pair_covered_sum", 0.0)) for k in stats_by_k)
    aggregate["avg_teacher_action_edge_pair_recall"] = float(edge_covered / max(edge_total, 1.0))
    vacancy_pair_total = sum(float(stats_by_k[k].get("teacher_vacancy_pair_count_sum", 0.0)) for k in stats_by_k)
    vacancy_pair_covered = sum(float(stats_by_k[k].get("teacher_vacancy_pair_covered_sum", 0.0)) for k in stats_by_k)
    aggregate["avg_teacher_vacancy_pair_recall"] = float(vacancy_pair_covered / max(vacancy_pair_total, 1.0))
    sequence_total = sum(float(stats_by_k[k].get("teacher_action_sequence_step_count_sum", 0.0)) for k in stats_by_k)
    sequence_covered = sum(float(stats_by_k[k].get("teacher_action_sequence_covered_sum", 0.0)) for k in stats_by_k)
    aggregate["avg_teacher_action_sequence_recall"] = float(sequence_covered / max(sequence_total, 1.0))
    aggregate["avg_realized_horizon"] = float(
        sum(float(stats_by_k[k].get("realized_horizon_sum", 0.0)) for k in stats_by_k) / max(total_samples, 1)
    )
    for k in sorted(stats_by_k):
        per_k = dict(stats_by_k[k])
        per_k["samples"] = int(len(samples_by_k[k]))
        aggregate["by_k"][str(k)] = per_k
    return aggregate


def _collect_segments_for_horizons(
    *,
    env_cfg: dict,
    segment_ks: list[int],
    num_segments_per_k: int,
    max_seed_vacancies: int,
    max_candidate_sites: int,
    seed: int,
    include_stepwise_path_summary: bool,
    summary_horizon_k: int,
    max_segments_per_rollout: int,
    teacher_candidate_neighbor_depth: int,
    teacher_candidate_augmentation: bool,
    teacher_mode: str,
    neural_teacher: Optional[torch.nn.Module],
    neural_teacher_device: str,
    neural_teacher_temperature: float,
    neural_teacher_epsilon: float,
    natural_teacher_backend: str = "kmc",
    adaptive_boundary_config: Optional[AdaptiveBoundaryConfig] = None,
    include_noop_segments: bool = False,
    keep_after_noop_segments: bool = False,
) -> tuple[list[MacroSegmentSample], dict[str, object]]:
    all_samples: list[MacroSegmentSample] = []
    samples_by_k: dict[int, list[MacroSegmentSample]] = {}
    stats_by_k: dict[int, dict[str, float]] = {}
    for offset, horizon_k in enumerate(segment_ks):
        env = MacroKMCEnv(copy.deepcopy(env_cfg))
        rng = np.random.default_rng(seed + 1009 * offset)
        samples, stats = _collect_segments(
            env=env,
            num_segments=num_segments_per_k,
            horizon_k=horizon_k,
            max_seed_vacancies=max_seed_vacancies,
            max_candidate_sites=max_candidate_sites,
            rng=rng,
            include_stepwise_path_summary=include_stepwise_path_summary,
            summary_horizon_k=summary_horizon_k,
            max_segments_per_rollout=max_segments_per_rollout,
            teacher_candidate_neighbor_depth=teacher_candidate_neighbor_depth,
            teacher_candidate_augmentation=teacher_candidate_augmentation,
            teacher_mode=teacher_mode,
            neural_teacher=neural_teacher,
            neural_teacher_device=neural_teacher_device,
            neural_teacher_temperature=neural_teacher_temperature,
            neural_teacher_epsilon=neural_teacher_epsilon,
            natural_teacher_backend=natural_teacher_backend,
            adaptive_boundary_config=adaptive_boundary_config,
            include_noop_segments=include_noop_segments,
            keep_after_noop_segments=keep_after_noop_segments,
        )
        samples_by_k[horizon_k] = samples
        stats_by_k[horizon_k] = stats
        all_samples.extend(samples)
    return all_samples, _aggregate_collection_stats(samples_by_k, stats_by_k)


def _ckpt_args_dict(raw_args: object) -> dict[str, object]:
    if isinstance(raw_args, dict):
        return dict(raw_args)
    if hasattr(raw_args, "__dict__"):
        return dict(vars(raw_args))
    raise TypeError(f"unsupported checkpoint args type: {type(raw_args)!r}")


def _build_planner_model_from_checkpoint(
    checkpoint_path: str,
    device: str,
) -> tuple[MacroDreamerEditModel, dict[str, object]]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_args = _ckpt_args_dict(ckpt["args"])
    include_stepwise_path_summary = str(ckpt_args.get("teacher_path_summary_mode", "stepwise")) == "stepwise"
    planner_segment_ks = _segment_ks_from_ckpt_args(ckpt_args)
    summary_horizon_k = _summary_horizon_k_from_segment_ks(planner_segment_ks)
    model = MacroDreamerEditModel(
        max_vacancies=int(ckpt_args["max_vacancies"]),
        max_defects=int(ckpt_args["max_defects"]),
        max_shells=int(ckpt_args["max_shells"]),
        stats_dim=int(ckpt_args["stats_dim"]),
        lattice_size=tuple(ckpt_args["lattice_size"]),
        neighbor_order=str(ckpt_args["neighbor_order"]),
        dim_latent=int(ckpt_args["dim_latent"]),
        graph_hidden_size=int(ckpt_args["graph_hidden_size"]),
        patch_hidden_size=int(ckpt_args["patch_hidden_size"]),
        patch_latent_dim=int(ckpt_args["patch_latent_dim"]),
        path_latent_dim=int(ckpt_args["path_latent_dim"]),
        global_summary_dim=16,
        teacher_path_summary_dim=teacher_path_summary_dim(
            summary_horizon_k,
            include_stepwise_features=include_stepwise_path_summary,
        ),
        max_macro_k=max(summary_horizon_k, 16),
    ).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    model.realized_tau_head_loaded = not any(key.startswith("realized_duration_head.") for key in missing)
    if missing:
        print(f"Planner-selected collector: missing checkpoint keys initialized from scratch: {missing}", flush=True)
    if unexpected:
        print(f"Planner-selected collector: unexpected checkpoint keys ignored: {unexpected}", flush=True)
    model.eval()
    return model, ckpt_args


def _build_planner_inference_tensors(
    *,
    env: MacroKMCEnv,
    max_seed_vacancies: int,
    max_candidate_sites: int,
    horizon_k: int,
    device: str,
) -> dict[str, torch.Tensor] | None:
    candidate_positions, depth_map, seeds = _build_candidate_positions(
        env,
        horizon_k,
        max_seed_vacancies=max_seed_vacancies,
        max_candidate_sites=max_candidate_sites,
    )
    if not candidate_positions:
        return None
    start_vacancies = env.env.get_vacancy_array().astype(np.int32)
    start_cu = env.env.get_cu_array().astype(np.int32)
    start_vac_set, start_cu_set = _positions_to_type_lookup(start_vacancies, start_cu)
    positions, nearest_offsets, reach_depth, is_start_vacancy, current_types, _, _ = _build_patch_features(
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
        "global_summary": torch.tensor(_global_summary(env)[None, :], dtype=torch.float32, device=device),
        "candidate_positions": torch.tensor(positions[None, ...], dtype=torch.float32, device=device),
        "nearest_vacancy_offset": torch.tensor(nearest_offsets[None, ...], dtype=torch.float32, device=device),
        "reach_depth": torch.tensor(reach_depth[None, ...], dtype=torch.float32, device=device),
        "is_start_vacancy": torch.tensor(is_start_vacancy[None, ...], dtype=torch.float32, device=device),
        "current_types": torch.tensor(current_types[None, ...], dtype=torch.long, device=device),
        "candidate_mask": torch.tensor(candidate_mask[None, ...], dtype=torch.float32, device=device),
        "box_dims": torch.tensor(np.asarray(env.env.dims, dtype=np.float32)[None, :], dtype=torch.float32, device=device),
        "horizon_k": torch.tensor([horizon_k], dtype=torch.long, device=device),
    }


def _compute_planner_selection_score(
    *,
    pred_reward_sum: float,
    reward_scale: float,
    model_expected_tau: float,
    baseline_expected_tau: float,
    horizon_k: int,
    planner_tau_source: str,
    planner_score_mode: str,
    planner_tau_residual_penalty: float,
    planner_k_penalty_power: float,
    planner_tau_blend_alpha: float = 1.0,
    planner_noop_risk_penalty: float = 0.0,
    noop_risk_prob: float = 0.0,
) -> tuple[float, float]:
    tau_for_score = _duration_from_source(
        model_expected_tau=model_expected_tau,
        baseline_expected_tau=baseline_expected_tau,
        source=planner_tau_source,
        blend_alpha=planner_tau_blend_alpha,
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


def _duration_from_source(
    *,
    model_expected_tau: float,
    baseline_expected_tau: float,
    source: str,
    blend_alpha: float = 1.0,
) -> float:
    model_tau = max(float(model_expected_tau), 1e-12)
    baseline_tau = max(float(baseline_expected_tau), 1e-12)
    if source == "model":
        return model_tau
    if source == "baseline":
        return baseline_tau
    if source == "blend":
        alpha = float(np.clip(blend_alpha, 0.0, 1.0))
        return float(np.exp((1.0 - alpha) * np.log(baseline_tau) + alpha * np.log(model_tau)))
    raise ValueError(f"Unknown duration source: {source}")


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
            return combine_action_endpoint_logits(action_source_logits, action_destination_logits)
        return action_support_logits if action_support_logits is not None else proposal_logits
    if source == "blend":
        alpha = float(np.clip(blend_alpha, 0.0, 1.0))
        return (1.0 - alpha) * change_logits + alpha * proposal_logits
    return change_logits


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


def _choose_planner_candidate(
    candidates: list[dict[str, object]],
    *,
    min_projected_changed_sites: int,
) -> dict[str, object] | None:
    legal = [
        item
        for item in candidates
        if float(item.get("reachability_violation", 1.0)) <= 0.0
        and float(item.get("projected_changed_count", 0.0)) >= float(min_projected_changed_sites)
    ]
    if not legal:
        return None
    return max(legal, key=lambda item: float(item.get("selection_score", -float("inf"))))


def _candidate_position_set(candidate: dict[str, object]) -> set[tuple[int, int, int]]:
    raw_positions = candidate.get("projected_changed_positions", [])
    positions: set[tuple[int, int, int]] = set()
    if not isinstance(raw_positions, list):
        return positions
    for item in raw_positions:
        try:
            coords = tuple(int(x) for x in item)
        except (TypeError, ValueError):
            continue
        if len(coords) == 3:
            positions.add(coords)
    return positions


def _site_overlap_metrics(
    projected_positions: set[tuple[int, int, int]],
    teacher_positions: set[tuple[int, int, int]],
) -> dict[str, float]:
    overlap = projected_positions & teacher_positions
    precision = float(len(overlap) / max(len(projected_positions), 1))
    recall = float(len(overlap) / max(len(teacher_positions), 1))
    f1 = float(2.0 * precision * recall / max(precision + recall, 1e-12))
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "overlap_count": float(len(overlap)),
        "teacher_changed_count": float(len(teacher_positions)),
        "projected_changed_count": float(len(projected_positions)),
    }


def _apply_teacher_overlap_rerank(
    *,
    env: MacroKMCEnv,
    candidates: list[dict[str, object]],
    start_vac_set: set[tuple[int, int, int]],
    start_cu_set: set[tuple[int, int, int]],
    rng: np.random.Generator,
    boundary_config: AdaptiveBoundaryConfig,
    weight: float,
    teacher_mode: str,
    neural_teacher: Optional[torch.nn.Module],
    neural_teacher_device: str,
    neural_teacher_temperature: float,
    neural_teacher_epsilon: float,
    natural_teacher_backend: str,
) -> dict[str, float]:
    stats = {
        "teacher_overlap_probe_count": 0.0,
        "teacher_overlap_f1_sum": 0.0,
        "teacher_overlap_precision_sum": 0.0,
        "teacher_overlap_recall_sum": 0.0,
        "teacher_overlap_probe_failures": 0.0,
    }
    if not candidates or float(weight) == 0.0:
        return stats
    rng_state = copy.deepcopy(rng.bit_generator.state)
    for candidate in candidates:
        projected_positions = _candidate_position_set(candidate)
        candidate["base_selection_score"] = float(candidate.get("selection_score", -float("inf")))
        try:
            probe_env = copy.deepcopy(env)
            probe_rng = np.random.default_rng()
            probe_rng.bit_generator.state = copy.deepcopy(rng_state)
            probe_adapter = _build_natural_teacher_adapter(
                backend=natural_teacher_backend,
                env=probe_env,
                teacher_mode=teacher_mode,
                neural_teacher=neural_teacher,
                neural_teacher_device=neural_teacher_device,
                neural_teacher_temperature=neural_teacher_temperature,
                neural_teacher_epsilon=neural_teacher_epsilon,
            )
            _, done, _, _, _, _, _, realized_horizon_k, _ = _rollout_teacher_path(
                adapter=probe_adapter,
                rng=probe_rng,
                max_horizon_k=int(candidate["segment_k"]),
                boundary_config=boundary_config,
            )
            if done or int(realized_horizon_k) <= 0:
                stats["teacher_overlap_probe_failures"] += 1.0
                continue
            end_vac_set, end_cu_set = _positions_to_type_lookup(
                probe_adapter.vacancy_positions(),
                probe_adapter.cu_positions(),
            )
            teacher_positions = _changed_positions_between(start_vac_set, start_cu_set, end_vac_set, end_cu_set)
            overlap = _site_overlap_metrics(projected_positions, teacher_positions)
            candidate["teacher_changed_positions_probe"] = [
                [int(pos[0]), int(pos[1]), int(pos[2])] for pos in sorted(teacher_positions)
            ]
            false_positive_positions = projected_positions - teacher_positions
            candidate["projected_false_positive_positions_probe"] = [
                [int(pos[0]), int(pos[1]), int(pos[2])] for pos in sorted(false_positive_positions)
            ]
            candidate["teacher_overlap_precision"] = float(overlap["precision"])
            candidate["teacher_overlap_recall"] = float(overlap["recall"])
            candidate["teacher_overlap_f1"] = float(overlap["f1"])
            candidate["teacher_overlap_count"] = float(overlap["overlap_count"])
            candidate["teacher_changed_count_probe"] = float(overlap["teacher_changed_count"])
            candidate["selection_score"] = float(candidate["base_selection_score"]) + float(weight) * float(overlap["f1"])
            stats["teacher_overlap_probe_count"] += 1.0
            stats["teacher_overlap_f1_sum"] += float(overlap["f1"])
            stats["teacher_overlap_precision_sum"] += float(overlap["precision"])
            stats["teacher_overlap_recall_sum"] += float(overlap["recall"])
        except Exception:
            stats["teacher_overlap_probe_failures"] += 1.0
            candidate["selection_score"] = float(candidate.get("base_selection_score", candidate.get("selection_score", -float("inf"))))
            continue
    rng.bit_generator.state = copy.deepcopy(rng_state)
    return stats


@torch.no_grad()
def _predict_planner_candidate_for_horizon(
    *,
    model: MacroDreamerEditModel,
    reward_model: MacroDreamerEditModel | None = None,
    duration_model: MacroDreamerEditModel | None = None,
    env: MacroKMCEnv,
    horizon_k: int,
    max_seed_vacancies: int,
    max_candidate_sites: int,
    reward_scale: float,
    device: str,
    duration_source: str,
    planner_tau_source: str,
    planner_score_mode: str,
    planner_tau_residual_penalty: float,
    planner_k_penalty_power: float,
    planner_noop_risk_penalty: float = 0.0,
    reward_prediction_source: str,
    reward_edit_context_source: str = "default",
    planner_duration_checkpoint_source: str = "duration",
    aux_projected_types_source: str = "aux",
    planner_projection_change_source: str = "change",
    planner_projection_change_blend_alpha: float = 0.5,
    planner_projection_topk_source: str = "none",
    planner_projection_topk_budget: int = 0,
    planner_proposal_score_weight: float = 0.0,
    planner_candidate_quality_score_weight: float = 0.0,
    planner_teacher_overlap_rerank_weight: float = 0.0,
    duration_blend_alpha: float = 1.0,
    planner_tau_blend_alpha: float = 1.0,
) -> dict[str, object] | None:
    tensors = _build_planner_inference_tensors(
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
    projection_change_logits = _projection_logits_from_source(
        change_logits=change_logits,
        proposal_logits=proposal_logits,
        action_support_logits=action_support_logits,
        action_source_logits=action_source_logits,
        action_destination_logits=action_destination_logits,
        source=planner_projection_change_source,
        blend_alpha=float(planner_projection_change_blend_alpha),
    )
    projection_type_logits = raw_type_logits
    projection_topk_mask = torch.zeros_like(tensors["candidate_mask"], dtype=torch.bool)
    if planner_projection_topk_source != "none" and int(planner_projection_topk_budget) > 0:
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
            projection_type_logits=raw_type_logits,
            ranking_logits=ranking_logits,
            candidate_mask=tensors["candidate_mask"],
            current_types=tensors["current_types"],
            topk_budget=int(planner_projection_topk_budget),
        )
    projected_types, projected_changed_mask, transport_cost, reachability_violation = project_types_by_inventory(
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
        reward_path_latent = reward_model.sample_path_latent(reward_prior_mu, reward_prior_logvar, deterministic=True)
        reward_next_pred = reward_model.predict_next_global(reward_global_latent, reward_path_latent, tensors["horizon_k"])
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
        if reward_model is not None and reward_model is not model and aux_projected_types_source == "aux":
            reward_projected_types, _, _, _ = project_types_by_inventory(
                current_types=tensors["current_types"],
                change_logits=reward_change_logits,
                type_logits=reward_type_logits,
                node_mask=tensors["candidate_mask"],
                positions=tensors["candidate_positions"],
                box_dims=tensors["box_dims"],
                horizon_k=tensors["horizon_k"],
                max_changed_sites=2 * tensors["horizon_k"],
            )
        _, reward_patch_latent = reward_head_model.encode_patch(
            positions=tensors["candidate_positions"],
            nearest_vacancy_offset=tensors["nearest_vacancy_offset"],
            reach_depth=tensors["reach_depth"],
            is_start_vacancy=tensors["is_start_vacancy"],
            type_ids=reward_projected_types,
            node_mask=tensors["candidate_mask"],
            global_summary=tensors["global_summary"],
            box_dims=tensors["box_dims"],
        )
        reward_change_logits, reward_type_logits = projected_edit_logits_from_types(
            current_types=tensors["current_types"],
            projected_types=reward_projected_types,
            candidate_mask=tensors["candidate_mask"],
        )
    reward_change_logits, reward_type_logits = _select_reward_edit_context(
        reward_edit_context_source,
        reward_change_logits,
        reward_type_logits,
    )
    primary_outputs = _predict_reward_and_duration_outputs(
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
    if duration_model is not None and duration_model is not reward_head_model:
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
                duration_projected_types, _, _, _ = project_types_by_inventory(
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
            duration_change_logits_for_head, duration_type_logits_for_head = projected_edit_logits_from_types(
                current_types=tensors["current_types"],
                projected_types=duration_projected_types,
                candidate_mask=tensors["candidate_mask"],
            )
        duration_change_logits_for_head, duration_type_logits_for_head = _select_reward_edit_context(
            reward_edit_context_source,
            duration_change_logits_for_head,
            duration_type_logits_for_head,
        )
        duration_outputs = _predict_reward_and_duration_outputs(
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
    noop_risk_prob = float(
        torch.sigmoid(primary_outputs.get("noop_risk_logit", torch.zeros_like(primary_outputs["reward"]))).item()
    )
    pred_reward = float((primary_outputs["reward"] * torch.sigmoid(primary_outputs["gate_logit"])).item())
    primary_expected_tau = float(torch.exp(primary_outputs["expected_tau_mu"]).item())
    model_expected_tau = float(torch.exp(duration_outputs["expected_tau_mu"]).item())
    model_realized_tau = float(torch.exp(duration_outputs["realized_tau_mu"]).item())
    baseline_log_tau = macro_duration_baseline_log_tau(tensors["global_summary"], tensors["horizon_k"])
    baseline_expected_tau = float(torch.exp(baseline_log_tau).item())
    pred_expected_tau = _duration_from_source(
        model_expected_tau=model_expected_tau,
        baseline_expected_tau=baseline_expected_tau,
        source=duration_source,
        blend_alpha=duration_blend_alpha,
    )
    pred_realized_tau = model_realized_tau
    if duration_source in {"baseline", "blend"}:
        pred_realized_tau = pred_expected_tau
    score_model_expected_tau = (
        primary_expected_tau
        if duration_model is not None
        and duration_model is not model
        and planner_duration_checkpoint_source == "primary"
        else model_expected_tau
    )
    selection_score, tau_for_score = _compute_planner_selection_score(
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
        planner_noop_risk_penalty=planner_noop_risk_penalty,
        noop_risk_prob=noop_risk_prob,
    )
    violation = float(reachability_violation.item())
    proposal_prob = torch.sigmoid(proposal_logits) * tensors["candidate_mask"]
    proposal_support_mass = float(proposal_prob.sum().item())
    if violation > 0.0:
        selection_score = -float("inf")
    else:
        if float(planner_proposal_score_weight) != 0.0:
            selection_score += float(planner_proposal_score_weight) * float(np.log1p(max(proposal_support_mass, 0.0)))
        if float(planner_candidate_quality_score_weight) != 0.0:
            selection_score += float(planner_candidate_quality_score_weight) * candidate_quality_score
    projected_mask_np = projected_changed_mask.squeeze(0).detach().cpu().numpy() > 0.5
    candidate_mask_np = tensors["candidate_mask"].squeeze(0).detach().cpu().numpy() > 0.5
    positions_np = np.rint(tensors["candidate_positions"].squeeze(0).detach().cpu().numpy()).astype(np.int32)
    projected_positions = [
        [int(pos[0]), int(pos[1]), int(pos[2])]
        for pos, changed, valid in zip(positions_np, projected_mask_np, candidate_mask_np)
        if bool(changed) and bool(valid)
    ]
    return {
        "segment_k": int(horizon_k),
        "predicted_reward_sum": pred_reward,
        "predicted_delta_e": float(pred_reward / max(float(reward_scale), 1e-12)),
        "predicted_reward_raw": reward_raw,
        "predicted_reward_gate_prob": reward_gate_prob,
        "predicted_noop_risk_prob": noop_risk_prob,
        "predicted_expected_tau": float(pred_expected_tau),
        "predicted_realized_tau": float(pred_realized_tau),
        "model_expected_tau": float(model_expected_tau),
        "primary_model_expected_tau": float(primary_expected_tau),
        "score_model_expected_tau": float(score_model_expected_tau),
        "model_realized_tau": float(model_realized_tau),
        "baseline_expected_tau": float(baseline_expected_tau),
        "planner_tau_for_score": float(tau_for_score),
        "duration_blend_alpha": float(duration_blend_alpha),
        "planner_tau_blend_alpha": float(planner_tau_blend_alpha),
        "reachability_violation": violation,
        "projected_changed_count": float(projected_changed_mask.sum().item()),
        "candidate_quality_score": candidate_quality_score,
        "proposal_support_mass": proposal_support_mass,
        "planner_projection_change_source": planner_projection_change_source,
        "planner_projection_topk_source": planner_projection_topk_source,
        "planner_projection_topk_budget": int(planner_projection_topk_budget),
        "planner_projection_topk_count": int(projection_topk_mask.sum().item()),
        "transport_cost": float(transport_cost.item()),
        "selection_score": float(selection_score),
        "projected_changed_positions": projected_positions,
    }


def _collect_planner_selected_segments(
    *,
    env: MacroKMCEnv,
    num_segments: int,
    segment_ks: list[int],
    planner_model: MacroDreamerEditModel,
    planner_reward_model: MacroDreamerEditModel | None = None,
    planner_duration_model: MacroDreamerEditModel | None = None,
    planner_device: str,
    max_seed_vacancies: int,
    max_candidate_sites: int,
    rng: np.random.Generator,
    include_stepwise_path_summary: bool,
    summary_horizon_k: int,
    max_segments_per_rollout: int,
    min_projected_changed_sites: int,
    duration_source: str,
    planner_tau_source: str,
    planner_score_mode: str,
    planner_tau_residual_penalty: float,
    planner_k_penalty_power: float,
    planner_noop_risk_penalty: float = 0.0,
    reward_prediction_source: str,
    reward_edit_context_source: str = "default",
    planner_duration_checkpoint_source: str = "duration",
    aux_projected_types_source: str = "aux",
    planner_projection_change_source: str = "change",
    planner_projection_change_blend_alpha: float = 0.5,
    planner_projection_topk_source: str = "none",
    planner_projection_topk_budget: int = 0,
    planner_proposal_score_weight: float = 0.0,
    planner_candidate_quality_score_weight: float = 0.0,
    planner_teacher_overlap_rerank_weight: float = 0.0,
    planner_selected_store_candidate_overlap_masks: bool = False,
    duration_blend_alpha: float = 1.0,
    planner_tau_blend_alpha: float = 1.0,
    allow_uncovered_reward_only: bool = False,
    teacher_candidate_augmentation: bool = True,
    teacher_candidate_neighbor_depth: int = 1,
    teacher_mode: str = "kmc",
    neural_teacher: Optional[torch.nn.Module] = None,
    neural_teacher_device: str = "cpu",
    neural_teacher_temperature: float = 1.0,
    neural_teacher_epsilon: float = 0.0,
    natural_teacher_backend: str = "kmc",
    adaptive_boundary_config: Optional[AdaptiveBoundaryConfig] = None,
    max_attempt_multiplier: int = 30,
    include_noop_segments: bool = False,
    keep_after_noop_segments: bool = False,
) -> tuple[list[MacroSegmentSample], dict[str, object]]:
    def restart_env(current_env: MacroKMCEnv) -> tuple[MacroKMCEnv, np.ndarray]:
        new_env = MacroKMCEnv(copy.deepcopy(current_env.cfg))
        return new_env, new_env.reset()

    samples: list[MacroSegmentSample] = []
    samples_by_k: dict[int, int] = {int(k): 0 for k in segment_ks}
    stats: dict[str, object] = {
        "attempts": 0,
        "skipped_no_planner_candidate": 0,
        "skipped_uncovered": 0,
        "reward_only_uncovered": 0,
        "skipped_terminal": 0,
        "skipped_noop": 0,
        "included_noop": 0,
        "candidate_size_sum": 0.0,
        "teacher_touched_count_sum": 0.0,
        "teacher_touched_mask_count_sum": 0.0,
        "teacher_touched_recall_sum": 0.0,
        "teacher_action_source_mask_count_sum": 0.0,
        "teacher_action_destination_mask_count_sum": 0.0,
        "teacher_action_source_recall_sum": 0.0,
        "teacher_action_destination_recall_sum": 0.0,
        "teacher_action_edge_pair_count_sum": 0.0,
        "teacher_action_edge_pair_covered_sum": 0.0,
        "teacher_action_edge_pair_unique_count_sum": 0.0,
        "teacher_action_edge_pair_support_count_sum": 0.0,
        "teacher_vacancy_pair_count_sum": 0.0,
        "teacher_vacancy_pair_covered_sum": 0.0,
        "teacher_vacancy_pair_unique_count_sum": 0.0,
        "teacher_action_sequence_step_count_sum": 0.0,
        "teacher_action_sequence_covered_sum": 0.0,
        "teacher_action_rollout_changed_count_sum": 0.0,
        "teacher_action_rollout_changed_f1_sum": 0.0,
        "planner_projected_changed_count_sum": 0.0,
        "planner_candidate_teacher_changed_count_sum": 0.0,
        "planner_candidate_false_positive_count_sum": 0.0,
        "planner_candidate_quality_target_sum": 0.0,
        "planner_candidate_quality_available_sum": 0.0,
        "planner_candidate_quality_score_sum": 0.0,
        "planner_score_sum": 0.0,
        "teacher_overlap_probe_count": 0.0,
        "teacher_overlap_probe_failures": 0.0,
        "teacher_overlap_f1_sum": 0.0,
        "teacher_overlap_precision_sum": 0.0,
        "teacher_overlap_recall_sum": 0.0,
        "teacher_overlap_selected_f1_sum": 0.0,
        "adaptive_boundary_hits": 0,
        "adaptive_truncated_at_max": 0,
        "realized_horizon_sum": 0.0,
    }
    selected_attempts_by_k: dict[int, int] = {int(k): 0 for k in segment_ks}
    skipped_uncovered_by_k: dict[int, int] = {int(k): 0 for k in segment_ks}
    reward_only_uncovered_by_k: dict[int, int] = {int(k): 0 for k in segment_ks}
    progress_every = max(20, min(100, num_segments // 10))
    obs = env.reset()
    segments_since_reset = 0
    stall_attempts = 0
    max_stall_attempts = 16
    attempts_limit = num_segments * max_attempt_multiplier
    while len(samples) < num_segments and int(stats["attempts"]) < attempts_limit:
        stats["attempts"] = int(stats["attempts"]) + 1
        start_obs = obs.copy()
        adapter = _build_natural_teacher_adapter(
            backend=natural_teacher_backend,
            env=env,
            teacher_mode=teacher_mode,
            neural_teacher=neural_teacher,
            neural_teacher_device=neural_teacher_device,
            neural_teacher_temperature=neural_teacher_temperature,
            neural_teacher_epsilon=neural_teacher_epsilon,
        )
        start_vacancies = adapter.vacancy_positions()
        start_cu = adapter.cu_positions()
        start_vac_set, start_cu_set = _positions_to_type_lookup(start_vacancies, start_cu)
        global_summary = adapter.global_summary()
        candidates: list[dict[str, object]] = []
        for horizon_k in segment_ks:
            candidate = _predict_planner_candidate_for_horizon(
                model=planner_model,
                reward_model=planner_reward_model,
                duration_model=planner_duration_model,
                env=env,
                horizon_k=int(horizon_k),
                max_seed_vacancies=max_seed_vacancies,
                max_candidate_sites=max_candidate_sites,
                reward_scale=float(env.cfg.get("reward_scale", 1.0)),
                device=planner_device,
                duration_source=duration_source,
                planner_tau_source=planner_tau_source,
                planner_score_mode=planner_score_mode,
                planner_tau_residual_penalty=planner_tau_residual_penalty,
                planner_k_penalty_power=planner_k_penalty_power,
                planner_noop_risk_penalty=planner_noop_risk_penalty,
                reward_prediction_source=reward_prediction_source,
                reward_edit_context_source=reward_edit_context_source,
                planner_duration_checkpoint_source=planner_duration_checkpoint_source,
                aux_projected_types_source=aux_projected_types_source,
                planner_projection_change_source=planner_projection_change_source,
                planner_projection_change_blend_alpha=planner_projection_change_blend_alpha,
                planner_projection_topk_source=planner_projection_topk_source,
                planner_projection_topk_budget=planner_projection_topk_budget,
                planner_proposal_score_weight=planner_proposal_score_weight,
                planner_candidate_quality_score_weight=planner_candidate_quality_score_weight,
                duration_blend_alpha=duration_blend_alpha,
                planner_tau_blend_alpha=planner_tau_blend_alpha,
            )
            if candidate is not None:
                candidates.append(candidate)
        boundary_config = adaptive_boundary_config or AdaptiveBoundaryConfig()
        if float(planner_teacher_overlap_rerank_weight) != 0.0:
            overlap_stats = _apply_teacher_overlap_rerank(
                env=env,
                candidates=candidates,
                start_vac_set=start_vac_set,
                start_cu_set=start_cu_set,
                rng=rng,
                boundary_config=boundary_config,
                weight=float(planner_teacher_overlap_rerank_weight),
                teacher_mode=teacher_mode,
                neural_teacher=neural_teacher,
                neural_teacher_device=neural_teacher_device,
                neural_teacher_temperature=neural_teacher_temperature,
                neural_teacher_epsilon=neural_teacher_epsilon,
                natural_teacher_backend=natural_teacher_backend,
            )
            for key, value in overlap_stats.items():
                stats[key] = float(stats.get(key, 0.0)) + float(value)
        selected = _choose_planner_candidate(
            candidates,
            min_projected_changed_sites=min_projected_changed_sites,
        )
        if selected is None:
            stats["skipped_no_planner_candidate"] = int(stats["skipped_no_planner_candidate"]) + 1
            env, obs = restart_env(env)
            segments_since_reset = 0
            continue
        horizon_k = int(selected["segment_k"])
        selected_attempts_by_k[horizon_k] = selected_attempts_by_k.get(horizon_k, 0) + 1
        if float(planner_teacher_overlap_rerank_weight) != 0.0:
            stats["teacher_overlap_selected_f1_sum"] = float(stats["teacher_overlap_selected_f1_sum"]) + float(
                selected.get("teacher_overlap_f1", 0.0)
            )
        candidate_positions, depth_map, seeds = adapter.build_candidate_positions(
            horizon_k=horizon_k,
            max_seed_vacancies=max_seed_vacancies,
            max_candidate_sites=max_candidate_sites,
        )
        if not candidate_positions:
            stats["skipped_no_planner_candidate"] = int(stats["skipped_no_planner_candidate"]) + 1
            env, obs = restart_env(env)
            segments_since_reset = 0
            stall_attempts = 0
            continue

        next_obs, done, path_infos, tau_exp, tau_real, reward_sum, touched_positions, realized_horizon_k, boundary_hit = (
            _rollout_teacher_path(
                adapter=adapter,
                rng=rng,
                max_horizon_k=horizon_k,
                boundary_config=boundary_config,
            )
        )
        if done:
            stats["skipped_terminal"] = int(stats["skipped_terminal"]) + 1
            env, obs = restart_env(env)
            segments_since_reset = 0
            stall_attempts = 0
            continue
        if realized_horizon_k <= 0:
            stats["skipped_terminal"] = int(stats["skipped_terminal"]) + 1
            env, obs = restart_env(env)
            segments_since_reset = 0
            stall_attempts = 0
            continue
        if boundary_config.mode == "adaptive_key_event":
            if boundary_hit:
                stats["adaptive_boundary_hits"] = int(stats["adaptive_boundary_hits"]) + 1
            elif realized_horizon_k >= horizon_k:
                stats["adaptive_truncated_at_max"] = int(stats["adaptive_truncated_at_max"]) + 1
        stats["realized_horizon_sum"] = float(stats["realized_horizon_sum"]) + float(realized_horizon_k)

        if boundary_config.mode == "adaptive_key_event" and boundary_config.candidate_horizon_source == "actual":
            candidate_positions = [
                pos for pos in candidate_positions if int(depth_map.get(pos, horizon_k + 1)) <= int(realized_horizon_k)
            ]
            depth_map = {pos: depth for pos, depth in depth_map.items() if int(depth) <= int(realized_horizon_k)}

        if teacher_candidate_augmentation:
            candidate_positions, depth_map, seeds = _augment_candidate_positions_with_teacher_path(
                candidate_positions=candidate_positions,
                depth_map=depth_map,
                seeds=seeds,
                touched_positions=touched_positions,
                box=adapter.box_dims(),
                nn1=adapter.nn1_offsets(),
                horizon_k=realized_horizon_k,
                max_candidate_sites=max_candidate_sites,
                teacher_neighbor_depth=teacher_candidate_neighbor_depth,
            )

        end_vacancies = adapter.vacancy_positions()
        end_cu = adapter.cu_positions()
        end_vac_set, end_cu_set = _positions_to_type_lookup(end_vacancies, end_cu)
        changed_positions = _changed_positions_between(start_vac_set, start_cu_set, end_vac_set, end_cu_set)
        if not changed_positions and not include_noop_segments:
            stats["skipped_noop"] = int(stats["skipped_noop"]) + 1
            env, obs = restart_env(env)
            segments_since_reset = 0
            stall_attempts = 0
            continue
        candidate_set = set(candidate_positions)
        reward_only_uncovered = False
        if not changed_positions.issubset(candidate_set):
            if allow_uncovered_reward_only:
                stats["reward_only_uncovered"] = int(stats["reward_only_uncovered"]) + 1
                reward_only_uncovered_by_k[horizon_k] = reward_only_uncovered_by_k.get(horizon_k, 0) + 1
                reward_only_uncovered = True
            else:
                stats["skipped_uncovered"] = int(stats["skipped_uncovered"]) + 1
                skipped_uncovered_by_k[horizon_k] = skipped_uncovered_by_k.get(horizon_k, 0) + 1
                stall_attempts += 1
                if stall_attempts >= max_stall_attempts:
                    env, obs = restart_env(env)
                    segments_since_reset = 0
                    stall_attempts = 0
                else:
                    obs = next_obs
                continue

        positions, nearest_offsets, reach_depth, is_start_vacancy, current_types, target_types, changed_mask = _build_patch_features(
            candidate_positions=candidate_positions,
            depth_map=depth_map,
            seeds=seeds,
            start_vac_set=start_vac_set,
            start_cu_set=start_cu_set,
            end_vac_set=end_vac_set,
            end_cu_set=end_cu_set,
            max_candidate_sites=max_candidate_sites,
            box=adapter.box_dims(),
            horizon_k=realized_horizon_k,
        )
        is_noop_sample = float(changed_mask.sum()) <= 0.0 and not reward_only_uncovered
        if is_noop_sample and not include_noop_segments:
            stats["skipped_noop"] = int(stats["skipped_noop"]) + 1
            env, obs = restart_env(env)
            segments_since_reset = 0
            stall_attempts = 0
            continue

        teacher_summary = _teacher_path_summary(
            path_infos,
            max_candidate_sites=max_candidate_sites,
            horizon_k=realized_horizon_k,
            include_stepwise_features=include_stepwise_path_summary,
            summary_horizon_k=summary_horizon_k,
        )
        mask = np.zeros((max_candidate_sites,), dtype=np.float32)
        mask[: len(candidate_positions)] = 1.0
        teacher_touched_mask = _candidate_mask_from_position_set(
            candidate_positions=positions,
            candidate_mask=mask,
            positions=touched_positions,
        )
        action_source_positions, action_destination_positions = _teacher_action_endpoint_sets(path_infos)
        teacher_action_source_mask = _candidate_mask_from_position_set(
            candidate_positions=positions,
            candidate_mask=mask,
            positions=action_source_positions,
        )
        teacher_action_destination_mask = _candidate_mask_from_position_set(
            candidate_positions=positions,
            candidate_mask=mask,
            positions=action_destination_positions,
        )
        (
            teacher_action_edge_pair_indices,
            teacher_action_edge_pair_mask,
            teacher_action_edge_pair_support_mask,
            teacher_action_edge_pair_moving_type,
            teacher_action_edge_pair_order,
            edge_pair_count,
            edge_pair_covered,
        ) = (
            _teacher_action_edge_pair_targets(
                candidate_positions=positions,
                candidate_mask=mask,
                changed_mask=changed_mask,
                path_infos=path_infos,
            )
        )
        (
            teacher_action_sequence_indices,
            teacher_action_sequence_mask,
            teacher_action_sequence_moving_type,
            teacher_action_sequence_order,
            teacher_action_rollout_changed_mask,
            action_sequence_step_count,
            action_sequence_covered,
        ) = _teacher_action_sequence_targets(
            candidate_positions=positions,
            candidate_mask=mask,
            current_types=current_types,
            path_infos=path_infos,
        )
        (
            teacher_vacancy_pair_indices,
            teacher_vacancy_pair_mask,
            teacher_vacancy_pair_moving_type,
            teacher_vacancy_pair_order,
            vacancy_pair_count,
            vacancy_pair_covered,
        ) = _teacher_vacancy_displacement_pair_targets_from_sequence(
            candidate_mask=mask,
            current_types=current_types,
            target_types=target_types,
            action_sequence_indices=teacher_action_sequence_indices,
            action_sequence_mask=teacher_action_sequence_mask,
            action_sequence_moving_type=teacher_action_sequence_moving_type,
            action_sequence_order=teacher_action_sequence_order,
        )
        rollout_hits = float(np.logical_and(teacher_action_rollout_changed_mask > 0.5, changed_mask > 0.5).sum())
        rollout_precision = rollout_hits / max(float(teacher_action_rollout_changed_mask.sum()), 1.0)
        rollout_recall = rollout_hits / max(float(changed_mask.sum()), 1.0)
        rollout_f1 = 2.0 * rollout_precision * rollout_recall / max(rollout_precision + rollout_recall, 1e-12)
        stats["teacher_touched_count_sum"] = float(stats["teacher_touched_count_sum"]) + float(len(touched_positions))
        stats["teacher_touched_mask_count_sum"] = float(stats["teacher_touched_mask_count_sum"]) + float(
            teacher_touched_mask.sum()
        )
        stats["teacher_touched_recall_sum"] = float(stats["teacher_touched_recall_sum"]) + float(
            teacher_touched_mask.sum() / max(len(touched_positions), 1)
        )
        stats["teacher_action_source_mask_count_sum"] = float(
            stats["teacher_action_source_mask_count_sum"]
        ) + float(teacher_action_source_mask.sum())
        stats["teacher_action_destination_mask_count_sum"] = float(
            stats["teacher_action_destination_mask_count_sum"]
        ) + float(teacher_action_destination_mask.sum())
        stats["teacher_action_source_recall_sum"] = float(
            stats["teacher_action_source_recall_sum"]
        ) + float(teacher_action_source_mask.sum() / max(len(action_source_positions), 1))
        stats["teacher_action_destination_recall_sum"] = float(
            stats["teacher_action_destination_recall_sum"]
        ) + float(teacher_action_destination_mask.sum() / max(len(action_destination_positions), 1))
        stats["teacher_action_edge_pair_count_sum"] = float(
            stats["teacher_action_edge_pair_count_sum"]
        ) + float(edge_pair_count)
        stats["teacher_action_edge_pair_covered_sum"] = float(
            stats["teacher_action_edge_pair_covered_sum"]
        ) + float(edge_pair_covered)
        stats["teacher_action_edge_pair_unique_count_sum"] = float(
            stats["teacher_action_edge_pair_unique_count_sum"]
        ) + float(teacher_action_edge_pair_mask.sum())
        stats["teacher_action_edge_pair_support_count_sum"] = float(
            stats["teacher_action_edge_pair_support_count_sum"]
        ) + float(teacher_action_edge_pair_support_mask.sum())
        stats["teacher_vacancy_pair_count_sum"] = float(stats["teacher_vacancy_pair_count_sum"]) + float(
            vacancy_pair_count
        )
        stats["teacher_vacancy_pair_covered_sum"] = float(stats["teacher_vacancy_pair_covered_sum"]) + float(
            vacancy_pair_covered
        )
        stats["teacher_vacancy_pair_unique_count_sum"] = float(
            stats["teacher_vacancy_pair_unique_count_sum"]
        ) + float(teacher_vacancy_pair_mask.sum())
        stats["teacher_action_sequence_step_count_sum"] = float(
            stats["teacher_action_sequence_step_count_sum"]
        ) + float(action_sequence_step_count)
        stats["teacher_action_sequence_covered_sum"] = float(
            stats["teacher_action_sequence_covered_sum"]
        ) + float(action_sequence_covered)
        stats["teacher_action_rollout_changed_count_sum"] = float(
            stats["teacher_action_rollout_changed_count_sum"]
        ) + float(teacher_action_rollout_changed_mask.sum())
        stats["teacher_action_rollout_changed_f1_sum"] = float(
            stats["teacher_action_rollout_changed_f1_sum"]
        ) + float(rollout_f1)
        stats["candidate_size_sum"] = float(stats["candidate_size_sum"]) + float(mask.sum())
        if is_noop_sample:
            stats["included_noop"] = int(stats["included_noop"]) + 1
        stats["planner_projected_changed_count_sum"] = float(stats["planner_projected_changed_count_sum"]) + float(
            selected.get("projected_changed_count", 0.0)
        )
        candidate_quality_available = float("teacher_overlap_f1" in selected)
        candidate_quality_target = float(selected.get("teacher_overlap_f1", 0.0))
        stats["planner_candidate_quality_target_sum"] = float(
            stats["planner_candidate_quality_target_sum"]
        ) + candidate_quality_target
        stats["planner_candidate_quality_available_sum"] = float(
            stats["planner_candidate_quality_available_sum"]
        ) + candidate_quality_available
        stats["planner_candidate_quality_score_sum"] = float(
            stats["planner_candidate_quality_score_sum"]
        ) + float(selected.get("candidate_quality_score", 0.0))
        stats["planner_score_sum"] = float(stats["planner_score_sum"]) + float(selected.get("selection_score", 0.0))
        planner_projected_changed_mask = _planner_projected_mask_for_sample(
            candidate_positions=positions,
            candidate_mask=mask,
            projected_positions=selected.get("projected_changed_positions", []),
        )
        if planner_selected_store_candidate_overlap_masks:
            planner_candidate_teacher_changed_mask, planner_candidate_false_positive_mask = (
                _planner_candidate_overlap_masks_for_sample(
                    candidate_positions=positions,
                    candidate_mask=mask,
                    candidates=candidates,
                )
            )
        else:
            planner_candidate_teacher_changed_mask = np.zeros_like(mask, dtype=np.float32)
            planner_candidate_false_positive_mask = np.zeros_like(mask, dtype=np.float32)
        stats["planner_candidate_teacher_changed_count_sum"] = float(
            stats["planner_candidate_teacher_changed_count_sum"]
        ) + float(planner_candidate_teacher_changed_mask.sum())
        stats["planner_candidate_false_positive_count_sum"] = float(
            stats["planner_candidate_false_positive_count_sum"]
        ) + float(planner_candidate_false_positive_mask.sum())
        samples_by_k[horizon_k] = samples_by_k.get(horizon_k, 0) + 1
        samples.append(
            MacroSegmentSample(
                start_obs=start_obs,
                next_obs=next_obs.copy(),
                start_vacancy_positions=start_vacancies.copy(),
                start_cu_positions=start_cu.copy(),
                global_summary=global_summary,
                teacher_path_summary=teacher_summary,
                candidate_positions=positions,
                nearest_vacancy_offset=nearest_offsets,
                reach_depth=reach_depth,
                is_start_vacancy=is_start_vacancy,
                current_types=current_types,
                target_types=target_types,
                candidate_mask=mask,
                changed_mask=changed_mask,
                tau_exp=float(tau_exp),
                tau_real=float(tau_real),
                reward_sum=float(reward_sum),
                horizon_k=int(realized_horizon_k),
                box_dims=adapter.box_dims().astype(np.float32),
                planner_projected_changed_mask=planner_projected_changed_mask,
                planner_teacher_overlap_f1=float(selected.get("teacher_overlap_f1", 0.0)),
                planner_candidate_teacher_changed_mask=planner_candidate_teacher_changed_mask,
                planner_candidate_false_positive_mask=planner_candidate_false_positive_mask,
                planner_candidate_quality_target=candidate_quality_target,
                planner_candidate_quality_available=candidate_quality_available,
                teacher_touched_mask=teacher_touched_mask,
                teacher_action_source_mask=teacher_action_source_mask,
                teacher_action_destination_mask=teacher_action_destination_mask,
                teacher_action_edge_pair_indices=teacher_action_edge_pair_indices,
                teacher_action_edge_pair_mask=teacher_action_edge_pair_mask,
                teacher_action_edge_pair_support_mask=teacher_action_edge_pair_support_mask,
                teacher_action_edge_pair_moving_type=teacher_action_edge_pair_moving_type,
                teacher_action_edge_pair_order=teacher_action_edge_pair_order,
                teacher_vacancy_pair_indices=teacher_vacancy_pair_indices,
                teacher_vacancy_pair_mask=teacher_vacancy_pair_mask,
                teacher_vacancy_pair_moving_type=teacher_vacancy_pair_moving_type,
                teacher_vacancy_pair_order=teacher_vacancy_pair_order,
                teacher_action_sequence_indices=teacher_action_sequence_indices,
                teacher_action_sequence_mask=teacher_action_sequence_mask,
                teacher_action_sequence_moving_type=teacher_action_sequence_moving_type,
                teacher_action_sequence_order=teacher_action_sequence_order,
                teacher_action_rollout_changed_mask=teacher_action_rollout_changed_mask,
            )
        )
        stall_attempts = 0
        if len(samples) % progress_every == 0 or len(samples) == num_segments:
            print(
                json.dumps(
                    {
                        "planner_selected_collect_progress": {
                            "samples": len(samples),
                            "target": num_segments,
                            "attempts": int(stats["attempts"]),
                            "coverage": float(len(samples) / max(int(stats["attempts"]), 1)),
                            "chosen_k_histogram": {str(k): int(v) for k, v in sorted(samples_by_k.items())},
                            "selected_attempt_k_histogram": {str(k): int(v) for k, v in sorted(selected_attempts_by_k.items())},
                            "reward_only_uncovered": int(stats["reward_only_uncovered"]),
                            "skipped_uncovered": int(stats["skipped_uncovered"]),
                            "teacher_overlap_avg_f1": float(stats["teacher_overlap_f1_sum"]) / max(float(stats["teacher_overlap_probe_count"]), 1.0),
                            "teacher_overlap_selected_avg_f1": float(stats["teacher_overlap_selected_f1_sum"]) / max(float(len(samples)), 1.0),
                        }
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        segments_since_reset += 1
        if is_noop_sample and not keep_after_noop_segments:
            env, obs = restart_env(env)
            segments_since_reset = 0
        elif max_segments_per_rollout > 0 and segments_since_reset >= max_segments_per_rollout:
            env, obs = restart_env(env)
            segments_since_reset = 0
        else:
            obs = next_obs

    denom = max(len(samples), 1)
    stats["samples"] = int(len(samples))
    stats["coverage"] = float(len(samples) / max(int(stats["attempts"]), 1))
    stats["avg_candidate_size"] = float(float(stats["candidate_size_sum"]) / denom)
    stats["avg_teacher_touched_count"] = float(float(stats["teacher_touched_count_sum"]) / denom)
    stats["avg_teacher_touched_mask_count"] = float(float(stats["teacher_touched_mask_count_sum"]) / denom)
    stats["avg_teacher_touched_recall"] = float(float(stats["teacher_touched_recall_sum"]) / denom)
    stats["avg_teacher_action_source_mask_count"] = float(float(stats["teacher_action_source_mask_count_sum"]) / denom)
    stats["avg_teacher_action_destination_mask_count"] = float(
        float(stats["teacher_action_destination_mask_count_sum"]) / denom
    )
    stats["avg_teacher_action_source_recall"] = float(float(stats["teacher_action_source_recall_sum"]) / denom)
    stats["avg_teacher_action_destination_recall"] = float(
        float(stats["teacher_action_destination_recall_sum"]) / denom
    )
    stats["avg_teacher_action_edge_pair_count"] = float(float(stats["teacher_action_edge_pair_count_sum"]) / denom)
    stats["avg_teacher_action_edge_pair_covered"] = float(
        float(stats["teacher_action_edge_pair_covered_sum"]) / denom
    )
    stats["avg_teacher_action_edge_pair_unique_count"] = float(
        float(stats["teacher_action_edge_pair_unique_count_sum"]) / denom
    )
    stats["avg_teacher_action_edge_pair_support_count"] = float(
        float(stats["teacher_action_edge_pair_support_count_sum"]) / denom
    )
    stats["avg_teacher_action_sequence_step_count"] = float(
        float(stats["teacher_action_sequence_step_count_sum"]) / denom
    )
    stats["avg_teacher_action_sequence_covered"] = float(
        float(stats["teacher_action_sequence_covered_sum"]) / denom
    )
    stats["avg_teacher_action_rollout_changed_count"] = float(
        float(stats["teacher_action_rollout_changed_count_sum"]) / denom
    )
    stats["avg_teacher_action_rollout_changed_f1"] = float(
        float(stats["teacher_action_rollout_changed_f1_sum"]) / denom
    )
    stats["avg_teacher_action_edge_pair_recall"] = float(
        float(stats["teacher_action_edge_pair_covered_sum"])
        / max(float(stats["teacher_action_edge_pair_count_sum"]), 1.0)
    )
    stats["avg_teacher_action_sequence_recall"] = float(
        float(stats["teacher_action_sequence_covered_sum"])
        / max(float(stats["teacher_action_sequence_step_count_sum"]), 1.0)
    )
    stats["avg_realized_horizon"] = float(float(stats["realized_horizon_sum"]) / denom)
    stats["avg_planner_projected_changed_count"] = float(float(stats["planner_projected_changed_count_sum"]) / denom)
    stats["avg_planner_candidate_teacher_changed_count"] = float(
        float(stats["planner_candidate_teacher_changed_count_sum"]) / denom
    )
    stats["avg_planner_candidate_false_positive_count"] = float(
        float(stats["planner_candidate_false_positive_count_sum"]) / denom
    )
    stats["avg_planner_candidate_quality_target"] = float(
        float(stats["planner_candidate_quality_target_sum"]) / max(float(stats["planner_candidate_quality_available_sum"]), 1.0)
    )
    stats["candidate_quality_available_frac"] = float(float(stats["planner_candidate_quality_available_sum"]) / denom)
    stats["avg_planner_candidate_quality_score"] = float(float(stats["planner_candidate_quality_score_sum"]) / denom)
    stats["avg_planner_score"] = float(float(stats["planner_score_sum"]) / denom)
    probe_count = max(float(stats.get("teacher_overlap_probe_count", 0.0)), 1.0)
    stats["avg_teacher_overlap_f1"] = float(float(stats.get("teacher_overlap_f1_sum", 0.0)) / probe_count)
    stats["avg_teacher_overlap_precision"] = float(float(stats.get("teacher_overlap_precision_sum", 0.0)) / probe_count)
    stats["avg_teacher_overlap_recall"] = float(float(stats.get("teacher_overlap_recall_sum", 0.0)) / probe_count)
    stats["avg_teacher_overlap_selected_f1"] = float(float(stats.get("teacher_overlap_selected_f1_sum", 0.0)) / denom)
    stats["chosen_k_histogram"] = {str(k): int(samples_by_k.get(k, 0)) for k in sorted(samples_by_k)}
    stats["selected_attempt_k_histogram"] = {str(k): int(selected_attempts_by_k.get(k, 0)) for k in sorted(selected_attempts_by_k)}
    stats["skipped_uncovered_by_k"] = {str(k): int(skipped_uncovered_by_k.get(k, 0)) for k in sorted(skipped_uncovered_by_k)}
    stats["reward_only_uncovered_by_k"] = {str(k): int(reward_only_uncovered_by_k.get(k, 0)) for k in sorted(reward_only_uncovered_by_k)}
    stats["by_k"] = {
        str(k): {"samples": int(samples_by_k.get(k, 0)), "sample_frac": float(samples_by_k.get(k, 0) / denom)}
        for k in sorted(samples_by_k)
    }
    return samples, stats


def _save_samples(samples: list[MacroSegmentSample], path: Path) -> None:
    payload = [asdict(sample) for sample in samples]
    torch.save(payload, path)


def _load_samples(path: Path) -> list[MacroSegmentSample]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return [MacroSegmentSample(**item) for item in payload]


def _planner_projected_mask_for_sample(
    candidate_positions: np.ndarray,
    candidate_mask: np.ndarray,
    projected_positions: object,
) -> np.ndarray:
    mask = np.zeros_like(candidate_mask, dtype=np.float32)
    if not projected_positions:
        return mask
    valid_indices = np.flatnonzero(candidate_mask > 0)
    pos_to_idx = {
        tuple(map(int, candidate_positions[idx].astype(np.int32).tolist())): int(idx)
        for idx in valid_indices.tolist()
    }
    for raw_pos in projected_positions:
        try:
            pos = tuple(map(int, raw_pos))
        except TypeError:
            continue
        idx = pos_to_idx.get(pos)
        if idx is not None:
            mask[idx] = 1.0
    return mask


def _planner_candidate_overlap_masks_for_sample(
    candidate_positions: np.ndarray,
    candidate_mask: np.ndarray,
    candidates: list[dict[str, object]],
) -> tuple[np.ndarray, np.ndarray]:
    teacher_changed_mask = np.zeros_like(candidate_mask, dtype=np.float32)
    false_positive_mask = np.zeros_like(candidate_mask, dtype=np.float32)
    valid_indices = np.flatnonzero(candidate_mask > 0)
    pos_to_idx = {
        tuple(map(int, candidate_positions[idx].astype(np.int32).tolist())): int(idx)
        for idx in valid_indices.tolist()
    }

    def mark(raw_positions: object, target: np.ndarray) -> None:
        if not raw_positions:
            return
        for raw_pos in raw_positions:
            try:
                pos = tuple(int(x) for x in raw_pos)
            except (TypeError, ValueError):
                continue
            if len(pos) != 3:
                continue
            idx = pos_to_idx.get(pos)
            if idx is not None:
                target[idx] = 1.0

    for candidate in candidates:
        mark(candidate.get("teacher_changed_positions_probe", []), teacher_changed_mask)
        mark(candidate.get("projected_false_positive_positions_probe", []), false_positive_mask)
    false_positive_mask = np.where(teacher_changed_mask > 0.5, 0.0, false_positive_mask).astype(np.float32)
    return teacher_changed_mask.astype(np.float32), false_positive_mask


def _batch_to_device(batch: list[MacroSegmentSample], device: str) -> dict[str, torch.Tensor]:
    planner_projected_masks = []
    planner_candidate_teacher_masks = []
    planner_candidate_false_positive_masks = []
    teacher_touched_masks = []
    teacher_action_source_masks = []
    teacher_action_destination_masks = []
    teacher_action_edge_pair_indices = []
    teacher_action_edge_pair_masks = []
    teacher_action_edge_pair_support_masks = []
    teacher_action_edge_pair_moving_types = []
    teacher_action_edge_pair_orders = []
    teacher_vacancy_pair_indices = []
    teacher_vacancy_pair_masks = []
    teacher_vacancy_pair_moving_types = []
    teacher_vacancy_pair_orders = []
    teacher_action_sequence_indices = []
    teacher_action_sequence_masks = []
    teacher_action_sequence_moving_types = []
    teacher_action_sequence_orders = []
    teacher_action_rollout_changed_masks = []
    for sample in batch:
        projected_mask = getattr(sample, "planner_projected_changed_mask", None)
        if projected_mask is None:
            projected_mask = np.zeros_like(sample.changed_mask, dtype=np.float32)
        planner_projected_masks.append(projected_mask)
        candidate_teacher_mask = getattr(sample, "planner_candidate_teacher_changed_mask", None)
        if candidate_teacher_mask is None:
            candidate_teacher_mask = np.zeros_like(sample.changed_mask, dtype=np.float32)
        planner_candidate_teacher_masks.append(candidate_teacher_mask)
        candidate_false_positive_mask = getattr(sample, "planner_candidate_false_positive_mask", None)
        if candidate_false_positive_mask is None:
            candidate_false_positive_mask = np.zeros_like(sample.changed_mask, dtype=np.float32)
        planner_candidate_false_positive_masks.append(candidate_false_positive_mask)
        teacher_touched_mask = getattr(sample, "teacher_touched_mask", None)
        if teacher_touched_mask is None:
            teacher_touched_mask = sample.changed_mask
        teacher_touched_masks.append(teacher_touched_mask)
        teacher_action_source_mask = getattr(sample, "teacher_action_source_mask", None)
        if teacher_action_source_mask is None:
            teacher_action_source_mask = teacher_touched_mask
        teacher_action_source_masks.append(teacher_action_source_mask)
        teacher_action_destination_mask = getattr(sample, "teacher_action_destination_mask", None)
        if teacher_action_destination_mask is None:
            teacher_action_destination_mask = teacher_touched_mask
        teacher_action_destination_masks.append(teacher_action_destination_mask)
        edge_pair_indices = getattr(sample, "teacher_action_edge_pair_indices", None)
        if edge_pair_indices is None:
            edge_pair_indices = np.full((sample.candidate_mask.shape[0], 2), -1, dtype=np.int64)
        teacher_action_edge_pair_indices.append(edge_pair_indices)
        edge_pair_mask = getattr(sample, "teacher_action_edge_pair_mask", None)
        if edge_pair_mask is None:
            edge_pair_mask = np.zeros_like(sample.candidate_mask, dtype=np.float32)
        teacher_action_edge_pair_masks.append(edge_pair_mask)
        edge_pair_support_mask = getattr(sample, "teacher_action_edge_pair_support_mask", None)
        if edge_pair_support_mask is None:
            edge_pair_support_mask = np.zeros_like(sample.candidate_mask, dtype=np.float32)
            valid_pairs = edge_pair_indices[(edge_pair_indices[:, 0] >= 0) & (edge_pair_indices[:, 1] >= 0)]
            for idx, (source_idx, dest_idx) in enumerate(valid_pairs.tolist()):
                edge_pair_support_mask[idx] = float(
                    sample.changed_mask[int(source_idx)] > 0.5 or sample.changed_mask[int(dest_idx)] > 0.5
                )
        teacher_action_edge_pair_support_masks.append(edge_pair_support_mask)
        edge_pair_moving_type = getattr(sample, "teacher_action_edge_pair_moving_type", None)
        if edge_pair_moving_type is None:
            edge_pair_moving_type = np.full_like(sample.candidate_mask, -1, dtype=np.int64)
        teacher_action_edge_pair_moving_types.append(edge_pair_moving_type)
        edge_pair_order = getattr(sample, "teacher_action_edge_pair_order", None)
        if edge_pair_order is None:
            edge_pair_order = np.zeros_like(sample.candidate_mask, dtype=np.float32)
        teacher_action_edge_pair_orders.append(edge_pair_order)
        vacancy_pair_indices = getattr(sample, "teacher_vacancy_pair_indices", None)
        if vacancy_pair_indices is None:
            vacancy_pair_indices = np.full((sample.candidate_mask.shape[0], 2), -1, dtype=np.int64)
        teacher_vacancy_pair_indices.append(vacancy_pair_indices)
        vacancy_pair_mask = getattr(sample, "teacher_vacancy_pair_mask", None)
        if vacancy_pair_mask is None:
            vacancy_pair_mask = np.zeros_like(sample.candidate_mask, dtype=np.float32)
        teacher_vacancy_pair_masks.append(vacancy_pair_mask)
        vacancy_pair_moving_type = getattr(sample, "teacher_vacancy_pair_moving_type", None)
        if vacancy_pair_moving_type is None:
            vacancy_pair_moving_type = np.full_like(sample.candidate_mask, -1, dtype=np.int64)
        teacher_vacancy_pair_moving_types.append(vacancy_pair_moving_type)
        vacancy_pair_order = getattr(sample, "teacher_vacancy_pair_order", None)
        if vacancy_pair_order is None:
            vacancy_pair_order = np.zeros_like(sample.candidate_mask, dtype=np.float32)
        teacher_vacancy_pair_orders.append(vacancy_pair_order)
        sequence_indices = getattr(sample, "teacher_action_sequence_indices", None)
        if sequence_indices is None:
            sequence_indices = edge_pair_indices
        teacher_action_sequence_indices.append(sequence_indices)
        sequence_mask = getattr(sample, "teacher_action_sequence_mask", None)
        if sequence_mask is None:
            sequence_mask = edge_pair_mask
        teacher_action_sequence_masks.append(sequence_mask)
        sequence_moving_type = getattr(sample, "teacher_action_sequence_moving_type", None)
        if sequence_moving_type is None:
            sequence_moving_type = edge_pair_moving_type
        teacher_action_sequence_moving_types.append(sequence_moving_type)
        sequence_order = getattr(sample, "teacher_action_sequence_order", None)
        if sequence_order is None:
            sequence_order = edge_pair_order
        teacher_action_sequence_orders.append(sequence_order)
        action_rollout_changed_mask = getattr(sample, "teacher_action_rollout_changed_mask", None)
        if action_rollout_changed_mask is None:
            action_rollout_changed_mask = sample.changed_mask
        teacher_action_rollout_changed_masks.append(action_rollout_changed_mask)
    return {
        "start_obs": torch.tensor(np.stack([sample.start_obs for sample in batch]), dtype=torch.float32, device=device),
        "next_obs": torch.tensor(np.stack([sample.next_obs for sample in batch]), dtype=torch.float32, device=device),
        "global_summary": torch.tensor(np.stack([sample.global_summary for sample in batch]), dtype=torch.float32, device=device),
        "teacher_path_summary": torch.tensor(np.stack([sample.teacher_path_summary for sample in batch]), dtype=torch.float32, device=device),
        "candidate_positions": torch.tensor(np.stack([sample.candidate_positions for sample in batch]), dtype=torch.float32, device=device),
        "nearest_vacancy_offset": torch.tensor(np.stack([sample.nearest_vacancy_offset for sample in batch]), dtype=torch.float32, device=device),
        "reach_depth": torch.tensor(np.stack([sample.reach_depth for sample in batch]), dtype=torch.float32, device=device),
        "is_start_vacancy": torch.tensor(np.stack([sample.is_start_vacancy for sample in batch]), dtype=torch.float32, device=device),
        "current_types": torch.tensor(np.stack([sample.current_types for sample in batch]), dtype=torch.long, device=device),
        "target_types": torch.tensor(np.stack([sample.target_types for sample in batch]), dtype=torch.long, device=device),
        "candidate_mask": torch.tensor(np.stack([sample.candidate_mask for sample in batch]), dtype=torch.float32, device=device),
        "changed_mask": torch.tensor(np.stack([sample.changed_mask for sample in batch]), dtype=torch.float32, device=device),
        "teacher_touched_mask": torch.tensor(np.stack(teacher_touched_masks), dtype=torch.float32, device=device),
        "teacher_action_source_mask": torch.tensor(
            np.stack(teacher_action_source_masks), dtype=torch.float32, device=device
        ),
        "teacher_action_destination_mask": torch.tensor(
            np.stack(teacher_action_destination_masks), dtype=torch.float32, device=device
        ),
        "teacher_action_edge_pair_indices": torch.tensor(
            np.stack(teacher_action_edge_pair_indices), dtype=torch.long, device=device
        ),
        "teacher_action_edge_pair_mask": torch.tensor(
            np.stack(teacher_action_edge_pair_masks), dtype=torch.float32, device=device
        ),
        "teacher_action_edge_pair_support_mask": torch.tensor(
            np.stack(teacher_action_edge_pair_support_masks), dtype=torch.float32, device=device
        ),
        "teacher_action_edge_pair_moving_type": torch.tensor(
            np.stack(teacher_action_edge_pair_moving_types), dtype=torch.long, device=device
        ),
        "teacher_action_edge_pair_order": torch.tensor(
            np.stack(teacher_action_edge_pair_orders), dtype=torch.float32, device=device
        ),
        "teacher_vacancy_pair_indices": torch.tensor(
            np.stack(teacher_vacancy_pair_indices), dtype=torch.long, device=device
        ),
        "teacher_vacancy_pair_mask": torch.tensor(
            np.stack(teacher_vacancy_pair_masks), dtype=torch.float32, device=device
        ),
        "teacher_vacancy_pair_moving_type": torch.tensor(
            np.stack(teacher_vacancy_pair_moving_types), dtype=torch.long, device=device
        ),
        "teacher_vacancy_pair_order": torch.tensor(
            np.stack(teacher_vacancy_pair_orders), dtype=torch.float32, device=device
        ),
        "teacher_action_sequence_indices": torch.tensor(
            np.stack(teacher_action_sequence_indices), dtype=torch.long, device=device
        ),
        "teacher_action_sequence_mask": torch.tensor(
            np.stack(teacher_action_sequence_masks), dtype=torch.float32, device=device
        ),
        "teacher_action_sequence_moving_type": torch.tensor(
            np.stack(teacher_action_sequence_moving_types), dtype=torch.long, device=device
        ),
        "teacher_action_sequence_order": torch.tensor(
            np.stack(teacher_action_sequence_orders), dtype=torch.float32, device=device
        ),
        "teacher_action_rollout_changed_mask": torch.tensor(
            np.stack(teacher_action_rollout_changed_masks), dtype=torch.float32, device=device
        ),
        "planner_projected_changed_mask": torch.tensor(np.stack(planner_projected_masks), dtype=torch.float32, device=device),
        "planner_candidate_teacher_changed_mask": torch.tensor(
            np.stack(planner_candidate_teacher_masks), dtype=torch.float32, device=device
        ),
        "planner_candidate_false_positive_mask": torch.tensor(
            np.stack(planner_candidate_false_positive_masks), dtype=torch.float32, device=device
        ),
        "planner_candidate_quality_target": torch.tensor(
            [float(getattr(sample, "planner_candidate_quality_target", 0.0)) for sample in batch],
            dtype=torch.float32,
            device=device,
        ),
        "planner_candidate_quality_available": torch.tensor(
            [float(getattr(sample, "planner_candidate_quality_available", 0.0)) for sample in batch],
            dtype=torch.float32,
            device=device,
        ),
        "tau_exp": torch.tensor([sample.tau_exp for sample in batch], dtype=torch.float32, device=device),
        "tau_real": torch.tensor([sample.tau_real for sample in batch], dtype=torch.float32, device=device),
        "reward_sum": torch.tensor([sample.reward_sum for sample in batch], dtype=torch.float32, device=device),
        "horizon_k": torch.tensor([sample.horizon_k for sample in batch], dtype=torch.long, device=device),
        "box_dims": torch.tensor(np.stack([sample.box_dims for sample in batch]), dtype=torch.float32, device=device),
    }


class _ProjectedStateEnv:
    def __init__(self, vacancies: np.ndarray, cu_positions: np.ndarray, dims: np.ndarray):
        self._vacancies = np.asarray(vacancies, dtype=np.int32)
        self._cu_positions = np.asarray(cu_positions, dtype=np.int32)
        self.dims = tuple(int(x) for x in np.asarray(dims, dtype=np.int32).tolist())
        self.V_TYPE = V_TYPE
        self.CU_TYPE = CU_TYPE

    def get_vacancy_array(self) -> np.ndarray:
        return self._vacancies

    def get_cu_array(self) -> np.ndarray:
        return self._cu_positions


def _apply_projected_types(sample: MacroSegmentSample, projected_types: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vac_set = {tuple(map(int, pos)) for pos in sample.start_vacancy_positions.tolist()}
    cu_set = {tuple(map(int, pos)) for pos in sample.start_cu_positions.tolist()}
    valid_indices = np.flatnonzero(sample.candidate_mask > 0)
    for idx in valid_indices.tolist():
        pos = tuple(map(int, sample.candidate_positions[idx].astype(np.int32).tolist()))
        vac_set.discard(pos)
        cu_set.discard(pos)
        new_type = int(projected_types[idx])
        if new_type == V_TYPE:
            vac_set.add(pos)
        elif new_type == CU_TYPE:
            cu_set.add(pos)
    vacancies = np.asarray(sorted(vac_set), dtype=np.int32) if vac_set else np.empty((0, 3), dtype=np.int32)
    cu_positions = np.asarray(sorted(cu_set), dtype=np.int32) if cu_set else np.empty((0, 3), dtype=np.int32)
    return vacancies, cu_positions


def _projected_global_latent_batch(
    *,
    batch: list[MacroSegmentSample],
    projected_types: torch.Tensor,
    model: MacroDreamerEditModel,
    device: str,
) -> torch.Tensor:
    shape = model.global_encoder.shape
    types_np = projected_types.detach().cpu().numpy()

    def _build_one(args: tuple[MacroSegmentSample, np.ndarray]) -> np.ndarray:
        sample, types = args
        vacancies, cu_positions = _apply_projected_types(sample, types)
        proxy_env = _ProjectedStateEnv(vacancies, cu_positions, sample.box_dims)
        share_obs = sample.start_obs[-shape.stats_dim:]
        return build_defect_graph_observation(proxy_env, shape=shape, share_obs=share_obs).astype(np.float32)

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(8, len(batch))) as executor:
        projected_obs = list(executor.map(_build_one, zip(batch, types_np)))

    projected_obs_t = torch.tensor(np.stack(projected_obs), dtype=torch.float32, device=device)
    return model.encode_global(projected_obs_t)


def _compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
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


def _compute_lognormal_distribution_metrics(mu: np.ndarray, log_sigma: np.ndarray, target: np.ndarray) -> dict[str, float]:
    eps = 1e-12
    mu = np.asarray(mu, dtype=np.float64)
    log_sigma = np.asarray(log_sigma, dtype=np.float64)
    target = np.clip(np.asarray(target, dtype=np.float64), eps, None)
    point_pred = np.exp(mu)
    linear_metrics = _compute_metrics(point_pred, target)
    log_metrics = _compute_log_metrics(point_pred, target)

    mu_t = torch.as_tensor(mu, dtype=torch.float64)
    log_sigma_t = torch.as_tensor(log_sigma, dtype=torch.float64).clamp(min=-6.0, max=2.0)
    sigma_t = torch.exp(log_sigma_t).clamp(min=1e-6)
    target_t = torch.as_tensor(target, dtype=torch.float64)
    log_target_t = torch.log(target_t)
    z = (log_target_t - mu_t) / sigma_t
    nll = (log_target_t + log_sigma_t + 0.5 * math.log(2.0 * math.pi) + 0.5 * z.square()).mean()
    pit = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    pit_sorted = torch.sort(pit).values
    if pit_sorted.numel() > 0:
        count = pit_sorted.numel()
        empirical_hi = torch.arange(1, count + 1, dtype=torch.float64) / float(count)
        empirical_lo = torch.arange(0, count, dtype=torch.float64) / float(count)
        pit_ks = float(torch.max(torch.maximum((empirical_hi - pit_sorted).abs(), (pit_sorted - empirical_lo).abs())).item())
        pit_mean = float(pit.mean().item())
        pit_var = float(pit.var(unbiased=False).item())
        coverage_68 = float((z.abs() <= 1.0).double().mean().item())
        coverage_95 = float((z.abs() <= 1.959963984540054).double().mean().item())
        mean_log_sigma = float(log_sigma_t.mean().item())
    else:
        pit_ks = 0.0
        pit_mean = 0.5
        pit_var = 0.0
        coverage_68 = 0.0
        coverage_95 = 0.0
        mean_log_sigma = 0.0
    return {
        **linear_metrics,
        **log_metrics,
        "nll": float(nll.item()),
        "coverage_68": coverage_68,
        "coverage_95": coverage_95,
        "pit_mean": pit_mean,
        "pit_var": pit_var,
        "pit_ks": pit_ks,
        "mean_log_sigma": mean_log_sigma,
        "predicted_median_mean": float(np.mean(point_pred)) if point_pred.size else 0.0,
    }


def _focal_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.75, gamma: float = 2.0) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = probs * targets + (1.0 - probs) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return (alpha_t * ((1.0 - pt).clamp(min=1e-6) ** gamma) * ce).mean()


def _proposal_target_from_tensors(tensors: dict[str, torch.Tensor], source: str) -> torch.Tensor:
    source_key = str(source)
    changed = tensors["changed_mask"].float()
    touched = tensors.get("teacher_touched_mask", changed).float()
    source_mask = tensors.get("teacher_action_source_mask", touched).float()
    destination_mask = tensors.get("teacher_action_destination_mask", touched).float()
    rollout_mask = tensors.get("teacher_action_rollout_changed_mask", changed).float()
    if source_key == "changed":
        return changed
    if source_key == "action_rollout":
        return rollout_mask
    if source_key == "touched":
        return touched
    if source_key == "action_source":
        return source_mask
    if source_key == "action_destination":
        return destination_mask
    if source_key == "action_endpoint":
        return torch.clamp(source_mask + destination_mask, min=0.0, max=1.0)
    if source_key in {"changed_or_touched", "union"}:
        return torch.clamp(changed + touched, min=0.0, max=1.0)
    raise ValueError(f"Unknown proposal_target_source={source!r}")


def _terminal_action_context_logits_from_tensors(
    tensors: dict[str, torch.Tensor],
    source: str,
    fallback_logits: torch.Tensor,
) -> torch.Tensor:
    if source == "action_endpoint":
        return fallback_logits
    if source == "teacher_rollout":
        rollout = tensors.get("teacher_action_rollout_changed_mask")
        if rollout is None:
            rollout = tensors["changed_mask"]
        rollout = rollout.float()
        logits = torch.where(
            rollout > 0.5,
            torch.full_like(rollout, 6.0),
            torch.full_like(rollout, -6.0),
        )
        return torch.where(tensors["candidate_mask"] > 0, logits, torch.full_like(logits, -20.0))
    raise ValueError(f"unknown terminal_edit_action_context_source={source!r}")


def _vacancy_displacement_target_from_tensors(tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    current = tensors["current_types"]
    target = tensors["target_types"]
    valid = tensors["candidate_mask"] > 0
    changed = current != target
    vacancy_edge = (current == V_TYPE) | (target == V_TYPE)
    return (valid & changed & vacancy_edge).float()


def _typed_diff_change_logits(type_logits: torch.Tensor, current_types: torch.Tensor) -> torch.Tensor:
    current_logits = type_logits.gather(-1, current_types.unsqueeze(-1)).squeeze(-1)
    noncopy_logits = type_logits.masked_fill(
        F.one_hot(current_types, num_classes=NUM_SITE_TYPES).bool(),
        -1.0e4,
    )
    return torch.logsumexp(noncopy_logits, dim=-1) - current_logits


def _terminal_typed_diff_loss(
    type_logits: torch.Tensor,
    *,
    target_types: torch.Tensor,
    current_types: torch.Tensor,
    target_mask: torch.Tensor,
    candidate_mask: torch.Tensor,
    copy_weight: float = 0.05,
    support_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    zero = type_logits.sum() * 0.0
    valid = candidate_mask > 0
    positive = valid & (target_mask > 0.5)
    negative = valid & ~positive
    if positive.any():
        pos_loss = F.cross_entropy(type_logits[positive], target_types[positive], reduction="mean")
        type_acc = (type_logits.argmax(dim=-1)[positive] == target_types[positive]).float().mean()
    else:
        pos_loss = zero
        type_acc = torch.ones((), dtype=type_logits.dtype, device=type_logits.device)
    if negative.any() and copy_weight > 0.0:
        neg_loss = F.cross_entropy(type_logits[negative], current_types[negative], reduction="mean")
        copy_acc = (type_logits.argmax(dim=-1)[negative] == current_types[negative]).float().mean()
    else:
        neg_loss = zero
        copy_acc = torch.ones((), dtype=type_logits.dtype, device=type_logits.device)
    change_logits = _typed_diff_change_logits(type_logits, current_types)
    support_terms = _proposal_support_loss(change_logits, target_mask, candidate_mask)
    support_loss = support_terms["loss"]
    return {
        "loss": pos_loss + float(copy_weight) * neg_loss + float(support_weight) * support_loss,
        "positive_type": pos_loss,
        "copy": neg_loss,
        "support": support_loss,
        "type_acc": type_acc,
        "copy_acc": copy_acc,
        "change_logits": change_logits,
        "topk_f1": support_terms["topk_f1"],
        "recall32": support_terms["recall32"],
    }


def _negative_action_edge_pair_indices(
    edge_pair_indices: torch.Tensor,
    *,
    candidate_positions: torch.Tensor | None = None,
    candidate_mask: torch.Tensor | None = None,
    box_dims: torch.Tensor | None = None,
    mode: str = "self",
) -> torch.Tensor:
    negative = edge_pair_indices.clone()
    negative[..., 1] = edge_pair_indices[..., 0]
    if mode == "self":
        return negative
    if mode != "same_source_nn1":
        raise ValueError(f"Unknown action_edge_pair_negative_mode={mode!r}")
    if candidate_positions is None or candidate_mask is None or box_dims is None:
        return negative

    squeezed = False
    edge_pairs = edge_pair_indices
    positions = candidate_positions
    mask = candidate_mask
    boxes = box_dims
    if edge_pairs.dim() == 2:
        edge_pairs = edge_pairs.unsqueeze(0)
        positions = positions.unsqueeze(0)
        mask = mask.unsqueeze(0)
        boxes = boxes.unsqueeze(0) if boxes.dim() == 1 else boxes
        squeezed = True
    if edge_pairs.dim() != 3 or positions.dim() != 3 or mask.dim() != 2 or boxes.dim() != 2:
        return negative

    pairs_cpu = edge_pairs.detach().cpu().long()
    negative_cpu = negative.detach().cpu().long()
    if squeezed:
        negative_cpu = negative_cpu.unsqueeze(0)
    positions_cpu = torch.round(positions.detach().cpu()).long()
    mask_cpu = mask.detach().cpu() > 0.5
    boxes_cpu = torch.round(boxes.detach().cpu()).long().clamp_min(1)
    offsets = torch.tensor(BCC_NN1_OFFSETS, dtype=torch.long)

    batch_size, max_pairs, _ = pairs_cpu.shape
    max_sites = positions_cpu.shape[1]
    for batch_idx in range(batch_size):
        lookup: dict[tuple[int, int, int], int] = {}
        for site_idx in torch.nonzero(mask_cpu[batch_idx], as_tuple=False).flatten().tolist():
            key = tuple(int(v) for v in positions_cpu[batch_idx, site_idx].tolist())
            lookup[key] = int(site_idx)
        if not lookup:
            continue
        box = boxes_cpu[batch_idx]
        for pair_idx in range(max_pairs):
            source_idx = int(pairs_cpu[batch_idx, pair_idx, 0].item())
            dest_idx = int(pairs_cpu[batch_idx, pair_idx, 1].item())
            if (
                source_idx < 0
                or dest_idx < 0
                or source_idx >= max_sites
                or dest_idx >= max_sites
                or not bool(mask_cpu[batch_idx, source_idx])
            ):
                continue
            source_pos = positions_cpu[batch_idx, source_idx]
            alternatives: list[int] = []
            for offset in offsets:
                candidate_pos = torch.remainder(source_pos + offset, box)
                candidate_idx = lookup.get(tuple(int(v) for v in candidate_pos.tolist()))
                if candidate_idx is None or candidate_idx in {source_idx, dest_idx}:
                    continue
                alternatives.append(int(candidate_idx))
            if alternatives:
                choice = alternatives[(pair_idx + source_idx) % len(alternatives)]
                negative_cpu[batch_idx, pair_idx, 1] = int(choice)
    if squeezed:
        negative_cpu = negative_cpu.squeeze(0)
    return negative_cpu.to(device=edge_pair_indices.device, dtype=edge_pair_indices.dtype)


def _negative_action_edge_pair_indices_list(
    edge_pair_indices: torch.Tensor,
    *,
    candidate_positions: torch.Tensor | None = None,
    candidate_mask: torch.Tensor | None = None,
    box_dims: torch.Tensor | None = None,
    mode: str = "self",
    count: int = 1,
) -> torch.Tensor:
    neg_count = max(int(count), 1)
    base_negative = _negative_action_edge_pair_indices(
        edge_pair_indices,
        candidate_positions=candidate_positions,
        candidate_mask=candidate_mask,
        box_dims=box_dims,
        mode=mode,
    )
    if neg_count == 1 or mode == "self":
        return base_negative.unsqueeze(-2).expand(*base_negative.shape[:-1], neg_count, 2).clone()
    if mode != "same_source_nn1" or candidate_positions is None or candidate_mask is None or box_dims is None:
        return base_negative.unsqueeze(-2).expand(*base_negative.shape[:-1], neg_count, 2).clone()

    negative = base_negative.unsqueeze(-2).expand(*base_negative.shape[:-1], neg_count, 2).clone()
    squeezed = False
    edge_pairs = edge_pair_indices
    positions = candidate_positions
    mask = candidate_mask
    boxes = box_dims
    if edge_pairs.dim() == 2:
        edge_pairs = edge_pairs.unsqueeze(0)
        positions = positions.unsqueeze(0)
        mask = mask.unsqueeze(0)
        boxes = boxes.unsqueeze(0) if boxes.dim() == 1 else boxes
        negative = negative.unsqueeze(0)
        squeezed = True
    if edge_pairs.dim() != 3 or positions.dim() != 3 or mask.dim() != 2 or boxes.dim() != 2:
        return negative.squeeze(0) if squeezed else negative

    pairs_cpu = edge_pairs.detach().cpu().long()
    negative_cpu = negative.detach().cpu().long()
    positions_cpu = torch.round(positions.detach().cpu()).long()
    mask_cpu = mask.detach().cpu() > 0.5
    boxes_cpu = torch.round(boxes.detach().cpu()).long().clamp_min(1)
    offsets = torch.tensor(BCC_NN1_OFFSETS, dtype=torch.long)

    batch_size, max_pairs, _ = pairs_cpu.shape
    max_sites = positions_cpu.shape[1]
    for batch_idx in range(batch_size):
        lookup: dict[tuple[int, int, int], int] = {}
        for site_idx in torch.nonzero(mask_cpu[batch_idx], as_tuple=False).flatten().tolist():
            key = tuple(int(v) for v in positions_cpu[batch_idx, site_idx].tolist())
            lookup[key] = int(site_idx)
        if not lookup:
            continue
        box = boxes_cpu[batch_idx]
        for pair_idx in range(max_pairs):
            source_idx = int(pairs_cpu[batch_idx, pair_idx, 0].item())
            dest_idx = int(pairs_cpu[batch_idx, pair_idx, 1].item())
            if (
                source_idx < 0
                or dest_idx < 0
                or source_idx >= max_sites
                or dest_idx >= max_sites
                or not bool(mask_cpu[batch_idx, source_idx])
            ):
                continue
            source_pos = positions_cpu[batch_idx, source_idx]
            alternatives: list[int] = []
            for offset in offsets:
                candidate_pos = torch.remainder(source_pos + offset, box)
                candidate_idx = lookup.get(tuple(int(v) for v in candidate_pos.tolist()))
                if candidate_idx is None or candidate_idx in {source_idx, dest_idx}:
                    continue
                alternatives.append(int(candidate_idx))
            if not alternatives:
                continue
            for neg_idx in range(neg_count):
                choice = alternatives[(pair_idx + source_idx + neg_idx) % len(alternatives)]
                negative_cpu[batch_idx, pair_idx, neg_idx, 0] = int(source_idx)
                negative_cpu[batch_idx, pair_idx, neg_idx, 1] = int(choice)
    if squeezed:
        negative_cpu = negative_cpu.squeeze(0)
    return negative_cpu.to(device=edge_pair_indices.device, dtype=edge_pair_indices.dtype)


def _dense_legal_action_edge_pair_negative_indices(
    edge_pair_indices: torch.Tensor,
    *,
    current_types: torch.Tensor,
    candidate_positions: torch.Tensor,
    candidate_mask: torch.Tensor,
    box_dims: torch.Tensor,
    count: int = 0,
) -> torch.Tensor:
    neg_count = max(int(count), 0)
    if neg_count <= 0:
        shape = (*edge_pair_indices.shape[:-1], 0, 2)
        return torch.empty(shape, dtype=edge_pair_indices.dtype, device=edge_pair_indices.device)
    fallback = _negative_action_edge_pair_indices_list(
        edge_pair_indices,
        candidate_positions=candidate_positions,
        candidate_mask=candidate_mask,
        box_dims=box_dims,
        mode="same_source_nn1",
        count=neg_count,
    )

    if edge_pair_indices.dim() != 3:
        return fallback
    pairs_cpu = edge_pair_indices.detach().cpu().long()
    negative_cpu = fallback.detach().cpu().long()
    positions_cpu = torch.round(candidate_positions.detach().cpu()).long()
    mask_cpu = candidate_mask.detach().cpu() > 0.5
    boxes_cpu = torch.round(box_dims.detach().cpu()).long().clamp_min(1)
    types_cpu = current_types.detach().cpu().long()
    offsets = torch.tensor(BCC_NN1_OFFSETS, dtype=torch.long)

    batch_size, max_pairs, _ = pairs_cpu.shape
    max_sites = positions_cpu.shape[1]
    for batch_idx in range(batch_size):
        lookup: dict[tuple[int, int, int], int] = {}
        valid_sites = torch.nonzero(mask_cpu[batch_idx], as_tuple=False).flatten().tolist()
        for site_idx in valid_sites:
            key = tuple(int(v) for v in positions_cpu[batch_idx, site_idx].tolist())
            lookup[key] = int(site_idx)
        if not lookup:
            continue
        positives: set[tuple[int, int]] = set()
        for pair_idx in range(max_pairs):
            source_idx = int(pairs_cpu[batch_idx, pair_idx, 0].item())
            dest_idx = int(pairs_cpu[batch_idx, pair_idx, 1].item())
            if (
                0 <= source_idx < max_sites
                and 0 <= dest_idx < max_sites
                and bool(mask_cpu[batch_idx, source_idx])
                and bool(mask_cpu[batch_idx, dest_idx])
            ):
                positives.add((source_idx, dest_idx))

        legal_pairs: list[tuple[int, int]] = []
        box = boxes_cpu[batch_idx]
        for source_idx in valid_sites:
            if int(types_cpu[batch_idx, source_idx].item()) != V_TYPE:
                continue
            source_pos = positions_cpu[batch_idx, source_idx]
            for offset in offsets:
                dest_pos = torch.remainder(source_pos + offset, box)
                dest_idx = lookup.get(tuple(int(v) for v in dest_pos.tolist()))
                if dest_idx is None:
                    continue
                dest_type = int(types_cpu[batch_idx, dest_idx].item())
                if dest_type not in (FE_TYPE, CU_TYPE):
                    continue
                pair = (int(source_idx), int(dest_idx))
                if pair not in positives:
                    legal_pairs.append(pair)
        if not legal_pairs:
            continue
        for pair_idx in range(max_pairs):
            source_idx = int(pairs_cpu[batch_idx, pair_idx, 0].item())
            dest_idx = int(pairs_cpu[batch_idx, pair_idx, 1].item())
            if (
                source_idx < 0
                or dest_idx < 0
                or source_idx >= max_sites
                or dest_idx >= max_sites
                or not bool(mask_cpu[batch_idx, source_idx])
            ):
                continue
            for neg_idx in range(neg_count):
                choice = legal_pairs[(pair_idx * neg_count + neg_idx + source_idx) % len(legal_pairs)]
                negative_cpu[batch_idx, pair_idx, neg_idx, 0] = int(choice[0])
                negative_cpu[batch_idx, pair_idx, neg_idx, 1] = int(choice[1])
    return negative_cpu.to(device=edge_pair_indices.device, dtype=edge_pair_indices.dtype)


def _dense_terminal_vacancy_pair_negative_indices(
    edge_pair_indices: torch.Tensor,
    *,
    current_types: torch.Tensor,
    candidate_mask: torch.Tensor,
    count: int = 0,
) -> torch.Tensor:
    neg_count = max(int(count), 0)
    if neg_count <= 0:
        shape = (*edge_pair_indices.shape[:-1], 0, 2)
        return torch.empty(shape, dtype=edge_pair_indices.dtype, device=edge_pair_indices.device)
    if edge_pair_indices.dim() != 3:
        shape = (*edge_pair_indices.shape[:-1], neg_count, 2)
        expanded = edge_pair_indices.unsqueeze(-2).expand(shape)
        return expanded.clone()

    pairs_cpu = edge_pair_indices.detach().cpu().long()
    mask_cpu = candidate_mask.detach().cpu() > 0.5
    types_cpu = current_types.detach().cpu().long()
    batch_size, max_pairs, _ = pairs_cpu.shape
    max_sites = int(types_cpu.shape[1])
    negative_cpu = pairs_cpu.unsqueeze(2).expand(batch_size, max_pairs, neg_count, 2).clone()
    for batch_idx in range(batch_size):
        positives: set[tuple[int, int]] = set()
        for pair_idx in range(max_pairs):
            source_idx = int(pairs_cpu[batch_idx, pair_idx, 0].item())
            dest_idx = int(pairs_cpu[batch_idx, pair_idx, 1].item())
            if (
                0 <= source_idx < max_sites
                and 0 <= dest_idx < max_sites
                and bool(mask_cpu[batch_idx, source_idx])
                and bool(mask_cpu[batch_idx, dest_idx])
            ):
                positives.add((source_idx, dest_idx))

        terminal_pairs: list[tuple[int, int]] = []
        valid_sites = torch.nonzero(mask_cpu[batch_idx], as_tuple=False).flatten().tolist()
        vacancy_sources = [
            int(idx)
            for idx in valid_sites
            if int(types_cpu[batch_idx, int(idx)].item()) == V_TYPE
        ]
        atom_destinations = [
            int(idx)
            for idx in valid_sites
            if int(types_cpu[batch_idx, int(idx)].item()) in (FE_TYPE, CU_TYPE)
        ]
        for source_idx in vacancy_sources:
            for dest_idx in atom_destinations:
                pair = (int(source_idx), int(dest_idx))
                if pair not in positives:
                    terminal_pairs.append(pair)
        if not terminal_pairs:
            continue
        for pair_idx in range(max_pairs):
            source_idx = int(pairs_cpu[batch_idx, pair_idx, 0].item())
            dest_idx = int(pairs_cpu[batch_idx, pair_idx, 1].item())
            if (
                source_idx < 0
                or dest_idx < 0
                or source_idx >= max_sites
                or dest_idx >= max_sites
                or not bool(mask_cpu[batch_idx, source_idx])
            ):
                continue
            for neg_idx in range(neg_count):
                choice = terminal_pairs[(pair_idx * neg_count + neg_idx + source_idx) % len(terminal_pairs)]
                negative_cpu[batch_idx, pair_idx, neg_idx, 0] = int(choice[0])
                negative_cpu[batch_idx, pair_idx, neg_idx, 1] = int(choice[1])
    return negative_cpu.to(device=edge_pair_indices.device, dtype=edge_pair_indices.dtype)


def _dense_vacancy_atom_pair_negative_indices(
    vacancy_pair_indices: torch.Tensor,
    *,
    current_types: torch.Tensor,
    candidate_mask: torch.Tensor,
    count: int = 1,
) -> torch.Tensor:
    neg_count = max(int(count), 0)
    if neg_count <= 0:
        shape = (*vacancy_pair_indices.shape[:-1], 0, 2)
        return torch.empty(shape, dtype=vacancy_pair_indices.dtype, device=vacancy_pair_indices.device)
    if vacancy_pair_indices.dim() != 3:
        shape = (*vacancy_pair_indices.shape[:-1], neg_count, 2)
        return torch.empty(shape, dtype=vacancy_pair_indices.dtype, device=vacancy_pair_indices.device)

    pairs_cpu = vacancy_pair_indices.detach().cpu().long()
    mask_cpu = candidate_mask.detach().cpu() > 0.5
    types_cpu = current_types.detach().cpu().long()
    batch_size, max_pairs, _ = pairs_cpu.shape
    max_sites = types_cpu.shape[1]
    negative_cpu = torch.zeros((batch_size, max_pairs, neg_count, 2), dtype=torch.long)

    for batch_idx in range(batch_size):
        valid_sites = torch.nonzero(mask_cpu[batch_idx], as_tuple=False).flatten().tolist()
        valid_pairs: list[tuple[int, int]] = []
        positives: set[tuple[int, int]] = set()
        for pair_idx in range(max_pairs):
            source_idx = int(pairs_cpu[batch_idx, pair_idx, 0].item())
            dest_idx = int(pairs_cpu[batch_idx, pair_idx, 1].item())
            if (
                0 <= source_idx < max_sites
                and 0 <= dest_idx < max_sites
                and bool(mask_cpu[batch_idx, source_idx])
                and bool(mask_cpu[batch_idx, dest_idx])
            ):
                positives.add((source_idx, dest_idx))
        for source_idx in valid_sites:
            if int(types_cpu[batch_idx, source_idx].item()) != V_TYPE:
                continue
            for dest_idx in valid_sites:
                if source_idx == dest_idx:
                    continue
                dest_type = int(types_cpu[batch_idx, dest_idx].item())
                if dest_type not in (FE_TYPE, CU_TYPE):
                    continue
                pair = (int(source_idx), int(dest_idx))
                if pair not in positives:
                    valid_pairs.append(pair)
        for pair_idx in range(max_pairs):
            source_idx = int(pairs_cpu[batch_idx, pair_idx, 0].item())
            dest_idx = int(pairs_cpu[batch_idx, pair_idx, 1].item())
            fallback = (
                source_idx
                if 0 <= source_idx < max_sites and bool(mask_cpu[batch_idx, source_idx])
                else (int(valid_sites[0]) if valid_sites else 0)
            )
            if valid_pairs:
                for neg_idx in range(neg_count):
                    choice = valid_pairs[(pair_idx * neg_count + neg_idx + fallback) % len(valid_pairs)]
                    negative_cpu[batch_idx, pair_idx, neg_idx, 0] = int(choice[0])
                    negative_cpu[batch_idx, pair_idx, neg_idx, 1] = int(choice[1])
            else:
                negative_cpu[batch_idx, pair_idx, :, 0] = int(fallback)
                negative_cpu[batch_idx, pair_idx, :, 1] = int(dest_idx if 0 <= dest_idx < max_sites else fallback)
    return negative_cpu.to(device=vacancy_pair_indices.device, dtype=vacancy_pair_indices.dtype)


def _structured_terminal_vacancy_pair_negative_indices(
    vacancy_pair_indices: torch.Tensor,
    *,
    current_types: torch.Tensor,
    candidate_mask: torch.Tensor,
    count_per_group: int = 0,
) -> torch.Tensor:
    """Build v109 hard negatives for terminal vacancy-displacement pairs.

    For each true source-vacancy -> destination-atom pair, generate negatives
    that share the source, share the destination, or mix teacher sources and
    destinations into an unpaired false terminal vacancy displacement.
    """
    group_count = max(int(count_per_group), 0)
    if group_count <= 0:
        shape = (*vacancy_pair_indices.shape[:-1], 0, 2)
        return torch.empty(shape, dtype=vacancy_pair_indices.dtype, device=vacancy_pair_indices.device)
    total_count = 3 * group_count
    fallback = _dense_terminal_vacancy_pair_negative_indices(
        vacancy_pair_indices,
        current_types=current_types,
        candidate_mask=candidate_mask,
        count=total_count,
    )
    if vacancy_pair_indices.dim() != 3:
        return fallback

    pairs_cpu = vacancy_pair_indices.detach().cpu().long()
    negative_cpu = fallback.detach().cpu().long()
    mask_cpu = candidate_mask.detach().cpu() > 0.5
    types_cpu = current_types.detach().cpu().long()
    batch_size, max_pairs, _ = pairs_cpu.shape
    max_sites = int(types_cpu.shape[1])

    for batch_idx in range(batch_size):
        valid_sites = torch.nonzero(mask_cpu[batch_idx], as_tuple=False).flatten().tolist()
        vacancy_sources = [
            int(idx)
            for idx in valid_sites
            if int(types_cpu[batch_idx, int(idx)].item()) == V_TYPE
        ]
        atom_destinations = [
            int(idx)
            for idx in valid_sites
            if int(types_cpu[batch_idx, int(idx)].item()) in (FE_TYPE, CU_TYPE)
        ]
        positives: set[tuple[int, int]] = set()
        teacher_sources: list[int] = []
        teacher_destinations: list[int] = []
        for pair_idx in range(max_pairs):
            source_idx = int(pairs_cpu[batch_idx, pair_idx, 0].item())
            dest_idx = int(pairs_cpu[batch_idx, pair_idx, 1].item())
            if (
                0 <= source_idx < max_sites
                and 0 <= dest_idx < max_sites
                and bool(mask_cpu[batch_idx, source_idx])
                and bool(mask_cpu[batch_idx, dest_idx])
            ):
                positives.add((source_idx, dest_idx))
                if source_idx not in teacher_sources:
                    teacher_sources.append(source_idx)
                if dest_idx not in teacher_destinations:
                    teacher_destinations.append(dest_idx)

        for pair_idx in range(max_pairs):
            source_idx = int(pairs_cpu[batch_idx, pair_idx, 0].item())
            dest_idx = int(pairs_cpu[batch_idx, pair_idx, 1].item())
            if (
                source_idx < 0
                or dest_idx < 0
                or source_idx >= max_sites
                or dest_idx >= max_sites
                or not bool(mask_cpu[batch_idx, source_idx])
                or not bool(mask_cpu[batch_idx, dest_idx])
            ):
                continue
            same_source = [
                (source_idx, int(candidate_dest))
                for candidate_dest in atom_destinations
                if int(candidate_dest) != dest_idx and (source_idx, int(candidate_dest)) not in positives
            ]
            same_destination = [
                (int(candidate_source), dest_idx)
                for candidate_source in vacancy_sources
                if int(candidate_source) != source_idx and (int(candidate_source), dest_idx) not in positives
            ]
            teacher_unpaired = [
                (int(candidate_source), int(candidate_dest))
                for candidate_source in teacher_sources
                for candidate_dest in teacher_destinations
                if (int(candidate_source), int(candidate_dest)) not in positives
                and not (int(candidate_source) == source_idx and int(candidate_dest) == dest_idx)
            ]
            groups = [same_source, same_destination, teacher_unpaired]
            for group_idx, group in enumerate(groups):
                if not group:
                    continue
                for neg_idx in range(group_count):
                    choice = group[(pair_idx + neg_idx + source_idx + dest_idx) % len(group)]
                    out_idx = group_idx * group_count + neg_idx
                    negative_cpu[batch_idx, pair_idx, out_idx, 0] = int(choice[0])
                    negative_cpu[batch_idx, pair_idx, out_idx, 1] = int(choice[1])
    return negative_cpu.to(device=vacancy_pair_indices.device, dtype=vacancy_pair_indices.dtype)


def _terminal_vacancy_pair_negative_indices(
    vacancy_pair_indices: torch.Tensor,
    *,
    current_types: torch.Tensor,
    candidate_mask: torch.Tensor,
    dense_count: int = 1,
    structured_count: int = 0,
) -> torch.Tensor:
    dense = _dense_terminal_vacancy_pair_negative_indices(
        vacancy_pair_indices,
        current_types=current_types,
        candidate_mask=candidate_mask,
        count=max(int(dense_count), 1),
    )
    if int(structured_count) <= 0:
        return dense
    structured = _structured_terminal_vacancy_pair_negative_indices(
        vacancy_pair_indices,
        current_types=current_types,
        candidate_mask=candidate_mask,
        count_per_group=int(structured_count),
    )
    if structured.numel() == 0:
        return dense
    return torch.cat([dense, structured], dim=-2)


def _decode_action_edge_pair_logits(
    decode_fn,
    *,
    site_latent: torch.Tensor,
    patch_latent: torch.Tensor,
    predicted_next_global: torch.Tensor,
    path_latent: torch.Tensor,
    horizon_k: torch.Tensor,
    current_types: torch.Tensor,
    edge_pair_indices: torch.Tensor,
) -> torch.Tensor:
    if edge_pair_indices.dim() == 4:
        batch, pairs, neg_count, _ = edge_pair_indices.shape
        flat_indices = edge_pair_indices.reshape(batch, pairs * neg_count, 2)
        flat_logits = decode_fn(
            site_latent=site_latent,
            patch_latent=patch_latent,
            predicted_next_global=predicted_next_global,
            path_latent=path_latent,
            horizon_k=horizon_k,
            current_types=current_types,
            edge_pair_indices=flat_indices,
        )
        return flat_logits.reshape(batch, pairs, neg_count)
    return decode_fn(
        site_latent=site_latent,
        patch_latent=patch_latent,
        predicted_next_global=predicted_next_global,
        path_latent=path_latent,
        horizon_k=horizon_k,
        current_types=current_types,
        edge_pair_indices=edge_pair_indices,
    )


def _action_edge_pair_target_tensors(
    tensors: dict[str, torch.Tensor],
    target_source: str,
) -> dict[str, torch.Tensor]:
    source = str(target_source).strip().lower().replace("-", "_")
    if source in {"action", "action_edge", "teacher_action_edge", "teacher_action"}:
        return {
            "indices": tensors["teacher_action_edge_pair_indices"],
            "mask": tensors["teacher_action_edge_pair_mask"],
            "support_mask": tensors["teacher_action_edge_pair_support_mask"],
            "moving_type": tensors["teacher_action_edge_pair_moving_type"],
            "order": tensors["teacher_action_edge_pair_order"],
            "is_terminal_vacancy_pair": torch.tensor(False, device=tensors["candidate_mask"].device),
        }
    if source in {"vacancy_pair", "terminal_vacancy_pair", "terminal_vacancy_displacement"}:
        mask = tensors["teacher_vacancy_pair_mask"]
        return {
            "indices": tensors["teacher_vacancy_pair_indices"],
            "mask": mask,
            "support_mask": mask,
            "moving_type": tensors["teacher_vacancy_pair_moving_type"],
            "order": tensors["teacher_vacancy_pair_order"],
            "is_terminal_vacancy_pair": torch.tensor(True, device=tensors["candidate_mask"].device),
        }
    raise ValueError(
        "unknown action edge-pair target source "
        f"{target_source!r}; expected action or vacancy_pair"
    )


def _action_edge_pair_supervision_loss(
    positive_logits: torch.Tensor,
    negative_logits: torch.Tensor,
    edge_pair_mask: torch.Tensor,
    *,
    negative_weight: float = 1.0,
    rank_margin_weight: float = 0.25,
    margin: float = 0.5,
) -> dict[str, torch.Tensor]:
    valid = edge_pair_mask > 0.5
    if not torch.any(valid):
        zero = positive_logits.sum() * 0.0
        return {
            "loss": zero,
            "positive": zero,
            "negative": zero,
            "rank_margin": zero,
            "pos_prob": zero,
            "neg_prob": zero,
            "rank_acc": zero,
            "available_frac": zero,
        }
    pos = positive_logits[valid]
    neg = negative_logits[valid]
    pos_for_rank = pos.unsqueeze(-1) if neg.dim() > 1 else pos
    pos_loss = F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos))
    neg_loss = F.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg))
    rank = F.relu(float(margin) - (pos_for_rank - neg)).mean()
    loss = pos_loss + float(negative_weight) * neg_loss + float(rank_margin_weight) * rank
    return {
        "loss": loss,
        "positive": pos_loss,
        "negative": neg_loss,
        "rank_margin": rank,
        "pos_prob": torch.sigmoid(pos).mean(),
        "neg_prob": torch.sigmoid(neg).mean(),
        "rank_acc": (pos_for_rank > neg).float().mean(),
        "available_frac": valid.float().mean(),
    }


def _pair_listwise_contrastive_loss(
    positive_logits: torch.Tensor,
    negative_logits: torch.Tensor,
    pair_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    valid = pair_mask > 0.5
    if negative_logits.dim() < positive_logits.dim() + 1 or negative_logits.shape[-1] <= 0 or not torch.any(valid):
        zero = positive_logits.sum() * 0.0
        return {
            "loss": zero,
            "acc": zero,
            "available_frac": zero,
        }
    pos = positive_logits[valid].unsqueeze(-1)
    neg = negative_logits[valid]
    logits = torch.cat([pos, neg], dim=-1)
    targets = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    loss = F.cross_entropy(logits, targets)
    acc = (torch.argmax(logits, dim=-1) == 0).float().mean()
    return {
        "loss": loss,
        "acc": acc,
        "available_frac": valid.float().mean(),
    }


def _action_edge_pair_support_loss(
    pair_logits: torch.Tensor,
    negative_logits: torch.Tensor,
    edge_pair_mask: torch.Tensor,
    edge_pair_support_mask: torch.Tensor,
    *,
    negative_weight: float = 1.0,
    rank_margin_weight: float = 0.25,
    margin: float = 0.5,
) -> dict[str, torch.Tensor]:
    valid = edge_pair_mask > 0.5
    if not torch.any(valid):
        zero = pair_logits.sum() * 0.0
        return {
            "loss": zero,
            "pair_bce": zero,
            "negative": zero,
            "rank_margin": zero,
            "support_prob": zero,
            "nonsupport_prob": zero,
            "neg_prob": zero,
            "rank_acc": zero,
            "support_frac": zero,
            "available_frac": zero,
        }
    support_target = edge_pair_support_mask.float().clamp(0.0, 1.0)
    pair_loss = F.binary_cross_entropy_with_logits(pair_logits[valid], support_target[valid])
    neg_loss = F.binary_cross_entropy_with_logits(negative_logits[valid], torch.zeros_like(negative_logits[valid]))
    support_valid = valid & (support_target > 0.5)
    if torch.any(support_valid):
        pos = pair_logits[support_valid]
        neg = negative_logits[support_valid]
        pos_for_rank = pos.unsqueeze(-1) if neg.dim() > 1 else pos
        rank = F.relu(float(margin) - (pos_for_rank - neg)).mean()
        support_prob = torch.sigmoid(pos).mean()
        rank_acc = (pos_for_rank > neg).float().mean()
    else:
        rank = pair_logits.sum() * 0.0
        support_prob = pair_logits.sum() * 0.0
        rank_acc = pair_logits.sum() * 0.0
    nonsupport_valid = valid & (support_target <= 0.5)
    nonsupport_prob = (
        torch.sigmoid(pair_logits[nonsupport_valid]).mean()
        if torch.any(nonsupport_valid)
        else pair_logits.sum() * 0.0
    )
    loss = pair_loss + float(negative_weight) * neg_loss + float(rank_margin_weight) * rank
    return {
        "loss": loss,
        "pair_bce": pair_loss,
        "negative": neg_loss,
        "rank_margin": rank,
        "support_prob": support_prob,
        "nonsupport_prob": nonsupport_prob,
        "neg_prob": torch.sigmoid(negative_logits[valid]).mean(),
        "rank_acc": rank_acc,
        "support_frac": support_valid.float().mean(),
        "available_frac": valid.float().mean(),
    }


def _action_edge_pair_semantic_loss(
    moving_type_logits: torch.Tensor,
    order_logits: torch.Tensor,
    edge_pair_mask: torch.Tensor,
    edge_pair_moving_type: torch.Tensor,
    edge_pair_order: torch.Tensor,
) -> dict[str, torch.Tensor]:
    valid = edge_pair_mask > 0.5
    zero = moving_type_logits.sum() * 0.0 + order_logits.sum() * 0.0
    if not torch.any(valid):
        return {
            "loss": zero,
            "moving_type": zero,
            "order": zero,
            "moving_type_acc": zero,
            "order_mae": zero,
            "available_frac": zero,
        }

    moving_type_target = edge_pair_moving_type.long()
    moving_type_valid = valid & (moving_type_target >= 0) & (moving_type_target < NUM_SITE_TYPES)
    if torch.any(moving_type_valid):
        moving_type_loss = F.cross_entropy(
            moving_type_logits[moving_type_valid],
            moving_type_target[moving_type_valid],
        )
        moving_type_pred = moving_type_logits[moving_type_valid].argmax(dim=-1)
        moving_type_acc = (moving_type_pred == moving_type_target[moving_type_valid]).float().mean()
    else:
        moving_type_loss = zero
        moving_type_acc = zero

    order_target = edge_pair_order.float().clamp(min=0.0, max=1.0)
    if torch.any(valid):
        order_pred = torch.sigmoid(order_logits[valid])
        order_loss = F.smooth_l1_loss(order_pred, order_target[valid])
        order_mae = torch.abs(order_pred - order_target[valid]).mean()
    else:
        order_loss = zero
        order_mae = zero

    return {
        "loss": moving_type_loss + order_loss,
        "moving_type": moving_type_loss,
        "order": order_loss,
        "moving_type_acc": moving_type_acc,
        "order_mae": order_mae,
        "available_frac": valid.float().mean(),
    }


def _log_target_tau(tau: torch.Tensor) -> torch.Tensor:
    return torch.log(tau.clamp(min=1e-10))


def _scheduled_aux_scale(epoch: int, total_epochs: int, start_fraction: float = 0.55, end_scale: float = 0.1) -> float:
    if total_epochs <= 1:
        return 1.0
    progress = float(max(epoch - 1, 0)) / float(max(total_epochs - 1, 1))
    if progress <= start_fraction:
        return 1.0
    tail_progress = (progress - start_fraction) / max(1.0 - start_fraction, 1e-6)
    return float(1.0 - (1.0 - end_scale) * min(max(tail_progress, 0.0), 1.0))


def _scheduled_posterior_tau_scale(epoch: int, total_epochs: int, start_fraction: float = 0.2, end_scale: float = 0.25) -> float:
    if total_epochs <= 1:
        return 1.0
    progress = float(max(epoch - 1, 0)) / float(max(total_epochs - 1, 1))
    if progress <= start_fraction:
        return 1.0
    tail_progress = (progress - start_fraction) / max(1.0 - start_fraction, 1e-6)
    return float(1.0 - (1.0 - end_scale) * min(max(tail_progress, 0.0), 1.0))


def _compute_reward_diagnostics(pred_reward: np.ndarray, true_reward: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred_reward, dtype=np.float64)
    true = np.asarray(true_reward, dtype=np.float64)
    if pred.size == 0 or true.size == 0:
        return {
            "mean_bias": 0.0,
            "zero_target_frac": 0.0,
            "negative_target_frac": 0.0,
            "zero_pred_mean": 0.0,
            "zero_pred_mean_abs": 0.0,
            "negative_pred_mean": 0.0,
            "negative_bias": 0.0,
            "nonzero_sign_acc": 1.0,
        }

    zero_mask = np.isclose(true, 0.0, atol=1e-8)
    negative_mask = true < -1e-8
    nonzero_mask = np.abs(true) > 1e-8
    if np.any(nonzero_mask):
        nonzero_sign_acc = float(np.mean(np.sign(pred[nonzero_mask]) == np.sign(true[nonzero_mask])))
    else:
        nonzero_sign_acc = 1.0
    return {
        "mean_bias": float(np.mean(pred - true)),
        "zero_target_frac": float(np.mean(zero_mask)),
        "negative_target_frac": float(np.mean(negative_mask)),
        "zero_pred_mean": float(np.mean(pred[zero_mask])) if np.any(zero_mask) else 0.0,
        "zero_pred_mean_abs": float(np.mean(np.abs(pred[zero_mask]))) if np.any(zero_mask) else 0.0,
        "negative_pred_mean": float(np.mean(pred[negative_mask])) if np.any(negative_mask) else 0.0,
        "negative_bias": float(np.mean(pred[negative_mask] - true[negative_mask])) if np.any(negative_mask) else 0.0,
        "nonzero_sign_acc": nonzero_sign_acc,
    }


def _selection_score(val_metrics: dict[str, float], dataset_stats: dict[str, object], *, proj_l1_score_weight: float = 80.0) -> float:
    coverage_penalty = max(0.0, 0.9 - float(dataset_stats.get("val", {}).get("coverage", 0.0)))
    projected_global_l1 = float(val_metrics.get("projected_global_l1", 0.0))
    unchanged_vacancy_copy_acc = float(val_metrics.get("unchanged_vacancy_copy_acc", 1.0))
    reward_corr = val_metrics.get("reward_corr")
    reward_corr_penalty = 0.0
    if reward_corr is not None:
        reward_corr_penalty = 0.5 * max(0.0, 0.45 - float(reward_corr))
    reward_zero_penalty = 0.5 * float(val_metrics.get("reward_zero_pred_mean_abs", 0.0))
    reward_negative_penalty = 0.25 * max(0.0, float(val_metrics.get("reward_negative_pred_mean", 0.0)))
    reward_bias_penalty = 0.25 * abs(float(val_metrics.get("reward_mean_bias", 0.0)))
    reward_sign_penalty = 0.25 * max(0.0, 0.9 - float(val_metrics.get("reward_nonzero_sign_acc", 1.0)))
    return (
        val_metrics["tau_log_mae"]
        + 0.5 * val_metrics["reward_mae"]
        + reward_corr_penalty
        + reward_zero_penalty
        + reward_negative_penalty
        + reward_bias_penalty
        + reward_sign_penalty
        + 0.5 * max(0.0, 1.0 - val_metrics["change_topk_f1"])
        + 0.5 * max(0.0, 1.0 - val_metrics["projected_change_f1"])
        + max(0.0, 1.0 - val_metrics["projected_changed_type_acc"])
        + 0.5 * max(0.0, 1.0 - unchanged_vacancy_copy_acc)
        + proj_l1_score_weight * projected_global_l1
        + 2.0 * val_metrics["reachability_violation_rate"]
        + coverage_penalty
    )


def _predict_reward_and_duration_outputs(
    model: nn.Module,
    global_latent: torch.Tensor,
    predicted_next_global: torch.Tensor,
    path_latent: torch.Tensor,
    global_summary: torch.Tensor,
    horizon_k: torch.Tensor,
    *,
    detach_duration_inputs: bool = False,
    patch_latent: torch.Tensor | None = None,
    change_logits: torch.Tensor | None = None,
    type_logits: torch.Tensor | None = None,
    current_types: torch.Tensor | None = None,
    candidate_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if hasattr(model, "predict_reward_and_durations"):
        outputs = model.predict_reward_and_durations(
            global_latent,
            predicted_next_global,
            path_latent,
            global_summary,
            horizon_k,
            detach_duration_inputs=detach_duration_inputs,
            patch_latent=patch_latent,
            change_logits=change_logits,
            type_logits=type_logits,
            current_types=current_types,
            candidate_mask=candidate_mask,
        )
        realized_tau_available = bool(getattr(model, "realized_tau_head_loaded", True))
        if not realized_tau_available:
            outputs = dict(outputs)
            outputs["realized_tau_mu"] = outputs["expected_tau_mu"]
            outputs["realized_tau_log_sigma"] = outputs["expected_tau_log_sigma"]
        if "noop_risk_logit" not in outputs:
            outputs = dict(outputs)
            outputs["noop_risk_logit"] = torch.zeros_like(outputs["reward"])
            outputs["noop_risk_available"] = False
        else:
            outputs["noop_risk_available"] = True
        outputs["realized_tau_available"] = realized_tau_available
        return outputs
    try:
        reward, tau_mu, tau_log_sigma, gate_logit = model.predict_reward_and_duration(
            global_latent,
            predicted_next_global,
            path_latent,
            global_summary,
            horizon_k,
            detach_duration_inputs=detach_duration_inputs,
            patch_latent=patch_latent,
            change_logits=change_logits,
            type_logits=type_logits,
            current_types=current_types,
            candidate_mask=candidate_mask,
        )
    except TypeError:
        reward, tau_mu, tau_log_sigma, gate_logit = model.predict_reward_and_duration(
            global_latent,
            predicted_next_global,
            path_latent,
            global_summary,
            horizon_k,
        )
    return {
        "reward": reward,
        "expected_tau_mu": tau_mu,
        "expected_tau_log_sigma": tau_log_sigma,
        "realized_tau_mu": tau_mu,
        "realized_tau_log_sigma": tau_log_sigma,
        "gate_logit": gate_logit,
        "noop_risk_logit": torch.zeros_like(reward),
        "noop_risk_available": False,
        "realized_tau_available": False,
    }


def _select_reward_edit_context(
    reward_edit_context_source: str,
    change_logits: torch.Tensor | None,
    type_logits: torch.Tensor | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if reward_edit_context_source == "none":
        return None, None
    if reward_edit_context_source != "default":
        raise ValueError(f"unknown reward_edit_context_source: {reward_edit_context_source}")
    return change_logits, type_logits


def _reward_supervision_losses(
    reward_hat: torch.Tensor,
    gate_logit: torch.Tensor,
    target_reward: torch.Tensor,
    *,
    reward_magnitude_weight: float,
    reward_gated_weight: float = 1.0,
    reward_gate_weight: float = 0.25,
    reward_zero_weight: float = 0.5,
    reward_sign_weight: float = 0.25,
) -> dict[str, torch.Tensor]:
    reward_nonzero_target = (target_reward.abs() > 1e-6).float()
    gated_reward = reward_hat * torch.sigmoid(gate_logit)
    gate_loss = F.binary_cross_entropy_with_logits(gate_logit, reward_nonzero_target)
    reward_gated_loss = F.smooth_l1_loss(gated_reward, target_reward)
    nonzero_mask = reward_nonzero_target > 0.5
    zero_mask = ~nonzero_mask
    if nonzero_mask.any():
        reward_magnitude_loss = F.smooth_l1_loss(reward_hat[nonzero_mask], target_reward[nonzero_mask])
        reward_sign_target = (target_reward[nonzero_mask] > 0).float()
        reward_sign_loss = F.binary_cross_entropy_with_logits(reward_hat[nonzero_mask], reward_sign_target)
    else:
        reward_magnitude_loss = torch.zeros((), device=reward_hat.device)
        reward_sign_loss = torch.zeros((), device=reward_hat.device)
    if zero_mask.any():
        reward_zero_loss = F.smooth_l1_loss(reward_hat[zero_mask], target_reward[zero_mask])
    else:
        reward_zero_loss = torch.zeros((), device=reward_hat.device)
    loss = (
        float(reward_gated_weight) * reward_gated_loss
        + float(reward_gate_weight) * gate_loss
        + 0.5 * reward_magnitude_weight * reward_magnitude_loss
        + float(reward_zero_weight) * reward_zero_loss
        + float(reward_sign_weight) * reward_sign_loss
    )
    return {
        "loss": loss,
        "gated_reward": gated_reward,
        "gate_loss": gate_loss,
        "magnitude_loss": reward_magnitude_loss,
        "zero_loss": reward_zero_loss,
        "sign_loss": reward_sign_loss,
    }


def _noop_risk_supervision_loss(
    noop_risk_logit: torch.Tensor,
    changed_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    target = (changed_mask.sum(dim=-1) <= 0.0).float()
    pos = target.sum()
    neg = target.numel() - pos
    if pos > 0:
        pos_weight = (neg / pos.clamp(min=1.0)).detach().clamp(min=1.0, max=20.0)
        loss = F.binary_cross_entropy_with_logits(noop_risk_logit, target, pos_weight=pos_weight)
    else:
        loss = F.binary_cross_entropy_with_logits(noop_risk_logit, target)
    prob = torch.sigmoid(noop_risk_logit.detach())
    if (target > 0.5).any():
        noop_prob = prob[target > 0.5].mean()
    else:
        noop_prob = torch.zeros((), device=noop_risk_logit.device)
    if (target <= 0.5).any():
        nonnoop_prob = prob[target <= 0.5].mean()
    else:
        nonnoop_prob = torch.zeros((), device=noop_risk_logit.device)
    return {
        "loss": loss,
        "target_frac": target.mean(),
        "noop_prob": noop_prob,
        "nonnoop_prob": nonnoop_prob,
    }


def _proposal_support_loss(
    proposal_logits: torch.Tensor,
    changed_mask: torch.Tensor,
    candidate_mask: torch.Tensor,
    hard_negative_mask: torch.Tensor | None = None,
    hard_negative_weight: float = 0.0,
    rank_margin_weight: float = 0.0,
    rank_margin: float = 1.0,
    candidate_positive_mask: torch.Tensor | None = None,
    candidate_false_positive_mask: torch.Tensor | None = None,
    candidate_positive_weight: float = 0.0,
    candidate_false_positive_weight: float = 0.0,
    candidate_rank_margin_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    valid = candidate_mask > 0
    if not valid.any():
        zero = torch.zeros((), device=proposal_logits.device)
        return {
            "loss": zero,
            "bce": zero,
            "focal": zero,
            "hard_negative": zero,
            "rank_margin": zero,
            "candidate_positive": zero,
            "candidate_false_positive": zero,
            "candidate_rank_margin": zero,
            "topk_f1": zero,
            "recall32": zero,
        }
    target = changed_mask.float()
    pos_count = target[valid].sum().clamp(min=1.0)
    neg_count = valid.float().sum().clamp(min=1.0) - pos_count + 1e-6
    pos_weight = (neg_count / pos_count).detach().clamp(min=1.0, max=200.0)
    bce = F.binary_cross_entropy_with_logits(proposal_logits[valid], target[valid], pos_weight=pos_weight)
    focal = _focal_bce_with_logits(proposal_logits[valid], target[valid])
    hard_negative_loss = torch.zeros((), device=proposal_logits.device)
    rank_loss = torch.zeros((), device=proposal_logits.device)
    candidate_positive_loss = torch.zeros((), device=proposal_logits.device)
    candidate_false_positive_loss = torch.zeros((), device=proposal_logits.device)
    candidate_rank_loss = torch.zeros((), device=proposal_logits.device)
    if hard_negative_mask is not None:
        hard_negative = (hard_negative_mask > 0.5) & valid & (target <= 0.5)
        if hard_negative.any():
            hard_negative_loss = F.binary_cross_entropy_with_logits(
                proposal_logits[hard_negative],
                torch.zeros_like(proposal_logits[hard_negative]),
            )
        if float(rank_margin_weight) > 0.0:
            sample_losses: list[torch.Tensor] = []
            for sample_idx in range(proposal_logits.shape[0]):
                pos_idx = (target[sample_idx] > 0.5) & valid[sample_idx]
                neg_idx = hard_negative[sample_idx]
                if pos_idx.any() and neg_idx.any():
                    pos_score = proposal_logits[sample_idx][pos_idx].mean()
                    neg_score = proposal_logits[sample_idx][neg_idx].mean()
                    sample_losses.append(F.relu(torch.tensor(float(rank_margin), device=proposal_logits.device) - pos_score + neg_score))
            if sample_losses:
                rank_loss = torch.stack(sample_losses).mean()
    if candidate_positive_mask is not None or candidate_false_positive_mask is not None:
        aux_positive = torch.zeros_like(target, dtype=torch.bool)
        aux_negative = torch.zeros_like(target, dtype=torch.bool)
        if candidate_positive_mask is not None:
            aux_positive = (candidate_positive_mask > 0.5) & valid
        if candidate_false_positive_mask is not None:
            aux_negative = (candidate_false_positive_mask > 0.5) & valid & (~aux_positive) & (target <= 0.5)
        if aux_positive.any():
            candidate_positive_loss = F.binary_cross_entropy_with_logits(
                proposal_logits[aux_positive],
                torch.ones_like(proposal_logits[aux_positive]),
            )
        if aux_negative.any():
            candidate_false_positive_loss = F.binary_cross_entropy_with_logits(
                proposal_logits[aux_negative],
                torch.zeros_like(proposal_logits[aux_negative]),
            )
        if float(candidate_rank_margin_weight) > 0.0:
            candidate_rank_losses: list[torch.Tensor] = []
            for sample_idx in range(proposal_logits.shape[0]):
                pos_idx = aux_positive[sample_idx]
                neg_idx = aux_negative[sample_idx]
                if pos_idx.any() and neg_idx.any():
                    pos_score = proposal_logits[sample_idx][pos_idx].mean()
                    neg_score = proposal_logits[sample_idx][neg_idx].mean()
                    candidate_rank_losses.append(
                        F.relu(torch.tensor(float(rank_margin), device=proposal_logits.device) - pos_score + neg_score)
                    )
            if candidate_rank_losses:
                candidate_rank_loss = torch.stack(candidate_rank_losses).mean()
    loss = (
        bce
        + 0.25 * focal
        + float(hard_negative_weight) * hard_negative_loss
        + float(rank_margin_weight) * rank_loss
        + float(candidate_positive_weight) * candidate_positive_loss
        + float(candidate_false_positive_weight) * candidate_false_positive_loss
        + float(candidate_rank_margin_weight) * candidate_rank_loss
    )

    probs = torch.sigmoid(proposal_logits.detach()) * candidate_mask
    topk_f1_scores: list[float] = []
    recall32_scores: list[float] = []
    for sample_idx in range(probs.shape[0]):
        valid_idx = torch.nonzero(valid[sample_idx], as_tuple=False).squeeze(-1)
        target_idx = torch.nonzero((target[sample_idx] > 0.5) & valid[sample_idx], as_tuple=False).squeeze(-1)
        if valid_idx.numel() == 0:
            continue
        if target_idx.numel() == 0:
            topk_f1_scores.append(1.0 if torch.sigmoid(proposal_logits[sample_idx, valid_idx]).max().item() < 0.5 else 0.0)
            recall32_scores.append(1.0)
            continue
        k = int(target_idx.numel())
        topk = min(k, int(valid_idx.numel()))
        selected = valid_idx[torch.topk(probs[sample_idx, valid_idx], k=topk).indices]
        hit = len(set(selected.detach().cpu().tolist()) & set(target_idx.detach().cpu().tolist()))
        precision = hit / max(topk, 1)
        recall = hit / max(k, 1)
        topk_f1_scores.append(2.0 * precision * recall / max(precision + recall, 1e-12))
        top32 = min(32, int(valid_idx.numel()))
        selected32 = valid_idx[torch.topk(probs[sample_idx, valid_idx], k=top32).indices]
        hit32 = len(set(selected32.detach().cpu().tolist()) & set(target_idx.detach().cpu().tolist()))
        recall32_scores.append(hit32 / max(k, 1))
    topk_f1 = torch.tensor(float(np.mean(topk_f1_scores)) if topk_f1_scores else 0.0, device=proposal_logits.device)
    recall32 = torch.tensor(float(np.mean(recall32_scores)) if recall32_scores else 0.0, device=proposal_logits.device)
    return {
        "loss": loss,
        "bce": bce,
        "focal": focal,
        "hard_negative": hard_negative_loss,
        "rank_margin": rank_loss,
        "candidate_positive": candidate_positive_loss,
        "candidate_false_positive": candidate_false_positive_loss,
        "candidate_rank_margin": candidate_rank_loss,
        "topk_f1": topk_f1,
        "recall32": recall32,
    }


def _candidate_quality_loss(
    quality_logit: torch.Tensor,
    target: torch.Tensor,
    available: torch.Tensor,
) -> dict[str, torch.Tensor]:
    valid = available > 0.5
    zero = torch.zeros((), device=quality_logit.device)
    if not valid.any():
        return {
            "loss": zero,
            "mae": zero,
            "corr": zero,
            "pred_mean": zero,
            "target_mean": zero,
            "available_frac": available.float().mean() if available.numel() else zero,
        }
    pred = torch.sigmoid(quality_logit[valid])
    clipped_target = target[valid].float().clamp(min=0.0, max=1.0)
    loss = F.smooth_l1_loss(pred, clipped_target)
    mae = torch.abs(pred - clipped_target).mean()
    if pred.numel() > 1 and float(torch.std(pred).item()) > 0.0 and float(torch.std(clipped_target).item()) > 0.0:
        corr = torch.corrcoef(torch.stack([pred, clipped_target]))[0, 1].clamp(min=-1.0, max=1.0)
    else:
        corr = zero
    return {
        "loss": loss,
        "mae": mae,
        "corr": corr,
        "pred_mean": pred.mean(),
        "target_mean": clipped_target.mean(),
        "available_frac": available.float().mean(),
    }


def _soft_typed_change_count(
    change_logits: torch.Tensor,
    type_logits: torch.Tensor,
    current_types: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    type_probs = F.softmax(type_logits, dim=-1)
    current_copy_prob = type_probs.gather(-1, current_types.unsqueeze(-1)).squeeze(-1)
    typed_change_mass = torch.sigmoid(change_logits) * (1.0 - current_copy_prob)
    return (typed_change_mass * candidate_mask).sum(dim=-1)


def _soft_directional_transition_counts(
    change_logits: torch.Tensor,
    type_logits: torch.Tensor,
    current_types: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    change_prob = torch.sigmoid(change_logits) * candidate_mask
    type_probs = F.softmax(type_logits, dim=-1)
    vacancy_mask = (current_types == V_TYPE).float() * candidate_mask
    fe_mask = (current_types == FE_TYPE).float() * candidate_mask
    cu_mask = (current_types == CU_TYPE).float() * candidate_mask
    return {
        "vac_to_fe": (change_prob * type_probs[..., FE_TYPE] * vacancy_mask).sum(dim=-1),
        "vac_to_cu": (change_prob * type_probs[..., CU_TYPE] * vacancy_mask).sum(dim=-1),
        "fe_to_vac": (change_prob * type_probs[..., V_TYPE] * fe_mask).sum(dim=-1),
        "cu_to_vac": (change_prob * type_probs[..., V_TYPE] * cu_mask).sum(dim=-1),
    }


def _target_directional_transition_counts(
    current_types: torch.Tensor,
    target_types: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    valid = candidate_mask > 0
    return {
        "vac_to_fe": ((current_types == V_TYPE) & (target_types == FE_TYPE) & valid).float().sum(dim=-1),
        "vac_to_cu": ((current_types == V_TYPE) & (target_types == CU_TYPE) & valid).float().sum(dim=-1),
        "fe_to_vac": ((current_types == FE_TYPE) & (target_types == V_TYPE) & valid).float().sum(dim=-1),
        "cu_to_vac": ((current_types == CU_TYPE) & (target_types == V_TYPE) & valid).float().sum(dim=-1),
    }


def _matched_pair_count_loss(
    change_logits: torch.Tensor,
    type_logits: torch.Tensor,
    current_types: torch.Tensor,
    target_types: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    pred_counts = _soft_directional_transition_counts(
        change_logits=change_logits,
        type_logits=type_logits,
        current_types=current_types,
        candidate_mask=candidate_mask,
    )
    target_counts = _target_directional_transition_counts(
        current_types=current_types,
        target_types=target_types,
        candidate_mask=candidate_mask,
    )
    vac_to_atom_loss = 0.5 * (
        F.smooth_l1_loss(pred_counts["vac_to_fe"], target_counts["vac_to_fe"])
        + F.smooth_l1_loss(pred_counts["vac_to_cu"], target_counts["vac_to_cu"])
    )
    atom_to_vac_loss = 0.5 * (
        F.smooth_l1_loss(pred_counts["fe_to_vac"], target_counts["fe_to_vac"])
        + F.smooth_l1_loss(pred_counts["cu_to_vac"], target_counts["cu_to_vac"])
    )
    pred_pair_count = torch.minimum(pred_counts["vac_to_fe"], pred_counts["fe_to_vac"]) + torch.minimum(
        pred_counts["vac_to_cu"], pred_counts["cu_to_vac"]
    )
    target_pair_count = torch.minimum(target_counts["vac_to_fe"], target_counts["fe_to_vac"]) + torch.minimum(
        target_counts["vac_to_cu"], target_counts["cu_to_vac"]
    )
    matched_pair_loss = F.smooth_l1_loss(pred_pair_count, target_pair_count)
    return 0.35 * vac_to_atom_loss + 0.65 * atom_to_vac_loss + 0.5 * matched_pair_loss


def _masked_type_cross_entropy(type_logits: torch.Tensor, target_types: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.any():
        return F.cross_entropy(type_logits[mask], target_types[mask])
    return torch.zeros((), device=type_logits.device)


def _edit_supervision_losses(
    *,
    change_logits: torch.Tensor,
    type_logits: torch.Tensor,
    current_types: torch.Tensor,
    target_types: torch.Tensor,
    changed_mask: torch.Tensor,
    candidate_mask: torch.Tensor,
    aux_scale: float,
    sparsity_weight: float = 0.0,
    count_loss_weight: float = 0.1,
    noop_change_weight: float = 0.0,
    noop_type_copy_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    device = change_logits.device
    valid = candidate_mask > 0
    changed_valid = valid & (changed_mask > 0)
    unchanged_valid = valid & (changed_mask <= 0)
    changed_atom_valid = changed_valid & (current_types != V_TYPE)
    changed_vacancy_valid = changed_valid & (current_types == V_TYPE)
    unchanged_atom_valid = unchanged_valid & (current_types != V_TYPE)
    unchanged_vacancy_valid = unchanged_valid & (current_types == V_TYPE)
    atom_to_vac_valid = changed_valid & (current_types != V_TYPE) & (target_types == V_TYPE)
    vac_to_atom_valid = changed_valid & (current_types == V_TYPE) & (target_types != V_TYPE)
    noop_sample = changed_mask.sum(dim=-1) <= 0.0
    noop_valid = valid & noop_sample.unsqueeze(-1)

    pos_count = changed_mask[valid].sum().clamp(min=1.0)
    neg_count = valid.float().sum().clamp(min=1.0) - pos_count + 1e-6
    pos_weight = (neg_count / pos_count).detach()
    mask_bce = F.binary_cross_entropy_with_logits(change_logits[valid], changed_mask[valid], pos_weight=pos_weight)
    mask_focal = _focal_bce_with_logits(change_logits[valid], changed_mask[valid])

    if changed_atom_valid.any():
        atom_change_loss = F.binary_cross_entropy_with_logits(
            change_logits[changed_atom_valid],
            torch.ones_like(change_logits[changed_atom_valid]),
        )
    else:
        atom_change_loss = torch.zeros((), device=device)
    if changed_vacancy_valid.any():
        vacancy_change_loss = F.binary_cross_entropy_with_logits(
            change_logits[changed_vacancy_valid],
            torch.ones_like(change_logits[changed_vacancy_valid]),
        )
    else:
        vacancy_change_loss = torch.zeros((), device=device)
    if unchanged_vacancy_valid.any():
        vacancy_static_loss = F.binary_cross_entropy_with_logits(
            change_logits[unchanged_vacancy_valid],
            torch.zeros_like(change_logits[unchanged_vacancy_valid]),
        )
    else:
        vacancy_static_loss = torch.zeros((), device=device)

    predicted_change_count = _soft_typed_change_count(
        change_logits=change_logits,
        type_logits=type_logits,
        current_types=current_types,
        candidate_mask=candidate_mask,
    )
    target_change_count = changed_mask.sum(dim=-1)
    count_loss = F.smooth_l1_loss(predicted_change_count, target_change_count)
    pair_count_loss = _matched_pair_count_loss(
        change_logits=change_logits,
        type_logits=type_logits,
        current_types=current_types,
        target_types=target_types,
        candidate_mask=candidate_mask,
    )
    if sparsity_weight > 0 and unchanged_valid.any():
        sparsity_loss = torch.sigmoid(change_logits[unchanged_valid]).mean()
    else:
        sparsity_loss = torch.zeros((), device=device)
    if noop_change_weight > 0 and noop_valid.any():
        noop_change_loss = torch.sigmoid(change_logits[noop_valid]).mean()
    else:
        noop_change_loss = torch.zeros((), device=device)
    mask_loss = (
        mask_bce
        + 0.25 * mask_focal
        + aux_scale * (0.4 * atom_change_loss + 0.4 * vacancy_change_loss + 0.2 * vacancy_static_loss)
        + count_loss_weight * count_loss
        + sparsity_weight * sparsity_loss
        + noop_change_weight * noop_change_loss
    )

    changed_type_loss = _masked_type_cross_entropy(type_logits, target_types, changed_valid)
    atom_to_vac_type_loss = _masked_type_cross_entropy(type_logits, target_types, atom_to_vac_valid)
    vac_to_atom_type_loss = _masked_type_cross_entropy(type_logits, target_types, vac_to_atom_valid)
    unchanged_copy_loss = _masked_type_cross_entropy(type_logits, current_types, unchanged_atom_valid)
    vacancy_type_static_loss = _masked_type_cross_entropy(type_logits, current_types, unchanged_vacancy_valid)
    noop_type_copy_loss = _masked_type_cross_entropy(type_logits, current_types, noop_valid)
    type_loss = (
        changed_type_loss
        + 0.5 * atom_to_vac_type_loss
        + 0.5 * vac_to_atom_type_loss
        + 0.05 * unchanged_copy_loss
        + 0.25 * vacancy_type_static_loss
        + noop_type_copy_weight * noop_type_copy_loss
    )

    return {
        "mask": mask_loss,
        "count": count_loss,
        "pair": pair_count_loss,
        "type": type_loss,
        "noop_change": noop_change_loss,
        "noop_type_copy": noop_type_copy_loss,
        "atom_to_vac_type": atom_to_vac_type_loss,
        "vac_to_atom_type": vac_to_atom_type_loss,
    }


def _projected_mask_distill_loss(
    change_logits: torch.Tensor,
    projected_changed_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    reachability_violation: torch.Tensor,
    target_changed_mask: torch.Tensor | None = None,
    projected_noop_fp_weight: float = 0.0,
) -> torch.Tensor:
    reachable = reachability_violation.unsqueeze(-1) <= 0
    positive_mask = valid_mask & (projected_changed_mask > 0) & reachable
    if target_changed_mask is not None:
        noop_sample = target_changed_mask.sum(dim=-1) <= 0.0
        positive_mask = positive_mask & (~noop_sample).unsqueeze(-1)
    if positive_mask.any():
        positive_loss = F.binary_cross_entropy_with_logits(change_logits[positive_mask], projected_changed_mask[positive_mask])
    else:
        positive_loss = torch.zeros((), device=change_logits.device)

    if target_changed_mask is None or projected_noop_fp_weight <= 0.0:
        return positive_loss
    noop_sample = target_changed_mask.sum(dim=-1) <= 0.0
    false_positive_mask = valid_mask & noop_sample.unsqueeze(-1) & (projected_changed_mask > 0) & reachable
    if false_positive_mask.any():
        false_positive_loss = F.binary_cross_entropy_with_logits(
            change_logits[false_positive_mask],
            torch.zeros_like(change_logits[false_positive_mask]),
        )
    else:
        false_positive_loss = torch.zeros((), device=change_logits.device)
    return positive_loss + float(projected_noop_fp_weight) * false_positive_loss


def _projected_state_alignment_loss(
    projected_patch_latent: torch.Tensor,
    target_patch_latent: torch.Tensor,
    projected_global: torch.Tensor,
    next_global: torch.Tensor,
    next_pred: torch.Tensor,
    projected_changed_mask: torch.Tensor,
    reachability_violation: torch.Tensor,
) -> torch.Tensor:
    has_projected_edit = projected_changed_mask.sum(dim=-1) > 0
    success_mask = (reachability_violation <= 0) & has_projected_edit
    if success_mask.any():
        return (
            F.smooth_l1_loss(projected_patch_latent[success_mask], target_patch_latent[success_mask])
            + 0.5 * F.smooth_l1_loss(projected_global[success_mask], next_global[success_mask])
            + 0.5 * F.smooth_l1_loss(projected_global[success_mask], next_pred[success_mask])
        )
    return torch.zeros((), device=projected_patch_latent.device)


def _validate_resume_args(args: argparse.Namespace, ckpt_args: dict[str, object]) -> None:
    resume_segment_ks = _segment_ks_from_ckpt_args(ckpt_args)
    current_segment_ks = _segment_ks_from_args(args)
    if resume_segment_ks != current_segment_ks:
        raise ValueError(
            f"Resume checkpoint segment_ks={resume_segment_ks} does not match current segment_ks={current_segment_ks}"
        )
    resume_summary_mode = ckpt_args.get("teacher_path_summary_mode")
    if resume_summary_mode is not None and str(resume_summary_mode) != str(args.teacher_path_summary_mode):
        raise ValueError(
            "Resume checkpoint teacher_path_summary_mode="
            f"{resume_summary_mode} does not match current teacher_path_summary_mode={args.teacher_path_summary_mode}"
        )


def _resize_tensor_prefix(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    resized = torch.zeros_like(target)
    slices = tuple(slice(0, min(source.shape[dim], target.shape[dim])) for dim in range(source.ndim))
    resized[slices] = source[slices].to(dtype=target.dtype, device=target.device)
    return resized


def _load_model_weights(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    *,
    allow_path_posterior_resize: bool = False,
) -> tuple[list[str], list[str], list[str], list[str]]:
    model_state = model.state_dict()
    load_state: dict[str, torch.Tensor] = {}
    unexpected: list[str] = []
    resized: list[str] = []
    skipped: list[str] = []
    for key, value in state_dict.items():
        if key not in model_state:
            unexpected.append(key)
            continue
        target = model_state[key]
        if tuple(value.shape) == tuple(target.shape):
            load_state[key] = value
        elif allow_path_posterior_resize and (key.startswith("path_posterior.") or key == "k_embed.weight"):
            load_state[key] = _resize_tensor_prefix(value, target)
            resized.append(key)
        else:
            skipped.append(key)
    missing, extra_unexpected = model.load_state_dict(load_state, strict=False)
    unexpected.extend(extra_unexpected)
    return list(missing), unexpected, resized, skipped


def _initialize_best_score_from_saved_best(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    max_changed_sites: int,
    dataset_stats: dict[str, object],
    save_dir: Path,
    checkpoint_best_score: Optional[float] = None,
    allow_checkpoint_best_score_fallback: bool = True,
    proj_l1_score_weight: float = 80.0,
    reward_prediction_source: str = "raw",
    reward_edit_context_source: str = "default",
    proposal_target_source: str = "changed",
    action_support_target_source: str = "touched",
    terminal_edit_action_context_source: str = "action_endpoint",
) -> tuple[float, str]:
    current_state = copy.deepcopy(model.state_dict())
    source = "resume checkpoint"
    best_model_path = save_dir / "best_model.pt"
    if best_model_path.exists():
        best_ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
        try:
            model.load_state_dict(best_ckpt["model"])
            source = "saved best model"
        except RuntimeError:
            source = "resume checkpoint (skipped incompatible saved best model)"
    best_metrics = _evaluate(
        model,
        loader,
        device,
        max_changed_sites,
        reward_prediction_source=reward_prediction_source,
        reward_edit_context_source=reward_edit_context_source,
        proposal_target_source=proposal_target_source,
        action_support_target_source=action_support_target_source,
        terminal_edit_action_context_source=terminal_edit_action_context_source,
    )
    best_score = _selection_score(best_metrics, dataset_stats, proj_l1_score_weight=proj_l1_score_weight)
    model.load_state_dict(current_state)
    if source.startswith("resume checkpoint") and checkpoint_best_score is not None and allow_checkpoint_best_score_fallback:
        best_score = min(best_score, float(checkpoint_best_score))
        source = f"{source} + stored best_score"
    return best_score, source


def _evaluate(
    model: MacroDreamerEditModel,
    loader: DataLoader,
    device: str,
    max_changed_sites: int,
    reward_prediction_source: str = "raw",
    reward_edit_context_source: str = "default",
    proposal_target_source: str = "changed",
    action_support_target_source: str = "touched",
    terminal_edit_action_context_source: str = "action_endpoint",
    action_edge_pair_negative_mode: str = "self",
    action_edge_pair_negative_count: int = 1,
    action_edge_pair_dense_negative_count: int = 0,
    action_edge_pair_target_source: str = "action",
    vacancy_pair_negative_count: int = 1,
    vacancy_pair_structured_negative_count: int = 0,
) -> dict[str, float]:
    model.eval()
    reward_pred = []
    reward_true = []
    noop_risk_pred = []
    noop_risk_true = []
    tau_pred = []
    tau_true = []
    realized_tau_mu_pred = []
    realized_tau_log_sigma_pred = []
    realized_tau_true = []
    realized_tau_available = True
    changed_f1_scores = []
    change_topk_f1_scores = []
    changed_type_acc_scores = []
    projected_change_f1_scores = []
    projected_changed_type_acc_scores = []
    proposal_topk_f1_scores = []
    proposal_recall32_scores = []
    action_support_topk_f1_scores = []
    action_support_recall32_scores = []
    action_source_topk_f1_scores = []
    action_source_recall32_scores = []
    action_destination_topk_f1_scores = []
    action_destination_recall32_scores = []
    action_endpoint_topk_f1_scores = []
    action_endpoint_recall32_scores = []
    terminal_edit_topk_f1_scores = []
    terminal_edit_recall32_scores = []
    terminal_typed_diff_type_acc_scores = []
    terminal_typed_diff_copy_acc_scores = []
    terminal_typed_diff_topk_f1_scores = []
    terminal_typed_diff_recall32_scores = []
    action_edge_pair_rank_acc_scores = []
    action_edge_pair_pos_prob_scores = []
    action_edge_pair_neg_prob_scores = []
    action_edge_pair_available_scores = []
    action_edge_pair_support_rank_acc_scores = []
    action_edge_pair_support_prob_scores = []
    action_edge_pair_support_nonsupport_prob_scores = []
    action_edge_pair_support_neg_prob_scores = []
    action_edge_pair_support_frac_scores = []
    action_edge_pair_moving_type_acc_scores = []
    action_edge_pair_order_mae_scores = []
    vacancy_pair_rank_acc_scores = []
    vacancy_pair_pos_prob_scores = []
    vacancy_pair_neg_prob_scores = []
    vacancy_pair_available_scores = []
    vacancy_pair_listwise_loss_scores = []
    vacancy_pair_listwise_acc_scores = []
    vacancy_pair_moving_type_acc_scores = []
    vacancy_pair_order_mae_scores = []
    candidate_quality_pred = []
    candidate_quality_true = []
    unchanged_copy_acc_scores = []
    unchanged_atom_copy_acc_scores = []
    unchanged_vacancy_copy_acc_scores = []
    raw_vac_to_fe_counts = []
    raw_fe_to_vac_counts = []
    raw_vac_to_cu_counts = []
    raw_cu_to_vac_counts = []
    raw_matched_pair_counts = []
    latent_losses = []
    projected_global_losses = []
    reachability_violations = []
    transport_costs = []
    with torch.no_grad():
        for batch in loader:
            tensors = _batch_to_device(batch, device)
            global_latent = model.encode_global(tensors["start_obs"])
            next_global = model.encode_global(tensors["next_obs"])
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
            if hasattr(model, "decode_proposal"):
                proposal_logits = model.decode_proposal(
                    site_latent=site_latent,
                    patch_latent=patch_latent,
                    predicted_next_global=next_pred,
                    path_latent=path_latent,
                    horizon_k=tensors["horizon_k"],
                    current_types=tensors["current_types"],
                )
            else:
                proposal_logits = change_logits
            if hasattr(model, "decode_action_support"):
                action_support_logits = model.decode_action_support(
                    site_latent=site_latent,
                    patch_latent=patch_latent,
                    predicted_next_global=next_pred,
                    path_latent=path_latent,
                    horizon_k=tensors["horizon_k"],
                    current_types=tensors["current_types"],
                )
            else:
                action_support_logits = proposal_logits
            if hasattr(model, "decode_action_source_support"):
                action_source_logits = model.decode_action_source_support(
                    site_latent=site_latent,
                    patch_latent=patch_latent,
                    predicted_next_global=next_pred,
                    path_latent=path_latent,
                    horizon_k=tensors["horizon_k"],
                    current_types=tensors["current_types"],
                )
            else:
                action_source_logits = action_support_logits
            if hasattr(model, "decode_action_destination_support"):
                action_destination_logits = model.decode_action_destination_support(
                    site_latent=site_latent,
                    patch_latent=patch_latent,
                    predicted_next_global=next_pred,
                    path_latent=path_latent,
                    horizon_k=tensors["horizon_k"],
                    current_types=tensors["current_types"],
                )
            else:
                action_destination_logits = action_support_logits
            action_endpoint_logits = combine_action_endpoint_logits(action_source_logits, action_destination_logits)
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
                quality_available = tensors["planner_candidate_quality_available"] > 0.5
                if quality_available.any():
                    candidate_quality_pred.extend(
                        torch.sigmoid(candidate_quality_logit[quality_available]).cpu().numpy().tolist()
                    )
                    candidate_quality_true.extend(
                        tensors["planner_candidate_quality_target"][quality_available].cpu().numpy().tolist()
                    )
            proposal_target = _proposal_target_from_tensors(tensors, proposal_target_source)
            proposal_terms = _proposal_support_loss(proposal_logits, proposal_target, tensors["candidate_mask"])
            proposal_topk_f1_scores.append(float(proposal_terms["topk_f1"].item()))
            proposal_recall32_scores.append(float(proposal_terms["recall32"].item()))
            action_support_target = _proposal_target_from_tensors(tensors, action_support_target_source)
            action_support_terms = _proposal_support_loss(
                action_support_logits,
                action_support_target,
                tensors["candidate_mask"],
            )
            action_support_topk_f1_scores.append(float(action_support_terms["topk_f1"].item()))
            action_support_recall32_scores.append(float(action_support_terms["recall32"].item()))
            action_source_terms = _proposal_support_loss(
                action_source_logits,
                _proposal_target_from_tensors(tensors, "action_source"),
                tensors["candidate_mask"],
            )
            action_source_topk_f1_scores.append(float(action_source_terms["topk_f1"].item()))
            action_source_recall32_scores.append(float(action_source_terms["recall32"].item()))
            action_destination_terms = _proposal_support_loss(
                action_destination_logits,
                _proposal_target_from_tensors(tensors, "action_destination"),
                tensors["candidate_mask"],
            )
            action_destination_topk_f1_scores.append(float(action_destination_terms["topk_f1"].item()))
            action_destination_recall32_scores.append(float(action_destination_terms["recall32"].item()))
            action_endpoint_terms = _proposal_support_loss(
                action_endpoint_logits,
                _proposal_target_from_tensors(tensors, "action_endpoint"),
                tensors["candidate_mask"],
            )
            action_endpoint_topk_f1_scores.append(float(action_endpoint_terms["topk_f1"].item()))
            action_endpoint_recall32_scores.append(float(action_endpoint_terms["recall32"].item()))
            if hasattr(model, "decode_terminal_edit_support"):
                terminal_action_context_logits = _terminal_action_context_logits_from_tensors(
                    tensors,
                    terminal_edit_action_context_source,
                    action_endpoint_logits,
                )
                terminal_edit_logits = model.decode_terminal_edit_support(
                    site_latent=site_latent,
                    patch_latent=patch_latent,
                    predicted_next_global=next_pred,
                    path_latent=path_latent,
                    horizon_k=tensors["horizon_k"],
                    current_types=tensors["current_types"],
                    action_sequence_logits=terminal_action_context_logits,
                )
                terminal_edit_terms = _proposal_support_loss(
                    terminal_edit_logits,
                    _proposal_target_from_tensors(tensors, "changed"),
                    tensors["candidate_mask"],
                )
                terminal_edit_topk_f1_scores.append(float(terminal_edit_terms["topk_f1"].item()))
                terminal_edit_recall32_scores.append(float(terminal_edit_terms["recall32"].item()))
                if hasattr(model, "decode_terminal_typed_diff"):
                    terminal_typed_logits = model.decode_terminal_typed_diff(
                        site_latent=site_latent,
                        patch_latent=patch_latent,
                        predicted_next_global=next_pred,
                        path_latent=path_latent,
                        horizon_k=tensors["horizon_k"],
                        current_types=tensors["current_types"],
                        action_sequence_logits=terminal_action_context_logits,
                    )
                    terminal_typed_terms = _terminal_typed_diff_loss(
                        terminal_typed_logits,
                        target_types=tensors["target_types"],
                        current_types=tensors["current_types"],
                        target_mask=_vacancy_displacement_target_from_tensors(tensors),
                        candidate_mask=tensors["candidate_mask"],
                    )
                    terminal_typed_diff_type_acc_scores.append(float(terminal_typed_terms["type_acc"].item()))
                    terminal_typed_diff_copy_acc_scores.append(float(terminal_typed_terms["copy_acc"].item()))
                    terminal_typed_diff_topk_f1_scores.append(float(terminal_typed_terms["topk_f1"].item()))
                    terminal_typed_diff_recall32_scores.append(float(terminal_typed_terms["recall32"].item()))
            if hasattr(model, "decode_action_edge_pairs"):
                edge_targets = _action_edge_pair_target_tensors(tensors, action_edge_pair_target_source)
                edge_pair_indices = edge_targets["indices"]
                edge_pair_mask = edge_targets["mask"]
                edge_pair_support_mask = edge_targets["support_mask"]
                edge_pair_moving_type = edge_targets["moving_type"]
                edge_pair_order = edge_targets["order"]
                edge_neg_indices = _negative_action_edge_pair_indices_list(
                    edge_pair_indices,
                    candidate_positions=tensors["candidate_positions"],
                    candidate_mask=tensors["candidate_mask"],
                    box_dims=tensors["box_dims"],
                    mode=action_edge_pair_negative_mode,
                    count=action_edge_pair_negative_count,
                )
                edge_energy_neg_indices = edge_neg_indices
                if int(action_edge_pair_dense_negative_count) > 0:
                    if str(action_edge_pair_target_source).strip().lower().replace("-", "_") in {
                        "vacancy_pair",
                        "terminal_vacancy_pair",
                        "terminal_vacancy_displacement",
                    }:
                        dense_edge_neg_indices = _dense_terminal_vacancy_pair_negative_indices(
                            edge_pair_indices,
                            current_types=tensors["current_types"],
                            candidate_mask=tensors["candidate_mask"],
                            count=int(action_edge_pair_dense_negative_count),
                        )
                    else:
                        dense_edge_neg_indices = _dense_legal_action_edge_pair_negative_indices(
                            edge_pair_indices,
                            current_types=tensors["current_types"],
                            candidate_positions=tensors["candidate_positions"],
                            candidate_mask=tensors["candidate_mask"],
                            box_dims=tensors["box_dims"],
                            count=int(action_edge_pair_dense_negative_count),
                        )
                    edge_energy_neg_indices = torch.cat([edge_neg_indices, dense_edge_neg_indices], dim=-2)
                edge_pos_logits = _decode_action_edge_pair_logits(
                    model.decode_action_edge_pairs,
                    site_latent=site_latent,
                    patch_latent=patch_latent,
                    predicted_next_global=next_pred,
                    path_latent=path_latent,
                    horizon_k=tensors["horizon_k"],
                    current_types=tensors["current_types"],
                    edge_pair_indices=edge_pair_indices,
                )
                edge_neg_logits = _decode_action_edge_pair_logits(
                    model.decode_action_edge_pairs,
                    site_latent=site_latent,
                    patch_latent=patch_latent,
                    predicted_next_global=next_pred,
                    path_latent=path_latent,
                    horizon_k=tensors["horizon_k"],
                    current_types=tensors["current_types"],
                    edge_pair_indices=edge_energy_neg_indices,
                )
                edge_terms = _action_edge_pair_supervision_loss(
                    edge_pos_logits,
                    edge_neg_logits,
                    edge_pair_mask,
                )
                action_edge_pair_rank_acc_scores.append(float(edge_terms["rank_acc"].item()))
                action_edge_pair_pos_prob_scores.append(float(edge_terms["pos_prob"].item()))
                action_edge_pair_neg_prob_scores.append(float(edge_terms["neg_prob"].item()))
                action_edge_pair_available_scores.append(float(edge_terms["available_frac"].item()))
                if hasattr(model, "decode_action_edge_pair_support"):
                    edge_support_logits = _decode_action_edge_pair_logits(
                        model.decode_action_edge_pair_support,
                        site_latent=site_latent,
                        patch_latent=patch_latent,
                        predicted_next_global=next_pred,
                        path_latent=path_latent,
                        horizon_k=tensors["horizon_k"],
                        current_types=tensors["current_types"],
                            edge_pair_indices=edge_pair_indices,
                    )
                    edge_support_neg_logits = _decode_action_edge_pair_logits(
                        model.decode_action_edge_pair_support,
                        site_latent=site_latent,
                        patch_latent=patch_latent,
                        predicted_next_global=next_pred,
                        path_latent=path_latent,
                        horizon_k=tensors["horizon_k"],
                        current_types=tensors["current_types"],
                        edge_pair_indices=edge_neg_indices,
                    )
                    edge_support_terms = _action_edge_pair_support_loss(
                        edge_support_logits,
                        edge_support_neg_logits,
                        edge_pair_mask,
                        edge_pair_support_mask,
                    )
                    action_edge_pair_support_rank_acc_scores.append(float(edge_support_terms["rank_acc"].item()))
                    action_edge_pair_support_prob_scores.append(float(edge_support_terms["support_prob"].item()))
                    action_edge_pair_support_nonsupport_prob_scores.append(
                        float(edge_support_terms["nonsupport_prob"].item())
                    )
                    action_edge_pair_support_neg_prob_scores.append(float(edge_support_terms["neg_prob"].item()))
                    action_edge_pair_support_frac_scores.append(float(edge_support_terms["support_frac"].item()))
                if hasattr(model, "decode_action_edge_pair_moving_type") and hasattr(model, "decode_action_edge_pair_order"):
                    edge_moving_type_logits = model.decode_action_edge_pair_moving_type(
                        site_latent=site_latent,
                        patch_latent=patch_latent,
                        predicted_next_global=next_pred,
                        path_latent=path_latent,
                        horizon_k=tensors["horizon_k"],
                        current_types=tensors["current_types"],
                            edge_pair_indices=edge_pair_indices,
                    )
                    edge_order_logits = model.decode_action_edge_pair_order(
                        site_latent=site_latent,
                        patch_latent=patch_latent,
                        predicted_next_global=next_pred,
                        path_latent=path_latent,
                        horizon_k=tensors["horizon_k"],
                        current_types=tensors["current_types"],
                            edge_pair_indices=edge_pair_indices,
                    )
                    edge_semantic_terms = _action_edge_pair_semantic_loss(
                        edge_moving_type_logits,
                        edge_order_logits,
                        edge_pair_mask,
                        edge_pair_moving_type,
                        edge_pair_order,
                    )
                    action_edge_pair_moving_type_acc_scores.append(
                        float(edge_semantic_terms["moving_type_acc"].item())
                    )
                    action_edge_pair_order_mae_scores.append(float(edge_semantic_terms["order_mae"].item()))
            if hasattr(model, "decode_vacancy_pairs"):
                vacancy_pair_neg_indices = _terminal_vacancy_pair_negative_indices(
                    tensors["teacher_vacancy_pair_indices"],
                    current_types=tensors["current_types"],
                    candidate_mask=tensors["candidate_mask"],
                    dense_count=vacancy_pair_negative_count,
                    structured_count=vacancy_pair_structured_negative_count,
                )
                vacancy_pair_logits = _decode_action_edge_pair_logits(
                    model.decode_vacancy_pairs,
                    site_latent=site_latent,
                    patch_latent=patch_latent,
                    predicted_next_global=next_pred,
                    path_latent=path_latent,
                    horizon_k=tensors["horizon_k"],
                    current_types=tensors["current_types"],
                    edge_pair_indices=tensors["teacher_vacancy_pair_indices"],
                )
                vacancy_pair_neg_logits = _decode_action_edge_pair_logits(
                    model.decode_vacancy_pairs,
                    site_latent=site_latent,
                    patch_latent=patch_latent,
                    predicted_next_global=next_pred,
                    path_latent=path_latent,
                    horizon_k=tensors["horizon_k"],
                    current_types=tensors["current_types"],
                    edge_pair_indices=vacancy_pair_neg_indices,
                )
                vacancy_pair_terms = _action_edge_pair_supervision_loss(
                    vacancy_pair_logits,
                    vacancy_pair_neg_logits,
                    tensors["teacher_vacancy_pair_mask"],
                )
                vacancy_pair_rank_acc_scores.append(float(vacancy_pair_terms["rank_acc"].item()))
                vacancy_pair_pos_prob_scores.append(float(vacancy_pair_terms["pos_prob"].item()))
                vacancy_pair_neg_prob_scores.append(float(vacancy_pair_terms["neg_prob"].item()))
                vacancy_pair_available_scores.append(float(vacancy_pair_terms["available_frac"].item()))
                vacancy_pair_listwise_terms = _pair_listwise_contrastive_loss(
                    vacancy_pair_logits,
                    vacancy_pair_neg_logits,
                    tensors["teacher_vacancy_pair_mask"],
                )
                vacancy_pair_listwise_loss_scores.append(float(vacancy_pair_listwise_terms["loss"].item()))
                vacancy_pair_listwise_acc_scores.append(float(vacancy_pair_listwise_terms["acc"].item()))
                if hasattr(model, "decode_vacancy_pair_moving_type") and hasattr(model, "decode_vacancy_pair_order"):
                    vacancy_pair_moving_type_logits = model.decode_vacancy_pair_moving_type(
                        site_latent=site_latent,
                        patch_latent=patch_latent,
                        predicted_next_global=next_pred,
                        path_latent=path_latent,
                        horizon_k=tensors["horizon_k"],
                        current_types=tensors["current_types"],
                        edge_pair_indices=tensors["teacher_vacancy_pair_indices"],
                    )
                    vacancy_pair_order_logits = model.decode_vacancy_pair_order(
                        site_latent=site_latent,
                        patch_latent=patch_latent,
                        predicted_next_global=next_pred,
                        path_latent=path_latent,
                        horizon_k=tensors["horizon_k"],
                        current_types=tensors["current_types"],
                        edge_pair_indices=tensors["teacher_vacancy_pair_indices"],
                    )
                    vacancy_pair_semantic_terms = _action_edge_pair_semantic_loss(
                        vacancy_pair_moving_type_logits,
                        vacancy_pair_order_logits,
                        tensors["teacher_vacancy_pair_mask"],
                        tensors["teacher_vacancy_pair_moving_type"],
                        tensors["teacher_vacancy_pair_order"],
                    )
                    vacancy_pair_moving_type_acc_scores.append(
                        float(vacancy_pair_semantic_terms["moving_type_acc"].item())
                    )
                    vacancy_pair_order_mae_scores.append(float(vacancy_pair_semantic_terms["order_mae"].item()))
            reward_change_logits, reward_type_logits = _select_reward_edit_context(
                reward_edit_context_source,
                change_logits,
                raw_type_logits,
            )
            duration_outputs = _predict_reward_and_duration_outputs(
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
            reward_hat = duration_outputs["reward"]
            tau_mu = duration_outputs["expected_tau_mu"]
            realized_tau_mu = duration_outputs["realized_tau_mu"]
            realized_tau_log_sigma = duration_outputs["realized_tau_log_sigma"]
            gate_logit = duration_outputs["gate_logit"]
            noop_risk_logit = duration_outputs["noop_risk_logit"]
            realized_tau_available = realized_tau_available and bool(duration_outputs.get("realized_tau_available", True))

            raw_change = (torch.sigmoid(change_logits) > 0.5).float() * tensors["candidate_mask"]
            target_change = tensors["changed_mask"]
            inter = (raw_change * target_change).sum(dim=-1)
            precision = inter / raw_change.sum(dim=-1).clamp(min=1.0)
            recall = inter / target_change.sum(dim=-1).clamp(min=1.0)
            f1 = 2.0 * precision * recall / (precision + recall).clamp(min=1e-6)
            changed_f1_scores.extend(f1.cpu().numpy().tolist())

            valid = tensors["candidate_mask"] > 0
            raw_types = raw_type_logits.argmax(dim=-1)
            changed_valid = valid & (tensors["changed_mask"] > 0)
            unchanged_valid = valid & (tensors["changed_mask"] <= 0)
            unchanged_atom_valid = unchanged_valid & (tensors["current_types"] != V_TYPE)
            unchanged_vacancy_valid = unchanged_valid & (tensors["current_types"] == V_TYPE)
            changed_type_acc = (raw_types[changed_valid] == tensors["target_types"][changed_valid]).float().mean().item() if changed_valid.any() else 1.0
            unchanged_copy_acc = (raw_types[unchanged_valid] == tensors["current_types"][unchanged_valid]).float().mean().item() if unchanged_valid.any() else 1.0
            unchanged_atom_copy_acc = (
                (raw_types[unchanged_atom_valid] == tensors["current_types"][unchanged_atom_valid]).float().mean().item()
                if unchanged_atom_valid.any()
                else 1.0
            )
            unchanged_vacancy_copy_acc = (
                (raw_types[unchanged_vacancy_valid] == tensors["current_types"][unchanged_vacancy_valid]).float().mean().item()
                if unchanged_vacancy_valid.any()
                else 1.0
            )
            changed_type_acc_scores.append(changed_type_acc)
            unchanged_copy_acc_scores.append(unchanged_copy_acc)
            unchanged_atom_copy_acc_scores.append(unchanged_atom_copy_acc)
            unchanged_vacancy_copy_acc_scores.append(unchanged_vacancy_copy_acc)

            for sample_idx in range(tensors["current_types"].shape[0]):
                sample_valid = valid[sample_idx]
                sample_current = tensors["current_types"][sample_idx, sample_valid]
                sample_pred = raw_types[sample_idx, sample_valid]
                vac_to_fe = float(((sample_current == V_TYPE) & (sample_pred == FE_TYPE)).sum().item())
                fe_to_vac = float(((sample_current == FE_TYPE) & (sample_pred == V_TYPE)).sum().item())
                vac_to_cu = float(((sample_current == V_TYPE) & (sample_pred == CU_TYPE)).sum().item())
                cu_to_vac = float(((sample_current == CU_TYPE) & (sample_pred == V_TYPE)).sum().item())
                raw_vac_to_fe_counts.append(vac_to_fe)
                raw_fe_to_vac_counts.append(fe_to_vac)
                raw_vac_to_cu_counts.append(vac_to_cu)
                raw_cu_to_vac_counts.append(cu_to_vac)
                raw_matched_pair_counts.append(min(vac_to_fe, fe_to_vac) + min(vac_to_cu, cu_to_vac))

            projected_types, _, proj_transport_cost, proj_violation = project_types_by_inventory(
                current_types=tensors["current_types"],
                change_logits=change_logits,
                type_logits=raw_type_logits,
                node_mask=tensors["candidate_mask"],
                positions=tensors["candidate_positions"],
                box_dims=tensors["box_dims"],
                horizon_k=tensors["horizon_k"],
                max_changed_sites=2 * tensors["horizon_k"],
            )
            projected_changed_mask = ((projected_types != tensors["current_types"]).float() * tensors["candidate_mask"])
            if reward_prediction_source == "projected":
                _, projected_patch_latent_for_reward = model.encode_patch(
                    positions=tensors["candidate_positions"],
                    nearest_vacancy_offset=tensors["nearest_vacancy_offset"],
                    reach_depth=tensors["reach_depth"],
                    is_start_vacancy=tensors["is_start_vacancy"],
                    type_ids=projected_types,
                    node_mask=tensors["candidate_mask"],
                    global_summary=tensors["global_summary"],
                    box_dims=tensors["box_dims"],
                )
                projected_change_logits, projected_type_logits = projected_edit_logits_from_types(
                    current_types=tensors["current_types"],
                    projected_types=projected_types,
                    candidate_mask=tensors["candidate_mask"],
                )
                projected_change_logits, projected_type_logits = _select_reward_edit_context(
                    reward_edit_context_source,
                    projected_change_logits,
                    projected_type_logits,
                )
                projected_duration_outputs = _predict_reward_and_duration_outputs(
                    model,
                    global_latent,
                    next_pred,
                    path_latent,
                    tensors["global_summary"],
                    tensors["horizon_k"],
                    patch_latent=projected_patch_latent_for_reward,
                    change_logits=projected_change_logits,
                    type_logits=projected_type_logits,
                    current_types=tensors["current_types"],
                    candidate_mask=tensors["candidate_mask"],
                )
                reward_hat = projected_duration_outputs["reward"]
                gate_logit = projected_duration_outputs["gate_logit"]
                noop_risk_logit = projected_duration_outputs["noop_risk_logit"]
                tau_mu = projected_duration_outputs["expected_tau_mu"]
                realized_tau_mu = projected_duration_outputs["realized_tau_mu"]
                realized_tau_log_sigma = projected_duration_outputs["realized_tau_log_sigma"]
                realized_tau_available = realized_tau_available and bool(
                    projected_duration_outputs.get("realized_tau_available", True)
                )
            gated_reward = reward_hat * torch.sigmoid(gate_logit)
            tau_pred.extend(torch.exp(tau_mu).cpu().numpy().tolist())
            tau_true.extend(tensors["tau_exp"].cpu().numpy().tolist())
            realized_tau_mu_pred.extend(realized_tau_mu.cpu().numpy().tolist())
            realized_tau_log_sigma_pred.extend(realized_tau_log_sigma.cpu().numpy().tolist())
            realized_tau_true.extend(tensors["tau_real"].cpu().numpy().tolist())
            reward_pred.extend(gated_reward.cpu().numpy().tolist())
            reward_true.extend(tensors["reward_sum"].cpu().numpy().tolist())
            noop_target = (tensors["changed_mask"].sum(dim=-1) <= 0.0).float()
            noop_risk_pred.extend(torch.sigmoid(noop_risk_logit).cpu().numpy().tolist())
            noop_risk_true.extend(noop_target.cpu().numpy().tolist())
            proj_changed_acc = (projected_types[changed_valid] == tensors["target_types"][changed_valid]).float().mean().item() if changed_valid.any() else 1.0
            projected_changed_type_acc_scores.append(proj_changed_acc)
            reachability_violations.extend(proj_violation.cpu().numpy().tolist())
            transport_costs.extend(proj_transport_cost.cpu().numpy().tolist())
            latent_losses.append(F.smooth_l1_loss(next_pred, next_global).item())
            projected_global = _projected_global_latent_batch(batch=batch, projected_types=projected_types, model=model, device=device)
            projected_global_losses.append(F.smooth_l1_loss(projected_global, next_global).item())

            change_probs = torch.sigmoid(change_logits)
            type_probs = torch.softmax(raw_type_logits, dim=-1)
            current_conf = type_probs.gather(-1, tensors["current_types"].unsqueeze(-1)).squeeze(-1)
            type_change_score = 1.0 - current_conf
            combined_scores = 0.5 * change_probs + 0.5 * type_change_score
            for sample_idx in range(tensors["current_types"].shape[0]):
                valid_idx = torch.nonzero(valid[sample_idx], as_tuple=False).squeeze(-1)
                if valid_idx.numel() == 0:
                    change_topk_f1_scores.append(1.0)
                    projected_change_f1_scores.append(1.0)
                    continue
                target_local = target_change[sample_idx, valid_idx]
                target_count = int(target_local.sum().item())
                if target_count <= 0:
                    change_topk_f1_scores.append(1.0)
                else:
                    ranked_local = torch.argsort(combined_scores[sample_idx, valid_idx], descending=True)[:target_count]
                    topk_pred = torch.zeros_like(target_local)
                    topk_pred[ranked_local] = 1.0
                    topk_inter = float((topk_pred * target_local).sum().item())
                    topk_precision = topk_inter / max(float(topk_pred.sum().item()), 1.0)
                    topk_recall = topk_inter / max(float(target_local.sum().item()), 1.0)
                    topk_f1 = 2.0 * topk_precision * topk_recall / max(topk_precision + topk_recall, 1e-6)
                    change_topk_f1_scores.append(float(topk_f1))

                proj_local = projected_changed_mask[sample_idx, valid_idx]
                proj_inter = float((proj_local * target_local).sum().item())
                proj_precision = proj_inter / max(float(proj_local.sum().item()), 1.0)
                proj_recall = proj_inter / max(float(target_local.sum().item()), 1.0)
                proj_f1 = 2.0 * proj_precision * proj_recall / max(proj_precision + proj_recall, 1e-6)
                projected_change_f1_scores.append(float(proj_f1))

    reward_pred_np = np.asarray(reward_pred, dtype=np.float64)
    reward_true_np = np.asarray(reward_true, dtype=np.float64)
    noop_risk_pred_np = np.asarray(noop_risk_pred, dtype=np.float64)
    noop_risk_true_np = np.asarray(noop_risk_true, dtype=np.float64)
    tau_pred_np = np.asarray(tau_pred, dtype=np.float64)
    tau_true_np = np.asarray(tau_true, dtype=np.float64)
    candidate_quality_pred_np = np.asarray(candidate_quality_pred, dtype=np.float64)
    candidate_quality_true_np = np.asarray(candidate_quality_true, dtype=np.float64)
    reward_metrics = _compute_metrics(reward_pred_np, reward_true_np)
    reward_diagnostics = _compute_reward_diagnostics(reward_pred_np, reward_true_np)
    tau_metrics = _compute_log_metrics(tau_pred_np, tau_true_np)
    realized_tau_metrics = _compute_lognormal_distribution_metrics(
        np.asarray(realized_tau_mu_pred),
        np.asarray(realized_tau_log_sigma_pred),
        np.asarray(realized_tau_true),
    )
    if not realized_tau_available:
        realized_tau_metrics["nll"] = 0.0
        realized_tau_metrics["coverage_68"] = 0.0
        realized_tau_metrics["coverage_95"] = 0.0
        realized_tau_metrics["pit_mean"] = 0.0
        realized_tau_metrics["pit_var"] = 0.0
        realized_tau_metrics["pit_ks"] = 0.0
        realized_tau_metrics["mean_log_sigma"] = 0.0
    return {
        "reward_mae": reward_metrics["mae"],
        "reward_rmse": reward_metrics["rmse"],
        "reward_corr": reward_metrics["corr"],
        "reward_mean_bias": reward_diagnostics["mean_bias"],
        "reward_zero_target_frac": reward_diagnostics["zero_target_frac"],
        "reward_negative_target_frac": reward_diagnostics["negative_target_frac"],
        "reward_zero_pred_mean": reward_diagnostics["zero_pred_mean"],
        "reward_zero_pred_mean_abs": reward_diagnostics["zero_pred_mean_abs"],
        "reward_negative_pred_mean": reward_diagnostics["negative_pred_mean"],
        "reward_negative_bias": reward_diagnostics["negative_bias"],
        "reward_nonzero_sign_acc": reward_diagnostics["nonzero_sign_acc"],
        "noop_risk_target_frac": float(np.mean(noop_risk_true_np)) if noop_risk_true_np.size else 0.0,
        "noop_risk_pred_mean": float(np.mean(noop_risk_pred_np)) if noop_risk_pred_np.size else 0.0,
        "noop_risk_noop_pred_mean": (
            float(np.mean(noop_risk_pred_np[noop_risk_true_np > 0.5]))
            if np.any(noop_risk_true_np > 0.5)
            else 0.0
        ),
        "noop_risk_nonnoop_pred_mean": (
            float(np.mean(noop_risk_pred_np[noop_risk_true_np <= 0.5]))
            if np.any(noop_risk_true_np <= 0.5)
            else 0.0
        ),
        "tau_log_mae": tau_metrics["log_mae"],
        "tau_log_rmse": tau_metrics["log_rmse"],
        "tau_log_corr": tau_metrics["log_corr"],
        "tau_scale_ratio": tau_metrics["scale_ratio"],
        "realized_tau_available": float(realized_tau_available),
        "realized_tau_mae": realized_tau_metrics["mae"],
        "realized_tau_rmse": realized_tau_metrics["rmse"],
        "realized_tau_corr": realized_tau_metrics["corr"],
        "realized_tau_log_mae": realized_tau_metrics["log_mae"],
        "realized_tau_log_rmse": realized_tau_metrics["log_rmse"],
        "realized_tau_log_corr": realized_tau_metrics["log_corr"],
        "realized_tau_scale_ratio": realized_tau_metrics["scale_ratio"],
        "realized_tau_nll": realized_tau_metrics["nll"],
        "realized_tau_coverage_68": realized_tau_metrics["coverage_68"],
        "realized_tau_coverage_95": realized_tau_metrics["coverage_95"],
        "realized_tau_pit_mean": realized_tau_metrics["pit_mean"],
        "realized_tau_pit_var": realized_tau_metrics["pit_var"],
        "realized_tau_pit_ks": realized_tau_metrics["pit_ks"],
        "realized_tau_mean_log_sigma": realized_tau_metrics["mean_log_sigma"],
        "change_f1": float(np.mean(changed_f1_scores)) if changed_f1_scores else 0.0,
        "change_topk_f1": float(np.mean(change_topk_f1_scores)) if change_topk_f1_scores else 0.0,
        "changed_type_acc": float(np.mean(changed_type_acc_scores)) if changed_type_acc_scores else 0.0,
        "projected_change_f1": float(np.mean(projected_change_f1_scores)) if projected_change_f1_scores else 0.0,
        "projected_changed_type_acc": float(np.mean(projected_changed_type_acc_scores)) if projected_changed_type_acc_scores else 0.0,
        "proposal_topk_f1": float(np.mean(proposal_topk_f1_scores)) if proposal_topk_f1_scores else 0.0,
        "proposal_recall32": float(np.mean(proposal_recall32_scores)) if proposal_recall32_scores else 0.0,
        "action_support_topk_f1": (
            float(np.mean(action_support_topk_f1_scores)) if action_support_topk_f1_scores else 0.0
        ),
        "action_support_recall32": (
            float(np.mean(action_support_recall32_scores)) if action_support_recall32_scores else 0.0
        ),
        "action_source_topk_f1": float(np.mean(action_source_topk_f1_scores)) if action_source_topk_f1_scores else 0.0,
        "action_source_recall32": float(np.mean(action_source_recall32_scores)) if action_source_recall32_scores else 0.0,
        "action_destination_topk_f1": (
            float(np.mean(action_destination_topk_f1_scores)) if action_destination_topk_f1_scores else 0.0
        ),
        "action_destination_recall32": (
            float(np.mean(action_destination_recall32_scores)) if action_destination_recall32_scores else 0.0
        ),
        "action_endpoint_topk_f1": (
            float(np.mean(action_endpoint_topk_f1_scores)) if action_endpoint_topk_f1_scores else 0.0
        ),
        "action_endpoint_recall32": (
            float(np.mean(action_endpoint_recall32_scores)) if action_endpoint_recall32_scores else 0.0
        ),
        "terminal_edit_topk_f1": (
            float(np.mean(terminal_edit_topk_f1_scores)) if terminal_edit_topk_f1_scores else 0.0
        ),
        "terminal_edit_recall32": (
            float(np.mean(terminal_edit_recall32_scores)) if terminal_edit_recall32_scores else 0.0
        ),
        "terminal_typed_diff_type_acc": (
            float(np.mean(terminal_typed_diff_type_acc_scores)) if terminal_typed_diff_type_acc_scores else 0.0
        ),
        "terminal_typed_diff_copy_acc": (
            float(np.mean(terminal_typed_diff_copy_acc_scores)) if terminal_typed_diff_copy_acc_scores else 0.0
        ),
        "terminal_typed_diff_topk_f1": (
            float(np.mean(terminal_typed_diff_topk_f1_scores)) if terminal_typed_diff_topk_f1_scores else 0.0
        ),
        "terminal_typed_diff_recall32": (
            float(np.mean(terminal_typed_diff_recall32_scores)) if terminal_typed_diff_recall32_scores else 0.0
        ),
        "action_edge_pair_rank_acc": (
            float(np.mean(action_edge_pair_rank_acc_scores)) if action_edge_pair_rank_acc_scores else 0.0
        ),
        "action_edge_pair_pos_prob": (
            float(np.mean(action_edge_pair_pos_prob_scores)) if action_edge_pair_pos_prob_scores else 0.0
        ),
        "action_edge_pair_neg_prob": (
            float(np.mean(action_edge_pair_neg_prob_scores)) if action_edge_pair_neg_prob_scores else 0.0
        ),
        "action_edge_pair_available_frac": (
            float(np.mean(action_edge_pair_available_scores)) if action_edge_pair_available_scores else 0.0
        ),
        "action_edge_pair_support_rank_acc": (
            float(np.mean(action_edge_pair_support_rank_acc_scores))
            if action_edge_pair_support_rank_acc_scores
            else 0.0
        ),
        "action_edge_pair_support_prob": (
            float(np.mean(action_edge_pair_support_prob_scores)) if action_edge_pair_support_prob_scores else 0.0
        ),
        "action_edge_pair_support_nonsupport_prob": (
            float(np.mean(action_edge_pair_support_nonsupport_prob_scores))
            if action_edge_pair_support_nonsupport_prob_scores
            else 0.0
        ),
        "action_edge_pair_support_neg_prob": (
            float(np.mean(action_edge_pair_support_neg_prob_scores))
            if action_edge_pair_support_neg_prob_scores
            else 0.0
        ),
        "action_edge_pair_support_frac": (
            float(np.mean(action_edge_pair_support_frac_scores)) if action_edge_pair_support_frac_scores else 0.0
        ),
        "action_edge_pair_moving_type_acc": (
            float(np.mean(action_edge_pair_moving_type_acc_scores))
            if action_edge_pair_moving_type_acc_scores
            else 0.0
        ),
        "action_edge_pair_order_mae": (
            float(np.mean(action_edge_pair_order_mae_scores)) if action_edge_pair_order_mae_scores else 0.0
        ),
        "vacancy_pair_rank_acc": (
            float(np.mean(vacancy_pair_rank_acc_scores)) if vacancy_pair_rank_acc_scores else 0.0
        ),
        "vacancy_pair_pos_prob": (
            float(np.mean(vacancy_pair_pos_prob_scores)) if vacancy_pair_pos_prob_scores else 0.0
        ),
        "vacancy_pair_neg_prob": (
            float(np.mean(vacancy_pair_neg_prob_scores)) if vacancy_pair_neg_prob_scores else 0.0
        ),
        "vacancy_pair_available_frac": (
            float(np.mean(vacancy_pair_available_scores)) if vacancy_pair_available_scores else 0.0
        ),
        "vacancy_pair_listwise_loss": (
            float(np.mean(vacancy_pair_listwise_loss_scores)) if vacancy_pair_listwise_loss_scores else 0.0
        ),
        "vacancy_pair_listwise_acc": (
            float(np.mean(vacancy_pair_listwise_acc_scores)) if vacancy_pair_listwise_acc_scores else 0.0
        ),
        "vacancy_pair_moving_type_acc": (
            float(np.mean(vacancy_pair_moving_type_acc_scores)) if vacancy_pair_moving_type_acc_scores else 0.0
        ),
        "vacancy_pair_order_mae": (
            float(np.mean(vacancy_pair_order_mae_scores)) if vacancy_pair_order_mae_scores else 0.0
        ),
        "candidate_quality_available": float(candidate_quality_true_np.size),
        "candidate_quality_mae": (
            float(np.mean(np.abs(candidate_quality_pred_np - candidate_quality_true_np)))
            if candidate_quality_true_np.size
            else 0.0
        ),
        "candidate_quality_corr": (
            float(np.corrcoef(candidate_quality_pred_np, candidate_quality_true_np)[0, 1])
            if candidate_quality_true_np.size > 1
            and np.std(candidate_quality_pred_np) > 0
            and np.std(candidate_quality_true_np) > 0
            else 0.0
        ),
        "candidate_quality_pred_mean": float(np.mean(candidate_quality_pred_np)) if candidate_quality_pred_np.size else 0.0,
        "candidate_quality_target_mean": float(np.mean(candidate_quality_true_np)) if candidate_quality_true_np.size else 0.0,
        "unchanged_copy_acc": float(np.mean(unchanged_copy_acc_scores)) if unchanged_copy_acc_scores else 0.0,
        "unchanged_atom_copy_acc": float(np.mean(unchanged_atom_copy_acc_scores)) if unchanged_atom_copy_acc_scores else 0.0,
        "unchanged_vacancy_copy_acc": float(np.mean(unchanged_vacancy_copy_acc_scores)) if unchanged_vacancy_copy_acc_scores else 0.0,
        "raw_vac_to_fe_count": float(np.mean(raw_vac_to_fe_counts)) if raw_vac_to_fe_counts else 0.0,
        "raw_fe_to_vac_count": float(np.mean(raw_fe_to_vac_counts)) if raw_fe_to_vac_counts else 0.0,
        "raw_vac_to_cu_count": float(np.mean(raw_vac_to_cu_counts)) if raw_vac_to_cu_counts else 0.0,
        "raw_cu_to_vac_count": float(np.mean(raw_cu_to_vac_counts)) if raw_cu_to_vac_counts else 0.0,
        "raw_matched_pair_count": float(np.mean(raw_matched_pair_counts)) if raw_matched_pair_counts else 0.0,
        "latent_l1": float(np.mean(latent_losses)) if latent_losses else 0.0,
        "projected_global_l1": float(np.mean(projected_global_losses)) if projected_global_losses else 0.0,
        "reachability_violation_rate": float(np.mean(reachability_violations)) if reachability_violations else 0.0,
        "mean_vacancy_transport_cost": float(np.mean(transport_costs)) if transport_costs else 0.0,
    }


def _train_epoch(
    model: MacroDreamerEditModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    max_changed_sites: int,
    weights: dict[str, float],
    *,
    epoch: int = 1,
    total_epochs: int = 1,
    tau_supervision_mode: str = "prior_main",
    proj_every_n_batches: int = 1,
    aux_anneal: bool = True,
    mask_sparsity_weight: float = 0.0,
    count_loss_weight: float = 0.1,
    detach_proj_encoder: bool = False,
    reward_magnitude_weight: float = 1.0,
    reward_gated_weight: float = 1.0,
    reward_gate_weight: float = 0.25,
    reward_zero_weight: float = 0.5,
    reward_sign_weight: float = 0.25,
    reward_prediction_source: str = "raw",
    reward_edit_context_source: str = "default",
    noop_change_weight: float = 0.0,
    noop_type_copy_weight: float = 0.0,
    projected_noop_fp_weight: float = 0.0,
    noop_risk_weight: float = 0.0,
    prior_noop_risk_weight: float = 0.0,
    proposal_support_weight: float = 0.0,
    prior_proposal_support_weight: float = 0.0,
    proposal_hard_negative_weight: float = 0.0,
    proposal_rank_margin_weight: float = 0.0,
    proposal_candidate_positive_weight: float = 0.0,
    proposal_candidate_negative_weight: float = 0.0,
    proposal_candidate_rank_margin_weight: float = 0.0,
    proposal_target_source: str = "changed",
    action_support_weight: float = 0.0,
    prior_action_support_weight: float = 0.0,
    action_support_hard_negative_weight: float = 0.0,
    action_support_rank_margin_weight: float = 0.0,
    action_support_candidate_positive_weight: float = 0.0,
    action_support_candidate_negative_weight: float = 0.0,
    action_support_candidate_rank_margin_weight: float = 0.0,
    action_support_target_source: str = "touched",
    terminal_edit_support_weight: float = 0.0,
    prior_terminal_edit_support_weight: float = 0.0,
    terminal_edit_support_hard_negative_weight: float = 0.0,
    terminal_edit_support_rank_margin_weight: float = 0.0,
    terminal_edit_support_candidate_positive_weight: float = 0.0,
    terminal_edit_support_candidate_negative_weight: float = 0.0,
    terminal_edit_support_candidate_rank_margin_weight: float = 0.0,
    terminal_edit_support_target_source: str = "changed",
    terminal_edit_action_context_source: str = "action_endpoint",
    terminal_typed_diff_weight: float = 0.0,
    prior_terminal_typed_diff_weight: float = 0.0,
    terminal_typed_diff_copy_weight: float = 0.05,
    terminal_typed_diff_support_weight: float = 0.0,
    action_source_support_weight: float = 0.0,
    prior_action_source_support_weight: float = 0.0,
    action_destination_support_weight: float = 0.0,
    prior_action_destination_support_weight: float = 0.0,
    action_edge_pair_weight: float = 0.0,
    prior_action_edge_pair_weight: float = 0.0,
    action_edge_pair_support_weight: float = 0.0,
    prior_action_edge_pair_support_weight: float = 0.0,
    action_edge_pair_semantic_weight: float = 0.0,
    prior_action_edge_pair_semantic_weight: float = 0.0,
    action_edge_pair_negative_weight: float = 1.0,
    action_edge_pair_rank_margin_weight: float = 0.25,
    action_edge_pair_negative_mode: str = "self",
    action_edge_pair_negative_count: int = 1,
    action_edge_pair_dense_negative_count: int = 0,
    action_edge_pair_target_source: str = "action",
    vacancy_pair_weight: float = 0.0,
    prior_vacancy_pair_weight: float = 0.0,
    vacancy_pair_semantic_weight: float = 0.0,
    prior_vacancy_pair_semantic_weight: float = 0.0,
    vacancy_pair_listwise_weight: float = 0.0,
    prior_vacancy_pair_listwise_weight: float = 0.0,
    vacancy_pair_interaction_weight: float = 0.0,
    prior_vacancy_pair_interaction_weight: float = 0.0,
    vacancy_pair_interaction_listwise_weight: float = 0.0,
    prior_vacancy_pair_interaction_listwise_weight: float = 0.0,
    vacancy_pair_negative_count: int = 1,
    vacancy_pair_structured_negative_count: int = 0,
    candidate_quality_weight: float = 0.0,
    prior_candidate_quality_weight: float = 0.0,
) -> dict[str, float]:
    model.train()
    aux_scale = _scheduled_aux_scale(epoch, total_epochs) if aux_anneal else 1.0
    logs = {
        "loss": 0.0,
        "mask": 0.0,
        "count": 0.0,
        "pair": 0.0,
        "prior_pair": 0.0,
        "proj_mask": 0.0,
        "type": 0.0,
        "noop_change": 0.0,
        "noop_type_copy": 0.0,
        "atom_to_vac_type": 0.0,
        "vac_to_atom_type": 0.0,
        "tau": 0.0,
        "tau_log_mu": 0.0,
        "tau_post": 0.0,
        "tau_prior": 0.0,
        "realized_tau": 0.0,
        "realized_tau_post": 0.0,
        "realized_tau_prior": 0.0,
        "tau_post_scale": 0.0,
        "reward": 0.0,
        "noop_risk": 0.0,
        "prior_noop_risk": 0.0,
        "proposal": 0.0,
        "prior_proposal": 0.0,
        "proposal_hard_negative": 0.0,
        "prior_proposal_hard_negative": 0.0,
        "proposal_rank_margin": 0.0,
        "prior_proposal_rank_margin": 0.0,
        "proposal_candidate_positive": 0.0,
        "prior_proposal_candidate_positive": 0.0,
        "proposal_candidate_negative": 0.0,
        "prior_proposal_candidate_negative": 0.0,
        "proposal_candidate_rank_margin": 0.0,
        "prior_proposal_candidate_rank_margin": 0.0,
        "action_support": 0.0,
        "prior_action_support": 0.0,
        "action_support_hard_negative": 0.0,
        "prior_action_support_hard_negative": 0.0,
        "action_support_rank_margin": 0.0,
        "prior_action_support_rank_margin": 0.0,
        "action_support_candidate_positive": 0.0,
        "prior_action_support_candidate_positive": 0.0,
        "action_support_candidate_negative": 0.0,
        "prior_action_support_candidate_negative": 0.0,
        "action_support_candidate_rank_margin": 0.0,
        "prior_action_support_candidate_rank_margin": 0.0,
        "terminal_edit_support": 0.0,
        "prior_terminal_edit_support": 0.0,
        "terminal_edit_topk_f1": 0.0,
        "terminal_edit_recall32": 0.0,
        "terminal_edit_candidate_positive": 0.0,
        "terminal_edit_candidate_negative": 0.0,
        "terminal_edit_candidate_rank_margin": 0.0,
        "terminal_typed_diff": 0.0,
        "prior_terminal_typed_diff": 0.0,
        "terminal_typed_diff_support": 0.0,
        "prior_terminal_typed_diff_support": 0.0,
        "terminal_typed_diff_type_acc": 0.0,
        "terminal_typed_diff_copy_acc": 0.0,
        "terminal_typed_diff_topk_f1": 0.0,
        "terminal_typed_diff_recall32": 0.0,
        "candidate_quality": 0.0,
        "prior_candidate_quality": 0.0,
        "candidate_quality_mae": 0.0,
        "candidate_quality_corr": 0.0,
        "candidate_quality_pred_mean": 0.0,
        "candidate_quality_target_mean": 0.0,
        "candidate_quality_available_frac": 0.0,
        "proposal_topk_f1": 0.0,
        "proposal_recall32": 0.0,
        "action_support_topk_f1": 0.0,
        "action_support_recall32": 0.0,
        "action_source": 0.0,
        "prior_action_source": 0.0,
        "action_source_topk_f1": 0.0,
        "action_source_recall32": 0.0,
        "action_destination": 0.0,
        "prior_action_destination": 0.0,
        "action_destination_topk_f1": 0.0,
        "action_destination_recall32": 0.0,
        "action_endpoint_topk_f1": 0.0,
        "action_endpoint_recall32": 0.0,
        "prior_action_endpoint_topk_f1": 0.0,
        "prior_action_endpoint_recall32": 0.0,
        "action_edge_pair": 0.0,
        "prior_action_edge_pair": 0.0,
        "action_edge_pair_positive": 0.0,
        "action_edge_pair_negative": 0.0,
        "action_edge_pair_rank_margin": 0.0,
        "action_edge_pair_rank_acc": 0.0,
        "action_edge_pair_available_frac": 0.0,
        "action_edge_pair_support": 0.0,
        "prior_action_edge_pair_support": 0.0,
        "action_edge_pair_support_bce": 0.0,
        "action_edge_pair_support_negative": 0.0,
        "action_edge_pair_support_rank_margin": 0.0,
        "action_edge_pair_support_rank_acc": 0.0,
        "action_edge_pair_support_prob": 0.0,
        "action_edge_pair_support_nonsupport_prob": 0.0,
        "action_edge_pair_support_frac": 0.0,
        "action_edge_pair_semantic": 0.0,
        "prior_action_edge_pair_semantic": 0.0,
        "action_edge_pair_moving_type": 0.0,
        "prior_action_edge_pair_moving_type": 0.0,
        "action_edge_pair_order": 0.0,
        "prior_action_edge_pair_order": 0.0,
        "action_edge_pair_moving_type_acc": 0.0,
        "action_edge_pair_order_mae": 0.0,
        "vacancy_pair": 0.0,
        "prior_vacancy_pair": 0.0,
        "vacancy_pair_positive": 0.0,
        "vacancy_pair_negative": 0.0,
        "vacancy_pair_rank_margin": 0.0,
        "vacancy_pair_rank_acc": 0.0,
        "vacancy_pair_available_frac": 0.0,
        "vacancy_pair_listwise": 0.0,
        "prior_vacancy_pair_listwise": 0.0,
        "vacancy_pair_listwise_acc": 0.0,
        "prior_vacancy_pair_listwise_acc": 0.0,
        "vacancy_pair_interaction": 0.0,
        "prior_vacancy_pair_interaction": 0.0,
        "vacancy_pair_interaction_rank_acc": 0.0,
        "prior_vacancy_pair_interaction_rank_acc": 0.0,
        "vacancy_pair_interaction_listwise": 0.0,
        "prior_vacancy_pair_interaction_listwise": 0.0,
        "vacancy_pair_interaction_listwise_acc": 0.0,
        "prior_vacancy_pair_interaction_listwise_acc": 0.0,
        "vacancy_pair_semantic": 0.0,
        "prior_vacancy_pair_semantic": 0.0,
        "vacancy_pair_moving_type": 0.0,
        "prior_vacancy_pair_moving_type": 0.0,
        "vacancy_pair_order": 0.0,
        "prior_vacancy_pair_order": 0.0,
        "vacancy_pair_moving_type_acc": 0.0,
        "vacancy_pair_order_mae": 0.0,
        "noop_risk_target_frac": 0.0,
        "noop_risk_noop_prob": 0.0,
        "noop_risk_nonnoop_prob": 0.0,
        "latent": 0.0,
        "proj": 0.0,
        "path": 0.0,
        "prior_edit": 0.0,
        "prior_latent": 0.0,
        "mask_aux_scale": 0.0,
    }
    count = 0
    proj_scale = float(max(proj_every_n_batches, 1))
    for batch_idx, batch in enumerate(loader):
        compute_proj = (proj_every_n_batches <= 1) or (batch_idx % proj_every_n_batches == 0)
        tensors = _batch_to_device(batch, device)
        global_latent = model.encode_global(tensors["start_obs"])
        next_global = model.encode_global(tensors["next_obs"]).detach()
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
        target_site_latent, target_patch_latent = model.encode_patch(
            positions=tensors["candidate_positions"],
            nearest_vacancy_offset=tensors["nearest_vacancy_offset"],
            reach_depth=tensors["reach_depth"],
            is_start_vacancy=tensors["is_start_vacancy"],
            type_ids=tensors["target_types"],
            node_mask=tensors["candidate_mask"],
            global_summary=tensors["global_summary"],
            box_dims=tensors["box_dims"],
        )
        prior_mu, prior_logvar = model.prior_stats(global_latent, tensors["global_summary"], tensors["horizon_k"])
        post_mu, post_logvar = model.posterior_stats(global_latent, next_global, tensors["teacher_path_summary"], tensors["horizon_k"])
        post_c = model.sample_path_latent(post_mu, post_logvar)
        prior_c = model.sample_path_latent(prior_mu, prior_logvar, deterministic=True)
        next_pred = model.predict_next_global(global_latent, post_c, tensors["horizon_k"])
        next_pred_prior = model.predict_next_global(global_latent, prior_c, tensors["horizon_k"])
        change_logits, raw_type_logits = model.decode_edit(
            site_latent=site_latent,
            patch_latent=patch_latent,
            predicted_next_global=next_pred,
            path_latent=post_c,
            horizon_k=tensors["horizon_k"],
            current_types=tensors["current_types"],
        )
        change_logits_prior, raw_type_logits_prior = model.decode_edit(
            site_latent=site_latent,
            patch_latent=patch_latent,
            predicted_next_global=next_pred_prior,
            path_latent=prior_c,
            horizon_k=tensors["horizon_k"],
            current_types=tensors["current_types"],
        )
        proposal_logits = model.decode_proposal(
            site_latent=site_latent,
            patch_latent=patch_latent,
            predicted_next_global=next_pred,
            path_latent=post_c,
            horizon_k=tensors["horizon_k"],
            current_types=tensors["current_types"],
        )
        proposal_logits_prior = model.decode_proposal(
            site_latent=site_latent,
            patch_latent=patch_latent,
            predicted_next_global=next_pred_prior,
            path_latent=prior_c,
            horizon_k=tensors["horizon_k"],
            current_types=tensors["current_types"],
        )
        action_support_logits = (
            model.decode_action_support(
                site_latent=site_latent,
                patch_latent=patch_latent,
                predicted_next_global=next_pred,
                path_latent=post_c,
                horizon_k=tensors["horizon_k"],
                current_types=tensors["current_types"],
            )
            if hasattr(model, "decode_action_support")
            else proposal_logits
        )
        action_support_logits_prior = (
            model.decode_action_support(
                site_latent=site_latent,
                patch_latent=patch_latent,
                predicted_next_global=next_pred_prior,
                path_latent=prior_c,
                horizon_k=tensors["horizon_k"],
                current_types=tensors["current_types"],
            )
            if hasattr(model, "decode_action_support")
            else proposal_logits_prior
        )
        action_source_logits = (
            model.decode_action_source_support(
                site_latent=site_latent,
                patch_latent=patch_latent,
                predicted_next_global=next_pred,
                path_latent=post_c,
                horizon_k=tensors["horizon_k"],
                current_types=tensors["current_types"],
            )
            if hasattr(model, "decode_action_source_support")
            else action_support_logits
        )
        action_source_logits_prior = (
            model.decode_action_source_support(
                site_latent=site_latent,
                patch_latent=patch_latent,
                predicted_next_global=next_pred_prior,
                path_latent=prior_c,
                horizon_k=tensors["horizon_k"],
                current_types=tensors["current_types"],
            )
            if hasattr(model, "decode_action_source_support")
            else action_support_logits_prior
        )
        action_destination_logits = (
            model.decode_action_destination_support(
                site_latent=site_latent,
                patch_latent=patch_latent,
                predicted_next_global=next_pred,
                path_latent=post_c,
                horizon_k=tensors["horizon_k"],
                current_types=tensors["current_types"],
            )
            if hasattr(model, "decode_action_destination_support")
            else action_support_logits
        )
        action_destination_logits_prior = (
            model.decode_action_destination_support(
                site_latent=site_latent,
                patch_latent=patch_latent,
                predicted_next_global=next_pred_prior,
                path_latent=prior_c,
                horizon_k=tensors["horizon_k"],
                current_types=tensors["current_types"],
            )
            if hasattr(model, "decode_action_destination_support")
            else action_support_logits_prior
        )
        action_endpoint_logits = combine_action_endpoint_logits(action_source_logits, action_destination_logits)
        action_endpoint_logits_prior = combine_action_endpoint_logits(
            action_source_logits_prior,
            action_destination_logits_prior,
        )
        if hasattr(model, "decode_terminal_edit_support"):
            terminal_action_context_logits = _terminal_action_context_logits_from_tensors(
                tensors,
                terminal_edit_action_context_source,
                action_endpoint_logits,
            )
            terminal_action_context_logits_prior = _terminal_action_context_logits_from_tensors(
                tensors,
                terminal_edit_action_context_source,
                action_endpoint_logits_prior,
            )
            terminal_edit_support_logits = model.decode_terminal_edit_support(
                site_latent=site_latent,
                patch_latent=patch_latent,
                predicted_next_global=next_pred,
                path_latent=post_c,
                horizon_k=tensors["horizon_k"],
                current_types=tensors["current_types"],
                action_sequence_logits=terminal_action_context_logits,
            )
            terminal_edit_support_logits_prior = model.decode_terminal_edit_support(
                site_latent=site_latent,
                patch_latent=patch_latent,
                predicted_next_global=next_pred_prior,
                path_latent=prior_c,
                horizon_k=tensors["horizon_k"],
                current_types=tensors["current_types"],
                action_sequence_logits=terminal_action_context_logits_prior,
            )
        else:
            terminal_edit_support_logits = proposal_logits
            terminal_edit_support_logits_prior = proposal_logits_prior
        if hasattr(model, "decode_terminal_typed_diff"):
            terminal_typed_diff_logits = model.decode_terminal_typed_diff(
                site_latent=site_latent,
                patch_latent=patch_latent,
                predicted_next_global=next_pred,
                path_latent=post_c,
                horizon_k=tensors["horizon_k"],
                current_types=tensors["current_types"],
                action_sequence_logits=terminal_action_context_logits,
            )
            terminal_typed_diff_logits_prior = model.decode_terminal_typed_diff(
                site_latent=site_latent,
                patch_latent=patch_latent,
                predicted_next_global=next_pred_prior,
                path_latent=prior_c,
                horizon_k=tensors["horizon_k"],
                current_types=tensors["current_types"],
                action_sequence_logits=terminal_action_context_logits_prior,
            )
        else:
            terminal_typed_diff_logits = raw_type_logits
            terminal_typed_diff_logits_prior = raw_type_logits_prior
        action_edge_pair_targets = _action_edge_pair_target_tensors(tensors, action_edge_pair_target_source)
        action_edge_pair_indices = action_edge_pair_targets["indices"]
        action_edge_pair_mask = action_edge_pair_targets["mask"]
        action_edge_pair_support_mask = action_edge_pair_targets["support_mask"]
        action_edge_pair_moving_type_target = action_edge_pair_targets["moving_type"]
        action_edge_pair_order_target = action_edge_pair_targets["order"]
        action_edge_pair_neg_indices = _negative_action_edge_pair_indices_list(
            action_edge_pair_indices,
            candidate_positions=tensors["candidate_positions"],
            candidate_mask=tensors["candidate_mask"],
            box_dims=tensors["box_dims"],
            mode=action_edge_pair_negative_mode,
            count=action_edge_pair_negative_count,
        )
        action_edge_pair_energy_neg_indices = action_edge_pair_neg_indices
        if int(action_edge_pair_dense_negative_count) > 0:
            if str(action_edge_pair_target_source).strip().lower().replace("-", "_") in {
                "vacancy_pair",
                "terminal_vacancy_pair",
                "terminal_vacancy_displacement",
            }:
                dense_action_edge_pair_neg_indices = _dense_terminal_vacancy_pair_negative_indices(
                    action_edge_pair_indices,
                    current_types=tensors["current_types"],
                    candidate_mask=tensors["candidate_mask"],
                    count=int(action_edge_pair_dense_negative_count),
                )
            else:
                dense_action_edge_pair_neg_indices = _dense_legal_action_edge_pair_negative_indices(
                    action_edge_pair_indices,
                    current_types=tensors["current_types"],
                    candidate_positions=tensors["candidate_positions"],
                    candidate_mask=tensors["candidate_mask"],
                    box_dims=tensors["box_dims"],
                    count=int(action_edge_pair_dense_negative_count),
                )
            action_edge_pair_energy_neg_indices = torch.cat(
                [action_edge_pair_neg_indices, dense_action_edge_pair_neg_indices],
                dim=-2,
            )
        if hasattr(model, "decode_action_edge_pairs"):
            action_edge_pair_logits = _decode_action_edge_pair_logits(
                model.decode_action_edge_pairs,
                site_latent=site_latent,
                patch_latent=patch_latent,
                predicted_next_global=next_pred,
                path_latent=post_c,
                horizon_k=tensors["horizon_k"],
                current_types=tensors["current_types"],
                edge_pair_indices=action_edge_pair_indices,
            )
            action_edge_pair_neg_logits = _decode_action_edge_pair_logits(
                model.decode_action_edge_pairs,
                site_latent=site_latent,
                patch_latent=patch_latent,
                predicted_next_global=next_pred,
                path_latent=post_c,
                horizon_k=tensors["horizon_k"],
                current_types=tensors["current_types"],
                edge_pair_indices=action_edge_pair_energy_neg_indices,
            )
            action_edge_pair_logits_prior = _decode_action_edge_pair_logits(
                model.decode_action_edge_pairs,
                site_latent=site_latent,
                patch_latent=patch_latent,
                predicted_next_global=next_pred_prior,
                path_latent=prior_c,
                horizon_k=tensors["horizon_k"],
                current_types=tensors["current_types"],
                edge_pair_indices=action_edge_pair_indices,
            )
            action_edge_pair_neg_logits_prior = _decode_action_edge_pair_logits(
                model.decode_action_edge_pairs,
                site_latent=site_latent,
                patch_latent=patch_latent,
                predicted_next_global=next_pred_prior,
                path_latent=prior_c,
                horizon_k=tensors["horizon_k"],
                current_types=tensors["current_types"],
                edge_pair_indices=action_edge_pair_energy_neg_indices,
            )
            if hasattr(model, "decode_action_edge_pair_support"):
                action_edge_pair_support_logits = _decode_action_edge_pair_logits(
                    model.decode_action_edge_pair_support,
                    site_latent=site_latent,
                    patch_latent=patch_latent,
                    predicted_next_global=next_pred,
                    path_latent=post_c,
                    horizon_k=tensors["horizon_k"],
                    current_types=tensors["current_types"],
                    edge_pair_indices=action_edge_pair_indices,
                )
                action_edge_pair_support_neg_logits = _decode_action_edge_pair_logits(
                    model.decode_action_edge_pair_support,
                    site_latent=site_latent,
                    patch_latent=patch_latent,
                    predicted_next_global=next_pred,
                    path_latent=post_c,
                    horizon_k=tensors["horizon_k"],
                    current_types=tensors["current_types"],
                    edge_pair_indices=action_edge_pair_neg_indices,
                )
                action_edge_pair_support_logits_prior = _decode_action_edge_pair_logits(
                    model.decode_action_edge_pair_support,
                    site_latent=site_latent,
                    patch_latent=patch_latent,
                    predicted_next_global=next_pred_prior,
                    path_latent=prior_c,
                    horizon_k=tensors["horizon_k"],
                    current_types=tensors["current_types"],
                    edge_pair_indices=action_edge_pair_indices,
                )
                action_edge_pair_support_neg_logits_prior = _decode_action_edge_pair_logits(
                    model.decode_action_edge_pair_support,
                    site_latent=site_latent,
                    patch_latent=patch_latent,
                    predicted_next_global=next_pred_prior,
                    path_latent=prior_c,
                    horizon_k=tensors["horizon_k"],
                    current_types=tensors["current_types"],
                    edge_pair_indices=action_edge_pair_neg_indices,
                )
            else:
                action_edge_pair_support_logits = torch.zeros_like(action_edge_pair_logits)
                action_edge_pair_support_neg_logits = torch.zeros_like(action_edge_pair_logits)
                action_edge_pair_support_logits_prior = torch.zeros_like(action_edge_pair_logits)
                action_edge_pair_support_neg_logits_prior = torch.zeros_like(action_edge_pair_logits)
            if hasattr(model, "decode_action_edge_pair_moving_type") and hasattr(model, "decode_action_edge_pair_order"):
                action_edge_pair_moving_type_logits = model.decode_action_edge_pair_moving_type(
                    site_latent=site_latent,
                    patch_latent=patch_latent,
                    predicted_next_global=next_pred,
                    path_latent=post_c,
                    horizon_k=tensors["horizon_k"],
                    current_types=tensors["current_types"],
                    edge_pair_indices=action_edge_pair_indices,
                )
                action_edge_pair_order_logits = model.decode_action_edge_pair_order(
                    site_latent=site_latent,
                    patch_latent=patch_latent,
                    predicted_next_global=next_pred,
                    path_latent=post_c,
                    horizon_k=tensors["horizon_k"],
                    current_types=tensors["current_types"],
                    edge_pair_indices=action_edge_pair_indices,
                )
                action_edge_pair_moving_type_logits_prior = model.decode_action_edge_pair_moving_type(
                    site_latent=site_latent,
                    patch_latent=patch_latent,
                    predicted_next_global=next_pred_prior,
                    path_latent=prior_c,
                    horizon_k=tensors["horizon_k"],
                    current_types=tensors["current_types"],
                    edge_pair_indices=action_edge_pair_indices,
                )
                action_edge_pair_order_logits_prior = model.decode_action_edge_pair_order(
                    site_latent=site_latent,
                    patch_latent=patch_latent,
                    predicted_next_global=next_pred_prior,
                    path_latent=prior_c,
                    horizon_k=tensors["horizon_k"],
                    current_types=tensors["current_types"],
                    edge_pair_indices=action_edge_pair_indices,
                )
            else:
                action_edge_pair_moving_type_logits = torch.zeros(
                    (*action_edge_pair_logits.shape, NUM_SITE_TYPES),
                    device=action_edge_pair_logits.device,
                    dtype=action_edge_pair_logits.dtype,
                )
                action_edge_pair_order_logits = torch.zeros_like(action_edge_pair_logits)
                action_edge_pair_moving_type_logits_prior = torch.zeros_like(action_edge_pair_moving_type_logits)
                action_edge_pair_order_logits_prior = torch.zeros_like(action_edge_pair_logits)
        else:
            action_edge_pair_logits = torch.zeros_like(action_edge_pair_mask)
            action_edge_pair_neg_logits = torch.zeros_like(action_edge_pair_logits)
            action_edge_pair_logits_prior = torch.zeros_like(action_edge_pair_logits)
            action_edge_pair_neg_logits_prior = torch.zeros_like(action_edge_pair_logits)
            action_edge_pair_support_logits = torch.zeros_like(action_edge_pair_logits)
            action_edge_pair_support_neg_logits = torch.zeros_like(action_edge_pair_logits)
            action_edge_pair_support_logits_prior = torch.zeros_like(action_edge_pair_logits)
            action_edge_pair_support_neg_logits_prior = torch.zeros_like(action_edge_pair_logits)
            action_edge_pair_moving_type_logits = torch.zeros(
                (*action_edge_pair_logits.shape, NUM_SITE_TYPES),
                device=action_edge_pair_logits.device,
                dtype=action_edge_pair_logits.dtype,
            )
            action_edge_pair_order_logits = torch.zeros_like(action_edge_pair_logits)
            action_edge_pair_moving_type_logits_prior = torch.zeros_like(action_edge_pair_moving_type_logits)
            action_edge_pair_order_logits_prior = torch.zeros_like(action_edge_pair_logits)
        vacancy_pair_neg_indices = _terminal_vacancy_pair_negative_indices(
            tensors["teacher_vacancy_pair_indices"],
            current_types=tensors["current_types"],
            candidate_mask=tensors["candidate_mask"],
            dense_count=vacancy_pair_negative_count,
            structured_count=vacancy_pair_structured_negative_count,
        )
        if hasattr(model, "decode_vacancy_pairs"):
            vacancy_pair_logits = _decode_action_edge_pair_logits(
                model.decode_vacancy_pairs,
                site_latent=site_latent,
                patch_latent=patch_latent,
                predicted_next_global=next_pred,
                path_latent=post_c,
                horizon_k=tensors["horizon_k"],
                current_types=tensors["current_types"],
                edge_pair_indices=tensors["teacher_vacancy_pair_indices"],
            )
            vacancy_pair_neg_logits = _decode_action_edge_pair_logits(
                model.decode_vacancy_pairs,
                site_latent=site_latent,
                patch_latent=patch_latent,
                predicted_next_global=next_pred,
                path_latent=post_c,
                horizon_k=tensors["horizon_k"],
                current_types=tensors["current_types"],
                edge_pair_indices=vacancy_pair_neg_indices,
            )
            vacancy_pair_logits_prior = _decode_action_edge_pair_logits(
                model.decode_vacancy_pairs,
                site_latent=site_latent,
                patch_latent=patch_latent,
                predicted_next_global=next_pred_prior,
                path_latent=prior_c,
                horizon_k=tensors["horizon_k"],
                current_types=tensors["current_types"],
                edge_pair_indices=tensors["teacher_vacancy_pair_indices"],
            )
            vacancy_pair_neg_logits_prior = _decode_action_edge_pair_logits(
                model.decode_vacancy_pairs,
                site_latent=site_latent,
                patch_latent=patch_latent,
                predicted_next_global=next_pred_prior,
                path_latent=prior_c,
                horizon_k=tensors["horizon_k"],
                current_types=tensors["current_types"],
                edge_pair_indices=vacancy_pair_neg_indices,
            )
            if hasattr(model, "decode_vacancy_pair_interaction"):
                vacancy_pair_interaction_logits = _decode_action_edge_pair_logits(
                    model.decode_vacancy_pair_interaction,
                    site_latent=site_latent,
                    patch_latent=patch_latent,
                    predicted_next_global=next_pred,
                    path_latent=post_c,
                    horizon_k=tensors["horizon_k"],
                    current_types=tensors["current_types"],
                    edge_pair_indices=tensors["teacher_vacancy_pair_indices"],
                )
                vacancy_pair_interaction_neg_logits = _decode_action_edge_pair_logits(
                    model.decode_vacancy_pair_interaction,
                    site_latent=site_latent,
                    patch_latent=patch_latent,
                    predicted_next_global=next_pred,
                    path_latent=post_c,
                    horizon_k=tensors["horizon_k"],
                    current_types=tensors["current_types"],
                    edge_pair_indices=vacancy_pair_neg_indices,
                )
                vacancy_pair_interaction_logits_prior = _decode_action_edge_pair_logits(
                    model.decode_vacancy_pair_interaction,
                    site_latent=site_latent,
                    patch_latent=patch_latent,
                    predicted_next_global=next_pred_prior,
                    path_latent=prior_c,
                    horizon_k=tensors["horizon_k"],
                    current_types=tensors["current_types"],
                    edge_pair_indices=tensors["teacher_vacancy_pair_indices"],
                )
                vacancy_pair_interaction_neg_logits_prior = _decode_action_edge_pair_logits(
                    model.decode_vacancy_pair_interaction,
                    site_latent=site_latent,
                    patch_latent=patch_latent,
                    predicted_next_global=next_pred_prior,
                    path_latent=prior_c,
                    horizon_k=tensors["horizon_k"],
                    current_types=tensors["current_types"],
                    edge_pair_indices=vacancy_pair_neg_indices,
                )
            else:
                vacancy_pair_interaction_logits = torch.zeros_like(vacancy_pair_logits)
                vacancy_pair_interaction_neg_logits = torch.zeros_like(vacancy_pair_neg_logits)
                vacancy_pair_interaction_logits_prior = torch.zeros_like(vacancy_pair_logits)
                vacancy_pair_interaction_neg_logits_prior = torch.zeros_like(vacancy_pair_neg_logits)
            if hasattr(model, "decode_vacancy_pair_moving_type") and hasattr(model, "decode_vacancy_pair_order"):
                vacancy_pair_moving_type_logits = model.decode_vacancy_pair_moving_type(
                    site_latent=site_latent,
                    patch_latent=patch_latent,
                    predicted_next_global=next_pred,
                    path_latent=post_c,
                    horizon_k=tensors["horizon_k"],
                    current_types=tensors["current_types"],
                    edge_pair_indices=tensors["teacher_vacancy_pair_indices"],
                )
                vacancy_pair_order_logits = model.decode_vacancy_pair_order(
                    site_latent=site_latent,
                    patch_latent=patch_latent,
                    predicted_next_global=next_pred,
                    path_latent=post_c,
                    horizon_k=tensors["horizon_k"],
                    current_types=tensors["current_types"],
                    edge_pair_indices=tensors["teacher_vacancy_pair_indices"],
                )
                vacancy_pair_moving_type_logits_prior = model.decode_vacancy_pair_moving_type(
                    site_latent=site_latent,
                    patch_latent=patch_latent,
                    predicted_next_global=next_pred_prior,
                    path_latent=prior_c,
                    horizon_k=tensors["horizon_k"],
                    current_types=tensors["current_types"],
                    edge_pair_indices=tensors["teacher_vacancy_pair_indices"],
                )
                vacancy_pair_order_logits_prior = model.decode_vacancy_pair_order(
                    site_latent=site_latent,
                    patch_latent=patch_latent,
                    predicted_next_global=next_pred_prior,
                    path_latent=prior_c,
                    horizon_k=tensors["horizon_k"],
                    current_types=tensors["current_types"],
                    edge_pair_indices=tensors["teacher_vacancy_pair_indices"],
                )
            else:
                vacancy_pair_moving_type_logits = torch.zeros(
                    (*vacancy_pair_logits.shape, NUM_SITE_TYPES),
                    device=vacancy_pair_logits.device,
                    dtype=vacancy_pair_logits.dtype,
                )
                vacancy_pair_order_logits = torch.zeros_like(vacancy_pair_logits)
                vacancy_pair_moving_type_logits_prior = torch.zeros_like(vacancy_pair_moving_type_logits)
                vacancy_pair_order_logits_prior = torch.zeros_like(vacancy_pair_logits)
        else:
            vacancy_pair_logits = torch.zeros_like(tensors["teacher_vacancy_pair_mask"])
            vacancy_pair_neg_logits = torch.zeros_like(vacancy_pair_logits)
            vacancy_pair_logits_prior = torch.zeros_like(vacancy_pair_logits)
            vacancy_pair_neg_logits_prior = torch.zeros_like(vacancy_pair_logits)
            vacancy_pair_interaction_logits = torch.zeros_like(vacancy_pair_logits)
            vacancy_pair_interaction_neg_logits = torch.zeros_like(vacancy_pair_logits)
            vacancy_pair_interaction_logits_prior = torch.zeros_like(vacancy_pair_logits)
            vacancy_pair_interaction_neg_logits_prior = torch.zeros_like(vacancy_pair_logits)
            vacancy_pair_moving_type_logits = torch.zeros(
                (*vacancy_pair_logits.shape, NUM_SITE_TYPES),
                device=vacancy_pair_logits.device,
                dtype=vacancy_pair_logits.dtype,
            )
            vacancy_pair_order_logits = torch.zeros_like(vacancy_pair_logits)
            vacancy_pair_moving_type_logits_prior = torch.zeros_like(vacancy_pair_moving_type_logits)
            vacancy_pair_order_logits_prior = torch.zeros_like(vacancy_pair_logits)
        candidate_quality_logit = model.decode_candidate_quality(
            site_latent=site_latent,
            patch_latent=patch_latent,
            predicted_next_global=next_pred,
            path_latent=post_c,
            horizon_k=tensors["horizon_k"],
            current_types=tensors["current_types"],
            candidate_mask=tensors["candidate_mask"],
        )
        candidate_quality_logit_prior = model.decode_candidate_quality(
            site_latent=site_latent,
            patch_latent=patch_latent,
            predicted_next_global=next_pred_prior,
            path_latent=prior_c,
            horizon_k=tensors["horizon_k"],
            current_types=tensors["current_types"],
            candidate_mask=tensors["candidate_mask"],
        )
        reward_change_logits, reward_type_logits = _select_reward_edit_context(
            reward_edit_context_source,
            change_logits,
            raw_type_logits,
        )
        reward_change_logits_prior, reward_type_logits_prior = _select_reward_edit_context(
            reward_edit_context_source,
            change_logits_prior,
            raw_type_logits_prior,
        )
        duration_outputs = _predict_reward_and_duration_outputs(
            model,
            global_latent,
            next_pred,
            post_c,
            tensors["global_summary"],
            tensors["horizon_k"],
            detach_duration_inputs=True,
            patch_latent=patch_latent,
            change_logits=reward_change_logits,
            type_logits=reward_type_logits,
            current_types=tensors["current_types"],
            candidate_mask=tensors["candidate_mask"],
        )
        prior_duration_outputs = _predict_reward_and_duration_outputs(
            model,
            global_latent,
            next_pred_prior,
            prior_c,
            tensors["global_summary"],
            tensors["horizon_k"],
            detach_duration_inputs=True,
            patch_latent=patch_latent,
            change_logits=reward_change_logits_prior,
            type_logits=reward_type_logits_prior,
            current_types=tensors["current_types"],
            candidate_mask=tensors["candidate_mask"],
        )
        reward_hat = duration_outputs["reward"]
        tau_mu = duration_outputs["expected_tau_mu"]
        tau_log_sigma = duration_outputs["expected_tau_log_sigma"]
        realized_tau_mu = duration_outputs["realized_tau_mu"]
        realized_tau_log_sigma = duration_outputs["realized_tau_log_sigma"]
        gate_logit = duration_outputs["gate_logit"]
        noop_risk_logit = duration_outputs["noop_risk_logit"]
        reward_hat_prior = prior_duration_outputs["reward"]
        tau_mu_prior = prior_duration_outputs["expected_tau_mu"]
        tau_log_sigma_prior = prior_duration_outputs["expected_tau_log_sigma"]
        realized_tau_mu_prior = prior_duration_outputs["realized_tau_mu"]
        realized_tau_log_sigma_prior = prior_duration_outputs["realized_tau_log_sigma"]
        gate_logit_prior = prior_duration_outputs["gate_logit"]
        noop_risk_logit_prior = prior_duration_outputs["noop_risk_logit"]
        if compute_proj:
            projected_types, _, transport_cost, reachability_violation = project_types_by_inventory(
                current_types=tensors["current_types"],
                change_logits=change_logits,
                type_logits=raw_type_logits,
                node_mask=tensors["candidate_mask"],
                positions=tensors["candidate_positions"],
                box_dims=tensors["box_dims"],
                horizon_k=tensors["horizon_k"],
                max_changed_sites=2 * tensors["horizon_k"],
            )
            projected_changed_mask = ((projected_types != tensors["current_types"]).float() * tensors["candidate_mask"]).detach()
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
            projected_global = _projected_global_latent_batch(batch=batch, projected_types=projected_types, model=model, device=device)
            projected_types_prior, _, _prior_transport_cost, prior_reachability_violation = project_types_by_inventory(
                current_types=tensors["current_types"],
                change_logits=change_logits_prior,
                type_logits=raw_type_logits_prior,
                node_mask=tensors["candidate_mask"],
                positions=tensors["candidate_positions"],
                box_dims=tensors["box_dims"],
                horizon_k=tensors["horizon_k"],
                max_changed_sites=2 * tensors["horizon_k"],
            )
            projected_changed_mask_prior = ((projected_types_prior != tensors["current_types"]).float() * tensors["candidate_mask"]).detach()
            _, projected_patch_latent_prior = model.encode_patch(
                positions=tensors["candidate_positions"],
                nearest_vacancy_offset=tensors["nearest_vacancy_offset"],
                reach_depth=tensors["reach_depth"],
                is_start_vacancy=tensors["is_start_vacancy"],
                type_ids=projected_types_prior,
                node_mask=tensors["candidate_mask"],
                global_summary=tensors["global_summary"],
                box_dims=tensors["box_dims"],
            )
            projected_global_prior = _projected_global_latent_batch(
                batch=batch,
                projected_types=projected_types_prior,
                model=model,
                device=device,
            )

        valid = tensors["candidate_mask"] > 0
        posterior_edit = _edit_supervision_losses(
            change_logits=change_logits,
            type_logits=raw_type_logits,
            current_types=tensors["current_types"],
            target_types=tensors["target_types"],
            changed_mask=tensors["changed_mask"],
            candidate_mask=tensors["candidate_mask"],
            aux_scale=aux_scale,
            sparsity_weight=mask_sparsity_weight,
            count_loss_weight=count_loss_weight,
            noop_change_weight=noop_change_weight,
            noop_type_copy_weight=noop_type_copy_weight,
        )
        prior_edit = _edit_supervision_losses(
            change_logits=change_logits_prior,
            type_logits=raw_type_logits_prior,
            current_types=tensors["current_types"],
            target_types=tensors["target_types"],
            changed_mask=tensors["changed_mask"],
            candidate_mask=tensors["candidate_mask"],
            aux_scale=aux_scale,
            sparsity_weight=mask_sparsity_weight,
            count_loss_weight=count_loss_weight,
            noop_change_weight=noop_change_weight,
            noop_type_copy_weight=noop_type_copy_weight,
        )
        count_loss = posterior_edit["count"]
        pair_count_loss = posterior_edit["pair"]
        if compute_proj:
            proj_mask_loss = _projected_mask_distill_loss(
                change_logits=change_logits,
                projected_changed_mask=projected_changed_mask,
                valid_mask=valid,
                reachability_violation=reachability_violation,
                target_changed_mask=tensors["changed_mask"],
                projected_noop_fp_weight=projected_noop_fp_weight,
            ) * proj_scale
            prior_proj_mask_loss = _projected_mask_distill_loss(
                change_logits=change_logits_prior,
                projected_changed_mask=projected_changed_mask_prior,
                valid_mask=valid,
                reachability_violation=prior_reachability_violation,
                target_changed_mask=tensors["changed_mask"],
                projected_noop_fp_weight=projected_noop_fp_weight,
            ) * proj_scale
        else:
            proj_mask_loss = torch.tensor(0.0, device=device)
            prior_proj_mask_loss = torch.tensor(0.0, device=device)
        mask_loss = posterior_edit["mask"] + 0.5 * proj_mask_loss

        type_loss = posterior_edit["type"]
        atom_to_vac_type_loss = posterior_edit["atom_to_vac_type"]
        vac_to_atom_type_loss = posterior_edit["vac_to_atom_type"]
        prior_pair_count_loss = prior_edit["pair"]
        prior_edit_loss = prior_edit["mask"] + 0.1 * prior_edit["count"] + (0.5 * prior_proj_mask_loss if compute_proj else 0.0) + prior_edit["type"]

        tau_log_target = torch.log(tensors["tau_exp"].clamp(min=1e-12))
        if reward_prediction_source == "projected" and compute_proj:
            projected_change_logits, projected_type_logits = projected_edit_logits_from_types(
                current_types=tensors["current_types"],
                projected_types=projected_types.detach(),
                candidate_mask=tensors["candidate_mask"],
            )
            projected_change_logits, projected_type_logits = _select_reward_edit_context(
                reward_edit_context_source,
                projected_change_logits,
                projected_type_logits,
            )
            projected_duration_outputs = _predict_reward_and_duration_outputs(
                model,
                global_latent,
                next_pred,
                post_c,
                tensors["global_summary"],
                tensors["horizon_k"],
                detach_duration_inputs=True,
                patch_latent=projected_patch_latent.detach() if detach_proj_encoder else projected_patch_latent,
                change_logits=projected_change_logits,
                type_logits=projected_type_logits,
                current_types=tensors["current_types"],
                candidate_mask=tensors["candidate_mask"],
            )
            reward_hat = projected_duration_outputs["reward"]
            gate_logit = projected_duration_outputs["gate_logit"]
            noop_risk_logit = projected_duration_outputs["noop_risk_logit"]
            tau_mu = projected_duration_outputs["expected_tau_mu"]
            tau_log_sigma = projected_duration_outputs["expected_tau_log_sigma"]
            realized_tau_mu = projected_duration_outputs["realized_tau_mu"]
            realized_tau_log_sigma = projected_duration_outputs["realized_tau_log_sigma"]
            projected_change_logits_prior, projected_type_logits_prior = projected_edit_logits_from_types(
                current_types=tensors["current_types"],
                projected_types=projected_types_prior.detach(),
                candidate_mask=tensors["candidate_mask"],
            )
            projected_change_logits_prior, projected_type_logits_prior = _select_reward_edit_context(
                reward_edit_context_source,
                projected_change_logits_prior,
                projected_type_logits_prior,
            )
            projected_prior_duration_outputs = _predict_reward_and_duration_outputs(
                model,
                global_latent,
                next_pred_prior,
                prior_c,
                tensors["global_summary"],
                tensors["horizon_k"],
                detach_duration_inputs=True,
                patch_latent=projected_patch_latent_prior.detach() if detach_proj_encoder else projected_patch_latent_prior,
                change_logits=projected_change_logits_prior,
                type_logits=projected_type_logits_prior,
                current_types=tensors["current_types"],
                candidate_mask=tensors["candidate_mask"],
            )
            reward_hat_prior = projected_prior_duration_outputs["reward"]
            gate_logit_prior = projected_prior_duration_outputs["gate_logit"]
            noop_risk_logit_prior = projected_prior_duration_outputs["noop_risk_logit"]
            tau_mu_prior = projected_prior_duration_outputs["expected_tau_mu"]
            tau_log_sigma_prior = projected_prior_duration_outputs["expected_tau_log_sigma"]
            realized_tau_mu_prior = projected_prior_duration_outputs["realized_tau_mu"]
            realized_tau_log_sigma_prior = projected_prior_duration_outputs["realized_tau_log_sigma"]
        tau_loss = lognormal_nll(tensors["tau_exp"], tau_mu, tau_log_sigma).mean()
        tau_log_mu_loss = F.l1_loss(tau_mu, tau_log_target)
        reward_terms = _reward_supervision_losses(
            reward_hat,
            gate_logit,
            tensors["reward_sum"],
            reward_magnitude_weight=reward_magnitude_weight,
            reward_gated_weight=reward_gated_weight,
            reward_gate_weight=reward_gate_weight,
            reward_zero_weight=reward_zero_weight,
            reward_sign_weight=reward_sign_weight,
        )
        reward_loss = reward_terms["loss"]
        noop_risk_terms = _noop_risk_supervision_loss(noop_risk_logit, tensors["changed_mask"])
        noop_risk_loss = noop_risk_terms["loss"]
        latent_loss = F.smooth_l1_loss(next_pred, next_global)
        if compute_proj:
            proj_state_loss = _projected_state_alignment_loss(
                projected_patch_latent=projected_patch_latent.detach() if detach_proj_encoder else projected_patch_latent,
                target_patch_latent=target_patch_latent.detach(),
                projected_global=projected_global.detach() if detach_proj_encoder else projected_global,
                next_global=next_global,
                next_pred=next_pred.detach(),
                projected_changed_mask=projected_changed_mask,
                reachability_violation=reachability_violation,
            ) * proj_scale
            prior_proj_state_loss = _projected_state_alignment_loss(
                projected_patch_latent=projected_patch_latent_prior.detach() if detach_proj_encoder else projected_patch_latent_prior,
                target_patch_latent=target_patch_latent.detach(),
                projected_global=projected_global_prior.detach() if detach_proj_encoder else projected_global_prior,
                next_global=next_global,
                next_pred=next_pred_prior.detach(),
                projected_changed_mask=projected_changed_mask_prior,
                reachability_violation=prior_reachability_violation,
            ) * proj_scale
        else:
            proj_state_loss = torch.tensor(0.0, device=device)
            prior_proj_state_loss = torch.tensor(0.0, device=device)
        prior_latent_loss = F.smooth_l1_loss(next_pred_prior, next_global) + 0.5 * prior_proj_state_loss
        path_loss = kl_divergence_diag_gaussian(post_mu, post_logvar, prior_mu, prior_logvar).mean()
        prior_tau_loss = lognormal_nll(tensors["tau_exp"], tau_mu_prior, tau_log_sigma_prior).mean()
        prior_tau_log_mu_loss = F.l1_loss(tau_mu_prior, tau_log_target)
        realized_tau_loss = lognormal_nll(tensors["tau_real"], realized_tau_mu, realized_tau_log_sigma).mean()
        prior_realized_tau_loss = lognormal_nll(tensors["tau_real"], realized_tau_mu_prior, realized_tau_log_sigma_prior).mean()
        prior_reward_terms = _reward_supervision_losses(
            reward_hat_prior,
            gate_logit_prior,
            tensors["reward_sum"],
            reward_magnitude_weight=reward_magnitude_weight,
            reward_gated_weight=reward_gated_weight,
            reward_gate_weight=reward_gate_weight,
            reward_zero_weight=reward_zero_weight,
            reward_sign_weight=reward_sign_weight,
        )
        prior_reward_loss = prior_reward_terms["loss"]
        prior_noop_risk_terms = _noop_risk_supervision_loss(noop_risk_logit_prior, tensors["changed_mask"])
        prior_noop_risk_loss = prior_noop_risk_terms["loss"]
        proposal_target = _proposal_target_from_tensors(tensors, proposal_target_source)
        proposal_terms = _proposal_support_loss(
            proposal_logits,
            proposal_target,
            tensors["candidate_mask"],
            hard_negative_mask=tensors["planner_projected_changed_mask"],
            hard_negative_weight=proposal_hard_negative_weight,
            rank_margin_weight=proposal_rank_margin_weight,
            candidate_positive_mask=tensors["planner_candidate_teacher_changed_mask"],
            candidate_false_positive_mask=tensors["planner_candidate_false_positive_mask"],
            candidate_positive_weight=proposal_candidate_positive_weight,
            candidate_false_positive_weight=proposal_candidate_negative_weight,
            candidate_rank_margin_weight=proposal_candidate_rank_margin_weight,
        )
        prior_proposal_terms = _proposal_support_loss(
            proposal_logits_prior,
            proposal_target,
            tensors["candidate_mask"],
            hard_negative_mask=tensors["planner_projected_changed_mask"],
            hard_negative_weight=proposal_hard_negative_weight,
            rank_margin_weight=proposal_rank_margin_weight,
            candidate_positive_mask=tensors["planner_candidate_teacher_changed_mask"],
            candidate_false_positive_mask=tensors["planner_candidate_false_positive_mask"],
            candidate_positive_weight=proposal_candidate_positive_weight,
            candidate_false_positive_weight=proposal_candidate_negative_weight,
            candidate_rank_margin_weight=proposal_candidate_rank_margin_weight,
        )
        action_support_target = _proposal_target_from_tensors(tensors, action_support_target_source)
        action_support_terms = _proposal_support_loss(
            action_support_logits,
            action_support_target,
            tensors["candidate_mask"],
            hard_negative_mask=tensors["planner_projected_changed_mask"],
            hard_negative_weight=action_support_hard_negative_weight,
            rank_margin_weight=action_support_rank_margin_weight,
            candidate_positive_mask=tensors["planner_candidate_teacher_changed_mask"],
            candidate_false_positive_mask=tensors["planner_candidate_false_positive_mask"],
            candidate_positive_weight=action_support_candidate_positive_weight,
            candidate_false_positive_weight=action_support_candidate_negative_weight,
            candidate_rank_margin_weight=action_support_candidate_rank_margin_weight,
        )
        prior_action_support_terms = _proposal_support_loss(
            action_support_logits_prior,
            action_support_target,
            tensors["candidate_mask"],
            hard_negative_mask=tensors["planner_projected_changed_mask"],
            hard_negative_weight=action_support_hard_negative_weight,
            rank_margin_weight=action_support_rank_margin_weight,
            candidate_positive_mask=tensors["planner_candidate_teacher_changed_mask"],
            candidate_false_positive_mask=tensors["planner_candidate_false_positive_mask"],
            candidate_positive_weight=action_support_candidate_positive_weight,
            candidate_false_positive_weight=action_support_candidate_negative_weight,
            candidate_rank_margin_weight=action_support_candidate_rank_margin_weight,
        )
        terminal_edit_target = _proposal_target_from_tensors(tensors, terminal_edit_support_target_source)
        terminal_edit_support_terms = _proposal_support_loss(
            terminal_edit_support_logits,
            terminal_edit_target,
            tensors["candidate_mask"],
            hard_negative_mask=tensors["planner_projected_changed_mask"],
            hard_negative_weight=terminal_edit_support_hard_negative_weight,
            rank_margin_weight=terminal_edit_support_rank_margin_weight,
            candidate_positive_mask=tensors["planner_candidate_teacher_changed_mask"],
            candidate_false_positive_mask=tensors["planner_candidate_false_positive_mask"],
            candidate_positive_weight=terminal_edit_support_candidate_positive_weight,
            candidate_false_positive_weight=terminal_edit_support_candidate_negative_weight,
            candidate_rank_margin_weight=terminal_edit_support_candidate_rank_margin_weight,
        )
        prior_terminal_edit_support_terms = _proposal_support_loss(
            terminal_edit_support_logits_prior,
            terminal_edit_target,
            tensors["candidate_mask"],
            hard_negative_mask=tensors["planner_projected_changed_mask"],
            hard_negative_weight=terminal_edit_support_hard_negative_weight,
            rank_margin_weight=terminal_edit_support_rank_margin_weight,
            candidate_positive_mask=tensors["planner_candidate_teacher_changed_mask"],
            candidate_false_positive_mask=tensors["planner_candidate_false_positive_mask"],
            candidate_positive_weight=terminal_edit_support_candidate_positive_weight,
            candidate_false_positive_weight=terminal_edit_support_candidate_negative_weight,
            candidate_rank_margin_weight=terminal_edit_support_candidate_rank_margin_weight,
        )
        vacancy_displacement_target = _vacancy_displacement_target_from_tensors(tensors)
        terminal_typed_diff_terms = _terminal_typed_diff_loss(
            terminal_typed_diff_logits,
            target_types=tensors["target_types"],
            current_types=tensors["current_types"],
            target_mask=vacancy_displacement_target,
            candidate_mask=tensors["candidate_mask"],
            copy_weight=terminal_typed_diff_copy_weight,
            support_weight=terminal_typed_diff_support_weight,
        )
        prior_terminal_typed_diff_terms = _terminal_typed_diff_loss(
            terminal_typed_diff_logits_prior,
            target_types=tensors["target_types"],
            current_types=tensors["current_types"],
            target_mask=vacancy_displacement_target,
            candidate_mask=tensors["candidate_mask"],
            copy_weight=terminal_typed_diff_copy_weight,
            support_weight=terminal_typed_diff_support_weight,
        )
        action_source_target = _proposal_target_from_tensors(tensors, "action_source")
        action_source_terms = _proposal_support_loss(
            action_source_logits,
            action_source_target,
            tensors["candidate_mask"],
            hard_negative_mask=tensors["planner_projected_changed_mask"],
            hard_negative_weight=action_support_hard_negative_weight,
            rank_margin_weight=action_support_rank_margin_weight,
        )
        prior_action_source_terms = _proposal_support_loss(
            action_source_logits_prior,
            action_source_target,
            tensors["candidate_mask"],
            hard_negative_mask=tensors["planner_projected_changed_mask"],
            hard_negative_weight=action_support_hard_negative_weight,
            rank_margin_weight=action_support_rank_margin_weight,
        )
        action_destination_target = _proposal_target_from_tensors(tensors, "action_destination")
        action_destination_terms = _proposal_support_loss(
            action_destination_logits,
            action_destination_target,
            tensors["candidate_mask"],
            hard_negative_mask=tensors["planner_projected_changed_mask"],
            hard_negative_weight=action_support_hard_negative_weight,
            rank_margin_weight=action_support_rank_margin_weight,
        )
        prior_action_destination_terms = _proposal_support_loss(
            action_destination_logits_prior,
            action_destination_target,
            tensors["candidate_mask"],
            hard_negative_mask=tensors["planner_projected_changed_mask"],
            hard_negative_weight=action_support_hard_negative_weight,
            rank_margin_weight=action_support_rank_margin_weight,
        )
        action_endpoint_terms = _proposal_support_loss(
            action_endpoint_logits,
            _proposal_target_from_tensors(tensors, "action_endpoint"),
            tensors["candidate_mask"],
        )
        prior_action_endpoint_terms = _proposal_support_loss(
            action_endpoint_logits_prior,
            _proposal_target_from_tensors(tensors, "action_endpoint"),
            tensors["candidate_mask"],
        )
        action_edge_pair_terms = _action_edge_pair_supervision_loss(
            action_edge_pair_logits,
            action_edge_pair_neg_logits,
            action_edge_pair_mask,
            negative_weight=action_edge_pair_negative_weight,
            rank_margin_weight=action_edge_pair_rank_margin_weight,
        )
        prior_action_edge_pair_terms = _action_edge_pair_supervision_loss(
            action_edge_pair_logits_prior,
            action_edge_pair_neg_logits_prior,
            action_edge_pair_mask,
            negative_weight=action_edge_pair_negative_weight,
            rank_margin_weight=action_edge_pair_rank_margin_weight,
        )
        action_edge_pair_support_terms = _action_edge_pair_support_loss(
            action_edge_pair_support_logits,
            action_edge_pair_support_neg_logits,
            action_edge_pair_mask,
            action_edge_pair_support_mask,
            negative_weight=action_edge_pair_negative_weight,
            rank_margin_weight=action_edge_pair_rank_margin_weight,
        )
        prior_action_edge_pair_support_terms = _action_edge_pair_support_loss(
            action_edge_pair_support_logits_prior,
            action_edge_pair_support_neg_logits_prior,
            action_edge_pair_mask,
            action_edge_pair_support_mask,
            negative_weight=action_edge_pair_negative_weight,
            rank_margin_weight=action_edge_pair_rank_margin_weight,
        )
        action_edge_pair_semantic_terms = _action_edge_pair_semantic_loss(
            action_edge_pair_moving_type_logits,
            action_edge_pair_order_logits,
            action_edge_pair_mask,
            action_edge_pair_moving_type_target,
            action_edge_pair_order_target,
        )
        prior_action_edge_pair_semantic_terms = _action_edge_pair_semantic_loss(
            action_edge_pair_moving_type_logits_prior,
            action_edge_pair_order_logits_prior,
            action_edge_pair_mask,
            action_edge_pair_moving_type_target,
            action_edge_pair_order_target,
        )
        vacancy_pair_terms = _action_edge_pair_supervision_loss(
            vacancy_pair_logits,
            vacancy_pair_neg_logits,
            tensors["teacher_vacancy_pair_mask"],
            negative_weight=action_edge_pair_negative_weight,
            rank_margin_weight=action_edge_pair_rank_margin_weight,
        )
        prior_vacancy_pair_terms = _action_edge_pair_supervision_loss(
            vacancy_pair_logits_prior,
            vacancy_pair_neg_logits_prior,
            tensors["teacher_vacancy_pair_mask"],
            negative_weight=action_edge_pair_negative_weight,
            rank_margin_weight=action_edge_pair_rank_margin_weight,
        )
        vacancy_pair_listwise_terms = _pair_listwise_contrastive_loss(
            vacancy_pair_logits,
            vacancy_pair_neg_logits,
            tensors["teacher_vacancy_pair_mask"],
        )
        prior_vacancy_pair_listwise_terms = _pair_listwise_contrastive_loss(
            vacancy_pair_logits_prior,
            vacancy_pair_neg_logits_prior,
            tensors["teacher_vacancy_pair_mask"],
        )
        vacancy_pair_interaction_terms = _action_edge_pair_supervision_loss(
            vacancy_pair_interaction_logits,
            vacancy_pair_interaction_neg_logits,
            tensors["teacher_vacancy_pair_mask"],
            negative_weight=action_edge_pair_negative_weight,
            rank_margin_weight=action_edge_pair_rank_margin_weight,
        )
        prior_vacancy_pair_interaction_terms = _action_edge_pair_supervision_loss(
            vacancy_pair_interaction_logits_prior,
            vacancy_pair_interaction_neg_logits_prior,
            tensors["teacher_vacancy_pair_mask"],
            negative_weight=action_edge_pair_negative_weight,
            rank_margin_weight=action_edge_pair_rank_margin_weight,
        )
        vacancy_pair_interaction_listwise_terms = _pair_listwise_contrastive_loss(
            vacancy_pair_interaction_logits,
            vacancy_pair_interaction_neg_logits,
            tensors["teacher_vacancy_pair_mask"],
        )
        prior_vacancy_pair_interaction_listwise_terms = _pair_listwise_contrastive_loss(
            vacancy_pair_interaction_logits_prior,
            vacancy_pair_interaction_neg_logits_prior,
            tensors["teacher_vacancy_pair_mask"],
        )
        vacancy_pair_semantic_terms = _action_edge_pair_semantic_loss(
            vacancy_pair_moving_type_logits,
            vacancy_pair_order_logits,
            tensors["teacher_vacancy_pair_mask"],
            tensors["teacher_vacancy_pair_moving_type"],
            tensors["teacher_vacancy_pair_order"],
        )
        prior_vacancy_pair_semantic_terms = _action_edge_pair_semantic_loss(
            vacancy_pair_moving_type_logits_prior,
            vacancy_pair_order_logits_prior,
            tensors["teacher_vacancy_pair_mask"],
            tensors["teacher_vacancy_pair_moving_type"],
            tensors["teacher_vacancy_pair_order"],
        )
        candidate_quality_terms = _candidate_quality_loss(
            candidate_quality_logit,
            tensors["planner_candidate_quality_target"],
            tensors["planner_candidate_quality_available"],
        )
        prior_candidate_quality_terms = _candidate_quality_loss(
            candidate_quality_logit_prior,
            tensors["planner_candidate_quality_target"],
            tensors["planner_candidate_quality_available"],
        )
        if tau_supervision_mode == "posterior_only":
            combined_tau_loss = tau_loss
            combined_tau_log_mu_loss = tau_log_mu_loss
            combined_realized_tau_loss = realized_tau_loss
            effective_tau_post_scale = 1.0
        else:
            combined_tau_loss = prior_tau_loss
            combined_tau_log_mu_loss = prior_tau_log_mu_loss
            combined_realized_tau_loss = prior_realized_tau_loss
            effective_tau_post_scale = 0.0

        prior_reward_weight = float(weights.get("prior_reward", 0.5 * weights["reward"]))

        main_loss = (
            weights["mask"] * mask_loss
            + weights["type"] * type_loss
            + weights["pair"] * pair_count_loss
            + float(weights.get("prior_pair", 0.0)) * prior_pair_count_loss
            + weights["reward"] * reward_loss
            + float(noop_risk_weight) * noop_risk_loss
            + weights["latent"] * latent_loss
            + weights["proj"] * proj_state_loss
            + weights["path"] * path_loss
            + weights["prior_edit"] * prior_edit_loss
            + weights["prior_latent"] * prior_latent_loss
            + prior_reward_weight * prior_reward_loss
            + float(prior_noop_risk_weight) * prior_noop_risk_loss
            + float(proposal_support_weight) * proposal_terms["loss"]
            + float(prior_proposal_support_weight) * prior_proposal_terms["loss"]
            + float(action_support_weight) * action_support_terms["loss"]
            + float(prior_action_support_weight) * prior_action_support_terms["loss"]
            + float(terminal_edit_support_weight) * terminal_edit_support_terms["loss"]
            + float(prior_terminal_edit_support_weight) * prior_terminal_edit_support_terms["loss"]
            + float(terminal_typed_diff_weight) * terminal_typed_diff_terms["loss"]
            + float(prior_terminal_typed_diff_weight) * prior_terminal_typed_diff_terms["loss"]
            + float(action_source_support_weight) * action_source_terms["loss"]
            + float(prior_action_source_support_weight) * prior_action_source_terms["loss"]
            + float(action_destination_support_weight) * action_destination_terms["loss"]
            + float(prior_action_destination_support_weight) * prior_action_destination_terms["loss"]
            + float(action_edge_pair_weight) * action_edge_pair_terms["loss"]
            + float(prior_action_edge_pair_weight) * prior_action_edge_pair_terms["loss"]
            + float(action_edge_pair_support_weight) * action_edge_pair_support_terms["loss"]
            + float(prior_action_edge_pair_support_weight) * prior_action_edge_pair_support_terms["loss"]
            + float(action_edge_pair_semantic_weight) * action_edge_pair_semantic_terms["loss"]
            + float(prior_action_edge_pair_semantic_weight) * prior_action_edge_pair_semantic_terms["loss"]
            + float(vacancy_pair_weight) * vacancy_pair_terms["loss"]
            + float(prior_vacancy_pair_weight) * prior_vacancy_pair_terms["loss"]
            + float(vacancy_pair_listwise_weight) * vacancy_pair_listwise_terms["loss"]
            + float(prior_vacancy_pair_listwise_weight) * prior_vacancy_pair_listwise_terms["loss"]
            + float(vacancy_pair_interaction_weight) * vacancy_pair_interaction_terms["loss"]
            + float(prior_vacancy_pair_interaction_weight) * prior_vacancy_pair_interaction_terms["loss"]
            + float(vacancy_pair_interaction_listwise_weight) * vacancy_pair_interaction_listwise_terms["loss"]
            + float(prior_vacancy_pair_interaction_listwise_weight)
            * prior_vacancy_pair_interaction_listwise_terms["loss"]
            + float(vacancy_pair_semantic_weight) * vacancy_pair_semantic_terms["loss"]
            + float(prior_vacancy_pair_semantic_weight) * prior_vacancy_pair_semantic_terms["loss"]
            + float(candidate_quality_weight) * candidate_quality_terms["loss"]
            + float(prior_candidate_quality_weight) * prior_candidate_quality_terms["loss"]
        )
        realized_tau_weight = float(weights.get("realized_tau", 0.0))
        tau_log_mu_weight = float(weights.get("tau_log_mu", 0.0))
        time_total_loss = (
            weights["tau"] * combined_tau_loss
            + tau_log_mu_weight * combined_tau_log_mu_loss
            + realized_tau_weight * combined_realized_tau_loss
        )
        loss = main_loss + time_total_loss
        optimizer.zero_grad()
        if loss.requires_grad:
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        batch_size = len(batch)
        count += batch_size
        logs["loss"] += float(loss.item()) * batch_size
        logs["mask"] += float(mask_loss.item()) * batch_size
        logs["count"] += float(count_loss.item()) * batch_size
        logs["pair"] += float(pair_count_loss.item()) * batch_size
        logs["prior_pair"] += float(prior_pair_count_loss.item()) * batch_size
        logs["proj_mask"] += float(proj_mask_loss.item()) * batch_size
        logs["type"] += float(type_loss.item()) * batch_size
        logs["noop_change"] += float(posterior_edit["noop_change"].item()) * batch_size
        logs["noop_type_copy"] += float(posterior_edit["noop_type_copy"].item()) * batch_size
        logs["atom_to_vac_type"] += float(atom_to_vac_type_loss.item()) * batch_size
        logs["vac_to_atom_type"] += float(vac_to_atom_type_loss.item()) * batch_size
        logs["tau"] += float(combined_tau_loss.item()) * batch_size
        logs["tau_log_mu"] += float(combined_tau_log_mu_loss.item()) * batch_size
        logs["tau_post"] += float(tau_loss.item()) * batch_size
        logs["tau_prior"] += float(prior_tau_loss.item()) * batch_size
        logs["realized_tau"] += float(combined_realized_tau_loss.item()) * batch_size
        logs["realized_tau_post"] += float(realized_tau_loss.item()) * batch_size
        logs["realized_tau_prior"] += float(prior_realized_tau_loss.item()) * batch_size
        logs["tau_post_scale"] += float(effective_tau_post_scale) * batch_size
        logs["reward"] += float(reward_loss.item()) * batch_size
        logs["noop_risk"] += float(noop_risk_loss.item()) * batch_size
        logs["prior_noop_risk"] += float(prior_noop_risk_loss.item()) * batch_size
        logs["proposal"] += float(proposal_terms["loss"].item()) * batch_size
        logs["prior_proposal"] += float(prior_proposal_terms["loss"].item()) * batch_size
        logs["proposal_hard_negative"] += float(proposal_terms["hard_negative"].item()) * batch_size
        logs["prior_proposal_hard_negative"] += float(prior_proposal_terms["hard_negative"].item()) * batch_size
        logs["proposal_rank_margin"] += float(proposal_terms["rank_margin"].item()) * batch_size
        logs["prior_proposal_rank_margin"] += float(prior_proposal_terms["rank_margin"].item()) * batch_size
        logs["proposal_candidate_positive"] += float(proposal_terms["candidate_positive"].item()) * batch_size
        logs["prior_proposal_candidate_positive"] += float(prior_proposal_terms["candidate_positive"].item()) * batch_size
        logs["proposal_candidate_negative"] += float(proposal_terms["candidate_false_positive"].item()) * batch_size
        logs["prior_proposal_candidate_negative"] += float(prior_proposal_terms["candidate_false_positive"].item()) * batch_size
        logs["proposal_candidate_rank_margin"] += float(proposal_terms["candidate_rank_margin"].item()) * batch_size
        logs["prior_proposal_candidate_rank_margin"] += float(prior_proposal_terms["candidate_rank_margin"].item()) * batch_size
        logs["action_support"] += float(action_support_terms["loss"].item()) * batch_size
        logs["prior_action_support"] += float(prior_action_support_terms["loss"].item()) * batch_size
        logs["action_support_hard_negative"] += float(action_support_terms["hard_negative"].item()) * batch_size
        logs["prior_action_support_hard_negative"] += float(prior_action_support_terms["hard_negative"].item()) * batch_size
        logs["action_support_rank_margin"] += float(action_support_terms["rank_margin"].item()) * batch_size
        logs["prior_action_support_rank_margin"] += float(prior_action_support_terms["rank_margin"].item()) * batch_size
        logs["action_support_candidate_positive"] += float(action_support_terms["candidate_positive"].item()) * batch_size
        logs["prior_action_support_candidate_positive"] += float(
            prior_action_support_terms["candidate_positive"].item()
        ) * batch_size
        logs["action_support_candidate_negative"] += float(
            action_support_terms["candidate_false_positive"].item()
        ) * batch_size
        logs["prior_action_support_candidate_negative"] += float(
            prior_action_support_terms["candidate_false_positive"].item()
        ) * batch_size
        logs["action_support_candidate_rank_margin"] += float(
            action_support_terms["candidate_rank_margin"].item()
        ) * batch_size
        logs["prior_action_support_candidate_rank_margin"] += float(
            prior_action_support_terms["candidate_rank_margin"].item()
        ) * batch_size
        logs["terminal_edit_support"] += float(terminal_edit_support_terms["loss"].item()) * batch_size
        logs["prior_terminal_edit_support"] += float(prior_terminal_edit_support_terms["loss"].item()) * batch_size
        logs["terminal_edit_topk_f1"] += float(terminal_edit_support_terms["topk_f1"].item()) * batch_size
        logs["terminal_edit_recall32"] += float(terminal_edit_support_terms["recall32"].item()) * batch_size
        logs["terminal_edit_candidate_positive"] += float(
            terminal_edit_support_terms["candidate_positive"].item()
        ) * batch_size
        logs["terminal_edit_candidate_negative"] += float(
            terminal_edit_support_terms["candidate_false_positive"].item()
        ) * batch_size
        logs["terminal_edit_candidate_rank_margin"] += float(
            terminal_edit_support_terms["candidate_rank_margin"].item()
        ) * batch_size
        logs["terminal_typed_diff"] += float(terminal_typed_diff_terms["loss"].item()) * batch_size
        logs["prior_terminal_typed_diff"] += float(prior_terminal_typed_diff_terms["loss"].item()) * batch_size
        logs["terminal_typed_diff_support"] += float(terminal_typed_diff_terms["support"].item()) * batch_size
        logs["prior_terminal_typed_diff_support"] += float(prior_terminal_typed_diff_terms["support"].item()) * batch_size
        logs["terminal_typed_diff_type_acc"] += float(terminal_typed_diff_terms["type_acc"].item()) * batch_size
        logs["terminal_typed_diff_copy_acc"] += float(terminal_typed_diff_terms["copy_acc"].item()) * batch_size
        logs["terminal_typed_diff_topk_f1"] += float(terminal_typed_diff_terms["topk_f1"].item()) * batch_size
        logs["terminal_typed_diff_recall32"] += float(terminal_typed_diff_terms["recall32"].item()) * batch_size
        logs["candidate_quality"] += float(candidate_quality_terms["loss"].item()) * batch_size
        logs["prior_candidate_quality"] += float(prior_candidate_quality_terms["loss"].item()) * batch_size
        logs["candidate_quality_mae"] += float(prior_candidate_quality_terms["mae"].item()) * batch_size
        logs["candidate_quality_corr"] += float(prior_candidate_quality_terms["corr"].item()) * batch_size
        logs["candidate_quality_pred_mean"] += float(prior_candidate_quality_terms["pred_mean"].item()) * batch_size
        logs["candidate_quality_target_mean"] += float(prior_candidate_quality_terms["target_mean"].item()) * batch_size
        logs["candidate_quality_available_frac"] += float(prior_candidate_quality_terms["available_frac"].item()) * batch_size
        logs["proposal_topk_f1"] += float(proposal_terms["topk_f1"].item()) * batch_size
        logs["proposal_recall32"] += float(proposal_terms["recall32"].item()) * batch_size
        logs["action_support_topk_f1"] += float(action_support_terms["topk_f1"].item()) * batch_size
        logs["action_support_recall32"] += float(action_support_terms["recall32"].item()) * batch_size
        logs["action_source"] += float(action_source_terms["loss"].item()) * batch_size
        logs["prior_action_source"] += float(prior_action_source_terms["loss"].item()) * batch_size
        logs["action_source_topk_f1"] += float(action_source_terms["topk_f1"].item()) * batch_size
        logs["action_source_recall32"] += float(action_source_terms["recall32"].item()) * batch_size
        logs["action_destination"] += float(action_destination_terms["loss"].item()) * batch_size
        logs["prior_action_destination"] += float(prior_action_destination_terms["loss"].item()) * batch_size
        logs["action_destination_topk_f1"] += float(action_destination_terms["topk_f1"].item()) * batch_size
        logs["action_destination_recall32"] += float(action_destination_terms["recall32"].item()) * batch_size
        logs["action_endpoint_topk_f1"] += float(action_endpoint_terms["topk_f1"].item()) * batch_size
        logs["action_endpoint_recall32"] += float(action_endpoint_terms["recall32"].item()) * batch_size
        logs["prior_action_endpoint_topk_f1"] += float(prior_action_endpoint_terms["topk_f1"].item()) * batch_size
        logs["prior_action_endpoint_recall32"] += float(prior_action_endpoint_terms["recall32"].item()) * batch_size
        logs["action_edge_pair"] += float(action_edge_pair_terms["loss"].item()) * batch_size
        logs["prior_action_edge_pair"] += float(prior_action_edge_pair_terms["loss"].item()) * batch_size
        logs["action_edge_pair_positive"] += float(action_edge_pair_terms["positive"].item()) * batch_size
        logs["action_edge_pair_negative"] += float(action_edge_pair_terms["negative"].item()) * batch_size
        logs["action_edge_pair_rank_margin"] += float(action_edge_pair_terms["rank_margin"].item()) * batch_size
        logs["action_edge_pair_rank_acc"] += float(action_edge_pair_terms["rank_acc"].item()) * batch_size
        logs["action_edge_pair_available_frac"] += float(action_edge_pair_terms["available_frac"].item()) * batch_size
        logs["action_edge_pair_support"] += float(action_edge_pair_support_terms["loss"].item()) * batch_size
        logs["prior_action_edge_pair_support"] += float(
            prior_action_edge_pair_support_terms["loss"].item()
        ) * batch_size
        logs["action_edge_pair_support_bce"] += float(
            action_edge_pair_support_terms["pair_bce"].item()
        ) * batch_size
        logs["action_edge_pair_support_negative"] += float(
            action_edge_pair_support_terms["negative"].item()
        ) * batch_size
        logs["action_edge_pair_support_rank_margin"] += float(
            action_edge_pair_support_terms["rank_margin"].item()
        ) * batch_size
        logs["action_edge_pair_support_rank_acc"] += float(
            action_edge_pair_support_terms["rank_acc"].item()
        ) * batch_size
        logs["action_edge_pair_support_prob"] += float(
            action_edge_pair_support_terms["support_prob"].item()
        ) * batch_size
        logs["action_edge_pair_support_nonsupport_prob"] += float(
            action_edge_pair_support_terms["nonsupport_prob"].item()
        ) * batch_size
        logs["action_edge_pair_support_frac"] += float(
            action_edge_pair_support_terms["support_frac"].item()
        ) * batch_size
        logs["action_edge_pair_semantic"] += float(action_edge_pair_semantic_terms["loss"].item()) * batch_size
        logs["prior_action_edge_pair_semantic"] += float(
            prior_action_edge_pair_semantic_terms["loss"].item()
        ) * batch_size
        logs["action_edge_pair_moving_type"] += float(
            action_edge_pair_semantic_terms["moving_type"].item()
        ) * batch_size
        logs["prior_action_edge_pair_moving_type"] += float(
            prior_action_edge_pair_semantic_terms["moving_type"].item()
        ) * batch_size
        logs["action_edge_pair_order"] += float(action_edge_pair_semantic_terms["order"].item()) * batch_size
        logs["prior_action_edge_pair_order"] += float(
            prior_action_edge_pair_semantic_terms["order"].item()
        ) * batch_size
        logs["action_edge_pair_moving_type_acc"] += float(
            action_edge_pair_semantic_terms["moving_type_acc"].item()
        ) * batch_size
        logs["action_edge_pair_order_mae"] += float(action_edge_pair_semantic_terms["order_mae"].item()) * batch_size
        logs["vacancy_pair"] += float(vacancy_pair_terms["loss"].item()) * batch_size
        logs["prior_vacancy_pair"] += float(prior_vacancy_pair_terms["loss"].item()) * batch_size
        logs["vacancy_pair_positive"] += float(vacancy_pair_terms["positive"].item()) * batch_size
        logs["vacancy_pair_negative"] += float(vacancy_pair_terms["negative"].item()) * batch_size
        logs["vacancy_pair_rank_margin"] += float(vacancy_pair_terms["rank_margin"].item()) * batch_size
        logs["vacancy_pair_rank_acc"] += float(vacancy_pair_terms["rank_acc"].item()) * batch_size
        logs["vacancy_pair_available_frac"] += float(vacancy_pair_terms["available_frac"].item()) * batch_size
        logs["vacancy_pair_listwise"] += float(vacancy_pair_listwise_terms["loss"].item()) * batch_size
        logs["prior_vacancy_pair_listwise"] += float(
            prior_vacancy_pair_listwise_terms["loss"].item()
        ) * batch_size
        logs["vacancy_pair_listwise_acc"] += float(vacancy_pair_listwise_terms["acc"].item()) * batch_size
        logs["prior_vacancy_pair_listwise_acc"] += float(
            prior_vacancy_pair_listwise_terms["acc"].item()
        ) * batch_size
        logs["vacancy_pair_interaction"] += float(vacancy_pair_interaction_terms["loss"].item()) * batch_size
        logs["prior_vacancy_pair_interaction"] += float(
            prior_vacancy_pair_interaction_terms["loss"].item()
        ) * batch_size
        logs["vacancy_pair_interaction_rank_acc"] += float(
            vacancy_pair_interaction_terms["rank_acc"].item()
        ) * batch_size
        logs["prior_vacancy_pair_interaction_rank_acc"] += float(
            prior_vacancy_pair_interaction_terms["rank_acc"].item()
        ) * batch_size
        logs["vacancy_pair_interaction_listwise"] += float(
            vacancy_pair_interaction_listwise_terms["loss"].item()
        ) * batch_size
        logs["prior_vacancy_pair_interaction_listwise"] += float(
            prior_vacancy_pair_interaction_listwise_terms["loss"].item()
        ) * batch_size
        logs["vacancy_pair_interaction_listwise_acc"] += float(
            vacancy_pair_interaction_listwise_terms["acc"].item()
        ) * batch_size
        logs["prior_vacancy_pair_interaction_listwise_acc"] += float(
            prior_vacancy_pair_interaction_listwise_terms["acc"].item()
        ) * batch_size
        logs["vacancy_pair_semantic"] += float(vacancy_pair_semantic_terms["loss"].item()) * batch_size
        logs["prior_vacancy_pair_semantic"] += float(
            prior_vacancy_pair_semantic_terms["loss"].item()
        ) * batch_size
        logs["vacancy_pair_moving_type"] += float(vacancy_pair_semantic_terms["moving_type"].item()) * batch_size
        logs["prior_vacancy_pair_moving_type"] += float(
            prior_vacancy_pair_semantic_terms["moving_type"].item()
        ) * batch_size
        logs["vacancy_pair_order"] += float(vacancy_pair_semantic_terms["order"].item()) * batch_size
        logs["prior_vacancy_pair_order"] += float(prior_vacancy_pair_semantic_terms["order"].item()) * batch_size
        logs["vacancy_pair_moving_type_acc"] += float(
            vacancy_pair_semantic_terms["moving_type_acc"].item()
        ) * batch_size
        logs["vacancy_pair_order_mae"] += float(vacancy_pair_semantic_terms["order_mae"].item()) * batch_size
        logs["noop_risk_target_frac"] += float(noop_risk_terms["target_frac"].item()) * batch_size
        logs["noop_risk_noop_prob"] += float(noop_risk_terms["noop_prob"].item()) * batch_size
        logs["noop_risk_nonnoop_prob"] += float(noop_risk_terms["nonnoop_prob"].item()) * batch_size
        logs["latent"] += float(latent_loss.item()) * batch_size
        logs["proj"] += float(proj_state_loss.item()) * batch_size
        logs["path"] += float(path_loss.item()) * batch_size
        logs["prior_edit"] += float(prior_edit_loss.item()) * batch_size
        logs["prior_latent"] += float(prior_latent_loss.item()) * batch_size
        logs["mask_aux_scale"] += float(aux_scale) * batch_size

    if count == 0:
        return logs
    if float(weights.get("realized_tau", 0.0)) > 0.0:
        setattr(model, "realized_tau_head_loaded", True)
    return {key: value / count for key, value in logs.items()}


def _build_loader(samples: list[MacroSegmentSample], batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(MacroSegmentDataset(samples), batch_size=batch_size, shuffle=shuffle, collate_fn=lambda batch: batch)


def _initialize_reward_heads(model: MacroDreamerEditModel, train_samples: list[MacroSegmentSample]) -> None:
    if not train_samples:
        return
    reward_values = np.asarray([sample.reward_sum for sample in train_samples], dtype=np.float64)
    nonzero_frac = float(np.mean(np.abs(reward_values) > 1e-6))
    nonzero_frac = min(max(nonzero_frac, 0.01), 0.99)
    gate_bias = math.log(nonzero_frac) - math.log(1.0 - nonzero_frac)
    zero_frac = float(np.mean(np.isclose(reward_values, 0.0, atol=1e-8)))
    pos_frac = float(np.mean(reward_values > 1e-6))
    neg_frac = float(np.mean(reward_values < -1e-6))
    noop_values = []
    for sample in train_samples:
        changed_mask = getattr(sample, "changed_mask", None)
        if changed_mask is not None:
            noop_values.append(float(np.sum(changed_mask) <= 0.0))
        else:
            # Lightweight tests and older helpers may only carry reward_sum.
            # Use zero reward as a conservative no-op proxy for initializing
            # the independent risk-head bias; real training samples use
            # changed_mask above.
            noop_values.append(float(abs(float(getattr(sample, "reward_sum", 0.0))) <= 1e-8))
    noop_values = np.asarray(noop_values, dtype=np.float64)
    noop_frac = min(max(float(np.mean(noop_values)), 0.01), 0.99)
    noop_bias = math.log(noop_frac) - math.log(1.0 - noop_frac)
    with torch.no_grad():
        model.reward_head[-1].weight.zero_()
        model.reward_head[-1].bias.zero_()
        model.reward_context_head[-1].weight.zero_()
        model.reward_context_head[-1].bias.zero_()
        model.reward_gate_head[-1].weight.zero_()
        model.reward_gate_head[-1].bias.fill_(gate_bias)
        if hasattr(model, "reward_gate_context_head"):
            model.reward_gate_context_head[-1].weight.zero_()
            model.reward_gate_context_head[-1].bias.zero_()
        if hasattr(model, "noop_risk_head"):
            model.noop_risk_head[-1].weight.zero_()
            model.noop_risk_head[-1].bias.fill_(noop_bias)
        if hasattr(model, "noop_risk_context_head"):
            model.noop_risk_context_head[-1].weight.zero_()
            model.noop_risk_context_head[-1].bias.zero_()
    print(
        "  Reward init: "
        f"mean={float(np.mean(reward_values)):.3f} zero_frac={zero_frac:.3f} "
        f"pos_frac={pos_frac:.3f} neg_frac={neg_frac:.3f} "
        f"nonzero_frac={nonzero_frac:.3f} gate_bias={gate_bias:.3f} "
        f"noop_frac={noop_frac:.3f} noop_bias={noop_bias:.3f}"
    )


def _initialize_output_heads(model: MacroDreamerEditModel, train_samples: list[MacroSegmentSample]) -> None:
    if not train_samples:
        return
    tau_residuals = []
    realized_tau_residuals = []
    for sample in train_samples:
        baseline_log_tau = math.log(max(int(sample.horizon_k), 1)) - float(sample.global_summary[10])
        tau_residuals.append(math.log(sample.tau_exp + 1e-12) - baseline_log_tau)
        realized_tau_residuals.append(math.log(sample.tau_real + 1e-12) - baseline_log_tau)
    tau_residual_mean = float(np.mean(tau_residuals))
    tau_residual_std = max(float(np.std(tau_residuals)), 0.2)
    realized_tau_residual_mean = float(np.mean(realized_tau_residuals))
    realized_tau_residual_std = max(float(np.std(realized_tau_residuals)), 0.35)
    valid_site_count = float(np.sum([sample.candidate_mask.sum() for sample in train_samples]))
    changed_site_count = float(np.sum([sample.changed_mask.sum() for sample in train_samples]))
    changed_rate = changed_site_count / max(valid_site_count, 1.0)
    sparse_prior = min(max(changed_rate, 1e-4), 5e-2)
    change_bias = math.log(sparse_prior) - math.log(1.0 - sparse_prior)
    with torch.no_grad():
        model.change_head.weight.zero_()
        model.change_head.bias.fill_(change_bias)
        model.type_head.weight.zero_()
        model.type_head.bias.zero_()
        model.duration_head[-1].weight.zero_()
        model.duration_head[-1].bias[0] = tau_residual_mean
        model.duration_head[-1].bias[1] = float(np.log(tau_residual_std))
        model.realized_duration_head[-1].weight.zero_()
        model.realized_duration_head[-1].bias[0] = realized_tau_residual_mean
        model.realized_duration_head[-1].bias[1] = float(np.log(realized_tau_residual_std))
        model.realized_tau_head_loaded = True
    _initialize_reward_heads(model, train_samples)


def _apply_output_head_initialization_policy(
    model: MacroDreamerEditModel,
    train_samples: list[MacroSegmentSample],
    *,
    resume: Optional[str],
    init_from: Optional[str],
    reinit_output_heads: bool,
    reinit_reward_heads: bool,
    freeze_duration_heads: bool,
) -> None:
    should_initialize_output_heads = (not resume) and (init_from is None or reinit_output_heads)
    if should_initialize_output_heads:
        _initialize_output_heads(model, train_samples)
    elif (not resume) and reinit_reward_heads:
        _initialize_reward_heads(model, train_samples)
    elif init_from is not None and not resume:
        print("  Output heads preserved from --init_from checkpoint", flush=True)
    if freeze_duration_heads:
        for module in (
            model.duration_head,
            model.realized_duration_head,
            model.duration_context_head,
            model.realized_duration_context_head,
        ):
            for param in module.parameters():
                param.requires_grad_(False)
        print("  Duration heads frozen", flush=True)


def _apply_reward_heads_only_training(model: MacroDreamerEditModel) -> None:
    reward_prefixes = (
        "reward_head.",
        "reward_context_head.",
        "reward_gate_head.",
        "reward_gate_context_head.",
    )
    trainable = 0
    total = 0
    for name, param in model.named_parameters():
        total += param.numel()
        is_reward_param = name.startswith(reward_prefixes)
        param.requires_grad_(is_reward_param)
        if is_reward_param:
            trainable += param.numel()
    print(f"  Reward-head-only training: trainable_params={trainable} / {total}", flush=True)


def _apply_noop_risk_heads_only_training(model: MacroDreamerEditModel) -> None:
    trainable_prefixes = (
        "noop_risk_head.",
        "noop_risk_context_head.",
    )
    trainable = 0
    total = 0
    for name, param in model.named_parameters():
        total += param.numel()
        should_train = name.startswith(trainable_prefixes)
        param.requires_grad_(should_train)
        if should_train:
            trainable += param.numel()
    print(f"  No-op-risk-head-only training: trainable_params={trainable} / {total}", flush=True)


def _apply_edit_heads_only_training(model: MacroDreamerEditModel) -> None:
    trainable_prefixes = (
        "edit_decoder.",
        "change_head.",
        "type_head.",
    )
    trainable = 0
    total = 0
    for name, param in model.named_parameters():
        total += param.numel()
        should_train = name.startswith(trainable_prefixes)
        param.requires_grad_(should_train)
        if should_train:
            trainable += param.numel()
    print(f"  Edit-head-only training: trainable_params={trainable} / {total}", flush=True)


def _apply_proposal_head_only_training(model: MacroDreamerEditModel) -> None:
    trainable_prefixes = ("proposal_head.",)
    trainable = 0
    total = 0
    for name, param in model.named_parameters():
        total += param.numel()
        should_train = name.startswith(trainable_prefixes)
        param.requires_grad_(should_train)
        if should_train:
            trainable += param.numel()
    print(f"  Proposal-head-only training: trainable_params={trainable} / {total}", flush=True)


def _apply_action_support_head_only_training(model: MacroDreamerEditModel) -> None:
    trainable_prefixes = ("action_support_head.",)
    trainable = 0
    total = 0
    for name, param in model.named_parameters():
        total += param.numel()
        should_train = name.startswith(trainable_prefixes)
        param.requires_grad_(should_train)
        if should_train:
            trainable += param.numel()
    print(f"  Action-support-head-only training: trainable_params={trainable} / {total}", flush=True)


def _apply_action_endpoint_heads_only_training(model: MacroDreamerEditModel) -> None:
    trainable_prefixes = (
        "action_source_head.",
        "action_destination_head.",
    )
    trainable = 0
    total = 0
    for name, param in model.named_parameters():
        total += param.numel()
        should_train = name.startswith(trainable_prefixes)
        param.requires_grad_(should_train)
        if should_train:
            trainable += param.numel()
    print(f"  Action-endpoint-heads-only training: trainable_params={trainable} / {total}", flush=True)


def _apply_action_edge_pair_head_only_training(model: MacroDreamerEditModel) -> None:
    trainable_prefixes = ("action_edge_pair_head.",)
    trainable = 0
    total = 0
    for name, param in model.named_parameters():
        total += param.numel()
        should_train = name.startswith(trainable_prefixes)
        param.requires_grad_(should_train)
        if should_train:
            trainable += param.numel()
    print(f"  Action-edge-pair-head-only training: trainable_params={trainable} / {total}", flush=True)


def _apply_action_edge_pair_support_head_only_training(model: MacroDreamerEditModel) -> None:
    trainable_prefixes = ("action_edge_pair_support_head.",)
    trainable = 0
    total = 0
    for name, param in model.named_parameters():
        total += param.numel()
        should_train = name.startswith(trainable_prefixes)
        param.requires_grad_(should_train)
        if should_train:
            trainable += param.numel()
    print(f"  Action-edge-pair-support-head-only training: trainable_params={trainable} / {total}", flush=True)


def _apply_action_edge_pair_dual_heads_only_training(model: MacroDreamerEditModel) -> None:
    trainable_prefixes = ("action_edge_pair_head.", "action_edge_pair_support_head.")
    trainable = 0
    total = 0
    for name, param in model.named_parameters():
        total += param.numel()
        should_train = name.startswith(trainable_prefixes)
        param.requires_grad_(should_train)
        if should_train:
            trainable += param.numel()
    print(f"  Action-edge-pair-dual-heads-only training: trainable_params={trainable} / {total}", flush=True)


def _apply_action_edge_pair_listwise_heads_only_training(model: MacroDreamerEditModel) -> None:
    trainable_prefixes = (
        "action_edge_pair_head.",
        "action_edge_pair_support_head.",
        "action_edge_pair_moving_type_head.",
        "action_edge_pair_order_head.",
        "candidate_quality_head.",
    )
    trainable = 0
    total = 0
    for name, param in model.named_parameters():
        total += param.numel()
        should_train = name.startswith(trainable_prefixes)
        param.requires_grad_(should_train)
        if should_train:
            trainable += param.numel()
    print(f"  Action-edge-pair-listwise-heads-only training: trainable_params={trainable} / {total}", flush=True)


def _apply_vacancy_pair_heads_only_training(model: MacroDreamerEditModel) -> None:
    trainable_prefixes = (
        "vacancy_pair_head.",
        "vacancy_pair_moving_type_head.",
        "vacancy_pair_order_head.",
    )
    trainable = 0
    total = 0
    for name, param in model.named_parameters():
        total += param.numel()
        should_train = name.startswith(trainable_prefixes)
        param.requires_grad_(should_train)
        if should_train:
            trainable += param.numel()
    print(f"  Vacancy-pair-heads-only training: trainable_params={trainable} / {total}", flush=True)


def _apply_vacancy_pair_interaction_head_only_training(model: MacroDreamerEditModel) -> None:
    trainable_prefixes = ("vacancy_pair_interaction_head.",)
    trainable = 0
    total = 0
    for name, param in model.named_parameters():
        total += param.numel()
        should_train = name.startswith(trainable_prefixes)
        param.requires_grad_(should_train)
        if should_train:
            trainable += param.numel()
    print(f"  Vacancy-pair-interaction-head-only training: trainable_params={trainable} / {total}", flush=True)


def _apply_candidate_quality_head_only_training(model: MacroDreamerEditModel) -> None:
    trainable_prefixes = ("candidate_quality_head.",)
    trainable = 0
    total = 0
    for name, param in model.named_parameters():
        total += param.numel()
        should_train = name.startswith(trainable_prefixes)
        param.requires_grad_(should_train)
        if should_train:
            trainable += param.numel()
    print(f"  Candidate-quality-head-only training: trainable_params={trainable} / {total}", flush=True)


def _apply_terminal_edit_support_head_only_training(model: MacroDreamerEditModel) -> None:
    trainable_prefixes = ("terminal_edit_support_head.",)
    trainable = 0
    total = 0
    for name, param in model.named_parameters():
        total += param.numel()
        should_train = name.startswith(trainable_prefixes)
        param.requires_grad_(should_train)
        if should_train:
            trainable += param.numel()
    print(f"  Terminal-edit-support-head-only training: trainable_params={trainable} / {total}", flush=True)


def _apply_terminal_typed_diff_head_only_training(model: MacroDreamerEditModel) -> None:
    trainable_prefixes = ("terminal_typed_diff_head.",)
    trainable = 0
    total = 0
    for name, param in model.named_parameters():
        total += param.numel()
        should_train = name.startswith(trainable_prefixes)
        param.requires_grad_(should_train)
        if should_train:
            trainable += param.numel()
    print(f"  Terminal-typed-diff-head-only training: trainable_params={trainable} / {total}", flush=True)


def _initialize_proposal_head_from_change_head(model: MacroDreamerEditModel) -> None:
    with torch.no_grad():
        model.proposal_head.weight.copy_(model.change_head.weight)
        model.proposal_head.bias.copy_(model.change_head.bias)
    print("  Initialized proposal_head from change_head", flush=True)


def _initialize_action_support_head_from_proposal_head(model: MacroDreamerEditModel) -> None:
    with torch.no_grad():
        model.action_support_head.weight.copy_(model.proposal_head.weight)
        model.action_support_head.bias.copy_(model.proposal_head.bias)
    print("  Initialized action_support_head from proposal_head", flush=True)


def _initialize_action_support_head_from_change_head(model: MacroDreamerEditModel) -> None:
    with torch.no_grad():
        model.action_support_head.weight.copy_(model.change_head.weight)
        model.action_support_head.bias.copy_(model.change_head.bias)
    print("  Initialized action_support_head from change_head", flush=True)


def _initialize_action_endpoint_heads_from_action_support_head(model: MacroDreamerEditModel) -> None:
    with torch.no_grad():
        model.action_source_head.weight.copy_(model.action_support_head.weight)
        model.action_source_head.bias.copy_(model.action_support_head.bias)
        model.action_destination_head.weight.copy_(model.action_support_head.weight)
        model.action_destination_head.bias.copy_(model.action_support_head.bias)
    print("  Initialized action endpoint heads from action_support_head", flush=True)


def _initialize_action_endpoint_heads_from_proposal_head(model: MacroDreamerEditModel) -> None:
    with torch.no_grad():
        model.action_source_head.weight.copy_(model.proposal_head.weight)
        model.action_source_head.bias.copy_(model.proposal_head.bias)
        model.action_destination_head.weight.copy_(model.proposal_head.weight)
        model.action_destination_head.bias.copy_(model.proposal_head.bias)
    print("  Initialized action endpoint heads from proposal_head", flush=True)


def _initialize_action_edge_pair_support_head_from_action_edge_pair_head(model: MacroDreamerEditModel) -> None:
    with torch.no_grad():
        model.action_edge_pair_support_head.load_state_dict(model.action_edge_pair_head.state_dict())
    print("  Initialized action_edge_pair_support_head from action_edge_pair_head", flush=True)


def _initialize_vacancy_pair_heads_from_action_edge_pair_heads(model: MacroDreamerEditModel) -> None:
    with torch.no_grad():
        if hasattr(model, "vacancy_pair_head"):
            model.vacancy_pair_head.load_state_dict(model.action_edge_pair_head.state_dict())
        if hasattr(model, "vacancy_pair_moving_type_head"):
            model.vacancy_pair_moving_type_head.load_state_dict(model.action_edge_pair_moving_type_head.state_dict())
        if hasattr(model, "vacancy_pair_order_head"):
            model.vacancy_pair_order_head.load_state_dict(model.action_edge_pair_order_head.state_dict())
    print("  Initialized vacancy_pair heads from action_edge_pair heads", flush=True)


def _apply_reward_duration_heads_only_training(model: MacroDreamerEditModel) -> None:
    trainable_prefixes = (
        "reward_head.",
        "reward_context_head.",
        "reward_gate_head.",
        "reward_gate_context_head.",
        "duration_head.",
        "realized_duration_head.",
        "duration_context_head.",
        "realized_duration_context_head.",
    )
    trainable = 0
    total = 0
    for name, param in model.named_parameters():
        total += param.numel()
        should_train = name.startswith(trainable_prefixes)
        param.requires_grad_(should_train)
        if should_train:
            trainable += param.numel()
    print(f"  Reward-duration-head-only training: trainable_params={trainable} / {total}", flush=True)


def _apply_duration_heads_only_training(model: MacroDreamerEditModel) -> None:
    trainable_prefixes = (
        "duration_head.",
        "realized_duration_head.",
        "duration_context_head.",
        "realized_duration_context_head.",
    )
    trainable = 0
    total = 0
    for name, param in model.named_parameters():
        total += param.numel()
        should_train = name.startswith(trainable_prefixes)
        param.requires_grad_(should_train)
        if should_train:
            trainable += param.numel()
    print(f"  Duration-head-only training: trainable_params={trainable} / {total}", flush=True)


def _apply_duration_prior_path_training(model: MacroDreamerEditModel) -> None:
    trainable_prefixes = (
        "k_embed.",
        "path_prior.",
        "macro_dynamics.",
        "duration_head.",
        "realized_duration_head.",
        "duration_context_head.",
        "realized_duration_context_head.",
    )
    trainable = 0
    total = 0
    for name, param in model.named_parameters():
        total += param.numel()
        should_train = name.startswith(trainable_prefixes)
        param.requires_grad_(should_train)
        if should_train:
            trainable += param.numel()
    print(f"  Duration-prior-path training: trainable_params={trainable} / {total}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dreamer multi-k macro edit training")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="results/atomworld_mirror_multik")
    parser.add_argument("--dataset_cache", type=str, default="results/atomworld_mirror_multik/segments.pt")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--init_from", type=str, default=None,
                        help="Load compatible model weights from a checkpoint without optimizer/epoch state. "
                             "Path-posterior tensors are prefix-resized for multi-k summary padding.")
    parser.add_argument("--reinit_output_heads", action="store_true",
                        help="When used with --init_from, reinitialize edit/reward/duration output heads from the new dataset. "
                             "By default, --init_from preserves compatible output heads for true weights-only warm start.")
    parser.add_argument("--reinit_reward_heads", action="store_true",
                        help="When used with --init_from, reinitialize only reward/gate heads from the new dataset, "
                             "preserving edit and duration heads.")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--lattice_size", type=int, nargs=3, default=[40, 40, 40])
    parser.add_argument("--cu_density", type=float, default=0.0134)
    parser.add_argument("--v_density", type=float, default=0.0002)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--reward_scale", type=float, default=10.0)
    parser.add_argument("--neighbor_order", type=str, default="2NN")
    parser.add_argument("--max_episode_steps", type=int, default=200)
    parser.add_argument("--max_vacancies", type=int, default=32)
    parser.add_argument("--max_defects", type=int, default=64)
    parser.add_argument("--max_shells", type=int, default=16)
    parser.add_argument("--stats_dim", type=int, default=10)
    parser.add_argument("--segment_k", type=int, default=4,
                        help="Legacy single-horizon fallback retained for old checkpoints and explicit ablations.")
    parser.add_argument("--segment_ks", type=int, nargs="+", default=[2, 4, 8],
                        help="Multi-k horizons for current AtomWorld-Mirror training. "
                             "Use a one-element list only for legacy single-horizon ablations.")
    parser.add_argument("--train_segments", type=int, default=2000)
    parser.add_argument("--val_segments", type=int, default=400)
    parser.add_argument("--train_segments_per_k", type=int, default=None,
                        help="Number of training segments to collect for each horizon in --segment_ks.")
    parser.add_argument("--val_segments_per_k", type=int, default=None,
                        help="Number of validation segments to collect for each horizon in --segment_ks.")
    parser.add_argument("--planner_selected_from", type=str, default=None,
                        help="Collect the segment cache from states where this checkpoint's multi-k planner chooses the horizon. "
                             "The teacher still rolls out the selected k for supervision.")
    parser.add_argument("--planner_selected_min_projected_changed_sites", type=int, default=2,
                        help="Reject planner-selected data candidates whose projected edit changes fewer sites than this value.")
    parser.add_argument("--planner_selected_duration_source", type=str, default="model", choices=["model", "baseline", "blend"],
                        help="Duration estimate used by the collector checkpoint for reported predicted tau.")
    parser.add_argument("--planner_selected_duration_blend_alpha", type=float, default=1.0,
                        help="For --planner_selected_duration_source blend, alpha in log_tau = (1-alpha)*baseline + alpha*model.")
    parser.add_argument("--planner_selected_tau_source", type=str, default=None, choices=["model", "baseline", "blend"],
                        help="Duration estimate used only for planner-selected data scoring. Defaults to --planner_selected_duration_source.")
    parser.add_argument("--planner_selected_tau_blend_alpha", type=float, default=None,
                        help="For --planner_selected_tau_source blend, alpha in log_tau. Defaults to --planner_selected_duration_blend_alpha.")
    parser.add_argument("--planner_selected_score_mode", type=str, default="energy_per_tau",
                        choices=["energy_per_tau", "energy_per_sqrt_tau", "energy"],
                        help="Score mode used when choosing k during planner-selected data collection.")
    parser.add_argument("--planner_selected_reward_prediction_source", type=str, default=None, choices=["raw", "projected"],
                        help="Reward source used by the collector checkpoint when choosing planner-selected horizons. "
                             "Defaults to the collector checkpoint's saved reward_prediction_source.")
    parser.add_argument("--planner_selected_tau_residual_penalty", type=float, default=0.0,
                        help="Collector score penalty exp(-w * |log(model_tau / baseline_tau)|).")
    parser.add_argument("--planner_selected_k_penalty_power", type=float, default=0.0,
                        help="Collector score penalty score /= k ** power.")
    parser.add_argument("--planner_selected_noop_risk_penalty", type=float, default=0.0,
                        help="Collector score penalty for explicit no-op/terminal-risk probability.")
    parser.add_argument("--planner_selected_reward_checkpoint", type=str, default=None,
                        help="Optional auxiliary checkpoint used only for planner-selected cache reward/gate/risk scoring.")
    parser.add_argument("--planner_selected_duration_checkpoint", type=str, default=None,
                        help="Optional auxiliary checkpoint used only for planner-selected cache duration scoring/reporting.")
    parser.add_argument("--planner_selected_planner_duration_checkpoint_source", type=str, default="duration",
                        choices=["primary", "duration"],
                        help="When an auxiliary duration checkpoint is set, choose whether collector scoring uses "
                             "the primary planner checkpoint tau or the auxiliary duration checkpoint tau.")
    parser.add_argument("--planner_selected_aux_projected_types_source", type=str, default="aux",
                        choices=["primary", "aux"],
                        help="When auxiliary reward/duration checkpoints are used with projected reward, choose "
                             "whether auxiliary heads evaluate primary projected types or their own projected types.")
    parser.add_argument("--planner_selected_projection_change_source", type=str, default="change",
                        choices=[
                            "change",
                            "proposal",
                            "action_support",
                            "action_source",
                            "action_destination",
                            "action_endpoint",
                            "blend",
                        ],
                        help="Collector projection support logits. Default preserves legacy change-head behavior.")
    parser.add_argument("--planner_selected_projection_change_blend_alpha", type=float, default=0.5,
                        help="Collector blend alpha for projection logits when source=blend.")
    parser.add_argument("--planner_selected_projection_topk_source", type=str, default="none",
                        choices=[
                            "none",
                            "change",
                            "proposal",
                            "action_support",
                            "action_source",
                            "action_destination",
                            "action_endpoint",
                            "blend",
                        ],
                        help="Collector optional top-k support gate source. Default disables top-k gating.")
    parser.add_argument("--planner_selected_projection_topk_budget", type=int, default=0,
                        help="Collector top-k support budget when --planner_selected_projection_topk_source is active.")
    parser.add_argument("--planner_selected_proposal_score_weight", type=float, default=0.0,
                        help="Optional additive collector score term log1p(proposal_support_mass) * weight.")
    parser.add_argument("--planner_selected_candidate_quality_score_weight", type=float, default=0.0,
                        help="Optional additive collector score term from the candidate-quality head sigmoid output. "
                             "Default 0 preserves existing planner behavior.")
    parser.add_argument("--planner_selected_teacher_overlap_rerank_weight", type=float, default=0.0,
                        help="Training-data-only teacher-probe rerank weight. When positive, the planner-selected "
                             "collector probes each horizon from the same start state, adds weight * site-overlap F1 "
                             "to the collector score, then replays the selected teacher rollout with the original RNG state.")
    parser.add_argument("--planner_selected_store_candidate_overlap_masks", action="store_true",
                        help="When teacher-overlap rerank is active, store union teacher-changed and projected false-positive "
                             "site masks from all probed horizon candidates for candidate-level proposal supervision.")
    parser.add_argument("--planner_selected_allow_uncovered_reward_only", action="store_true",
                        help="For planner-selected calibration, keep samples whose teacher endpoint changes fall outside "
                             "the candidate set and use them only for reward/tau head calibration. This is intended for "
                             "--train_reward_duration_heads_only runs; edit targets remain candidate-local.")
    parser.add_argument("--max_seed_vacancies", type=int, default=8)
    parser.add_argument("--max_candidate_sites", type=int, default=128)
    parser.add_argument("--dim_latent", type=int, default=16)
    parser.add_argument("--graph_hidden_size", type=int, default=32)
    parser.add_argument("--patch_hidden_size", type=int, default=96)
    parser.add_argument("--patch_latent_dim", type=int, default=64)
    parser.add_argument("--path_latent_dim", type=int, default=32)
    parser.add_argument("--teacher_path_summary_mode", type=str, default="stepwise", choices=["stepwise", "legacy"])
    parser.add_argument("--tau_supervision_mode", type=str, default="prior_main", choices=["prior_main", "posterior_only"])
    parser.add_argument("--tau_weight", type=float, default=1.0,
                        help="Training loss weight for tau_exp duration supervision (default: 1.0).")
    parser.add_argument("--tau_log_mu_weight", type=float, default=0.0,
                        help="Optional direct L1 weight on expected_duration_mu vs log(tau_exp). "
                             "Use this when NLL improves sigma without improving tau_log_mae.")
    parser.add_argument("--freeze_duration_heads", action="store_true",
                        help="Freeze expected/realized duration heads after checkpoint loading or output-head initialization.")
    parser.add_argument("--train_reward_heads_only", action="store_true",
                        help="Freeze all parameters except reward/gate heads after checkpoint loading and optional head reinitialization.")
    parser.add_argument("--train_noop_risk_heads_only", action="store_true",
                        help="Freeze all parameters except explicit no-op/terminal-risk heads for planner-risk diagnostics.")
    parser.add_argument("--train_edit_heads_only", action="store_true",
                        help="Freeze all parameters except edit decoder/change/type heads. "
                             "Use to absorb teacher-path augmented positive edit/type supervision without drifting reward, duration, encoder, or dynamics.")
    parser.add_argument("--train_proposal_head_only", action="store_true",
                        help="Freeze all parameters except the independent candidate proposal/support head.")
    parser.add_argument("--train_action_support_head_only", action="store_true",
                        help="Freeze all parameters except the independent teacher action-support generator head.")
    parser.add_argument("--train_action_endpoint_heads_only", action="store_true",
                        help="Freeze all parameters except source/destination endpoint action-support heads.")
    parser.add_argument("--train_action_edge_pair_head_only", action="store_true",
                        help="Freeze all parameters except the edge-pair action generator head.")
    parser.add_argument("--train_action_edge_pair_support_head_only", action="store_true",
                        help="Freeze all parameters except the terminal-support edge-pair action generator head.")
    parser.add_argument("--train_action_edge_pair_dual_heads_only", action="store_true",
                        help="Freeze all parameters except both energy/action and terminal-support edge-pair heads.")
    parser.add_argument("--train_action_edge_pair_listwise_heads_only", action="store_true",
                        help="Freeze all parameters except edge-pair energy/support/semantic heads and the candidate-quality ranking head.")
    parser.add_argument("--train_vacancy_pair_heads_only", action="store_true",
                        help="Freeze all parameters except terminal vacancy-displacement pair selector heads.")
    parser.add_argument("--train_vacancy_pair_interaction_head_only", action="store_true",
                        help="Freeze all parameters except the v114 terminal vacancy-pair interaction head.")
    parser.add_argument("--train_candidate_quality_head_only", action="store_true",
                        help="Freeze all parameters except the candidate-level overlap-quality ranking head.")
    parser.add_argument("--train_terminal_edit_support_head_only", action="store_true",
                        help="Freeze all parameters except the terminal sparse-edit support decoder head.")
    parser.add_argument("--train_terminal_typed_diff_head_only", action="store_true",
                        help="Freeze all parameters except the terminal typed final-diff decoder head.")
    parser.add_argument("--init_proposal_from_change_head", action="store_true",
                        help="Copy change_head weights into proposal_head after checkpoint loading before training.")
    parser.add_argument("--init_action_support_from_proposal_head", action="store_true",
                        help="Copy proposal_head weights into action_support_head after checkpoint loading before training.")
    parser.add_argument("--init_action_support_from_change_head", action="store_true",
                        help="Copy change_head weights into action_support_head after checkpoint loading before training.")
    parser.add_argument("--init_action_endpoint_from_action_support_head", action="store_true",
                        help="Copy action_support_head weights into source/destination endpoint heads before training.")
    parser.add_argument("--init_action_endpoint_from_proposal_head", action="store_true",
                        help="Copy proposal_head weights into source/destination endpoint heads before training.")
    parser.add_argument("--init_action_edge_pair_support_from_action_edge_pair_head", action="store_true",
                        help="Copy edge-pair action head weights into the terminal-support edge-pair head before training.")
    parser.add_argument("--init_vacancy_pair_heads_from_action_edge_pair_heads", action="store_true",
                        help="Copy action-edge pair score/semantic heads into terminal vacancy-pair heads before training.")
    parser.add_argument("--train_reward_duration_heads_only", action="store_true",
                        help="Freeze all parameters except reward/gate and expected/realized duration heads. "
                             "Use for planner-selected calibration runs.")
    parser.add_argument("--train_duration_heads_only", action="store_true",
                        help="Freeze all parameters except expected/realized duration heads for time-only calibration.")
    parser.add_argument("--train_duration_prior_path_only", action="store_true",
                        help="Freeze encoders/edit/reward heads but train k embedding, path prior, macro dynamics, "
                             "and duration heads for prior-side duration calibration.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--proj_every_n_batches", type=int, default=1,
                        help="Compute projection losses every N batches (1=every batch). Higher values reduce CPU overhead from projected-global re-encoding.")
    parser.add_argument("--no_aux_anneal", action="store_true",
                        help="Disable aux_scale annealing; keep auxiliary mask losses at full strength throughout training.")
    parser.add_argument("--mask_sparsity_weight", type=float, default=0.0,
                        help="Weight for L1 sparsity penalty on change probabilities of unchanged sites.")
    parser.add_argument("--count_loss_weight", type=float, default=0.1,
                        help="Weight for count_loss within mask_loss (default: 0.1).")
    parser.add_argument("--noop_change_weight", type=float, default=0.0,
                        help="Extra no-op hard-negative penalty on predicted change probabilities for teacher no-op samples.")
    parser.add_argument("--noop_type_copy_weight", type=float, default=0.0,
                        help="Extra type-copy cross-entropy weight for teacher no-op samples.")
    parser.add_argument("--projected_noop_fp_weight", type=float, default=0.0,
                        help="Penalty for projected false-positive edits on teacher no-op samples in projection distillation.")
    parser.add_argument("--noop_risk_weight", type=float, default=0.0,
                        help="Training loss weight for explicit posterior no-op/terminal-risk classification.")
    parser.add_argument("--prior_noop_risk_weight", type=float, default=0.0,
                        help="Training loss weight for explicit prior no-op/terminal-risk classification.")
    parser.add_argument("--proposal_support_weight", type=float, default=0.0,
                        help="Independent posterior site-support proposal supervision weight.")
    parser.add_argument("--prior_proposal_support_weight", type=float, default=0.0,
                        help="Independent prior site-support proposal supervision weight.")
    parser.add_argument("--proposal_target_source", type=str, default="changed",
                        choices=[
                            "changed",
                            "touched",
                            "changed_or_touched",
                            "union",
                            "action_source",
                            "action_destination",
                            "action_endpoint",
                        ],
                        help="Target mask for proposal support supervision and proposal validation metrics. "
                             "Default changed preserves terminal sparse-edit behavior; touched/changed_or_touched "
                             "enable teacher path/action support-compressor diagnostics.")
    parser.add_argument("--proposal_hard_negative_weight", type=float, default=0.0,
                        help="Extra proposal loss on planner-selected projected sites that are not teacher changed sites. "
                             "Default 0 preserves existing behavior.")
    parser.add_argument("--proposal_rank_margin_weight", type=float, default=0.0,
                        help="Pairwise proposal ranking loss weight: teacher changed sites should score above "
                             "planner-selected projected false positives. Default 0 preserves existing behavior.")
    parser.add_argument("--proposal_candidate_positive_weight", type=float, default=0.0,
                        help="Extra proposal BCE weight for sites changed by any teacher-probed planner candidate. "
                             "Requires --planner_selected_store_candidate_overlap_masks.")
    parser.add_argument("--proposal_candidate_negative_weight", type=float, default=0.0,
                        help="Extra proposal BCE weight for sites projected by probed candidates but not changed by their teacher probes. "
                             "Requires --planner_selected_store_candidate_overlap_masks.")
    parser.add_argument("--proposal_candidate_rank_margin_weight", type=float, default=0.0,
                        help="Pairwise ranking weight between candidate-level teacher-changed sites and candidate-level false positives. "
                             "Default 0 preserves existing behavior.")
    parser.add_argument("--action_support_weight", type=float, default=0.0,
                        help="Posterior action-support generator supervision weight. Default 0 preserves existing behavior.")
    parser.add_argument("--prior_action_support_weight", type=float, default=0.0,
                        help="Prior action-support generator supervision weight. This is the planner-facing branch.")
    parser.add_argument("--action_support_target_source", type=str, default="touched",
                        choices=[
                            "changed",
                            "touched",
                            "changed_or_touched",
                            "union",
                            "action_source",
                            "action_destination",
                            "action_endpoint",
                        ],
                        help="Target mask for action-support supervision and validation metrics. "
                             "Default touched learns teacher path action support instead of terminal changed support.")
    parser.add_argument("--action_support_hard_negative_weight", type=float, default=0.0,
                        help="Extra action-support loss on planner-selected projected sites that are not teacher changed sites.")
    parser.add_argument("--action_support_rank_margin_weight", type=float, default=0.0,
                        help="Pairwise action-support ranking weight between target positives and selected false positives.")
    parser.add_argument("--action_support_candidate_positive_weight", type=float, default=0.0,
                        help="Extra action-support BCE weight for sites changed by any teacher-probed planner candidate.")
    parser.add_argument("--action_support_candidate_negative_weight", type=float, default=0.0,
                        help="Extra action-support BCE weight for candidate-projected false positives.")
    parser.add_argument("--action_support_candidate_rank_margin_weight", type=float, default=0.0,
                        help="Pairwise ranking weight between candidate-level teacher-changed sites and candidate-level false positives.")
    parser.add_argument("--terminal_edit_support_weight", type=float, default=0.0,
                        help="Posterior terminal sparse-edit support decoder loss weight. Default 0 preserves existing behavior.")
    parser.add_argument("--prior_terminal_edit_support_weight", type=float, default=0.0,
                        help="Prior terminal sparse-edit support decoder loss weight for planner-facing support.")
    parser.add_argument("--terminal_edit_support_target_source", type=str, default="changed",
                        choices=[
                            "changed",
                            "touched",
                            "changed_or_touched",
                            "union",
                            "action_source",
                            "action_destination",
                            "action_endpoint",
                            "action_rollout",
                        ],
                        help="Target mask for terminal sparse-edit support. Default changed keeps this branch terminal-edit specific.")
    parser.add_argument("--terminal_edit_action_context_source", type=str, default="action_endpoint",
                        choices=["action_endpoint", "teacher_rollout"],
                        help="Action-context logits used by terminal_edit_support during train/eval.")
    parser.add_argument("--terminal_edit_support_hard_negative_weight", type=float, default=0.0,
                        help="Extra terminal-edit loss on planner-selected projected sites that are not teacher changed sites.")
    parser.add_argument("--terminal_edit_support_rank_margin_weight", type=float, default=0.0,
                        help="Pairwise terminal-edit ranking weight between target positives and selected false positives.")
    parser.add_argument("--terminal_edit_support_candidate_positive_weight", type=float, default=0.0,
                        help="Extra terminal-edit BCE weight for sites changed by any teacher-probed planner candidate.")
    parser.add_argument("--terminal_edit_support_candidate_negative_weight", type=float, default=0.0,
                        help="Extra terminal-edit BCE weight for candidate-projected false positives.")
    parser.add_argument("--terminal_edit_support_candidate_rank_margin_weight", type=float, default=0.0,
                        help="Pairwise terminal-edit ranking weight between candidate teacher positives and false positives.")
    parser.add_argument("--terminal_typed_diff_weight", type=float, default=0.0,
                        help="Posterior terminal typed final-diff CE loss weight on vacancy-displacement sites.")
    parser.add_argument("--prior_terminal_typed_diff_weight", type=float, default=0.0,
                        help="Prior terminal typed final-diff CE loss weight on vacancy-displacement sites.")
    parser.add_argument("--terminal_typed_diff_copy_weight", type=float, default=0.05,
                        help="Copy CE weight on non-vacancy-displacement candidate sites for terminal typed-diff head.")
    parser.add_argument("--terminal_typed_diff_support_weight", type=float, default=0.0,
                        help="Extra support BCE/focal loss weight on terminal typed-diff noncopy logits. "
                             "Default 0 preserves the v94 CE-only typed-diff behavior.")
    parser.add_argument("--action_source_support_weight", type=float, default=0.0,
                        help="Posterior source endpoint support loss weight for teacher action old_pos sites.")
    parser.add_argument("--prior_action_source_support_weight", type=float, default=0.0,
                        help="Prior source endpoint support loss weight for planner-facing old_pos sites.")
    parser.add_argument("--action_destination_support_weight", type=float, default=0.0,
                        help="Posterior destination endpoint support loss weight for teacher action new_pos sites.")
    parser.add_argument("--prior_action_destination_support_weight", type=float, default=0.0,
                        help="Prior destination endpoint support loss weight for planner-facing new_pos sites.")
    parser.add_argument("--action_edge_pair_weight", type=float, default=0.0,
                        help="Posterior edge-pair action loss weight for teacher old_pos -> new_pos pairs.")
    parser.add_argument("--prior_action_edge_pair_weight", type=float, default=0.0,
                        help="Prior edge-pair action loss weight for planner-facing old_pos -> new_pos pairs.")
    parser.add_argument("--action_edge_pair_support_weight", type=float, default=0.0,
                        help="Posterior terminal-support edge-pair loss weight. Default 0 preserves existing behavior.")
    parser.add_argument("--prior_action_edge_pair_support_weight", type=float, default=0.0,
                        help="Prior terminal-support edge-pair loss weight for planner-facing support scoring.")
    parser.add_argument("--action_edge_pair_semantic_weight", type=float, default=0.0,
                        help="Posterior moving-type/action-order semantic loss weight for teacher old_pos->new_pos pairs.")
    parser.add_argument("--prior_action_edge_pair_semantic_weight", type=float, default=0.0,
                        help="Prior moving-type/action-order semantic loss weight for planner-facing edge-pair semantics.")
    parser.add_argument("--action_edge_pair_negative_weight", type=float, default=1.0,
                        help="Negative-pair BCE weight for edge-pair action supervision.")
    parser.add_argument("--action_edge_pair_rank_margin_weight", type=float, default=0.25,
                        help="Margin-ranking weight between true action edges and selected negative edge pairs.")
    parser.add_argument("--action_edge_pair_negative_mode", type=str, default="self",
                        choices=["self", "same_source_nn1"],
                        help="Negative edge-pair construction. self preserves old_pos->old_pos negatives; "
                             "same_source_nn1 uses a legal wrong 1NN destination under the same source when available.")
    parser.add_argument("--action_edge_pair_negative_count", type=int, default=1,
                        help="Number of same-source legal destination negatives per positive edge pair. "
                             "Values above 1 enable listwise edge-pair ranking; default preserves legacy behavior.")
    parser.add_argument("--action_edge_pair_dense_negative_count", type=int, default=0,
                        help="Extra global KMC-legal vacancy->atom negative pairs per positive edge for the "
                             "energy edge-pair head only. Default 0 preserves existing behavior.")
    parser.add_argument("--action_edge_pair_target_source", type=str, default="action",
                        choices=[
                            "action",
                            "action_edge",
                            "teacher_action_edge",
                            "vacancy_pair",
                            "terminal_vacancy_pair",
                            "terminal_vacancy_displacement",
                        ],
                        help="Pair target for action_edge_pair heads. Default action uses teacher micro-action edges; "
                             "vacancy_pair trains the same heads on terminal vacancy-displacement pairs.")
    parser.add_argument("--vacancy_pair_weight", type=float, default=0.0,
                        help="Posterior terminal vacancy-displacement pair selector loss weight.")
    parser.add_argument("--prior_vacancy_pair_weight", type=float, default=0.0,
                        help="Prior terminal vacancy-displacement pair selector loss weight for planner-facing scoring.")
    parser.add_argument("--vacancy_pair_semantic_weight", type=float, default=0.0,
                        help="Posterior moving-type/order semantic loss weight for terminal vacancy-displacement pairs.")
    parser.add_argument("--prior_vacancy_pair_semantic_weight", type=float, default=0.0,
                        help="Prior moving-type/order semantic loss weight for terminal vacancy-displacement pairs.")
    parser.add_argument("--vacancy_pair_listwise_weight", type=float, default=0.0,
                        help="Posterior listwise contrastive loss weight that ranks each true terminal vacancy pair "
                             "above sampled false pairs from the same state/candidate.")
    parser.add_argument("--prior_vacancy_pair_listwise_weight", type=float, default=0.0,
                        help="Prior planner-facing listwise contrastive loss weight for terminal vacancy pairs.")
    parser.add_argument("--vacancy_pair_interaction_weight", type=float, default=0.0,
                        help="Posterior v114 conditional-compatibility pair-interaction loss weight.")
    parser.add_argument("--prior_vacancy_pair_interaction_weight", type=float, default=0.0,
                        help="Prior planner-facing v114 conditional-compatibility pair-interaction loss weight.")
    parser.add_argument("--vacancy_pair_interaction_listwise_weight", type=float, default=0.0,
                        help="Posterior v114 listwise loss weight for the interaction pair scorer.")
    parser.add_argument("--prior_vacancy_pair_interaction_listwise_weight", type=float, default=0.0,
                        help="Prior planner-facing v114 listwise loss weight for the interaction pair scorer.")
    parser.add_argument("--vacancy_pair_negative_count", type=int, default=4,
                        help="Dense terminal vacancy->atom negative pairs per true vacancy-displacement pair.")
    parser.add_argument("--vacancy_pair_structured_negative_count", type=int, default=0,
                        help="Structured same-source, same-destination, and teacher-unpaired false vacancy-pair "
                             "negatives per true pair. Default 0 preserves previous behavior.")
    parser.add_argument("--candidate_quality_weight", type=float, default=0.0,
                        help="Posterior candidate-quality regression weight against selected teacher-overlap F1. "
                             "Requires teacher-overlap rerank data.")
    parser.add_argument("--prior_candidate_quality_weight", type=float, default=0.0,
                        help="Prior candidate-quality regression weight against selected teacher-overlap F1. "
                             "This is the planner-facing quality branch.")
    parser.add_argument("--pair_weight", type=float, default=0.0,
                        help="Independent training loss weight for posterior directional vacancy/atom pair-count balance.")
    parser.add_argument("--prior_pair_weight", type=float, default=None,
                        help="Independent training loss weight for prior-side directional pair-count balance. Defaults to --pair_weight.")
    parser.add_argument("--detach_proj_encoder", action="store_true",
                        help="Detach encoder outputs (projected_global, projected_patch_latent) in proj_state_loss, preventing projection consistency gradients from flowing into the encoder.")
    parser.add_argument("--proj_weight", type=float, default=0.5,
                        help="Training loss weight for proj_state_loss (default: 0.5). Higher values strengthen encoder projection consistency.")
    parser.add_argument("--proj_l1_score_weight", type=float, default=80.0,
                        help="Weight for proj_global_l1 in the selection score formula (default: 80.0). Lower values reduce proj_l1 dominance in model selection.")
    parser.add_argument("--reward_weight", type=float, default=0.5,
                        help="Training loss weight for reward prediction (default: 0.5). Higher values strengthen energy/reward prediction.")
    parser.add_argument("--prior_reward_weight", type=float, default=0.5,
                        help="Training loss weight for prior-side reward prediction (default: 0.5). Higher values reduce posterior/prior reward mismatch at inference.")
    parser.add_argument("--prior_edit_weight", type=float, default=0.25,
                        help="Training loss weight for prior-side sparse edit prediction (default: 0.25).")
    parser.add_argument("--prior_latent_weight", type=float, default=0.25,
                        help="Training loss weight for prior-side latent/projection prediction (default: 0.25).")
    parser.add_argument("--path_weight", type=float, default=0.05,
                        help="Training loss weight for posterior-prior path KL (default: 0.05).")
    parser.add_argument("--reward_magnitude_weight", type=float, default=1.0,
                        help="Extra multiplier for reward magnitude loss relative to gate loss (default: 1.0). "
                             "Higher values prioritise predicting correct dE amplitude over zero/nonzero classification.")
    parser.add_argument("--reward_gated_weight", type=float, default=1.0,
                        help="Internal weight for the gated reward regression term inside reward supervision.")
    parser.add_argument("--reward_gate_weight", type=float, default=0.25,
                        help="Internal weight for reward zero/nonzero gate BCE. Increase when zero-reward samples are overpredicted.")
    parser.add_argument("--reward_zero_weight", type=float, default=0.5,
                        help="Internal weight for raw reward regression on zero-reward samples.")
    parser.add_argument("--reward_sign_weight", type=float, default=0.25,
                        help="Internal weight for nonzero reward sign classification.")
    parser.add_argument("--reward_prediction_source", type=str, default="raw", choices=["raw", "projected"],
                        help="Reward/gate context source. 'projected' trains and evaluates reward on the projection-closed macro edit.")
    parser.add_argument("--reward_edit_context_source", type=str, default="default", choices=["default", "none"],
                        help="Edit-summary features passed into reward/duration/risk context heads. "
                             "'default' keeps the current raw/projected edit context; 'none' keeps patch+k context but zeros edit-summary features.")
    parser.add_argument("--reward_branch_version", type=int, default=2,
                        help="Version tag for reward/gate architecture and initialization. Changing it forces reward-head recalibration on resume.")
    parser.add_argument("--realized_tau_weight", type=float, default=0.25,
                        help="Auxiliary loss weight for the tau_real lognormal distribution head (default: 0.25). "
                             "tau_exp remains the primary time-supervision target.")
    parser.add_argument("--natural_teacher_backend", type=str, default="kmc", choices=["kmc"],
                        help="Simulator-teacher adapter backend. KMC is the first backend; non-KMC simulators should implement the same state/support/path/duration contract.")
    parser.add_argument("--teacher_mode", type=str, default="kmc", choices=["kmc", "neural"],
                        help="Action source inside the KMC natural-teacher backend: 'kmc' = rate-proportional, 'neural' = learned policy.")
    parser.add_argument("--neural_teacher_path", type=str, default=None,
                        help="Path to DreamerKMCAgent checkpoint for neural teacher mode.")
    parser.add_argument("--neural_teacher_temperature", type=float, default=1.0,
                        help="Temperature for neural teacher softmax sampling (higher = more random).")
    parser.add_argument("--neural_teacher_epsilon", type=float, default=0.0,
                        help="Epsilon-greedy: probability of using KMC rate-proportional sampling instead of neural teacher.")
    parser.add_argument("--segment_boundary_mode", type=str, default="fixed_k", choices=["fixed_k", "adaptive_key_event"],
                        help="Macro-boundary policy. fixed_k preserves the existing Multi-K protocol; adaptive_key_event lets the teacher stop at key events before the nominal horizon.")
    parser.add_argument("--adaptive_min_k", type=int, default=1,
                        help="Minimum teacher micro steps before adaptive_key_event can stop a segment.")
    parser.add_argument("--adaptive_candidate_horizon_source", type=str, default="actual", choices=["nominal", "actual"],
                        help="For adaptive_key_event, filter candidate support by the nominal max horizon or the realized actual horizon.")
    parser.add_argument("--adaptive_key_moving_types", type=int, nargs="*", default=[CU_TYPE],
                        help="Moving type ids that trigger adaptive_key_event after --adaptive_min_k. Default is Cu for the KMC backend.")
    parser.add_argument("--adaptive_min_touched_sites", type=int, default=0,
                        help="Optional adaptive boundary trigger once this many unique sites are touched; 0 disables it.")
    parser.add_argument("--adaptive_abs_delta_e_threshold", type=float, default=0.0,
                        help="Optional single-step |delta_E| adaptive boundary trigger; 0 disables it.")
    parser.add_argument("--adaptive_cumulative_abs_delta_e_threshold", type=float, default=0.0,
                        help="Optional cumulative |delta_E| adaptive boundary trigger; 0 disables it.")
    parser.add_argument("--gate_warmup_epochs", type=int, default=0,
                        help="Number of initial epochs where only gate loss is trained (reward_magnitude_weight=0).")
    parser.add_argument("--eval_freq", type=int, default=5)
    parser.add_argument("--save_freq", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_segments_per_rollout", type=int, default=50,
                        help="Maximum number of accepted macro segments to draw from one teacher rollout before rebuilding the env. Use larger values for more natural long trajectories; 0 disables forced resets.")
    parser.add_argument("--teacher_candidate_neighbor_depth", type=int, default=1,
                        help="During training-data construction, expand the candidate set around teacher-touched sites by this many 1-hop shells.")
    parser.add_argument("--disable_teacher_candidate_augmentation", action="store_true",
                        help="Do not inject future teacher-touched sites into candidate support; use for prior-only planner train/eval parity.")
    parser.add_argument("--include_noop_segments", action="store_true",
                        help="Keep start==end macro segments as explicit zero-edit/no-op supervision samples. "
                             "Default remains to skip them to protect ordinary sparse-edit training.")
    parser.add_argument("--keep_after_noop_segments", action="store_true",
                        help="When --include_noop_segments is active, continue collecting from the same visible "
                             "state after an accepted no-op segment instead of immediately rebuilding the env. "
                             "This is intended for hard-negative no-op support diagnostics; default is restart.")
    return parser.parse_args()


def _dataset_signature(args: argparse.Namespace) -> dict[str, object]:
    segment_ks = _segment_ks_from_args(args)
    train_segments_per_k = _split_segments_per_k(args, "train", segment_ks)
    val_segments_per_k = _split_segments_per_k(args, "val", segment_ks)
    planner_selected_from = getattr(args, "planner_selected_from", None)
    planner_tau_source = getattr(args, "planner_selected_tau_source", None) or getattr(
        args,
        "planner_selected_duration_source",
        "model",
    )
    signature = {
        "dataset_version": 18,
        "dataset_mode": "planner_selected" if planner_selected_from else "teacher_marginal",
        "seed": int(args.seed),
        "lattice_size": list(args.lattice_size),
        "cu_density": float(args.cu_density),
        "v_density": float(args.v_density),
        "segment_k": int(segment_ks[0]) if len(segment_ks) == 1 else int(max(segment_ks)),
        "segment_ks": segment_ks,
        "summary_horizon_k": int(_summary_horizon_k_from_segment_ks(segment_ks)),
        "max_seed_vacancies": int(args.max_seed_vacancies),
        "max_candidate_sites": int(args.max_candidate_sites),
        "max_episode_steps": int(args.max_episode_steps),
        "max_vacancies": int(args.max_vacancies),
        "max_defects": int(args.max_defects),
        "max_shells": int(args.max_shells),
        "neighbor_order": str(args.neighbor_order),
        "reward_scale": float(args.reward_scale),
        "temperature": float(args.temperature),
        "stats_dim": int(args.stats_dim),
        "train_segments": int(_split_total_segments(args, "train", segment_ks)),
        "val_segments": int(_split_total_segments(args, "val", segment_ks)),
        "train_segments_per_k": int(train_segments_per_k),
        "val_segments_per_k": int(val_segments_per_k),
        "planner_selected_from": str(planner_selected_from) if planner_selected_from else None,
        "planner_selected_checkpoint": _path_fingerprint(planner_selected_from),
        "planner_selected_min_projected_changed_sites": int(
            getattr(args, "planner_selected_min_projected_changed_sites", 2)
        ),
        "planner_selected_duration_source": str(getattr(args, "planner_selected_duration_source", "model")),
        "planner_selected_tau_source": str(planner_tau_source),
        "planner_selected_score_mode": str(getattr(args, "planner_selected_score_mode", "energy_per_tau")),
        "planner_selected_reward_prediction_source": (
            str(args.planner_selected_reward_prediction_source)
            if getattr(args, "planner_selected_reward_prediction_source", None)
            else None
        ),
        "planner_selected_tau_residual_penalty": float(getattr(args, "planner_selected_tau_residual_penalty", 0.0)),
        "planner_selected_k_penalty_power": float(getattr(args, "planner_selected_k_penalty_power", 0.0)),
        "planner_selected_noop_risk_penalty": float(getattr(args, "planner_selected_noop_risk_penalty", 0.0)),
        "planner_selected_reward_checkpoint": _path_fingerprint(
            getattr(args, "planner_selected_reward_checkpoint", None)
        ),
        "planner_selected_duration_checkpoint": _path_fingerprint(
            getattr(args, "planner_selected_duration_checkpoint", None)
        ),
        "planner_selected_planner_duration_checkpoint_source": str(
            getattr(args, "planner_selected_planner_duration_checkpoint_source", "duration")
        ),
        "planner_selected_aux_projected_types_source": str(
            getattr(args, "planner_selected_aux_projected_types_source", "aux")
        ),
        "planner_selected_projection_change_source": str(
            getattr(args, "planner_selected_projection_change_source", "change")
        ),
        "planner_selected_projection_change_blend_alpha": float(
            getattr(args, "planner_selected_projection_change_blend_alpha", 0.5)
        ),
        "planner_selected_projection_topk_source": str(
            getattr(args, "planner_selected_projection_topk_source", "none")
        ),
        "planner_selected_projection_topk_budget": int(
            getattr(args, "planner_selected_projection_topk_budget", 0)
        ),
        "planner_selected_proposal_score_weight": float(
            getattr(args, "planner_selected_proposal_score_weight", 0.0)
        ),
        "planner_selected_candidate_quality_score_weight": float(
            getattr(args, "planner_selected_candidate_quality_score_weight", 0.0)
        ),
        "planner_selected_allow_uncovered_reward_only": bool(
            getattr(args, "planner_selected_allow_uncovered_reward_only", False)
        ),
        "include_noop_segments": bool(getattr(args, "include_noop_segments", False)),
        "keep_after_noop_segments": bool(getattr(args, "keep_after_noop_segments", False)),
        "teacher_path_summary_mode": str(args.teacher_path_summary_mode),
        "teacher_mode": str(getattr(args, "teacher_mode", "kmc")),
        "neural_teacher_checkpoint": _path_fingerprint(getattr(args, "neural_teacher_path", None)),
        "neural_teacher_temperature": float(getattr(args, "neural_teacher_temperature", 1.0)),
        "neural_teacher_epsilon": float(getattr(args, "neural_teacher_epsilon", 0.0)),
        "max_segments_per_rollout": int(getattr(args, "max_segments_per_rollout", 50)),
        "teacher_candidate_neighbor_depth": int(getattr(args, "teacher_candidate_neighbor_depth", 1)),
        "teacher_candidate_augmentation": not bool(getattr(args, "disable_teacher_candidate_augmentation", False)),
        "natural_teacher_backend": str(getattr(args, "natural_teacher_backend", "kmc")),
        "segment_boundary_mode": str(getattr(args, "segment_boundary_mode", "fixed_k")),
        "adaptive_min_k": int(getattr(args, "adaptive_min_k", 1)),
        "adaptive_candidate_horizon_source": str(getattr(args, "adaptive_candidate_horizon_source", "nominal")),
        "adaptive_key_moving_types": list(getattr(args, "adaptive_key_moving_types", [CU_TYPE]) or []),
        "adaptive_min_touched_sites": int(getattr(args, "adaptive_min_touched_sites", 0)),
        "adaptive_abs_delta_e_threshold": float(getattr(args, "adaptive_abs_delta_e_threshold", 0.0)),
        "adaptive_cumulative_abs_delta_e_threshold": float(
            getattr(args, "adaptive_cumulative_abs_delta_e_threshold", 0.0)
        ),
    }
    teacher_overlap_rerank_weight = float(getattr(args, "planner_selected_teacher_overlap_rerank_weight", 0.0))
    if teacher_overlap_rerank_weight != 0.0:
        signature["planner_selected_teacher_overlap_rerank_weight"] = teacher_overlap_rerank_weight
    if bool(getattr(args, "planner_selected_store_candidate_overlap_masks", False)):
        signature["planner_selected_store_candidate_overlap_masks"] = True
    if str(getattr(args, "planner_selected_duration_source", "model")) == "blend":
        signature["planner_selected_duration_blend_alpha"] = float(
            getattr(args, "planner_selected_duration_blend_alpha", 1.0)
        )
    if str(planner_tau_source) == "blend":
        tau_blend_alpha = getattr(args, "planner_selected_tau_blend_alpha", None)
        if tau_blend_alpha is None:
            tau_blend_alpha = getattr(args, "planner_selected_duration_blend_alpha", 1.0)
        signature["planner_selected_tau_blend_alpha"] = float(tau_blend_alpha)
    return signature


def _adaptive_boundary_config_from_args(args: argparse.Namespace) -> AdaptiveBoundaryConfig:
    mode = str(getattr(args, "segment_boundary_mode", "fixed_k"))
    return AdaptiveBoundaryConfig(
        mode=mode,
        min_k=max(1, int(getattr(args, "adaptive_min_k", 1))),
        candidate_horizon_source=str(getattr(args, "adaptive_candidate_horizon_source", "nominal")),
        key_moving_types=tuple(int(item) for item in (getattr(args, "adaptive_key_moving_types", [CU_TYPE]) or [])),
        min_touched_sites=max(0, int(getattr(args, "adaptive_min_touched_sites", 0))),
        abs_delta_e_threshold=max(0.0, float(getattr(args, "adaptive_abs_delta_e_threshold", 0.0))),
        cumulative_abs_delta_e_threshold=max(
            0.0,
            float(getattr(args, "adaptive_cumulative_abs_delta_e_threshold", 0.0)),
        ),
    )


def main() -> None:
    args = parse_args()
    if args.resume and args.init_from:
        raise ValueError("--resume and --init_from are mutually exclusive: use --resume for homogeneous continuation or --init_from for weights-only migration")
    head_only_flags = [
        bool(args.train_reward_heads_only),
        bool(args.train_noop_risk_heads_only),
        bool(args.train_edit_heads_only),
        bool(args.train_proposal_head_only),
        bool(args.train_action_support_head_only),
        bool(args.train_action_endpoint_heads_only),
        bool(args.train_action_edge_pair_head_only),
        bool(args.train_action_edge_pair_support_head_only),
        bool(args.train_action_edge_pair_dual_heads_only),
        bool(args.train_action_edge_pair_listwise_heads_only),
        bool(args.train_vacancy_pair_heads_only),
        bool(args.train_vacancy_pair_interaction_head_only),
        bool(args.train_candidate_quality_head_only),
        bool(args.train_terminal_edit_support_head_only),
        bool(args.train_terminal_typed_diff_head_only),
        bool(args.train_reward_duration_heads_only),
        bool(args.train_duration_heads_only),
        bool(args.train_duration_prior_path_only),
    ]
    if sum(head_only_flags) > 1:
        raise ValueError(
            "--train_reward_heads_only, --train_noop_risk_heads_only, --train_edit_heads_only, "
            "--train_proposal_head_only, --train_action_support_head_only, --train_action_endpoint_heads_only, "
            "--train_action_edge_pair_head_only, --train_action_edge_pair_support_head_only, "
            "--train_action_edge_pair_dual_heads_only, --train_action_edge_pair_listwise_heads_only, "
            "--train_vacancy_pair_heads_only, --train_vacancy_pair_interaction_head_only, "
            "--train_candidate_quality_head_only, --train_terminal_edit_support_head_only, "
            "--train_terminal_typed_diff_head_only, "
            "--train_reward_duration_heads_only, --train_duration_heads_only, and "
            "--train_duration_prior_path_only are mutually exclusive"
        )
    if args.freeze_duration_heads and (
        args.train_reward_duration_heads_only
        or args.train_duration_heads_only
        or args.train_duration_prior_path_only
    ):
        raise ValueError("--freeze_duration_heads conflicts with duration-head calibration modes")
    if args.planner_selected_allow_uncovered_reward_only and not (
        args.train_reward_duration_heads_only
        or args.train_duration_heads_only
        or args.train_duration_prior_path_only
    ):
        raise ValueError(
            "--planner_selected_allow_uncovered_reward_only is only supported with "
            "duration/reward-duration head-only calibration, because uncovered samples do not provide complete sparse-edit targets"
        )
    candidate_level_weights = (
        float(getattr(args, "proposal_candidate_positive_weight", 0.0))
        + float(getattr(args, "proposal_candidate_negative_weight", 0.0))
        + float(getattr(args, "proposal_candidate_rank_margin_weight", 0.0))
        + float(getattr(args, "action_support_candidate_positive_weight", 0.0))
        + float(getattr(args, "action_support_candidate_negative_weight", 0.0))
        + float(getattr(args, "action_support_candidate_rank_margin_weight", 0.0))
    )
    if candidate_level_weights > 0.0:
        if not bool(getattr(args, "planner_selected_store_candidate_overlap_masks", False)):
            raise ValueError(
                "candidate-level proposal weights require --planner_selected_store_candidate_overlap_masks"
            )
        if float(getattr(args, "planner_selected_teacher_overlap_rerank_weight", 0.0)) == 0.0:
            raise ValueError(
                "candidate-level proposal weights require --planner_selected_teacher_overlap_rerank_weight > 0"
            )
    candidate_quality_weights = (
        float(getattr(args, "candidate_quality_weight", 0.0))
        + float(getattr(args, "prior_candidate_quality_weight", 0.0))
    )
    if candidate_quality_weights > 0.0 and float(getattr(args, "planner_selected_teacher_overlap_rerank_weight", 0.0)) == 0.0:
        raise ValueError("candidate-quality training requires --planner_selected_teacher_overlap_rerank_weight > 0")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    segment_ks = _segment_ks_from_args(args)
    if getattr(args, "segment_ks", None) is not None:
        args.segment_ks = segment_ks
        args.segment_k = max(segment_ks)
    summary_horizon_k = _summary_horizon_k_from_segment_ks(segment_ks)
    train_segments_per_k = _split_segments_per_k(args, "train", segment_ks)
    val_segments_per_k = _split_segments_per_k(args, "val", segment_ks)
    train_segments_total = _split_total_segments(args, "train", segment_ks)
    val_segments_total = _split_total_segments(args, "val", segment_ks)
    planner_selected_tau_source = args.planner_selected_tau_source or args.planner_selected_duration_source
    planner_selected_tau_blend_alpha = (
        float(args.planner_selected_duration_blend_alpha)
        if args.planner_selected_tau_blend_alpha is None
        else float(args.planner_selected_tau_blend_alpha)
    )
    adaptive_boundary_config = _adaptive_boundary_config_from_args(args)
    if adaptive_boundary_config.mode == "adaptive_key_event" and adaptive_boundary_config.min_k > summary_horizon_k:
        raise ValueError(
            f"adaptive_min_k={adaptive_boundary_config.min_k} exceeds max segment horizon {summary_horizon_k}"
        )
    if int(args.max_episode_steps) < int(summary_horizon_k):
        raise ValueError(
            f"max_episode_steps={int(args.max_episode_steps)} is smaller than max segment horizon "
            f"{int(summary_horizon_k)}; increase --max_episode_steps before collecting large-k macro segments"
        )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    dataset_cache = Path(args.dataset_cache)
    dataset_cache.parent.mkdir(parents=True, exist_ok=True)
    dataset_signature = _dataset_signature(args)

    env_cfg = {
        "lattice_size": tuple(args.lattice_size),
        "max_episode_steps": args.max_episode_steps,
        "max_vacancies": args.max_vacancies,
        "max_defects": args.max_defects,
        "max_shells": args.max_shells,
        "stats_dim": args.stats_dim,
        "temperature": args.temperature,
        "reward_scale": args.reward_scale,
        "cu_density": args.cu_density,
        "v_density": args.v_density,
        "rlkmc_topk": 16,
        "neighbor_order": args.neighbor_order,
    }
    include_stepwise_path_summary = args.teacher_path_summary_mode == "stepwise"

    if dataset_cache.exists():
        payload = torch.load(dataset_cache, map_location="cpu", weights_only=False)
        cached_signature = payload.get("signature")
        if cached_signature != dataset_signature:
            print("Cached dataset signature mismatch; regenerating dataset.")
            dataset_cache.unlink()
            payload = None
        else:
            train_samples = [MacroSegmentSample(**item) for item in payload["train"]]
            val_samples = [MacroSegmentSample(**item) for item in payload["val"]]
            dataset_stats = payload.get("stats", {})
            print(f"Loaded cached dataset from {dataset_cache}")
    else:
        payload = None
    if payload is None:
        # Load neural teacher if requested
        neural_teacher = None
        if args.teacher_mode == "neural":
            if not args.neural_teacher_path:
                raise ValueError("--neural_teacher_path is required when --teacher_mode=neural")
            neural_teacher = _load_neural_teacher(args.neural_teacher_path, env_cfg, args.device)

        if args.planner_selected_from:
            planner_model, planner_ckpt_args = _build_planner_model_from_checkpoint(args.planner_selected_from, args.device)
            planner_reward_model = None
            planner_duration_model = None
            if args.planner_selected_reward_checkpoint:
                planner_reward_model, _ = _build_planner_model_from_checkpoint(
                    args.planner_selected_reward_checkpoint,
                    args.device,
                )
            if args.planner_selected_duration_checkpoint:
                if args.planner_selected_duration_checkpoint == args.planner_selected_reward_checkpoint and planner_reward_model is not None:
                    planner_duration_model = planner_reward_model
                else:
                    planner_duration_model, _ = _build_planner_model_from_checkpoint(
                        args.planner_selected_duration_checkpoint,
                        args.device,
                    )
            planner_reward_prediction_source = str(
                args.planner_selected_reward_prediction_source
                or planner_ckpt_args.get("reward_prediction_source", "raw")
            )
            print(
                "Collecting planner-selected calibration cache: "
                f"source={args.planner_selected_from} "
                f"train={train_segments_total} val={val_segments_total} "
                f"score_mode={args.planner_selected_score_mode} "
                f"tau_source={planner_selected_tau_source} "
                f"tau_blend_alpha={planner_selected_tau_blend_alpha:.3f} "
                f"reward_source={planner_reward_prediction_source} "
                f"reward_ckpt={args.planner_selected_reward_checkpoint or 'primary'} "
                f"duration_ckpt={args.planner_selected_duration_checkpoint or 'primary'} "
                f"projection_source={args.planner_selected_projection_change_source} "
                f"topk={args.planner_selected_projection_topk_source}:{args.planner_selected_projection_topk_budget} "
                f"teacher_overlap_rerank={args.planner_selected_teacher_overlap_rerank_weight:.3f} "
                f"allow_uncovered_reward_only={args.planner_selected_allow_uncovered_reward_only}",
                flush=True,
            )
            train_samples, train_stats = _collect_planner_selected_segments(
                env=MacroKMCEnv(copy.deepcopy(env_cfg)),
                num_segments=train_segments_total,
                segment_ks=segment_ks,
                planner_model=planner_model,
                planner_reward_model=planner_reward_model,
                planner_duration_model=planner_duration_model,
                planner_device=args.device,
                max_seed_vacancies=args.max_seed_vacancies,
                max_candidate_sites=args.max_candidate_sites,
                rng=np.random.default_rng(args.seed),
                include_stepwise_path_summary=include_stepwise_path_summary,
                summary_horizon_k=summary_horizon_k,
                max_segments_per_rollout=args.max_segments_per_rollout,
                min_projected_changed_sites=args.planner_selected_min_projected_changed_sites,
                duration_source=args.planner_selected_duration_source,
                planner_tau_source=planner_selected_tau_source,
                planner_score_mode=args.planner_selected_score_mode,
                planner_tau_residual_penalty=args.planner_selected_tau_residual_penalty,
                planner_k_penalty_power=args.planner_selected_k_penalty_power,
                planner_noop_risk_penalty=args.planner_selected_noop_risk_penalty,
                reward_prediction_source=planner_reward_prediction_source,
                reward_edit_context_source=args.reward_edit_context_source,
                planner_duration_checkpoint_source=args.planner_selected_planner_duration_checkpoint_source,
                aux_projected_types_source=args.planner_selected_aux_projected_types_source,
                planner_projection_change_source=args.planner_selected_projection_change_source,
                planner_projection_change_blend_alpha=args.planner_selected_projection_change_blend_alpha,
                planner_projection_topk_source=args.planner_selected_projection_topk_source,
                planner_projection_topk_budget=args.planner_selected_projection_topk_budget,
                planner_proposal_score_weight=args.planner_selected_proposal_score_weight,
                planner_candidate_quality_score_weight=args.planner_selected_candidate_quality_score_weight,
                planner_teacher_overlap_rerank_weight=args.planner_selected_teacher_overlap_rerank_weight,
                planner_selected_store_candidate_overlap_masks=args.planner_selected_store_candidate_overlap_masks,
                duration_blend_alpha=args.planner_selected_duration_blend_alpha,
                planner_tau_blend_alpha=planner_selected_tau_blend_alpha,
                allow_uncovered_reward_only=args.planner_selected_allow_uncovered_reward_only,
                teacher_candidate_augmentation=not args.disable_teacher_candidate_augmentation,
                teacher_candidate_neighbor_depth=args.teacher_candidate_neighbor_depth,
                teacher_mode=args.teacher_mode,
                neural_teacher=neural_teacher,
                neural_teacher_device=args.device,
                neural_teacher_temperature=args.neural_teacher_temperature,
                neural_teacher_epsilon=args.neural_teacher_epsilon,
                natural_teacher_backend=args.natural_teacher_backend,
                adaptive_boundary_config=adaptive_boundary_config,
                include_noop_segments=args.include_noop_segments,
                keep_after_noop_segments=args.keep_after_noop_segments,
            )
            val_samples, val_stats = _collect_planner_selected_segments(
                env=MacroKMCEnv(copy.deepcopy(env_cfg)),
                num_segments=val_segments_total,
                segment_ks=segment_ks,
                planner_model=planner_model,
                planner_reward_model=planner_reward_model,
                planner_duration_model=planner_duration_model,
                planner_device=args.device,
                max_seed_vacancies=args.max_seed_vacancies,
                max_candidate_sites=args.max_candidate_sites,
                rng=np.random.default_rng(args.seed + 1),
                include_stepwise_path_summary=include_stepwise_path_summary,
                summary_horizon_k=summary_horizon_k,
                max_segments_per_rollout=args.max_segments_per_rollout,
                min_projected_changed_sites=args.planner_selected_min_projected_changed_sites,
                duration_source=args.planner_selected_duration_source,
                planner_tau_source=planner_selected_tau_source,
                planner_score_mode=args.planner_selected_score_mode,
                planner_tau_residual_penalty=args.planner_selected_tau_residual_penalty,
                planner_k_penalty_power=args.planner_selected_k_penalty_power,
                planner_noop_risk_penalty=args.planner_selected_noop_risk_penalty,
                reward_prediction_source=planner_reward_prediction_source,
                reward_edit_context_source=args.reward_edit_context_source,
                planner_duration_checkpoint_source=args.planner_selected_planner_duration_checkpoint_source,
                aux_projected_types_source=args.planner_selected_aux_projected_types_source,
                planner_projection_change_source=args.planner_selected_projection_change_source,
                planner_projection_change_blend_alpha=args.planner_selected_projection_change_blend_alpha,
                planner_projection_topk_source=args.planner_selected_projection_topk_source,
                planner_projection_topk_budget=args.planner_selected_projection_topk_budget,
                planner_proposal_score_weight=args.planner_selected_proposal_score_weight,
                planner_candidate_quality_score_weight=args.planner_selected_candidate_quality_score_weight,
                planner_teacher_overlap_rerank_weight=args.planner_selected_teacher_overlap_rerank_weight,
                planner_selected_store_candidate_overlap_masks=args.planner_selected_store_candidate_overlap_masks,
                duration_blend_alpha=args.planner_selected_duration_blend_alpha,
                planner_tau_blend_alpha=planner_selected_tau_blend_alpha,
                allow_uncovered_reward_only=args.planner_selected_allow_uncovered_reward_only,
                teacher_candidate_augmentation=not args.disable_teacher_candidate_augmentation,
                teacher_candidate_neighbor_depth=args.teacher_candidate_neighbor_depth,
                teacher_mode=args.teacher_mode,
                neural_teacher=neural_teacher,
                neural_teacher_device=args.device,
                neural_teacher_temperature=args.neural_teacher_temperature,
                neural_teacher_epsilon=args.neural_teacher_epsilon,
                natural_teacher_backend=args.natural_teacher_backend,
                adaptive_boundary_config=adaptive_boundary_config,
                include_noop_segments=args.include_noop_segments,
                keep_after_noop_segments=args.keep_after_noop_segments,
            )
        else:
            train_samples, train_stats = _collect_segments_for_horizons(
                env_cfg=env_cfg,
                segment_ks=segment_ks,
                num_segments_per_k=train_segments_per_k,
                max_seed_vacancies=args.max_seed_vacancies,
                max_candidate_sites=args.max_candidate_sites,
                seed=args.seed,
                include_stepwise_path_summary=include_stepwise_path_summary,
                summary_horizon_k=summary_horizon_k,
                max_segments_per_rollout=args.max_segments_per_rollout,
                teacher_candidate_neighbor_depth=args.teacher_candidate_neighbor_depth,
                teacher_candidate_augmentation=not args.disable_teacher_candidate_augmentation,
                teacher_mode=args.teacher_mode,
                neural_teacher=neural_teacher,
                neural_teacher_device=args.device,
                neural_teacher_temperature=args.neural_teacher_temperature,
                neural_teacher_epsilon=args.neural_teacher_epsilon,
                natural_teacher_backend=args.natural_teacher_backend,
                adaptive_boundary_config=adaptive_boundary_config,
                include_noop_segments=args.include_noop_segments,
                keep_after_noop_segments=args.keep_after_noop_segments,
            )
            val_samples, val_stats = _collect_segments_for_horizons(
                env_cfg=env_cfg,
                segment_ks=segment_ks,
                num_segments_per_k=val_segments_per_k,
                max_seed_vacancies=args.max_seed_vacancies,
                max_candidate_sites=args.max_candidate_sites,
                seed=args.seed + 1,
                include_stepwise_path_summary=include_stepwise_path_summary,
                summary_horizon_k=summary_horizon_k,
                max_segments_per_rollout=args.max_segments_per_rollout,
                teacher_candidate_neighbor_depth=args.teacher_candidate_neighbor_depth,
                teacher_candidate_augmentation=not args.disable_teacher_candidate_augmentation,
                teacher_mode=args.teacher_mode,
                neural_teacher=neural_teacher,
                neural_teacher_device=args.device,
                neural_teacher_temperature=args.neural_teacher_temperature,
                neural_teacher_epsilon=args.neural_teacher_epsilon,
                natural_teacher_backend=args.natural_teacher_backend,
                adaptive_boundary_config=adaptive_boundary_config,
                include_noop_segments=args.include_noop_segments,
                keep_after_noop_segments=args.keep_after_noop_segments,
            )
        dataset_stats = {"train": train_stats, "val": val_stats}
        torch.save(
            {
                "train": [asdict(sample) for sample in train_samples],
                "val": [asdict(sample) for sample in val_samples],
                "stats": dataset_stats,
                "signature": dataset_signature,
            },
            dataset_cache,
        )
        print(json.dumps({"dataset_stats": dataset_stats}, ensure_ascii=False), flush=True)

    train_loader = _build_loader(train_samples, args.batch_size, shuffle=True)
    val_loader = _build_loader(val_samples, args.batch_size, shuffle=False)
    model = MacroDreamerEditModel(
        max_vacancies=args.max_vacancies,
        max_defects=args.max_defects,
        max_shells=args.max_shells,
        stats_dim=args.stats_dim,
        lattice_size=tuple(args.lattice_size),
        neighbor_order=args.neighbor_order,
        dim_latent=args.dim_latent,
        graph_hidden_size=args.graph_hidden_size,
        patch_hidden_size=args.patch_hidden_size,
        patch_latent_dim=args.patch_latent_dim,
        path_latent_dim=args.path_latent_dim,
        global_summary_dim=16,
        teacher_path_summary_dim=teacher_path_summary_dim(summary_horizon_k, include_stepwise_features=include_stepwise_path_summary),
        max_macro_k=max(summary_horizon_k, 16),
    ).to(args.device)
    model.realized_tau_head_loaded = True
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    weights = {
        "mask": 1.0,
        "type": 1.0,
        "pair": args.pair_weight,
        "prior_pair": args.pair_weight if args.prior_pair_weight is None else args.prior_pair_weight,
        "tau": args.tau_weight,
        "tau_log_mu": args.tau_log_mu_weight,
        "realized_tau": args.realized_tau_weight,
        "reward": args.reward_weight,
        "prior_reward": args.prior_reward_weight,
        "noop_risk": args.noop_risk_weight,
        "prior_noop_risk": args.prior_noop_risk_weight,
        "proposal": args.proposal_support_weight,
        "prior_proposal": args.prior_proposal_support_weight,
        "latent": 0.5,
        "proj": args.proj_weight,
        "path": args.path_weight,
        "prior_edit": args.prior_edit_weight,
        "prior_latent": args.prior_latent_weight,
    }
    max_changed_sites = 2 * summary_horizon_k
    best_score = float("inf")
    start_epoch = 1
    if args.init_from:
        ckpt = torch.load(args.init_from, map_location=args.device, weights_only=False)
        missing, unexpected, resized, skipped = _load_model_weights(
            model,
            ckpt["model"],
            allow_path_posterior_resize=True,
        )
        model.realized_tau_head_loaded = not any(key.startswith("realized_duration_head.") for key in missing)
        if resized:
            print(f"Init-from: prefix-resized compatible tensors: {resized}", flush=True)
        if skipped:
            print(f"Init-from: skipped incompatible tensors: {skipped}", flush=True)
        if missing:
            print(f"Init-from: parameters left at initialization: {missing}", flush=True)
        if unexpected:
            print(f"Init-from: unexpected keys ignored: {unexpected}", flush=True)
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=args.device, weights_only=False)
        _validate_resume_args(args, ckpt.get("args", {}))
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        model.realized_tau_head_loaded = not any(key.startswith("realized_duration_head.") for key in missing)
        if missing:
            print(f"Resume: new parameters initialized from scratch: {missing}")
        if unexpected:
            print(f"Resume: unexpected keys ignored: {unexpected}")
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except ValueError as e:
            print(f"Resume: optimizer state incompatible ({e}), reinitializing optimizer (weights loaded)")
        start_epoch = int(ckpt["epoch"]) + 1
        if not args.eval_only:
            resume_reward_branch_version = int(ckpt.get("args", {}).get("reward_branch_version", 1))
            reward_branch_missing = any(
                key.startswith("reward_context_head.") or key.startswith("reward_gate_context_head.")
                for key in missing
            )
            if resume_reward_branch_version != int(args.reward_branch_version) or reward_branch_missing:
                _initialize_reward_heads(model, train_samples)
                if resume_reward_branch_version != int(args.reward_branch_version):
                    print(
                        "Resume: reinitialized reward heads due to reward_branch_version mismatch "
                        f"({resume_reward_branch_version} -> {int(args.reward_branch_version)})"
                    )
                else:
                    print("Resume: reinitialized reward heads because reward context parameters were missing")
        if not args.eval_only:
            allow_checkpoint_best_score_fallback = Path(args.resume).resolve().parent == save_dir.resolve()
            best_score, score_source = _initialize_best_score_from_saved_best(
                model=model,
                loader=val_loader,
                device=args.device,
                max_changed_sites=max_changed_sites,
                dataset_stats=dataset_stats,
                save_dir=save_dir,
                checkpoint_best_score=ckpt.get("best_score"),
                allow_checkpoint_best_score_fallback=allow_checkpoint_best_score_fallback,
                proj_l1_score_weight=args.proj_l1_score_weight,
                reward_prediction_source=args.reward_prediction_source,
                reward_edit_context_source=args.reward_edit_context_source,
                proposal_target_source=args.proposal_target_source,
                action_support_target_source=args.action_support_target_source,
                terminal_edit_action_context_source=args.terminal_edit_action_context_source,
            )
            print(f"Initialized best_score from {score_source} under current selection metric: {best_score:.4f}")
        else:
            best_score = float(ckpt.get("best_score", best_score))

    _apply_output_head_initialization_policy(
        model,
        train_samples,
        resume=args.resume,
        init_from=args.init_from,
        reinit_output_heads=args.reinit_output_heads,
        reinit_reward_heads=args.reinit_reward_heads,
        freeze_duration_heads=args.freeze_duration_heads,
    )
    if args.init_proposal_from_change_head:
        _initialize_proposal_head_from_change_head(model)
    if args.init_action_support_from_proposal_head:
        _initialize_action_support_head_from_proposal_head(model)
    if args.init_action_support_from_change_head:
        _initialize_action_support_head_from_change_head(model)
    if args.init_action_endpoint_from_action_support_head:
        _initialize_action_endpoint_heads_from_action_support_head(model)
    if args.init_action_endpoint_from_proposal_head:
        _initialize_action_endpoint_heads_from_proposal_head(model)
    if args.init_action_edge_pair_support_from_action_edge_pair_head:
        _initialize_action_edge_pair_support_head_from_action_edge_pair_head(model)
    if args.init_vacancy_pair_heads_from_action_edge_pair_heads:
        _initialize_vacancy_pair_heads_from_action_edge_pair_heads(model)
    if args.train_reward_heads_only:
        _apply_reward_heads_only_training(model)
    if args.train_noop_risk_heads_only:
        _apply_noop_risk_heads_only_training(model)
    if args.train_edit_heads_only:
        _apply_edit_heads_only_training(model)
    if args.train_proposal_head_only:
        _apply_proposal_head_only_training(model)
    if args.train_action_support_head_only:
        _apply_action_support_head_only_training(model)
    if args.train_action_endpoint_heads_only:
        _apply_action_endpoint_heads_only_training(model)
    if args.train_action_edge_pair_head_only:
        _apply_action_edge_pair_head_only_training(model)
    if args.train_action_edge_pair_support_head_only:
        _apply_action_edge_pair_support_head_only_training(model)
    if args.train_action_edge_pair_dual_heads_only:
        _apply_action_edge_pair_dual_heads_only_training(model)
    if args.train_action_edge_pair_listwise_heads_only:
        _apply_action_edge_pair_listwise_heads_only_training(model)
    if args.train_vacancy_pair_heads_only:
        _apply_vacancy_pair_heads_only_training(model)
    if args.train_vacancy_pair_interaction_head_only:
        _apply_vacancy_pair_interaction_head_only_training(model)
    if args.train_candidate_quality_head_only:
        _apply_candidate_quality_head_only_training(model)
    if args.train_terminal_edit_support_head_only:
        _apply_terminal_edit_support_head_only_training(model)
    if args.train_terminal_typed_diff_head_only:
        _apply_terminal_typed_diff_head_only_training(model)
    if args.train_reward_duration_heads_only:
        _apply_reward_duration_heads_only_training(model)
    if args.train_duration_heads_only:
        _apply_duration_heads_only_training(model)
    if args.train_duration_prior_path_only:
        _apply_duration_prior_path_training(model)

    if args.eval_only:
        metrics = _evaluate(
            model,
            val_loader,
            args.device,
            max_changed_sites,
            reward_prediction_source=args.reward_prediction_source,
            reward_edit_context_source=args.reward_edit_context_source,
            proposal_target_source=args.proposal_target_source,
            action_support_target_source=args.action_support_target_source,
            terminal_edit_action_context_source=args.terminal_edit_action_context_source,
            action_edge_pair_negative_mode=args.action_edge_pair_negative_mode,
            action_edge_pair_negative_count=args.action_edge_pair_negative_count,
            action_edge_pair_dense_negative_count=args.action_edge_pair_dense_negative_count,
            action_edge_pair_target_source=args.action_edge_pair_target_source,
            vacancy_pair_negative_count=args.vacancy_pair_negative_count,
            vacancy_pair_structured_negative_count=args.vacancy_pair_structured_negative_count,
        )
        print(json.dumps({"val": metrics, "dataset": dataset_stats}, ensure_ascii=False, indent=2))
        return

    log_path = save_dir / "training_log.txt"
    metrics_path = save_dir / "metrics.json"
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_metrics = _train_epoch(
            model,
            train_loader,
            optimizer,
            args.device,
            max_changed_sites,
            weights,
            epoch=epoch,
            total_epochs=args.epochs,
            tau_supervision_mode=args.tau_supervision_mode,
            proj_every_n_batches=args.proj_every_n_batches,
            aux_anneal=not args.no_aux_anneal,
            mask_sparsity_weight=args.mask_sparsity_weight,
            count_loss_weight=args.count_loss_weight,
            detach_proj_encoder=args.detach_proj_encoder,
            reward_magnitude_weight=args.reward_magnitude_weight if epoch > args.gate_warmup_epochs else 0.0,
            reward_gated_weight=args.reward_gated_weight,
            reward_gate_weight=args.reward_gate_weight,
            reward_zero_weight=args.reward_zero_weight,
            reward_sign_weight=args.reward_sign_weight,
            reward_prediction_source=args.reward_prediction_source,
            reward_edit_context_source=args.reward_edit_context_source,
            noop_change_weight=args.noop_change_weight,
            noop_type_copy_weight=args.noop_type_copy_weight,
            projected_noop_fp_weight=args.projected_noop_fp_weight,
            noop_risk_weight=args.noop_risk_weight,
            prior_noop_risk_weight=args.prior_noop_risk_weight,
            proposal_support_weight=args.proposal_support_weight,
            prior_proposal_support_weight=args.prior_proposal_support_weight,
            proposal_hard_negative_weight=args.proposal_hard_negative_weight,
            proposal_rank_margin_weight=args.proposal_rank_margin_weight,
            proposal_candidate_positive_weight=args.proposal_candidate_positive_weight,
            proposal_candidate_negative_weight=args.proposal_candidate_negative_weight,
            proposal_candidate_rank_margin_weight=args.proposal_candidate_rank_margin_weight,
            proposal_target_source=args.proposal_target_source,
            action_support_weight=args.action_support_weight,
            prior_action_support_weight=args.prior_action_support_weight,
            action_support_hard_negative_weight=args.action_support_hard_negative_weight,
            action_support_rank_margin_weight=args.action_support_rank_margin_weight,
            action_support_candidate_positive_weight=args.action_support_candidate_positive_weight,
            action_support_candidate_negative_weight=args.action_support_candidate_negative_weight,
            action_support_candidate_rank_margin_weight=args.action_support_candidate_rank_margin_weight,
            action_support_target_source=args.action_support_target_source,
            terminal_edit_support_weight=args.terminal_edit_support_weight,
            prior_terminal_edit_support_weight=args.prior_terminal_edit_support_weight,
            terminal_edit_support_hard_negative_weight=args.terminal_edit_support_hard_negative_weight,
            terminal_edit_support_rank_margin_weight=args.terminal_edit_support_rank_margin_weight,
            terminal_edit_support_candidate_positive_weight=args.terminal_edit_support_candidate_positive_weight,
            terminal_edit_support_candidate_negative_weight=args.terminal_edit_support_candidate_negative_weight,
            terminal_edit_support_candidate_rank_margin_weight=args.terminal_edit_support_candidate_rank_margin_weight,
            terminal_edit_support_target_source=args.terminal_edit_support_target_source,
            terminal_edit_action_context_source=args.terminal_edit_action_context_source,
            terminal_typed_diff_weight=args.terminal_typed_diff_weight,
            prior_terminal_typed_diff_weight=args.prior_terminal_typed_diff_weight,
            terminal_typed_diff_copy_weight=args.terminal_typed_diff_copy_weight,
            terminal_typed_diff_support_weight=args.terminal_typed_diff_support_weight,
            action_source_support_weight=args.action_source_support_weight,
            prior_action_source_support_weight=args.prior_action_source_support_weight,
            action_destination_support_weight=args.action_destination_support_weight,
            prior_action_destination_support_weight=args.prior_action_destination_support_weight,
            action_edge_pair_weight=args.action_edge_pair_weight,
            prior_action_edge_pair_weight=args.prior_action_edge_pair_weight,
            action_edge_pair_support_weight=args.action_edge_pair_support_weight,
            prior_action_edge_pair_support_weight=args.prior_action_edge_pair_support_weight,
            action_edge_pair_semantic_weight=args.action_edge_pair_semantic_weight,
            prior_action_edge_pair_semantic_weight=args.prior_action_edge_pair_semantic_weight,
            action_edge_pair_negative_weight=args.action_edge_pair_negative_weight,
            action_edge_pair_rank_margin_weight=args.action_edge_pair_rank_margin_weight,
            action_edge_pair_negative_mode=args.action_edge_pair_negative_mode,
            action_edge_pair_negative_count=args.action_edge_pair_negative_count,
            action_edge_pair_dense_negative_count=args.action_edge_pair_dense_negative_count,
            action_edge_pair_target_source=args.action_edge_pair_target_source,
            vacancy_pair_weight=args.vacancy_pair_weight,
            prior_vacancy_pair_weight=args.prior_vacancy_pair_weight,
            vacancy_pair_semantic_weight=args.vacancy_pair_semantic_weight,
            prior_vacancy_pair_semantic_weight=args.prior_vacancy_pair_semantic_weight,
            vacancy_pair_listwise_weight=args.vacancy_pair_listwise_weight,
            prior_vacancy_pair_listwise_weight=args.prior_vacancy_pair_listwise_weight,
            vacancy_pair_interaction_weight=args.vacancy_pair_interaction_weight,
            prior_vacancy_pair_interaction_weight=args.prior_vacancy_pair_interaction_weight,
            vacancy_pair_interaction_listwise_weight=args.vacancy_pair_interaction_listwise_weight,
            prior_vacancy_pair_interaction_listwise_weight=args.prior_vacancy_pair_interaction_listwise_weight,
            vacancy_pair_negative_count=args.vacancy_pair_negative_count,
            vacancy_pair_structured_negative_count=args.vacancy_pair_structured_negative_count,
            candidate_quality_weight=args.candidate_quality_weight,
            prior_candidate_quality_weight=args.prior_candidate_quality_weight,
        )
        elapsed = time.time() - t0
        train_msg = (
            f"[Epoch {epoch:03d}/{args.epochs}] loss={train_metrics['loss']:.4f} "
            f"mask={train_metrics['mask']:.4f} count={train_metrics['count']:.4f} pair={train_metrics['pair']:.4f} "
            f"prior_pair={train_metrics['prior_pair']:.4f} proj_mask={train_metrics['proj_mask']:.4f} type={train_metrics['type']:.4f} "
            f"noop_change={train_metrics['noop_change']:.4f} noop_type={train_metrics['noop_type_copy']:.4f} "
            f"prior_edit={train_metrics['prior_edit']:.4f} "
            f"tau={train_metrics['tau']:.4f} tau_prior={train_metrics['tau_prior']:.4f} "
            f"tau_mu={train_metrics['tau_log_mu']:.4f} "
            f"tau_post={train_metrics['tau_post']:.4f} real_tau={train_metrics['realized_tau']:.4f} "
            f"real_tau_prior={train_metrics['realized_tau_prior']:.4f} real_tau_post={train_metrics['realized_tau_post']:.4f} "
            f"tau_post_scale={train_metrics['tau_post_scale']:.2f} reward={train_metrics['reward']:.4f} "
            f"noop_risk={train_metrics['noop_risk']:.4f}/{train_metrics['prior_noop_risk']:.4f} "
            f"proposal={train_metrics['proposal']:.4f}/{train_metrics['prior_proposal']:.4f} "
            f"prop_cand={train_metrics['proposal_candidate_positive']:.4f}/"
            f"{train_metrics['proposal_candidate_negative']:.4f}/"
            f"{train_metrics['proposal_candidate_rank_margin']:.4f} "
            f"action={train_metrics['action_support']:.4f}/{train_metrics['prior_action_support']:.4f} "
            f"action_cand={train_metrics['action_support_candidate_positive']:.4f}/"
            f"{train_metrics['action_support_candidate_negative']:.4f}/"
            f"{train_metrics['action_support_candidate_rank_margin']:.4f} "
            f"candq={train_metrics['candidate_quality']:.4f}/"
            f"{train_metrics['prior_candidate_quality']:.4f} "
            f"candq_mae={train_metrics['candidate_quality_mae']:.4f} "
            f"candq_corr={train_metrics['candidate_quality_corr']:.4f} "
            f"prop_topk={train_metrics['proposal_topk_f1']:.4f} prop_r32={train_metrics['proposal_recall32']:.4f} "
            f"action_topk={train_metrics['action_support_topk_f1']:.4f} "
            f"action_r32={train_metrics['action_support_recall32']:.4f} "
            f"src_topk={train_metrics['action_source_topk_f1']:.4f} "
            f"src_r32={train_metrics['action_source_recall32']:.4f} "
            f"dst_topk={train_metrics['action_destination_topk_f1']:.4f} "
            f"dst_r32={train_metrics['action_destination_recall32']:.4f} "
            f"endpoint_topk={train_metrics['action_endpoint_topk_f1']:.4f} "
            f"endpoint_r32={train_metrics['action_endpoint_recall32']:.4f} "
            f"term_edit={train_metrics['terminal_edit_support']:.4f}/"
            f"{train_metrics['prior_terminal_edit_support']:.4f} "
            f"term_topk={train_metrics['terminal_edit_topk_f1']:.4f} "
            f"term_r32={train_metrics['terminal_edit_recall32']:.4f} "
            f"typed={train_metrics['terminal_typed_diff']:.4f}/"
            f"{train_metrics['prior_terminal_typed_diff']:.4f} "
            f"typed_sup={train_metrics['terminal_typed_diff_support']:.4f}/"
            f"{train_metrics['prior_terminal_typed_diff_support']:.4f} "
            f"typed_acc={train_metrics['terminal_typed_diff_type_acc']:.4f} "
            f"typed_f1={train_metrics['terminal_typed_diff_topk_f1']:.4f} "
            f"edge_pair={train_metrics['action_edge_pair']:.4f}/"
            f"{train_metrics['prior_action_edge_pair']:.4f} "
            f"edge_acc={train_metrics['action_edge_pair_rank_acc']:.4f} "
            f"edge_sup={train_metrics['action_edge_pair_support']:.4f}/"
            f"{train_metrics['prior_action_edge_pair_support']:.4f} "
            f"edge_sup_acc={train_metrics['action_edge_pair_support_rank_acc']:.4f} "
            f"edge_sup_prob={train_metrics['action_edge_pair_support_prob']:.3f}/"
            f"{train_metrics['action_edge_pair_support_nonsupport_prob']:.3f} "
            f"edge_sem={train_metrics['action_edge_pair_semantic']:.4f}/"
            f"{train_metrics['prior_action_edge_pair_semantic']:.4f} "
            f"edge_type_acc={train_metrics['action_edge_pair_moving_type_acc']:.4f} "
            f"edge_order_mae={train_metrics['action_edge_pair_order_mae']:.4f} "
            f"vac_pair={train_metrics['vacancy_pair']:.4f}/"
            f"{train_metrics['prior_vacancy_pair']:.4f} "
            f"vac_acc={train_metrics['vacancy_pair_rank_acc']:.4f} "
            f"vac_int={train_metrics['vacancy_pair_interaction']:.4f}/"
            f"{train_metrics['prior_vacancy_pair_interaction']:.4f} "
            f"vac_int_acc={train_metrics['vacancy_pair_interaction_rank_acc']:.4f} "
            f"vac_int_lw={train_metrics['vacancy_pair_interaction_listwise']:.4f}/"
            f"{train_metrics['prior_vacancy_pair_interaction_listwise']:.4f} "
            f"vac_int_lw_acc={train_metrics['vacancy_pair_interaction_listwise_acc']:.4f} "
            f"vac_sem={train_metrics['vacancy_pair_semantic']:.4f}/"
            f"{train_metrics['prior_vacancy_pair_semantic']:.4f} "
            f"vac_type_acc={train_metrics['vacancy_pair_moving_type_acc']:.4f} "
            f"vac_order_mae={train_metrics['vacancy_pair_order_mae']:.4f} "
            f"latent={train_metrics['latent']:.4f} proj={train_metrics['proj']:.4f} "
            f"path={train_metrics['path']:.4f} prior_latent={train_metrics['prior_latent']:.4f} "
            f"mask_aux={train_metrics['mask_aux_scale']:.2f} time={elapsed:.1f}s"
        )
        print(train_msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as fp:
            fp.write(train_msg + "\n")

        if epoch % args.eval_freq == 0 or epoch == 1:
            val_metrics = _evaluate(
                model,
                val_loader,
                args.device,
                max_changed_sites,
                reward_prediction_source=args.reward_prediction_source,
                reward_edit_context_source=args.reward_edit_context_source,
                proposal_target_source=args.proposal_target_source,
                action_support_target_source=args.action_support_target_source,
                terminal_edit_action_context_source=args.terminal_edit_action_context_source,
                action_edge_pair_negative_mode=args.action_edge_pair_negative_mode,
                action_edge_pair_negative_count=args.action_edge_pair_negative_count,
                action_edge_pair_dense_negative_count=args.action_edge_pair_dense_negative_count,
                action_edge_pair_target_source=args.action_edge_pair_target_source,
                vacancy_pair_negative_count=args.vacancy_pair_negative_count,
                vacancy_pair_structured_negative_count=args.vacancy_pair_structured_negative_count,
            )
            if val_metrics.get("realized_tau_available", 1.0) >= 0.5:
                realized_tau_msg = (
                    f"real_tau_nll={val_metrics['realized_tau_nll']:.4f} "
                    f"real_tau_log_mae={val_metrics['realized_tau_log_mae']:.4f} "
                    f"real_tau_cov68={val_metrics['realized_tau_coverage_68']:.4f} "
                    f"real_tau_pit_ks={val_metrics['realized_tau_pit_ks']:.4f} "
                )
            else:
                realized_tau_msg = "real_tau=unavailable "
            val_msg = (
                f"  >>> VAL reward_mae={val_metrics['reward_mae']:.4f} reward_corr={val_metrics['reward_corr']:.4f} "
                f"tau_log_mae={val_metrics['tau_log_mae']:.4f} tau_log_corr={val_metrics['tau_log_corr']:.4f} "
                f"tau_scale={val_metrics['tau_scale_ratio']:.2f} {realized_tau_msg}change_f1={val_metrics['change_f1']:.4f} "
                f"change_topk_f1={val_metrics['change_topk_f1']:.4f} proj_change_f1={val_metrics['projected_change_f1']:.4f} "
                f"chg_type_acc={val_metrics['changed_type_acc']:.4f} proj_chg_type_acc={val_metrics['projected_changed_type_acc']:.4f} "
                f"proposal_topk={val_metrics['proposal_topk_f1']:.4f} proposal_r32={val_metrics['proposal_recall32']:.4f} "
                f"action_topk={val_metrics['action_support_topk_f1']:.4f} "
                f"action_r32={val_metrics['action_support_recall32']:.4f} "
                f"src_topk={val_metrics['action_source_topk_f1']:.4f} src_r32={val_metrics['action_source_recall32']:.4f} "
                f"dst_topk={val_metrics['action_destination_topk_f1']:.4f} dst_r32={val_metrics['action_destination_recall32']:.4f} "
                f"endpoint_topk={val_metrics['action_endpoint_topk_f1']:.4f} endpoint_r32={val_metrics['action_endpoint_recall32']:.4f} "
                f"term_topk={val_metrics['terminal_edit_topk_f1']:.4f} term_r32={val_metrics['terminal_edit_recall32']:.4f} "
                f"typed_acc={val_metrics['terminal_typed_diff_type_acc']:.4f} "
                f"typed_f1={val_metrics['terminal_typed_diff_topk_f1']:.4f} "
                f"edge_acc={val_metrics['action_edge_pair_rank_acc']:.4f} "
                f"edge_prob={val_metrics['action_edge_pair_pos_prob']:.3f}/"
                f"{val_metrics['action_edge_pair_neg_prob']:.3f} "
                f"edge_sup_acc={val_metrics['action_edge_pair_support_rank_acc']:.4f} "
                f"edge_sup_prob={val_metrics['action_edge_pair_support_prob']:.3f}/"
                f"{val_metrics['action_edge_pair_support_nonsupport_prob']:.3f} "
                f"edge_type_acc={val_metrics['action_edge_pair_moving_type_acc']:.4f} "
                f"edge_order_mae={val_metrics['action_edge_pair_order_mae']:.4f} "
                f"vac_acc={val_metrics['vacancy_pair_rank_acc']:.4f} "
                f"vac_prob={val_metrics['vacancy_pair_pos_prob']:.3f}/"
                f"{val_metrics['vacancy_pair_neg_prob']:.3f} "
                f"vac_type_acc={val_metrics['vacancy_pair_moving_type_acc']:.4f} "
                f"vac_order_mae={val_metrics['vacancy_pair_order_mae']:.4f} "
                f"noop_risk={val_metrics['noop_risk_noop_pred_mean']:.3f}/{val_metrics['noop_risk_nonnoop_pred_mean']:.3f} "
                f"pair_fe={val_metrics['raw_vac_to_fe_count']:.2f}/{val_metrics['raw_fe_to_vac_count']:.2f} "
                f"pair_cu={val_metrics['raw_vac_to_cu_count']:.2f}/{val_metrics['raw_cu_to_vac_count']:.2f} "
                f"matched_pair={val_metrics['raw_matched_pair_count']:.2f} "
                f"unchg_copy_acc={val_metrics['unchanged_copy_acc']:.4f} vac_copy_acc={val_metrics['unchanged_vacancy_copy_acc']:.4f} "
                f"reach_violation={val_metrics['reachability_violation_rate']:.4f} proj_global_l1={val_metrics['projected_global_l1']:.4f}"
            )
            print(val_msg, flush=True)
            with open(log_path, "a", encoding="utf-8") as fp:
                fp.write(val_msg + "\n")
            metrics_payload = {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "dataset": dataset_stats,
            }
            score = _selection_score(val_metrics, dataset_stats, proj_l1_score_weight=args.proj_l1_score_weight)
            metrics_payload["selection_score"] = score
            metrics_path.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            if score < best_score:
                best_score = score
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "best_score": best_score,
                        "args": vars(args),
                        "dataset": dataset_stats,
                    },
                    save_dir / "best_model.pt",
                )
                print(f"  >>> New best model: score={best_score:.4f}", flush=True)

        if epoch % args.save_freq == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_score": best_score,
                    "args": vars(args),
                    "dataset": dataset_stats,
                },
                save_dir / f"checkpoint_{epoch}.pt",
            )

    final_metrics = _evaluate(
        model,
        val_loader,
        args.device,
        max_changed_sites,
        reward_prediction_source=args.reward_prediction_source,
        reward_edit_context_source=args.reward_edit_context_source,
        proposal_target_source=args.proposal_target_source,
        action_support_target_source=args.action_support_target_source,
        terminal_edit_action_context_source=args.terminal_edit_action_context_source,
        action_edge_pair_negative_mode=args.action_edge_pair_negative_mode,
        action_edge_pair_negative_count=args.action_edge_pair_negative_count,
        action_edge_pair_dense_negative_count=args.action_edge_pair_dense_negative_count,
        action_edge_pair_target_source=args.action_edge_pair_target_source,
        vacancy_pair_negative_count=args.vacancy_pair_negative_count,
        vacancy_pair_structured_negative_count=args.vacancy_pair_structured_negative_count,
    )
    print(json.dumps({"final_val": final_metrics, "dataset": dataset_stats}, ensure_ascii=False, indent=2), flush=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": args.epochs,
            "best_score": best_score,
            "args": vars(args),
            "dataset": dataset_stats,
            "final_val": final_metrics,
        },
        save_dir / "final_model.pt",
    )


if __name__ == "__main__":
    main()
