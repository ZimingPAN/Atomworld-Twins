from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train_dreamer_macro_edit as mod
import eval_macro_time_alignment as eval_mod
import eval_macro_long_trajectory as long_eval_mod
from dreamer4.macro_edit import (
    MacroDreamerEditModel,
    macro_duration_baseline_log_tau,
    project_types_by_inventory,
    projected_edit_logits_from_types,
    teacher_path_summary_dim,
)
from train_dreamer_macro_edit import (
    MacroKMCEnv,
    _build_loader,
    _compute_reward_diagnostics,
    _collect_segments,
    _edit_supervision_losses,
    _evaluate,
    _initialize_reward_heads,
    _initialize_best_score_from_saved_best,
    _load_model_weights,
    _matched_pair_count_loss,
    _normalize_segment_ks,
    _projected_mask_distill_loss,
    _projected_state_alignment_loss,
    _reward_supervision_losses,
    _selection_score,
    _soft_directional_transition_counts,
    _soft_typed_change_count,
    _train_epoch,
    _validate_resume_args,
)


def test_inventory_projection_preserves_patch_counts():
    current_types = torch.tensor([[0, 1, 2, 0, 1]], dtype=torch.long)
    change_logits = torch.tensor([[8.0, 6.0, 5.0, -4.0, -5.0]])
    type_logits = torch.tensor(
        [
            [
                [0.1, 3.0, 0.2],
                [2.5, 0.1, 0.3],
                [0.2, 0.3, 2.7],
                [1.5, 0.1, 0.1],
                [0.2, 1.6, 0.2],
            ]
        ],
        dtype=torch.float32,
    )
    node_mask = torch.ones_like(change_logits)
    positions = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0], [4.0, 4.0, 4.0]]])
    box_dims = torch.tensor([[20.0, 20.0, 20.0]])
    horizon_k = torch.tensor([3])
    final_types, _, transport_cost, violations = project_types_by_inventory(
        current_types=current_types,
        change_logits=change_logits,
        type_logits=type_logits,
        node_mask=node_mask,
        positions=positions,
        box_dims=box_dims,
        horizon_k=horizon_k,
        max_changed_sites=3,
    )
    current_counts = torch.bincount(current_types[0], minlength=3)
    final_counts = torch.bincount(final_types[0], minlength=3)
    assert torch.equal(current_counts, final_counts)
    assert float(violations[0].item()) in {0.0, 1.0}
    assert float(transport_cost[0].item()) >= 0.0


def test_projection_skips_violation_when_change_mass_has_no_typed_swap_support():
    current_types = torch.tensor([[2, 0, 2, 1, 0, 0, 1, 0]], dtype=torch.long)
    change_logits = torch.zeros((1, 8), dtype=torch.float32)
    type_logits = torch.full((1, 8, 3), -4.0, dtype=torch.float32)
    for idx, type_id in enumerate(current_types[0].tolist()):
        type_logits[0, idx, type_id] = 4.0
    node_mask = torch.ones_like(change_logits)
    positions = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 0.0, 0.0], [3.0, 1.0, 1.0], [4.0, 0.0, 0.0], [5.0, 1.0, 1.0], [6.0, 0.0, 0.0], [7.0, 1.0, 1.0]]],
        dtype=torch.float32,
    )
    box_dims = torch.tensor([[20.0, 20.0, 20.0]], dtype=torch.float32)
    horizon_k = torch.tensor([4], dtype=torch.long)

    final_types, changed_mask, _transport_cost, violations = project_types_by_inventory(
        current_types=current_types,
        change_logits=change_logits,
        type_logits=type_logits,
        node_mask=node_mask,
        positions=positions,
        box_dims=box_dims,
        horizon_k=horizon_k,
        max_changed_sites=8,
    )

    assert torch.equal(final_types, current_types)
    assert torch.equal(changed_mask, torch.zeros_like(node_mask))
    assert float(violations[0].item()) == 0.0


def test_soft_typed_change_count_requires_noncopy_type_support():
    change_logits = torch.zeros((1, 4), dtype=torch.float32)
    current_types = torch.tensor([[0, 1, 2, 0]], dtype=torch.long)
    candidate_mask = torch.ones((1, 4), dtype=torch.float32)
    copy_type_logits = torch.full((1, 4, 3), -6.0, dtype=torch.float32)
    for idx, type_id in enumerate(current_types[0].tolist()):
        copy_type_logits[0, idx, type_id] = 6.0

    count = _soft_typed_change_count(
        change_logits=change_logits,
        type_logits=copy_type_logits,
        current_types=current_types,
        candidate_mask=candidate_mask,
    )

    assert float(count.item()) < 1e-3


def test_soft_directional_transition_counts_track_both_swap_halves():
    change_logits = torch.full((1, 4), 8.0, dtype=torch.float32)
    current_types = torch.tensor([[2, 0, 2, 1]], dtype=torch.long)
    candidate_mask = torch.ones((1, 4), dtype=torch.float32)
    type_logits = torch.full((1, 4, 3), -6.0, dtype=torch.float32)
    type_logits[0, 0, 0] = 6.0
    type_logits[0, 1, 2] = 6.0
    type_logits[0, 2, 1] = 6.0
    type_logits[0, 3, 2] = 6.0

    counts = _soft_directional_transition_counts(
        change_logits=change_logits,
        type_logits=type_logits,
        current_types=current_types,
        candidate_mask=candidate_mask,
    )

    assert float(counts["vac_to_fe"].item()) > 0.95
    assert float(counts["fe_to_vac"].item()) > 0.95
    assert float(counts["vac_to_cu"].item()) > 0.95
    assert float(counts["cu_to_vac"].item()) > 0.95


def test_matched_pair_count_loss_is_low_when_directional_counts_match_targets():
    change_logits = torch.full((1, 2), 8.0, dtype=torch.float32)
    current_types = torch.tensor([[2, 0]], dtype=torch.long)
    target_types = torch.tensor([[0, 2]], dtype=torch.long)
    candidate_mask = torch.ones((1, 2), dtype=torch.float32)
    type_logits = torch.full((1, 2, 3), -6.0, dtype=torch.float32)
    type_logits[0, 0, 0] = 6.0
    type_logits[0, 1, 2] = 6.0

    loss = _matched_pair_count_loss(
        change_logits=change_logits,
        type_logits=type_logits,
        current_types=current_types,
        target_types=target_types,
        candidate_mask=candidate_mask,
    )

    assert float(loss.item()) < 1e-2


def test_matched_pair_count_loss_penalizes_single_sided_prediction():
    change_logits = torch.full((1, 2), 8.0, dtype=torch.float32)
    current_types = torch.tensor([[2, 0]], dtype=torch.long)
    target_types = torch.tensor([[0, 2]], dtype=torch.long)
    candidate_mask = torch.ones((1, 2), dtype=torch.float32)
    type_logits = torch.full((1, 2, 3), -6.0, dtype=torch.float32)
    type_logits[0, 0, 0] = 6.0
    type_logits[0, 1, 0] = 6.0

    loss = _matched_pair_count_loss(
        change_logits=change_logits,
        type_logits=type_logits,
        current_types=current_types,
        target_types=target_types,
        candidate_mask=candidate_mask,
    )

    assert float(loss.item()) > 0.3


def test_macro_edit_smoke_train_eval():
    cfg = {
        "lattice_size": (10, 10, 10),
        "max_episode_steps": 40,
        "max_vacancies": 8,
        "max_defects": 32,
        "max_shells": 8,
        "stats_dim": 10,
        "temperature": 300.0,
        "reward_scale": 1.0,
        "cu_density": 0.01,
        "v_density": 0.001,
        "rlkmc_topk": 8,
        "neighbor_order": "2NN",
    }
    env = MacroKMCEnv(cfg)
    env.reset()
    rng = np.random.default_rng(0)
    samples, stats = _collect_segments(
        env=env,
        num_segments=6,
        horizon_k=2,
        max_seed_vacancies=8,
        max_candidate_sites=48,
        rng=rng,
        max_attempt_multiplier=20,
    )
    assert len(samples) >= 2
    assert stats["coverage"] > 0.0

    train_samples = samples[:-1]
    val_samples = samples[-1:]
    train_loader = _build_loader(train_samples, batch_size=2, shuffle=False)
    val_loader = _build_loader(val_samples, batch_size=1, shuffle=False)
    model = MacroDreamerEditModel(
        max_vacancies=cfg["max_vacancies"],
        max_defects=cfg["max_defects"],
        max_shells=cfg["max_shells"],
        stats_dim=cfg["stats_dim"],
        lattice_size=cfg["lattice_size"],
        neighbor_order=cfg["neighbor_order"],
        dim_latent=8,
        graph_hidden_size=16,
        patch_hidden_size=32,
        patch_latent_dim=24,
        path_latent_dim=12,
        global_summary_dim=16,
        teacher_path_summary_dim=teacher_path_summary_dim(2),
        max_macro_k=4,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    weights = {
        "mask": 1.0,
        "type": 1.0,
        "pair": 0.0,
        "tau": 1.0,
        "reward": 0.5,
        "latent": 0.5,
        "proj": 0.5,
        "path": 0.05,
        "prior_edit": 0.25,
        "prior_latent": 0.25,
    }
    train_metrics = _train_epoch(model, train_loader, optimizer, "cpu", max_changed_sites=4, weights=weights)
    eval_metrics = _evaluate(model, val_loader, "cpu", max_changed_sites=4)
    assert np.isfinite(train_metrics["loss"])
    assert np.isfinite(eval_metrics["tau_log_mae"])
    assert np.isfinite(eval_metrics["reward_mae"])
    assert "projected_changed_type_acc" in eval_metrics
    assert "unchanged_vacancy_copy_acc" in eval_metrics
    assert "reachability_violation_rate" in eval_metrics
    assert "raw_vac_to_fe_count" in eval_metrics
    assert "raw_fe_to_vac_count" in eval_metrics
    assert "raw_matched_pair_count" in eval_metrics
    assert "pair" in train_metrics
    assert "tau_prior" in train_metrics
    assert "tau_post" in train_metrics


def test_teacher_path_summary_keeps_stepwise_time_sketch():
    path_infos = [
        {
            "dir_idx": 0,
            "moving_type": 1,
            "total_rate": 10.0,
            "expected_delta_t": 0.1,
            "delta_E": 0.5,
            "old_pos": np.asarray([0, 0, 0], dtype=np.int32),
            "new_pos": np.asarray([1, 1, 1], dtype=np.int32),
            "vac_idx": 0,
        },
        {
            "dir_idx": 3,
            "moving_type": 0,
            "total_rate": 4.0,
            "expected_delta_t": 0.25,
            "delta_E": -0.2,
            "old_pos": np.asarray([1, 1, 1], dtype=np.int32),
            "new_pos": np.asarray([2, 2, 2], dtype=np.int32),
            "vac_idx": 1,
        },
    ]

    summary = mod._teacher_path_summary(path_infos, max_candidate_sites=32, horizon_k=4)

    assert summary.shape == (teacher_path_summary_dim(4),)
    assert np.isclose(summary[18], math.log(0.1 + 1e-12), atol=1e-6)
    assert np.isclose(summary[19], math.log(0.25 + 1e-12), atol=1e-6)
    assert np.isclose(summary[22], 0.5, atol=1e-6)
    assert np.isclose(summary[23], -0.2, atol=1e-6)


def test_teacher_path_summary_legacy_mode_keeps_base_dim_only():
    summary = mod._teacher_path_summary([], max_candidate_sites=32, horizon_k=4, include_stepwise_features=False)

    assert summary.shape == (teacher_path_summary_dim(4, include_stepwise_features=False),)
    assert summary.shape == (18,)


def test_teacher_action_sequence_targets_preserve_order_and_rollout_changes():
    candidate_positions = np.asarray(
        [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]],
        dtype=np.int32,
    )
    candidate_mask = np.ones((4,), dtype=np.float32)
    current_types = np.asarray([mod.V_TYPE, mod.CU_TYPE, mod.FE_TYPE, mod.FE_TYPE], dtype=np.int64)
    path_infos = [
        {
            "old_pos": np.asarray([0, 0, 0], dtype=np.int32),
            "new_pos": np.asarray([1, 1, 1], dtype=np.int32),
            "moving_type": mod.CU_TYPE,
        },
        {
            "old_pos": np.asarray([1, 1, 1], dtype=np.int32),
            "new_pos": np.asarray([2, 2, 2], dtype=np.int32),
            "moving_type": mod.FE_TYPE,
        },
    ]

    indices, mask, moving_type, order, rollout_changed, step_count, covered = mod._teacher_action_sequence_targets(
        candidate_positions=candidate_positions,
        candidate_mask=candidate_mask,
        current_types=current_types,
        path_infos=path_infos,
    )

    assert step_count == 2
    assert covered == 2
    assert indices[:2].tolist() == [[0, 1], [1, 2]]
    assert mask[:2].tolist() == [1.0, 1.0]
    assert moving_type[:2].tolist() == [mod.CU_TYPE, mod.FE_TYPE]
    assert order[:2].tolist() == pytest.approx([0.0, 1.0])
    assert rollout_changed.tolist() == [1.0, 1.0, 1.0, 0.0]


def test_terminal_action_context_can_use_teacher_rollout_mask():
    fallback_logits = torch.zeros((1, 3), dtype=torch.float32)
    tensors = {
        "candidate_mask": torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32),
        "teacher_action_rollout_changed_mask": torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float32),
    }

    logits = mod._terminal_action_context_logits_from_tensors(
        tensors=tensors,
        source="teacher_rollout",
        fallback_logits=fallback_logits,
    )

    assert float(logits[0, 0].item()) == pytest.approx(6.0)
    assert float(logits[0, 1].item()) == pytest.approx(-6.0)
    assert float(logits[0, 2].item()) == pytest.approx(-20.0)


def test_adaptive_boundary_reached_on_key_moving_type_after_min_k():
    config = mod.AdaptiveBoundaryConfig(
        mode="adaptive_key_event",
        min_k=2,
        key_moving_types=(mod.CU_TYPE,),
    )
    info = {"moving_type": mod.CU_TYPE, "delta_E": 0.0}

    assert not mod._adaptive_boundary_reached(
        config=config,
        step_count=1,
        latest_info=info,
        touched_positions={(0, 0, 0), (1, 0, 0)},
        cumulative_abs_delta_e=0.0,
    )
    assert mod._adaptive_boundary_reached(
        config=config,
        step_count=2,
        latest_info=info,
        touched_positions={(0, 0, 0), (1, 0, 0)},
        cumulative_abs_delta_e=0.0,
    )


def test_collect_segments_adaptive_key_event_uses_realized_horizon(monkeypatch):
    class FakeInner:
        dims = np.array([10, 10, 10], dtype=np.int32)
        NN1 = np.array([[1, 0, 0]], dtype=np.int32)

        def __init__(self):
            self.after = False

        def get_vacancy_array(self):
            if self.after:
                return np.array([[1, 0, 0]], dtype=np.int32)
            return np.array([[0, 0, 0]], dtype=np.int32)

        def get_cu_array(self):
            if self.after:
                return np.array([[0, 0, 0]], dtype=np.int32)
            return np.array([[1, 0, 0]], dtype=np.int32)

    class FakeEnv:
        cfg = {"reward_scale": 10.0}

        def __init__(self):
            self.env = FakeInner()

        def reset(self):
            self.env.after = False
            return np.zeros((2, 2), dtype=np.float32)

        def obs(self):
            return np.zeros((2, 2), dtype=np.float32)

        def step(self, _action):
            self.env.after = True
            info = {
                "expected_delta_t": 0.25,
                "delta_t": 0.3,
                "total_rate": 4.0,
                "delta_E": 0.2,
                "dir_idx": 0,
                "moving_type": mod.CU_TYPE,
                "old_pos": np.array([0, 0, 0], dtype=np.int32),
                "new_pos": np.array([1, 0, 0], dtype=np.int32),
                "vac_idx": 0,
            }
            return np.ones((2, 2), dtype=np.float32), 2.0, False, info

    monkeypatch.setattr(
        mod,
        "_build_candidate_positions",
        lambda *_args, **_kwargs: (
            [(0, 0, 0), (1, 0, 0), (2, 0, 0)],
            {(0, 0, 0): 0, (1, 0, 0): 1, (2, 0, 0): 2},
            np.array([[0, 0, 0]], dtype=np.int32),
        ),
    )
    monkeypatch.setattr(mod, "_global_summary", lambda _env: np.zeros((16,), dtype=np.float32))
    monkeypatch.setattr(mod, "_sample_teacher_action", lambda _env, _rng: object())

    samples, stats = _collect_segments(
        env=FakeEnv(),
        num_segments=1,
        horizon_k=8,
        max_seed_vacancies=1,
        max_candidate_sites=4,
        rng=np.random.default_rng(0),
        max_segments_per_rollout=0,
        teacher_candidate_augmentation=False,
        adaptive_boundary_config=mod.AdaptiveBoundaryConfig(
            mode="adaptive_key_event",
            min_k=1,
            candidate_horizon_source="actual",
            key_moving_types=(mod.CU_TYPE,),
        ),
    )

    assert len(samples) == 1
    assert samples[0].horizon_k == 1
    assert stats["adaptive_boundary_hits"] == 1
    assert stats["adaptive_truncated_at_max"] == 0
    assert stats["avg_realized_horizon"] == pytest.approx(1.0)
    assert samples[0].candidate_mask.sum() == pytest.approx(2.0)


def test_macro_duration_baseline_matches_k_over_total_rate():
    global_summary = torch.zeros((2, 16), dtype=torch.float32)
    global_summary[:, 10] = torch.log(torch.tensor([100.0, 25.0], dtype=torch.float32))
    horizon_k = torch.tensor([4, 2], dtype=torch.long)

    baseline = macro_duration_baseline_log_tau(global_summary, horizon_k)
    expected = torch.log(torch.tensor([4.0 / 100.0, 2.0 / 25.0], dtype=torch.float32))

    assert torch.allclose(baseline, expected, atol=1e-6)


def test_reward_diagnostics_capture_zero_drift_and_sign_errors():
    diagnostics = _compute_reward_diagnostics(
        pred_reward=np.asarray([0.6, 0.2, -0.1, 0.0], dtype=np.float64),
        true_reward=np.asarray([0.0, -1.0, 1.0, 0.0], dtype=np.float64),
    )

    assert diagnostics["zero_target_frac"] == pytest.approx(0.5)
    assert diagnostics["negative_target_frac"] == pytest.approx(0.25)
    assert diagnostics["zero_pred_mean_abs"] == pytest.approx(0.3)
    assert diagnostics["negative_pred_mean"] == pytest.approx(0.2)
    assert diagnostics["nonzero_sign_acc"] == pytest.approx(0.0)


def test_selection_score_penalizes_reward_zero_drift():
    base_metrics = {
        "tau_log_mae": 0.1,
        "reward_mae": 0.5,
        "reward_corr": 0.6,
        "reward_zero_pred_mean_abs": 0.0,
        "reward_negative_pred_mean": -0.2,
        "reward_mean_bias": 0.0,
        "reward_nonzero_sign_acc": 1.0,
        "change_topk_f1": 0.95,
        "projected_change_f1": 0.95,
        "projected_changed_type_acc": 0.95,
        "unchanged_vacancy_copy_acc": 0.99,
        "projected_global_l1": 0.001,
        "reachability_violation_rate": 0.0,
    }
    dataset_stats = {"val": {"coverage": 0.99}}

    drifted_metrics = dict(base_metrics)
    drifted_metrics["reward_zero_pred_mean_abs"] = 0.5
    drifted_metrics["reward_negative_pred_mean"] = 0.2
    drifted_metrics["reward_mean_bias"] = 0.3
    drifted_metrics["reward_nonzero_sign_acc"] = 0.5

    assert _selection_score(drifted_metrics, dataset_stats) > _selection_score(base_metrics, dataset_stats)


def test_predict_reward_and_duration_uses_physical_baseline_when_residual_is_zero():
    model = MacroDreamerEditModel(
        max_vacancies=4,
        max_defects=8,
        max_shells=4,
        stats_dim=10,
        lattice_size=(10, 10, 10),
        neighbor_order="2NN",
        dim_latent=4,
        graph_hidden_size=8,
        patch_hidden_size=16,
        patch_latent_dim=8,
        path_latent_dim=4,
        global_summary_dim=16,
        teacher_path_summary_dim=teacher_path_summary_dim(2),
        max_macro_k=4,
    )
    with torch.no_grad():
        model.duration_head[-1].weight.zero_()
        model.duration_head[-1].bias.zero_()

    global_summary = torch.zeros((1, 16), dtype=torch.float32)
    global_summary[0, 10] = math.log(100.0)
    _reward, tau_mu, _tau_log_sigma, _gate_logit = model.predict_reward_and_duration(
        global_latent=torch.zeros((1, model.global_latent_dim), dtype=torch.float32),
        predicted_next_global=torch.zeros((1, model.global_latent_dim), dtype=torch.float32),
        path_latent=torch.zeros((1, model.path_latent_dim), dtype=torch.float32),
        global_summary=global_summary,
        horizon_k=torch.tensor([4], dtype=torch.long),
    )

    assert torch.allclose(tau_mu, torch.tensor([math.log(4.0 / 100.0)], dtype=torch.float32), atol=1e-6)


def test_initialize_reward_heads_zero_centers_reward_branch():
    model = MacroDreamerEditModel(
        max_vacancies=4,
        max_defects=8,
        max_shells=4,
        stats_dim=10,
        lattice_size=(10, 10, 10),
        neighbor_order="2NN",
        dim_latent=4,
        graph_hidden_size=8,
        patch_hidden_size=16,
        patch_latent_dim=8,
        path_latent_dim=4,
        global_summary_dim=16,
        teacher_path_summary_dim=teacher_path_summary_dim(2),
        max_macro_k=4,
    )

    _initialize_reward_heads(
        model,
        [
            SimpleNamespace(reward_sum=0.0),
            SimpleNamespace(reward_sum=1.0),
            SimpleNamespace(reward_sum=-1.0),
        ],
    )

    assert torch.allclose(model.reward_head[-1].bias, torch.zeros_like(model.reward_head[-1].bias))
    assert torch.allclose(model.reward_context_head[-1].bias, torch.zeros_like(model.reward_context_head[-1].bias))
    assert torch.allclose(model.reward_gate_context_head[-1].bias, torch.zeros_like(model.reward_gate_context_head[-1].bias))
    assert torch.isfinite(model.reward_gate_head[-1].bias).all()


def test_predict_reward_and_durations_returns_expected_and_realized_heads():
    model = MacroDreamerEditModel(
        max_vacancies=4,
        max_defects=8,
        max_shells=4,
        stats_dim=10,
        lattice_size=(10, 10, 10),
        neighbor_order="2NN",
        dim_latent=4,
        graph_hidden_size=8,
        patch_hidden_size=16,
        patch_latent_dim=8,
        path_latent_dim=4,
        global_summary_dim=16,
        teacher_path_summary_dim=teacher_path_summary_dim(2),
        max_macro_k=4,
    )
    with torch.no_grad():
        model.duration_head[-1].weight.zero_()
        model.realized_duration_head[-1].weight.zero_()

    global_summary = torch.zeros((1, 16), dtype=torch.float32)
    global_summary[0, 10] = math.log(100.0)
    outputs = model.predict_reward_and_durations(
        global_latent=torch.zeros((1, model.global_latent_dim), dtype=torch.float32),
        predicted_next_global=torch.zeros((1, model.global_latent_dim), dtype=torch.float32),
        path_latent=torch.zeros((1, model.path_latent_dim), dtype=torch.float32),
        global_summary=global_summary,
        horizon_k=torch.tensor([4], dtype=torch.long),
    )

    assert set(outputs) == {
        "reward",
        "expected_tau_mu",
        "expected_tau_log_sigma",
        "realized_tau_mu",
        "realized_tau_log_sigma",
        "gate_logit",
        "noop_risk_logit",
    }
    baseline = torch.tensor([math.log(4.0 / 100.0)], dtype=torch.float32)
    assert torch.allclose(outputs["expected_tau_mu"], baseline, atol=1e-6)
    assert torch.allclose(outputs["realized_tau_mu"], baseline, atol=1e-6)
    assert torch.allclose(outputs["expected_tau_log_sigma"], torch.tensor([-2.0], dtype=torch.float32), atol=1e-6)
    assert torch.allclose(outputs["realized_tau_log_sigma"], torch.tensor([-1.0], dtype=torch.float32), atol=1e-6)


def test_detached_duration_inputs_block_backbone_gradients():
    model = MacroDreamerEditModel(
        max_vacancies=4,
        max_defects=8,
        max_shells=4,
        stats_dim=10,
        lattice_size=(10, 10, 10),
        neighbor_order="2NN",
        dim_latent=4,
        graph_hidden_size=8,
        patch_hidden_size=16,
        patch_latent_dim=8,
        path_latent_dim=4,
        global_summary_dim=16,
        teacher_path_summary_dim=teacher_path_summary_dim(2),
        max_macro_k=4,
    )

    global_summary = torch.zeros((1, 16), dtype=torch.float32)
    global_summary[0, 10] = math.log(80.0)

    global_latent = torch.randn((1, model.global_latent_dim), dtype=torch.float32, requires_grad=True)
    next_global = torch.randn((1, model.global_latent_dim), dtype=torch.float32, requires_grad=True)
    path_latent = torch.randn((1, model.path_latent_dim), dtype=torch.float32, requires_grad=True)
    _reward, tau_mu, tau_log_sigma, _gate_logit = model.predict_reward_and_duration(
        global_latent=global_latent,
        predicted_next_global=next_global,
        path_latent=path_latent,
        global_summary=global_summary,
        horizon_k=torch.tensor([4], dtype=torch.long),
        detach_duration_inputs=True,
    )
    (tau_mu.sum() + tau_log_sigma.sum()).backward()

    assert global_latent.grad is None or torch.allclose(global_latent.grad, torch.zeros_like(global_latent.grad))
    assert next_global.grad is None or torch.allclose(next_global.grad, torch.zeros_like(next_global.grad))
    assert path_latent.grad is None or torch.allclose(path_latent.grad, torch.zeros_like(path_latent.grad))
    assert model.duration_head[-1].weight.grad is not None
    assert float(model.duration_head[-1].weight.grad.abs().sum().item()) > 0.0


def test_apply_projected_types_handles_non_prefix_origin_candidate():
    sample = mod.MacroSegmentSample(
        start_obs=np.zeros((4,), dtype=np.float32),
        next_obs=np.zeros((4,), dtype=np.float32),
        start_vacancy_positions=np.asarray([[1, 1, 1]], dtype=np.int32),
        start_cu_positions=np.asarray([[0, 0, 0]], dtype=np.int32),
        global_summary=np.zeros((16,), dtype=np.float32),
        teacher_path_summary=np.zeros((teacher_path_summary_dim(2),), dtype=np.float32),
        candidate_positions=np.asarray([[2, 2, 2], [9, 9, 9], [0, 0, 0]], dtype=np.float32),
        nearest_vacancy_offset=np.zeros((3, 3), dtype=np.float32),
        reach_depth=np.zeros((3,), dtype=np.float32),
        is_start_vacancy=np.zeros((3,), dtype=np.float32),
        current_types=np.asarray([0, 0, 1], dtype=np.int64),
        target_types=np.asarray([0, 0, 2], dtype=np.int64),
        candidate_mask=np.asarray([1.0, 0.0, 1.0], dtype=np.float32),
        changed_mask=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        tau_exp=1.0,
        tau_real=1.0,
        reward_sum=0.0,
        horizon_k=2,
        box_dims=np.asarray([10.0, 10.0, 10.0], dtype=np.float32),
    )

    vacancies, cu_positions = mod._apply_projected_types(sample, np.asarray([0, 0, 2], dtype=np.int64))

    assert [0, 0, 0] in vacancies.tolist()
    assert [0, 0, 0] not in cu_positions.tolist()


def test_projected_mask_distill_loss_skips_violation_samples():
    change_logits = torch.zeros((2, 2), dtype=torch.float32)
    projected_changed_mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
    valid_mask = torch.ones((2, 2), dtype=torch.bool)
    reachability_violation = torch.tensor([1.0, 0.0], dtype=torch.float32)

    loss = _projected_mask_distill_loss(
        change_logits=change_logits,
        projected_changed_mask=projected_changed_mask,
        valid_mask=valid_mask,
        reachability_violation=reachability_violation,
    )

    expected = F.binary_cross_entropy_with_logits(change_logits[1, :1], projected_changed_mask[1, :1])
    assert torch.isclose(loss, expected)


def test_projected_state_alignment_loss_skips_violation_samples():
    projected_patch_latent = torch.tensor([[9.0, 9.0], [1.0, 1.0]], dtype=torch.float32)
    target_patch_latent = torch.tensor([[0.0, 0.0], [2.0, 2.0]], dtype=torch.float32)
    projected_global = torch.tensor([[9.0, 9.0], [1.0, 1.0]], dtype=torch.float32)
    next_global = torch.tensor([[0.0, 0.0], [3.0, 3.0]], dtype=torch.float32)
    next_pred = torch.tensor([[0.0, 0.0], [2.0, 2.0]], dtype=torch.float32)
    projected_changed_mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
    reachability_violation = torch.tensor([1.0, 0.0], dtype=torch.float32)

    loss = _projected_state_alignment_loss(
        projected_patch_latent=projected_patch_latent,
        target_patch_latent=target_patch_latent,
        projected_global=projected_global,
        next_global=next_global,
        next_pred=next_pred,
        projected_changed_mask=projected_changed_mask,
        reachability_violation=reachability_violation,
    )

    expected = (
        F.smooth_l1_loss(projected_patch_latent[1:], target_patch_latent[1:])
        + 0.5 * F.smooth_l1_loss(projected_global[1:], next_global[1:])
        + 0.5 * F.smooth_l1_loss(projected_global[1:], next_pred[1:])
    )
    assert torch.isclose(loss, expected)


def test_projected_state_alignment_loss_skips_empty_projected_edit():
    loss = _projected_state_alignment_loss(
        projected_patch_latent=torch.tensor([[1.0, 1.0]], dtype=torch.float32),
        target_patch_latent=torch.tensor([[2.0, 2.0]], dtype=torch.float32),
        projected_global=torch.tensor([[1.0, 1.0]], dtype=torch.float32),
        next_global=torch.tensor([[3.0, 3.0]], dtype=torch.float32),
        next_pred=torch.tensor([[2.0, 2.0]], dtype=torch.float32),
        projected_changed_mask=torch.zeros((1, 2), dtype=torch.float32),
        reachability_violation=torch.zeros((1,), dtype=torch.float32),
    )

    assert torch.isclose(loss, torch.zeros((), dtype=torch.float32))


def test_decode_edit_returns_raw_type_logits_without_copy_prior():
    model = MacroDreamerEditModel(
        max_vacancies=4,
        max_defects=8,
        max_shells=4,
        stats_dim=10,
        lattice_size=(10, 10, 10),
        neighbor_order="2NN",
        dim_latent=4,
        graph_hidden_size=8,
        patch_hidden_size=16,
        patch_latent_dim=8,
        path_latent_dim=4,
        global_summary_dim=16,
        teacher_path_summary_dim=teacher_path_summary_dim(2),
        max_macro_k=4,
    )
    with torch.no_grad():
        model.type_head.weight.zero_()
        model.type_head.bias.zero_()

    current_types = torch.tensor([[0, 1, 2]], dtype=torch.long)
    _change_logits, raw_type_logits = model.decode_edit(
        site_latent=torch.zeros((1, 3, 8), dtype=torch.float32),
        patch_latent=torch.zeros((1, 8), dtype=torch.float32),
        predicted_next_global=torch.zeros((1, model.global_latent_dim), dtype=torch.float32),
        path_latent=torch.zeros((1, 4), dtype=torch.float32),
        horizon_k=torch.tensor([2], dtype=torch.long),
        current_types=current_types,
    )

    assert torch.allclose(raw_type_logits, torch.zeros_like(raw_type_logits))


def test_apply_type_copy_bias_prefers_copy_when_type_head_is_neutral():
    model = MacroDreamerEditModel(
        max_vacancies=4,
        max_defects=8,
        max_shells=4,
        stats_dim=10,
        lattice_size=(10, 10, 10),
        neighbor_order="2NN",
        dim_latent=4,
        graph_hidden_size=8,
        patch_hidden_size=16,
        patch_latent_dim=8,
        path_latent_dim=4,
        global_summary_dim=16,
        teacher_path_summary_dim=teacher_path_summary_dim(2),
        max_macro_k=4,
    )
    current_types = torch.tensor([[0, 1, 2]], dtype=torch.long)
    raw_type_logits = torch.zeros((1, 3, 3), dtype=torch.float32)

    type_logits = model.apply_type_copy_bias(raw_type_logits, current_types)

    assert torch.equal(type_logits.argmax(dim=-1), current_types)


def test_masked_type_cross_entropy_ignores_copy_prior_offset():
    current_types = torch.tensor([[0, 2]], dtype=torch.long)
    target_types = torch.tensor([[2, 0]], dtype=torch.long)
    mask = torch.tensor([[True, True]])
    raw_type_logits = torch.zeros((1, 2, 3), dtype=torch.float32)
    model = MacroDreamerEditModel(
        max_vacancies=4,
        max_defects=8,
        max_shells=4,
        stats_dim=10,
        lattice_size=(10, 10, 10),
        neighbor_order="2NN",
        dim_latent=4,
        graph_hidden_size=8,
        patch_hidden_size=16,
        patch_latent_dim=8,
        path_latent_dim=4,
        global_summary_dim=16,
        teacher_path_summary_dim=teacher_path_summary_dim(2),
        max_macro_k=4,
    )
    biased_type_logits = model.apply_type_copy_bias(raw_type_logits, current_types)

    raw_loss = mod._masked_type_cross_entropy(raw_type_logits, target_types, mask)
    biased_loss = mod._masked_type_cross_entropy(biased_type_logits, target_types, mask)

    assert torch.isclose(raw_loss, torch.tensor(math.log(3.0), dtype=torch.float32), atol=1e-6)
    assert float(biased_loss.item()) > float(raw_loss.item()) + 1.0


def test_edit_supervision_adds_vacancy_to_atom_type_term():
    change_logits = torch.full((1, 2), 8.0, dtype=torch.float32)
    current_types = torch.tensor([[2, 0]], dtype=torch.long)
    target_types = torch.tensor([[0, 2]], dtype=torch.long)
    changed_mask = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
    candidate_mask = torch.ones((1, 2), dtype=torch.float32)
    type_logits = torch.full((1, 2, 3), -6.0, dtype=torch.float32)
    type_logits[0, 0, 2] = 6.0
    type_logits[0, 1, 2] = 6.0

    losses = _edit_supervision_losses(
        change_logits=change_logits,
        type_logits=type_logits,
        current_types=current_types,
        target_types=target_types,
        changed_mask=changed_mask,
        candidate_mask=candidate_mask,
        aux_scale=1.0,
    )

    assert float(losses["vac_to_atom_type"].item()) > 5.0
    assert float(losses["atom_to_vac_type"].item()) < 1e-2


def test_evaluate_uses_raw_type_logits_for_type_metrics(monkeypatch):
    sample = mod.MacroSegmentSample(
        start_obs=np.zeros((4,), dtype=np.float32),
        next_obs=np.zeros((4,), dtype=np.float32),
        start_vacancy_positions=np.asarray([[1, 1, 1]], dtype=np.int32),
        start_cu_positions=np.empty((0, 3), dtype=np.int32),
        global_summary=np.zeros((16,), dtype=np.float32),
        teacher_path_summary=np.zeros((teacher_path_summary_dim(2),), dtype=np.float32),
        candidate_positions=np.asarray([[0, 0, 0], [1, 1, 1]], dtype=np.float32),
        nearest_vacancy_offset=np.zeros((2, 3), dtype=np.float32),
        reach_depth=np.zeros((2,), dtype=np.float32),
        is_start_vacancy=np.asarray([0.0, 1.0], dtype=np.float32),
        current_types=np.asarray([0, 2], dtype=np.int64),
        target_types=np.asarray([2, 2], dtype=np.int64),
        candidate_mask=np.asarray([1.0, 1.0], dtype=np.float32),
        changed_mask=np.asarray([1.0, 0.0], dtype=np.float32),
        tau_exp=1.0,
        tau_real=1.0,
        reward_sum=0.0,
        horizon_k=2,
        box_dims=np.asarray([10.0, 10.0, 10.0], dtype=np.float32),
    )

    class FakeModel:
        def eval(self):
            return self

        def encode_global(self, obs):
            return torch.zeros((obs.shape[0], 2), dtype=torch.float32, device=obs.device)

        def encode_patch(self, *, positions, nearest_vacancy_offset, reach_depth, is_start_vacancy, type_ids, node_mask, global_summary, box_dims):
            batch, sites = positions.shape[:2]
            return (
                torch.zeros((batch, sites, 2), dtype=torch.float32, device=positions.device),
                torch.zeros((batch, 2), dtype=torch.float32, device=positions.device),
            )

        def prior_stats(self, global_latent, global_summary, horizon_k):
            batch = global_latent.shape[0]
            return (
                torch.zeros((batch, 1), dtype=torch.float32, device=global_latent.device),
                torch.zeros((batch, 1), dtype=torch.float32, device=global_latent.device),
            )

        def sample_path_latent(self, mu, logvar, deterministic=False):
            return mu

        def predict_next_global(self, global_latent, path_latent, horizon_k):
            return global_latent

        def decode_edit(self, *, site_latent, patch_latent, predicted_next_global, path_latent, horizon_k, current_types):
            batch = current_types.shape[0]
            change_logits = torch.tensor([[8.0, -8.0]], dtype=torch.float32, device=current_types.device).expand(batch, -1).clone()
            raw_type_logits = torch.tensor(
                [[[1.5, -5.0, 2.0], [-5.0, -5.0, 4.0]]],
                dtype=torch.float32,
                device=current_types.device,
            ).expand(batch, -1, -1).clone()
            return change_logits, raw_type_logits

        def apply_type_copy_bias(self, raw_type_logits, current_types):
            copy_prior = F.one_hot(current_types, num_classes=3).float() * 2.0
            return raw_type_logits + copy_prior

        def predict_reward_and_duration(self, global_latent, predicted_next_global, path_latent, global_summary, horizon_k):
            batch = global_latent.shape[0]
            zeros = torch.zeros((batch,), dtype=torch.float32, device=global_latent.device)
            return zeros, zeros, zeros, zeros

    def fake_project_types_by_inventory(*, current_types, change_logits, type_logits, node_mask, positions, box_dims, horizon_k, max_changed_sites):
        batch = current_types.shape[0]
        return current_types.clone(), torch.zeros_like(node_mask), torch.zeros((batch,), dtype=torch.float32), torch.zeros((batch,), dtype=torch.float32)

    def fake_projected_global_latent_batch(*, batch, projected_types, model, device):
        return torch.zeros((len(batch), 2), dtype=torch.float32, device=device)

    monkeypatch.setattr(mod, "project_types_by_inventory", fake_project_types_by_inventory)
    monkeypatch.setattr(mod, "_projected_global_latent_batch", fake_projected_global_latent_batch)

    metrics = mod._evaluate(FakeModel(), _build_loader([sample], batch_size=1, shuffle=False), "cpu", max_changed_sites=2)

    assert metrics["changed_type_acc"] == pytest.approx(1.0)
    assert metrics["unchanged_vacancy_copy_acc"] == pytest.approx(1.0)
    assert metrics["raw_fe_to_vac_count"] == pytest.approx(1.0)


def test_initialize_output_heads_tracks_empirical_changed_rate_without_extra_sparsification():
    sample = mod.MacroSegmentSample(
        start_obs=np.zeros((4,), dtype=np.float32),
        next_obs=np.zeros((4,), dtype=np.float32),
        start_vacancy_positions=np.asarray([[0, 0, 0]], dtype=np.int32),
        start_cu_positions=np.asarray([[1, 1, 1]], dtype=np.int32),
        global_summary=np.zeros((16,), dtype=np.float32),
        teacher_path_summary=np.zeros((teacher_path_summary_dim(2),), dtype=np.float32),
        candidate_positions=np.zeros((50, 3), dtype=np.float32),
        nearest_vacancy_offset=np.zeros((50, 3), dtype=np.float32),
        reach_depth=np.zeros((50,), dtype=np.float32),
        is_start_vacancy=np.zeros((50,), dtype=np.float32),
        current_types=np.zeros((50,), dtype=np.int64),
        target_types=np.zeros((50,), dtype=np.int64),
        candidate_mask=np.ones((50,), dtype=np.float32),
        changed_mask=np.asarray([1.0] + [0.0] * 49, dtype=np.float32),
        tau_exp=1.0,
        tau_real=1.0,
        reward_sum=0.0,
        horizon_k=2,
        box_dims=np.asarray([10.0, 10.0, 10.0], dtype=np.float32),
    )
    model = MacroDreamerEditModel(
        max_vacancies=4,
        max_defects=8,
        max_shells=4,
        stats_dim=10,
        lattice_size=(10, 10, 10),
        neighbor_order="2NN",
        dim_latent=4,
        graph_hidden_size=8,
        patch_hidden_size=16,
        patch_latent_dim=8,
        path_latent_dim=4,
        global_summary_dim=16,
        teacher_path_summary_dim=teacher_path_summary_dim(2),
        max_macro_k=4,
    )

    mod._initialize_output_heads(model, [sample])

    expected_changed_rate = 1.0 / 50.0
    assert math.isclose(torch.sigmoid(model.change_head.bias).item(), expected_changed_rate, rel_tol=1e-6)


def test_init_from_reward_reinit_policy_runs_after_loaded_weights():
    sample = mod.MacroSegmentSample(
        start_obs=np.zeros((4,), dtype=np.float32),
        next_obs=np.zeros((4,), dtype=np.float32),
        start_vacancy_positions=np.asarray([[0, 0, 0]], dtype=np.int32),
        start_cu_positions=np.asarray([[1, 1, 1]], dtype=np.int32),
        global_summary=np.zeros((16,), dtype=np.float32),
        teacher_path_summary=np.zeros((teacher_path_summary_dim(2),), dtype=np.float32),
        candidate_positions=np.zeros((4, 3), dtype=np.float32),
        nearest_vacancy_offset=np.zeros((4, 3), dtype=np.float32),
        reach_depth=np.zeros((4,), dtype=np.float32),
        is_start_vacancy=np.zeros((4,), dtype=np.float32),
        current_types=np.zeros((4,), dtype=np.int64),
        target_types=np.zeros((4,), dtype=np.int64),
        candidate_mask=np.ones((4,), dtype=np.float32),
        changed_mask=np.asarray([1.0, 1.0, 0.0, 0.0], dtype=np.float32),
        tau_exp=1.0,
        tau_real=1.0,
        reward_sum=1.0,
        horizon_k=2,
        box_dims=np.asarray([10.0, 10.0, 10.0], dtype=np.float32),
    )
    model = MacroDreamerEditModel(
        max_vacancies=4,
        max_defects=8,
        max_shells=4,
        stats_dim=10,
        lattice_size=(10, 10, 10),
        neighbor_order="2NN",
        dim_latent=4,
        graph_hidden_size=8,
        patch_hidden_size=16,
        patch_latent_dim=8,
        path_latent_dim=4,
        global_summary_dim=16,
        teacher_path_summary_dim=teacher_path_summary_dim(2),
        max_macro_k=4,
    )
    with torch.no_grad():
        model.reward_head[-1].bias.fill_(7.0)
        model.reward_gate_head[-1].bias.fill_(7.0)
        model.duration_head[-1].bias.fill_(3.0)

    mod._apply_output_head_initialization_policy(
        model,
        [sample],
        resume=None,
        init_from="previous.pt",
        reinit_output_heads=False,
        reinit_reward_heads=True,
        freeze_duration_heads=False,
    )

    assert torch.allclose(model.reward_head[-1].bias, torch.zeros_like(model.reward_head[-1].bias))
    assert not torch.allclose(model.reward_gate_head[-1].bias, torch.full_like(model.reward_gate_head[-1].bias, 7.0))
    assert torch.allclose(model.duration_head[-1].bias, torch.full_like(model.duration_head[-1].bias, 3.0))


def test_reward_heads_only_training_freezes_non_reward_parameters():
    model = MacroDreamerEditModel(
        max_vacancies=4,
        max_defects=8,
        max_shells=4,
        stats_dim=10,
        lattice_size=(10, 10, 10),
        neighbor_order="2NN",
        dim_latent=4,
        graph_hidden_size=8,
        patch_hidden_size=16,
        patch_latent_dim=8,
        path_latent_dim=4,
        global_summary_dim=16,
        teacher_path_summary_dim=teacher_path_summary_dim(2),
        max_macro_k=4,
    )

    mod._apply_reward_heads_only_training(model)

    reward_prefixes = (
        "reward_head.",
        "reward_context_head.",
        "reward_gate_head.",
        "reward_gate_context_head.",
    )
    for name, param in model.named_parameters():
        assert param.requires_grad == name.startswith(reward_prefixes), name


def test_reward_duration_heads_only_training_freezes_non_calibration_parameters():
    model = MacroDreamerEditModel(
        max_vacancies=4,
        max_defects=8,
        max_shells=4,
        stats_dim=10,
        lattice_size=(10, 10, 10),
        neighbor_order="2NN",
        dim_latent=4,
        graph_hidden_size=8,
        patch_hidden_size=16,
        patch_latent_dim=8,
        path_latent_dim=4,
        global_summary_dim=16,
        teacher_path_summary_dim=teacher_path_summary_dim(2),
        max_macro_k=4,
    )

    mod._apply_reward_duration_heads_only_training(model)

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
    for name, param in model.named_parameters():
        assert param.requires_grad == name.startswith(trainable_prefixes), name


def test_duration_heads_only_training_freezes_reward_parameters():
    model = MacroDreamerEditModel(
        max_vacancies=4,
        max_defects=8,
        max_shells=4,
        stats_dim=10,
        lattice_size=(10, 10, 10),
        neighbor_order="2NN",
        dim_latent=4,
        graph_hidden_size=8,
        patch_hidden_size=16,
        patch_latent_dim=8,
        path_latent_dim=4,
        global_summary_dim=16,
        teacher_path_summary_dim=teacher_path_summary_dim(2),
        max_macro_k=4,
    )

    mod._apply_duration_heads_only_training(model)

    trainable_prefixes = (
        "duration_head.",
        "realized_duration_head.",
        "duration_context_head.",
        "realized_duration_context_head.",
    )
    for name, param in model.named_parameters():
        assert param.requires_grad == name.startswith(trainable_prefixes), name


def test_duration_prior_path_training_freezes_encoders_and_reward_parameters():
    model = MacroDreamerEditModel(
        max_vacancies=4,
        max_defects=8,
        max_shells=4,
        stats_dim=10,
        lattice_size=(10, 10, 10),
        neighbor_order="2NN",
        dim_latent=4,
        graph_hidden_size=8,
        patch_hidden_size=16,
        patch_latent_dim=8,
        path_latent_dim=4,
        global_summary_dim=16,
        teacher_path_summary_dim=teacher_path_summary_dim(2),
        max_macro_k=4,
    )

    mod._apply_duration_prior_path_training(model)

    trainable_prefixes = (
        "k_embed.",
        "path_prior.",
        "macro_dynamics.",
        "duration_head.",
        "realized_duration_head.",
        "duration_context_head.",
        "realized_duration_context_head.",
    )
    for name, param in model.named_parameters():
        assert param.requires_grad == name.startswith(trainable_prefixes), name


def test_action_support_head_only_training_freezes_non_action_support_parameters():
    model = MacroDreamerEditModel(
        max_vacancies=4,
        max_defects=8,
        max_shells=4,
        stats_dim=10,
        lattice_size=(10, 10, 10),
        neighbor_order="2NN",
        dim_latent=4,
        graph_hidden_size=8,
        patch_hidden_size=16,
        patch_latent_dim=8,
        path_latent_dim=4,
        global_summary_dim=16,
        teacher_path_summary_dim=teacher_path_summary_dim(2),
        max_macro_k=4,
    )

    mod._apply_action_support_head_only_training(model)

    for name, param in model.named_parameters():
        assert param.requires_grad == name.startswith("action_support_head."), name


def test_action_endpoint_heads_only_training_freezes_non_endpoint_parameters():
    model = MacroDreamerEditModel(
        max_vacancies=4,
        max_defects=8,
        max_shells=4,
        stats_dim=10,
        lattice_size=(10, 10, 10),
        neighbor_order="2NN",
        dim_latent=4,
        graph_hidden_size=8,
        patch_hidden_size=16,
        patch_latent_dim=8,
        path_latent_dim=4,
        global_summary_dim=16,
        teacher_path_summary_dim=teacher_path_summary_dim(2),
        max_macro_k=4,
    )

    mod._apply_action_endpoint_heads_only_training(model)

    trainable_prefixes = ("action_source_head.", "action_destination_head.")
    for name, param in model.named_parameters():
        assert param.requires_grad == name.startswith(trainable_prefixes), name


def test_terminal_edit_support_head_only_training_freezes_non_terminal_parameters():
    model = MacroDreamerEditModel(
        max_vacancies=4,
        max_defects=8,
        max_shells=4,
        stats_dim=10,
        lattice_size=(10, 10, 10),
        neighbor_order="2NN",
        dim_latent=4,
        graph_hidden_size=8,
        patch_hidden_size=16,
        patch_latent_dim=8,
        path_latent_dim=4,
        global_summary_dim=16,
        teacher_path_summary_dim=teacher_path_summary_dim(2),
        max_macro_k=4,
    )

    mod._apply_terminal_edit_support_head_only_training(model)

    for name, param in model.named_parameters():
        assert param.requires_grad == name.startswith("terminal_edit_support_head."), name


def test_terminal_edit_support_decoder_conditions_on_action_sequence_logits():
    model = MacroDreamerEditModel(
        max_vacancies=4,
        max_defects=8,
        max_shells=4,
        stats_dim=10,
        lattice_size=(10, 10, 10),
        neighbor_order="2NN",
        dim_latent=4,
        graph_hidden_size=8,
        patch_hidden_size=16,
        patch_latent_dim=8,
        path_latent_dim=4,
        global_summary_dim=16,
        teacher_path_summary_dim=teacher_path_summary_dim(2),
        max_macro_k=4,
    )
    site_latent = torch.randn(2, 5, 8)
    patch_latent = torch.randn(2, 8)
    predicted_next_global = torch.randn(2, model.global_latent_dim)
    path_latent = torch.randn(2, 4)
    horizon_k = torch.tensor([2, 4], dtype=torch.long)
    current_types = torch.zeros((2, 5), dtype=torch.long)

    low_action = torch.full((2, 5), -8.0)
    high_action = torch.full((2, 5), 8.0)
    low_logits = model.decode_terminal_edit_support(
        site_latent=site_latent,
        patch_latent=patch_latent,
        predicted_next_global=predicted_next_global,
        path_latent=path_latent,
        horizon_k=horizon_k,
        current_types=current_types,
        action_sequence_logits=low_action,
    )
    high_logits = model.decode_terminal_edit_support(
        site_latent=site_latent,
        patch_latent=patch_latent,
        predicted_next_global=predicted_next_global,
        path_latent=path_latent,
        horizon_k=horizon_k,
        current_types=current_types,
        action_sequence_logits=high_action,
    )

    assert low_logits.shape == (2, 5)
    assert not torch.allclose(low_logits, high_logits)


def test_action_edge_pair_support_head_only_training_freezes_non_support_parameters():
    model = MacroDreamerEditModel(
        max_vacancies=4,
        max_defects=8,
        max_shells=4,
        stats_dim=10,
        lattice_size=(10, 10, 10),
        neighbor_order="2NN",
        dim_latent=4,
        graph_hidden_size=8,
        patch_hidden_size=16,
        patch_latent_dim=8,
        path_latent_dim=4,
        global_summary_dim=16,
        teacher_path_summary_dim=teacher_path_summary_dim(2),
        max_macro_k=4,
    )

    mod._apply_action_edge_pair_support_head_only_training(model)

    for name, param in model.named_parameters():
        assert param.requires_grad == name.startswith("action_edge_pair_support_head."), name


def test_action_edge_pair_dual_heads_only_training_freezes_other_parameters():
    model = MacroDreamerEditModel(
        max_vacancies=4,
        max_defects=8,
        max_shells=4,
        stats_dim=10,
        lattice_size=(10, 10, 10),
        neighbor_order="2NN",
        dim_latent=4,
        graph_hidden_size=8,
        patch_hidden_size=16,
        patch_latent_dim=8,
        path_latent_dim=4,
        global_summary_dim=16,
        teacher_path_summary_dim=teacher_path_summary_dim(2),
        max_macro_k=4,
    )

    mod._apply_action_edge_pair_dual_heads_only_training(model)

    trainable_prefixes = ("action_edge_pair_head.", "action_edge_pair_support_head.")
    for name, param in model.named_parameters():
        assert param.requires_grad == name.startswith(trainable_prefixes), name


def test_action_edge_pair_listwise_heads_only_training_freezes_other_parameters():
    model = MacroDreamerEditModel(
        max_vacancies=4,
        max_defects=8,
        max_shells=4,
        stats_dim=10,
        lattice_size=(10, 10, 10),
        neighbor_order="2NN",
        dim_latent=4,
        graph_hidden_size=8,
        patch_hidden_size=16,
        patch_latent_dim=8,
        path_latent_dim=4,
        global_summary_dim=16,
        teacher_path_summary_dim=teacher_path_summary_dim(2),
        max_macro_k=4,
    )

    mod._apply_action_edge_pair_listwise_heads_only_training(model)

    trainable_prefixes = (
        "action_edge_pair_head.",
        "action_edge_pair_support_head.",
        "action_edge_pair_moving_type_head.",
        "action_edge_pair_order_head.",
        "candidate_quality_head.",
    )
    for name, param in model.named_parameters():
        assert param.requires_grad == name.startswith(trainable_prefixes), name


def test_vacancy_pair_heads_only_training_freezes_other_parameters():
    model = MacroDreamerEditModel(
        max_vacancies=4,
        max_defects=8,
        max_shells=4,
        stats_dim=10,
        lattice_size=(10, 10, 10),
        neighbor_order="2NN",
        dim_latent=4,
        graph_hidden_size=8,
        patch_hidden_size=16,
        patch_latent_dim=8,
        path_latent_dim=4,
        global_summary_dim=16,
        teacher_path_summary_dim=teacher_path_summary_dim(2),
        max_macro_k=4,
    )

    mod._apply_vacancy_pair_heads_only_training(model)

    trainable_prefixes = (
        "vacancy_pair_head.",
        "vacancy_pair_moving_type_head.",
        "vacancy_pair_order_head.",
    )
    for name, param in model.named_parameters():
        assert param.requires_grad == name.startswith(trainable_prefixes), name


def test_vacancy_pair_interaction_head_only_training_freezes_other_parameters():
    model = MacroDreamerEditModel(
        max_vacancies=4,
        max_defects=8,
        max_shells=4,
        stats_dim=10,
        lattice_size=(10, 10, 10),
        neighbor_order="2NN",
        dim_latent=4,
        graph_hidden_size=8,
        patch_hidden_size=16,
        patch_latent_dim=8,
        path_latent_dim=4,
        global_summary_dim=16,
        teacher_path_summary_dim=teacher_path_summary_dim(2),
        max_macro_k=4,
    )

    mod._apply_vacancy_pair_interaction_head_only_training(model)

    for name, param in model.named_parameters():
        assert param.requires_grad == name.startswith("vacancy_pair_interaction_head."), name


def test_vacancy_pair_decoders_return_pairwise_shapes():
    model = MacroDreamerEditModel(
        max_vacancies=4,
        max_defects=8,
        max_shells=4,
        stats_dim=10,
        lattice_size=(10, 10, 10),
        neighbor_order="2NN",
        dim_latent=4,
        graph_hidden_size=8,
        patch_hidden_size=16,
        patch_latent_dim=8,
        path_latent_dim=4,
        global_summary_dim=16,
        teacher_path_summary_dim=teacher_path_summary_dim(2),
        max_macro_k=4,
    )
    site_latent = torch.randn(2, 5, model.patch_latent_dim)
    patch_latent = torch.randn(2, model.patch_latent_dim)
    next_global = torch.randn(2, model.global_latent_dim)
    path_latent = torch.randn(2, 4)
    horizon_k = torch.tensor([2, 4], dtype=torch.long)
    current_types = torch.tensor(
        [
            [mod.V_TYPE, mod.FE_TYPE, mod.CU_TYPE, mod.FE_TYPE, mod.CU_TYPE],
            [mod.V_TYPE, mod.CU_TYPE, mod.FE_TYPE, mod.FE_TYPE, mod.CU_TYPE],
        ],
        dtype=torch.long,
    )
    pair_indices = torch.tensor([[[0, 1], [0, 2], [0, 4]], [[0, 1], [0, 2], [0, 4]]])

    scores = model.decode_vacancy_pairs(
        site_latent=site_latent,
        patch_latent=patch_latent,
        predicted_next_global=next_global,
        path_latent=path_latent,
        horizon_k=horizon_k,
        current_types=current_types,
        edge_pair_indices=pair_indices,
    )
    interaction_scores = model.decode_vacancy_pair_interaction(
        site_latent=site_latent,
        patch_latent=patch_latent,
        predicted_next_global=next_global,
        path_latent=path_latent,
        horizon_k=horizon_k,
        current_types=current_types,
        edge_pair_indices=pair_indices,
    )
    moving_type_logits = model.decode_vacancy_pair_moving_type(
        site_latent=site_latent,
        patch_latent=patch_latent,
        predicted_next_global=next_global,
        path_latent=path_latent,
        horizon_k=horizon_k,
        current_types=current_types,
        edge_pair_indices=pair_indices,
    )
    order_logits = model.decode_vacancy_pair_order(
        site_latent=site_latent,
        patch_latent=patch_latent,
        predicted_next_global=next_global,
        path_latent=path_latent,
        horizon_k=horizon_k,
        current_types=current_types,
        edge_pair_indices=pair_indices,
    )

    assert scores.shape == (2, 3)
    assert interaction_scores.shape == (2, 3)
    assert moving_type_logits.shape == (2, 3, mod.NUM_SITE_TYPES)
    assert order_logits.shape == (2, 3)


def test_action_edge_pair_support_loss_separates_terminal_support_edges():
    pair_logits = torch.tensor([[2.0, -2.0, 0.5]], dtype=torch.float32)
    negative_logits = torch.tensor([[-1.0, 0.25, -0.5]], dtype=torch.float32)
    edge_mask = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32)
    support_mask = torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float32)

    terms = mod._action_edge_pair_support_loss(
        pair_logits,
        negative_logits,
        edge_mask,
        support_mask,
        negative_weight=1.0,
        rank_margin_weight=1.0,
        margin=0.5,
    )

    assert terms["support_prob"].item() > terms["nonsupport_prob"].item()
    assert terms["rank_acc"].item() == pytest.approx(1.0)
    assert terms["support_frac"].item() == pytest.approx(2.0 / 3.0)
    assert terms["loss"].item() > 0.0


def test_action_edge_pair_semantic_loss_uses_moving_type_and_order():
    moving_type_logits = torch.tensor(
        [[[0.0, 4.0, -1.0], [3.0, -2.0, 0.0]]],
        dtype=torch.float32,
    )
    order_logits = torch.tensor([[3.0, -3.0]], dtype=torch.float32)
    edge_mask = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
    moving_type = torch.tensor([[1, 0]], dtype=torch.long)
    order = torch.tensor([[1.0, 0.0]], dtype=torch.float32)

    terms = mod._action_edge_pair_semantic_loss(
        moving_type_logits,
        order_logits,
        edge_mask,
        moving_type,
        order,
    )

    assert terms["moving_type_acc"].item() == pytest.approx(1.0)
    assert terms["order_mae"].item() < 0.06
    assert terms["loss"].item() > 0.0


def test_terminal_vacancy_pair_targets_track_vacancy_identity_across_sequence():
    candidate_mask = np.ones((4,), dtype=np.float32)
    current_types = np.asarray([mod.V_TYPE, mod.FE_TYPE, mod.CU_TYPE, mod.FE_TYPE], dtype=np.int64)
    target_types = np.asarray([mod.FE_TYPE, mod.CU_TYPE, mod.V_TYPE, mod.FE_TYPE], dtype=np.int64)
    action_sequence_indices = np.asarray(
        [
            [0, 1],
            [1, 2],
            [-1, -1],
            [-1, -1],
        ],
        dtype=np.int64,
    )
    action_sequence_mask = np.asarray([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    action_sequence_moving_type = np.asarray([mod.FE_TYPE, mod.CU_TYPE, -1, -1], dtype=np.int64)
    action_sequence_order = np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

    pair_indices, pair_mask, pair_moving_type, pair_order, total_pairs, covered_pairs = (
        mod._teacher_vacancy_displacement_pair_targets_from_sequence(
            candidate_mask=candidate_mask,
            current_types=current_types,
            target_types=target_types,
            action_sequence_indices=action_sequence_indices,
            action_sequence_mask=action_sequence_mask,
            action_sequence_moving_type=action_sequence_moving_type,
            action_sequence_order=action_sequence_order,
        )
    )

    assert total_pairs == 1
    assert covered_pairs == 1
    assert pair_mask.tolist() == [1.0, 0.0, 0.0, 0.0]
    assert pair_indices[0].tolist() == [0, 2]
    assert int(pair_moving_type[0]) == mod.FE_TYPE
    assert float(pair_order[0]) == pytest.approx(1.0)


def test_projection_logits_can_use_action_support_head():
    change_logits = torch.tensor([[0.1, 0.2, 0.3]])
    proposal_logits = torch.tensor([[1.0, 1.1, 1.2]])
    action_support_logits = torch.tensor([[2.0, 2.1, 2.2]])

    selected = mod._projection_logits_from_source(
        change_logits=change_logits,
        proposal_logits=proposal_logits,
        action_support_logits=action_support_logits,
        source="action_support",
        blend_alpha=0.5,
    )

    assert torch.equal(selected, action_support_logits)


def test_projection_logits_can_use_action_endpoint_heads():
    change_logits = torch.tensor([[0.1, 0.2, 0.3]])
    proposal_logits = torch.tensor([[1.0, 1.1, 1.2]])
    action_support_logits = torch.tensor([[2.0, 2.1, 2.2]])
    action_source_logits = torch.tensor([[3.0, 0.5, 4.0]])
    action_destination_logits = torch.tensor([[0.1, 5.0, 2.0]])

    selected = mod._projection_logits_from_source(
        change_logits=change_logits,
        proposal_logits=proposal_logits,
        action_support_logits=action_support_logits,
        action_source_logits=action_source_logits,
        action_destination_logits=action_destination_logits,
        source="action_endpoint",
        blend_alpha=0.5,
    )

    assert torch.equal(selected, torch.maximum(action_source_logits, action_destination_logits))


def test_same_source_nn1_edge_pair_negative_uses_legal_wrong_destination():
    edge_pairs = torch.tensor([[[0, 1], [3, 4]]], dtype=torch.long)
    positions = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [39.0, 39.0, 39.0],
                [5.0, 5.0, 5.0],
                [6.0, 6.0, 6.0],
                [4.0, 4.0, 4.0],
            ]
        ],
        dtype=torch.float32,
    )
    candidate_mask = torch.ones((1, 6), dtype=torch.float32)
    box_dims = torch.tensor([[40.0, 40.0, 40.0]], dtype=torch.float32)

    negatives = mod._negative_action_edge_pair_indices(
        edge_pairs,
        candidate_positions=positions,
        candidate_mask=candidate_mask,
        box_dims=box_dims,
        mode="same_source_nn1",
    )

    assert negatives[0, 0].tolist() == [0, 2]
    assert negatives[0, 1].tolist() == [3, 5]


def test_same_source_nn1_edge_pair_negative_falls_back_to_self_pair():
    edge_pairs = torch.tensor([[[0, 1]]], dtype=torch.long)
    positions = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]], dtype=torch.float32)
    candidate_mask = torch.ones((1, 2), dtype=torch.float32)
    box_dims = torch.tensor([[40.0, 40.0, 40.0]], dtype=torch.float32)

    negatives = mod._negative_action_edge_pair_indices(
        edge_pairs,
        candidate_positions=positions,
        candidate_mask=candidate_mask,
        box_dims=box_dims,
        mode="same_source_nn1",
    )

    assert negatives[0, 0].tolist() == [0, 0]


def test_same_source_nn1_edge_pair_negative_list_returns_multiple_destinations():
    edge_pairs = torch.tensor([[[0, 1]]], dtype=torch.long)
    positions = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [39.0, 39.0, 39.0],
                [1.0, 39.0, 1.0],
            ]
        ],
        dtype=torch.float32,
    )
    candidate_mask = torch.ones((1, 4), dtype=torch.float32)
    box_dims = torch.tensor([[40.0, 40.0, 40.0]], dtype=torch.float32)

    negatives = mod._negative_action_edge_pair_indices_list(
        edge_pairs,
        candidate_positions=positions,
        candidate_mask=candidate_mask,
        box_dims=box_dims,
        mode="same_source_nn1",
        count=3,
    )

    assert negatives.shape == (1, 1, 3, 2)
    assert negatives[0, 0, :, 0].tolist() == [0, 0, 0]
    assert set(negatives[0, 0, :, 1].tolist()).issubset({2, 3})
    assert 1 not in negatives[0, 0, :, 1].tolist()


def test_dense_legal_edge_pair_negatives_use_global_vacancy_atom_pairs():
    edge_pairs = torch.tensor([[[0, 1]]], dtype=torch.long)
    current_types = torch.tensor([[mod.V_TYPE, mod.CU_TYPE, mod.V_TYPE, mod.CU_TYPE, mod.FE_TYPE]], dtype=torch.long)
    positions = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [5.0, 5.0, 5.0],
                [6.0, 6.0, 6.0],
                [4.0, 4.0, 4.0],
            ]
        ],
        dtype=torch.float32,
    )
    candidate_mask = torch.ones((1, 5), dtype=torch.float32)
    box_dims = torch.tensor([[40.0, 40.0, 40.0]], dtype=torch.float32)

    negatives = mod._dense_legal_action_edge_pair_negative_indices(
        edge_pairs,
        current_types=current_types,
        candidate_positions=positions,
        candidate_mask=candidate_mask,
        box_dims=box_dims,
        count=2,
    )

    assert negatives.shape == (1, 1, 2, 2)
    pairs = {tuple(item) for item in negatives[0, 0].tolist()}
    assert pairs.issubset({(2, 3), (2, 4)})
    assert (0, 1) not in pairs


def test_dense_terminal_vacancy_pair_negatives_allow_nonlocal_destinations():
    edge_pairs = torch.tensor([[[0, 3]]], dtype=torch.long)
    current_types = torch.tensor([[mod.V_TYPE, mod.CU_TYPE, mod.V_TYPE, mod.CU_TYPE, mod.FE_TYPE]], dtype=torch.long)
    candidate_mask = torch.ones((1, 5), dtype=torch.float32)

    negatives = mod._dense_terminal_vacancy_pair_negative_indices(
        edge_pairs,
        current_types=current_types,
        candidate_mask=candidate_mask,
        count=3,
    )

    assert negatives.shape == (1, 1, 3, 2)
    pairs = {tuple(item) for item in negatives[0, 0].tolist()}
    assert pairs.issubset({(0, 1), (0, 4), (2, 1), (2, 3), (2, 4)})
    assert (0, 3) not in pairs


def test_structured_terminal_vacancy_pair_negatives_cover_source_destination_and_unpaired_cases():
    vacancy_pairs = torch.tensor([[[0, 1], [3, 4]]], dtype=torch.long)
    current_types = torch.tensor(
        [[mod.V_TYPE, mod.CU_TYPE, mod.FE_TYPE, mod.V_TYPE, mod.CU_TYPE, mod.FE_TYPE]],
        dtype=torch.long,
    )
    candidate_mask = torch.ones((1, 6), dtype=torch.float32)

    negatives = mod._structured_terminal_vacancy_pair_negative_indices(
        vacancy_pairs,
        current_types=current_types,
        candidate_mask=candidate_mask,
        count_per_group=1,
    )

    assert negatives.shape == (1, 2, 3, 2)
    positive_pairs = {(0, 1), (3, 4)}
    for pair in negatives.reshape(-1, 2).tolist():
        assert tuple(pair) not in positive_pairs

    first = [tuple(pair) for pair in negatives[0, 0].tolist()]
    assert first[0][0] == 0 and first[0][1] != 1
    assert first[1][1] == 1 and first[1][0] != 0
    assert first[2] in {(0, 4), (3, 1)}


def test_pair_listwise_contrastive_loss_prefers_true_pair_over_negatives():
    pair_mask = torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32)
    good_pos = torch.tensor([[4.0, 3.0, 0.0]], dtype=torch.float32)
    good_neg = torch.tensor(
        [
            [
                [0.0, -1.0, -2.0],
                [0.5, 0.0, -1.0],
                [9.0, 9.0, 9.0],
            ]
        ],
        dtype=torch.float32,
    )
    bad_pos = torch.tensor([[0.0, -1.0, 0.0]], dtype=torch.float32)
    bad_neg = torch.tensor(
        [
            [
                [3.0, 2.0, 1.0],
                [2.5, 1.0, 0.0],
                [9.0, 9.0, 9.0],
            ]
        ],
        dtype=torch.float32,
    )

    good = mod._pair_listwise_contrastive_loss(good_pos, good_neg, pair_mask)
    bad = mod._pair_listwise_contrastive_loss(bad_pos, bad_neg, pair_mask)
    empty = mod._pair_listwise_contrastive_loss(good_pos, good_neg, torch.zeros_like(pair_mask))

    assert good["acc"].item() == 1.0
    assert bad["acc"].item() == 0.0
    assert good["loss"].item() < bad["loss"].item()
    assert empty["loss"].item() == 0.0


def test_action_edge_pair_target_tensors_can_select_terminal_vacancy_pairs():
    tensors = {
        "candidate_mask": torch.ones((1, 3), dtype=torch.float32),
        "teacher_action_edge_pair_indices": torch.tensor([[[0, 1], [-1, -1], [-1, -1]]]),
        "teacher_action_edge_pair_mask": torch.tensor([[1.0, 0.0, 0.0]]),
        "teacher_action_edge_pair_support_mask": torch.tensor([[1.0, 0.0, 0.0]]),
        "teacher_action_edge_pair_moving_type": torch.tensor([[mod.CU_TYPE, -1, -1]]),
        "teacher_action_edge_pair_order": torch.tensor([[0.0, 0.0, 0.0]]),
        "teacher_vacancy_pair_indices": torch.tensor([[[0, 2], [-1, -1], [-1, -1]]]),
        "teacher_vacancy_pair_mask": torch.tensor([[1.0, 0.0, 0.0]]),
        "teacher_vacancy_pair_moving_type": torch.tensor([[mod.FE_TYPE, -1, -1]]),
        "teacher_vacancy_pair_order": torch.tensor([[1.0, 0.0, 0.0]]),
    }

    selected = mod._action_edge_pair_target_tensors(tensors, "vacancy_pair")

    assert torch.equal(selected["indices"], tensors["teacher_vacancy_pair_indices"])
    assert torch.equal(selected["support_mask"], tensors["teacher_vacancy_pair_mask"])
    assert int(selected["moving_type"][0, 0].item()) == mod.FE_TYPE


def test_action_edge_pair_losses_accept_listwise_negatives():
    pair_logits = torch.tensor([[2.0, 0.0]], dtype=torch.float32)
    negative_logits = torch.tensor([[[0.0, -1.0], [1.0, 2.0]]], dtype=torch.float32)
    edge_mask = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
    support_mask = torch.tensor([[1.0, 0.0]], dtype=torch.float32)

    edge_terms = mod._action_edge_pair_supervision_loss(
        pair_logits,
        negative_logits,
        edge_mask,
        negative_weight=1.0,
        rank_margin_weight=1.0,
        margin=0.5,
    )
    support_terms = mod._action_edge_pair_support_loss(
        pair_logits,
        negative_logits,
        edge_mask,
        support_mask,
        negative_weight=1.0,
        rank_margin_weight=1.0,
        margin=0.5,
    )

    assert edge_terms["loss"].item() > 0.0
    assert edge_terms["rank_acc"].item() == pytest.approx(0.5)
    assert support_terms["loss"].item() > 0.0
    assert support_terms["rank_acc"].item() == pytest.approx(1.0)


def test_edge_pair_multiobjective_completion_source_maps_to_multiobjective():
    assert (
        long_eval_mod._edge_pair_completion_score_source("action_edge_pair_multiobjective_completion")
        == "multiobjective"
    )


def test_vacancy_atom_pair_filter_requires_kmc_source_and_destination_types():
    current_types = torch.tensor([[mod.V_TYPE, mod.CU_TYPE, mod.FE_TYPE, mod.V_TYPE]], dtype=torch.long)

    assert long_eval_mod._is_valid_vacancy_atom_pair(
        current_types,
        0,
        0,
        1,
        require_vacancy_atom_pair=True,
    )
    assert long_eval_mod._is_valid_vacancy_atom_pair(
        current_types,
        0,
        0,
        2,
        require_vacancy_atom_pair=True,
    )
    assert not long_eval_mod._is_valid_vacancy_atom_pair(
        current_types,
        0,
        1,
        0,
        require_vacancy_atom_pair=True,
    )
    assert not long_eval_mod._is_valid_vacancy_atom_pair(
        current_types,
        0,
        0,
        3,
        require_vacancy_atom_pair=True,
    )
    assert long_eval_mod._is_valid_vacancy_atom_pair(
        current_types,
        0,
        1,
        3,
        require_vacancy_atom_pair=False,
    )


def test_planner_selected_dataset_signature_records_mode_and_counts():
    args = SimpleNamespace(
        seed=3,
        lattice_size=[10, 10, 10],
        cu_density=0.01,
        v_density=0.001,
        segment_k=4,
        segment_ks=[2, 4, 8],
        max_seed_vacancies=16,
        max_candidate_sites=256,
        max_episode_steps=200,
        max_vacancies=8,
        max_defects=16,
        max_shells=4,
        neighbor_order="2NN",
        reward_scale=10.0,
        temperature=300.0,
        stats_dim=10,
        train_segments=999,
        val_segments=999,
        train_segments_per_k=11,
        val_segments_per_k=5,
        planner_selected_from="results/source/final_model.pt",
        planner_selected_min_projected_changed_sites=2,
        planner_selected_duration_source="model",
        planner_selected_tau_source="baseline",
        planner_selected_score_mode="energy_per_sqrt_tau",
        planner_selected_reward_prediction_source="projected",
        planner_selected_tau_residual_penalty=0.25,
        planner_selected_k_penalty_power=0.5,
        planner_selected_allow_uncovered_reward_only=True,
        teacher_path_summary_mode="stepwise",
        teacher_mode="kmc",
        natural_teacher_backend="kmc",
        segment_boundary_mode="adaptive_key_event",
        adaptive_min_k=2,
        adaptive_candidate_horizon_source="actual",
        adaptive_key_moving_types=[1],
        adaptive_min_touched_sites=4,
        adaptive_abs_delta_e_threshold=0.1,
        adaptive_cumulative_abs_delta_e_threshold=0.3,
        neural_teacher_temperature=1.0,
        neural_teacher_epsilon=0.0,
        max_segments_per_rollout=50,
        teacher_candidate_neighbor_depth=1,
        disable_teacher_candidate_augmentation=True,
    )

    signature = mod._dataset_signature(args)

    assert signature["dataset_mode"] == "planner_selected"
    assert signature["train_segments"] == 33
    assert signature["val_segments"] == 15
    assert signature["planner_selected_from"] == "results/source/final_model.pt"
    assert signature["planner_selected_checkpoint"] == {
        "path": "results/source/final_model.pt",
        "exists": False,
    }
    assert signature["planner_selected_score_mode"] == "energy_per_sqrt_tau"
    assert signature["planner_selected_tau_source"] == "baseline"
    assert signature["planner_selected_reward_prediction_source"] == "projected"
    assert signature["planner_selected_allow_uncovered_reward_only"] is True
    assert signature["natural_teacher_backend"] == "kmc"
    assert signature["segment_boundary_mode"] == "adaptive_key_event"
    assert signature["adaptive_min_k"] == 2
    assert signature["adaptive_candidate_horizon_source"] == "actual"
    assert signature["adaptive_key_moving_types"] == [1]


def test_train_planner_candidate_selection_matches_long_eval_semantics():
    candidates = [
        {"segment_k": 2, "selection_score": 1.0, "reachability_violation": 0.0, "projected_changed_count": 2.0},
        {"segment_k": 4, "selection_score": 5.0, "reachability_violation": 1.0, "projected_changed_count": 4.0},
        {"segment_k": 8, "selection_score": 2.0, "reachability_violation": 0.0, "projected_changed_count": 1.0},
    ]

    selected = mod._choose_planner_candidate(candidates, min_projected_changed_sites=2)

    assert selected["segment_k"] == 2


def test_planner_selected_reward_only_keeps_uncovered_segments(monkeypatch):
    class FakeInner:
        dims = np.array([10, 10, 10], dtype=np.int32)

        def __init__(self):
            self.after = False

        def get_vacancy_array(self):
            if self.after:
                return np.array([[2, 0, 0]], dtype=np.int32)
            return np.array([[0, 0, 0]], dtype=np.int32)

        def get_cu_array(self):
            return np.array([[1, 0, 0]], dtype=np.int32)

    class FakeEnv:
        cfg = {"reward_scale": 10.0}

        def __init__(self):
            self.env = FakeInner()

        def reset(self):
            self.env.after = False
            return np.zeros((2, 2), dtype=np.float32)

        def step(self, _action):
            self.env.after = True
            info = {
                "expected_delta_t": 0.25,
                "delta_t": 0.3,
                "total_rate": 4.0,
                "delta_E": 0.2,
                "dir_idx": 0,
                "moving_type": 0,
                "old_pos": np.array([1, 0, 0], dtype=np.int32),
                "new_pos": np.array([0, 0, 0], dtype=np.int32),
                "vac_idx": 0,
            }
            return np.ones((2, 2), dtype=np.float32), 2.0, False, info

    monkeypatch.setattr(
        mod,
        "_predict_planner_candidate_for_horizon",
        lambda **_: {
            "segment_k": 8,
            "selection_score": 1.0,
            "reachability_violation": 0.0,
            "projected_changed_count": 2.0,
        },
    )
    monkeypatch.setattr(
        mod,
        "_build_candidate_positions",
        lambda *_args, **_kwargs: (
            [(0, 0, 0), (1, 0, 0)],
            {(0, 0, 0): 0, (1, 0, 0): 1},
            np.array([[0, 0, 0]], dtype=np.int32),
        ),
    )
    monkeypatch.setattr(mod, "_global_summary", lambda _env: np.zeros((18,), dtype=np.float32))
    monkeypatch.setattr(mod, "_sample_teacher_action", lambda _env, _rng: object())

    samples, stats = mod._collect_planner_selected_segments(
        env=FakeEnv(),
        num_segments=1,
        segment_ks=[2, 4, 8],
        planner_model=torch.nn.Identity(),
        planner_device="cpu",
        max_seed_vacancies=1,
        max_candidate_sites=4,
        rng=np.random.default_rng(0),
        include_stepwise_path_summary=True,
        summary_horizon_k=8,
        max_segments_per_rollout=0,
        min_projected_changed_sites=2,
        duration_source="model",
        planner_tau_source="model",
        planner_score_mode="energy_per_tau",
        planner_tau_residual_penalty=0.0,
        planner_k_penalty_power=0.0,
        reward_prediction_source="projected",
        allow_uncovered_reward_only=True,
        teacher_candidate_augmentation=False,
    )

    assert len(samples) == 1
    assert stats["reward_only_uncovered"] == 1
    assert stats["skipped_uncovered"] == 0
    assert stats["selected_attempt_k_histogram"]["8"] == 1
    assert stats["chosen_k_histogram"]["8"] == 1


def test_planner_selected_collector_augments_candidates_with_teacher_path(monkeypatch):
    class FakeInner:
        dims = np.array([10, 10, 10], dtype=np.int32)
        NN1 = np.zeros((1, 3), dtype=np.int32)

        def __init__(self):
            self.after = False

        def get_vacancy_array(self):
            if self.after:
                return np.array([[2, 0, 0]], dtype=np.int32)
            return np.array([[0, 0, 0]], dtype=np.int32)

        def get_cu_array(self):
            return np.array([[1, 0, 0]], dtype=np.int32)

    class FakeEnv:
        cfg = {"reward_scale": 10.0}

        def __init__(self):
            self.env = FakeInner()

        def reset(self):
            self.env.after = False
            return np.zeros((2, 2), dtype=np.float32)

        def step(self, _action):
            self.env.after = True
            info = {
                "expected_delta_t": 0.25,
                "delta_t": 0.3,
                "total_rate": 4.0,
                "delta_E": 0.2,
                "dir_idx": 0,
                "moving_type": 0,
                "old_pos": np.array([1, 0, 0], dtype=np.int32),
                "new_pos": np.array([0, 0, 0], dtype=np.int32),
                "vac_idx": 0,
            }
            return np.ones((2, 2), dtype=np.float32), 2.0, False, info

    augment_calls = []
    monkeypatch.setattr(
        mod,
        "_predict_planner_candidate_for_horizon",
        lambda **_: {
            "segment_k": 8,
            "selection_score": 1.0,
            "reachability_violation": 0.0,
            "projected_changed_count": 2.0,
        },
    )
    monkeypatch.setattr(
        mod,
        "_build_candidate_positions",
        lambda *_args, **_kwargs: (
            [(0, 0, 0), (1, 0, 0)],
            {(0, 0, 0): 0, (1, 0, 0): 1},
            np.array([[0, 0, 0]], dtype=np.int32),
        ),
    )

    def fake_augment(**kwargs):
        augment_calls.append(set(kwargs["touched_positions"]))
        return (
            [(0, 0, 0), (1, 0, 0), (2, 0, 0)],
            {(0, 0, 0): 0, (1, 0, 0): 1, (2, 0, 0): 1},
            kwargs["seeds"],
        )

    monkeypatch.setattr(mod, "_augment_candidate_positions_with_teacher_path", fake_augment)
    monkeypatch.setattr(mod, "_global_summary", lambda _env: np.zeros((18,), dtype=np.float32))
    monkeypatch.setattr(mod, "_sample_teacher_action", lambda _env, _rng: object())

    samples, stats = mod._collect_planner_selected_segments(
        env=FakeEnv(),
        num_segments=1,
        segment_ks=[2, 4, 8],
        planner_model=torch.nn.Identity(),
        planner_device="cpu",
        max_seed_vacancies=1,
        max_candidate_sites=4,
        rng=np.random.default_rng(0),
        include_stepwise_path_summary=True,
        summary_horizon_k=8,
        max_segments_per_rollout=0,
        min_projected_changed_sites=2,
        duration_source="model",
        planner_tau_source="model",
        planner_score_mode="energy_per_tau",
        planner_tau_residual_penalty=0.0,
        planner_k_penalty_power=0.0,
        reward_prediction_source="projected",
        allow_uncovered_reward_only=False,
        teacher_candidate_augmentation=True,
    )

    assert len(samples) == 1
    assert stats["skipped_uncovered"] == 0
    assert stats["reward_only_uncovered"] == 0
    assert augment_calls
    assert (1, 0, 0) in augment_calls[0]
    assert (0, 0, 0) in augment_calls[0]


def test_validate_resume_args_rejects_segment_k_mismatch():
    with pytest.raises(ValueError, match="segment_k"):
        _validate_resume_args(SimpleNamespace(segment_k=4), {"segment_k": 2})


def test_validate_resume_args_rejects_path_summary_mode_mismatch():
    with pytest.raises(ValueError, match="teacher_path_summary_mode"):
        _validate_resume_args(
            SimpleNamespace(segment_k=4, teacher_path_summary_mode="stepwise"),
            {"segment_k": 4, "teacher_path_summary_mode": "legacy"},
        )


def test_initialize_best_score_from_saved_best_uses_best_checkpoint(tmp_path, monkeypatch):
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(0.9)

    best_model = torch.nn.Linear(1, 1, bias=False)
    best_model.weight.data.fill_(0.1)
    torch.save({"model": best_model.state_dict()}, tmp_path / "best_model.pt")

    def fake_evaluate(eval_model, loader, device, max_changed_sites, **_kwargs):
        tau_log_mae = float(eval_model.weight.detach().cpu().item())
        return {
            "tau_log_mae": tau_log_mae,
            "reward_mae": 0.0,
            "change_topk_f1": 1.0,
            "projected_change_f1": 1.0,
            "projected_changed_type_acc": 1.0,
            "reachability_violation_rate": 0.0,
        }

    monkeypatch.setattr(mod, "_evaluate", fake_evaluate)
    score, source = _initialize_best_score_from_saved_best(
        model=model,
        loader=None,
        device="cpu",
        max_changed_sites=4,
        dataset_stats={"val": {"coverage": 1.0}},
        save_dir=tmp_path,
    )

    assert math.isclose(score, 0.1, rel_tol=1e-6)
    assert source == "saved best model"
    assert math.isclose(float(model.weight.detach().cpu().item()), 0.9, rel_tol=1e-6)


def test_initialize_best_score_from_checkpoint_fallback_uses_stored_best_score(tmp_path, monkeypatch):
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(0.9)

    def fake_evaluate(eval_model, loader, device, max_changed_sites, **_kwargs):
        tau_log_mae = float(eval_model.weight.detach().cpu().item())
        return {
            "tau_log_mae": tau_log_mae,
            "reward_mae": 0.0,
            "change_topk_f1": 1.0,
            "projected_change_f1": 1.0,
            "projected_changed_type_acc": 1.0,
            "reachability_violation_rate": 0.0,
        }

    monkeypatch.setattr(mod, "_evaluate", fake_evaluate)
    score, source = _initialize_best_score_from_saved_best(
        model=model,
        loader=None,
        device="cpu",
        max_changed_sites=4,
        dataset_stats={"val": {"coverage": 1.0}},
        save_dir=tmp_path,
        checkpoint_best_score=0.2,
    )

    assert math.isclose(score, 0.2, rel_tol=1e-6)
    assert source == "resume checkpoint + stored best_score"
    assert math.isclose(float(model.weight.detach().cpu().item()), 0.9, rel_tol=1e-6)


def test_initialize_best_score_from_checkpoint_skips_stored_best_score_for_new_save_dir(tmp_path, monkeypatch):
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(0.9)

    def fake_evaluate(eval_model, loader, device, max_changed_sites, **_kwargs):
        tau_log_mae = float(eval_model.weight.detach().cpu().item())
        return {
            "tau_log_mae": tau_log_mae,
            "reward_mae": 0.0,
            "change_topk_f1": 1.0,
            "projected_change_f1": 1.0,
            "projected_changed_type_acc": 1.0,
            "reachability_violation_rate": 0.0,
        }

    monkeypatch.setattr(mod, "_evaluate", fake_evaluate)
    score, source = _initialize_best_score_from_saved_best(
        model=model,
        loader=None,
        device="cpu",
        max_changed_sites=4,
        dataset_stats={"val": {"coverage": 1.0}},
        save_dir=tmp_path,
        checkpoint_best_score=0.2,
        allow_checkpoint_best_score_fallback=False,
    )

    assert math.isclose(score, 0.9, rel_tol=1e-6)
    assert source == "resume checkpoint"
    assert math.isclose(float(model.weight.detach().cpu().item()), 0.9, rel_tol=1e-6)


def test_initialize_best_score_skips_incompatible_saved_best_model(tmp_path, monkeypatch):
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(0.9)

    incompatible_best_model = torch.nn.Linear(2, 1, bias=False)
    torch.save({"model": incompatible_best_model.state_dict()}, tmp_path / "best_model.pt")

    def fake_evaluate(eval_model, loader, device, max_changed_sites, **_kwargs):
        tau_log_mae = float(eval_model.weight.detach().cpu().item())
        return {
            "tau_log_mae": tau_log_mae,
            "reward_mae": 0.0,
            "change_topk_f1": 1.0,
            "projected_change_f1": 1.0,
            "projected_changed_type_acc": 1.0,
            "reachability_violation_rate": 0.0,
        }

    monkeypatch.setattr(mod, "_evaluate", fake_evaluate)
    score, source = _initialize_best_score_from_saved_best(
        model=model,
        loader=None,
        device="cpu",
        max_changed_sites=4,
        dataset_stats={"val": {"coverage": 1.0}},
        save_dir=tmp_path,
    )

    assert math.isclose(score, 0.9, rel_tol=1e-6)
    assert "skipped incompatible saved best model" in source


def test_eval_cache_validation_rejects_segment_k_mismatch(tmp_path):
    payload = {
        "train": [],
        "val": [],
        "stats": {},
        "signature": {"dataset_version": 7, "segment_k": 2},
    }
    cache_path = tmp_path / "segments.pt"
    torch.save(payload, cache_path)

    with pytest.raises(ValueError, match="segment_k"):
        eval_mod._load_samples(cache_path, "val", 0, expected_segment_k=4)


def test_selection_score_penalizes_projected_global_l1():
    base_metrics = {
        "tau_log_mae": 2.0,
        "reward_mae": 1.0,
        "change_topk_f1": 0.2,
        "projected_change_f1": 0.1,
        "projected_changed_type_acc": 0.1,
        "projected_global_l1": 0.002,
        "unchanged_vacancy_copy_acc": 0.95,
        "reachability_violation_rate": 0.0,
    }
    worse_metrics = dict(base_metrics)
    worse_metrics["projected_global_l1"] = 0.008

    base_score = _selection_score(base_metrics, {"val": {"coverage": 1.0}})
    worse_score = _selection_score(worse_metrics, {"val": {"coverage": 1.0}})

    assert worse_score > base_score


def test_selection_score_penalizes_unchanged_vacancy_copy_acc():
    base_metrics = {
        "tau_log_mae": 2.0,
        "reward_mae": 1.0,
        "change_topk_f1": 0.2,
        "projected_change_f1": 0.1,
        "projected_changed_type_acc": 0.1,
        "projected_global_l1": 0.002,
        "unchanged_vacancy_copy_acc": 0.95,
        "reachability_violation_rate": 0.0,
    }
    worse_metrics = dict(base_metrics)
    worse_metrics["unchanged_vacancy_copy_acc"] = 0.60

    base_score = _selection_score(base_metrics, {"val": {"coverage": 1.0}})
    worse_score = _selection_score(worse_metrics, {"val": {"coverage": 1.0}})

    assert worse_score > base_score


def test_compute_lognormal_distribution_metrics_reports_calibrated_center_case():
    mu = np.log(np.asarray([1.0, 2.0, 4.0], dtype=np.float64))
    log_sigma = np.log(np.asarray([0.2, 0.2, 0.2], dtype=np.float64))
    target = np.asarray([1.0, 2.0, 4.0], dtype=np.float64)

    metrics = mod._compute_lognormal_distribution_metrics(mu, log_sigma, target)

    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["log_mae"] == pytest.approx(0.0)
    assert metrics["coverage_68"] == pytest.approx(1.0)
    assert metrics["coverage_95"] == pytest.approx(1.0)
    assert metrics["pit_mean"] == pytest.approx(0.5)
    assert metrics["pit_ks"] == pytest.approx(0.5, rel=1e-6)


def test_teacher_path_summary_pads_multik_stepwise_features():
    path_infos = [
        {
            "dir_idx": 0,
            "moving_type": 1,
            "total_rate": 10.0,
            "expected_delta_t": 0.1,
            "delta_E": 0.5,
            "old_pos": np.asarray([0, 0, 0], dtype=np.int32),
            "new_pos": np.asarray([1, 1, 1], dtype=np.int32),
            "vac_idx": 0,
        },
        {
            "dir_idx": 3,
            "moving_type": 0,
            "total_rate": 4.0,
            "expected_delta_t": 0.25,
            "delta_E": -0.2,
            "old_pos": np.asarray([1, 1, 1], dtype=np.int32),
            "new_pos": np.asarray([2, 2, 2], dtype=np.int32),
            "vac_idx": 1,
        },
    ]

    summary = mod._teacher_path_summary(path_infos, max_candidate_sites=32, horizon_k=2, summary_horizon_k=8)

    assert summary.shape == (teacher_path_summary_dim(8),)
    assert summary[17] == pytest.approx(1.0)
    assert summary[18] == pytest.approx(math.log(0.1 + 1e-12))
    assert summary[19] == pytest.approx(math.log(0.25 + 1e-12))
    assert np.allclose(summary[20:26], -27.0)
    assert summary[26] == pytest.approx(0.5)
    assert summary[27] == pytest.approx(-0.2)
    assert np.allclose(summary[28:34], 0.0)


def test_eval_cache_validation_accepts_mixed_k_signature(tmp_path):
    payload = {
        "train": [],
        "val": [],
        "stats": {},
        "signature": {"dataset_version": 9, "segment_ks": [2, 4, 8], "summary_horizon_k": 8},
    }
    cache_path = tmp_path / "segments.pt"
    torch.save(payload, cache_path)

    samples, _stats, signature = eval_mod._load_samples(
        cache_path,
        "val",
        0,
        expected_segment_ks=[8, 2, 4],
        expected_summary_horizon_k=8,
    )

    assert samples == []
    assert signature["segment_ks"] == [2, 4, 8]
    with pytest.raises(ValueError, match="segment_ks"):
        eval_mod._load_samples(cache_path, "val", 0, expected_segment_ks=[4])
    with pytest.raises(ValueError, match="summary_horizon_k"):
        eval_mod._load_samples(
            cache_path,
            "val",
            0,
            expected_segment_ks=[2, 4, 8],
            expected_summary_horizon_k=4,
        )


def test_eval_cache_validation_accepts_adaptive_realized_horizon(tmp_path):
    sample = mod.MacroSegmentSample(
        start_obs=np.zeros((4,), dtype=np.float32),
        next_obs=np.zeros((4,), dtype=np.float32),
        start_vacancy_positions=np.asarray([[0, 0, 0]], dtype=np.int32),
        start_cu_positions=np.asarray([[1, 0, 0]], dtype=np.int32),
        global_summary=np.zeros((16,), dtype=np.float32),
        teacher_path_summary=np.zeros((teacher_path_summary_dim(8),), dtype=np.float32),
        candidate_positions=np.zeros((4, 3), dtype=np.float32),
        nearest_vacancy_offset=np.zeros((4, 3), dtype=np.float32),
        reach_depth=np.zeros((4,), dtype=np.float32),
        is_start_vacancy=np.zeros((4,), dtype=np.float32),
        current_types=np.zeros((4,), dtype=np.int64),
        target_types=np.zeros((4,), dtype=np.int64),
        candidate_mask=np.ones((4,), dtype=np.float32),
        changed_mask=np.zeros((4,), dtype=np.float32),
        tau_exp=1.0,
        tau_real=1.0,
        reward_sum=0.0,
        horizon_k=1,
        box_dims=np.asarray([10.0, 10.0, 10.0], dtype=np.float32),
    )
    payload = {
        "train": [],
        "val": [sample.__dict__],
        "stats": {},
        "signature": {
            "dataset_version": 12,
            "segment_ks": [4, 8],
            "summary_horizon_k": 8,
            "segment_boundary_mode": "adaptive_key_event",
        },
    }
    cache_path = tmp_path / "adaptive_segments.pt"
    torch.save(payload, cache_path)

    samples, _stats, signature = eval_mod._load_samples(
        cache_path,
        "val",
        0,
        expected_segment_ks=[4, 8],
        expected_summary_horizon_k=8,
    )

    assert len(samples) == 1
    assert samples[0].horizon_k == 1
    assert signature["segment_boundary_mode"] == "adaptive_key_event"

    payload["val"][0] = dict(payload["val"][0])
    payload["val"][0]["horizon_k"] = 9
    torch.save(payload, cache_path)
    with pytest.raises(ValueError, match="adaptive sample"):
        eval_mod._load_samples(
            cache_path,
            "val",
            0,
            expected_segment_ks=[4, 8],
            expected_summary_horizon_k=8,
        )


def test_init_from_resizes_path_posterior_for_multik_summary():
    old_model = MacroDreamerEditModel(
        max_vacancies=4,
        max_defects=8,
        max_shells=4,
        stats_dim=10,
        lattice_size=(10, 10, 10),
        neighbor_order="2NN",
        dim_latent=4,
        graph_hidden_size=8,
        patch_hidden_size=16,
        patch_latent_dim=8,
        path_latent_dim=4,
        global_summary_dim=16,
        teacher_path_summary_dim=teacher_path_summary_dim(4),
        max_macro_k=8,
    )
    new_model = MacroDreamerEditModel(
        max_vacancies=4,
        max_defects=8,
        max_shells=4,
        stats_dim=10,
        lattice_size=(10, 10, 10),
        neighbor_order="2NN",
        dim_latent=4,
        graph_hidden_size=8,
        patch_hidden_size=16,
        patch_latent_dim=8,
        path_latent_dim=4,
        global_summary_dim=16,
        teacher_path_summary_dim=teacher_path_summary_dim(8),
        max_macro_k=8,
    )
    old_state = old_model.state_dict()

    _missing, _unexpected, resized, skipped = _load_model_weights(
        new_model,
        old_state,
        allow_path_posterior_resize=True,
    )

    assert any(key.startswith("path_posterior.") for key in resized)
    assert not skipped
    loaded = new_model.state_dict()["path_posterior.net.1.weight"]
    old = old_state["path_posterior.net.1.weight"]
    assert torch.allclose(loaded[: old.shape[0], : old.shape[1]], old)
    assert torch.allclose(loaded[:, old.shape[1] :], torch.zeros_like(loaded[:, old.shape[1] :]))


def test_init_from_resizes_k_embedding_for_larger_horizon_set():
    old_model = MacroDreamerEditModel(
        max_vacancies=4,
        max_defects=8,
        max_shells=4,
        stats_dim=10,
        lattice_size=(10, 10, 10),
        neighbor_order="2NN",
        dim_latent=4,
        graph_hidden_size=8,
        patch_hidden_size=16,
        patch_latent_dim=8,
        path_latent_dim=4,
        global_summary_dim=16,
        teacher_path_summary_dim=teacher_path_summary_dim(8),
        max_macro_k=8,
    )
    new_model = MacroDreamerEditModel(
        max_vacancies=4,
        max_defects=8,
        max_shells=4,
        stats_dim=10,
        lattice_size=(10, 10, 10),
        neighbor_order="2NN",
        dim_latent=4,
        graph_hidden_size=8,
        patch_hidden_size=16,
        patch_latent_dim=8,
        path_latent_dim=4,
        global_summary_dim=16,
        teacher_path_summary_dim=teacher_path_summary_dim(1024),
        max_macro_k=1024,
    )
    old_state = old_model.state_dict()

    _missing, _unexpected, resized, skipped = _load_model_weights(
        new_model,
        old_state,
        allow_path_posterior_resize=True,
    )

    assert "k_embed.weight" in resized
    assert not skipped
    loaded = new_model.state_dict()["k_embed.weight"]
    old = old_state["k_embed.weight"]
    assert torch.allclose(loaded[: old.shape[0]], old)
    assert torch.allclose(loaded[old.shape[0] :], torch.zeros_like(loaded[old.shape[0] :]))


def test_large_horizon_candidate_build_stops_at_candidate_cap():
    class FakeInner:
        dims = np.array([64, 64, 64], dtype=np.int32)
        NN1 = np.array(
            [
                [1, 0, 0],
                [-1, 0, 0],
                [0, 1, 0],
                [0, -1, 0],
                [0, 0, 1],
                [0, 0, -1],
            ],
            dtype=np.int32,
        )
        diffusion_rates = [[1.0 for _ in range(6)]]

        def _ensure_diffusion_rates(self):
            return None

        def get_vacancy_array(self):
            return np.array([[32, 32, 32]], dtype=np.int32)

    class FakeEnv:
        env = FakeInner()

    positions, depth_map, _seeds = mod._build_candidate_positions(
        FakeEnv(),
        horizon_k=1024,
        max_seed_vacancies=1,
        max_candidate_sites=32,
    )

    assert len(positions) == 32
    assert len(depth_map) >= 32
    assert max(depth_map[pos] for pos in positions) < 1024


def test_projection_uses_per_sample_max_changed_sites_cap():
    current_types = torch.tensor([[2, 0, 2, 0], [2, 0, 2, 0]], dtype=torch.long)
    change_logits = torch.full((2, 4), 8.0, dtype=torch.float32)
    type_logits = torch.full((2, 4, 3), -6.0, dtype=torch.float32)
    type_logits[:, 0, 0] = 6.0
    type_logits[:, 1, 2] = 6.0
    type_logits[:, 2, 0] = 6.0
    type_logits[:, 3, 2] = 6.0
    node_mask = torch.ones((2, 4), dtype=torch.float32)
    positions = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [4.0, 0.0, 0.0], [5.0, 1.0, 1.0]],
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [4.0, 0.0, 0.0], [5.0, 1.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    box_dims = torch.tensor([[20.0, 20.0, 20.0], [20.0, 20.0, 20.0]], dtype=torch.float32)

    _types, changed_mask, _cost, violations = project_types_by_inventory(
        current_types=current_types,
        change_logits=change_logits,
        type_logits=type_logits,
        node_mask=node_mask,
        positions=positions,
        box_dims=box_dims,
        horizon_k=torch.tensor([4, 4], dtype=torch.long),
        max_changed_sites=torch.tensor([2, 4], dtype=torch.long),
    )

    assert changed_mask[0].sum().item() == pytest.approx(2.0)
    assert changed_mask[1].sum().item() == pytest.approx(4.0)
    assert torch.all(violations == 0)


def test_long_eval_planner_chooses_best_legal_energy_per_time_candidate():
    candidates = [
        {"segment_k": 2, "selection_score": 1.0, "reachability_violation": 0.0, "projected_changed_count": 2.0},
        {"segment_k": 4, "selection_score": 5.0, "reachability_violation": 1.0, "projected_changed_count": 4.0},
        {"segment_k": 8, "selection_score": 2.0, "reachability_violation": 0.0, "projected_changed_count": 2.0},
    ]

    selected = long_eval_mod._choose_planner_candidate(candidates)

    assert selected["segment_k"] == 8


def test_long_eval_planner_rejects_all_illegal_candidates():
    candidates = [
        {"segment_k": 2, "selection_score": 1.0, "reachability_violation": 1.0, "projected_changed_count": 2.0},
        {"segment_k": 4, "selection_score": 2.0, "reachability_violation": 1.0, "projected_changed_count": 2.0},
    ]

    assert long_eval_mod._choose_planner_candidate(candidates) is None


def test_long_eval_planner_rejects_projected_noop_candidates():
    candidates = [
        {"segment_k": 2, "selection_score": 4.0, "reachability_violation": 0.0, "projected_changed_count": 0.0},
        {"segment_k": 4, "selection_score": 3.0, "reachability_violation": 0.0, "projected_changed_count": 1.0},
    ]

    assert long_eval_mod._choose_planner_candidate(candidates, min_projected_changed_sites=2) is None


def test_candidate_joint_diagnostic_summary_tracks_selected_and_oracle_bounds():
    segments = [
        {
            "index": 0,
            "planner_candidates": [
                {
                    "candidate_joint_diagnostic": {
                        "segment_k": 8,
                        "selected_by_planner": True,
                        "pre_oracle_selection_score": 3.0,
                        "site_f1": 0.2,
                        "vacancy_pair_f1": 0.1,
                        "teacher_reward_sum": 1.0,
                        "teacher_tau_exp": 2.0,
                        "projected_changed_count": 100.0,
                        "teacher_changed_count": 20.0,
                        "vacancy_pair_selected_count": 50.0,
                        "vacancy_pair_teacher_count": 10.0,
                    },
                    "vacancy_pair_projection_diagnostic": {
                        "candidate_pair_count": 4,
                        "selected_pair_count": 2,
                        "selected_pairs": [{"source_position": [0, 0, 0], "destination_position": [1, 0, 0]}],
                        "factorized_pair_score_count": 4,
                        "factorized_pair_scores": [
                            {
                                "rank": 1,
                                "source_position": [0, 0, 0],
                                "destination_position": [1, 0, 0],
                                "score": 1.0,
                                "source_score": 0.4,
                                "destination_score": 0.3,
                            }
                        ],
                    },
                },
                {
                    "candidate_joint_diagnostic": {
                        "segment_k": 32,
                        "selected_by_planner": False,
                        "pre_oracle_selection_score": 1.0,
                        "site_f1": 0.6,
                        "vacancy_pair_f1": 0.7,
                        "teacher_reward_sum": 0.5,
                        "teacher_tau_exp": 1.0,
                        "projected_changed_count": 18.0,
                        "teacher_changed_count": 20.0,
                        "vacancy_pair_selected_count": 9.0,
                        "vacancy_pair_teacher_count": 10.0,
                    }
                },
            ],
        }
    ]

    summary = long_eval_mod._candidate_joint_diagnostic_summary(segments)

    assert summary["candidate_count"] == 2
    assert summary["selected_by_planner"]["avg_site_f1"] == pytest.approx(0.2)
    assert summary["selector_upper_bounds"]["site_f1"]["avg_site_f1"] == pytest.approx(0.6)
    assert summary["selector_upper_bounds"]["pre_oracle_selection_score"]["avg_site_f1"] == pytest.approx(0.2)
    compact = long_eval_mod._planner_candidates_for_output(
        segments[0]["planner_candidates"],
        compact=True,
    )
    assert compact[0]["candidate_joint_diagnostic"]["selected_by_planner"] is True
    assert "selected_pairs" not in compact[0].get("vacancy_pair_projection_diagnostic", {})
    assert compact[0]["vacancy_pair_projection_diagnostic"]["factorized_pair_score_count"] == 4
    assert compact[0]["vacancy_pair_projection_diagnostic"]["factorized_pair_scores"][0]["source_score"] == 0.4


def test_vacancy_pair_rank_summary_flattens_into_candidate_joint_diagnostics():
    selected = {
        "segment_k": 8,
        "selection_score": 1.0,
        "vacancy_pair_projection_diagnostic": {
            "ranked_pair_score_count": 4,
            "selected_pair_count": 2,
            "ranked_pair_scores": [
                {
                    "rank": 1,
                    "source_position": [0, 0, 0],
                    "destination_position": [1, 0, 0],
                    "moving_type": 2,
                    "score": 0.9,
                },
                {
                    "rank": 2,
                    "source_position": [2, 0, 0],
                    "destination_position": [3, 0, 0],
                    "moving_type": 1,
                    "score": 0.8,
                },
                {
                    "rank": 3,
                    "source_position": [0, 0, 0],
                    "destination_position": [3, 0, 0],
                    "moving_type": 2,
                    "score": 0.7,
                },
            ],
            "factorized_pair_score_count": 3,
            "factorized_pair_scores": [
                {
                    "rank": 1,
                    "source_position": [0, 0, 0],
                    "destination_position": [1, 0, 0],
                    "moving_type": 2,
                    "score": 0.9,
                    "source_score": 0.4,
                    "destination_score": 0.2,
                },
                {
                    "rank": 3,
                    "source_position": [0, 0, 0],
                    "destination_position": [3, 0, 0],
                    "moving_type": 2,
                    "score": 0.7,
                    "source_score": 0.4,
                    "destination_score": 0.1,
                },
            ],
        },
    }
    teacher_segment = {
        "reward_sum": 1.0,
        "tau_exp": 2.0,
        "vacancy_pair_positions": [
            {
                "source": [0, 0, 0],
                "destination": [1, 0, 0],
                "moving_type": 2,
            },
            {
                "source": [2, 0, 0],
                "destination": [3, 0, 0],
                "moving_type": 1,
            },
        ],
    }

    rank_summary = long_eval_mod._vacancy_pair_rank_summary(selected, teacher_segment)
    record = long_eval_mod._candidate_joint_diagnostic_record(
        selected,
        {"f1": 0.25, "projected_changed_count": 2.0, "teacher_changed_count": 4.0},
        {"f1": 0.5, "selected_pair_count": 2.0, "teacher_pair_count": 2.0},
        teacher_segment,
        rank_summary,
    )
    grouped = long_eval_mod._candidate_joint_group_summary([record])

    assert rank_summary["teacher_pair_found_recall"] == pytest.approx(1.0)
    assert rank_summary["teacher_pair_rank_mean"] == pytest.approx(1.5)
    assert rank_summary["teacher_pair_mrr"] == pytest.approx(0.75)
    assert rank_summary["teacher_pair_typed_rank_accuracy"] == pytest.approx(1.0)
    assert rank_summary["teacher_pair_recall_at_rank"]["8"] == pytest.approx(1.0)
    assert rank_summary["topk_false_positive_rate"]["8"] == pytest.approx(1.0 / 3.0)
    assert rank_summary["topk_true_pair_count"]["8"] == 2
    assert rank_summary["topk_source_hard_negative_count"]["8"] == 1
    assert rank_summary["topk_destination_hard_negative_count"]["8"] == 1
    assert rank_summary["topk_source_destination_unpaired_count"]["8"] == 1
    assert rank_summary["topk_true_score_mean"]["8"] == pytest.approx(0.85)
    assert rank_summary["topk_false_score_mean"]["8"] == pytest.approx(0.7)
    assert record["vacancy_pair_rank_mean"] == pytest.approx(1.5)
    assert record["vacancy_pair_rank_recall_at_8"] == pytest.approx(1.0)
    assert grouped["avg_vacancy_pair_rank_percentile_mean"] == pytest.approx(0.375)

    selected["teacher_overlap_oracle"] = {"vacancy_pair_rank": rank_summary}
    long_eval_mod._annotate_factorized_vacancy_pairs(selected, teacher_segment)
    factorized = selected["vacancy_pair_projection_diagnostic"]["factorized_pair_scores"]
    assert factorized[0]["pair_label"] == "true_pair"
    assert factorized[1]["pair_label"] == "source_destination_unpaired"
    compact = long_eval_mod._planner_candidates_for_output([selected], compact=True)
    compact_rank = compact[0]["teacher_overlap_oracle"]["vacancy_pair_rank"]
    assert compact_rank["teacher_pair_mrr"] == pytest.approx(0.75)
    assert compact_rank["topk_source_destination_unpaired_count"]["8"] == 1
    compact_factorized = compact[0]["vacancy_pair_projection_diagnostic"]["factorized_pair_scores"]
    assert compact_factorized[0]["pair_label"] == "true_pair"


def _dummy_v125_pareto_spec(weight_idx: int = 8) -> dict[str, object]:
    models = {}
    for target in long_eval_mod._PARETO_TARGETS:
        weights = [0.0] * 46
        weights[weight_idx] = 1.0
        models[target] = {
            "weights": weights,
            "mean": [0.0] * 45,
            "std": [1.0] * 45,
            "target_mean": 0.0,
            "target_std": 1.0,
        }
    return {"feature_dim": 45, "models": models}


def _dummy_v131_recall_floor_spec() -> dict[str, object]:
    models = {}
    for target in long_eval_mod._PARETO_TARGETS:
        weights = [0.0] * 46
        if target == "pair_recall":
            # Feature 30 is the raw budget in the 45-D live candidate-budget vector.
            weights[30] = 0.01
        models[target] = {
            "weights": weights,
            "mean": [0.0] * 45,
            "std": [1.0] * 45,
            "target_mean": 0.0,
            "target_std": 1.0,
        }
    return {
        "feature_dim": 45,
        "policy": {"pair_recall_floor": 0.6},
        "models": models,
    }


def _dummy_v125_candidate(score: float, selection_score: float) -> dict[str, object]:
    pairs = [
        {"score": score, "vacancy_score": score, "energy_score": score * 0.5},
        {"score": score - 0.1, "vacancy_score": score - 0.1, "energy_score": score * 0.25},
    ]
    return {
        "segment_k": 8,
        "selection_score": selection_score,
        "reachability_violation": 0.0,
        "projected_changed_count": 4.0,
        "predicted_reward_sum": score,
        "predicted_delta_e": score,
        "predicted_expected_tau": 2.0,
        "predicted_noop_risk_prob": 0.1,
        "candidate_quality_score": score,
        "proposal_support_density": 0.2,
        "planner_edge_completion_support_count": 32.0,
        "planner_projection_change_source": "vacancy_pair_completion",
        "vacancy_pair_projection_diagnostic": {
            "selected_pair_count": 32,
            "factorized_pair_scores": pairs,
        },
    }


def _dummy_v131_candidate(score: float, selection_score: float) -> dict[str, object]:
    candidate = _dummy_v125_candidate(score=score, selection_score=selection_score)
    candidate["vacancy_pair_projection_diagnostic"] = {
        "selected_pair_count": 128,
        "factorized_pair_scores": [
            {"score": float(score) - 0.001 * float(idx), "vacancy_score": float(score), "energy_score": float(score)}
            for idx in range(128)
        ],
    }
    return candidate


def test_v125_pareto_selector_diagnostic_builds_label_clean_features_without_changing_scores():
    candidates = [
        _dummy_v125_candidate(score=0.2, selection_score=10.0),
        _dummy_v125_candidate(score=0.9, selection_score=1.0),
    ]

    stats = long_eval_mod._apply_candidate_pareto_selector(
        candidates,
        selector_spec=_dummy_v125_pareto_spec(),
        mode="diagnostic",
        weight=1.0,
        pair_score_field="score",
        min_projected_changed_sites=2,
    )

    assert stats["enabled"] is True
    assert stats["feature_row_count"] == 2 * len(long_eval_mod._PARETO_BUDGETS)
    assert stats["missing_pair_score_candidate_count"] == 0
    assert stats["applied"] is False
    assert candidates[0]["selection_score"] == pytest.approx(10.0)
    assert candidates[1]["selection_score"] == pytest.approx(1.0)
    selector_record = candidates[1]["planner_candidate_pareto_selector"]
    assert selector_record["feature_dim"] == 45
    assert selector_record["budget_applied_to_projection"] is False
    assert selector_record["teacher_label_fields_used"] is False


def test_v125_pareto_selector_replace_updates_candidate_scores_from_live_pair_scores():
    candidates = [
        _dummy_v125_candidate(score=0.2, selection_score=10.0),
        _dummy_v125_candidate(score=0.9, selection_score=1.0),
    ]

    stats = long_eval_mod._apply_candidate_pareto_selector(
        candidates,
        selector_spec=_dummy_v125_pareto_spec(),
        mode="replace",
        weight=1.0,
        pair_score_field="score",
        min_projected_changed_sites=2,
    )
    selected = long_eval_mod._choose_planner_candidate(candidates, min_projected_changed_sites=2)

    assert stats["applied"] is True
    assert candidates[1]["selection_score"] > candidates[0]["selection_score"]
    assert selected is candidates[1]


def test_v131_recall_floor_selector_uses_spec_floor_before_balanced_budget_choice():
    candidates = [_dummy_v131_candidate(score=0.5, selection_score=10.0)]

    stats = long_eval_mod._apply_candidate_pareto_selector(
        candidates,
        selector_spec=_dummy_v131_recall_floor_spec(),
        mode="diagnostic",
        weight=1.0,
        pair_score_field="score",
        selector_policy="recall_floor_balanced",
        recall_floor=None,
        min_projected_changed_sites=2,
    )

    selector_record = candidates[0]["planner_candidate_pareto_selector"]
    assert stats["selector_policy"] == "recall_floor_balanced"
    assert stats["recall_floor"] == pytest.approx(0.6)
    assert stats["selected_budget"] == 64
    assert stats["selected_prediction_pair_recall"] == pytest.approx(0.64)
    assert stats["selected_pair_recall_floor_passed"] is True
    assert selector_record["selector_policy"] == "recall_floor_balanced"
    assert selector_record["budget"] == 64
    assert selector_record["pair_recall_floor_passed"] is True


def test_v134_recall_floor_selector_respects_min_budget_guard():
    candidates = [_dummy_v131_candidate(score=0.5, selection_score=10.0)]

    stats = long_eval_mod._apply_candidate_pareto_selector(
        candidates,
        selector_spec=_dummy_v131_recall_floor_spec(),
        mode="diagnostic",
        weight=1.0,
        pair_score_field="score",
        selector_policy="recall_floor_balanced",
        recall_floor=None,
        min_budget=96,
        clip_probability_predictions=True,
        min_projected_changed_sites=2,
    )

    selector_record = candidates[0]["planner_candidate_pareto_selector"]
    assert stats["selected_budget"] == 96
    assert stats["selected_prediction_pair_recall"] == pytest.approx(0.96)
    assert stats["selected_min_budget_passed"] is True
    assert selector_record["min_budget"] == 96
    assert selector_record["min_budget_passed"] is True
    assert selector_record["clip_probability_predictions"] is True


def test_v137_pareto_teacher_label_after_budget_validation_allows_zero_weight_add_mode():
    args = SimpleNamespace(
        planner_candidate_pareto_teacher_label_after_budget_projection=True,
        planner_candidate_pareto_selector_spec="selector.json",
        planner_candidate_pareto_apply_budget_to_projection=True,
        planner_teacher_overlap_oracle_mode="add",
        planner_teacher_overlap_oracle_weight=0.0,
    )

    long_eval_mod._validate_pareto_teacher_label_diagnostic(args)


def test_v137_pareto_teacher_label_after_budget_validation_rejects_selection_changing_oracle():
    args = SimpleNamespace(
        planner_candidate_pareto_teacher_label_after_budget_projection=True,
        planner_candidate_pareto_selector_spec="selector.json",
        planner_candidate_pareto_apply_budget_to_projection=True,
        planner_teacher_overlap_oracle_mode="replace",
        planner_teacher_overlap_oracle_weight=0.0,
    )

    with pytest.raises(ValueError, match="oracle_mode add"):
        long_eval_mod._validate_pareto_teacher_label_diagnostic(args)

    args.planner_teacher_overlap_oracle_mode = "add"
    args.planner_teacher_overlap_oracle_weight = 0.5
    with pytest.raises(ValueError, match="oracle_weight 0.0"):
        long_eval_mod._validate_pareto_teacher_label_diagnostic(args)


def test_long_eval_teacher_segment_marks_lattice_noop(monkeypatch):
    class InnerEnv:
        def __init__(self):
            self.vacancies = np.asarray([[0, 0, 0]], dtype=np.int32)
            self.cu = np.asarray([[1, 0, 0]], dtype=np.int32)

        def get_vacancy_array(self):
            return self.vacancies.copy()

        def get_cu_array(self):
            return self.cu.copy()

    class FakeEnv:
        def __init__(self):
            self.env = InnerEnv()

        def step(self, action):
            return np.zeros((1,), dtype=np.float32), 0.0, False, {
                "expected_delta_t": 0.25,
                "delta_t": 0.5,
            }

    monkeypatch.setattr(long_eval_mod.mod, "_sample_teacher_action", lambda env, rng: 0)

    segment = long_eval_mod._collect_teacher_segment(FakeEnv(), horizon_k=2, rng=np.random.default_rng(0))

    assert segment["is_noop"] is True
    assert segment["changed_site_count"] == 0
    assert segment["tau_exp"] == pytest.approx(0.5)
    assert segment["tau_real"] == pytest.approx(1.0)


def test_long_eval_teacher_segment_marks_lattice_change(monkeypatch):
    class InnerEnv:
        def __init__(self):
            self.vacancies = np.asarray([[0, 0, 0]], dtype=np.int32)
            self.cu = np.asarray([[1, 0, 0]], dtype=np.int32)

        def get_vacancy_array(self):
            return self.vacancies.copy()

        def get_cu_array(self):
            return self.cu.copy()

    class FakeEnv:
        def __init__(self):
            self.env = InnerEnv()

        def step(self, action):
            self.env.vacancies = np.asarray([[0, 0, 1]], dtype=np.int32)
            return np.zeros((1,), dtype=np.float32), 1.0, False, {
                "expected_delta_t": 0.25,
                "delta_t": 0.5,
            }

    monkeypatch.setattr(long_eval_mod.mod, "_sample_teacher_action", lambda env, rng: 0)

    segment = long_eval_mod._collect_teacher_segment(FakeEnv(), horizon_k=1, rng=np.random.default_rng(0))

    assert segment["is_noop"] is False
    assert segment["changed_site_count"] > 0
    assert segment["reward_sum"] == pytest.approx(1.0)


def test_long_eval_planner_can_score_with_baseline_tau_source():
    model_score, model_tau = long_eval_mod._compute_selection_score(
        pred_reward_sum=2.0,
        reward_scale=10.0,
        model_expected_tau=0.2,
        baseline_expected_tau=0.1,
        horizon_k=4,
        planner_tau_source="model",
    )
    baseline_score, baseline_tau = long_eval_mod._compute_selection_score(
        pred_reward_sum=2.0,
        reward_scale=10.0,
        model_expected_tau=0.2,
        baseline_expected_tau=0.1,
        horizon_k=4,
        planner_tau_source="baseline",
    )

    assert model_tau == pytest.approx(0.2)
    assert baseline_tau == pytest.approx(0.1)
    assert baseline_score == pytest.approx(2.0 * model_score)


def test_long_eval_planner_can_score_with_blended_tau_source():
    score, tau = long_eval_mod._compute_selection_score(
        pred_reward_sum=2.0,
        reward_scale=10.0,
        model_expected_tau=0.2,
        baseline_expected_tau=0.05,
        horizon_k=4,
        planner_tau_source="blend",
        planner_tau_blend_alpha=0.5,
    )

    assert tau == pytest.approx(0.1)
    assert score == pytest.approx(2.0)


def test_long_eval_duration_log_offset_scales_model_component():
    model_tau = long_eval_mod._duration_from_source(
        model_expected_tau=0.2,
        baseline_expected_tau=0.05,
        source="model",
        duration_log_offset=math.log(0.5),
    )
    blend_tau = long_eval_mod._duration_from_source(
        model_expected_tau=0.2,
        baseline_expected_tau=0.05,
        source="blend",
        blend_alpha=0.5,
        duration_log_offset=math.log(0.5),
    )

    assert model_tau == pytest.approx(0.1)
    assert blend_tau == pytest.approx(math.sqrt(0.05 * 0.1))


def test_long_eval_estimates_duration_log_offset_from_calibration_segments():
    offset = long_eval_mod._estimate_duration_log_offset(
        base_log_offset=0.0,
        predicted_tau=[2.0, 4.0],
        target_tau=[1.0, 2.0],
    )

    assert offset == pytest.approx(math.log(0.5))


def test_long_eval_planner_duration_residual_penalty_is_multiplicative():
    unpenalized, _ = long_eval_mod._compute_selection_score(
        pred_reward_sum=2.0,
        reward_scale=10.0,
        model_expected_tau=0.2,
        baseline_expected_tau=0.1,
        horizon_k=4,
        planner_tau_residual_penalty=0.0,
    )
    penalized, _ = long_eval_mod._compute_selection_score(
        pred_reward_sum=2.0,
        reward_scale=10.0,
        model_expected_tau=0.2,
        baseline_expected_tau=0.1,
        horizon_k=4,
        planner_tau_residual_penalty=1.0,
    )

    assert penalized < unpenalized
    assert penalized == pytest.approx(unpenalized * 0.5)


def test_reward_supervision_loss_weights_affect_zero_gate_terms():
    reward_hat = torch.tensor([1.0, 0.5], dtype=torch.float32)
    gate_logit = torch.tensor([1.0, -1.0], dtype=torch.float32)
    target_reward = torch.tensor([0.0, 2.0], dtype=torch.float32)

    base = _reward_supervision_losses(
        reward_hat,
        gate_logit,
        target_reward,
        reward_magnitude_weight=1.0,
        reward_gate_weight=0.25,
        reward_zero_weight=0.5,
    )
    stronger_zero_gate = _reward_supervision_losses(
        reward_hat,
        gate_logit,
        target_reward,
        reward_magnitude_weight=1.0,
        reward_gate_weight=2.0,
        reward_zero_weight=2.0,
    )

    assert stronger_zero_gate["loss"].item() > base["loss"].item()
    assert torch.allclose(stronger_zero_gate["gated_reward"], reward_hat * torch.sigmoid(gate_logit))


def test_projected_edit_logits_encode_projection_closed_context():
    current_types = torch.tensor([[2, 0, 1, 0]], dtype=torch.long)
    projected_types = torch.tensor([[0, 2, 1, 0]], dtype=torch.long)
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]], dtype=torch.float32)

    change_logits, type_logits = projected_edit_logits_from_types(current_types, projected_types, mask, logit=5.0)

    assert change_logits[0, 0].item() == pytest.approx(5.0)
    assert change_logits[0, 1].item() == pytest.approx(5.0)
    assert change_logits[0, 2].item() == pytest.approx(-5.0)
    assert change_logits[0, 3].item() == pytest.approx(-5.0)
    assert type_logits.argmax(dim=-1).tolist() == projected_types.tolist()


def test_normalize_segment_ks_sorts_and_deduplicates():
    assert _normalize_segment_ks(4, [8, 2, 4, 2]) == [2, 4, 8]
    with pytest.raises(ValueError):
        _normalize_segment_ks(4, [0, 2])
