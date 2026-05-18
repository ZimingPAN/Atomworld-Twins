#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
GPU_ID="${GPU_ID:-1}"
DEVICE="${DEVICE:-cuda}"
RESULT_ROOT="${RESULT_ROOT:-results/natural_teacher_support32_noopsupport_0515}"
STAGE="${STAGE:-v58_adaptive2048_noop_support}"
SAVE_DIR="${RESULT_ROOT}/${STAGE}"
INIT_CKPT="${INIT_CKPT:-results/natural_teacher_support32_adapt2048_from_fixed_0515/v58_adaptive2048_from_fixed_balanced/final_model.pt}"
PLANNER_CKPT="${PLANNER_CKPT:-${INIT_CKPT}}"
CACHE="${CACHE:-${SAVE_DIR}/segments.pt}"
LOG="${SAVE_DIR}/pipeline.log"

mkdir -p "${SAVE_DIR}"

{
  echo "=== ${STAGE} START $(date -Is) ==="
  echo "init=${INIT_CKPT}"
  echo "planner=${PLANNER_CKPT}"
  echo "cache=${CACHE}"
  echo "fix=adaptive support32 planner-selected no-op support with explicit zero-edit/zero-reward samples"
} | tee -a "${LOG}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" train_dreamer_macro_edit.py \
  --save_dir "${SAVE_DIR}" \
  --dataset_cache "${CACHE}" \
  --init_from "${INIT_CKPT}" \
  --planner_selected_from "${PLANNER_CKPT}" \
  --planner_selected_min_projected_changed_sites 2 \
  --planner_selected_duration_source baseline \
  --planner_selected_tau_source baseline \
  --planner_selected_score_mode energy_per_sqrt_tau \
  --planner_selected_reward_prediction_source projected \
  --segment_ks 8 16 32 \
  --train_segments_per_k 192 \
  --val_segments_per_k 48 \
  --max_episode_steps 1024 \
  --max_segments_per_rollout 8 \
  --max_seed_vacancies 32 \
  --max_candidate_sites 2048 \
  --teacher_path_summary_mode stepwise \
  --tau_supervision_mode prior_main \
  --reward_prediction_source projected \
  --include_noop_segments \
  --realized_tau_weight 0.05 \
  --tau_log_mu_weight 0.5 \
  --count_loss_weight 0.5 \
  --mask_sparsity_weight 0.005 \
  --pair_weight 0.1 \
  --prior_pair_weight 0.1 \
  --reward_weight 1.0 \
  --prior_reward_weight 1.0 \
  --reward_gate_weight 1.0 \
  --reward_zero_weight 2.0 \
  --prior_edit_weight 0.45 \
  --prior_latent_weight 0.3 \
  --proj_weight 0.7 \
  --no_aux_anneal \
  --natural_teacher_backend kmc \
  --segment_boundary_mode adaptive_key_event \
  --adaptive_min_k 8 \
  --adaptive_candidate_horizon_source actual \
  --adaptive_key_moving_types 1 \
  --disable_teacher_candidate_augmentation \
  --lr 2e-5 \
  --batch_size 2 \
  --epochs 6 \
  --device "${DEVICE}" \
  2>&1 | tee -a "${LOG}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" eval_macro_time_alignment.py \
  --checkpoint "${SAVE_DIR}/final_model.pt" \
  --cache "${CACHE}" \
  --split val \
  --batch_size 2 \
  --device "${DEVICE}" \
  --output "${SAVE_DIR}/eval_time_alignment_final.json" \
  --save_all_samples \
  2>&1 | tee -a "${LOG}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" eval_macro_long_trajectory.py \
  --checkpoint "${SAVE_DIR}/final_model.pt" \
  --planner_segment_ks 8 16 32 \
  --rollout_segments 80 \
  --max_episode_steps_override 1024 \
  --duration_source model \
  --planner_tau_source model \
  --planner_score_mode energy_per_sqrt_tau \
  --min_projected_changed_sites 2 \
  --device "${DEVICE}" \
  --output "${SAVE_DIR}/eval_long_trajectory.json" \
  2>&1 | tee -a "${LOG}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" eval_macro_long_trajectory.py \
  --checkpoint "${SAVE_DIR}/final_model.pt" \
  --planner_segment_ks 8 16 32 \
  --rollout_segments 80 \
  --max_episode_steps_override 1024 \
  --duration_source baseline \
  --planner_tau_source baseline \
  --planner_score_mode energy_per_sqrt_tau \
  --min_projected_changed_sites 2 \
  --device "${DEVICE}" \
  --output "${SAVE_DIR}/eval_long_trajectory_baselinetau.json" \
  2>&1 | tee -a "${LOG}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" eval_macro_long_trajectory.py \
  --checkpoint "${SAVE_DIR}/final_model.pt" \
  --planner_segment_ks 8 16 32 \
  --rollout_segments 80 \
  --max_episode_steps_override 1024 \
  --duration_source model \
  --planner_tau_source baseline \
  --planner_score_mode energy_per_sqrt_tau \
  --min_projected_changed_sites 2 \
  --device "${DEVICE}" \
  --output "${SAVE_DIR}/eval_long_model_plannerbaseline.json" \
  2>&1 | tee -a "${LOG}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" eval_macro_long_trajectory.py \
  --checkpoint "${SAVE_DIR}/final_model.pt" \
  --planner_segment_ks 8 16 32 \
  --rollout_segments 20 \
  --max_episode_steps_override 1024 \
  --duration_source model \
  --planner_tau_source model \
  --planner_score_mode energy_per_sqrt_tau \
  --min_projected_changed_sites 0 \
  --device "${DEVICE}" \
  --output "${SAVE_DIR}/eval_long_min0_diagnostic.json" \
  2>&1 | tee -a "${LOG}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" eval_macro_long_trajectory.py \
  --checkpoint "${SAVE_DIR}/final_model.pt" \
  --planner_segment_ks 8 16 32 \
  --rollout_segments 80 \
  --max_episode_steps_override 1024 \
  --duration_source baseline \
  --planner_tau_source baseline \
  --planner_score_mode energy_per_sqrt_tau \
  --min_projected_changed_sites 2 \
  --allow_teacher_noop_segments \
  --device "${DEVICE}" \
  --output "${SAVE_DIR}/eval_long_baseline_allownoop80.json" \
  2>&1 | tee -a "${LOG}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" eval_macro_long_trajectory.py \
  --checkpoint "${SAVE_DIR}/final_model.pt" \
  --planner_segment_ks 8 16 32 \
  --rollout_segments 20 \
  --max_episode_steps_override 1024 \
  --duration_source model \
  --planner_tau_source baseline \
  --planner_score_mode energy_per_sqrt_tau \
  --min_projected_changed_sites 0 \
  --allow_teacher_noop_segments \
  --device "${DEVICE}" \
  --output "${SAVE_DIR}/eval_long_min0_model_plannerbaseline_allownoop20.json" \
  2>&1 | tee -a "${LOG}"

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

save_dir = Path("${SAVE_DIR}")
summary = {
    "stage": "${STAGE}",
    "init_checkpoint": "${INIT_CKPT}",
    "planner_checkpoint": "${PLANNER_CKPT}",
    "dataset_cache": "${CACHE}",
    "segment_ks": [8, 16, 32],
    "boundary": "adaptive_key_event",
    "support": "planner_selected_noop_support_no_teacher_path_aug",
    "max_candidate_sites": 2048,
    "files": {},
}
summary_keys = [
    "num_samples",
    "reward_corr",
    "tau_log_mae",
    "tau_log_corr",
    "tau_scale_ratio",
    "change_f1",
    "projected_change_f1",
    "projected_changed_type_acc",
    "reachability_violation_rate",
    "completed_segments",
    "requested_segments",
    "completed_rollout_segments",
    "requested_rollout_segments",
    "stop_reason",
    "skipped_noop",
    "allow_teacher_noop_segments",
    "duration_source",
    "planner_tau_source",
    "expected_time_ratio",
    "delta_e_ratio",
    "cumulative_pred_delta_e",
    "cumulative_teacher_delta_e",
    "pred_tau_sum",
    "teacher_tau_sum",
    "chosen_horizon_counts",
    "chosen_k_histogram",
]
for name in [
    "metrics.json",
    "eval_time_alignment_final.json",
    "eval_long_trajectory.json",
    "eval_long_trajectory_baselinetau.json",
    "eval_long_model_plannerbaseline.json",
    "eval_long_min0_diagnostic.json",
    "eval_long_baseline_allownoop80.json",
    "eval_long_min0_model_plannerbaseline_allownoop20.json",
]:
    path = save_dir / name
    if not path.exists():
        continue
    data = json.loads(path.read_text())
    item = {key: data.get(key) for key in summary_keys if key in data}
    if "reward_sum" in data:
        item["reward_sum"] = data["reward_sum"]
    if "tau_expected" in data:
        item["tau_expected"] = data["tau_expected"]
    if "val" in data:
        val = data["val"]
        item["val"] = {key: val.get(key) for key in summary_keys if key in val}
    summary["files"][name] = item
(save_dir / "stage_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
print(f"Saved summary to {save_dir / 'stage_summary.json'}")
PY

echo "=== ${STAGE} DONE $(date -Is) ===" | tee -a "${LOG}"
