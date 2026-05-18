#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-cuda}"
RESULT_ROOT="${RESULT_ROOT:-results/largek_dualsupport_0516}"
STAGE="${STAGE:-v57_teacheraug_positive_riskguard}"
SAVE_DIR="${RESULT_ROOT}/${STAGE}"
INIT_CKPT="${INIT_CKPT:-results/largek_nooprisk_0516/v56_noop_horizon_augmented_riskhead/final_model.pt}"
PLANNER_CKPT="${PLANNER_CKPT:-results/largek_nooprisk_0516/v56_noop_horizon_augmented_riskhead/final_model.pt}"
CACHE="${CACHE:-${SAVE_DIR}/segments.pt}"
EPOCHS="${EPOCHS:-6}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LR="${LR:-5e-6}"
LOG="${SAVE_DIR}/pipeline.log"

mkdir -p "${SAVE_DIR}"

{
  echo "=== ${STAGE} START $(date -Is) ==="
  echo "init=${INIT_CKPT}"
  echo "planner_signature=${PLANNER_CKPT}"
  echo "cache=${CACHE}"
  echo "fix=large-k dual-support diagnostic: inference-support hard negatives plus teacher-path augmented positives"
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
  --segment_ks 128 256 512 1024 \
  --train_segments_per_k 96 \
  --val_segments_per_k 24 \
  --max_episode_steps 4096 \
  --max_segments_per_rollout 4 \
  --max_candidate_sites 1024 \
  --teacher_candidate_neighbor_depth 1 \
  --teacher_path_summary_mode stepwise \
  --tau_supervision_mode prior_main \
  --reward_prediction_source projected \
  --include_noop_segments \
  --keep_after_noop_segments \
  --freeze_duration_heads \
  --realized_tau_weight 0.05 \
  --tau_log_mu_weight 0.25 \
  --count_loss_weight 1.0 \
  --mask_sparsity_weight 0.012 \
  --noop_change_weight 0.35 \
  --noop_type_copy_weight 0.15 \
  --projected_noop_fp_weight 0.75 \
  --noop_risk_weight 0.5 \
  --prior_noop_risk_weight 0.5 \
  --pair_weight 0.35 \
  --prior_pair_weight 0.35 \
  --reward_weight 1.0 \
  --prior_reward_weight 1.0 \
  --reward_gate_weight 6.0 \
  --reward_zero_weight 10.0 \
  --reward_sign_weight 0.5 \
  --reward_magnitude_weight 1.0 \
  --prior_edit_weight 0.95 \
  --prior_latent_weight 0.3 \
  --proj_weight 1.1 \
  --no_aux_anneal \
  --lr "${LR}" \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --device "${DEVICE}" \
  2>&1 | tee -a "${LOG}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" eval_macro_time_alignment.py \
  --checkpoint "${SAVE_DIR}/final_model.pt" \
  --cache "${CACHE}" \
  --split val \
  --batch_size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --output "${SAVE_DIR}/eval_time_alignment_final.json" \
  --save_all_samples \
  2>&1 | tee -a "${LOG}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" eval_macro_long_trajectory.py \
  --checkpoint "${SAVE_DIR}/final_model.pt" \
  --planner_segment_ks 128 256 512 1024 \
  --rollout_segments 80 \
  --max_episode_steps_override 4096 \
  --duration_source model \
  --planner_tau_source model \
  --planner_score_mode energy_per_sqrt_tau \
  --min_projected_changed_sites 2 \
  --device "${DEVICE}" \
  --output "${SAVE_DIR}/eval_long_trajectory.json" \
  2>&1 | tee -a "${LOG}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" eval_macro_long_trajectory.py \
  --checkpoint "${SAVE_DIR}/final_model.pt" \
  --planner_segment_ks 128 256 512 1024 \
  --rollout_segments 80 \
  --max_episode_steps_override 4096 \
  --duration_source model \
  --planner_tau_source model \
  --planner_score_mode energy_per_sqrt_tau \
  --planner_noop_risk_penalty 1.0 \
  --min_projected_changed_sites 2 \
  --device "${DEVICE}" \
  --output "${SAVE_DIR}/eval_long_riskpen1.json" \
  2>&1 | tee -a "${LOG}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" eval_macro_long_trajectory.py \
  --checkpoint "${SAVE_DIR}/final_model.pt" \
  --planner_segment_ks 128 256 512 1024 \
  --rollout_segments 80 \
  --max_episode_steps_override 4096 \
  --duration_source model \
  --planner_tau_source baseline \
  --planner_score_mode energy_per_tau \
  --planner_noop_risk_penalty 1.0 \
  --min_projected_changed_sites 10 \
  --device "${DEVICE}" \
  --output "${SAVE_DIR}/eval_long_model_baselineplanner_min10_riskpen1.json" \
  2>&1 | tee -a "${LOG}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" eval_macro_long_trajectory.py \
  --checkpoint "${SAVE_DIR}/final_model.pt" \
  --planner_segment_ks 128 256 512 1024 \
  --rollout_segments 80 \
  --max_episode_steps_override 4096 \
  --duration_source baseline \
  --planner_tau_source baseline \
  --planner_score_mode energy_per_tau \
  --planner_noop_risk_penalty 1.0 \
  --min_projected_changed_sites 10 \
  --device "${DEVICE}" \
  --output "${SAVE_DIR}/eval_long_baseline_min10_riskpen1.json" \
  2>&1 | tee -a "${LOG}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" eval_macro_long_trajectory.py \
  --checkpoint "${SAVE_DIR}/final_model.pt" \
  --planner_segment_ks 128 256 512 1024 \
  --rollout_segments 20 \
  --max_episode_steps_override 4096 \
  --duration_source baseline \
  --planner_tau_source baseline \
  --planner_score_mode energy_per_tau \
  --planner_noop_risk_penalty 1.0 \
  --min_projected_changed_sites 10 \
  --allow_teacher_noop_segments \
  --device "${DEVICE}" \
  --output "${SAVE_DIR}/eval_long_baseline_min10_riskpen1_allownoop20.json" \
  2>&1 | tee -a "${LOG}"

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

save_dir = Path("${SAVE_DIR}")
summary = {
    "stage": "${STAGE}",
    "init_checkpoint": "${INIT_CKPT}",
    "planner_signature_checkpoint": "${PLANNER_CKPT}",
    "dataset_cache": "${CACHE}",
    "segment_ks": [128, 256, 512, 1024],
    "support": "inference_support_hard_negatives_plus_teacher_path_augmented_positives",
    "teacher_candidate_augmentation": True,
    "teacher_candidate_neighbor_depth": 1,
    "loss_patch": {
        "freeze_duration_heads": True,
        "projected_noop_fp_weight": 0.75,
        "noop_risk_weight": 0.5,
        "prior_noop_risk_weight": 0.5,
    },
    "files": {},
}
for name in [
    "metrics.json",
    "eval_time_alignment_final.json",
    "eval_long_trajectory.json",
    "eval_long_riskpen1.json",
    "eval_long_model_baselineplanner_min10_riskpen1.json",
    "eval_long_baseline_min10_riskpen1.json",
    "eval_long_baseline_min10_riskpen1_allownoop20.json",
]:
    path = save_dir / name
    if not path.exists():
        continue
    data = json.loads(path.read_text())
    item = {}
    for key in [
        "val",
        "dataset",
        "noop_risk",
        "reward_sum",
        "tau_expected",
        "cumulative",
        "chosen_k_histogram",
        "completed_rollout_segments",
        "requested_rollout_segments",
        "stop_reason",
        "skipped_noop",
        "planner_score_mode",
        "planner_noop_risk_penalty",
        "planner_tau_source",
        "duration_source",
        "effective_min_projected_changed_sites",
    ]:
        if key in data:
            item[key] = data[key]
    summary["files"][name] = item
(save_dir / "stage_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
print(f"Saved summary to {save_dir / 'stage_summary.json'}")
PY

echo "=== ${STAGE} DONE $(date -Is) ===" | tee -a "${LOG}"
