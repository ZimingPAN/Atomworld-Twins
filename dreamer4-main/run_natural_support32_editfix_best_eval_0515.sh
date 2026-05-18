#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
GPU_ID="${GPU_ID:-2}"
DEVICE="${DEVICE:-cuda}"
SAVE_DIR="${SAVE_DIR:-results/natural_teacher_support32_editfix_0515/v57_adaptive_support32_editfix}"
CHECKPOINT="${CHECKPOINT:-${SAVE_DIR}/best_model.pt}"
CACHE="${CACHE:-results/natural_teacher_support32_0515/v57_adaptive_support32_noaug/segments.pt}"
LOG="${SAVE_DIR}/best_eval.log"

{
  echo "=== best eval START $(date -Is) ==="
  echo "checkpoint=${CHECKPOINT}"
  echo "cache=${CACHE}"
} | tee -a "${LOG}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" eval_macro_time_alignment.py \
  --checkpoint "${CHECKPOINT}" \
  --cache "${CACHE}" \
  --split val \
  --batch_size 2 \
  --device "${DEVICE}" \
  --output "${SAVE_DIR}/eval_time_alignment_best.json" \
  --save_all_samples \
  2>&1 | tee -a "${LOG}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" eval_macro_long_trajectory.py \
  --checkpoint "${CHECKPOINT}" \
  --planner_segment_ks 8 16 32 \
  --rollout_segments 80 \
  --max_episode_steps_override 1024 \
  --duration_source model \
  --planner_tau_source model \
  --planner_score_mode energy_per_sqrt_tau \
  --min_projected_changed_sites 2 \
  --device "${DEVICE}" \
  --output "${SAVE_DIR}/eval_long_trajectory_best.json" \
  2>&1 | tee -a "${LOG}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" eval_macro_long_trajectory.py \
  --checkpoint "${CHECKPOINT}" \
  --planner_segment_ks 8 16 32 \
  --rollout_segments 20 \
  --max_episode_steps_override 1024 \
  --duration_source model \
  --planner_tau_source model \
  --planner_score_mode energy_per_sqrt_tau \
  --min_projected_changed_sites 0 \
  --device "${DEVICE}" \
  --output "${SAVE_DIR}/eval_long_min0_best.json" \
  2>&1 | tee -a "${LOG}"

echo "=== best eval DONE $(date -Is) ===" | tee -a "${LOG}"
