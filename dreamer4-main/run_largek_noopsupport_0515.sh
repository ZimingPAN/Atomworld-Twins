#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-cuda}"
RESULT_ROOT="${RESULT_ROOT:-results/largek_noopsupport_0515}"
STAGE="${STAGE:-v56_noop_support}"
SAVE_DIR="${RESULT_ROOT}/${STAGE}"
INIT_CKPT="${INIT_CKPT:-results/largek_priorpathfix_0515/v56_projected_priorpath_duration/final_model.pt}"
PLANNER_CKPT="${PLANNER_CKPT:-${INIT_CKPT}}"
CACHE="${SAVE_DIR}/segments.pt"
LOG="${SAVE_DIR}/pipeline.log"

mkdir -p "${SAVE_DIR}"

{
  echo "=== ${STAGE} START $(date -Is) ==="
  echo "init=${INIT_CKPT}"
  echo "planner=${PLANNER_CKPT}"
  echo "cache=${CACHE}"
  echo "fix=planner-selected projected no-op support with explicit zero-edit/zero-reward samples"
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
  --teacher_path_summary_mode stepwise \
  --tau_supervision_mode prior_main \
  --reward_prediction_source projected \
  --include_noop_segments \
  --realized_tau_weight 0.1 \
  --tau_log_mu_weight 2.0 \
  --count_loss_weight 0.5 \
  --mask_sparsity_weight 0.005 \
  --pair_weight 0.1 \
  --prior_pair_weight 0.1 \
  --reward_weight 1.0 \
  --prior_reward_weight 1.0 \
  --reward_gate_weight 1.0 \
  --reward_zero_weight 2.0 \
  --prior_edit_weight 0.75 \
  --prior_latent_weight 0.5 \
  --proj_weight 1.0 \
  --no_aux_anneal \
  --lr 5e-5 \
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
  --duration_source baseline \
  --planner_tau_source baseline \
  --planner_score_mode energy_per_sqrt_tau \
  --min_projected_changed_sites 2 \
  --device "${DEVICE}" \
  --output "${SAVE_DIR}/eval_long_trajectory_baselinetau.json" \
  2>&1 | tee -a "${LOG}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" eval_macro_long_trajectory.py \
  --checkpoint "${SAVE_DIR}/final_model.pt" \
  --planner_segment_ks 128 256 512 1024 \
  --rollout_segments 80 \
  --max_episode_steps_override 4096 \
  --duration_source model \
  --planner_tau_source baseline \
  --planner_score_mode energy_per_sqrt_tau \
  --min_projected_changed_sites 2 \
  --device "${DEVICE}" \
  --output "${SAVE_DIR}/eval_long_model_plannerbaseline.json" \
  2>&1 | tee -a "${LOG}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" eval_macro_long_trajectory.py \
  --checkpoint "${SAVE_DIR}/final_model.pt" \
  --planner_segment_ks 128 256 512 1024 \
  --rollout_segments 80 \
  --max_episode_steps_override 4096 \
  --duration_source blend \
  --duration_blend_alpha 0.25 \
  --planner_tau_source blend \
  --planner_tau_blend_alpha 0.25 \
  --planner_score_mode energy_per_sqrt_tau \
  --min_projected_changed_sites 2 \
  --device "${DEVICE}" \
  --output "${SAVE_DIR}/eval_long_blend025.json" \
  2>&1 | tee -a "${LOG}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" eval_macro_long_trajectory.py \
  --checkpoint "${SAVE_DIR}/final_model.pt" \
  --planner_segment_ks 128 256 512 1024 \
  --rollout_segments 20 \
  --max_episode_steps_override 4096 \
  --duration_source baseline \
  --planner_tau_source baseline \
  --planner_score_mode energy_per_sqrt_tau \
  --min_projected_changed_sites 2 \
  --allow_teacher_noop_segments \
  --device "${DEVICE}" \
  --output "${SAVE_DIR}/eval_long_baseline_allownoop20.json" \
  2>&1 | tee -a "${LOG}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" eval_macro_long_trajectory.py \
  --checkpoint "${SAVE_DIR}/final_model.pt" \
  --planner_segment_ks 128 256 512 1024 \
  --rollout_segments 20 \
  --max_episode_steps_override 4096 \
  --duration_source model \
  --planner_tau_source baseline \
  --planner_score_mode energy_per_sqrt_tau \
  --min_projected_changed_sites 0 \
  --allow_teacher_noop_segments \
  --device "${DEVICE}" \
  --output "${SAVE_DIR}/eval_long_min0_model_plannerbaseline_allownoop20.json" \
  2>&1 | tee -a "${LOG}"

echo "=== ${STAGE} DONE $(date -Is) ===" | tee -a "${LOG}"
