#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-cuda}"
RESULT_ROOT="${RESULT_ROOT:-results/largek_priorpathfix_0515}"
STAGE="${STAGE:-v56_projected_priorpath_duration}"
SAVE_DIR="${RESULT_ROOT}/${STAGE}"
INIT_CKPT="${INIT_CKPT:-results/largek_projectedtaufix_0515/v56_projected_tau_consistent_heads/final_model.pt}"
CACHE_SOURCE="${CACHE_SOURCE:-results/largek_contig_0515/v56_teacher_contig4/segments.pt}"
CACHE="${SAVE_DIR}/segments.pt"
LOG="${SAVE_DIR}/pipeline.log"

mkdir -p "${SAVE_DIR}"
ln -sf "../../largek_contig_0515/v56_teacher_contig4/segments.pt" "${CACHE}"

{
  echo "=== ${STAGE} START $(date -Is) ==="
  echo "init=${INIT_CKPT}"
  echo "cache=${CACHE_SOURCE}"
  echo "fix=train prior path, macro dynamics, k embedding, and duration heads under projected duration supervision"
} | tee -a "${LOG}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" train_dreamer_macro_edit.py \
  --save_dir "${SAVE_DIR}" \
  --dataset_cache "${CACHE}" \
  --init_from "${INIT_CKPT}" \
  --segment_ks 128 256 512 1024 \
  --train_segments_per_k 96 \
  --val_segments_per_k 24 \
  --max_episode_steps 4096 \
  --max_segments_per_rollout 4 \
  --max_candidate_sites 1024 \
  --teacher_path_summary_mode stepwise \
  --tau_supervision_mode prior_main \
  --reward_prediction_source projected \
  --realized_tau_weight 0.1 \
  --tau_log_mu_weight 2.0 \
  --count_loss_weight 0.2 \
  --prior_edit_weight 0.75 \
  --prior_latent_weight 0.5 \
  --proj_weight 1.0 \
  --no_aux_anneal \
  --train_duration_prior_path_only \
  --lr 1e-4 \
  --batch_size 2 \
  --epochs 8 \
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

echo "=== ${STAGE} DONE $(date -Is) ===" | tee -a "${LOG}"
