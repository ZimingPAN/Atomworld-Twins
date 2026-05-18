#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-cuda}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

SRC_DIR="results/largek_noopsupport_0515/v56_noop_support"
SAVE_DIR="results/largek_noopaug_0516/v56_noop_horizon_augmented_sigfix"
CACHE="$SAVE_DIR/segments_noop_horizon_aug.pt"
LOG="$SAVE_DIR/pipeline.log"

mkdir -p "$SAVE_DIR"

{
  echo "=== v56_noop_horizon_augmented START $(date -Is) ==="
  echo "source=$SRC_DIR"
  echo "cache=$CACHE"
  ../.venv/bin/python augment_noop_horizon_cache.py \
    --input "$SRC_DIR/segments.pt" \
    --output "$CACHE" \
    --horizons 128,256,512,1024

  ../.venv/bin/python train_dreamer_macro_edit.py \
    --save_dir "$SAVE_DIR" \
    --dataset_cache "$CACHE" \
    --init_from "$SRC_DIR/final_model.pt" \
    --planner_selected_from results/largek_priorpathfix_0515/v56_projected_priorpath_duration/final_model.pt \
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
    --freeze_duration_heads \
    --realized_tau_weight 0.05 \
    --tau_log_mu_weight 0.25 \
    --count_loss_weight 0.9 \
    --mask_sparsity_weight 0.015 \
    --noop_change_weight 0.35 \
    --noop_type_copy_weight 0.15 \
    --projected_noop_fp_weight 0.75 \
    --pair_weight 0.25 \
    --prior_pair_weight 0.25 \
    --reward_weight 1.5 \
    --prior_reward_weight 1.5 \
    --reward_gate_weight 6.0 \
    --reward_zero_weight 10.0 \
    --reward_sign_weight 0.5 \
    --reward_magnitude_weight 1.0 \
    --prior_edit_weight 0.75 \
    --prior_latent_weight 0.3 \
    --proj_weight 1.0 \
    --no_aux_anneal \
    --lr 8e-6 \
    --batch_size 2 \
    --epochs 4 \
    --device "$DEVICE"

  ../.venv/bin/python eval_macro_time_alignment.py \
    --checkpoint "$SAVE_DIR/final_model.pt" \
    --cache "$CACHE" \
    --split val \
    --batch_size 2 \
    --device "$DEVICE" \
    --output "$SAVE_DIR/eval_time_alignment_final.json" \
    --save_all_samples

  ../.venv/bin/python eval_macro_long_trajectory.py \
    --checkpoint "$SAVE_DIR/final_model.pt" \
    --planner_segment_ks 128 256 512 1024 \
    --rollout_segments 80 \
    --max_episode_steps_override 4096 \
    --duration_source model \
    --planner_tau_source model \
    --planner_score_mode energy_per_sqrt_tau \
    --min_projected_changed_sites 2 \
    --device "$DEVICE" \
    --output "$SAVE_DIR/eval_long_trajectory.json"

  ../.venv/bin/python eval_macro_long_trajectory.py \
    --checkpoint "$SAVE_DIR/final_model.pt" \
    --planner_segment_ks 128 256 512 1024 \
    --rollout_segments 80 \
    --max_episode_steps_override 4096 \
    --duration_source baseline \
    --planner_tau_source baseline \
    --planner_score_mode energy_per_sqrt_tau \
    --min_projected_changed_sites 2 \
    --device "$DEVICE" \
    --output "$SAVE_DIR/eval_long_trajectory_baselinetau.json"

  ../.venv/bin/python eval_macro_long_trajectory.py \
    --checkpoint "$SAVE_DIR/final_model.pt" \
    --planner_segment_ks 128 256 512 1024 \
    --rollout_segments 20 \
    --max_episode_steps_override 4096 \
    --duration_source baseline \
    --planner_tau_source baseline \
    --planner_score_mode energy_per_sqrt_tau \
    --min_projected_changed_sites 2 \
    --allow_teacher_noop_segments \
    --device "$DEVICE" \
    --output "$SAVE_DIR/eval_long_baseline_allownoop20.json"

  ../.venv/bin/python eval_macro_long_trajectory.py \
    --checkpoint "$SAVE_DIR/final_model.pt" \
    --planner_segment_ks 128 256 512 1024 \
    --rollout_segments 20 \
    --max_episode_steps_override 4096 \
    --duration_source model \
    --planner_tau_source baseline \
    --planner_score_mode energy_per_sqrt_tau \
    --min_projected_changed_sites 0 \
    --allow_teacher_noop_segments \
    --device "$DEVICE" \
    --output "$SAVE_DIR/eval_long_min0_model_plannerbaseline_allownoop20.json"

  echo "=== v56_noop_horizon_augmented DONE $(date -Is) ==="
} 2>&1 | tee -a "$LOG"
