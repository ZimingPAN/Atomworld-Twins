#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
GPU_ID="${GPU_ID:-1}"
DEVICE="${DEVICE:-cuda}"
RESULT_ROOT="${RESULT_ROOT:-results/natural_teacher_support32_onpolicy_proposal_0516}"
STAGE="${STAGE:-v70_onpolicy_topk128_proposal_smoke}"
SAVE_DIR="${RESULT_ROOT}/${STAGE}"
INIT_CKPT="${INIT_CKPT:-results/natural_teacher_support32_proposalhead_0516/v69_proposal_smoke1/final_model.pt}"
PLANNER_CKPT="${PLANNER_CKPT:-results/natural_teacher_support32_proposalhead_0516/v69_proposal_smoke1/final_model.pt}"
PROTECTED_CKPT="${PROTECTED_CKPT:-results/natural_teacher_support32_nooprisk_0516/v63_nooprisk_head_plannerpenalty/final_model.pt}"
CACHE="${CACHE:-${SAVE_DIR}/segments.pt}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LR="${LR:-5e-5}"
TRAIN_PER_K="${TRAIN_PER_K:-8}"
VAL_PER_K="${VAL_PER_K:-4}"
TEACHER_OVERLAP_RERANK_WEIGHT="${TEACHER_OVERLAP_RERANK_WEIGHT:-0.0}"
LOG="${SAVE_DIR}/pipeline.log"

mkdir -p "${SAVE_DIR}"

{
  echo "=== ${STAGE} START $(date -Is) ==="
  echo "init=${INIT_CKPT}"
  echo "planner=${PLANNER_CKPT}"
  echo "protected_reward_tau=${PROTECTED_CKPT}"
  echo "cache=${CACHE}"
  echo "teacher_overlap_rerank_weight=${TEACHER_OVERLAP_RERANK_WEIGHT}"
  echo "fix=on-policy planner-selected cache parity with v69 proposal topk128 + v63 protected reward/tau/risk"
} | tee -a "${LOG}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" train_dreamer_macro_edit.py \
  --save_dir "${SAVE_DIR}" \
  --dataset_cache "${CACHE}" \
  --init_from "${INIT_CKPT}" \
  --planner_selected_from "${PLANNER_CKPT}" \
  --planner_selected_reward_checkpoint "${PROTECTED_CKPT}" \
  --planner_selected_duration_checkpoint "${PROTECTED_CKPT}" \
  --planner_selected_planner_duration_checkpoint_source duration \
  --planner_selected_aux_projected_types_source primary \
  --planner_selected_projection_change_source proposal \
  --planner_selected_projection_topk_source proposal \
  --planner_selected_projection_topk_budget 128 \
  --planner_selected_noop_risk_penalty 1.0 \
  --planner_selected_teacher_overlap_rerank_weight "${TEACHER_OVERLAP_RERANK_WEIGHT}" \
  --planner_selected_min_projected_changed_sites 2 \
  --planner_selected_duration_source model \
  --planner_selected_tau_source model \
  --planner_selected_score_mode energy \
  --planner_selected_reward_prediction_source projected \
  --segment_ks 8 16 32 \
  --train_segments_per_k "${TRAIN_PER_K}" \
  --val_segments_per_k "${VAL_PER_K}" \
  --max_episode_steps 1024 \
  --max_segments_per_rollout 12 \
  --max_seed_vacancies 32 \
  --max_candidate_sites 2048 \
  --teacher_candidate_neighbor_depth 1 \
  --teacher_path_summary_mode stepwise \
  --tau_supervision_mode prior_main \
  --reward_prediction_source projected \
  --include_noop_segments \
  --keep_after_noop_segments \
  --train_proposal_head_only \
  --proposal_support_weight 1.0 \
  --prior_proposal_support_weight 1.0 \
  --reward_weight 0.0 \
  --prior_reward_weight 0.0 \
  --tau_weight 0.0 \
  --realized_tau_weight 0.0 \
  --tau_log_mu_weight 0.0 \
  --pair_weight 0.0 \
  --prior_pair_weight 0.0 \
  --prior_edit_weight 0.0 \
  --prior_latent_weight 0.0 \
  --proj_weight 0.0 \
  --path_weight 0.0 \
  --natural_teacher_backend kmc \
  --segment_boundary_mode adaptive_key_event \
  --adaptive_min_k 8 \
  --adaptive_candidate_horizon_source actual \
  --adaptive_key_moving_types 1 \
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
  --reward_checkpoint "${PROTECTED_CKPT}" \
  --duration_checkpoint "${PROTECTED_CKPT}" \
  --aux_projected_types_source primary \
  --planner_segment_ks 8 16 32 \
  --rollout_segments 20 \
  --max_episode_steps_override 1024 \
  --duration_source model \
  --planner_tau_source model \
  --planner_score_mode energy \
  --planner_noop_risk_penalty 1.0 \
  --planner_projection_change_source proposal \
  --planner_projection_topk_source proposal \
  --planner_projection_topk_budget 128 \
  --proposal_diagnostic \
  --min_projected_changed_sites 2 \
  --device "${DEVICE}" \
  --output "${SAVE_DIR}/eval_long_energy_topk128_20.json" \
  2>&1 | tee -a "${LOG}"

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

save_dir = Path("${SAVE_DIR}")
summary = {
    "stage": "${STAGE}",
    "init_checkpoint": "${INIT_CKPT}",
    "planner_checkpoint": "${PLANNER_CKPT}",
    "protected_reward_tau_checkpoint": "${PROTECTED_CKPT}",
    "dataset_cache": "${CACHE}",
    "collector": "v69 proposal topk128 with v63 protected reward/tau/risk",
    "teacher_overlap_rerank_weight": float("${TEACHER_OVERLAP_RERANK_WEIGHT}"),
    "segment_ks": [8, 16, 32],
    "train_segments_per_k": int("${TRAIN_PER_K}"),
    "val_segments_per_k": int("${VAL_PER_K}"),
    "files": {},
}
for name in [
    "metrics.json",
    "eval_time_alignment_final.json",
    "eval_long_energy_topk128_20.json",
]:
    path = save_dir / name
    if not path.exists():
        continue
    data = json.loads(path.read_text())
    item = {}
    for key in [
        "dataset",
        "val",
        "reward_sum",
        "tau_expected",
        "cumulative",
        "chosen_k_histogram",
        "completed_rollout_segments",
        "requested_rollout_segments",
        "stop_reason",
        "skipped_noop",
    ]:
        if key in data:
            item[key] = data[key]
    summary["files"][name] = item
(save_dir / "stage_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
print(f"Saved summary to {save_dir / 'stage_summary.json'}")
PY

echo "=== ${STAGE} DONE $(date -Is) ===" | tee -a "${LOG}"
