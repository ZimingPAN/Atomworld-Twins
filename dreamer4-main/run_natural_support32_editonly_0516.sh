#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
GPU_ID="${GPU_ID:-2}"
DEVICE="${DEVICE:-cuda}"
RESULT_ROOT="${RESULT_ROOT:-results/natural_teacher_support32_editonly_0516}"
STAGE="${STAGE:-v65_teacheraug_editonly_guard}"
SAVE_DIR="${RESULT_ROOT}/${STAGE}"
INIT_CKPT="${INIT_CKPT:-results/natural_teacher_support32_nooprisk_0516/v63_nooprisk_head_plannerpenalty/final_model.pt}"
PLANNER_CKPT="${PLANNER_CKPT:-results/natural_teacher_support32_nooprisk_0516/v63_nooprisk_head_plannerpenalty/final_model.pt}"
CACHE="${CACHE:-results/natural_teacher_support32_dualsupport_0516/v64_teacheraug_positive_riskguard/segments.pt}"
EPOCHS="${EPOCHS:-4}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LR="${LR:-3e-6}"
ROLLOUT_SEGMENTS="${ROLLOUT_SEGMENTS:-80}"
TAU_WEIGHT="${TAU_WEIGHT:-0.5}"
TAU_LOG_MU_WEIGHT="${TAU_LOG_MU_WEIGHT:-0.25}"
REALIZED_TAU_WEIGHT="${REALIZED_TAU_WEIGHT:-0.02}"
REWARD_WEIGHT="${REWARD_WEIGHT:-0.4}"
PRIOR_REWARD_WEIGHT="${PRIOR_REWARD_WEIGHT:-0.4}"
NOOP_RISK_WEIGHT="${NOOP_RISK_WEIGHT:-0.1}"
PRIOR_NOOP_RISK_WEIGHT="${PRIOR_NOOP_RISK_WEIGHT:-0.1}"
PROJ_WEIGHT="${PROJ_WEIGHT:-0.25}"
LOG="${SAVE_DIR}/pipeline.log"

mkdir -p "${SAVE_DIR}"

{
  echo "=== ${STAGE} START $(date -Is) ==="
  echo "init=${INIT_CKPT}"
  echo "planner_signature=${PLANNER_CKPT}"
  echo "cache=${CACHE}"
  echo "fix=teacher-aug positive edit/type recovery with reward-duration-risk protected edit-head-only training"
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
  --max_segments_per_rollout 12 \
  --max_seed_vacancies 32 \
  --max_candidate_sites 2048 \
  --teacher_candidate_neighbor_depth 1 \
  --teacher_path_summary_mode stepwise \
  --tau_supervision_mode prior_main \
  --reward_prediction_source projected \
  --include_noop_segments \
  --keep_after_noop_segments \
  --train_edit_heads_only \
  --tau_weight "${TAU_WEIGHT}" \
  --realized_tau_weight "${REALIZED_TAU_WEIGHT}" \
  --tau_log_mu_weight "${TAU_LOG_MU_WEIGHT}" \
  --reward_weight "${REWARD_WEIGHT}" \
  --prior_reward_weight "${PRIOR_REWARD_WEIGHT}" \
  --noop_risk_weight "${NOOP_RISK_WEIGHT}" \
  --prior_noop_risk_weight "${PRIOR_NOOP_RISK_WEIGHT}" \
  --path_weight 0.0 \
  --prior_latent_weight 0.0 \
  --count_loss_weight 0.8 \
  --mask_sparsity_weight 0.008 \
  --noop_change_weight 0.25 \
  --noop_type_copy_weight 0.10 \
  --projected_noop_fp_weight 0.50 \
  --pair_weight 0.35 \
  --prior_pair_weight 0.35 \
  --prior_edit_weight 0.90 \
  --proj_weight "${PROJ_WEIGHT}" \
  --no_aux_anneal \
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
  --planner_segment_ks 8 16 32 \
  --rollout_segments "${ROLLOUT_SEGMENTS}" \
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
  --rollout_segments "${ROLLOUT_SEGMENTS}" \
  --max_episode_steps_override 1024 \
  --duration_source model \
  --planner_tau_source model \
  --planner_score_mode energy \
  --planner_noop_risk_penalty 1.0 \
  --min_projected_changed_sites 2 \
  --device "${DEVICE}" \
  --output "${SAVE_DIR}/eval_long_energy_riskpen1.json" \
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
    "segment_ks": [8, 16, 32],
    "boundary": "adaptive_key_event",
    "support": "teacher_path_augmented_positives_reused_from_v64",
    "training_mode": "edit_heads_only",
    "reward_duration_risk_protected": True,
    "anchor_weights": {
        "tau": float("${TAU_WEIGHT}"),
        "tau_log_mu": float("${TAU_LOG_MU_WEIGHT}"),
        "realized_tau": float("${REALIZED_TAU_WEIGHT}"),
        "reward": float("${REWARD_WEIGHT}"),
        "prior_reward": float("${PRIOR_REWARD_WEIGHT}"),
        "noop_risk": float("${NOOP_RISK_WEIGHT}"),
        "prior_noop_risk": float("${PRIOR_NOOP_RISK_WEIGHT}"),
        "proj": float("${PROJ_WEIGHT}"),
    },
    "max_candidate_sites": 2048,
    "teacher_candidate_augmentation": True,
    "teacher_candidate_neighbor_depth": 1,
    "rollout_segments": int("${ROLLOUT_SEGMENTS}"),
    "files": {},
}
for name in [
    "metrics.json",
    "eval_time_alignment_final.json",
    "eval_long_trajectory.json",
    "eval_long_energy_riskpen1.json",
]:
    path = save_dir / name
    if not path.exists():
        continue
    data = json.loads(path.read_text())
    item = {}
    if "val" in data:
        item["val"] = data["val"]
    if "dataset" in data:
        item["dataset"] = data["dataset"]
    if "noop_risk" in data:
        item["noop_risk"] = data["noop_risk"]
    for key in [
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
    ]:
        if key in data:
            item[key] = data[key]
    summary["files"][name] = item
(save_dir / "stage_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
print(f"Saved summary to {save_dir / 'stage_summary.json'}")
PY

echo "=== ${STAGE} DONE $(date -Is) ===" | tee -a "${LOG}"
