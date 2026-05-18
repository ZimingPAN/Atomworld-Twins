#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-cuda}"
RESULT_ROOT="${RESULT_ROOT:-results/largek_candidate_rank_0516}"
STAGE="${STAGE:-v59_candidate_overlaprank_smoke}"
SAVE_DIR="${RESULT_ROOT}/${STAGE}"
INIT_CKPT="${INIT_CKPT:-results/largek_proposalhead_0516/v58_v57cache_proposal_topk128_sigfix/final_model.pt}"
PLANNER_CKPT="${PLANNER_CKPT:-results/largek_proposalhead_0516/v58_v57cache_proposal_topk128_sigfix/final_model.pt}"
CACHE="${CACHE:-${SAVE_DIR}/segments.pt}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LR="${LR:-5e-5}"
TRAIN_PER_K="${TRAIN_PER_K:-4}"
VAL_PER_K="${VAL_PER_K:-2}"
PROPOSAL_TOPK="${PROPOSAL_TOPK:-256}"
TEACHER_OVERLAP_RERANK_WEIGHT="${TEACHER_OVERLAP_RERANK_WEIGHT:-0.5}"
PROPOSAL_CANDIDATE_POSITIVE_WEIGHT="${PROPOSAL_CANDIDATE_POSITIVE_WEIGHT:-1.0}"
PROPOSAL_CANDIDATE_NEGATIVE_WEIGHT="${PROPOSAL_CANDIDATE_NEGATIVE_WEIGHT:-1.0}"
PROPOSAL_CANDIDATE_RANK_MARGIN_WEIGHT="${PROPOSAL_CANDIDATE_RANK_MARGIN_WEIGHT:-0.25}"
PROPOSAL_HARD_NEGATIVE_WEIGHT="${PROPOSAL_HARD_NEGATIVE_WEIGHT:-0.5}"
PROPOSAL_RANK_MARGIN_WEIGHT="${PROPOSAL_RANK_MARGIN_WEIGHT:-0.1}"
LOG="${SAVE_DIR}/pipeline.log"

mkdir -p "${SAVE_DIR}"

{
  echo "=== ${STAGE} START $(date -Is) ==="
  echo "init=${INIT_CKPT}"
  echo "planner=${PLANNER_CKPT}"
  echo "cache=${CACHE}"
  echo "proposal_topk=${PROPOSAL_TOPK}"
  echo "teacher_overlap_rerank_weight=${TEACHER_OVERLAP_RERANK_WEIGHT}"
  echo "proposal_candidate_positive_weight=${PROPOSAL_CANDIDATE_POSITIVE_WEIGHT}"
  echo "proposal_candidate_negative_weight=${PROPOSAL_CANDIDATE_NEGATIVE_WEIGHT}"
  echo "proposal_candidate_rank_margin_weight=${PROPOSAL_CANDIDATE_RANK_MARGIN_WEIGHT}"
  echo "fix=large-k candidate-level support ranking smoke: teacher-overlap positives plus projected false-positive hard negatives"
} | tee -a "${LOG}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" train_dreamer_macro_edit.py \
  --save_dir "${SAVE_DIR}" \
  --dataset_cache "${CACHE}" \
  --init_from "${INIT_CKPT}" \
  --planner_selected_from "${PLANNER_CKPT}" \
  --planner_selected_projection_change_source proposal \
  --planner_selected_projection_topk_source proposal \
  --planner_selected_projection_topk_budget "${PROPOSAL_TOPK}" \
  --planner_selected_noop_risk_penalty 1.0 \
  --planner_selected_teacher_overlap_rerank_weight "${TEACHER_OVERLAP_RERANK_WEIGHT}" \
  --planner_selected_store_candidate_overlap_masks \
  --planner_selected_min_projected_changed_sites 2 \
  --planner_selected_duration_source baseline \
  --planner_selected_tau_source baseline \
  --planner_selected_score_mode energy_per_tau \
  --planner_selected_reward_prediction_source projected \
  --segment_ks 128 256 512 1024 \
  --train_segments_per_k "${TRAIN_PER_K}" \
  --val_segments_per_k "${VAL_PER_K}" \
  --max_episode_steps 4096 \
  --max_segments_per_rollout 4 \
  --max_seed_vacancies 8 \
  --max_candidate_sites 1024 \
  --teacher_candidate_neighbor_depth 1 \
  --teacher_path_summary_mode stepwise \
  --tau_supervision_mode prior_main \
  --reward_prediction_source projected \
  --include_noop_segments \
  --keep_after_noop_segments \
  --train_proposal_head_only \
  --proposal_support_weight 1.0 \
  --prior_proposal_support_weight 1.0 \
  --proposal_hard_negative_weight "${PROPOSAL_HARD_NEGATIVE_WEIGHT}" \
  --proposal_rank_margin_weight "${PROPOSAL_RANK_MARGIN_WEIGHT}" \
  --proposal_candidate_positive_weight "${PROPOSAL_CANDIDATE_POSITIVE_WEIGHT}" \
  --proposal_candidate_negative_weight "${PROPOSAL_CANDIDATE_NEGATIVE_WEIGHT}" \
  --proposal_candidate_rank_margin_weight "${PROPOSAL_CANDIDATE_RANK_MARGIN_WEIGHT}" \
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
  --noop_risk_weight 0.0 \
  --prior_noop_risk_weight 0.0 \
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
  --planner_noop_risk_penalty 1.0 \
  --planner_projection_change_source proposal \
  --planner_projection_topk_source proposal \
  --planner_projection_topk_budget "${PROPOSAL_TOPK}" \
  --proposal_diagnostic \
  --proposal_diagnostic_max_sites 64 \
  --min_projected_changed_sites 2 \
  --device "${DEVICE}" \
  --output "${SAVE_DIR}/eval_long_candidate_rank_topk256_modeltau_riskpen1.json" \
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
  --planner_projection_change_source proposal \
  --planner_projection_topk_source proposal \
  --planner_projection_topk_budget "${PROPOSAL_TOPK}" \
  --proposal_diagnostic \
  --proposal_diagnostic_max_sites 64 \
  --min_projected_changed_sites 2 \
  --device "${DEVICE}" \
  --output "${SAVE_DIR}/eval_long_candidate_rank_topk256_baseline_min2_riskpen1.json" \
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
  --planner_projection_change_source proposal \
  --planner_projection_topk_source proposal \
  --planner_projection_topk_budget "${PROPOSAL_TOPK}" \
  --proposal_diagnostic \
  --proposal_diagnostic_max_sites 64 \
  --min_projected_changed_sites 2 \
  --allow_teacher_noop_segments \
  --device "${DEVICE}" \
  --output "${SAVE_DIR}/eval_long_candidate_rank_topk256_baseline_min2_riskpen1_allownoop20.json" \
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
    "segment_ks": [128, 256, 512, 1024],
    "train_segments_per_k": int("${TRAIN_PER_K}"),
    "val_segments_per_k": int("${VAL_PER_K}"),
    "proposal_topk": int("${PROPOSAL_TOPK}"),
    "teacher_overlap_rerank_weight": float("${TEACHER_OVERLAP_RERANK_WEIGHT}"),
    "proposal_candidate_positive_weight": float("${PROPOSAL_CANDIDATE_POSITIVE_WEIGHT}"),
    "proposal_candidate_negative_weight": float("${PROPOSAL_CANDIDATE_NEGATIVE_WEIGHT}"),
    "proposal_candidate_rank_margin_weight": float("${PROPOSAL_CANDIDATE_RANK_MARGIN_WEIGHT}"),
    "files": {},
}
for name in [
    "metrics.json",
    "eval_time_alignment_final.json",
    "eval_long_candidate_rank_topk256_modeltau_riskpen1.json",
    "eval_long_candidate_rank_topk256_baseline_min2_riskpen1.json",
    "eval_long_candidate_rank_topk256_baseline_min2_riskpen1_allownoop20.json",
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
        "noop_risk",
        "cumulative",
        "chosen_k_histogram",
        "completed_rollout_segments",
        "requested_rollout_segments",
        "stop_reason",
        "skipped_noop",
        "duration_source",
        "planner_tau_source",
        "planner_score_mode",
        "planner_projection_change_source",
        "planner_projection_topk_source",
        "planner_projection_topk_budget",
        "effective_min_projected_changed_sites",
    ]:
        if key in data:
            item[key] = data[key]
    summary["files"][name] = item
(save_dir / "stage_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
print(f"Saved summary to {save_dir / 'stage_summary.json'}")
PY

echo "=== ${STAGE} DONE $(date -Is) ===" | tee -a "${LOG}"
