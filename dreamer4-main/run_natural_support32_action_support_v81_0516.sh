#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-cuda}"
RESULT_ROOT="${RESULT_ROOT:-results/natural_teacher_support32_actionproposal_0516}"
STAGE="${STAGE:-v81_independent_action_support_smoke1b}"
SAVE_DIR="${RESULT_ROOT}/${STAGE}"
INIT_CKPT="${INIT_CKPT:-results/natural_teacher_support32_touchedproposal_0516/v80_actionrank_smoke1/final_model.pt}"
PLANNER_CKPT="${PLANNER_CKPT:-results/natural_teacher_support32_touchedproposal_0516/v80_actionrank_smoke1/final_model.pt}"
PROTECTED_CKPT="${PROTECTED_CKPT:-results/natural_teacher_support32_nooprisk_0516/v63_nooprisk_head_plannerpenalty/final_model.pt}"
SOURCE_CACHE="${SOURCE_CACHE:-results/natural_teacher_support32_touchedproposal_0516/v80_actionrank_smoke1/segments.pt}"
CACHE="${CACHE:-${SAVE_DIR}/segments.pt}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LR="${LR:-5e-5}"
ACTION_SUPPORT_TARGET_SOURCE="${ACTION_SUPPORT_TARGET_SOURCE:-changed_or_touched}"
ACTION_SUPPORT_HARD_NEGATIVE_WEIGHT="${ACTION_SUPPORT_HARD_NEGATIVE_WEIGHT:-0.5}"
ACTION_SUPPORT_RANK_MARGIN_WEIGHT="${ACTION_SUPPORT_RANK_MARGIN_WEIGHT:-0.25}"
ACTION_SUPPORT_CANDIDATE_POSITIVE_WEIGHT="${ACTION_SUPPORT_CANDIDATE_POSITIVE_WEIGHT:-1.0}"
ACTION_SUPPORT_CANDIDATE_NEGATIVE_WEIGHT="${ACTION_SUPPORT_CANDIDATE_NEGATIVE_WEIGHT:-1.0}"
ACTION_SUPPORT_CANDIDATE_RANK_MARGIN_WEIGHT="${ACTION_SUPPORT_CANDIDATE_RANK_MARGIN_WEIGHT:-0.25}"
LOG="${SAVE_DIR}/pipeline.log"

mkdir -p "${SAVE_DIR}"

{
  echo "=== ${STAGE} START $(date -Is) ==="
  echo "init=${INIT_CKPT}"
  echo "planner=${PLANNER_CKPT}"
  echo "protected_reward_tau=${PROTECTED_CKPT}"
  echo "cache=${CACHE}"
  echo "action_support_target_source=${ACTION_SUPPORT_TARGET_SOURCE}"
  echo "action_support_hard_negative_weight=${ACTION_SUPPORT_HARD_NEGATIVE_WEIGHT}"
  echo "action_support_rank_margin_weight=${ACTION_SUPPORT_RANK_MARGIN_WEIGHT}"
  echo "action_support_candidate_positive_weight=${ACTION_SUPPORT_CANDIDATE_POSITIVE_WEIGHT}"
  echo "action_support_candidate_negative_weight=${ACTION_SUPPORT_CANDIDATE_NEGATIVE_WEIGHT}"
  echo "action_support_candidate_rank_margin_weight=${ACTION_SUPPORT_CANDIDATE_RANK_MARGIN_WEIGHT}"
  echo "fix=independent action_support_head initialized from trained proposal_head, with old proposal logits frozen"
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
  --planner_selected_projection_topk_budget 96 \
  --planner_selected_noop_risk_penalty 1.0 \
  --planner_selected_teacher_overlap_rerank_weight 0.5 \
  --planner_selected_store_candidate_overlap_masks \
  --planner_selected_min_projected_changed_sites 2 \
  --planner_selected_duration_source model \
  --planner_selected_tau_source model \
  --planner_selected_score_mode energy \
  --planner_selected_reward_prediction_source projected \
  --segment_ks 8 16 32 \
  --train_segments_per_k 4 \
  --val_segments_per_k 2 \
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
  --train_action_support_head_only \
  --init_action_support_from_proposal_head \
  --action_support_target_source "${ACTION_SUPPORT_TARGET_SOURCE}" \
  --action_support_weight 1.0 \
  --prior_action_support_weight 1.0 \
  --action_support_hard_negative_weight "${ACTION_SUPPORT_HARD_NEGATIVE_WEIGHT}" \
  --action_support_rank_margin_weight "${ACTION_SUPPORT_RANK_MARGIN_WEIGHT}" \
  --action_support_candidate_positive_weight "${ACTION_SUPPORT_CANDIDATE_POSITIVE_WEIGHT}" \
  --action_support_candidate_negative_weight "${ACTION_SUPPORT_CANDIDATE_NEGATIVE_WEIGHT}" \
  --action_support_candidate_rank_margin_weight "${ACTION_SUPPORT_CANDIDATE_RANK_MARGIN_WEIGHT}" \
  --proposal_support_weight 0.0 \
  --prior_proposal_support_weight 0.0 \
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

for B in 96 128; do
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
    --planner_projection_change_source action_support \
    --planner_projection_topk_source action_support \
    --planner_projection_topk_budget "${B}" \
    --proposal_diagnostic \
    --min_projected_changed_sites 2 \
    --device "${DEVICE}" \
    --output "${SAVE_DIR}/eval_long_action_as_change_topk${B}_20.json" \
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
    --planner_projection_change_source change \
    --planner_projection_topk_source action_support \
    --planner_projection_topk_budget "${B}" \
    --proposal_diagnostic \
    --min_projected_changed_sites 2 \
    --device "${DEVICE}" \
    --output "${SAVE_DIR}/eval_long_change_inside_action_gate_topk${B}_20.json" \
    2>&1 | tee -a "${LOG}"
done

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
    "action_support_target_source": "${ACTION_SUPPORT_TARGET_SOURCE}",
    "action_support_hard_negative_weight": float("${ACTION_SUPPORT_HARD_NEGATIVE_WEIGHT}"),
    "action_support_rank_margin_weight": float("${ACTION_SUPPORT_RANK_MARGIN_WEIGHT}"),
    "action_support_candidate_positive_weight": float("${ACTION_SUPPORT_CANDIDATE_POSITIVE_WEIGHT}"),
    "action_support_candidate_negative_weight": float("${ACTION_SUPPORT_CANDIDATE_NEGATIVE_WEIGHT}"),
    "action_support_candidate_rank_margin_weight": float("${ACTION_SUPPORT_CANDIDATE_RANK_MARGIN_WEIGHT}"),
    "segment_ks": [8, 16, 32],
    "files": {},
}
for path in sorted(save_dir.glob("*.json")):
    if path.name == "stage_summary.json":
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
    if "segments" in data:
        overlaps = []
        projected_counts = []
        teacher_counts = []
        for seg in data.get("segments", []):
            po = seg.get("proposal_overlap")
            if isinstance(po, dict):
                overlaps.append(float(po.get("f1", 0.0)))
            if "projected_changed_count" in seg:
                projected_counts.append(float(seg["projected_changed_count"]))
            if "traditional_changed_site_count" in seg:
                teacher_counts.append(float(seg["traditional_changed_site_count"]))
        item["site_f1_mean"] = sum(overlaps) / len(overlaps) if overlaps else None
        item["projected_changed_count_mean"] = sum(projected_counts) / len(projected_counts) if projected_counts else None
        item["teacher_changed_site_count_mean"] = sum(teacher_counts) / len(teacher_counts) if teacher_counts else None
    summary["files"][path.name] = item
(save_dir / "stage_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
print(f"Saved summary to {save_dir / 'stage_summary.json'}")
PY

echo "=== ${STAGE} DONE $(date -Is) ===" | tee -a "${LOG}"
