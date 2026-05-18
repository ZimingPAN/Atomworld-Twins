#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="/home/likun-share/panziming/pydeps:/home/likun-share/panziming/AtomWorld-Mirror/kmcteacher_backend:/home/likun-share/panziming/AtomWorld-Mirror/LightZero-main:/home/likun-share/panziming/AtomWorld-Mirror/dreamer4-main:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-cuda}"
RESULT_ROOT="${RESULT_ROOT:-results/natural_teacher_support32_sequence_rollout_0517}"
STAGE="${STAGE:-v95_terminal_typed_diff_support_smoke1}"
SAVE_DIR="${RESULT_ROOT}/${STAGE}"
INIT_CKPT="${INIT_CKPT:-results/natural_teacher_support32_sequence_rollout_0517/v90_sequence_rollout_residual_smoke1/final_model.pt}"
PLANNER_CKPT="${PLANNER_CKPT:-results/natural_teacher_support32_sequence_rollout_0517/v90_sequence_rollout_residual_smoke1/final_model.pt}"
PROTECTED_CKPT="${PROTECTED_CKPT:-results/natural_teacher_support32_nooprisk_0516/v63_nooprisk_head_plannerpenalty/final_model.pt}"
CACHE="${CACHE:-results/natural_teacher_support32_sequence_rollout_0517/v90_sequence_rollout_residual_smoke1/segments_v17.pt}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LR="${LR:-8e-5}"
ROLLOUT_SEGMENTS="${ROLLOUT_SEGMENTS:-20}"
EDGE_MULTI_TYPE_WEIGHT="${EDGE_MULTI_TYPE_WEIGHT:-0.15}"
EDGE_MULTI_ORDER_WEIGHT="${EDGE_MULTI_ORDER_WEIGHT:-0.10}"
LOG="${SAVE_DIR}/pipeline.log"

mkdir -p "${SAVE_DIR}"

{
  echo "=== ${STAGE} START $(date -Is) ==="
  echo "init=${INIT_CKPT}"
  echo "planner=${PLANNER_CKPT}"
  echo "protected_reward_tau=${PROTECTED_CKPT}"
  echo "cache=${CACHE}"
  echo "rollout_segments=${ROLLOUT_SEGMENTS}"
  echo "fix=v95 terminal typed-diff decoder with explicit vacancy-displacement support loss"
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
  --planner_selected_projection_change_source action_endpoint \
  --planner_selected_projection_topk_source action_endpoint \
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
  --train_terminal_typed_diff_head_only \
  --terminal_edit_action_context_source action_endpoint \
  --terminal_typed_diff_weight 1.0 \
  --prior_terminal_typed_diff_weight 1.0 \
  --terminal_typed_diff_copy_weight 0.02 \
  --terminal_typed_diff_support_weight 1.0 \
  --proposal_support_weight 0.0 \
  --prior_proposal_support_weight 0.0 \
  --action_support_weight 0.0 \
  --prior_action_support_weight 0.0 \
  --terminal_edit_support_weight 0.0 \
  --prior_terminal_edit_support_weight 0.0 \
  --action_source_support_weight 0.0 \
  --prior_action_source_support_weight 0.0 \
  --action_destination_support_weight 0.0 \
  --prior_action_destination_support_weight 0.0 \
  --action_edge_pair_weight 0.0 \
  --prior_action_edge_pair_weight 0.0 \
  --action_edge_pair_support_weight 0.0 \
  --prior_action_edge_pair_support_weight 0.0 \
  --action_edge_pair_semantic_weight 0.0 \
  --prior_action_edge_pair_semantic_weight 0.0 \
  --candidate_quality_weight 0.0 \
  --prior_candidate_quality_weight 0.0 \
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

run_eval() {
  local name="$1"
  local change_source="$2"
  local topk_source="$3"
  local topk_budget="$4"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" eval_macro_long_trajectory.py \
    --checkpoint "${SAVE_DIR}/final_model.pt" \
    --reward_checkpoint "${PROTECTED_CKPT}" \
    --duration_checkpoint "${PROTECTED_CKPT}" \
    --aux_projected_types_source primary \
    --planner_segment_ks 8 16 32 \
    --rollout_segments "${ROLLOUT_SEGMENTS}" \
    --max_episode_steps_override 1024 \
    --duration_source model \
    --planner_tau_source model \
    --planner_score_mode energy \
    --planner_noop_risk_penalty 1.0 \
    --planner_projection_change_source "${change_source}" \
    --planner_projection_topk_source "${topk_source}" \
    --planner_projection_topk_budget "${topk_budget}" \
    --planner_edge_completion_anchor_source action_source \
    --planner_edge_completion_anchor_budget 32 \
    --planner_edge_completion_destinations_per_anchor 4 \
    --planner_projection_change_blend_alpha 0.5 \
    --planner_edge_pair_multiobjective_type_weight "${EDGE_MULTI_TYPE_WEIGHT}" \
    --planner_edge_pair_multiobjective_order_weight "${EDGE_MULTI_ORDER_WEIGHT}" \
    --proposal_diagnostic \
    --proposal_diagnostic_max_sites 384 \
    --min_projected_changed_sites 2 \
    --device "${DEVICE}" \
    --output "${SAVE_DIR}/eval_long_${name}_${ROLLOUT_SEGMENTS}.json" \
    2>&1 | tee -a "${LOG}"
}

run_eval "terminal_typed_diff_direct" "terminal_typed_diff" "none" 0
run_eval "terminal_typed_diff_endpoint_gate96" "terminal_typed_diff" "action_endpoint" 96
run_eval "terminal_typed_diff_endpoint_gate128" "terminal_typed_diff" "action_endpoint" 128
run_eval "terminal_typed_diff_twostage_gate96" "terminal_typed_diff" "two_stage_vacancy_displacement" 96
run_eval "edgepair_multi_inside_typed_gate96" "action_edge_pair_multiobjective_completion" "terminal_typed_diff" 96

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
    "rollout_segments": int("${ROLLOUT_SEGMENTS}"),
    "terminal_typed_diff_target": "vacancy_displacement",
    "terminal_typed_diff_support_weight": 1.0,
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
        "planner_projection_change_source",
        "planner_projection_topk_source",
        "planner_projection_topk_budget",
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
                projected_counts.append(float(po.get("projected_changed_count", 0.0)))
                teacher_counts.append(float(po.get("teacher_changed_count", 0.0)))
        if overlaps:
            item["selected_site_f1_mean"] = sum(overlaps) / len(overlaps)
            item["projected_changed_count_mean"] = sum(projected_counts) / max(len(projected_counts), 1)
            item["teacher_changed_count_mean"] = sum(teacher_counts) / max(len(teacher_counts), 1)
    summary["files"][path.name] = item
save_dir.joinpath("stage_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
print(json.dumps(summary, indent=2, sort_keys=True))
PY

echo "=== ${STAGE} DONE $(date -Is) ===" | tee -a "${LOG}"
