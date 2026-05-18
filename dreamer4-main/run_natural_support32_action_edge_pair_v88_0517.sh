#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="/home/likun-share/panziming/pydeps:/home/likun-share/panziming/AtomWorld-Mirror/kmcteacher_backend:/home/likun-share/panziming/AtomWorld-Mirror/LightZero-main:/home/likun-share/panziming/AtomWorld-Mirror/dreamer4-main:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-cuda}"
RESULT_ROOT="${RESULT_ROOT:-results/natural_teacher_support32_actionendpoint_0517}"
STAGE="${STAGE:-v88_edge_pair_multiobjective_smoke1}"
SAVE_DIR="${RESULT_ROOT}/${STAGE}"
INIT_CKPT="${INIT_CKPT:-results/natural_teacher_support32_actionendpoint_0517/v87_edge_pair_support_listwise_smoke1/final_model.pt}"
if [ ! -f "${INIT_CKPT}" ]; then
  INIT_CKPT="results/natural_teacher_support32_actionendpoint_0517/v87_edge_pair_support_split_smoke1/final_model.pt"
fi
if [ ! -f "${INIT_CKPT}" ]; then
  INIT_CKPT="results/natural_teacher_support32_actionendpoint_0517/v86_action_edge_pair_samesource_neg_smoke1/final_model.pt"
fi
PLANNER_CKPT="${PLANNER_CKPT:-results/natural_teacher_support32_actionendpoint_0517/v83_action_endpoint_smoke1b/final_model.pt}"
PROTECTED_CKPT="${PROTECTED_CKPT:-results/natural_teacher_support32_nooprisk_0516/v63_nooprisk_head_plannerpenalty/final_model.pt}"
CACHE="${CACHE:-results/natural_teacher_support32_actionendpoint_0517/v87_edge_pair_support_listwise_smoke1/segments_v16.pt}"
if [ ! -f "${CACHE}" ]; then
  CACHE="${SAVE_DIR}/segments_v17.pt"
fi
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LR="${LR:-8e-5}"
ACTION_EDGE_PAIR_NEGATIVE_MODE="${ACTION_EDGE_PAIR_NEGATIVE_MODE:-same_source_nn1}"
ACTION_EDGE_PAIR_NEGATIVE_COUNT="${ACTION_EDGE_PAIR_NEGATIVE_COUNT:-4}"
ACTION_EDGE_PAIR_NEGATIVE_WEIGHT="${ACTION_EDGE_PAIR_NEGATIVE_WEIGHT:-2.0}"
ACTION_EDGE_PAIR_RANK_MARGIN_WEIGHT="${ACTION_EDGE_PAIR_RANK_MARGIN_WEIGHT:-1.0}"
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
  echo "action_edge_pair_negative_mode=${ACTION_EDGE_PAIR_NEGATIVE_MODE}"
  echo "action_edge_pair_negative_count=${ACTION_EDGE_PAIR_NEGATIVE_COUNT}"
  echo "action_edge_pair_negative_weight=${ACTION_EDGE_PAIR_NEGATIVE_WEIGHT}"
  echo "action_edge_pair_rank_margin_weight=${ACTION_EDGE_PAIR_RANK_MARGIN_WEIGHT}"
  echo "edge_multi_type_weight=${EDGE_MULTI_TYPE_WEIGHT}"
  echo "edge_multi_order_weight=${EDGE_MULTI_ORDER_WEIGHT}"
  echo "fix=listwise multi-objective edge heads: energy/support/moving_type/order/candidate_quality"
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
  --train_action_edge_pair_listwise_heads_only \
  --action_edge_pair_weight 0.35 \
  --prior_action_edge_pair_weight 0.35 \
  --action_edge_pair_support_weight 0.35 \
  --prior_action_edge_pair_support_weight 0.35 \
  --action_edge_pair_semantic_weight 0.50 \
  --prior_action_edge_pair_semantic_weight 0.50 \
  --candidate_quality_weight 0.80 \
  --prior_candidate_quality_weight 0.80 \
  --action_edge_pair_negative_weight "${ACTION_EDGE_PAIR_NEGATIVE_WEIGHT}" \
  --action_edge_pair_rank_margin_weight "${ACTION_EDGE_PAIR_RANK_MARGIN_WEIGHT}" \
  --action_edge_pair_negative_mode "${ACTION_EDGE_PAIR_NEGATIVE_MODE}" \
  --action_edge_pair_negative_count "${ACTION_EDGE_PAIR_NEGATIVE_COUNT}" \
  --proposal_support_weight 0.0 \
  --prior_proposal_support_weight 0.0 \
  --action_support_weight 0.0 \
  --prior_action_support_weight 0.0 \
  --action_source_support_weight 0.0 \
  --prior_action_source_support_weight 0.0 \
  --action_destination_support_weight 0.0 \
  --prior_action_destination_support_weight 0.0 \
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
  local anchor_budget="$5"
  local dest_per_anchor="$6"
  local candq_weight="$7"
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
    --planner_projection_change_source "${change_source}" \
    --planner_projection_topk_source "${topk_source}" \
    --planner_projection_topk_budget "${topk_budget}" \
    --planner_edge_completion_anchor_source action_source \
    --planner_edge_completion_anchor_budget "${anchor_budget}" \
    --planner_edge_completion_destinations_per_anchor "${dest_per_anchor}" \
    --planner_projection_change_blend_alpha 0.5 \
    --planner_edge_pair_multiobjective_type_weight "${EDGE_MULTI_TYPE_WEIGHT}" \
    --planner_edge_pair_multiobjective_order_weight "${EDGE_MULTI_ORDER_WEIGHT}" \
    --planner_candidate_quality_score_weight "${candq_weight}" \
    --proposal_diagnostic \
    --proposal_diagnostic_max_sites 384 \
    --min_projected_changed_sites 2 \
    --device "${DEVICE}" \
    --output "${SAVE_DIR}/eval_long_${name}_20.json" \
    2>&1 | tee -a "${LOG}"
}

run_eval "edgepair_multi_as_change_a32_d4" "action_edge_pair_multiobjective_completion" "none" 0 32 4 0.0
run_eval "edgepair_multi_as_change_a32_d8" "action_edge_pair_multiobjective_completion" "none" 0 32 8 0.0
run_eval "edgepair_multi_as_change_a64_d4" "action_edge_pair_multiobjective_completion" "none" 0 64 4 0.0
run_eval "edgepair_multi_candq05_as_change_a32_d4" "action_edge_pair_multiobjective_completion" "none" 0 32 4 0.5
run_eval "change_inside_edgepair_multi_gate_a32_d4_topk96" "change" "action_edge_pair_multiobjective_completion" 96 32 4 0.0
run_eval "change_inside_edgepair_multi_gate_candq05_a32_d4_topk96" "change" "action_edge_pair_multiobjective_completion" 96 32 4 0.5

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
    "action_edge_pair_negative_mode": "${ACTION_EDGE_PAIR_NEGATIVE_MODE}",
    "action_edge_pair_negative_count": int("${ACTION_EDGE_PAIR_NEGATIVE_COUNT}"),
    "action_edge_pair_negative_weight": float("${ACTION_EDGE_PAIR_NEGATIVE_WEIGHT}"),
    "action_edge_pair_rank_margin_weight": float("${ACTION_EDGE_PAIR_RANK_MARGIN_WEIGHT}"),
    "edge_multi_type_weight": float("${EDGE_MULTI_TYPE_WEIGHT}"),
    "edge_multi_order_weight": float("${EDGE_MULTI_ORDER_WEIGHT}"),
    "listwise_heads": ["energy", "terminal_support", "moving_type", "path_order", "candidate_quality"],
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
        "planner_candidate_quality_score_weight",
    ]:
        if key in data:
            item[key] = data[key]
    if "segments" in data:
        overlaps = []
        projected_counts = []
        teacher_counts = []
        edge_counts = []
        delta_ratios = []
        time_ratios = []
        for seg in data.get("segments", []):
            po = seg.get("proposal_overlap")
            if isinstance(po, dict):
                overlaps.append(float(po.get("f1", 0.0)))
            if "projected_changed_count" in seg:
                projected_counts.append(float(seg["projected_changed_count"]))
            if "traditional_changed_site_count" in seg:
                teacher_counts.append(float(seg["traditional_changed_site_count"]))
            if "planner_edge_completion_support_count" in seg:
                edge_counts.append(float(seg["planner_edge_completion_support_count"]))
            if "delta_e_ratio" in seg:
                delta_ratios.append(float(seg["delta_e_ratio"]))
            if "expected_time_ratio" in seg:
                time_ratios.append(float(seg["expected_time_ratio"]))
        item["site_f1_mean"] = sum(overlaps) / len(overlaps) if overlaps else None
        item["projected_changed_count_mean"] = sum(projected_counts) / len(projected_counts) if projected_counts else None
        item["teacher_changed_site_count_mean"] = sum(teacher_counts) / len(teacher_counts) if teacher_counts else None
        item["edge_completion_support_count_mean"] = sum(edge_counts) / len(edge_counts) if edge_counts else None
        item["delta_e_ratio_mean"] = sum(delta_ratios) / len(delta_ratios) if delta_ratios else None
        item["expected_time_ratio_mean"] = sum(time_ratios) / len(time_ratios) if time_ratios else None
    summary["files"][path.name] = item
(save_dir / "stage_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
print(f"Saved summary to {save_dir / 'stage_summary.json'}")
PY

echo "=== ${STAGE} DONE $(date -Is) ===" | tee -a "${LOG}"
