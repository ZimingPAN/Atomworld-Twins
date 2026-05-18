#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="/home/likun-share/panziming/pydeps:/home/likun-share/panziming/AtomWorld-Mirror/kmcteacher_backend:/home/likun-share/panziming/AtomWorld-Mirror/LightZero-main:/home/likun-share/panziming/AtomWorld-Mirror/dreamer4-main:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-cuda}"
RESULT_ROOT="${RESULT_ROOT:-results/natural_teacher_support32_sequence_rollout_0517}"
STAGE="${STAGE:-v93_two_stage_vacancy_displacement_smoke20}"
SAVE_DIR="${RESULT_ROOT}/${STAGE}"
CHECKPOINT="${CHECKPOINT:-results/natural_teacher_support32_sequence_rollout_0517/v90_sequence_rollout_residual_smoke1/final_model.pt}"
if [ ! -f "${CHECKPOINT}" ]; then
  CHECKPOINT="results/natural_teacher_support32_terminaldecouple_0517/v89_terminal_sparse_edit_decoupled_smoke1/final_model.pt"
fi
PROTECTED_CKPT="${PROTECTED_CKPT:-results/natural_teacher_support32_nooprisk_0516/v63_nooprisk_head_plannerpenalty/final_model.pt}"
EDGE_MULTI_TYPE_WEIGHT="${EDGE_MULTI_TYPE_WEIGHT:-0.15}"
EDGE_MULTI_ORDER_WEIGHT="${EDGE_MULTI_ORDER_WEIGHT:-0.10}"
ROLLOUT_SEGMENTS="${ROLLOUT_SEGMENTS:-20}"
LOG="${SAVE_DIR}/pipeline.log"

mkdir -p "${SAVE_DIR}"

{
  echo "=== ${STAGE} START $(date -Is) ==="
  echo "checkpoint=${CHECKPOINT}"
  echo "protected_reward_tau=${PROTECTED_CKPT}"
  echo "rollout_segments=${ROLLOUT_SEGMENTS}"
  echo "fix=v93 eval-only two-stage support gate + typed vacancy-displacement projection"
} | tee -a "${LOG}"

run_eval() {
  local name="$1"
  local change_source="$2"
  local topk_source="$3"
  local topk_budget="$4"
  local anchor_source="$5"
  local anchor_budget="$6"
  local dest_per_anchor="$7"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" eval_macro_long_trajectory.py \
    --checkpoint "${CHECKPOINT}" \
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
    --planner_edge_completion_anchor_source "${anchor_source}" \
    --planner_edge_completion_anchor_budget "${anchor_budget}" \
    --planner_edge_completion_destinations_per_anchor "${dest_per_anchor}" \
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

run_eval "twostage_vacdisp_a16_d4" "two_stage_vacancy_displacement" "none" 0 "action_source" 16 4
run_eval "twostage_vacdisp_a32_d4" "two_stage_vacancy_displacement" "none" 0 "action_source" 32 4
run_eval "twostage_vacdisp_a32_d8" "two_stage_vacancy_displacement" "none" 0 "action_source" 32 8
run_eval "change_inside_twostage_gate_topk96" "change" "two_stage_vacancy_displacement" 96 "action_source" 32 4
run_eval "edgepair_multi_inside_twostage_gate_topk96" "action_edge_pair_multiobjective_completion" "two_stage_vacancy_displacement" 96 "action_source" 32 4

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

save_dir = Path("${SAVE_DIR}")
summary = {
    "stage": "${STAGE}",
    "checkpoint": "${CHECKPOINT}",
    "protected_reward_tau_checkpoint": "${PROTECTED_CKPT}",
    "rollout_segments": int("${ROLLOUT_SEGMENTS}"),
    "two_stage_projection": "support gate plus typed vacancy-displacement final diff",
    "files": {},
}
for path in sorted(save_dir.glob("*.json")):
    if path.name == "stage_summary.json":
        continue
    data = json.loads(path.read_text())
    item = {}
    for key in [
        "completed_rollout_segments",
        "requested_rollout_segments",
        "stop_reason",
        "chosen_k_histogram",
        "planner_projection_change_source",
        "planner_projection_topk_source",
        "planner_projection_topk_budget",
        "cumulative",
        "tau_expected",
        "reward_sum",
    ]:
        if key in data:
            item[key] = data[key]
    overlaps = []
    projected_counts = []
    teacher_counts = []
    edge_counts = []
    for seg in data.get("segments", []):
        overlap = seg.get("proposal_overlap")
        if isinstance(overlap, dict):
            overlaps.append(float(overlap.get("f1", 0.0)))
            projected_counts.append(float(overlap.get("projected_changed_count", 0.0)))
            teacher_counts.append(float(overlap.get("teacher_changed_count", 0.0)))
        edge_counts.append(float(seg.get("planner_edge_completion_support_count", 0.0)))
    if overlaps:
        item["selected_site_f1_mean"] = sum(overlaps) / len(overlaps)
        item["projected_changed_count_mean"] = sum(projected_counts) / max(len(projected_counts), 1)
        item["teacher_changed_count_mean"] = sum(teacher_counts) / max(len(teacher_counts), 1)
    if edge_counts:
        item["edge_completion_support_count_mean"] = sum(edge_counts) / len(edge_counts)
    summary["files"][path.name] = item
save_dir.joinpath("stage_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
print(json.dumps(summary, indent=2, sort_keys=True))
PY

echo "=== ${STAGE} DONE $(date -Is) ===" | tee -a "${LOG}"
