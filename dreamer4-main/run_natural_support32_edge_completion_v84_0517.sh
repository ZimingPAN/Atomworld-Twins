#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="/home/likun-share/panziming/pydeps:/home/likun-share/panziming/AtomWorld-Mirror/kmcteacher_backend:/home/likun-share/panziming/AtomWorld-Mirror/LightZero-main:/home/likun-share/panziming/AtomWorld-Mirror/dreamer4-main:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-cuda}"
RESULT_ROOT="${RESULT_ROOT:-results/natural_teacher_support32_actionendpoint_0517}"
STAGE="${STAGE:-v84_edge_completion_upperbound20}"
SAVE_DIR="${RESULT_ROOT}/${STAGE}"
CHECKPOINT="${CHECKPOINT:-results/natural_teacher_support32_actionendpoint_0517/v83_action_endpoint_smoke1b/final_model.pt}"
PROTECTED_CKPT="${PROTECTED_CKPT:-results/natural_teacher_support32_nooprisk_0516/v63_nooprisk_head_plannerpenalty/final_model.pt}"
LOG="${SAVE_DIR}/pipeline.log"

mkdir -p "${SAVE_DIR}"

{
  echo "=== ${STAGE} START $(date -Is) ==="
  echo "checkpoint=${CHECKPOINT}"
  echo "protected_reward_tau=${PROTECTED_CKPT}"
  echo "diagnostic=edge completion from action_source anchors and action_destination NN1 endpoints"
} | tee -a "${LOG}"

run_eval() {
  local name="$1"
  local change_source="$2"
  local topk_source="$3"
  local topk_budget="$4"
  local anchor_budget="$5"
  local dest_per_anchor="$6"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" eval_macro_long_trajectory.py \
    --checkpoint "${CHECKPOINT}" \
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
    --planner_edge_completion_destination_source action_destination \
    --planner_edge_completion_anchor_budget "${anchor_budget}" \
    --planner_edge_completion_destinations_per_anchor "${dest_per_anchor}" \
    --proposal_diagnostic \
    --proposal_diagnostic_max_sites 384 \
    --min_projected_changed_sites 2 \
    --device "${DEVICE}" \
    --output "${SAVE_DIR}/eval_long_${name}_20.json" \
    2>&1 | tee -a "${LOG}"
}

run_eval "edge_as_change_a16_d8" "action_edge_completion" "none" 0 16 8
run_eval "edge_as_change_a32_d4" "action_edge_completion" "none" 0 32 4
run_eval "edge_as_change_a32_d8" "action_edge_completion" "none" 0 32 8
run_eval "change_inside_edge_gate_a32_d4_topk96" "change" "action_edge_completion" 96 32 4
run_eval "change_inside_edge_gate_a32_d4_topk128" "change" "action_edge_completion" 128 32 4

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

save_dir = Path("${SAVE_DIR}")
summary = {
    "stage": "${STAGE}",
    "checkpoint": "${CHECKPOINT}",
    "protected_reward_tau_checkpoint": "${PROTECTED_CKPT}",
    "segment_ks": [8, 16, 32],
    "files": {},
}
for path in sorted(save_dir.glob("eval_long_*_20.json")):
    data = json.loads(path.read_text())
    item = {}
    for key in [
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
    overlaps = []
    projected_counts = []
    teacher_counts = []
    edge_counts = []
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
    item["site_f1_mean"] = sum(overlaps) / len(overlaps) if overlaps else None
    item["projected_changed_count_mean"] = sum(projected_counts) / len(projected_counts) if projected_counts else None
    item["teacher_changed_site_count_mean"] = sum(teacher_counts) / len(teacher_counts) if teacher_counts else None
    item["edge_completion_support_count_mean"] = sum(edge_counts) / len(edge_counts) if edge_counts else None
    summary["files"][path.name] = item
(save_dir / "stage_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
print(f"Saved summary to {save_dir / 'stage_summary.json'}")
PY

echo "=== ${STAGE} DONE $(date -Is) ===" | tee -a "${LOG}"
