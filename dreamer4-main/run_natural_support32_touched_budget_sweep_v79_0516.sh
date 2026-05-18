#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-cuda}"
RESULT_ROOT="${RESULT_ROOT:-results/natural_teacher_support32_touchedproposal_0516}"
STAGE="${STAGE:-v79_touched_budget_sweep20}"
SAVE_DIR="${RESULT_ROOT}/${STAGE}"
MODEL_CKPT="${MODEL_CKPT:-results/natural_teacher_support32_touchedproposal_0516/v78_touchedsupport_smoke1/final_model.pt}"
PROTECTED_CKPT="${PROTECTED_CKPT:-results/natural_teacher_support32_nooprisk_0516/v63_nooprisk_head_plannerpenalty/final_model.pt}"
ROLLS="${ROLLS:-20}"
BUDGETS="${BUDGETS:-32 64 96 128 192 256}"
LOG="${SAVE_DIR}/pipeline.log"

mkdir -p "${SAVE_DIR}"

{
  echo "=== ${STAGE} START $(date -Is) ==="
  echo "model=${MODEL_CKPT}"
  echo "protected_reward_tau=${PROTECTED_CKPT}"
  echo "rollout_segments=${ROLLS}"
  echo "budgets=${BUDGETS}"
  echo "diagnostic=eval-only touched proposal support-budget sweep; no training and no cache collection"
} | tee -a "${LOG}"

for B in ${BUDGETS}; do
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" eval_macro_long_trajectory.py \
    --checkpoint "${MODEL_CKPT}" \
    --reward_checkpoint "${PROTECTED_CKPT}" \
    --duration_checkpoint "${PROTECTED_CKPT}" \
    --aux_projected_types_source primary \
    --planner_segment_ks 8 16 32 \
    --rollout_segments "${ROLLS}" \
    --max_episode_steps_override 1024 \
    --duration_source model \
    --planner_tau_source model \
    --planner_score_mode energy \
    --planner_noop_risk_penalty 1.0 \
    --planner_projection_change_source proposal \
    --planner_projection_topk_source proposal \
    --planner_projection_topk_budget "${B}" \
    --proposal_diagnostic \
    --min_projected_changed_sites 2 \
    --device "${DEVICE}" \
    --output "${SAVE_DIR}/eval_proposal_as_change_topk${B}_${ROLLS}.json" \
    2>&1 | tee -a "${LOG}"

  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" eval_macro_long_trajectory.py \
    --checkpoint "${MODEL_CKPT}" \
    --reward_checkpoint "${PROTECTED_CKPT}" \
    --duration_checkpoint "${PROTECTED_CKPT}" \
    --aux_projected_types_source primary \
    --planner_segment_ks 8 16 32 \
    --rollout_segments "${ROLLS}" \
    --max_episode_steps_override 1024 \
    --duration_source model \
    --planner_tau_source model \
    --planner_score_mode energy \
    --planner_noop_risk_penalty 1.0 \
    --planner_projection_change_source change \
    --planner_projection_topk_source proposal \
    --planner_projection_topk_budget "${B}" \
    --proposal_diagnostic \
    --min_projected_changed_sites 2 \
    --device "${DEVICE}" \
    --output "${SAVE_DIR}/eval_change_inside_proposal_gate_topk${B}_${ROLLS}.json" \
    2>&1 | tee -a "${LOG}"
done

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

save_dir = Path("${SAVE_DIR}")
summary = {
    "stage": "${STAGE}",
    "model_checkpoint": "${MODEL_CKPT}",
    "protected_reward_tau_checkpoint": "${PROTECTED_CKPT}",
    "rollout_segments": int("${ROLLS}"),
    "budgets": [int(x) for x in "${BUDGETS}".split()],
    "files": {},
}
for path in sorted(save_dir.glob("eval_*_*.json")):
    data = json.loads(path.read_text())
    cumulative = data.get("cumulative", {})
    segments = data.get("segments", [])
    overlaps = []
    projected_counts = []
    teacher_counts = []
    for seg in segments:
        po = seg.get("proposal_overlap")
        if isinstance(po, dict):
            overlaps.append(float(po.get("f1", 0.0)))
        if "projected_changed_count" in seg:
            projected_counts.append(float(seg["projected_changed_count"]))
        if "traditional_changed_site_count" in seg:
            teacher_counts.append(float(seg["traditional_changed_site_count"]))
    summary["files"][path.name] = {
        "completed_rollout_segments": data.get("completed_rollout_segments"),
        "requested_rollout_segments": data.get("requested_rollout_segments"),
        "stop_reason": data.get("stop_reason"),
        "chosen_k_histogram": data.get("chosen_k_histogram"),
        "delta_e_ratio": cumulative.get("delta_e_ratio"),
        "expected_time_ratio": cumulative.get("expected_time_ratio"),
        "tau_scale_ratio": (data.get("tau_expected") or {}).get("scale_ratio"),
        "site_f1_mean": sum(overlaps) / len(overlaps) if overlaps else None,
        "projected_changed_count_mean": sum(projected_counts) / len(projected_counts) if projected_counts else None,
        "teacher_changed_site_count_mean": sum(teacher_counts) / len(teacher_counts) if teacher_counts else None,
    }
(save_dir / "stage_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
print(f"Saved summary to {save_dir / 'stage_summary.json'}")
PY

echo "=== ${STAGE} DONE $(date -Is) ===" | tee -a "${LOG}"
