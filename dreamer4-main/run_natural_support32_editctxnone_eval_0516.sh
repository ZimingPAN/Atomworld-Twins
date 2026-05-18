#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
GPU_ID="${GPU_ID:-3}"
DEVICE="${DEVICE:-cuda}"
RESULT_ROOT="${RESULT_ROOT:-results/natural_teacher_support32_editctxnone_0516}"
STAGE="${STAGE:-v66_editctxnone_evalonly}"
SAVE_DIR="${RESULT_ROOT}/${STAGE}"
CKPT="${CKPT:-results/natural_teacher_support32_editonly_0516/v65_editonly_smoke1/final_model.pt}"
CACHE="${CACHE:-results/natural_teacher_support32_dualsupport_0516/v64_teacheraug_positive_riskguard/segments.pt}"
ROLLOUT_SEGMENTS="${ROLLOUT_SEGMENTS:-80}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LOG="${SAVE_DIR}/pipeline.log"

mkdir -p "${SAVE_DIR}"

{
  echo "=== ${STAGE} START $(date -Is) ==="
  echo "checkpoint=${CKPT}"
  echo "cache=${CACHE}"
  echo "diagnostic=reward/tau edit-summary context disabled; projected patch context retained"
} | tee -a "${LOG}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" eval_macro_time_alignment.py \
  --checkpoint "${CKPT}" \
  --cache "${CACHE}" \
  --split val \
  --batch_size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --reward_edit_context_source none \
  --output "${SAVE_DIR}/eval_time_alignment_editctxnone.json" \
  --save_all_samples \
  2>&1 | tee -a "${LOG}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" eval_macro_long_trajectory.py \
  --checkpoint "${CKPT}" \
  --planner_segment_ks 8 16 32 \
  --rollout_segments "${ROLLOUT_SEGMENTS}" \
  --max_episode_steps_override 1024 \
  --duration_source model \
  --planner_tau_source model \
  --planner_score_mode energy_per_sqrt_tau \
  --min_projected_changed_sites 2 \
  --reward_edit_context_source none \
  --device "${DEVICE}" \
  --output "${SAVE_DIR}/eval_long_editctxnone80.json" \
  2>&1 | tee -a "${LOG}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" eval_macro_long_trajectory.py \
  --checkpoint "${CKPT}" \
  --planner_segment_ks 8 16 32 \
  --rollout_segments "${ROLLOUT_SEGMENTS}" \
  --max_episode_steps_override 1024 \
  --duration_source model \
  --planner_tau_source model \
  --planner_score_mode energy \
  --planner_noop_risk_penalty 1.0 \
  --min_projected_changed_sites 2 \
  --reward_edit_context_source none \
  --device "${DEVICE}" \
  --output "${SAVE_DIR}/eval_long_energy_editctxnone80.json" \
  2>&1 | tee -a "${LOG}"

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

save_dir = Path("${SAVE_DIR}")
summary = {
    "stage": "${STAGE}",
    "checkpoint": "${CKPT}",
    "cache": "${CACHE}",
    "reward_edit_context_source": "none",
    "diagnostic": "disable edit-summary context for reward/tau/risk heads while retaining projected patch context",
    "files": {},
}
for name in [
    "eval_time_alignment_editctxnone.json",
    "eval_long_editctxnone80.json",
    "eval_long_energy_editctxnone80.json",
]:
    path = save_dir / name
    if not path.exists():
        continue
    data = json.loads(path.read_text())
    item = {}
    for key in [
        "reward_sum",
        "tau_expected",
        "noop_risk",
        "cumulative",
        "chosen_k_histogram",
        "completed_rollout_segments",
        "requested_rollout_segments",
        "stop_reason",
        "reward_edit_context_source",
    ]:
        if key in data:
            item[key] = data[key]
    summary["files"][name] = item
(save_dir / "stage_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
print(f"Saved summary to {save_dir / 'stage_summary.json'}")
PY

echo "=== ${STAGE} DONE $(date -Is) ===" | tee -a "${LOG}"
