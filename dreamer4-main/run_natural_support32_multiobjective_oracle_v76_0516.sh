#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
GPU_ID="${GPU_ID:-1}"
DEVICE="${DEVICE:-cuda}"
RESULT_ROOT="${RESULT_ROOT:-results/proposal_diagnostics_0516}"
STAGE="${STAGE:-v76_multiobjective_oracle_smoke20}"
SAVE_DIR="${RESULT_ROOT}/${STAGE}"
PRIMARY_CKPT="${PRIMARY_CKPT:-results/natural_teacher_support32_proposalhead_0516/v69_proposal_smoke1/final_model.pt}"
PROTECTED_CKPT="${PROTECTED_CKPT:-results/natural_teacher_support32_nooprisk_0516/v63_nooprisk_head_plannerpenalty/final_model.pt}"
ROLLOUT_SEGMENTS="${ROLLOUT_SEGMENTS:-20}"
ORACLE_MODE="${ORACLE_MODE:-replace}"
ORACLE_ADD_WEIGHT="${ORACLE_ADD_WEIGHT:-0.5}"
METRICS="${METRICS:-overlap_f1 teacher_reward teacher_reward_per_tau teacher_reward_per_sqrt_tau overlap_reward_norm}"
LOG="${SAVE_DIR}/pipeline.log"

mkdir -p "${SAVE_DIR}"

{
  echo "=== ${STAGE} START $(date -Is) ==="
  echo "primary=${PRIMARY_CKPT}"
  echo "protected_reward_tau=${PROTECTED_CKPT}"
  echo "rollout_segments=${ROLLOUT_SEGMENTS}"
  echo "oracle_mode=${ORACLE_MODE}"
  echo "oracle_add_weight=${ORACLE_ADD_WEIGHT}"
  echo "metrics=${METRICS}"
  echo "diagnostic=eval-only multi-objective teacher oracle sweep for support32 candidate selection"
} | tee -a "${LOG}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" eval_macro_long_trajectory.py \
  --checkpoint "${PRIMARY_CKPT}" \
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
  --planner_projection_change_source proposal \
  --planner_projection_topk_source proposal \
  --planner_projection_topk_budget 128 \
  --proposal_diagnostic \
  --min_projected_changed_sites 2 \
  --device "${DEVICE}" \
  --output "${SAVE_DIR}/eval_long_energy_topk128_modelscore_${ROLLOUT_SEGMENTS}.json" \
  2>&1 | tee -a "${LOG}"

for metric in ${METRICS}; do
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" eval_macro_long_trajectory.py \
    --checkpoint "${PRIMARY_CKPT}" \
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
    --planner_projection_change_source proposal \
    --planner_projection_topk_source proposal \
    --planner_projection_topk_budget 128 \
    --planner_teacher_overlap_oracle_mode "${ORACLE_MODE}" \
    --planner_teacher_overlap_oracle_weight "${ORACLE_ADD_WEIGHT}" \
    --planner_teacher_overlap_oracle_metric "${metric}" \
    --proposal_diagnostic \
    --min_projected_changed_sites 2 \
    --device "${DEVICE}" \
    --output "${SAVE_DIR}/eval_long_energy_topk128_oracle_${ORACLE_MODE}_${metric}_${ROLLOUT_SEGMENTS}.json" \
    2>&1 | tee -a "${LOG}"
done

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

save_dir = Path("${SAVE_DIR}")
summary = {
    "stage": "${STAGE}",
    "primary_checkpoint": "${PRIMARY_CKPT}",
    "protected_reward_tau_checkpoint": "${PROTECTED_CKPT}",
    "rollout_segments": int("${ROLLOUT_SEGMENTS}"),
    "oracle_mode": "${ORACLE_MODE}",
    "oracle_add_weight": float("${ORACLE_ADD_WEIGHT}"),
    "files": {},
}
for path in sorted(save_dir.glob("eval_long_energy_topk128_*.json")):
    data = json.loads(path.read_text())
    vals = [
        float((seg.get("proposal_overlap") or {}).get("f1"))
        for seg in data.get("segments", [])
        if "f1" in (seg.get("proposal_overlap") or {})
    ]
    cumulative = data.get("cumulative") or {}
    tau_expected = data.get("tau_expected") or {}
    summary["files"][path.name] = {
        "completed_rollout_segments": data.get("completed_rollout_segments"),
        "stop_reason": data.get("stop_reason"),
        "chosen_k_histogram": data.get("chosen_k_histogram"),
        "planner_teacher_overlap_oracle_mode": data.get("planner_teacher_overlap_oracle_mode"),
        "planner_teacher_overlap_oracle_metric": data.get("planner_teacher_overlap_oracle_metric"),
        "planner_teacher_overlap_oracle_weight": data.get("planner_teacher_overlap_oracle_weight"),
        "teacher_overlap_oracle": data.get("teacher_overlap_oracle"),
        "delta_e_ratio": cumulative.get("delta_e_ratio"),
        "expected_time_ratio": cumulative.get("expected_time_ratio"),
        "tau_log_mae": tau_expected.get("log_mae"),
        "tau_scale_ratio": tau_expected.get("scale_ratio"),
        "site_f1_mean": sum(vals) / len(vals) if vals else None,
        "cumulative": cumulative,
    }
(save_dir / "stage_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
print(f"Saved summary to {save_dir / 'stage_summary.json'}")
PY

echo "=== ${STAGE} DONE $(date -Is) ===" | tee -a "${LOG}"
