#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-cuda}"
RESULT_ROOT="${RESULT_ROOT:-results/largek_candidate_oracle_0516}"
STAGE="${STAGE:-v59_candidate_oracle_smoke20}"
SAVE_DIR="${RESULT_ROOT}/${STAGE}"
PRIMARY_CKPT="${PRIMARY_CKPT:-results/largek_candidate_rank_0516/v59_candidate_overlaprank_smoke/final_model.pt}"
ROLLOUT_SEGMENTS="${ROLLOUT_SEGMENTS:-20}"
PROPOSAL_TOPK="${PROPOSAL_TOPK:-256}"
ORACLE_ADD_WEIGHT="${ORACLE_ADD_WEIGHT:-0.5}"
LOG="${SAVE_DIR}/pipeline.log"

mkdir -p "${SAVE_DIR}"

{
  echo "=== ${STAGE} START $(date -Is) ==="
  echo "primary=${PRIMARY_CKPT}"
  echo "rollout_segments=${ROLLOUT_SEGMENTS}"
  echo "proposal_topk=${PROPOSAL_TOPK}"
  echo "oracle_add_weight=${ORACLE_ADD_WEIGHT}"
  echo "diagnostic=eval-only teacher-overlap oracle for large-k candidate selection; no training and no cache collection"
} | tee -a "${LOG}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" eval_macro_long_trajectory.py \
  --checkpoint "${PRIMARY_CKPT}" \
  --planner_segment_ks 128 256 512 1024 \
  --rollout_segments "${ROLLOUT_SEGMENTS}" \
  --max_episode_steps_override 4096 \
  --duration_source model \
  --planner_tau_source model \
  --planner_score_mode energy_per_sqrt_tau \
  --planner_noop_risk_penalty 1.0 \
  --planner_projection_change_source proposal \
  --planner_projection_topk_source proposal \
  --planner_projection_topk_budget "${PROPOSAL_TOPK}" \
  --planner_teacher_overlap_oracle_mode replace \
  --proposal_diagnostic \
  --proposal_diagnostic_max_sites 64 \
  --min_projected_changed_sites 2 \
  --device "${DEVICE}" \
  --output "${SAVE_DIR}/eval_long_modeltau_topk${PROPOSAL_TOPK}_oracle_replace_${ROLLOUT_SEGMENTS}.json" \
  2>&1 | tee -a "${LOG}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" eval_macro_long_trajectory.py \
  --checkpoint "${PRIMARY_CKPT}" \
  --planner_segment_ks 128 256 512 1024 \
  --rollout_segments "${ROLLOUT_SEGMENTS}" \
  --max_episode_steps_override 4096 \
  --duration_source baseline \
  --planner_tau_source baseline \
  --planner_score_mode energy_per_tau \
  --planner_noop_risk_penalty 1.0 \
  --planner_projection_change_source proposal \
  --planner_projection_topk_source proposal \
  --planner_projection_topk_budget "${PROPOSAL_TOPK}" \
  --planner_teacher_overlap_oracle_mode replace \
  --proposal_diagnostic \
  --proposal_diagnostic_max_sites 64 \
  --min_projected_changed_sites 2 \
  --device "${DEVICE}" \
  --output "${SAVE_DIR}/eval_long_baseline_topk${PROPOSAL_TOPK}_oracle_replace_${ROLLOUT_SEGMENTS}.json" \
  2>&1 | tee -a "${LOG}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" eval_macro_long_trajectory.py \
  --checkpoint "${PRIMARY_CKPT}" \
  --planner_segment_ks 128 256 512 1024 \
  --rollout_segments "${ROLLOUT_SEGMENTS}" \
  --max_episode_steps_override 4096 \
  --duration_source baseline \
  --planner_tau_source baseline \
  --planner_score_mode energy_per_tau \
  --planner_noop_risk_penalty 1.0 \
  --planner_projection_change_source proposal \
  --planner_projection_topk_source proposal \
  --planner_projection_topk_budget "${PROPOSAL_TOPK}" \
  --planner_teacher_overlap_oracle_mode add \
  --planner_teacher_overlap_oracle_weight "${ORACLE_ADD_WEIGHT}" \
  --proposal_diagnostic \
  --proposal_diagnostic_max_sites 64 \
  --min_projected_changed_sites 2 \
  --device "${DEVICE}" \
  --output "${SAVE_DIR}/eval_long_baseline_topk${PROPOSAL_TOPK}_oracle_add${ORACLE_ADD_WEIGHT}_${ROLLOUT_SEGMENTS}.json" \
  2>&1 | tee -a "${LOG}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" eval_macro_long_trajectory.py \
  --checkpoint "${PRIMARY_CKPT}" \
  --planner_segment_ks 128 256 512 1024 \
  --rollout_segments "${ROLLOUT_SEGMENTS}" \
  --max_episode_steps_override 4096 \
  --duration_source baseline \
  --planner_tau_source baseline \
  --planner_score_mode energy_per_tau \
  --planner_noop_risk_penalty 1.0 \
  --planner_projection_change_source proposal \
  --planner_projection_topk_source proposal \
  --planner_projection_topk_budget "${PROPOSAL_TOPK}" \
  --planner_teacher_overlap_oracle_mode replace \
  --proposal_diagnostic \
  --proposal_diagnostic_max_sites 64 \
  --min_projected_changed_sites 2 \
  --allow_teacher_noop_segments \
  --device "${DEVICE}" \
  --output "${SAVE_DIR}/eval_long_baseline_topk${PROPOSAL_TOPK}_oracle_replace_allownoop_${ROLLOUT_SEGMENTS}.json" \
  2>&1 | tee -a "${LOG}"

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

save_dir = Path("${SAVE_DIR}")
summary = {
    "stage": "${STAGE}",
    "primary_checkpoint": "${PRIMARY_CKPT}",
    "rollout_segments": int("${ROLLOUT_SEGMENTS}"),
    "proposal_topk": int("${PROPOSAL_TOPK}"),
    "oracle_add_weight": float("${ORACLE_ADD_WEIGHT}"),
    "files": {},
}
for path in sorted(save_dir.glob("eval_long_*.json")):
    data = json.loads(path.read_text())
    vals = [
        float((seg.get("proposal_overlap") or {}).get("f1"))
        for seg in data.get("segments", [])
        if "f1" in (seg.get("proposal_overlap") or {})
    ]
    first = (data.get("segments") or [{}])[0]
    summary["files"][path.name] = {
        "completed_rollout_segments": data.get("completed_rollout_segments"),
        "requested_rollout_segments": data.get("requested_rollout_segments"),
        "stop_reason": data.get("stop_reason"),
        "chosen_k_histogram": data.get("chosen_k_histogram"),
        "duration_source": data.get("duration_source"),
        "planner_tau_source": data.get("planner_tau_source"),
        "planner_score_mode": data.get("planner_score_mode"),
        "planner_teacher_overlap_oracle_mode": data.get("planner_teacher_overlap_oracle_mode"),
        "teacher_overlap_oracle": data.get("teacher_overlap_oracle"),
        "cumulative": data.get("cumulative"),
        "tau_expected": data.get("tau_expected"),
        "site_f1_mean": sum(vals) / len(vals) if vals else None,
        "first_segment": {
            "segment_k": first.get("segment_k"),
            "projected_changed_count": first.get("projected_changed_count"),
            "traditional_changed_site_count": first.get("traditional_changed_site_count"),
            "predicted_delta_e": first.get("predicted_delta_e"),
            "traditional_kmc_delta_e": first.get("traditional_kmc_delta_e"),
            "predicted_expected_tau": first.get("predicted_expected_tau"),
            "traditional_kmc_expected_tau": first.get("traditional_kmc_expected_tau"),
            "proposal_overlap": first.get("proposal_overlap"),
        },
    }
(save_dir / "stage_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
print(f"Saved summary to {save_dir / 'stage_summary.json'}")
PY

echo "=== ${STAGE} DONE $(date -Is) ===" | tee -a "${LOG}"
