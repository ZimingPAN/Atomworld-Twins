#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-cuda}"
SAVE_DIR="${SAVE_DIR:-results/largek_proposalhead_0516/v58_v57cache_proposal_topk128_sigfix}"
CKPT="${CKPT:-${SAVE_DIR}/final_model.pt}"
LOG="${SAVE_DIR}/proposal_sweep.log"

mkdir -p "${SAVE_DIR}"

run_eval() {
  local name="$1"
  local topk="$2"
  local duration_source="$3"
  local planner_tau_source="$4"
  local score_mode="$5"
  local min_changed="$6"
  local rollout_segments="$7"
  shift 7
  {
    echo "=== ${name} START $(date -Is) ==="
    echo "topk=${topk} duration=${duration_source} planner_tau=${planner_tau_source} score=${score_mode} min_changed=${min_changed}"
  } | tee -a "${LOG}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" eval_macro_long_trajectory.py \
    --checkpoint "${CKPT}" \
    --planner_segment_ks 128 256 512 1024 \
    --rollout_segments "${rollout_segments}" \
    --max_episode_steps_override 4096 \
    --duration_source "${duration_source}" \
    --planner_tau_source "${planner_tau_source}" \
    --planner_score_mode "${score_mode}" \
    --planner_noop_risk_penalty 1.0 \
    --planner_projection_change_source proposal \
    --planner_projection_topk_source proposal \
    --planner_projection_topk_budget "${topk}" \
    --proposal_diagnostic \
    --proposal_diagnostic_max_sites 64 \
    --min_projected_changed_sites "${min_changed}" \
    --device "${DEVICE}" \
    --output "${SAVE_DIR}/${name}.json" \
    "$@" \
    2>&1 | tee -a "${LOG}"
  echo "=== ${name} DONE $(date -Is) ===" | tee -a "${LOG}"
}

echo "=== large-k proposal sweep START $(date -Is) ===" | tee -a "${LOG}"
echo "checkpoint=${CKPT}" | tee -a "${LOG}"

run_eval eval_long_proposal_topk256_modeltau 256 model model energy_per_sqrt_tau 2 80
run_eval eval_long_proposal_topk512_modeltau 512 model model energy_per_sqrt_tau 2 80
run_eval eval_long_proposal_topk128_baseline_min2_riskpen1 128 baseline baseline energy_per_tau 2 80
run_eval eval_long_proposal_topk256_baseline_min2_riskpen1 256 baseline baseline energy_per_tau 2 80
run_eval eval_long_proposal_topk512_baseline_min2_riskpen1 512 baseline baseline energy_per_tau 2 80
run_eval eval_long_proposal_topk256_baseline_min2_riskpen1_allownoop20 256 baseline baseline energy_per_tau 2 20 --allow_teacher_noop_segments

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

save_dir = Path("${SAVE_DIR}")
names = [
    "eval_long_proposal_topk256_modeltau.json",
    "eval_long_proposal_topk512_modeltau.json",
    "eval_long_proposal_topk128_baseline_min2_riskpen1.json",
    "eval_long_proposal_topk256_baseline_min2_riskpen1.json",
    "eval_long_proposal_topk512_baseline_min2_riskpen1.json",
    "eval_long_proposal_topk256_baseline_min2_riskpen1_allownoop20.json",
]
summary = {"checkpoint": "${CKPT}", "files": {}}
for name in names:
    path = save_dir / name
    if not path.exists():
        continue
    data = json.loads(path.read_text())
    item = {}
    for key in [
        "completed_rollout_segments",
        "requested_rollout_segments",
        "stop_reason",
        "chosen_k_histogram",
        "duration_source",
        "planner_tau_source",
        "planner_score_mode",
        "planner_projection_topk_budget",
        "effective_min_projected_changed_sites",
        "allow_teacher_noop_segments",
        "cumulative",
    ]:
        if key in data:
            item[key] = data[key]
    first = (data.get("segments") or [{}])[0]
    if first:
        item["first_segment"] = {
            "segment_k": first.get("segment_k"),
            "predicted_delta_e": first.get("predicted_delta_e"),
            "teacher_delta_e": first.get("traditional_kmc_delta_e"),
            "predicted_expected_tau": first.get("predicted_expected_tau"),
            "teacher_expected_tau": first.get("traditional_kmc_expected_tau"),
            "projected_changed_count": first.get("projected_changed_count"),
            "teacher_changed_site_count": first.get("traditional_changed_site_count"),
            "proposal_overlap": first.get("proposal_overlap"),
        }
    summary["files"][name] = item
(save_dir / "proposal_sweep_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
print(f"Saved summary to {save_dir / 'proposal_sweep_summary.json'}")
PY

echo "=== large-k proposal sweep DONE $(date -Is) ===" | tee -a "${LOG}"
