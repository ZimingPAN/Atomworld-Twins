#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
RESULT_ROOT="${RESULT_ROOT:-results/natural_teacher_support32_sequence_rollout_0517}"
INPUT_STAGE="${INPUT_STAGE:-v111_pair_factorized_readonly_smoke20}"
STAGE="${STAGE:-v112_pair_conditional_readonly}"
SAVE_DIR="${RESULT_ROOT}/${STAGE}"
INPUT_DIR="${RESULT_ROOT}/${INPUT_STAGE}"
LOG="${SAVE_DIR}/pipeline.log"

mkdir -p "${SAVE_DIR}"

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

{
  echo "=== ${STAGE} START $(timestamp) ==="
  echo "input_stage=${INPUT_STAGE}"
  echo "mode=read-only conditional source/destination vacancy-pair diagnostic"
} | tee -a "${LOG}"

"${PYTHON_BIN}" diagnose_vacancy_pair_conditional_v112.py \
  --eval-json "${INPUT_DIR}/eval_long_vacancy_factorized_20.json" \
  --name vacancy_factorized \
  --eval-json "${INPUT_DIR}/eval_long_energy_factorized_20.json" \
  --name energy_factorized \
  --output "${SAVE_DIR}/conditional_pair_rank_v112.json" \
  2>&1 | tee -a "${LOG}"

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

save_dir = Path("${SAVE_DIR}")
summary = json.loads((save_dir / "conditional_pair_rank_v112.json").read_text(encoding="utf-8"))

def slim_group(group):
    if not isinstance(group, dict):
        return {}
    out = {
        "candidate_count": group.get("candidate_count"),
        "avg_site_f1": group.get("avg_site_f1"),
        "avg_pair_precision": group.get("avg_vacancy_pair_precision"),
        "avg_pair_f1": group.get("avg_vacancy_pair_f1"),
        "best_field_by_recall_at_8": group.get("best_field_by_recall_at_8"),
        "conditions": {},
    }
    cond = group.get("conditional_rank", {})
    for cond_name, payload in cond.items() if isinstance(cond, dict) else []:
        out["conditions"][cond_name] = {}
        for field, ranks in payload.items() if isinstance(payload, dict) else []:
            recall = ranks.get("recall_at_rank", {}) if isinstance(ranks, dict) else {}
            labels = ranks.get("top32_label_count_mean", {}) if isinstance(ranks, dict) else {}
            out["conditions"][cond_name][field] = {
                "rank_mean": ranks.get("rank_mean") if isinstance(ranks, dict) else None,
                "rank_percentile": ranks.get("rank_percentile_mean") if isinstance(ranks, dict) else None,
                "group_size_mean": ranks.get("group_size_mean") if isinstance(ranks, dict) else None,
                "recall1": recall.get("1") if isinstance(recall, dict) else None,
                "recall4": recall.get("4") if isinstance(recall, dict) else None,
                "recall8": recall.get("8") if isinstance(recall, dict) else None,
                "recall32": recall.get("32") if isinstance(recall, dict) else None,
                "top32_true_pair": labels.get("true_pair") if isinstance(labels, dict) else None,
                "top32_unpaired": labels.get("source_destination_unpaired") if isinstance(labels, dict) else None,
            }
    return out

stage = {
    "stage": "${STAGE}",
    "input_stage": "${INPUT_STAGE}",
    "mode": "read-only v112 conditional pair diagnostic",
    "combined": {
        "selected_by_planner": slim_group(summary.get("combined", {}).get("selected_by_planner", {})),
        "all_candidates": slim_group(summary.get("combined", {}).get("all_candidates", {})),
    },
    "inputs": {},
}
for name, payload in summary.get("inputs", {}).items():
    stage["inputs"][name] = {
        "completed_segments": payload.get("completed_rollout_segments"),
        "stop_reason": payload.get("stop_reason"),
        "chosen_k_histogram": payload.get("chosen_k_histogram"),
        "cumulative": payload.get("cumulative"),
        "tau_expected": payload.get("tau_expected"),
        "selected_by_planner": slim_group(payload.get("selected_by_planner", {})),
    }
out = save_dir / "stage_summary.json"
out.write_text(json.dumps(stage, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(stage, ensure_ascii=False, indent=2))
PY

echo "=== ${STAGE} END $(timestamp) ===" | tee -a "${LOG}"
