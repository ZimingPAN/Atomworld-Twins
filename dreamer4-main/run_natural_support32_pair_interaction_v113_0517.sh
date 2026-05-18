#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
RESULT_ROOT="${RESULT_ROOT:-results/natural_teacher_support32_sequence_rollout_0517}"
INPUT_STAGE="${INPUT_STAGE:-v111_pair_factorized_readonly_smoke20}"
STAGE="${STAGE:-v113_pair_interaction_readonly}"
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
  echo "mode=read-only conditional compatibility / pair-interaction calibration"
} | tee -a "${LOG}"

"${PYTHON_BIN}" diagnose_vacancy_pair_interaction_v113.py \
  --eval-json "${INPUT_DIR}/eval_long_vacancy_factorized_20.json" \
  --name vacancy_factorized \
  --eval-json "${INPUT_DIR}/eval_long_energy_factorized_20.json" \
  --name energy_factorized \
  --output "${SAVE_DIR}/pair_interaction_v113.json" \
  2>&1 | tee -a "${LOG}"

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

save_dir = Path("${SAVE_DIR}")
summary = json.loads((save_dir / "pair_interaction_v113.json").read_text(encoding="utf-8"))

def slim_score(payload):
    if not isinstance(payload, dict):
        return {}
    global_rank = payload.get("global_rank", {})
    cond = payload.get("conditional_rank", {})
    recall = global_rank.get("recall_at_rank", {}) if isinstance(global_rank, dict) else {}
    fp = global_rank.get("topk_false_positive_rate", {}) if isinstance(global_rank, dict) else {}
    out = {
        "rank_mean": global_rank.get("true_pair_rank_mean"),
        "rank_percentile": global_rank.get("true_pair_rank_percentile_mean"),
        "recall128": recall.get("128") if isinstance(recall, dict) else None,
        "top128_pair_precision": global_rank.get("top128_pair_precision"),
        "top128_pair_recall": global_rank.get("top128_pair_recall"),
        "top128_pair_f1": global_rank.get("top128_pair_f1"),
        "top128_false_positive_rate": fp.get("128") if isinstance(fp, dict) else None,
        "condition": {},
    }
    for name, cond_payload in cond.items() if isinstance(cond, dict) else []:
        cond_recall = cond_payload.get("recall_at_rank", {}) if isinstance(cond_payload, dict) else {}
        labels = cond_payload.get("top32_label_count_mean", {}) if isinstance(cond_payload, dict) else {}
        out["condition"][name] = {
            "rank_mean": cond_payload.get("rank_mean") if isinstance(cond_payload, dict) else None,
            "rank_percentile": cond_payload.get("rank_percentile_mean") if isinstance(cond_payload, dict) else None,
            "group_size_mean": cond_payload.get("group_size_mean") if isinstance(cond_payload, dict) else None,
            "recall8": cond_recall.get("8") if isinstance(cond_recall, dict) else None,
            "recall32": cond_recall.get("32") if isinstance(cond_recall, dict) else None,
            "top32_unpaired": labels.get("source_destination_unpaired") if isinstance(labels, dict) else None,
        }
    return out

def slim_group(group):
    if not isinstance(group, dict):
        return {}
    score_summaries = group.get("score_summaries", {})
    return {
        "candidate_count": group.get("candidate_count"),
        "avg_site_f1": group.get("avg_site_f1"),
        "avg_pair_precision": group.get("avg_vacancy_pair_precision"),
        "avg_pair_f1": group.get("avg_vacancy_pair_f1"),
        "scores": {
            name: slim_score(score_summaries.get(name, {}))
            for name in ["score", "destination_score", "moving_type_score", "calibrated_interaction_score"]
        },
    }

stage = {
    "stage": "${STAGE}",
    "input_stage": "${INPUT_STAGE}",
    "mode": "read-only v113 conditional compatibility calibration",
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
