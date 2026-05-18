#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="/home/likun-share/panziming/pydeps:/home/likun-share/panziming/AtomWorld-Mirror/kmcteacher_backend:/home/likun-share/panziming/AtomWorld-Mirror/LightZero-main:/home/likun-share/panziming/AtomWorld-Mirror/dreamer4-main:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-cuda}"
RESULT_ROOT="${RESULT_ROOT:-results/natural_teacher_support32_sequence_rollout_0517}"
PARETO_MODE="${PARETO_MODE:-diagnostic}"
PARETO_WEIGHT="${PARETO_WEIGHT:-1.0}"
STAGE="${STAGE:-v137_full_true_pair_rank_budgeted_${PARETO_MODE}_smoke20}"
SAVE_DIR="${RESULT_ROOT}/${STAGE}"
CHECKPOINT="${CHECKPOINT:-results/natural_teacher_support32_sequence_rollout_0517/v110_pair_listwise_contrastive_smoke1/final_model.pt}"
if [ ! -f "${CHECKPOINT}" ]; then
  CHECKPOINT="results/natural_teacher_support32_sequence_rollout_0517/v101b_terminal_vacancy_pair_selector_smoke1/final_model.pt"
fi
PROTECTED_CKPT="${PROTECTED_CKPT:-results/natural_teacher_support32_nooprisk_0516/v63_nooprisk_head_plannerpenalty/final_model.pt}"
SELECTOR_SPEC="${SELECTOR_SPEC:-results/natural_teacher_support32_sequence_rollout_0517/v130_recall_floor_selector_replay_readonly/candidate_recall_floor_selector_spec_v130.json}"
RECALL_FLOOR="${RECALL_FLOOR:-0.6}"
MIN_BUDGET="${MIN_BUDGET:-48}"
ROLLOUT_SEGMENTS="${ROLLOUT_SEGMENTS:-20}"
LOG="${SAVE_DIR}/pipeline.log"

mkdir -p "${SAVE_DIR}"

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

{
  echo "=== ${STAGE} START $(timestamp) ==="
  echo "checkpoint=${CHECKPOINT}"
  echo "protected_reward_tau=${PROTECTED_CKPT}"
  echo "selector_spec=${SELECTOR_SPEC}"
  echo "mode=v137 budgeted projection with post-budget teacher true-pair rank labels"
  echo "pareto_mode=${PARETO_MODE}"
  echo "pareto_weight=${PARETO_WEIGHT}"
  echo "recall_floor=${RECALL_FLOOR}"
  echo "min_budget=${MIN_BUDGET}"
  echo "rollout_segments=${ROLLOUT_SEGMENTS}"
} | tee -a "${LOG}"

EVAL_JSON="${SAVE_DIR}/eval_long_full_true_pair_rank_budgeted_${PARETO_MODE}_20.json"
RANK_SUMMARY="${SAVE_DIR}/rank_summary_budgeted_${PARETO_MODE}.json"
FACTORIZED_SUMMARY="${SAVE_DIR}/factorized_summary_budgeted_${PARETO_MODE}.json"

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
  --planner_projection_change_source vacancy_pair_completion \
  --planner_projection_topk_source none \
  --planner_projection_topk_budget 0 \
  --planner_edge_completion_anchor_source action_source \
  --planner_edge_completion_destination_source action_destination \
  --planner_edge_completion_anchor_budget 32 \
  --planner_edge_completion_destinations_per_anchor 4 \
  --planner_edge_completion_destination_scope global_atom \
  --planner_edge_completion_global_pair_budget 0 \
  --planner_projection_change_blend_alpha 0.5 \
  --planner_edge_pair_multiobjective_type_weight 0.15 \
  --planner_edge_pair_multiobjective_order_weight 0.10 \
  --planner_vacancy_pair_rank_diagnostic \
  --planner_vacancy_pair_rank_max_pairs 0 \
  --planner_vacancy_pair_factorized_diagnostic \
  --planner_vacancy_pair_factorized_max_pairs 0 \
  --planner_candidate_pareto_selector_spec "${SELECTOR_SPEC}" \
  --planner_candidate_pareto_selector_mode "${PARETO_MODE}" \
  --planner_candidate_pareto_selector_policy recall_floor_balanced \
  --planner_candidate_pareto_recall_floor "${RECALL_FLOOR}" \
  --planner_candidate_pareto_min_budget "${MIN_BUDGET}" \
  --planner_candidate_pareto_live_score_scale_normalize \
  --planner_candidate_pareto_clip_probability_predictions \
  --planner_candidate_pareto_selector_weight "${PARETO_WEIGHT}" \
  --planner_candidate_pareto_pair_score_field score \
  --planner_candidate_pareto_apply_budget_to_projection \
  --planner_candidate_pareto_teacher_label_after_budget_projection \
  --proposal_diagnostic \
  --proposal_diagnostic_max_sites 384 \
  --planner_teacher_overlap_oracle_mode add \
  --planner_teacher_overlap_oracle_weight 0.0 \
  --planner_teacher_overlap_oracle_metric overlap_reward_norm \
  --planner_candidate_joint_diagnostic \
  --planner_candidate_joint_compact_candidates \
  --min_projected_changed_sites 2 \
  --print_segments 0 \
  --progress_every 5 \
  --device "${DEVICE}" \
  --output "${EVAL_JSON}" \
  2>&1 | tee -a "${LOG}"

"${PYTHON_BIN}" diagnose_vacancy_pair_rank_v108.py \
  --eval-json "${EVAL_JSON}" \
  --output "${RANK_SUMMARY}" \
  2>&1 | tee -a "${LOG}"

"${PYTHON_BIN}" diagnose_vacancy_pair_factorized_v111.py \
  --eval-json "${EVAL_JSON}" \
  --output "${FACTORIZED_SUMMARY}" \
  2>&1 | tee -a "${LOG}"

"${PYTHON_BIN}" - <<PY
import json
from collections import Counter
from pathlib import Path

save_dir = Path("${SAVE_DIR}")
eval_path = Path("${EVAL_JSON}")
rank_path = Path("${RANK_SUMMARY}")
factorized_path = Path("${FACTORIZED_SUMMARY}")
payload = json.loads(eval_path.read_text(encoding="utf-8"))
rank = json.loads(rank_path.read_text(encoding="utf-8"))
factorized = json.loads(factorized_path.read_text(encoding="utf-8"))

def mean(values):
    values = [float(v) for v in values if v is not None]
    return sum(values) / len(values) if values else 0.0

def nested(payload, *keys, default=None):
    value = payload
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
    return default if value is None else value

site_f1 = []
pair_precision = []
pair_recall = []
pair_f1 = []
typed_acc = []
selected_pair_count = []
selected_budget_hist = Counter()
selected_pred_recall = []
selected_pred_precision = []
selected_pred_f1 = []
segment_hit_flags = []

for segment in payload.get("segments", []):
    if not isinstance(segment, dict):
        continue
    site_f1.append(nested(segment, "proposal_overlap", "f1", default=0.0))
    overlap = segment.get("vacancy_pair_overlap") or {}
    pair_precision.append(overlap.get("precision", 0.0))
    pair_recall.append(overlap.get("recall", 0.0))
    pair_f1.append(overlap.get("f1", 0.0))
    typed_acc.append(overlap.get("typed_endpoint_accuracy", 0.0))
    selected_pair_count.append(overlap.get("selected_pair_count", 0.0))
    segment_hit_flags.append(
        float(overlap.get("overlap_pair_count", overlap.get("intersection_count", 0.0)) or 0.0) > 0.0
    )
    selected = segment.get("selected") if isinstance(segment.get("selected"), dict) else None
    if selected is None:
        candidates = [item for item in segment.get("planner_candidates", []) if isinstance(item, dict)]
        selected = max(candidates, key=lambda item: float(item.get("selection_score", float("-inf"))), default={})
    selector = selected.get("planner_candidate_pareto_selector") if isinstance(selected, dict) else {}
    if isinstance(selector, dict):
        selected_budget_hist[str(int(selector.get("budget", -1) or -1))] += 1
        preds = selector.get("predictions") if isinstance(selector.get("predictions"), dict) else {}
        selected_pred_recall.append(preds.get("pair_recall", 0.0))
        selected_pred_precision.append(preds.get("pair_precision", 0.0))
        selected_pred_f1.append(preds.get("pair_f1", 0.0))

selected_rank = rank.get("selected_by_planner_rank") if isinstance(rank.get("selected_by_planner_rank"), dict) else {}
all_rank = rank.get("all_candidates_rank") if isinstance(rank.get("all_candidates_rank"), dict) else {}
selected_factorized = factorized.get("selected_by_planner") if isinstance(factorized.get("selected_by_planner"), dict) else {}
all_factorized = factorized.get("all_candidates") if isinstance(factorized.get("all_candidates"), dict) else {}
selector_summary = payload.get("planner_candidate_pareto_selector") if isinstance(payload.get("planner_candidate_pareto_selector"), dict) else {}

def rank_core(group):
    recall = group.get("avg_recall_at_rank") if isinstance(group.get("avg_recall_at_rank"), dict) else {}
    false_rate = group.get("avg_topk_false_positive_rate") if isinstance(group.get("avg_topk_false_positive_rate"), dict) else {}
    true_count = group.get("avg_topk_true_pair_count") if isinstance(group.get("avg_topk_true_pair_count"), dict) else {}
    return {
        "count": group.get("count"),
        "avg_site_f1": group.get("avg_site_f1"),
        "avg_vacancy_pair_precision": group.get("avg_vacancy_pair_precision"),
        "avg_vacancy_pair_recall": group.get("avg_vacancy_pair_recall"),
        "avg_vacancy_pair_f1": group.get("avg_vacancy_pair_f1"),
        "avg_true_pair_found_recall": group.get("avg_true_pair_found_recall"),
        "avg_true_pair_rank_mean": group.get("avg_true_pair_rank_mean"),
        "avg_true_pair_rank_percentile_mean": group.get("avg_true_pair_rank_percentile_mean"),
        "avg_true_pair_mrr": group.get("avg_true_pair_mrr"),
        "avg_true_pair_typed_rank_accuracy": group.get("avg_true_pair_typed_rank_accuracy"),
        "recall_at_32": recall.get("32"),
        "recall_at_64": recall.get("64"),
        "recall_at_128": recall.get("128"),
        "recall_at_512": recall.get("512"),
        "top128_false_positive_rate": false_rate.get("128"),
        "top128_true_pair_count": true_count.get("128"),
    }

def factor_core(group):
    factors = group.get("factor_rank_summaries") if isinstance(group.get("factor_rank_summaries"), dict) else {}
    compact = {}
    for name, item in factors.items():
        if not isinstance(item, dict):
            continue
        recall = item.get("recall_at_rank") if isinstance(item.get("recall_at_rank"), dict) else {}
        compact[name] = {
            "recall_at_128": recall.get("128"),
            "rank_percentile": item.get("true_pair_rank_percentile_mean"),
            "top128_false_positive_rate": (
                item.get("topk_false_positive_rate", {}).get("128")
                if isinstance(item.get("topk_false_positive_rate"), dict)
                else None
            ),
        }
    return {
        "candidate_count": group.get("candidate_count"),
        "best_factor_by_recall_at_128": group.get("best_factor_by_recall_at_128"),
        "best_factor_by_rank_percentile": group.get("best_factor_by_rank_percentile"),
        "factors": compact,
    }

summary = {
    "stage": "${STAGE}",
    "mode": "v137_budgeted_projection_post_budget_teacher_rank_labels",
    "checkpoint": "${CHECKPOINT}",
    "protected_reward_tau_checkpoint": "${PROTECTED_CKPT}",
    "selector_spec": "${SELECTOR_SPEC}",
    "pareto_mode": "${PARETO_MODE}",
    "recall_floor": float("${RECALL_FLOOR}"),
    "min_budget": int("${MIN_BUDGET}"),
    "eval": {
        "completed_segments": payload.get("completed_rollout_segments"),
        "requested_rollout_segments": payload.get("requested_rollout_segments"),
        "stop_reason": payload.get("stop_reason"),
        "chosen_k_histogram": payload.get("chosen_k_histogram"),
        "cumulative": payload.get("cumulative"),
        "tau_expected": payload.get("tau_expected"),
        "planner_candidate_pareto_selector": selector_summary,
        "teacher_overlap_oracle": payload.get("teacher_overlap_oracle"),
    },
    "selected_overlap_actual": {
        "site_f1": mean(site_f1),
        "pair_precision": mean(pair_precision),
        "pair_recall": mean(pair_recall),
        "pair_f1": mean(pair_f1),
        "typed_endpoint_accuracy": mean(typed_acc),
        "selected_pair_count": mean(selected_pair_count),
        "hit_segment_count": int(sum(1 for item in segment_hit_flags if item)),
    },
    "selected_selector_prediction": {
        "selected_budget_histogram": dict(sorted(selected_budget_hist.items())),
        "pred_pair_recall_mean": mean(selected_pred_recall),
        "pred_pair_precision_mean": mean(selected_pred_precision),
        "pred_pair_f1_mean": mean(selected_pred_f1),
    },
    "true_pair_rank": {
        "selected_by_planner": rank_core(selected_rank),
        "all_candidates": rank_core(all_rank),
    },
    "factorized_rank": {
        "selected_by_planner": factor_core(selected_factorized),
        "all_candidates": factor_core(all_factorized),
    },
    "artifacts": {
        "eval_json": str(eval_path),
        "rank_summary": str(rank_path),
        "factorized_summary": str(factorized_path),
    },
    "interpretation": {
        "teacher_labels_after_budget_projection": bool(
            payload.get("planner_candidate_pareto_teacher_label_after_budget_projection", False)
        ),
        "budget_applied_to_projection": bool(selector_summary.get("budget_applied_to_projection", False)),
        "teacher_label_fields_used": bool(selector_summary.get("teacher_label_fields_used", False)),
    },
}

out = save_dir / "stage_summary.json"
out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

echo "=== ${STAGE} END $(timestamp) ===" | tee -a "${LOG}"
