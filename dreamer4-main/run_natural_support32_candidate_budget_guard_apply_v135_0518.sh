#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="/home/likun-share/panziming/pydeps:/home/likun-share/panziming/AtomWorld-Mirror/kmcteacher_backend:/home/likun-share/panziming/AtomWorld-Mirror/LightZero-main:/home/likun-share/panziming/AtomWorld-Mirror/dreamer4-main:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-cuda}"
RESULT_ROOT="${RESULT_ROOT:-results/natural_teacher_support32_sequence_rollout_0517}"
STAGE="${STAGE:-v135_score_scale_minbudget48_budgetapply_smoke20}"
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
  echo "mode=v135 score-scale normalized recall-floor selector with budget-to-projection"
  echo "recall_floor=${RECALL_FLOOR}"
  echo "min_budget=${MIN_BUDGET}"
  echo "rollout_segments=${ROLLOUT_SEGMENTS}"
} | tee -a "${LOG}"

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
  --planner_vacancy_pair_factorized_diagnostic \
  --planner_vacancy_pair_factorized_max_pairs 0 \
  --planner_candidate_pareto_selector_spec "${SELECTOR_SPEC}" \
  --planner_candidate_pareto_selector_mode diagnostic \
  --planner_candidate_pareto_selector_policy recall_floor_balanced \
  --planner_candidate_pareto_recall_floor "${RECALL_FLOOR}" \
  --planner_candidate_pareto_min_budget "${MIN_BUDGET}" \
  --planner_candidate_pareto_live_score_scale_normalize \
  --planner_candidate_pareto_clip_probability_predictions \
  --planner_candidate_pareto_apply_budget_to_projection \
  --planner_candidate_pareto_pair_score_field score \
  --proposal_diagnostic \
  --proposal_diagnostic_max_sites 384 \
  --planner_candidate_joint_compact_candidates \
  --min_projected_changed_sites 2 \
  --print_segments 0 \
  --progress_every 5 \
  --device "${DEVICE}" \
  --output "${SAVE_DIR}/eval_long_score_scale_minbudget48_budgetapply_20.json" \
  2>&1 | tee -a "${LOG}"

"${PYTHON_BIN}" - <<PY
import json
from collections import Counter
from pathlib import Path

save_dir = Path("${SAVE_DIR}")
eval_path = save_dir / "eval_long_score_scale_minbudget48_budgetapply_20.json"
payload = json.loads(eval_path.read_text(encoding="utf-8"))
segments = payload.get("segments", [])
recall_floor = float("${RECALL_FLOOR}")
min_budget = int("${MIN_BUDGET}")

def mean(values):
    values = [float(v) for v in values if v is not None]
    return sum(values) / len(values) if values else 0.0

def legal_candidate(candidate):
    return (
        float(candidate.get("reachability_violation", 1.0) or 0.0) <= 0.0
        and float(candidate.get("projected_changed_count", 0.0) or 0.0) >= 2.0
    )

site_f1 = []
pair_precision = []
pair_recall = []
pair_f1 = []
typed_acc = []
selected_pair_count = []
feature_dims = Counter()
selector_budget_hist = Counter()
selector_floor_pass_hist = Counter()
selector_min_budget_pass_hist = Counter()
selector_suggested_k_hist = Counter()
selected_k_hist = Counter()
selected_budget_hist = Counter()
selected_projection_budget_hist = Counter()
candidate_count_hist = Counter()
candidate_records = 0
selector_records = 0
applied_selector_records = 0
selector_pred_pair_recall = []
selector_pred_pair_precision = []
selector_pred_pair_f1 = []
selected_pred_pair_recall = []
selected_pred_pair_precision = []
selected_pred_pair_f1 = []
selector_tiny_high_recall_count = 0
selector_pair_recall_gt1_count = 0

for segment in segments:
    site_f1.append((segment.get("proposal_overlap") or {}).get("f1", 0.0))
    overlap = segment.get("vacancy_pair_overlap") or {}
    pair_precision.append(overlap.get("precision", 0.0))
    pair_recall.append(overlap.get("recall", 0.0))
    pair_f1.append(overlap.get("f1", 0.0))
    typed_acc.append(overlap.get("typed_endpoint_accuracy", 0.0))
    selected_pair_count.append(overlap.get("selected_pair_count", 0.0))
    selected_k_hist[str(int(segment.get("segment_k", 0)))] += 1
    candidates = [item for item in segment.get("planner_candidates", []) if isinstance(item, dict)]
    candidate_count_hist[str(len(candidates))] += 1
    legal = [item for item in candidates if legal_candidate(item)]
    selected_pick = max(
        legal,
        key=lambda item: float(item.get("selection_score", float("-inf"))),
        default=None,
    )
    selector_pick = max(
        [
            item for item in legal
            if isinstance(item.get("planner_candidate_pareto_selector"), dict)
        ],
        key=lambda item: float(item["planner_candidate_pareto_selector"].get("score", float("-inf"))),
        default=None,
    )
    if selected_pick is not None:
        selector = selected_pick.get("planner_candidate_pareto_selector") or {}
        selected_budget_hist[str(int(selector.get("budget", -1)))] += 1
        selected_projection_budget_hist[str(int(selector.get("projection_rerun_pair_budget", -1)))] += 1
        predictions = selector.get("predictions") or {}
        selected_pred_pair_recall.append(float(predictions.get("pair_recall", 0.0) or 0.0))
        selected_pred_pair_precision.append(float(predictions.get("pair_precision", 0.0) or 0.0))
        selected_pred_pair_f1.append(float(predictions.get("pair_f1", 0.0) or 0.0))
    if selector_pick is not None:
        selector = selector_pick.get("planner_candidate_pareto_selector") or {}
        selector_suggested_k_hist[str(int(selector_pick.get("segment_k", 0)))] += 1
        budget = int(selector.get("budget", -1))
        selector_budget_hist[str(budget)] += 1
        selector_floor_pass_hist[str(bool(selector.get("pair_recall_floor_passed", False)))] += 1
        selector_min_budget_pass_hist[str(bool(selector.get("min_budget_passed", False)))] += 1
        predictions = selector.get("predictions") or {}
        pred_recall = float(predictions.get("pair_recall", 0.0) or 0.0)
        pred_precision = float(predictions.get("pair_precision", 0.0) or 0.0)
        pred_f1 = float(predictions.get("pair_f1", 0.0) or 0.0)
        selector_pred_pair_recall.append(pred_recall)
        selector_pred_pair_precision.append(pred_precision)
        selector_pred_pair_f1.append(pred_f1)
        if budget <= 12 and pred_recall >= recall_floor:
            selector_tiny_high_recall_count += 1
        if pred_recall > 1.0:
            selector_pair_recall_gt1_count += 1
    for candidate in candidates:
        candidate_records += 1
        selector = candidate.get("planner_candidate_pareto_selector")
        if isinstance(selector, dict):
            selector_records += 1
            feature_dims[str(int(selector.get("feature_dim", -1)))] += 1
            if bool(selector.get("budget_applied_to_projection", False)):
                applied_selector_records += 1

selector_summary = payload.get("planner_candidate_pareto_selector") or {}
summary = {
    "stage": "${STAGE}",
    "mode": "budget_to_projection_score_scale_normalized_min_budget_guard",
    "checkpoint": "${CHECKPOINT}",
    "protected_reward_tau_checkpoint": "${PROTECTED_CKPT}",
    "selector_spec": "${SELECTOR_SPEC}",
    "recall_floor": recall_floor,
    "min_budget": min_budget,
    "eval": {
        "completed_segments": payload.get("completed_rollout_segments"),
        "requested_rollout_segments": payload.get("requested_rollout_segments"),
        "stop_reason": payload.get("stop_reason"),
        "chosen_k_histogram": payload.get("chosen_k_histogram"),
        "cumulative": payload.get("cumulative"),
        "tau_expected": payload.get("tau_expected"),
        "planner_candidate_pareto_selector": selector_summary,
    },
    "selected_overlap": {
        "site_f1": mean(site_f1),
        "pair_precision": mean(pair_precision),
        "pair_recall": mean(pair_recall),
        "pair_f1": mean(pair_f1),
        "typed_endpoint_accuracy": mean(typed_acc),
        "selected_pair_count": mean(selected_pair_count),
    },
    "diagnostic_integrity": {
        "candidate_count_histogram": dict(sorted(candidate_count_hist.items())),
        "candidate_records": int(candidate_records),
        "selector_records": int(selector_records),
        "applied_selector_records": int(applied_selector_records),
        "feature_dim_histogram": dict(sorted(feature_dims.items())),
        "selected_k_histogram": dict(sorted(selected_k_hist.items())),
        "selected_budget_histogram": dict(sorted(selected_budget_hist.items())),
        "selected_projection_budget_histogram": dict(sorted(selected_projection_budget_hist.items())),
        "selector_suggested_k_histogram": dict(sorted(selector_suggested_k_hist.items())),
        "selector_suggested_budget_histogram": dict(sorted(selector_budget_hist.items())),
        "selector_suggested_floor_pass_histogram": dict(sorted(selector_floor_pass_hist.items())),
        "selector_suggested_min_budget_pass_histogram": dict(sorted(selector_min_budget_pass_hist.items())),
        "selector_suggested_pred_pair_recall_mean": mean(selector_pred_pair_recall),
        "selector_suggested_pred_pair_precision_mean": mean(selector_pred_pair_precision),
        "selector_suggested_pred_pair_f1_mean": mean(selector_pred_pair_f1),
        "selected_pred_pair_recall_mean": mean(selected_pred_pair_recall),
        "selected_pred_pair_precision_mean": mean(selected_pred_pair_precision),
        "selected_pred_pair_f1_mean": mean(selected_pred_pair_f1),
        "selector_tiny_high_recall_count": int(selector_tiny_high_recall_count),
        "selector_pair_recall_gt1_count": int(selector_pair_recall_gt1_count),
        "budget_applied_to_projection": bool(selector_summary.get("budget_applied_to_projection", False)),
        "budget_projection_rerun_count": int(selector_summary.get("budget_projection_rerun_count", 0)),
        "budget_projection_rerun_failed_count": int(
            selector_summary.get("budget_projection_rerun_failed_count", 0)
        ),
        "teacher_label_fields_used": bool(selector_summary.get("teacher_label_fields_used", True)),
    },
    "guard_checks": {
        "no_tiny_high_recall": int(selector_tiny_high_recall_count) == 0,
        "no_pair_recall_gt_1": int(selector_pair_recall_gt1_count) == 0,
        "all_selector_picks_pass_min_budget": selector_min_budget_pass_hist.get("False", 0) == 0,
        "budget_projection_rerun_complete": int(selector_summary.get("budget_projection_rerun_failed_count", 0)) == 0,
    },
}
out = save_dir / "stage_summary.json"
out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

echo "=== ${STAGE} END $(timestamp) ===" | tee -a "${LOG}"
