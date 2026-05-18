#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-cuda}"
RESULT_ROOT="${RESULT_ROOT:-results/largek_proposalhead_0516}"
STAGE="${STAGE:-v58_v57cache_proposal_topk128_sigfix}"
SAVE_DIR="${RESULT_ROOT}/${STAGE}"
INIT_CKPT="${INIT_CKPT:-results/largek_dualsupport_0516/v57_teacheraug_positive_riskguard/final_model.pt}"
PLANNER_CKPT="${PLANNER_CKPT:-results/largek_nooprisk_0516/v56_noop_horizon_augmented_riskhead/final_model.pt}"
SOURCE_CACHE="${SOURCE_CACHE:-results/largek_dualsupport_0516/v57_teacheraug_positive_riskguard/segments.pt}"
CACHE="${CACHE:-${SAVE_DIR}/segments.pt}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LR="${LR:-3e-5}"
PROPOSAL_TOPK="${PROPOSAL_TOPK:-128}"
LOG="${SAVE_DIR}/pipeline.log"

mkdir -p "${SAVE_DIR}"
if [[ ! -f "${CACHE}" ]]; then
  cp "${SOURCE_CACHE}" "${CACHE}"
fi
"${PYTHON_BIN}" - <<PY
from pathlib import Path
import torch

cache = Path("${CACHE}")
payload = torch.load(cache, map_location="cpu", weights_only=False)
sig = payload.setdefault("signature", {})
sig.update({
    "planner_selected_reward_checkpoint": None,
    "planner_selected_duration_checkpoint": None,
    "planner_selected_planner_duration_checkpoint_source": "duration",
    "planner_selected_aux_projected_types_source": "aux",
    "planner_selected_projection_change_source": "change",
    "planner_selected_projection_change_blend_alpha": 0.5,
    "planner_selected_projection_topk_source": "none",
    "planner_selected_projection_topk_budget": 0,
    "planner_selected_proposal_score_weight": 0.0,
    "planner_selected_noop_risk_penalty": 0.0,
})
torch.save(payload, cache)
print(f"Normalized copied cache signature at {cache}")
PY

{
  echo "=== ${STAGE} START $(date -Is) ==="
  echo "init=${INIT_CKPT}"
  echo "planner_signature=${PLANNER_CKPT}"
  echo "source_cache=${SOURCE_CACHE}"
  echo "cache=${CACHE}"
  echo "proposal_topk=${PROPOSAL_TOPK}"
  echo "fix=large-k diagnostic: train proposal_head only on v57 planner-selected cache, then eval proposal top-k closed-loop support"
} | tee -a "${LOG}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" train_dreamer_macro_edit.py \
  --save_dir "${SAVE_DIR}" \
  --dataset_cache "${CACHE}" \
  --init_from "${INIT_CKPT}" \
  --planner_selected_from "${PLANNER_CKPT}" \
  --planner_selected_min_projected_changed_sites 2 \
  --planner_selected_duration_source baseline \
  --planner_selected_tau_source baseline \
  --planner_selected_score_mode energy_per_sqrt_tau \
  --planner_selected_reward_prediction_source projected \
  --segment_ks 128 256 512 1024 \
  --train_segments_per_k 96 \
  --val_segments_per_k 24 \
  --max_episode_steps 4096 \
  --max_segments_per_rollout 4 \
  --max_seed_vacancies 8 \
  --max_candidate_sites 1024 \
  --teacher_candidate_neighbor_depth 1 \
  --teacher_path_summary_mode stepwise \
  --tau_supervision_mode prior_main \
  --reward_prediction_source projected \
  --include_noop_segments \
  --keep_after_noop_segments \
  --train_proposal_head_only \
  --init_proposal_from_change_head \
  --proposal_support_weight 1.0 \
  --prior_proposal_support_weight 1.0 \
  --reward_weight 0.0 \
  --prior_reward_weight 0.0 \
  --tau_weight 0.0 \
  --realized_tau_weight 0.0 \
  --tau_log_mu_weight 0.0 \
  --pair_weight 0.0 \
  --prior_pair_weight 0.0 \
  --prior_edit_weight 0.0 \
  --prior_latent_weight 0.0 \
  --proj_weight 0.0 \
  --path_weight 0.0 \
  --noop_risk_weight 0.0 \
  --prior_noop_risk_weight 0.0 \
  --no_aux_anneal \
  --lr "${LR}" \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --device "${DEVICE}" \
  2>&1 | tee -a "${LOG}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" eval_macro_time_alignment.py \
  --checkpoint "${SAVE_DIR}/final_model.pt" \
  --cache "${CACHE}" \
  --split val \
  --batch_size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --output "${SAVE_DIR}/eval_time_alignment_final.json" \
  --save_all_samples \
  2>&1 | tee -a "${LOG}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" eval_macro_long_trajectory.py \
  --checkpoint "${SAVE_DIR}/final_model.pt" \
  --planner_segment_ks 128 256 512 1024 \
  --rollout_segments 80 \
  --max_episode_steps_override 4096 \
  --duration_source model \
  --planner_tau_source model \
  --planner_score_mode energy_per_sqrt_tau \
  --planner_projection_change_source proposal \
  --planner_projection_topk_source proposal \
  --planner_projection_topk_budget "${PROPOSAL_TOPK}" \
  --proposal_diagnostic \
  --min_projected_changed_sites 2 \
  --device "${DEVICE}" \
  --output "${SAVE_DIR}/eval_long_proposal_topk128_modeltau.json" \
  2>&1 | tee -a "${LOG}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" eval_macro_long_trajectory.py \
  --checkpoint "${SAVE_DIR}/final_model.pt" \
  --planner_segment_ks 128 256 512 1024 \
  --rollout_segments 80 \
  --max_episode_steps_override 4096 \
  --duration_source model \
  --planner_tau_source model \
  --planner_score_mode energy_per_sqrt_tau \
  --planner_noop_risk_penalty 1.0 \
  --planner_projection_change_source proposal \
  --planner_projection_topk_source proposal \
  --planner_projection_topk_budget "${PROPOSAL_TOPK}" \
  --proposal_diagnostic \
  --min_projected_changed_sites 2 \
  --device "${DEVICE}" \
  --output "${SAVE_DIR}/eval_long_proposal_topk128_modeltau_riskpen1.json" \
  2>&1 | tee -a "${LOG}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" eval_macro_long_trajectory.py \
  --checkpoint "${SAVE_DIR}/final_model.pt" \
  --planner_segment_ks 128 256 512 1024 \
  --rollout_segments 80 \
  --max_episode_steps_override 4096 \
  --duration_source baseline \
  --planner_tau_source baseline \
  --planner_score_mode energy_per_tau \
  --planner_noop_risk_penalty 1.0 \
  --planner_projection_change_source proposal \
  --planner_projection_topk_source proposal \
  --planner_projection_topk_budget "${PROPOSAL_TOPK}" \
  --proposal_diagnostic \
  --min_projected_changed_sites 10 \
  --device "${DEVICE}" \
  --output "${SAVE_DIR}/eval_long_proposal_topk128_baseline_min10_riskpen1.json" \
  2>&1 | tee -a "${LOG}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" eval_macro_long_trajectory.py \
  --checkpoint "${SAVE_DIR}/final_model.pt" \
  --planner_segment_ks 128 256 512 1024 \
  --rollout_segments 20 \
  --max_episode_steps_override 4096 \
  --duration_source baseline \
  --planner_tau_source baseline \
  --planner_score_mode energy_per_tau \
  --planner_noop_risk_penalty 1.0 \
  --planner_projection_change_source proposal \
  --planner_projection_topk_source proposal \
  --planner_projection_topk_budget "${PROPOSAL_TOPK}" \
  --proposal_diagnostic \
  --min_projected_changed_sites 10 \
  --allow_teacher_noop_segments \
  --device "${DEVICE}" \
  --output "${SAVE_DIR}/eval_long_proposal_topk128_baseline_min10_riskpen1_allownoop20.json" \
  2>&1 | tee -a "${LOG}"

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

save_dir = Path("${SAVE_DIR}")
summary = {
    "stage": "${STAGE}",
    "init_checkpoint": "${INIT_CKPT}",
    "planner_signature_checkpoint": "${PLANNER_CKPT}",
    "source_cache": "${SOURCE_CACHE}",
    "dataset_cache": "${CACHE}",
    "segment_ks": [128, 256, 512, 1024],
    "proposal_topk": int("${PROPOSAL_TOPK}"),
    "fix": "proposal_head_only_on_v57_cache",
    "files": {},
}
for name in [
    "metrics.json",
    "eval_time_alignment_final.json",
    "eval_long_proposal_topk128_modeltau.json",
    "eval_long_proposal_topk128_modeltau_riskpen1.json",
    "eval_long_proposal_topk128_baseline_min10_riskpen1.json",
    "eval_long_proposal_topk128_baseline_min10_riskpen1_allownoop20.json",
]:
    path = save_dir / name
    if not path.exists():
        continue
    data = json.loads(path.read_text())
    item = {}
    for key in [
        "dataset",
        "val",
        "reward_sum",
        "tau_expected",
        "noop_risk",
        "cumulative",
        "chosen_k_histogram",
        "completed_rollout_segments",
        "requested_rollout_segments",
        "stop_reason",
        "skipped_noop",
        "duration_source",
        "planner_tau_source",
        "planner_score_mode",
        "planner_projection_change_source",
        "planner_projection_topk_source",
        "planner_projection_topk_budget",
        "proposal_diagnostic",
        "effective_min_projected_changed_sites",
    ]:
        if key in data:
            item[key] = data[key]
    summary["files"][name] = item
(save_dir / "stage_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
print(f"Saved summary to {save_dir / 'stage_summary.json'}")
PY

echo "=== ${STAGE} DONE $(date -Is) ===" | tee -a "${LOG}"
