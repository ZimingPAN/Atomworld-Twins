#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="/home/likun-share/panziming/pydeps:/home/likun-share/panziming/AtomWorld-Mirror/kmcteacher_backend:/home/likun-share/panziming/AtomWorld-Mirror/LightZero-main:/home/likun-share/panziming/AtomWorld-Mirror/dreamer4-main:${PYTHONPATH:-}"

PYTHON="${PYTHON:-/home/likun-share/panziming/AtomWorld-Mirror/.venv/bin/python}"
STAGE="${STAGE:-v82_action_edge_support_diag20}"
OUT_DIR="results/natural_teacher_support32_actionproposal_0516/${STAGE}"
CHECKPOINT="${CHECKPOINT:-results/natural_teacher_support32_actionproposal_0516/v81_independent_action_support_smoke1b/final_model.pt}"
SAMPLES_PER_K="${SAMPLES_PER_K:-8}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "${OUT_DIR}"
{
  echo "=== ${STAGE} START $(date -Iseconds) ==="
  echo "checkpoint=${CHECKPOINT}"
  echo "samples_per_k=${SAMPLES_PER_K}"
  "${PYTHON}" diagnose_action_edge_support.py \
    --checkpoint "${CHECKPOINT}" \
    --output "${OUT_DIR}/stage_summary.json" \
    --segment_ks 8 16 32 \
    --samples_per_k "${SAMPLES_PER_K}" \
    --topk_budgets 32 64 96 128 256 \
    --sources change proposal action_support \
    --seed 0 \
    --device "${DEVICE}" \
    --lattice_size 40 40 40 \
    --cu_density 0.0134 \
    --v_density 0.0002 \
    --max_episode_steps 1024 \
    --max_seed_vacancies 32 \
    --max_candidate_sites 2048 \
    --segment_boundary_mode adaptive_key_event \
    --adaptive_min_k 8 \
    --adaptive_candidate_horizon_source actual
  echo "=== ${STAGE} DONE $(date -Iseconds) ==="
} 2>&1 | tee "${OUT_DIR}/pipeline.log"
