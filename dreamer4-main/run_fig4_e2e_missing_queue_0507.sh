#!/usr/bin/env bash
set -u

BASE="/home/likun/panziming/AtomWorld-Twins/dreamer4-main"
PYTHON="/home/likun/panziming/AtomWorld-Twins/conda/bin/python3"
PYTHONPATH_VALUE="/home/likun/panziming/pydeps:/home/likun/panziming/AtomWorld-Twins/kmcteacher_backend:/home/likun/panziming/AtomWorld-Twins/LightZero-main:/home/likun/panziming/AtomWorld-Twins/dreamer4-main"
RESULT_DIR="results/dreamer_macro_edit_v52_planner_selected_duration_covered_aug_1024"
CHECKPOINT="${RESULT_DIR}/final_model.pt"
MAX_FIG4_PROCS=2
SLEEP_SECONDS=180

cd "${BASE}" || exit 1

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*"
}

json_complete() {
  local output="$1"
  "${PYTHON}" - "$output" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
if not path.exists():
    raise SystemExit(1)
try:
    data = json.loads(path.read_text())
except Exception:
    raise SystemExit(1)
if data.get("complete") is True:
    raise SystemExit(0)
total = data.get("total_cases")
completed = data.get("completed_cases")
rows = data.get("rows", [])
if total is not None and completed == total and len(rows) == total:
    raise SystemExit(0)
raise SystemExit(1)
PY
}

output_running() {
  local output="$1"
  pgrep -f "benchmark_fig4_case_grid.py .*--output ${output}" >/dev/null 2>&1
}

fig4_proc_count() {
  pgrep -fc "benchmark_fig4_case_grid.py" || true
}

wait_for_slot() {
  while true; do
    local count
    count="$(fig4_proc_count)"
    if [ "${count}" -lt "${MAX_FIG4_PROCS}" ]; then
      return 0
    fi
    log "waiting: ${count} fig4 benchmark processes active; max=${MAX_FIG4_PROCS}"
    free -h | sed -n '1,2p'
    sleep "${SLEEP_SECONDS}"
  done
}

run_case() {
  local label="$1"
  local temp="$2"
  local edge="$3"
  local seed="$4"
  local gpu="$5"
  local output="${RESULT_DIR}/benchmark_fig4_case_grid_cu005_end_to_end_t${temp}_l${edge}_k8_s1_r1_only_0507.json"
  local logfile="${RESULT_DIR}/benchmark_fig4_case_grid_cu005_end_to_end_t${temp}_l${edge}_k8_s1_r1_only_0507.log"

  if json_complete "${output}"; then
    log "skip ${label}: complete JSON exists at ${output}"
    return 0
  fi

  while output_running "${output}"; do
    log "wait ${label}: existing process is still writing ${output}"
    sleep "${SLEEP_SECONDS}"
    if json_complete "${output}"; then
      log "skip ${label}: existing process completed ${output}"
      return 0
    fi
  done

  wait_for_slot
  log "start ${label}: T=${temp}K lattice=${edge}^3 seed=${seed} gpu=${gpu} output=${output}"
  free -h | sed -n '1,2p'
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${PYTHONPATH_VALUE}" "${PYTHON}" benchmark_fig4_case_grid.py \
    --checkpoint "${CHECKPOINT}" \
    --output "${output}" \
    --cu_densities 0.005 \
    --temperatures "${temp}" \
    --lattice_edges "${edge}" \
    --segment_k 8 \
    --samples 1 \
    --teacher_repeats 1 \
    --teacher_warmup_segments 0 \
    --macro_repeats 1 \
    --macro_warmup 0 \
    --candidate_limit 128 \
    --macro_timing_scope end_to_end \
    --device cuda \
    --amp_dtype bfloat16 \
    --seed "${seed}" 2>&1 | tee -a "${logfile}"
  local status=${PIPESTATUS[0]}
  if json_complete "${output}"; then
    log "complete ${label}: ${output}"
  else
    log "incomplete ${label}: exit=${status}, output not complete"
  fi
  return 0
}

log "fig4 missing-case queue started"

run_case "800_t263" 263 800 3000 1
run_case "800_t333" 333 800 23000 3
run_case "800_t373" 373 800 33000 4
run_case "1000_t293" 293 1000 14000 5
run_case "1000_t333" 333 1000 24000 6
run_case "1000_t373" 373 1000 34000 7
run_case "800_t293_retry_if_needed" 293 800 13000 2
run_case "1000_t263_retry_if_needed" 263 1000 4000 0

log "fig4 missing-case queue finished"
