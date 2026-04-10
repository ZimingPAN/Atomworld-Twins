#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════
# Evaluate v24 and v25 after training completes
# Run on GPU server after training is done
# ═══════════════════════════════════════════════════════════

set -e
cd /home/likun/panziming/SwarmEcosystem/dreamer4-main

echo "═══════════════════════════════════════════════════════════"
echo "Evaluating v24 and v25"
echo "═══════════════════════════════════════════════════════════"

# ─── v24 eval (val split, on KMC teacher segments) ───
echo "=== v24 eval (val split) ==="
python eval_macro_time_alignment.py \
    --checkpoint results/dreamer_macro_edit_v24_energy_fix/best_model.pt \
    --cache results/dreamer_macro_edit_v15/segments.pt \
    --split val --save_all_samples \
    --output results/dreamer_macro_edit_v24_energy_fix/eval_full_samples.json \
    2>&1 | tee results/dreamer_macro_edit_v24_energy_fix/eval.log

# ─── v25 eval (val split, on its own neural-teacher segments) ───
echo "=== v25 eval (val split, neural teacher segments) ==="
python eval_macro_time_alignment.py \
    --checkpoint results/dreamer_macro_edit_v25_neural_teacher/best_model.pt \
    --cache results/dreamer_macro_edit_v25_neural_teacher/segments.pt \
    --split val --save_all_samples \
    --output results/dreamer_macro_edit_v25_neural_teacher/eval_full_samples.json \
    2>&1 | tee results/dreamer_macro_edit_v25_neural_teacher/eval.log

# ─── v25 cross-eval (on KMC segments for fair comparison) ───
echo "=== v25 cross-eval (KMC segments) ==="
python eval_macro_time_alignment.py \
    --checkpoint results/dreamer_macro_edit_v25_neural_teacher/best_model.pt \
    --cache results/dreamer_macro_edit_v15/segments.pt \
    --split val --save_all_samples \
    --output results/dreamer_macro_edit_v25_neural_teacher/eval_kmc_segments.json \
    2>&1 | tee results/dreamer_macro_edit_v25_neural_teacher/eval_kmc.log

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "Evaluation complete. Results:"
echo "  v24: results/dreamer_macro_edit_v24_energy_fix/eval_full_samples.json"
echo "  v25: results/dreamer_macro_edit_v25_neural_teacher/eval_full_samples.json"
echo "  v25 (KMC): results/dreamer_macro_edit_v25_neural_teacher/eval_kmc_segments.json"
echo ""
echo "Copy results back to local machine:"
echo "  scp -r likun@server:~/panziming/SwarmEcosystem/dreamer4-main/results/dreamer_macro_edit_v24_energy_fix/ results/"
echo "  scp -r likun@server:~/panziming/SwarmEcosystem/dreamer4-main/results/dreamer_macro_edit_v25_neural_teacher/ results/"
echo "═══════════════════════════════════════════════════════════"
