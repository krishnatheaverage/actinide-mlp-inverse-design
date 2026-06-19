#!/bin/bash
# Wait for the review-cycle refinement computes, run T1, fill all numbers, rebuild.
cd /Users/krishnaharish/actinide-mlp-inverse-design
PY="$HOME/miniforge3/envs/actinide/bin/python"
export KMP_DUPLICATE_LIB_OK=TRUE
log(){ echo "[final $(date +%H:%M:%S)] $*"; }

log "waiting for rescore / fine-FE / decontracted to finish..."
while pgrep -f "run_rescore.py" >/dev/null || pgrep -f "fine_fe.py" >/dev/null || pgrep -f "rel_effect_decontracted.py" >/dev/null; do
  sleep 30
done
log "refinement computes done; running T1 diagnostic (CCSD)"
OMP_NUM_THREADS=8 "$PY" src/01_relativistic_dft/t1_diagnostic.py > results/dft/t1.log 2>&1
log "T1 rc=$?"

"$PY" src/common/fill_macros.py > results/macros_fill.log 2>&1
"$PY" src/common/make_figures.py > results/figures_make.log 2>&1
cd paper && tectonic main.tex > /dev/null 2>&1
log "FINALIZE_DONE pdf=$(ls -la main.pdf | awk '{print $5}') bytes"
