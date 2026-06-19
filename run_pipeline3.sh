#!/bin/bash
# Final stages: oracle -> generative -> rescore. MACE + dataset already done.
cd /Users/krishnaharish/actinide-mlp-inverse-design
PY="$HOME/miniforge3/envs/actinide/bin/python"
export KMP_DUPLICATE_LIB_OK=TRUE
log(){ echo "[pipe3 $(date +%H:%M:%S)] $*"; }

log "V_min surrogate oracle (~246 ligands, resumes from cache)"
OMP_NUM_THREADS=4 "$PY" src/04_generative/build_surrogate_data.py > results/generative/oracle.log 2>&1
log "oracle done (rc=$?)"

log "generative design (GB-GA + baselines + metrics)"
OMP_NUM_THREADS=4 "$PY" src/04_generative/run_generative.py > results/generative/generative.log 2>&1
log "generative done (rc=$?)"

log "X2C-DFT An/Ln rescore of top candidates"
OMP_NUM_THREADS=8 "$PY" src/04_generative/run_rescore.py > results/generative/rescore.log 2>&1
log "rescore done (rc=$?)"

"$PY" src/common/fill_macros.py > results/macros_fill.log 2>&1
"$PY" src/common/make_figures.py > results/figures_make.log 2>&1
log "PIPELINE3_DONE"
