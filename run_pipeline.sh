#!/bin/bash
# Autonomous orchestration of the remaining pipeline. Run detached (nohup).
cd /Users/krishnaharish/actinide-mlp-inverse-design
PY="$HOME/miniforge3/envs/actinide/bin/python"
export KMP_DUPLICATE_LIB_OK=TRUE   # torch+conda libomp clash workaround

log(){ echo "[pipeline $(date +%H:%M:%S)] $*"; }

log "waiting for MLP dataset (DATASET_DONE)..."
while ! grep -q DATASET_DONE results/mlp/gen_dataset.log 2>/dev/null; do
  pgrep -f "gen_dataset.py" >/dev/null || { grep -q DATASET_DONE results/mlp/gen_dataset.log 2>/dev/null || log "WARN dataset process gone"; break; }
  sleep 30
done
log "dataset stage finished ($(grep -c 'energy=' results/mlp/uo2_dataset.extxyz) frames)"

# Stage 4 oracle (CPU) in background; MACE training (GPU/MPS) in foreground
log "launching V_min oracle (CPU) + MACE training (GPU)"
NWORK=5 OMP_NUM_THREADS=1 nohup "$PY" src/04_generative/build_surrogate_data.py > results/generative/oracle.log 2>&1 &
OPID=$!
OMP_NUM_THREADS=4 "$PY" src/02_mlp/train_mace.py > results/mlp/train_mace.log 2>&1
log "MACE training+eval finished (rc=$?)"
wait $OPID
log "oracle finished"

log "generative design (GB-GA + baselines + metrics)"
OMP_NUM_THREADS=4 "$PY" src/04_generative/run_generative.py > results/generative/generative.log 2>&1
log "generative finished (rc=$?)"

log "X2C-DFT An/Ln rescore of top candidates"
OMP_NUM_THREADS=6 "$PY" src/04_generative/run_rescore.py > results/generative/rescore.log 2>&1
log "rescore finished (rc=$?)"

log "regenerating macros + figures"
"$PY" src/common/fill_macros.py > /dev/null 2>&1
"$PY" src/common/make_figures.py > /dev/null 2>&1
log "PIPELINE_DONE"
