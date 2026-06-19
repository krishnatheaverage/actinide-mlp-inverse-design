"""Build the surrogate training set: assemble a diverse ligand pool (real
extractants + RDKit/GB-GA-generated analogues + donor fragments), then label each
with the DFT V_min donor-strength descriptor. Parallel over CPU cores, resumable
(caches to results/generative/vmin_labels.json).
"""
import os, sys, json, random
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(__file__))
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
from extractant_library import load_validated
import gb_ga
from dft_descriptor import vmin_descriptor

CACHE = "results/generative/vmin_labels.json"
MAX_HEAVY = 18          # cap for DFT affordability
N_AUG = 60              # GB-GA analogues to add

def assemble_pool(seed=3):
    random.seed(seed)
    rows, _ = load_validated(verbose=False)
    pool = set()
    for r in rows:                                   # real extractants + fragments
        m = Chem.MolFromSmiles(r["smiles"])
        if m and m.GetNumHeavyAtoms() <= MAX_HEAVY:
            pool.add(r["smiles"])
    seeds = [r["smiles"] for r in rows if r["cls"] != "model"]
    mols = [Chem.MolFromSmiles(s) for s in seeds]
    tries = 0
    while len([p for p in pool]) < len(rows) + N_AUG and tries < N_AUG*40:
        tries += 1
        if random.random() < 0.5:
            c = gb_ga.mutate(random.choice(mols))
        else:
            a, b = random.sample(mols, 2); c = gb_ga.crossover(a, b)
        if c is None: continue
        if c.GetNumHeavyAtoms() > MAX_HEAVY or c.GetNumHeavyAtoms() < 4: continue
        pool.add(Chem.MolToSmiles(c))
    return sorted(pool)

def _label(smi):
    try:
        r = vmin_descriptor(smi)
        if r and "vmin" in r:
            return smi, r
    except Exception:
        pass
    return smi, None

if __name__ == "__main__":
    os.makedirs("results/generative", exist_ok=True)
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    pool = assemble_pool()
    todo = [s for s in pool if s not in cache]
    print(f"pool={len(pool)} cached={len(cache)} todo={len(todo)}", flush=True)
    # SEQUENTIAL (robust on macOS; multiprocessing.Pool + PySCF spawn is fragile).
    # PySCF uses OMP threads per molecule for intra-molecule parallelism.
    import time; t0 = time.time()
    for i, smi in enumerate(todo):
        _, r = _label(smi)
        if r is not None:
            cache[smi] = r
        if i % 5 == 0:
            json.dump(cache, open(CACHE, "w"))
            print(f"  labelled {i+1}/{len(todo)} ok={len(cache)} ({time.time()-t0:.0f}s)", flush=True)
    json.dump(cache, open(CACHE, "w"))
    vmins = [v["vmin"] for v in cache.values() if "vmin" in v]
    print(f"DONE: {len(cache)} labelled; V_min range [{min(vmins):.3f},{max(vmins):.3f}] a.u.",
          flush=True)
