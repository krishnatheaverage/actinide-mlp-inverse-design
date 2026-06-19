"""Stage 4 driver: train donor-strength surrogate on DFT V_min labels, run
surrogate-guided GB-GA extractant design, and compare against baselines
(GB-GA on a control objective; random-from-prior). Reports the standard de novo
metrics (validity / uniqueness / novelty / internal diversity / SAscore) and saves
top candidates for X2C-DFT actinide/lanthanide re-scoring.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
from extractant_library import load_validated
import gb_ga
from surrogate import DonorSurrogate
from score import make_scorer, molecular_terms, sa_score

CACHE = "results/generative/vmin_labels.json"

def fp(smi):
    m = Chem.MolFromSmiles(smi)
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) if m else None

def internal_diversity(smiles, n=300):
    s = smiles[:n]; fps = [fp(x) for x in s if fp(x)]
    if len(fps) < 2: return 0.0
    sims = []
    for i in range(len(fps)):
        sims += list(DataStructs.BulkTanimotoSimilarity(fps[i], fps[i+1:]))
    return float(1 - np.mean(sims)) if sims else 0.0

def metrics(generated, reference_canon):
    canon = []
    for s in generated:
        m = Chem.MolFromSmiles(s)
        if m: canon.append(Chem.MolToSmiles(m))
    n = len(generated)
    uniq = set(canon)
    novel = [c for c in uniq if c not in reference_canon]
    return dict(n=n, validity=len(canon)/max(n,1),
                uniqueness=len(uniq)/max(len(canon),1),
                novelty=len(novel)/max(len(uniq),1),
                int_div=internal_diversity(list(uniq)))

if __name__ == "__main__":
    labels = {s: v["vmin"] for s, v in json.load(open(CACHE)).items() if "vmin" in v}
    print(f"surrogate training labels: {len(labels)}", flush=True)
    surr = DonorSurrogate()
    surr_metrics = surr.fit(list(labels.keys()), labels)
    print("surrogate (scaffold split):", surr_metrics, flush=True)

    rows, _ = load_validated(verbose=False)
    seeds = [r["smiles"] for r in rows if r["cls"] != "model"]
    known_canon = set(Chem.MolToSmiles(Chem.MolFromSmiles(r["smiles"])) for r in rows)
    train_canon = set(Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in labels)
    reference = known_canon | train_canon

    scorer = make_scorer(surr, objective="extractant")
    control = make_scorer(surr, objective="control_logp")

    print("\n=== GB-GA: extractant objective ===", flush=True)
    recs_ext, hist_ext = gb_ga.run_ga(seeds, scorer, pop_size=80, n_gen=25, rng_seed=1)
    print("\n=== GB-GA: control objective (logP only) ===", flush=True)
    recs_ctl, hist_ctl = gb_ga.run_ga(seeds, control, pop_size=80, n_gen=25, rng_seed=1)

    # random-from-prior baseline: mutate seeds without selection
    import random; random.seed(0)
    mols = [Chem.MolFromSmiles(s) for s in seeds]
    rand_lib = []
    while len(rand_lib) < 1500:
        c = gb_ga.mutate(random.choice(mols))
        if c: rand_lib.append(Chem.MolToSmiles(c))

    ext_lib = [r["smiles"] for r in recs_ext]
    ctl_lib = [r["smiles"] for r in recs_ctl]
    res = dict(
        surrogate=surr_metrics,
        metrics_extractant=metrics(ext_lib, reference),
        metrics_control=metrics(ctl_lib, reference),
        metrics_random=metrics(rand_lib, reference),
        history_extractant=hist_ext, history_control=hist_ctl,
    )

    # characterise top extractant candidates (real scored properties)
    top = []
    for r in recs_ext[:15]:
        t = molecular_terms(r["smiles"])
        top.append(dict(smiles=r["smiles"], score=r["score"],
                        vmin_pred=surr.predict(r["smiles"]),
                        n_donor=t["n_donor"], logp=round(t["logp"],2),
                        nF=t["nF"], mw=round(t["mw"],1), sa=round(t["sa"],2),
                        novel=Chem.MolToSmiles(Chem.MolFromSmiles(r["smiles"])) not in reference))
    res["top_candidates"] = top
    # summary: mean donor strength of best-10 per objective
    def mean_donor(recs):
        vs = [surr.predict(r["smiles"]) for r in recs[:10]]
        vs = [v for v in vs if v is not None]
        return float(np.mean(vs))
    res["mean_vmin_top10_extractant"] = mean_donor(recs_ext)
    res["mean_vmin_top10_control"] = mean_donor(recs_ctl)

    json.dump(res, open("results/generative/generative_results.json", "w"), indent=2)
    print("\n=== top extractant candidates ===")
    for t in top[:10]:
        print(f"  score={t['score']:.2f} vmin_pred={t['vmin_pred']:.3f} "
              f"donors={t['n_donor']} logP={t['logp']} F={t['nF']} SA={t['sa']} "
              f"novel={t['novel']}  {t['smiles']}")
    json.dump([t["smiles"] for t in top], open("results/generative/top_candidates.json","w"))
    print("GENERATIVE_DONE", flush=True)
