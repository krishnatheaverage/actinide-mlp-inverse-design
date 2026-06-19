"""
Graph-based genetic algorithm for molecules (Jensen, Chem. Sci. 10, 3567, 2019),
reimplemented on modern RDKit using FragmentOnBonds + molzip for crossover.

This is the field-standard strong baseline that any de novo generator must beat
(Reviewer-2 demands it). Used here both as a baseline AND as the search engine
driven by the extractant scoring function.
"""
import random
import numpy as np
from rdkit import Chem
from rdkit.Chem import RWMol, BRICS
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

# ---------- mutation operators (reaction SMARTS) ----------
_MUT_RXNS = [
    # append atom
    "[*;!H0:1]>>[*:1]-[C]", "[*;!H0:1]>>[*:1]-[N]", "[*;!H0:1]>>[*:1]-[O]",
    "[*;!H0:1]>>[*:1]=[C]", "[*;!H0:1]>>[*:1]-[F]",
    # insert atom into a bond
    "[*:1]-[*:2]>>[*:1]-[C]-[*:2]", "[*:1]-[*:2]>>[*:1]-[N]-[*:2]",
    "[*:1]-[*:2]>>[*:1]-[O]-[*:2]",
    # change atom identity
    "[#6:1]>>[#7:1]", "[#7:1]>>[#6:1]", "[#6:1]>>[#8:1]", "[#8:1]>>[#6:1]",
    # change bond order
    "[*:1]-[*:2]>>[*:1]=[*:2]", "[*:1]=[*:2]>>[*:1]-[*:2]",
    # delete a terminal atom
    "[*:1]-[*;D1]>>[*:1]",
]
from rdkit.Chem import AllChem
_MUT = [AllChem.ReactionFromSmarts(s) for s in _MUT_RXNS]


def _sane(mol):
    if mol is None:
        return None
    try:
        smi = Chem.MolToSmiles(mol)
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        # reject radicals / disconnected / too small/large
        if "." in smi:
            return None
        n = m.GetNumAtoms()
        if n < 3 or n > 70:
            return None
        if any(a.GetNumRadicalElectrons() for a in m.GetAtoms()):
            return None
        return m
    except Exception:
        return None


def mutate(mol, n_try=10):
    for _ in range(n_try):
        rxn = random.choice(_MUT)
        try:
            prods = rxn.RunReactants((mol,))
        except Exception:
            continue
        if not prods:
            continue
        cand = _sane(random.choice(prods)[0])
        if cand is not None:
            return cand
    return None


def crossover(a, b, n_try=10):
    """Cut one non-ring single bond in each parent, recombine via molzip."""
    for _ in range(n_try):
        try:
            fa = _cut(a)
            fb = _cut(b)
            if fa is None or fb is None:
                continue
            # take one piece from each and zip the dummy ends
            pa = random.choice(fa)
            pb = random.choice(fb)
            combo = Chem.CombineMols(pa, pb)
            params = Chem.MolzipParams()
            params.label = Chem.MolzipLabel.Isotope
            zipped = Chem.molzip(combo, params)
            cand = _sane(zipped)
            if cand is not None:
                return cand
        except Exception:
            continue
    return None


def _cut(mol):
    bonds = [bd.GetIdx() for bd in mol.GetBonds()
             if not bd.IsInRing()
             and bd.GetBondType() == Chem.BondType.SINGLE]
    if not bonds:
        return None
    bidx = random.choice(bonds)
    # fragment with dummy atoms labelled so molzip can rejoin
    frag = Chem.FragmentOnBonds(mol, [bidx], addDummies=True, dummyLabels=[(1, 1)])
    pieces = Chem.GetMolFrags(frag, asMols=True, sanitizeFrags=False)
    good = []
    for p in pieces:
        try:
            Chem.SanitizeMol(p)
            good.append(p)
        except Exception:
            pass
    return good or None


def run_ga(seed_smiles, score_fn, pop_size=60, n_gen=20, mut_rate=0.4,
           elitism=0.2, rng_seed=0, log=print):
    """Evolve a population maximising score_fn(smiles)->float.

    Returns (best_records, history) where best_records is a list of
    dicts {smiles, score} sorted descending, and history is per-gen best/mean.
    """
    random.seed(rng_seed); np.random.seed(rng_seed)
    pop = []
    seen = set()
    for smi in seed_smiles:
        m = Chem.MolFromSmiles(smi)
        if m:
            cs = Chem.MolToSmiles(m)
            if cs not in seen:
                seen.add(cs); pop.append(m)
    # pad population by mutating seeds
    while len(pop) < pop_size:
        base = random.choice(pop)
        c = mutate(base)
        if c:
            cs = Chem.MolToSmiles(c)
            if cs not in seen:
                seen.add(cs); pop.append(c)
        if len(seen) > pop_size * 50:
            break

    def evaluate(mols):
        out = []
        for m in mols:
            smi = Chem.MolToSmiles(m)
            try:
                s = score_fn(smi)
            except Exception:
                s = -1e9
            out.append((m, smi, s))
        return out

    scored = evaluate(pop)
    history = []
    all_seen = {smi: s for (_, smi, s) in scored}
    for gen in range(n_gen):
        scored.sort(key=lambda t: t[2], reverse=True)
        n_elite = max(1, int(elitism * pop_size))
        elites = scored[:n_elite]
        # fitness-proportional-ish selection from top half
        parents = [t for t in scored[:max(2, pop_size // 2)]]
        children = list(elites)
        guard = 0
        while len(children) < pop_size and guard < pop_size * 40:
            guard += 1
            if random.random() < mut_rate or len(parents) < 2:
                p = random.choice(parents)[0]
                c = mutate(p)
            else:
                pa, pb = random.sample(parents, 2)
                c = crossover(pa[0], pb[0])
                if c and random.random() < 0.5:
                    c = mutate(c) or c
            if c is None:
                continue
            cs = Chem.MolToSmiles(c)
            if cs in all_seen:
                continue
            try:
                s = score_fn(cs)
            except Exception:
                continue
            all_seen[cs] = s
            children.append((c, cs, s))
        scored = children
        best = max(scored, key=lambda t: t[2])
        mean = float(np.mean([t[2] for t in scored]))
        history.append(dict(gen=gen, best=best[2], mean=mean, best_smiles=best[1]))
        log(f"  gen {gen:2d}  best={best[2]:.3f}  mean={mean:.3f}  {best[1]}")

    recs = sorted([dict(smiles=s, score=v) for s, v in all_seen.items()],
                  key=lambda d: d["score"], reverse=True)
    return recs, history


if __name__ == "__main__":
    # self-test: do crossover/mutation actually produce valid novel molecules?
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from extractant_library import load_validated
    rows, _ = load_validated()
    seeds = [r["smiles"] for r in rows if r["cls"] != "model"]
    print(f"{len(seeds)} non-model seeds")
    n_mut_ok = n_cx_ok = 0
    mols = [Chem.MolFromSmiles(s) for s in seeds]
    for _ in range(200):
        m = mutate(random.choice(mols))
        if m: n_mut_ok += 1
    for _ in range(200):
        a, b = random.sample(mols, 2)
        c = crossover(a, b)
        if c: n_cx_ok += 1
    print(f"mutate success {n_mut_ok}/200, crossover success {n_cx_ok}/200")
    # tiny GA driven by a trivial score (maximise #N donors - heavy penalty on size)
    from rdkit.Chem import Descriptors
    def toy_score(smi):
        m = Chem.MolFromSmiles(smi)
        nN = sum(a.GetSymbol() == "N" for a in m.GetAtoms())
        return nN - 0.02 * Descriptors.MolWt(m)
    recs, hist = run_ga(seeds, toy_score, pop_size=40, n_gen=6)
    print("top toy:", recs[0])
