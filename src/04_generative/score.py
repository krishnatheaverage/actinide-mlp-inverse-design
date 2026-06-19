import os, sys
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, RDConfig

try:
    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer
    def sa_score(m): return sascorer.calculateScore(m)
except Exception:
    def sa_score(m):
        return 1 + 0.005*m.GetNumHeavyAtoms() + Descriptors.NumRotatableBonds(m)*0.05

_DONOR = Chem.MolFromSmarts("[$([O,N,F,S,P;!$([N+]);!$([O+])])]")

def molecular_terms(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None: return None
    nd = len(m.GetSubstructMatches(_DONOR))
    logp = Crippen.MolLogP(m)
    nF = sum(a.GetSymbol() == "F" for a in m.GetAtoms())
    mw = Descriptors.MolWt(m)
    return dict(m=m, n_donor=nd, logp=logp, nF=nF, mw=mw, sa=sa_score(m),
                n_heavy=m.GetNumHeavyAtoms())

def make_scorer(surrogate, weights=None, objective="extractant"):
    w = dict(donor=10.0, dent=0.35, co2=1.0, sa=0.15)
    if weights: w.update(weights)

    def co2_philicity(logp, nF):

        win = np.exp(-0.5*((logp - 4.0)/2.5)**2)
        fluor = np.tanh(nF/6.0)
        return win + 0.6*fluor

    def denticity(nd):
        return min(nd, 6) - 0.4*max(0, nd-6)

    def size_penalty(mw, nh):
        p = 0.0
        if mw > 700: p += (mw-700)/200.0
        if nh < 6:   p += (6-nh)*0.5
        return p

    def scorer(smiles):
        t = molecular_terms(smiles)
        if t is None: return -1e6
        if objective == "control_logp":
            return t["logp"] - 0.1*t["sa"]
        vmin = surrogate.predict(smiles) if surrogate else -0.05
        if vmin is None: return -1e6
        donor_strength = -vmin
        s = (w["donor"]*donor_strength
             + w["dent"]*denticity(t["n_donor"])
             + w["co2"]*co2_philicity(t["logp"], t["nF"])
             - w["sa"]*max(0, t["sa"]-3.0)
             - size_penalty(t["mw"], t["n_heavy"]))
        return float(s)
    return scorer
