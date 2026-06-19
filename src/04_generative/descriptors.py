import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen

FEATURE_NAMES = [
    "MW","logP","TPSA","HBA","HBD","nRot","nAromRings","nRings","FracCsp3",
    "nN","nO","nF","nP","nS","nDonorAtoms","nCarbonyl","nAmide","nEther",
    "maxAbsPartialChargeGasteiger","nHeavy","nHeteroatoms",
]

_CARBONYL = Chem.MolFromSmarts("[CX3]=[OX1]")
_AMIDE    = Chem.MolFromSmarts("[NX3][CX3]=[OX1]")
_ETHER    = Chem.MolFromSmarts("[OD2]([#6])[#6]")
_DONOR    = Chem.MolFromSmarts("[$([O,N,F,S,P;!$([N+]);!$([O+])])]")

def featurize(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    try:
        from rdkit.Chem import AllChem
        AllChem.ComputeGasteigerCharges(m)
        gast = max(abs(float(a.GetPropsAsDict().get("_GasteigerCharge", 0.0)))
                   for a in m.GetAtoms())
        if not np.isfinite(gast): gast = 0.0
    except Exception:
        gast = 0.0
    def cnt(sym): return sum(a.GetSymbol() == sym for a in m.GetAtoms())
    f = [
        Descriptors.MolWt(m), Crippen.MolLogP(m), rdMolDescriptors.CalcTPSA(m),
        rdMolDescriptors.CalcNumHBA(m), rdMolDescriptors.CalcNumHBD(m),
        rdMolDescriptors.CalcNumRotatableBonds(m),
        rdMolDescriptors.CalcNumAromaticRings(m), rdMolDescriptors.CalcNumRings(m),
        rdMolDescriptors.CalcFractionCSP3(m),
        cnt("N"), cnt("O"), cnt("F"), cnt("P"), cnt("S"),
        len(m.GetSubstructMatches(_DONOR)),
        len(m.GetSubstructMatches(_CARBONYL)), len(m.GetSubstructMatches(_AMIDE)),
        len(m.GetSubstructMatches(_ETHER)),
        gast, m.GetNumHeavyAtoms(),
        sum(a.GetAtomicNum() not in (1, 6) for a in m.GetAtoms()),
    ]
    return np.array(f, float)

def featurize_many(smiles_list):
    X, keep = [], []
    for s in smiles_list:
        v = featurize(s)
        if v is not None and np.all(np.isfinite(v)):
            X.append(v); keep.append(s)
    return np.array(X), keep
