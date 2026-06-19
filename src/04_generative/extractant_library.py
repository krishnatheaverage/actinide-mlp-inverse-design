"""
Curated library of REAL f-element extractant ligands (literature molecules),
as SMILES with name + process class. Hand-curated from the solvent-extraction
literature (PUREX/TRUEX/DIAMEX/DGA/SANEX/TALSPEAK families) and RDKit-validated
at load time. Anchor identities are best-effort literature SMILES; the loader
canonicalises and drops anything RDKit cannot parse, and reports the result.

These serve as (a) the seed population / prior for the generator and
(b) the "known scaffold" set against which generated-ligand NOVELTY is measured.
"""
from rdkit import Chem
from rdkit.Chem import Descriptors

# name, process/class, SMILES (donor set noted in comments)
EXTRACTANTS = [
    # --- neutral organophosphorus (PUREX / TRUEX) : P=O O-donor ---
    ("TBP",        "PUREX",   "O=P(OCCCC)(OCCCC)OCCCC"),
    ("TiBP",       "PUREX",   "O=P(OCC(C)C)(OCC(C)C)OCC(C)C"),
    ("TEHP",       "neutral-OP","O=P(OCC(CC)CCCC)(OCC(CC)CCCC)OCC(CC)CCCC"),
    ("TOPO",       "neutral-OP","O=P(CCCCCCCC)(CCCCCCCC)CCCCCCCC"),
    ("CMPO",       "TRUEX",   "CC(C)CN(CC(C)C)C(=O)CP(=O)(c1ccccc1)CCCCCCCC"),
    # --- acidic organophosphorus (REE / TALSPEAK) ---
    ("HDEHP",      "acidic-OP","CCCCC(CC)COP(=O)(O)OCC(CC)CCCC"),
    ("HEH/EHP",    "acidic-OP","CCCCC(CC)CP(=O)(O)OCC(CC)CCCC"),
    ("Cyanex272",  "acidic-OP","CC(C)(C)CC(C)CP(=O)(O)CC(C)CC(C)(C)C"),
    # --- diamides / malonamides (DIAMEX) : C=O O-donor ---
    ("DMDBTDMA",   "DIAMEX",  "CCCCN(C)C(=O)C(CCCCCCCCCCCC)C(=O)N(C)CCCC"),
    ("DMDOHEMA",   "DIAMEX",  "CCCCCCCCN(C)C(=O)C(CCOCCCCCC)C(=O)N(C)CCCCCCCC"),
    # --- diglycolamides (DGA) : O,O,O tridentate ---
    ("TODGA",      "DGA",     "CCCCCCCCN(CCCCCCCC)C(=O)COCC(=O)N(CCCCCCCC)CCCCCCCC"),
    ("T2EHDGA",    "DGA",     "CCCCC(CC)CN(CC(CC)CCCC)C(=O)COCC(=O)N(CC(CC)CCCC)CC(CC)CCCC"),
    ("TEDGA",      "DGA",     "CCN(CC)C(=O)COCC(=O)N(CC)CC"),
    # --- soft-N donors (SANEX) : triazinyl-pyridine / -bipyridine ---
    ("n-Pr-BTP",   "SANEX",   "CCCc1nnc(-c2cccc(-c3nnc(CCC)c(CCC)n3)n2)c(CCC)n1"),
    ("CyMe4-BTBP", "SANEX",   "CC1(C)CCC(C)(C)c2nc(-c3ccc(-c4ccc(-c5nc6c(nn5)C(C)(C)CCC6(C)C)nc4)nc3)nnc21"),
    ("CyMe4-BTPhen","SANEX",  "CC1(C)CCC(C)(C)c2nc(-c3nnc(-c4ccc5ccc6cccnc6c5n4)c4nnc(-c5nc7c(nn5)C(C)(C)CCC7(C)C)nc34)nc21"),
    # --- aqueous holdback / hydrophilic donors ---
    ("DTPA",       "TALSPEAK","OC(=O)CN(CC(=O)O)CCN(CC(=O)O)CCN(CC(=O)O)CC(=O)O"),
    ("SO3-Ph-BTP", "i-SANEX", "CCc1nnc(-c2cccc(-c3nnc(CC)c(-c4ccc(S(=O)(=O)O)cc4)n3)n2)c1-c1ccc(S(=O)(=O)O)cc1"),
    # --- small donor fragments / models (also used as DFT-tractable donors) ---
    ("formamide",  "model",   "O=CN"),
    ("acetamide",  "model",   "CC(=O)N"),
    ("formate",    "model",   "[O-]C=O"),
    ("acetate",    "model",   "CC(=O)[O-]"),
    ("glycolamide","model",   "NC(=O)CO"),
    ("diglycolamide-core","model","O=C(N)COCC(=O)N"),
    ("nitrate",    "model",   "[O-][N+]([O-])=O"),
    ("pyridine",   "model",   "c1ccncc1"),
    ("bipyridine", "model",   "c1ccc(-c2ccccn2)nc1"),
    ("triazine",   "model",   "c1cnncn1"),
]


def load_validated(verbose=True):
    """Return list of dicts with RDKit-canonical SMILES; drop unparsable."""
    out, dropped = [], []
    for name, cls, smi in EXTRACTANTS:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            dropped.append((name, smi)); continue
        can = Chem.MolToSmiles(m)
        out.append(dict(
            name=name, cls=cls, smiles=can,
            mw=round(Descriptors.MolWt(m), 1),
            n_N=sum(a.GetSymbol() == "N" for a in m.GetAtoms()),
            n_O=sum(a.GetSymbol() == "O" for a in m.GetAtoms()),
            n_P=sum(a.GetSymbol() == "P" for a in m.GetAtoms()),
            n_F=sum(a.GetSymbol() == "F" for a in m.GetAtoms()),
            logp=round(Descriptors.MolLogP(m), 2),
        ))
    if verbose:
        print(f"Loaded {len(out)} valid extractants; dropped {len(dropped)}: "
              f"{[d[0] for d in dropped]}")
    return out, dropped


if __name__ == "__main__":
    rows, dropped = load_validated()
    for r in rows:
        print(f"{r['name']:18s} {r['cls']:10s} MW={r['mw']:7.1f} "
              f"N{r['n_N']} O{r['n_O']} P{r['n_P']} F{r['n_F']} logP={r['logp']:6.2f}  {r['smiles']}")
