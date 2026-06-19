"""Re-score top generated candidates with REAL X2C-PBE0(DF) actinide/lanthanide
binding. For each candidate we estimate the complexation energy to a hard
trivalent cation at fixed (RDKit/MMFF) ligand geometry with the metal placed at
the donor centroid (a vertical/rigid estimate -- fast, and the rigid-geometry
error largely cancels in the An/Ln DIFFERENCE):

  dE(M)  = E[M.L]^(3+) - E[L] - E[M]^3+      (M = La(III,4f0), Ac(III,5f0))
  ddE    = dE(Ac) - dE(La)                    < 0  => An(III)-selective

This ties the generative pipeline back to the relativistic-DFT core (Stage 1) and
gives a genuine, if approximate, An/Ln covalency-driven selectivity number.
"""
import os, sys, json
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from pyscf import gto, dft
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from relbasis import mixed_basis
HARTREE2KJ = 2625.499639
DONORS = {"N", "O", "S", "P"}

def x2c_energy(atoms, charge, spin=0, light="def2-SVP"):
    basis = mixed_basis([a[0] for a in atoms], heavy_basis="SARC-DKH2", light_basis=light)
    mol = gto.M(atom=atoms, basis=basis, charge=charge, spin=spin, verbose=0, max_memory=11000)
    mf = (dft.UKS(mol) if spin else dft.RKS(mol)).x2c().density_fit()
    mf.xc = "PBE0"; mf.conv_tol = 1e-8; mf.max_cycle = 200
    e = mf.kernel()
    return e if mf.converged else np.nan

def ligand_geom(smiles, seed=1):
    m = Chem.MolFromSmiles(smiles)
    if m is None: return None
    m = Chem.AddHs(m)
    if AllChem.EmbedMolecule(m, randomSeed=seed) != 0: return None
    AllChem.MMFFOptimizeMolecule(m, maxIters=500)
    conf = m.GetConformer()
    syms = [a.GetSymbol() for a in m.GetAtoms()]
    xyz = np.array([list(conf.GetAtomPosition(i)) for i in range(m.GetNumAtoms())])
    donors = [i for i,s in enumerate(syms) if s in DONORS]
    return syms, xyz, donors

def metal_position(xyz, donors, k=3):
    # place metal outside the donor centroid, >=2.2 A from the nearest donor
    sel = donors[:k] if len(donors) >= k else donors
    c = xyz[sel].mean(0); com = xyz.mean(0)
    v = c - com; nv = np.linalg.norm(v)
    v = v/nv if nv > 0.3 else np.array([1.0, 0.0, 0.0])
    for d in np.arange(0.0, 3.2, 0.2):
        p = c + d*v
        if min(np.linalg.norm(xyz[i]-p) for i in sel) >= 2.2:
            return p
    return c + 2.4*v

def rescore(smiles, metals=(("La",3),("Ac",3)), max_heavy=24):
    g = ligand_geom(smiles)
    if g is None: return dict(smiles=smiles, error="embed_fail")
    syms, xyz, donors = g
    if not donors or sum(s not in ("H",) for s in syms) > max_heavy:
        return dict(smiles=smiles, error="no_donor_or_too_big",
                    n_heavy=sum(s!="H" for s in syms))
    lig_atoms = [[syms[i], tuple(xyz[i])] for i in range(len(syms))]
    e_L = x2c_energy(lig_atoms, charge=0)
    mpos = metal_position(xyz, donors)
    out = dict(smiles=smiles, n_heavy=sum(s!="H" for s in syms), dE={})
    for M, q in metals:
        e_M = x2c_energy([[M,(0,0,0)]], charge=q)
        cmplx = [[M, tuple(mpos)]] + lig_atoms
        e_C = x2c_energy(cmplx, charge=q)
        out["dE"][M] = float((e_C - e_L - e_M)*HARTREE2KJ)
    if "La" in out["dE"] and "Ac" in out["dE"]:
        out["ddE_AcLa_kJ"] = out["dE"]["Ac"] - out["dE"]["La"]
    return out

if __name__ == "__main__":
    top = json.load(open("results/generative/top_candidates.json"))
    # Focus on small canonical donor motifs: fast AND a chemically meaningful An/Ln
    # trend across hard-O (amide/diglycolamide) vs soft-N (pyridine/triazine) donors.
    refs = ["O=CN", "CC(=O)N", "NC(=O)CO", "NC(=O)COCC(=O)N", "c1ccncc1", "c1cnncn1"]
    results = []
    for i, smi in enumerate((refs + top[:2])):
        print(f"[{i+1}] rescoring {smi}", flush=True)
        try:
            r = rescore(smi); results.append(r)
            print("   ", r.get("dE"), "ddE(Ac-La)=", r.get("ddE_AcLa_kJ"), flush=True)
        except Exception as e:
            print("    FAIL", str(e)[:80]); results.append(dict(smiles=smi, error=str(e)[:80]))
    json.dump(results, open("results/generative/rescore_results.json","w"), indent=2)
    print("RESCORE_DONE", flush=True)
