"""Reviewer-1 response: a real T1 coupled-cluster diagnostic for uranyl, to TEST
(not assert) the adequacy of single-reference treatment for the closed-shell f0
actinyl. T1 << 0.02 supports single-reference; larger flags multireference need.
Frozen-core DF-CCSD on the X2C-RHF reference, def2-SVP ligand basis for tractability.
"""
import json, numpy as np
from pyscf import gto, scf, cc
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from relbasis import mixed_basis

atoms = [["U",(0,0,0)],["O",(0,0,1.685)],["O",(0,0,-1.685)]]
basis = mixed_basis([a[0] for a in atoms], heavy_basis="SARC-DKH2", light_basis="def2-SVP")
mol = gto.M(atom=atoms, basis=basis, charge=2, spin=0, verbose=3, max_memory=11000)
mf = scf.RHF(mol).x2c().density_fit()
mf.conv_tol = 1e-9
mf.kernel()
# freeze deep core (U has many core electrons); freeze ~ up to 5p of U + O 1s
nfroze = 43   # approx: U [Rn]-core-ish frozen; conservative
mycc = cc.CCSD(mf, frozen=nfroze)
mycc.max_cycle = 80
mycc.kernel()
nocc = mycc.t1.shape[0]
t1diag = float(np.linalg.norm(mycc.t1) / np.sqrt(2*nocc))
print(f"T1 diagnostic = {t1diag:.4f}  (nocc_corr={nocc}, frozen={nfroze})", flush=True)
json.dump(dict(t1_diagnostic=t1diag, n_corr_occ=nocc, frozen=nfroze,
               e_ccsd=float(mycc.e_tot)), open("results/dft/t1_diagnostic.json","w"), indent=2)
print("T1_DONE", flush=True)
