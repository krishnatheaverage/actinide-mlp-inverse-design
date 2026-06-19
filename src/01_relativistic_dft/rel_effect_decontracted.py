import json, numpy as np
from pyscf import gto, dft
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))

def uranyl(r): return [["U",(0,0,0)],["O",(0,0,r)],["O",(0,0,-r)]]

def basis_dict(decontract):
    bU = gto.basis.load("SARC-DKH2", "U")
    bO = gto.basis.load("def2-TZVP", "O")
    if decontract:
        bU = gto.uncontract(bU); bO = gto.uncontract(bO)
    return {"U": bU, "O": bO}

def scan(x2c, decontract):
    rs = np.arange(1.66, 1.92, 0.02); es = []
    bd = basis_dict(decontract)
    for r in rs:
        mol = gto.M(atom=uranyl(r), basis=bd, charge=2, spin=0, verbose=0, max_memory=11000)
        mf = dft.RKS(mol)
        if x2c: mf = mf.x2c()
        mf = mf.density_fit(); mf.xc = "PBE0"; mf.conv_tol = 1e-9
        es.append(mf.kernel())
    es = np.array(es); i = es.argmin(); sl = slice(max(0,i-2), i+3)
    c = np.polyfit(rs[sl], es[sl], 2); return float(-c[1]/(2*c[0]))

if __name__ == "__main__":
    out = {}
    for tag, dec in [("contracted", False), ("decontracted", True)]:
        rx = scan(True, dec); rn = scan(False, dec)
        out[tag] = dict(r_x2c=rx, r_nonrel=rn, contraction_pm=(rn-rx)*100)
        print(f"{tag}: r_X2C={rx:.4f} r_nonrel={rn:.4f} contraction={ (rn-rx)*100:.2f} pm", flush=True)
    json.dump(out, open("results/dft/rel_effect_decontracted.json", "w"), indent=2)
    print("DECON_DONE", flush=True)
