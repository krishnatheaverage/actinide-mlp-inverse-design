import sys, os, json, time
import numpy as np
from pyscf import gto, dft
from pyscf.geomopt.geometric_solver import optimize
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from relbasis import mixed_basis

HARTREE2KJ = 2625.499639

def make_mf(atoms, charge, spin, heavy="SARC-DKH2", light="def2-TZVP", xc="PBE0"):
    elems = [a[0] for a in atoms]
    basis = mixed_basis(elems, heavy_basis=heavy, light_basis=light)
    mol = gto.M(atom=atoms, basis=basis, charge=charge, spin=spin,
                verbose=0, max_memory=12000)
    mf = dft.UKS(mol) if spin else dft.RKS(mol)
    mf = mf.x2c().density_fit()
    mf.xc = xc; mf.conv_tol = 1e-9; mf.max_cycle = 150
    return mol, mf

def sp_energy(atoms, charge, spin, **kw):
    _, mf = make_mf(atoms, charge, spin, **kw)
    return mf.kernel()

def opt_geometry(atoms, charge, spin, maxsteps=80, **kw):
    _, mf = make_mf(atoms, charge, spin, **kw)
    mol_eq = optimize(mf, maxsteps=maxsteps)
    new_atoms = [[mol_eq.atom_symbol(i), tuple(mol_eq.atom_coord(i, unit="Angstrom"))]
                 for i in range(mol_eq.natm)]
    e = sp_energy(new_atoms, charge, spin, **kw)
    return new_atoms, e

def uranyl(r_uo=1.78):
    return [["U",(0,0,0)],["O",(0,0,r_uo)],["O",(0,0,-r_uo)]]

def relativistic_effect():
    print("\n[A] Relativistic effect on uranyl U=O bond", flush=True)
    out = {}

    rs = np.arange(1.66, 1.92, 0.02)
    for tag, x2c in (("x2c", True), ("nonrel", False)):
        es = []
        for r in rs:
            elems=["U","O","O"]; basis=mixed_basis(elems)
            mol=gto.M(atom=uranyl(r),basis=basis,charge=2,spin=0,verbose=0,max_memory=12000)
            mf=dft.RKS(mol); mf=mf.x2c() if x2c else mf; mf=mf.density_fit()
            mf.xc="PBE0"; mf.conv_tol=1e-9
            es.append(mf.kernel())
        es=np.array(es)

        i=es.argmin(); sl=slice(max(0,i-2),i+3)
        c=np.polyfit(rs[sl],es[sl],2); rmin=-c[1]/(2*c[0])
        out[tag]=dict(r_min=float(rmin), e_min=float(es.min()),
                      rs=rs.tolist(), es=es.tolist())
        print(f"  {tag}: r(U=O)_min = {rmin:.4f} A", flush=True)
    out["contraction_A"]=out["nonrel"]["r_min"]-out["x2c"]["r_min"]
    print(f"  relativistic bond contraction = {out['contraction_A']*100:.2f} pm", flush=True)
    return out

def water(): return [["O",(0,0,0)],["H",(0.757,0.586,0)],["H",(-0.757,0.586,0)]]
def acetate():
    return [["C",(0,0,0)],["O",(1.25,0,0)],["O",(-0.6,1.08,0)],
            ["C",(-0.7,-1.28,0)],["H",(-1.79,-1.22,0)],["H",(-0.36,-1.82,0.89)],["H",(-0.36,-1.82,-0.89)]]

def binding_energy(metal, mq, lig_atoms, lig_charge, lig_spin, n_lig=1,
                   m_template=None, label=""):
    t0=time.time()

    e_M = sp_energy([[metal,(0,0,0)]], mq, 0)
    e_L = sp_energy(lig_atoms, lig_charge, lig_spin)

    complex_atoms = [[metal,(0.0,0.0,0.0)]]

    import math
    base = np.array([a[1] for a in lig_atoms]); syms=[a[0] for a in lig_atoms]

    shift = np.array([2.4,0,0]) - base[0]
    for k in range(n_lig):
        ang = 2*math.pi*k/max(n_lig,1)
        R = np.array([[math.cos(ang),-math.sin(ang),0],[math.sin(ang),math.cos(ang),0],[0,0,1]])
        for s,c in zip(syms, base+shift):
            complex_atoms.append([s, tuple(R@np.array(c))])
    tot_charge = mq + n_lig*lig_charge
    eq_atoms, e_complex = opt_geometry(complex_atoms, tot_charge, 0)
    dE = (e_complex - e_M - n_lig*e_L)*HARTREE2KJ
    out=dict(label=label, metal=metal, mq=mq, n_lig=n_lig,
             e_complex=e_complex, e_M=e_M, e_L=e_L, dE_kJ=float(dE),
             geom=eq_atoms, wall_s=round(time.time()-t0,1))
    print(f"  [{label}] {metal}{mq}+ + {n_lig}L: complexation dE = {dE:.1f} kJ/mol "
          f"({out['wall_s']}s)", flush=True)
    return out

if __name__=="__main__":
    results={}
    results["relativistic_effect"]=relativistic_effect()
    print("\n[B/C] Complexation + An/Ln selectivity (acetate donor)", flush=True)
    sel=[]

    for metal, mq in (("La",3),("Ac",3),("Th",4)):
        try:
            sel.append(binding_energy(metal, mq, acetate(), -1, 0, n_lig=1,
                                      label=f"{metal}{mq}+_acetate"))
        except Exception as e:
            import traceback; traceback.print_exc()
    results["selectivity"]=sel
    with open("results/dft/dft_results.json","w") as f:
        json.dump(results, f, indent=2)
    print("DFT_PRODUCTION_DONE", flush=True)
