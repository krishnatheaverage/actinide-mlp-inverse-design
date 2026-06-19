import time, traceback, os, sys, numpy as np
from pyscf import gto, dft
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from relbasis import mixed_basis

def uranyl(n_water=0, r_uo=1.78):
    import math
    atoms = [["U", (0,0,0)], ["O", (0,0, r_uo)], ["O", (0,0,-r_uo)]]
    for k in range(n_water):
        a = 2*math.pi*k/max(n_water,1); r = 2.45
        ox,oy = r*math.cos(a), r*math.sin(a)
        atoms += [["O",(ox,oy,0)], ["H",(ox*1.15,oy*1.15, 0.59)], ["H",(ox*1.15,oy*1.15,-0.59)]]
    return atoms

def run(label, atoms, heavy="SARC-DKH2", light="def2-TZVP", xc="PBE0", df=False, x2c=True, grad=True):
    t0=time.time()
    elems = [a[0] for a in atoms]
    basis = mixed_basis(elems, heavy_basis=heavy, light_basis=light)
    mol = gto.M(atom=atoms, basis=basis, charge=2, spin=0, verbose=0, max_memory=12000)
    mf = dft.RKS(mol)
    if x2c: mf = mf.x2c()
    mf.xc = xc; mf.max_cycle=100; mf.conv_tol=1e-8
    if df:
        mf = mf.density_fit()
    e = mf.kernel(); t_scf=time.time()-t0; conv=mf.converged; nao=mol.nao_nr()
    tg=None; gmax=None; gmode="skip"
    if grad:
        try:
            t1=time.time(); g=mf.nuc_grad_method().kernel(); tg=time.time()-t1
            gmax=float(np.abs(g).max()); gmode="analytic"
        except Exception as ge:
            gmode=f"FAIL:{type(ge).__name__}:{str(ge)[:50]}"
    print(f"[{label}] nao={nao} conv={conv} E={e:.6f} x2c={x2c} df={df} "
          f"t_scf={t_scf:.1f}s grad={gmode} t_grad={None if tg is None else round(tg,1)} gmax={gmax}",
          flush=True)
    return dict(label=label,nao=nao,conv=conv,E=e,t_scf=t_scf,t_grad=tg,gmode=gmode)

if __name__=="__main__":
    for heavy in ("SARC-DKH2", "ANO-RCC-VDZP"):
        print(f"\n##### heavy basis = {heavy} (light=def2-TZVP) #####", flush=True)
        try:
            print("--- bare uranyl ---")
            run("UO2^2+ x2c", uranyl(0), heavy=heavy)
            print("--- bare uranyl + DF ---")
            try: run("UO2^2+ x2c+DF", uranyl(0), heavy=heavy, df=True)
            except Exception: traceback.print_exc()
            print("--- uranyl + 2 H2O (MLP-size system) ---")
            run("UO2(H2O)2 x2c", uranyl(2), heavy=heavy)
        except Exception:
            traceback.print_exc()
    print("=== relativistic effect: X2C vs nonrel, bare uranyl (SARC) ===")
    run("UO2 X2C  ", uranyl(0), heavy="SARC-DKH2", grad=False)
    run("UO2 NONREL", uranyl(0), heavy="SARC-DKH2", x2c=False, grad=False)
    print("BENCH_DONE", flush=True)
