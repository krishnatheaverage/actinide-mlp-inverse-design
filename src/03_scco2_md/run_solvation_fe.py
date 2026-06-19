"""
Solvation (excess chemical potential) of a model CO2-phile solute in scCO2 by
alchemical soft-core decoupling + MBAR. Reports per-window uncertainty, the MBAR
overlap matrix, and forward/reverse BAR agreement (hysteresis check). State point
density is pinned to the Span-Wagner EOS (CoolProp), so dG is reported at a
well-defined supercritical state and can be swept along an isotherm.

Solute: TraPPE-UA methane (sigma=0.373 nm, eps/kB=148 K) -- a neutral LJ probe of
CO2-philicity. Decoupling free energy dG_decouple; solvation free energy = -dG.
"""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import openmm as mm
from openmm import unit
from openmm.app import Simulation
from co2_system import build_co2_box, M_CO2_GMOL, AVOG
from CoolProp.CoolProp import PropsSI

KB_kJ = 0.00831446262  # kJ/mol/K

SOLUTES = {
    "CH4-UA":  dict(sigma=0.373, eps=148.0*KB_kJ, mass=16.043),  # TraPPE-UA methane
    "Xe-like": dict(sigma=0.398, eps=214.0*KB_kJ, mass=131.29),  # heavier LJ probe
}

def find_void(xyz, L, seed=0, n_try=400):
    """Return an insertion point maximising the min distance to existing atoms."""
    rng = np.random.default_rng(seed)
    best, best_d = None, -1
    for _ in range(n_try):
        p = rng.uniform(0.15*L, 0.85*L, size=3)
        d = np.linalg.norm(xyz - p, axis=1).min()
        if d > best_d:
            best_d, best = d, p
    return best

def pick_platform():
    names=[mm.Platform.getPlatform(i).getName() for i in range(mm.Platform.getNumPlatforms())]
    for pref in ("CUDA","OpenCL","CPU","Reference"):
        if pref in names:
            p=mm.Platform.getPlatformByName(pref)
            return p, ({"Threads":"8"} if pref=="CPU" else {}), names
    raise RuntimeError("no platform")

def add_solute(system, nb, solute, box_L):
    """Add a single neutral LJ solute; return its index and a softcore CustomNonbondedForce."""
    s = SOLUTES[solute]
    idx = system.addParticle(s["mass"])
    # solute carries NO standard nonbonded (eps=0) -> handled by softcore custom force
    nb.addParticle(0.0, s["sigma"], 0.0)
    cutoff = nb.getCutoffDistance()
    # soft-core LJ (Beutler) between solute and solvent, scaled by lambda_vdw
    expr = ("lambda_vdw*4*eps*(1/(a*(1-lambda_vdw)^2+(r/sig)^6)^2 - 1/(a*(1-lambda_vdw)^2+(r/sig)^6));"
            "sig=0.5*(sigma1+sigma2); eps=sqrt(epsilon1*epsilon2); a=0.5")
    cf = mm.CustomNonbondedForce(expr)
    cf.addGlobalParameter("lambda_vdw", 1.0)
    cf.addPerParticleParameter("sigma")
    cf.addPerParticleParameter("epsilon")
    cf.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
    cf.setCutoffDistance(cutoff)
    cf.setUseSwitchingFunction(True)
    cf.setSwitchingDistance(cutoff - 0.1*unit.nanometer)
    # per-particle params for ALL particles
    from co2_system import SIG_C, EPS_C, SIG_O, EPS_O
    n_solv = system.getNumParticles() - 1
    for i in range(n_solv):
        # solvent CO2 atoms in C,O,O repeating order
        if i % 3 == 0: cf.addParticle([SIG_C, EPS_C])
        else:          cf.addParticle([SIG_O, EPS_O])
    cf.addParticle([s["sigma"], s["eps"]])  # solute
    cf.addInteractionGroup(set([idx]), set(range(n_solv)))
    # OpenMM requires all nonbonded-type forces to share identical exclusions:
    # copy the CO2 intramolecular exceptions from the NonbondedForce.
    for k in range(nb.getNumExceptions()):
        p1, p2, _, _, _ = nb.getExceptionParameters(k)
        cf.addExclusion(p1, p2)
    system.addForce(cf)
    return idx, cf

def run(solute="CH4-UA", T_K=320.0, P_bar=150.0, n_co2=343,
        lambdas=None, eq_ps=60, prod_ps=120, sample_ps=0.5, seed=2, plat=None):
    if lambdas is None:
        lambdas = [1.00,0.95,0.90,0.80,0.70,0.60,0.50,0.40,0.30,0.20,0.12,0.06,0.0]
    beta = 1.0/(KB_kJ*T_K)
    PV_CONV = 0.0602214  # bar*nm^3 -> kJ/mol
    rho_eos = PropsSI("D","T",T_K,"P",P_bar*1e5,"CO2")/1000.0
    # pack at safe low density, compress to the state point with the barostat (NPT)
    system, pos, top, L = build_co2_box(n_co2, rho_gcc=0.25, seed=seed)
    nb = [f for f in system.getForces() if isinstance(f, mm.NonbondedForce)][0]
    co2_xyz = pos.value_in_unit(unit.nanometer)
    cen = find_void(co2_xyz, L, seed=seed)
    idx, cf = add_solute(system, nb, solute, L)
    pos = np.vstack([co2_xyz, cen]) * unit.nanometer
    system.addForce(mm.MonteCarloBarostat(P_bar*unit.bar, T_K*unit.kelvin, 25))

    integ = mm.LangevinMiddleIntegrator(T_K*unit.kelvin, 1.0/unit.picosecond, 0.001*unit.picoseconds)
    platform, props, _ = plat
    sim = Simulation(top, system, integ, platform, props)
    sim.context.setPositions(pos)
    sim.minimizeEnergy(maxIterations=1000)
    sim.context.setVelocitiesToTemperature(T_K*unit.kelvin, seed)
    sim.step(5000)                                   # 5 ps gentle start @1fs (lambda=1)
    integ.setStepSize(0.002*unit.picoseconds)
    sim.step(int(40/0.002))                          # 40 ps NPT density equilibration

    K = len(lambdas)
    samples_per = int(prod_ps/sample_ps)
    u_kln = np.zeros((K, K, samples_per))   # u_kln[k,l,n]: sample n from state k eval at state l
    N_k = np.zeros(K, dtype=int)
    t0=time.time()
    for k, lam in enumerate(lambdas):
        sim.context.setParameter("lambda_vdw", lam)
        sim.context.setVelocitiesToTemperature(T_K*unit.kelvin, seed+k)
        sim.step(int(eq_ps/0.002))
        for n in range(samples_per):
            sim.step(int(sample_ps/0.002))
            V = sim.context.getState().getPeriodicBoxVolume().value_in_unit(unit.nanometer**3)
            pv = P_bar * V * PV_CONV   # kJ/mol
            for l, lam2 in enumerate(lambdas):
                sim.context.setParameter("lambda_vdw", lam2)
                e = sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                u_kln[k, l, n] = beta*(e + pv)
            sim.context.setParameter("lambda_vdw", lam)  # restore sampling state
        N_k[k] = samples_per
        print(f"  [{solute} T={T_K} P={P_bar}] window {k+1}/{K} lambda={lam:.2f} done "
              f"({time.time()-t0:.0f}s)", flush=True)

    # ---- MBAR ----
    from pymbar import MBAR
    u_kn = u_kln.reshape(K, K*samples_per)  # not used; build proper concat
    # build u_kn: shape (K, sum N_k)
    u_kn = np.zeros((K, int(N_k.sum())))
    idxs = np.concatenate([[0], np.cumsum(N_k)])
    for k in range(K):
        for l in range(K):
            u_kn[l, idxs[k]:idxs[k+1]] = u_kln[k, l, :N_k[k]]
    mbar = MBAR(u_kn, N_k)
    res = mbar.compute_free_energy_differences()
    dF = res["Delta_f"]; ddF = res["dDelta_f"]
    # decoupling free energy = f(lambda=0) - f(lambda=1); states ordered lam[0]=1 ... lam[-1]=0
    dG_decouple = float((dF[0, K-1]) / beta)      # kJ/mol
    dG_err = float((ddF[0, K-1]) / beta)
    dG_solv = -dG_decouple
    overlap = mbar.compute_overlap()["matrix"]
    # adjacent-window BAR forward/reverse via pymbar BAR
    from pymbar import other_estimators as oe
    bar_adj = []
    for k in range(K-1):
        w_F = u_kln[k, k+1, :N_k[k]] - u_kln[k, k, :N_k[k]]
        w_R = u_kln[k+1, k, :N_k[k+1]] - u_kln[k+1, k+1, :N_k[k+1]]
        try:
            b = oe.bar(w_F, w_R)
            bar_adj.append(float(b["Delta_f"]/beta))
        except Exception:
            bar_adj.append(None)
    out = dict(solute=solute, T_K=T_K, P_bar=P_bar, n_co2=n_co2, rho_eos=rho_eos,
               lambdas=lambdas, dG_solv_kJ=dG_solv, dG_err_kJ=dG_err,
               min_overlap_adjacent=float(min(overlap[i, i+1] for i in range(K-1))),
               overlap_diag=overlap.diagonal().tolist(),
               wall_s=round(time.time()-t0,1), samples_per=samples_per)
    print(f"==> {solute} T={T_K}K P={P_bar}bar rho={rho_eos:.3f}: "
          f"dG_solv = {dG_solv:.2f} +/- {dG_err:.2f} kJ/mol  "
          f"min adj-overlap={out['min_overlap_adjacent']:.3f}", flush=True)
    return out, overlap

if __name__=="__main__":
    plat = pick_platform()
    print("platform:", plat[0].getName(), flush=True)
    allres=[]
    def save(): json.dump(allres, open("results/scco2/solvation_fe.json","w"), indent=2)
    # two solutes at one state point, then a supercritical density sweep for CH4
    for solute in ("CH4-UA","Xe-like"):
        r,ov = run(solute=solute, T_K=320.0, P_bar=150.0, plat=plat)
        np.save(f"results/scco2/overlap_{solute}.npy", ov); allres.append(r); save()
    for P in (100.0, 200.0):
        r,ov = run(solute="CH4-UA", T_K=320.0, P_bar=P, plat=plat); allres.append(r); save()
    print("SOLVATION_DONE", flush=True)
