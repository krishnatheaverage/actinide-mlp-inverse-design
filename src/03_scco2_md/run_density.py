import sys, os, time, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import openmm as mm
from openmm import unit
from openmm.app import Simulation
from co2_system import build_co2_box, M_CO2_GMOL, AVOG
from CoolProp.CoolProp import PropsSI

def pick_platform():
    names = [mm.Platform.getPlatform(i).getName() for i in range(mm.Platform.getNumPlatforms())]
    for pref in ("CUDA", "OpenCL", "CPU", "Reference"):
        if pref in names:
            p = mm.Platform.getPlatformByName(pref)
            props = {}
            if pref == "CPU":
                props = {"Threads": "8"}
            return p, props, names
    raise RuntimeError("no platform")

def density_gcc(system, box_vec_nm, n_co2):

    vol_nm3 = box_vec_nm**3
    vol_cm3 = vol_nm3 * 1e-21
    mass_g = n_co2 * M_CO2_GMOL / AVOG
    return mass_g / vol_cm3

def run_state_point(T_K, P_bar, n_co2=500, eq_ps=120, prod_ps=300, seed=1, plat=None):
    rho_eos = PropsSI("D", "T", T_K, "P", P_bar*1e5, "CO2") / 1000.0

    system, pos, top, L = build_co2_box(n_co2, rho_gcc=0.25, seed=seed)
    system.addForce(mm.MonteCarloBarostat(P_bar*unit.bar, T_K*unit.kelvin, 25))
    integ = mm.LangevinMiddleIntegrator(T_K*unit.kelvin, 1.0/unit.picosecond, 0.001*unit.picoseconds)
    platform, props, _ = plat
    sim = Simulation(top, system, integ, platform, props)
    sim.context.setPositions(pos)
    sim.minimizeEnergy(maxIterations=1000)
    sim.context.setVelocitiesToTemperature(T_K*unit.kelvin, seed)
    sim.step(5000)
    integ.setStepSize(0.002*unit.picoseconds)
    sim.step(int(eq_ps/0.002))

    nsteps = int(prod_ps/0.002); stride = 500
    dens = []
    t0 = time.time()
    for _ in range(nsteps//stride):
        sim.step(stride)
        st = sim.context.getState()
        box = st.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
        edge = box[0][0]
        dens.append(density_gcc(system, edge, n_co2))
    dens = np.array(dens)
    wall = time.time()-t0
    out = dict(T_K=T_K, P_bar=P_bar, n_co2=n_co2,
               rho_md=float(dens.mean()), rho_md_std=float(dens.std()),
               rho_eos=float(rho_eos), pct_err=float(100*(dens.mean()-rho_eos)/rho_eos),
               n_samples=len(dens), prod_ps=prod_ps, wall_s=round(wall,1))
    print(f"T={T_K}K P={P_bar}bar  rho_MD={out['rho_md']:.4f}+/-{out['rho_md_std']:.4f}  "
          f"rho_EOS={out['rho_eos']:.4f} g/cc  err={out['pct_err']:+.1f}%  ({wall:.0f}s)", flush=True)
    return out

if __name__ == "__main__":
    plat = pick_platform()
    print("platforms:", plat[2], "-> using", plat[0].getName(), flush=True)

    state_points = [(320, 80), (320, 100), (320, 150), (320, 200), (320, 300), (310, 75)]
    results = []
    for T, P in state_points:
        try:
            results.append(run_state_point(T, P, plat=plat))
        except Exception as e:
            import traceback; traceback.print_exc()
    with open("results/scco2/density_validation.json", "w") as f:
        json.dump(results, f, indent=2)
    print("DENSITY_DONE", flush=True)
