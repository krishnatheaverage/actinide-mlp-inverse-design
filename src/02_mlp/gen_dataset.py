"""
Stage 2 data generation (PARALLEL): build a labelled configuration set for an MLP
of the closed-shell uranyl aqua complex UO2(H2O)4^2+, with X2C-PBE0(DF) energies+
forces. DFT labelling and the numerical Hessian are parallelised over a process
pool for ~Nworker speed-up on the M-series CPU.

Sampling sources (kept SEPARATE -> leakage-controlled split + OOD test):
  NMS-T : thermal normal-mode sampling at T = 300, 600, 1000 K
  SCAN  : U=O and U-O(water) bond-stretch scans (held out as OOD)
The numerical Hessian also yields real uranyl stretch frequencies (vs experiment).

Output: results/mlp/uo2_dataset.extxyz (energy eV, forces eV/A, config_type tag).
"""
import os, sys, json, time
os.environ.setdefault("OMP_NUM_THREADS", "1")  # single-thread per worker
import numpy as np
from multiprocessing import Pool
from pyscf import gto, dft
from pyscf.geomopt.geometric_solver import optimize
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from relbasis import mixed_basis
from ase import Atoms
from ase.io import write

HA2EV = 27.211386245988
BOHR2A = 0.529177210903
F_CONV = HA2EV / BOHR2A
LIGHT_BASIS = "def2-SVP"
NWORK = int(os.environ.get("NWORK", "5"))

def make_mf(atoms, charge=2, spin=0):
    basis = mixed_basis([a[0] for a in atoms], heavy_basis="SARC-DKH2", light_basis=LIGHT_BASIS)
    mol = gto.M(atom=atoms, basis=basis, charge=charge, spin=spin, verbose=0, max_memory=4000)
    mf = dft.RKS(mol).x2c().density_fit()
    mf.xc = "PBE0"; mf.conv_tol = 1e-9; mf.max_cycle = 120
    return mol, mf

def energy_grad(atoms, charge=2, spin=0):
    _, mf = make_mf(atoms, charge, spin)
    e = mf.kernel()
    if not mf.converged:
        raise RuntimeError("SCF not converged")
    g = mf.nuc_grad_method().kernel()
    return e, g

# ---- top-level worker (picklable) ----
def _eg(payload):
    syms, coords, ctype = payload
    atoms = [[syms[i], tuple(coords[i])] for i in range(len(syms))]
    try:
        e, g = energy_grad(atoms)
        return ctype, e, np.asarray(g)
    except Exception as ex:
        return ctype, None, str(ex)[:60]

def init_uo2_4w():
    a = [["U",(0,0,0)],["O",(0,0,1.76)],["O",(0,0,-1.76)]]
    import math
    for k in range(4):
        ang = math.pi/2*k; r = 2.45; ox,oy = r*math.cos(ang), r*math.sin(ang)
        a.append(["O",(ox,oy,0)]); hx,hy = ox*1.18, oy*1.18
        a.append(["H",(hx,hy, 0.59)]); a.append(["H",(hx,hy,-0.59)])
    return a

def parallel_hessian(syms, coords, pool, disp=0.01):
    """Central-difference Hessian from analytic gradients, parallel. disp in Angstrom."""
    n = len(syms); d_bohr = disp / BOHR2A
    jobs = []
    for i in range(n):
        for c in range(3):
            for sgn in (+1, -1):
                cc = coords.copy(); cc[i, c] += sgn*disp
                jobs.append((syms, cc, f"hess_{i}_{c}_{sgn}"))
    results = pool.map(_eg, jobs)
    grads = {}
    for (ctype, e, g) in results:
        if e is None: raise RuntimeError(f"hessian grad failed: {g}")
        grads[ctype] = g.reshape(-1)
    H = np.zeros((3*n, 3*n))
    for i in range(n):
        for c in range(3):
            gp = grads[f"hess_{i}_{c}_1"]; gm = grads[f"hess_{i}_{c}_-1"]
            H[3*i+c, :] = (gp - gm) / (2*d_bohr)
    return 0.5*(H + H.T)

def normal_modes(H, syms):
    masses = np.array([Atoms(s).get_masses()[0] for s in syms])
    m3 = np.repeat(masses, 3)
    Hmw = H / np.sqrt(np.outer(m3, m3))
    w2, V = np.linalg.eigh(Hmw)
    conv = 5140.4871
    freqs = np.sign(w2)*np.sqrt(np.abs(w2))*conv
    return freqs, V, masses

def sample_nms(coords, syms, freqs, V, masses, T, n_samples, rng):
    kB = 3.166811563e-6; conv = 5140.4871
    m3 = np.repeat(masses, 3)
    real_idx = [i for i in range(len(freqs)) if freqs[i] > 80.0]
    confs = []
    for _ in range(n_samples):
        dq = np.zeros(3*len(syms))
        for i in real_idx:
            w2 = (freqs[i]/conv)**2
            dq += rng.normal(0, np.sqrt(kB*T/w2)) * V[:, i]
        dx_ang = (dq/np.sqrt(m3)).reshape(-1, 3) * BOHR2A
        confs.append(coords + dx_ang)
    return confs

def scan_bond(coords, syms, i, j, deltas):
    v = coords[j]-coords[i]; v = v/np.linalg.norm(v); r0 = np.linalg.norm(coords[j]-coords[i])
    out = []
    for d in deltas:
        c = coords.copy(); c[j] = c[i] + (r0+d)*v; out.append(c)
    return out

if __name__ == "__main__":
    rng = np.random.default_rng(7); t0 = time.time()
    print(f"[1] optimise UO2(H2O)4^2+ (NWORK={NWORK})", flush=True)
    _, mf = make_mf(init_uo2_4w())
    mol_eq = optimize(mf, maxsteps=100)
    syms = [mol_eq.atom_symbol(i) for i in range(mol_eq.natm)]
    coords = np.array([mol_eq.atom_coord(i, unit="Angstrom") for i in range(mol_eq.natm)])
    print(f"    optimised ({time.time()-t0:.0f}s)", flush=True)

    pool = Pool(NWORK)
    print("[2] parallel numerical Hessian + normal modes", flush=True)
    H = parallel_hessian(syms, coords, pool)
    freqs, V, masses = normal_modes(H, syms)
    top_freqs = sorted([f for f in freqs if f > 80], reverse=True)[:6]
    print(f"    top freqs (cm^-1): {[round(f,1) for f in top_freqs]} ({time.time()-t0:.0f}s)", flush=True)

    print("[3] build configurations", flush=True)
    frames = []
    for T, n in [(300, 90), (600, 95), (1000, 70)]:
        for c in sample_nms(coords, syms, freqs, V, masses, T, n, rng):
            frames.append((syms, c, f"NMS-{T}"))
    for c in scan_bond(coords, syms, 0, 1, np.linspace(-0.25, 0.6, 24)):
        frames.append((syms, c, "SCAN-UO"))
    for c in scan_bond(coords, syms, 0, 3, np.linspace(-0.4, 1.0, 24)):
        frames.append((syms, c, "SCAN-UOw"))
    print(f"    {len(frames)} configs to label", flush=True)

    print("[4] parallel X2C-PBE0(DF) labelling", flush=True)
    ase_frames = []; ok = 0
    for n, (ctype, e, g) in enumerate(pool.imap(_eg, frames, chunksize=1)):
        if e is None:
            print(f"    frame {n} ({ctype}) FAIL: {g}"); continue
        c = frames[n][1]
        a = Atoms(symbols=syms, positions=c)
        a.info["energy"] = e*HA2EV; a.info["config_type"] = ctype
        a.arrays["forces"] = -g*F_CONV
        ase_frames.append(a); ok += 1
        if n % 25 == 0:
            write("results/mlp/uo2_dataset.extxyz", ase_frames)
            print(f"    labelled {n+1}/{len(frames)} ({time.time()-t0:.0f}s)", flush=True)
    pool.close(); pool.join()
    write("results/mlp/uo2_dataset.extxyz", ase_frames)
    meta = dict(system="UO2(H2O)4^2+", n_atoms=len(syms), n_frames=ok,
                light_basis=LIGHT_BASIS, top_freqs_cm=top_freqs,
                wall_min=round((time.time()-t0)/60, 1))
    json.dump(meta, open("results/mlp/dataset_meta.json", "w"), indent=2)
    print(f"DATASET_DONE: {ok} frames in {meta['wall_min']} min", flush=True)
