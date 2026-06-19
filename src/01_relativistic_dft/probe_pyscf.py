import time, sys, traceback
import numpy as np

def section(t): print(f"\n{'='*72}\n{t}\n{'='*72}", flush=True)

section("PySCF / environment")
import pyscf
from pyscf import gto, dft, scf
print("pyscf", pyscf.__version__)
try:
    import basis_set_exchange as bse
    print("basis_set_exchange", bse.version())
    HAVE_BSE = True
except Exception as e:
    print("basis_set_exchange MISSING:", e)
    HAVE_BSE = False

CANDIDATES = ["x2c-SVPall", "x2c-TZVPall", "ANO-RCC-VDZP", "ANO-RCC-VTZP",
              "SARC-DKH2", "sarc2-dkh-qzvp", "jorge-DZP", "dyall-vdz"]

def try_pyscf_internal(name, elem="U"):
    try:
        b = gto.basis.load(name, elem)
        return ("internal", sum(len(s)-1 for s in b))
    except Exception as e:
        return ("fail-internal", str(e)[:60])

def try_bse(name, elem="U"):
    if not HAVE_BSE: return ("no-bse", "")
    try:
        s = bse.get_basis(name, elements=[elem], fmt="nwchem", header=False)
        b = gto.basis.parse(s)
        return ("bse-ok", len(b))
    except Exception as e:
        return ("fail-bse", str(e)[:80])

section("Basis availability for U (uranium, Z=92)")
usable = []
for name in CANDIDATES:
    pi = try_pyscf_internal(name)
    be = try_bse(name)
    print(f"{name:18s} internal={pi[0]:14s} bse={be[0]}")
    if pi[0] == "internal" or be[0] == "bse-ok":
        usable.append(name)
print("USABLE:", usable)

def get_basis_dict(name, elements):
    out = {}
    for el in elements:
        try:
            out[el] = gto.basis.load(name, el)
        except Exception:
            if not HAVE_BSE:
                raise
            s = bse.get_basis(name, elements=[el], fmt="nwchem", header=False)
            out[el] = gto.basis.parse(s)
    return out

def build_uranyl(basis_name, n_water=0):

    atoms = [["U", (0.0, 0.0, 0.0)],
             ["O", (0.0, 0.0,  1.78)],
             ["O", (0.0, 0.0, -1.78)]]
    import math
    for k in range(n_water):
        ang = 2*math.pi*k/max(n_water,1)
        r = 2.45
        ox = (r*math.cos(ang), r*math.sin(ang), 0.0)
        atoms.append(["O", ox])

        atoms.append(["H", (ox[0]*1.10+0.3, ox[1]*1.10, 0.59)])
        atoms.append(["H", (ox[0]*1.10+0.3, ox[1]*1.10, -0.59)])
    bdict = get_basis_dict(basis_name, ["U", "O", "H"])
    mol = gto.M(atom=atoms, basis=bdict, charge=2, spin=0,
                verbose=0, max_memory=12000)
    return mol

section("Timing test: scalar-X2C PBE0 on UO2^2+ (+waters)")
if not usable:
    print("No usable actinide basis found -- STOP, need to fix basis access.")
    sys.exit(1)

for basis_name in usable[:2]:
    for nw in (0, 2):
        try:
            t0 = time.time()
            mol = build_uranyl(basis_name, n_water=nw)
            mf = dft.RKS(mol).x2c()
            mf.xc = "PBE0"
            mf.max_cycle = 80
            e = mf.kernel()
            t_scf = time.time() - t0
            conv = mf.converged
            nao = mol.nao_nr()

            tg, gmax, gmode = None, None, "none"
            try:
                t1 = time.time()
                g = mf.nuc_grad_method().kernel()
                tg = time.time() - t1
                gmax = float(np.abs(g).max()); gmode = "analytic"
            except Exception as ge:
                gmode = f"analytic-fail:{type(ge).__name__}"
            print(f"[{basis_name} +{nw}H2O] nao={nao} conv={conv} "
                  f"E={e:.6f} Ha  t_scf={t_scf:.1f}s  grad={gmode} t_grad={tg}")
        except Exception:
            print(f"[{basis_name} +{nw}H2O] FAILED:")
            traceback.print_exc()

print("\nPROBE_DONE", flush=True)
