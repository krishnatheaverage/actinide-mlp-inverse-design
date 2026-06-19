import numpy as np
import openmm as mm
from openmm import unit
from openmm.app import Topology, Element

KB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA

SIG_C, EPS_C, Q_C = 0.280, 27.0 * 0.00831446262, +0.70
SIG_O, EPS_O, Q_O = 0.305, 79.0 * 0.00831446262, -0.35
R_CO = 0.116
R_OO = 2.0 * R_CO

K_OO = 80000.0
M_C, M_O = 12.011, 15.999
M_CO2_GMOL = M_C + 2 * M_O

AVOG = 6.02214076e23

def box_edge_for_density(n_co2, rho_gcc):
    mass_g = n_co2 * M_CO2_GMOL / AVOG
    vol_cm3 = mass_g / rho_gcc
    vol_nm3 = vol_cm3 * 1e21
    return vol_nm3 ** (1.0 / 3.0)

def build_co2_box(n_co2, rho_gcc=0.70, seed=0, cutoff_nm=1.3):
    rng = np.random.default_rng(seed)
    L = box_edge_for_density(n_co2, rho_gcc)

    n_side = int(np.ceil(n_co2 ** (1.0 / 3.0)))
    spacing = L / n_side
    centers = []
    for i in range(n_side):
        for j in range(n_side):
            for k in range(n_side):
                if len(centers) < n_co2:
                    centers.append(np.array([i, j, k]) * spacing + spacing * 0.5)
    centers = np.array(centers)

    system = mm.System()
    box = L
    system.setDefaultPeriodicBoxVectors(mm.Vec3(box, 0, 0),
                                        mm.Vec3(0, box, 0),
                                        mm.Vec3(0, 0, box))
    nb = mm.NonbondedForce()
    nb.setNonbondedMethod(mm.NonbondedForce.PME)
    nb.setCutoffDistance(cutoff_nm * unit.nanometer)
    nb.setUseSwitchingFunction(True)
    nb.setSwitchingDistance((cutoff_nm - 0.1) * unit.nanometer)
    nb.setUseDispersionCorrection(True)
    nb.setEwaldErrorTolerance(1e-4)

    positions = []
    top = Topology()
    chain = top.addChain()
    elemC, elemO = Element.getBySymbol("C"), Element.getBySymbol("O")

    oo_bond = mm.HarmonicBondForce()
    for c in centers:

        v = rng.normal(size=3); v /= np.linalg.norm(v)
        pos_C = c
        pos_O1 = c + v * R_CO
        pos_O2 = c - v * R_CO
        iC = system.addParticle(M_C)
        iO1 = system.addParticle(M_O)
        iO2 = system.addParticle(M_O)
        nb.addParticle(Q_C, SIG_C, EPS_C)
        nb.addParticle(Q_O, SIG_O, EPS_O)
        nb.addParticle(Q_O, SIG_O, EPS_O)

        system.addConstraint(iC, iO1, R_CO)
        system.addConstraint(iC, iO2, R_CO)
        oo_bond.addBond(iO1, iO2, R_OO, K_OO)

        nb.addException(iC, iO1, 0.0, 1.0, 0.0)
        nb.addException(iC, iO2, 0.0, 1.0, 0.0)
        nb.addException(iO1, iO2, 0.0, 1.0, 0.0)
        positions += [pos_C, pos_O1, pos_O2]
        res = top.addResidue("CO2", chain)
        top.addAtom("C", elemC, res)
        top.addAtom("O1", elemO, res)
        top.addAtom("O2", elemO, res)

    system.addForce(nb)
    system.addForce(oo_bond)
    top.setPeriodicBoxVectors([[box, 0, 0], [0, box, 0], [0, 0, box]] * unit.nanometer)
    return system, np.array(positions) * unit.nanometer, top, L

if __name__ == "__main__":
    for n in (256, 500, 1000):
        L = box_edge_for_density(n, 0.70)
        print(f"n_co2={n:5d}  rho=0.70 g/cc -> box edge {L:.3f} nm")
    s, p, t, L = build_co2_box(256, 0.70)
    print("built system: particles", s.getNumParticles(),
          "constraints", s.getNumConstraints(), "box", round(L, 3), "nm")
