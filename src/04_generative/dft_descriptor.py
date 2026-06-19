import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from pyscf import gto, dft

DONORS = {"N", "O", "F", "S", "P"}
ANG2BOHR = 1.0/0.529177210903

def embed_3d(smiles, seed=1):
    m = Chem.MolFromSmiles(smiles)
    if m is None: return None
    m = Chem.AddHs(m)
    if AllChem.EmbedMolecule(m, randomSeed=seed) != 0:
        if AllChem.EmbedMolecule(m, randomSeed=seed, useRandomCoords=True) != 0:
            return None
    try: AllChem.MMFFOptimizeMolecule(m, maxIters=400)
    except Exception: pass
    return m

def probe_points(coords, syms, n_sphere=12, radii=(1.6, 2.0)):
    idx = [i for i,s in enumerate(syms) if s in DONORS]
    if not idx: return None
    pts = []
    gold = np.pi*(3-np.sqrt(5))
    for i in idx:
        for R in radii:
            for k in range(n_sphere):
                y = 1 - 2*(k+0.5)/n_sphere; r = np.sqrt(max(0,1-y*y))
                th = gold*k
                d = np.array([np.cos(th)*r, y, np.sin(th)*r])
                pts.append(coords[i] + R*d)
    return np.array(pts)

def esp_at_points(mol, dm, pts_ang):
    pts = pts_ang*ANG2BOHR

    Vele = -np.einsum('pij,ij->p', mol.intor('int1e_grids', grids=pts), dm)

    Vnuc = np.zeros(len(pts))
    coords = mol.atom_coords()
    charges = mol.atom_charges()
    for A in range(mol.natm):
        d = np.linalg.norm(pts - coords[A], axis=1)
        Vnuc += charges[A]/np.maximum(d, 1e-9)
    return Vnuc + Vele

def vmin_descriptor(smiles, xc="PBE", basis="def2-SVP", seed=1):
    m = embed_3d(smiles, seed)
    if m is None: return None
    conf = m.GetConformer()
    syms = [a.GetSymbol() for a in m.GetAtoms()]
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(m.GetNumAtoms())])
    pts = probe_points(coords, syms)
    if pts is None: return None
    atom = [[syms[i], tuple(coords[i])] for i in range(len(syms))]
    try:
        mol = gto.M(atom=atom, basis=basis, charge=0, spin=0, verbose=0, max_memory=8000)
        mf = dft.RKS(mol).density_fit(); mf.xc = xc; mf.conv_tol = 1e-8
        mf.kernel()
        dm = mf.make_rdm1()

        keep = np.ones(len(pts), bool)
        cb = mol.atom_coords()/ANG2BOHR
        for A in range(mol.natm):
            keep &= np.linalg.norm(pts - cb[A], axis=1) > 1.2
        V = esp_at_points(mol, dm, pts[keep])
        return dict(smiles=smiles, vmin=float(V.min()),
                    homo=float(mf.mo_energy[mf.mo_occ>0].max()),
                    dipole=float(np.linalg.norm(mf.dip_moment(verbose=0))),
                    n_atoms=mol.natm)
    except Exception as e:
        return dict(smiles=smiles, error=str(e)[:80])

if __name__ == "__main__":
    import time
    for smi in ["CC(N)=O", "O=CN", "c1ccncc1", "CCO", "O=P(OC)(OC)OC", "FC(F)F"]:
        t0=time.time(); r = vmin_descriptor(smi); dt=time.time()-t0
        print(f"{smi:18s} {dt:5.1f}s  {r}")
