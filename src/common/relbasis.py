"""Mixed relativistic basis helper: an all-electron relativistic basis on the
heavy (actinide/lanthanide) centre, a segmented def2 basis on light ligand atoms,
all used under PySCF's scalar-X2C Hamiltonian. PySCF resolves each element name
(routing to basis_set_exchange for SARC/ANO-RCC as needed).
"""
LIGHT = {"H", "C", "N", "O", "F", "S", "P", "Cl"}

def mixed_basis(elements, heavy_basis="SARC-DKH2", light_basis="def2-TZVP",
                heavy_elems=("La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
                             "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm")):
    """Return {element: basis_name} dict for gto.M(basis=...)."""
    b = {}
    for el in set(elements):
        b[el] = heavy_basis if el in heavy_elems else light_basis
    return b
