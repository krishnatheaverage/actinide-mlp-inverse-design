LIGHT = {"H", "C", "N", "O", "F", "S", "P", "Cl"}

def mixed_basis(elements, heavy_basis="SARC-DKH2", light_basis="def2-TZVP",
                heavy_elems=("La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
                             "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm")):
    b = {}
    for el in set(elements):
        b[el] = heavy_basis if el in heavy_elems else light_basis
    return b
