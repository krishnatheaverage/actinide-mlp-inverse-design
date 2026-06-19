# Hyper-Relativistic MLP / Inverse-Design Workflow for Actinide Extractants in scCO₂

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20764583.svg)](https://doi.org/10.5281/zenodo.20764583)

A fully open, laptop-scale, end-to-end and **reproducible** computational workflow:
relativistic DFT reference data → machine-learning interatomic potential → supercritical-CO₂
free-energy MD → surrogate-guided generative ligand design, with an X2C-DFT actinide/lanthanide
binding re-score. Built and run entirely on a single Apple M4 (16 GB).

> **Honesty statement.** "Hyper-relativistic" and "transient" in the paper title are *aspirational*.
> The delivered content is a scalar-X2C / MACE / TraPPE-CO₂ / graph-GA proof of concept on **static,
> closed-shell f⁰ model systems**, with **no spin–orbit coupling, no transient species, and no
> experimental validation**. The value of this repository is an honest, reproducible scaffold and an
> explicit catalogue of where such pipelines break. Every number in the manuscript is produced by the
> code here; none is hand-entered.

## Key results (all computed, reproducible)

| Stage | Quantity | Value |
|---|---|---|
| Relativistic DFT | uranyl r(U=O), X2C-PBE0/SARC-DKH2 | 1.685 Å |
| | relativistic shift (contracted vs decontracted) | −5.1 pm / −5.1 pm (not a basis artifact) |
| | uranyl T₁ coupled-cluster diagnostic | 0.031 (> 0.02 → mild multireference) |
| ML potential (MACE) | in-distribution test E / F RMSE | 2.4 meV/atom / 36 meV/Å |
| | out-of-distribution (bond scans) F RMSE | 1287 meV/Å (~35× worse) |
| | leakage check (test↔train vs intra-train) | 0.59 / 0.59 Å (no leakage) |
| scCO₂ MD | TraPPE-CO₂ density vs Span–Wagner EOS | +5% (30 MPa) → +219% (near-critical) |
| | CH₄ solvation ΔG (MBAR, 13- & 24-window) | ≈ −1 kJ/mol; overlap 0.06–0.11 (intrinsically poor) |
| Inverse design | donor-strength surrogate R² (scaffold split) | −0.12 (does not beat the mean) |
| | GB-GA validity / uniqueness / novelty / IntDiv | 1.00 / 1.00 / 0.99 / 0.79 (random: 0.53 unique) |
| | top candidates | organophosphorus, SA 2.5–4.1, logP≈4, novel |
| | An/Ln ΔΔE(Ac−La), rigid single-point | +176 ± 128 kJ/mol (large scatter; f⁰ baseline) |

## Repository layout

```
src/01_relativistic_dft/   X2C-PBE0 reference data, relativistic effect, T1 diagnostic
src/02_mlp/                MLP dataset generation, MACE training/eval, leakage check
src/03_scco2_md/           TraPPE-CO2 system, density validation, MBAR solvation FE
src/04_generative/         extractant library, V_min surrogate, GB-GA, An/Ln re-score
src/common/                relativistic basis helper, figure + macro generation
results/                   all JSON outputs, logs, dataset, figures
paper/                     LaTeX manuscript (main.tex) + auto-filled numbers + PDF
run_pipeline*.sh           orchestration scripts
```

## Reproduce

```bash
# 1. environment (Miniforge / conda-forge)
conda create -n actinide -c conda-forge python=3.11 pyscf rdkit openmm ase xtb \
    mdtraj pymbar scikit-learn matplotlib h5py geometric basis_set_exchange
conda activate actinide
pip install mace-torch CoolProp
export KMP_DUPLICATE_LIB_OK=TRUE   # torch + conda libomp coexistence

# 2. run stages (each writes JSON to results/)
python src/01_relativistic_dft/run_dft.py
python src/02_mlp/gen_dataset.py && python src/02_mlp/train_mace.py
python src/03_scco2_md/run_density.py && python src/03_scco2_md/run_solvation_fe.py
python src/04_generative/build_surrogate_data.py && \
    python src/04_generative/run_generative.py && python src/04_generative/run_rescore.py

# 3. assemble numbers + figures + PDF
python src/common/fill_macros.py && python src/common/make_figures.py
cd paper && tectonic main.tex
```

## Adversarial review record

The manuscript was hardened through five Reviewer-2-style cycles (theory; data/leakage;
solvation/sampling; ligand validation; claims/reproducibility). Each revision is a *real*
code or text change — e.g. the leakage check, the decontracted relativistic recompute, the
T₁ diagnostic, the 24-window free-energy re-run, and the synthesizability audit were all
added in response to the critiques.

## Limitations (read before reusing)

Scalar-X2C only (no SOC); single-reference DFT on f⁰ centers (T₁=0.031 flags residual
multireference error); fixed-charge non-polarizable CO₂ (invalid near the critical point and
for charged solutes); donor-strength surrogate does not generalize (R²<0); An/Ln re-score is a
rigid single-point estimate on the f⁰ model pair, not the chemically decisive Am/Eu; no ligand
is synthesized, assayed, or checked for radiolytic/hydrolytic stability.

## License & citation

Code released under the MIT License. This is a methodological demonstration; if you build on it,
cite the repository and the upstream method papers listed in `paper/refs.bib`
(PySCF, MACE, TraPPE-CO₂, MBAR, GB-GA, X2C, SARC, Span–Wagner).
