import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "results"; F = "figures"
os.makedirs(F, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 140})

def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{F}/{name}.pdf"); fig.savefig(f"{F}/{name}.png", dpi=160)
    plt.close(fig); print("wrote", name)

def jload(p):
    return json.load(open(p)) if os.path.exists(p) else None

d = jload(f"{R}/dft/relativistic_effect.json")
if d:
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    for tag, c in (("x2c", "C3"), ("nonrel", "C0")):
        rs = np.array(d[tag]["rs"]); es = np.array(d[tag]["es"])
        es = (es - es.min())*2625.5
        lab = f"X2C scalar-rel.\n($r_{{min}}$={d['x2c']['r_min']:.3f} Å)" if tag=="x2c" \
              else f"non-relativistic\n($r_{{min}}$={d['nonrel']['r_min']:.3f} Å)"
        ax.plot(rs, es, "o-", ms=3, color=c, label=lab)
    ax.set_xlabel("U=O distance (Å)"); ax.set_ylabel("Rel. energy (kJ/mol)")
    ax.set_title(f"Uranyl U=O bond (PBE0/SARC-DKH2)\nΔr = {d['contraction_A']*100:.1f} pm")
    ax.legend(fontsize=7)
    save(fig, "fig1_relativistic_effect")

d = jload(f"{R}/scco2/density_validation.json")
if d:
    d320 = [x for x in d if x["T_K"] == 320]
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    P = [x["P_bar"]/10 for x in d320]
    ax.errorbar(P, [x["rho_md"] for x in d320], yerr=[x["rho_md_std"] for x in d320],
                fmt="o", color="C3", label="TraPPE-CO$_2$ MD", capsize=3)
    ax.plot(P, [x["rho_eos"] for x in d320], "s--", color="k", label="Span–Wagner EOS")
    ax.set_xlabel("Pressure (MPa)"); ax.set_ylabel("Density (g/cm$^3$)")
    ax.set_title("scCO$_2$ density, T = 320 K"); ax.legend(fontsize=8)
    save(fig, "fig2_scco2_density")

d = jload(f"{R}/mlp/mlp_eval.json")
if d and "parity" in d:
    fig, axs = plt.subplots(1, 2, figsize=(7.5, 3.4))
    for ax, key, lab in [(axs[0], "energy", "Energy (eV/atom)"),
                         (axs[1], "force", "Force (eV/Å)")]:
        for split, c in (("test", "C0"), ("ood", "C3")):
            p = d["parity"].get(split, {}).get(key)
            if p: ax.scatter(p["ref"], p["pred"], s=5, alpha=0.4, color=c,
                             label=f"{split} (RMSE {d['rmse'][split][key]:.3g})")
        lo = min(ax.get_xlim()[0], ax.get_ylim()[0]); hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax.set_xlabel(f"DFT {lab}"); ax.set_ylabel(f"MACE {lab}"); ax.legend(fontsize=7)
    fig.suptitle("MACE vs X2C-DFT (split-by-source test + OOD scan)")
    save(fig, "fig3_mlp_parity")

d = jload(f"{R}/scco2/solvation_fe.json")
if d:
    ch4 = [x for x in d if x["solute"] == "CH4-UA"]
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.errorbar([x["rho_eos"] for x in ch4], [x["dG_solv_kJ"] for x in ch4],
                yerr=[x["dG_err_kJ"] for x in ch4], fmt="o-", capsize=3, color="C2")
    ax.set_xlabel("scCO$_2$ density (g/cm$^3$)"); ax.set_ylabel("$\\Delta G_{solv}$ (kJ/mol)")
    ax.set_title("CH$_4$ solvation free energy vs scCO$_2$ density")
    save(fig, "fig4_solvation_fe")

d = jload(f"{R}/generative/generative_results.json")
if d:
    fig, axs = plt.subplots(1, 2, figsize=(7.8, 3.3))
    for tag, c in (("history_extractant", "C3"), ("history_control", "C0")):
        h = d.get(tag)
        if h:
            axs[0].plot([x["gen"] for x in h], [x["best"] for x in h], "-",
                        color=c, label=tag.replace("history_", ""))
    axs[0].set_xlabel("GA generation"); axs[0].set_ylabel("best score"); axs[0].legend(fontsize=8)
    axs[0].set_title("GB-GA optimisation")

    ms = {k: d[f"metrics_{k}"] for k in ("extractant", "control", "random") if f"metrics_{k}" in d}
    keys = ["uniqueness", "novelty", "int_div"]
    xpos = np.arange(len(keys)); w = 0.25
    for i, (lab, m) in enumerate(ms.items()):
        axs[1].bar(xpos + i*w, [m[k] for k in keys], w, label=lab)
    axs[1].set_xticks(xpos + w); axs[1].set_xticklabels(keys, fontsize=8)
    axs[1].legend(fontsize=7); axs[1].set_title("de novo metrics")
    save(fig, "fig5_generative")

print("figures done")
