"""Train a MACE potential on the UO2(H2O)4^2+ X2C-DFT dataset and evaluate it
under a leakage-aware protocol:
  - train/val/test drawn from the normal-mode-sampled frames, which are IID draws
    (not an autocorrelated MD trajectory), so a random split among them does not
    leak correlated near-duplicates;
  - the U=O / U-O(water) bond-stretch SCANS are held out ENTIRELY as an
    out-of-distribution (OOD) extrapolation test;
  - errors are reported element-resolved (U centre vs O vs H) so heavy-atom error
    is not masked by abundant light atoms.
Runs on the Apple GPU (MPS).
"""
import os, sys, json, subprocess, shutil
import numpy as np
from ase.io import read, write

DATA = "results/mlp/uo2_dataset.extxyz"
OUT = "results/mlp"
NAME = "uo2_mace"

def split():
    frames = read(DATA, ":")
    nms = [a for a in frames if a.info["config_type"].startswith("NMS")]
    ood = [a for a in frames if a.info["config_type"].startswith("SCAN")]
    rng = np.random.default_rng(0); idx = rng.permutation(len(nms))
    n_test = max(20, int(0.15*len(nms)))
    test = [nms[i] for i in idx[:n_test]]
    trainval = [nms[i] for i in idx[n_test:]]
    write(f"{OUT}/train.xyz", trainval); write(f"{OUT}/test.xyz", test)
    write(f"{OUT}/ood.xyz", ood)
    print(f"split: train+val={len(trainval)} test(in-dist)={len(test)} ood(scan)={len(ood)}")
    return len(trainval), len(test), len(ood)

def train(max_epochs=250):
    py = sys.executable
    cmd = [py, "-m", "mace.cli.run_train",
        "--name", NAME, "--train_file", f"{OUT}/train.xyz",
        "--valid_fraction", "0.15", "--test_file", f"{OUT}/test.xyz",
        "--E0s", "average", "--model", "MACE",
        "--num_interactions", "2", "--num_channels", "32",
        "--max_L", "1", "--correlation", "3", "--r_max", "5.0",
        "--max_num_epochs", str(max_epochs), "--batch_size", "8",
        "--swa", "--start_swa", str(int(max_epochs*0.75)),
        "--energy_key", "energy", "--forces_key", "forces",
        "--energy_weight", "1.0", "--forces_weight", "25.0",
        "--device", "cpu", "--default_dtype", "float64", "--seed", "1",
        "--model_dir", OUT, "--log_dir", f"{OUT}/logs",
        "--checkpoints_dir", f"{OUT}/ckpt", "--results_dir", f"{OUT}/mace_results"]
    print("running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

def evaluate():
    from mace.calculators import MACECalculator
    model_path = None
    for cand in (f"{OUT}/{NAME}.model", f"{OUT}/{NAME}_swa.model",
                 f"{OUT}/{NAME}_stagetwo.model", f"{OUT}/{NAME}_compiled.model"):
        if os.path.exists(cand): model_path = cand
    if model_path is None:
        cands = [f for f in os.listdir(OUT) if f.endswith(".model")]
        model_path = os.path.join(OUT, cands[0]) if cands else None
    print("loading model:", model_path, flush=True)
    calc = MACECalculator(model_paths=model_path, device="cpu", default_dtype="float64")

    def errs(xyz):
        frames = read(xyz, ":")
        dE, dFsq, fel = [], [], {"U": [], "O": [], "H": []}
        e_ref_all, e_pred_all, f_ref_all, f_pred_all = [], [], [], []
        for a in frames:
            n = len(a)
            # ASE extxyz round-trip attaches energy/forces to a SinglePointCalculator
            e_ref = a.info.get("energy", a.get_potential_energy())
            f_ref = a.arrays["forces"] if "forces" in a.arrays else a.get_forces()
            a2 = a.copy(); a2.calc = calc
            e_pred = a2.get_potential_energy(); f_pred = a2.get_forces()
            dE.append((e_pred - e_ref)/n)
            dFsq += list(((f_pred - f_ref)**2).sum(1))
            for i, s in enumerate(a.get_chemical_symbols()):
                fel.setdefault(s, []).append(np.sqrt(((f_pred[i]-f_ref[i])**2).sum()))
            e_ref_all.append(e_ref/n); e_pred_all.append(e_pred/n)
            f_ref_all += list(f_ref.ravel()); f_pred_all += list(f_pred.ravel())
        return dict(
            e_rmse=float(np.sqrt(np.mean(np.array(dE)**2))),
            f_rmse=float(np.sqrt(np.mean(dFsq))),
            f_rmse_by_element={s: float(np.sqrt(np.mean(np.array(v)**2)))
                               for s, v in fel.items() if v},
            parity=dict(energy=dict(ref=e_ref_all, pred=e_pred_all),
                        force=dict(ref=f_ref_all[:2000], pred=f_pred_all[:2000])))
    res = dict(test=errs(f"{OUT}/test.xyz"), ood=errs(f"{OUT}/ood.xyz"))
    out = dict(
        rmse=dict(test=dict(energy=res["test"]["e_rmse"], force=res["test"]["f_rmse"]),
                  ood=dict(energy=res["ood"]["e_rmse"], force=res["ood"]["f_rmse"])),
        f_rmse_by_element=dict(test=res["test"]["f_rmse_by_element"],
                               ood=res["ood"]["f_rmse_by_element"]),
        parity=dict(test=res["test"]["parity"], ood=res["ood"]["parity"]),
        meta=json.load(open(f"{OUT}/dataset_meta.json")) if os.path.exists(f"{OUT}/dataset_meta.json") else {}
    )
    json.dump(out, open(f"{OUT}/mlp_eval.json", "w"), indent=2)
    print("=== MLP eval ===")
    print(f"  test : E {out['rmse']['test']['energy']*1000:.2f} meV/atom  "
          f"F {out['rmse']['test']['force']*1000:.1f} meV/A")
    print(f"  OOD  : E {out['rmse']['ood']['energy']*1000:.2f} meV/atom  "
          f"F {out['rmse']['ood']['force']*1000:.1f} meV/A")
    print(f"  force by element (test): {out['f_rmse_by_element']['test']}")
    print("MLP_DONE")

if __name__ == "__main__":
    split(); train(); evaluate()
