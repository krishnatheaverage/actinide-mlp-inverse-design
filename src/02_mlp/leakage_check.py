import json, numpy as np
from ase.io import read
from scipy.spatial.distance import pdist, cdist

def descriptor(a):

    return np.sort(pdist(a.get_positions()))

frames = read("results/mlp/uo2_dataset.extxyz", ":")
nms = [a for a in frames if a.info["config_type"].startswith("NMS")]
ood = [a for a in frames if a.info["config_type"].startswith("SCAN")]
rng = np.random.default_rng(0); idx = rng.permutation(len(nms))
n_test = max(20, int(0.15*len(nms)))
test = [nms[i] for i in idx[:n_test]]; train = [nms[i] for i in idx[n_test:]]

Dtr = np.array([descriptor(a) for a in train])
Dte = np.array([descriptor(a) for a in test])
Doo = np.array([descriptor(a) for a in ood])

nn_test = cdist(Dte, Dtr).min(1)
nn_ood = cdist(Doo, Dtr).min(1)

intra = []
M = cdist(Dtr, Dtr); np.fill_diagonal(M, np.inf)
intra = M.min(1)

out = dict(
    nn_test_min=float(nn_test.min()), nn_test_med=float(np.median(nn_test)),
    nn_ood_min=float(nn_ood.min()), nn_ood_med=float(np.median(nn_ood)),
    intra_train_med=float(np.median(intra)),
    n_train=len(train), n_test=len(test), n_ood=len(ood))
print(f"nearest-train descriptor distance (Angstrom-scale):")
print(f"  test  : min {out['nn_test_min']:.3f}  median {out['nn_test_med']:.3f}")
print(f"  OOD   : min {out['nn_ood_min']:.3f}  median {out['nn_ood_med']:.3f}")
print(f"  intra-train median NN: {out['intra_train_med']:.3f}")
json.dump(out, open("results/mlp/leakage_check.json","w"), indent=2)
print("LEAKAGE_DONE")
