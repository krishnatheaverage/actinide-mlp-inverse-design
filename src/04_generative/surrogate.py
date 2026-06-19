import numpy as np
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from descriptors import featurize_many

def scaffold_split(smiles, frac_train=0.8, seed=0):
    buckets = defaultdict(list)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        try:
            scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
        except Exception:
            scaf = ""
        buckets[scaf or f"_singleton_{i}"].append(i)
    groups = sorted(buckets.values(), key=len, reverse=True)
    rng = np.random.default_rng(seed)

    n = len(smiles); n_train = int(frac_train*n)
    train, test = [], []
    for g in groups:
        (train if len(train) < n_train else test).extend(g)
    return np.array(train), np.array(test)

class DonorSurrogate:
    def __init__(self):
        self.model = GradientBoostingRegressor(n_estimators=400, max_depth=3,
                                               learning_rate=0.03, subsample=0.8,
                                               random_state=0)
        self.mu = self.sd = None

    def fit(self, smiles, y):
        X, keep = featurize_many(smiles)
        y = np.array([y[s] for s in keep])
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Xs = (X - self.mu) / self.sd
        tr, te = scaffold_split(keep)
        self.model.fit(Xs[tr], y[tr])
        pred = self.model.predict(Xs[te])
        metrics = dict(r2_scaffold=float(r2_score(y[te], pred)),
                       mae_scaffold=float(mean_absolute_error(y[te], pred)),
                       n_train=int(len(tr)), n_test=int(len(te)),
                       y_std=float(y.std()))

        self.model.fit(Xs, y)
        return metrics

    def predict(self, smiles):
        v = featurize_many([smiles])
        if len(v[1]) == 0:
            return None
        Xs = (v[0] - self.mu) / self.sd
        return float(self.model.predict(Xs)[0])
