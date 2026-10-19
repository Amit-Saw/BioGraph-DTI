"""Fake model registry / reference index so the Flask app can be tested without checkpoints."""
from types import SimpleNamespace

import numpy as np


def _fake_p(smiles):
    return 5.0 + (sum(map(ord, smiles)) % 50) / 10.0          # deterministic value in 5.0 .. 9.9


class FakeCache:
    def get(self, seq):
        return np.zeros(320, dtype=np.float32)


class FakeRegistry:
    def __init__(self):
        self.cache = FakeCache()
        self.models = {}
        for ep, split, name, rmse in [("Ki", "cold_drug", "attentivefp_esm", 0.98),
                                      ("Ki", "random", "mpnn_esm", 0.92),
                                      ("IC50", "cold_drug", "attentivefp_esm", 1.05),
                                      ("IC50", "random", "attentivefp_esm", 0.95)]:
            self.models[(ep, split, name)] = SimpleNamespace(endpoint=ep, split=split, name=name, model=None,
                                                             test_rmse=rmse, val_rmse=rmse, epochs_run=50)

    def available(self):
        return [{"endpoint": m.endpoint, "split": m.split, "model": m.name, "test_rmse": m.test_rmse,
                 "val_rmse": m.val_rmse, "epochs_run": m.epochs_run} for m in self.models.values()]

    def get(self, ep, split, name):
        try:
            return self.models[(ep, split, name)]
        except KeyError:
            raise KeyError(f"No checkpoint loaded for {ep}/{split}/{name}")

    def best_model(self, ep, split):
        c = [m for m in self.models.values() if m.endpoint == ep and m.split == split]
        return min(c, key=lambda m: m.val_rmse).name if c else None

    def predict_many(self, ep, split, name, smiles_list, seq, batch_size=256, progress=None):
        out = np.array([_fake_p(s) for s in smiles_list], dtype=np.float32)
        if progress:
            progress(len(smiles_list), len(smiles_list))
        return out

    def predict_pairs(self, ep, split, name, smiles_list, seqs, batch_size=256, progress=None):
        return self.predict_many(ep, split, name, smiles_list, None, batch_size, progress)

    def predict_pair(self, ep, split, name, smiles, seq):
        return float(_fake_p(smiles))

    def predict_data(self, ep, split, name, data_list, batch_size=256):
        return np.array([6.0 + 0.01 * float(d.x.sum()) for d in data_list], dtype=np.float32)


class FakeReference:
    def available(self, ep):
        return True

    def _load(self, ep):
        return None

    def applicability(self, ep, smiles):
        return {"max_tanimoto": 0.2, "level": "low", "message": "far from training data"}

    def nearest(self, ep, smiles, k=5, seq=None):
        return [{"smiles": "CCO", "similarity": 0.2, "median_pAffinity": 6.1, "n_targets": 2,
                 "n_measurements": 3, "same_target_pAffinity": None}]
