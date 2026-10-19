"""Loads every checkpoint found in checkpoints/<endpoint>_<split>/<model>/ once and serves predictions."""
import json
import threading
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from torch_geometric.data import Batch

import config
from dti.encoders import DTIModel, ENCODERS
from dti.featurize import smiles_to_data
from dti.smiles import validate_smiles


@dataclass
class LoadedModel:
    endpoint: str
    split: str
    name: str
    model: Any
    test_rmse: Optional[float]
    val_rmse: Optional[float]
    epochs_run: Optional[int]


class ModelRegistry:
    def __init__(self, cache, device=None):
        self.cache = cache                                        # EmbeddingCache
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}                                          # (endpoint, split, name) -> LoadedModel
        self.infer_lock = threading.Lock()

    # ------------------------------------------------------------ loading
    def load_all(self, checkpoint_dir=None):
        root = checkpoint_dir or config.CHECKPOINT_DIR
        for ckpt in sorted(root.glob("*/*/best.pt")):
            folder, name = ckpt.parent.parent.name, ckpt.parent.name
            if name not in ENCODERS or "_" not in folder:
                continue
            endpoint, split = folder.split("_", 1)               # "Ki_cold_drug" -> Ki, cold_drug
            model = DTIModel(name)
            model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True), strict=True)
            model.to(self.device).eval()
            metrics = ckpt.parent / "test_metrics.json"
            test_rmse = val_rmse = epochs = None
            if metrics.exists():
                r = json.loads(metrics.read_text())
                test_rmse, val_rmse, epochs = r["test"]["RMSE"], r["val"]["RMSE"], r.get("epochs_run")
            self.models[(endpoint, split, name)] = LoadedModel(endpoint, split, name, model,
                                                               test_rmse, val_rmse, epochs)
        return len(self.models)

    def available(self):
        return [{"endpoint": m.endpoint, "split": m.split, "model": m.name,
                 "test_rmse": m.test_rmse, "val_rmse": m.val_rmse, "epochs_run": m.epochs_run}
                for m in self.models.values()]

    def get(self, endpoint, split, name):
        try:
            return self.models[(endpoint, split, name)]
        except KeyError:
            raise KeyError(f"No checkpoint loaded for {endpoint}/{split}/{name}")

    def best_model(self, endpoint, split):
        """Name of the loaded model with the lowest validation RMSE for this endpoint+split."""
        cands = [m for m in self.models.values() if m.endpoint == endpoint and m.split == split]
        if not cands:
            return None
        return min(cands, key=lambda m: m.val_rmse if m.val_rmse is not None else 1e9).name

    # ------------------------------------------------------------ prediction
    def predict_data(self, endpoint, split, name, data_list, batch_size=256):
        """pAffinity for a list of ready PyG Data objects (used by the explainer)."""
        lm = self.get(endpoint, split, name)
        out = []
        with self.infer_lock, torch.inference_mode():
            for k in range(0, len(data_list), batch_size):
                batch = Batch.from_data_list(data_list[k:k + batch_size]).to(self.device)
                out.append(lm.model(batch).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)

    def predict_pairs(self, endpoint, split, name, smiles_list, seqs, batch_size=256, progress=None):
        """pAffinity for aligned lists of SMILES and (already cleaned) sequences.
        Invalid SMILES -> NaN.  Protein embeddings are looked up once per distinct sequence."""
        emb = {s: self.cache.get(s) for s in set(seqs)}
        out = np.full(len(smiles_list), np.nan, dtype=np.float32)
        items = []
        for i, (smi, seq) in enumerate(zip(smiles_list, seqs)):
            if validate_smiles(smi)[0]:
                items.append((i, smiles_to_data(smi, emb[seq])))
        for k in range(0, len(items), batch_size):
            chunk = items[k:k + batch_size]
            preds = self.predict_data(endpoint, split, name, [d for _, d in chunk], batch_size)
            for (i, _), p in zip(chunk, preds):
                out[i] = p
            if progress:
                progress(min(k + batch_size, len(items)), len(items))
        return out

    def predict_many(self, endpoint, split, name, smiles_list, seq, batch_size=256, progress=None):
        """Many drugs, one protein."""
        return self.predict_pairs(endpoint, split, name, smiles_list, [seq] * len(smiles_list),
                                  batch_size, progress)

    def predict_pair(self, endpoint, split, name, smiles, seq):
        return float(self.predict_many(endpoint, split, name, [smiles], seq)[0])
