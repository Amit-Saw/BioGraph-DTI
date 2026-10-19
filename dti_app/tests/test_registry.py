"""Milestone 3 check: the app must reproduce the predictions saved during training/testing.
Needs the files from the Drive copy step:
  checkpoints/<endpoint>_<split>/<model>/best.pt
  data/results/predicted_vs_actual/<endpoint>_<split>_<model>.csv   (columns: canonical_smiles, protein_sequence,
                                                                     actual_p, predicted_p, ...)
"""
import numpy as np
import pandas as pd
import pytest

import config
from dti.protein import ESMEmbedder, EmbeddingCache, clean_sequence
from dti.registry import ModelRegistry

N_ROWS = 200
TOL = 5e-3          # recomputed ESM embeddings can differ slightly from the stored ones


@pytest.fixture(scope="module")
def registry():
    reg = ModelRegistry(EmbeddingCache(ESMEmbedder()))
    if reg.load_all() == 0:
        pytest.skip("no checkpoints found in checkpoints/")
    return reg


def _cases():
    root = config.DATA_DIR / "results" / "predicted_vs_actual"
    return sorted(root.glob("*.csv")) if root.exists() else []


@pytest.mark.parametrize("csv", _cases(), ids=lambda p: p.stem)
def test_reproduces_saved_predictions(registry, csv):
    endpoint, rest = csv.stem.split("_", 1)                  # Ki_cold_drug_attentivefp_esm
    split, name = rest.rsplit("_", 2)[0], "_".join(rest.rsplit("_", 2)[1:])
    if (endpoint, split, name) not in registry.models:
        pytest.skip(f"{endpoint}/{split}/{name} not loaded")
    df = pd.read_csv(csv).head(N_ROWS)
    seqs = [clean_sequence(s) for s in df.protein_sequence]
    pred = registry.predict_pairs(endpoint, split, name, df.canonical_smiles.tolist(), seqs)
    diff = np.abs(pred - df.predicted_p.to_numpy())
    assert np.nanmax(diff) < TOL, f"max abs diff {np.nanmax(diff):.4f} (tolerance {TOL})"
