"""Calibrate the "similar to training data?" warning.

For each endpoint it takes validation pairs (cold_drug split), predicts them with the main model, measures each drug's
highest Tanimoto similarity to the training drugs, and shows how the error changes with similarity.  The thresholds
are the smallest similarities from which the error is clearly lower than average.

    python scripts/calibrate_applicability.py [--n 3000]

Needs: checkpoints, data/reference/<ep>_train.csv and <ep>_validation.csv.
Writes: data/results/applicability.json  (read by config.py and the About page)."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config                                            # noqa: E402

EDGES = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0001]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000, help="validation rows per endpoint")
    a = ap.parse_args()

    from dti.protein import ESMEmbedder, EmbeddingCache, clean_sequence
    from dti.registry import ModelRegistry
    from services import predict_service
    from services.chem import ReferenceIndex

    reg = ModelRegistry(EmbeddingCache(ESMEmbedder()))
    reg.load_all()
    ref = ReferenceIndex()
    table, highs, mids = [], [], []

    for ep in config.ENDPOINTS:
        val_path = config.REFERENCE_DIR / f"{ep}_validation.csv"
        primary, _ = predict_service.select_models(reg, ep)
        if primary is None or not ref.available(ep) or not val_path.exists():
            print(f"{ep}: skipped (needs a model, {ep}_train.csv and {ep}_validation.csv)")
            continue
        df = pd.read_csv(val_path, usecols=["canonical_smiles", "protein_sequence", "p_affinity"])
        df = df.sample(min(a.n, len(df)), random_state=0).reset_index(drop=True)
        seqs = [clean_sequence(s) for s in df.protein_sequence]
        print(f"{ep}: predicting {len(df)} validation pairs with {primary[1]} ({primary[0]}) ...")
        pred = reg.predict_pairs(ep, primary[0], primary[1], df.canonical_smiles.tolist(), seqs)
        sim = np.array([np.nan if (v := ref.max_similarity(ep, s)) is None else v for s in df.canonical_smiles])
        err = pred - df.p_affinity.to_numpy()
        ok = ~np.isnan(err) & ~np.isnan(sim)
        err, sim = err[ok], sim[ok]
        overall = float(np.sqrt(np.mean(err ** 2)))
        print(f"  overall RMSE {overall:.3f} on {len(err)} pairs")

        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            m = (sim >= lo) & (sim < hi)
            if m.sum() == 0:
                continue
            row = {"endpoint": ep, "bin": f"{lo:.1f}-{min(hi, 1.0):.1f}", "n": int(m.sum()),
                   "mae": round(float(np.abs(err[m]).mean()), 3),
                   "rmse": round(float(np.sqrt((err[m] ** 2).mean())), 3)}
            table.append(row)
            print(f"  similarity {row['bin']:>8}: n={row['n']:5d}  MAE {row['mae']:.3f}  RMSE {row['rmse']:.3f}")

        def first_edge(candidates, factor):
            for e in candidates:
                m = sim >= e
                if m.sum() >= 100 and np.sqrt(np.mean(err[m] ** 2)) <= factor * overall:
                    return e
            return None

        h = first_edge([0.4, 0.5, 0.6, 0.7, 0.8], 0.90)
        m_ = first_edge([0.2, 0.3, 0.4, 0.5], 1.00)
        if h is not None: highs.append(h)
        if m_ is not None: mids.append(m_)

    if not table:
        sys.exit("Nothing calibrated.")
    sim_high = float(np.median(highs)) if highs else config.SIM_HIGH
    sim_mid = float(np.median(mids)) if mids else config.SIM_MID
    sim_mid = min(sim_mid, sim_high)
    out = {"sim_high": sim_high, "sim_mid": sim_mid, "table": table}
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "applicability.json").write_text(json.dumps(out, indent=2))
    print(f"\nsim_high = {sim_high}, sim_mid = {sim_mid}  ->  data/results/applicability.json")
    print("Restart the app to use the new thresholds.")


if __name__ == "__main__":
    main()
