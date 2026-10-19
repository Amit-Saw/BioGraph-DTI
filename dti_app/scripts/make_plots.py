"""Draw predicted-vs-actual plots for the About page.

    python scripts/make_plots.py

Reads  data/results/predicted_vs_actual/<endpoint>_<split>_<model>.csv   (columns actual_p, predicted_p)
Writes static/img/plots/<name>.png"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

import config


def main():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("matplotlib is not installed:  pip install matplotlib")

    src = config.RESULTS_DIR / "predicted_vs_actual"
    files = sorted(src.glob("*.csv")) if src.exists() else []
    if not files:
        sys.exit(f"No CSV files in {src}. Run the test cell in Colab, then scripts/setup_data.py.")
    out = config.BASE_DIR / "static" / "img" / "plots"
    out.mkdir(parents=True, exist_ok=True)

    for f in files:
        df = pd.read_csv(f, usecols=["actual_p", "predicted_p"])
        y, p = df.actual_p.to_numpy(), df.predicted_p.to_numpy()
        r = float(np.corrcoef(y, p)[0, 1]); rmse = float(np.sqrt(np.mean((p - y) ** 2)))
        fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=130)
        hb = ax.hexbin(y, p, gridsize=55, bins="log", cmap="viridis", mincnt=1)
        lo, hi = float(min(y.min(), p.min())), float(max(y.max(), p.max()))
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
        ax.set(xlabel="Actual pAffinity", ylabel="Predicted pAffinity", title=f.stem.replace("_", " "))
        ax.text(0.04, 0.95, f"Pearson {r:.3f}\nRMSE {rmse:.3f}\nn = {len(y):,}", transform=ax.transAxes, va="top",
                bbox=dict(boxstyle="round", fc="white", alpha=0.85))
        fig.colorbar(hb, ax=ax, label="log10 count")
        fig.tight_layout()
        fig.savefig(out / f"{f.stem}.png")
        plt.close(fig)
        print("wrote", out / f"{f.stem}.png")


if __name__ == "__main__":
    main()
