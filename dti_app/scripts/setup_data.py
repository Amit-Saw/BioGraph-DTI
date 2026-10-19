"""Copy the files the app needs from your Drive training folder into this project.

    python scripts/setup_data.py --src "G:/My Drive/Colab Notebooks/Deep Learning 2/Project/DTI_Project"

Which models to copy is set in MODELS below (edit if you extended / retrained others).
Only best.pt, config.json, test_metrics.json are copied for each model (never latest.pt)."""
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (endpoint, split, model) -> folder checkpoints/<endpoint>_<split>/<model>/
MODELS = [("Ki", "cold_drug", "attentivefp_esm"), ("Ki", "random", "mpnn_esm"),
          ("IC50", "cold_drug", "attentivefp_esm"), ("IC50", "random", "attentivefp_esm")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Drive folder that contains 13_Models, 10_Splits_v3, 08_Embeddings")
    ap.add_argument("--dst", default=str(ROOT), help="project folder (default: this project)")
    ap.add_argument("--no-embeddings", action="store_true", help="skip the (large) protein_embeddings.pkl")
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    if not src.exists():
        sys.exit(f"Source folder not found: {src}")

    missing, copied = [], 0

    def cp(s, d, required=True):
        nonlocal copied
        if not s.exists():
            if required:
                missing.append(str(s))
            else:
                print(f"  (optional, not found) {s.name}")
            return
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, d)
        copied += 1
        print(f"  ok  {d.relative_to(dst)}")

    print("Model checkpoints")
    for ep, split, name in MODELS:
        m = src / "13_Models" / f"{ep}_{split}" / name
        t = dst / "checkpoints" / f"{ep}_{split}" / name
        for f in ("best.pt", "config.json", "test_metrics.json"):
            cp(m / f, t / f, required=(f != "config.json"))
        cp(m / "predicted_vs_actual.csv",
           dst / "data/results/predicted_vs_actual" / f"{ep}_{split}_{name}.csv", required=False)
        cp(m / "history.json", dst / "data/results/history" / f"{ep}_{split}_{name}.json", required=False)

    print("Reference data (training drugs for the similarity check)")
    for ep in ("Ki", "IC50"):
        d = src / "10_Splits_v3" / f"{ep}_cold_drug"
        cp(d / "train.csv", dst / "data/reference" / f"{ep}_train.csv")
        cp(d / "validation.csv", dst / "data/reference" / f"{ep}_validation.csv", required=False)

    print("Results table")
    cp(src / "13_Models" / "comparison_table.csv", dst / "data/results/comparison_table.csv", required=False)

    if not a.no_embeddings:
        print("Protein embeddings (optional, saves recomputing ESM-2)")
        cp(src / "08_Embeddings" / "protein_embeddings.pkl", dst / "cache/protein_embeddings.pkl", required=False)

    print(f"\nCopied {copied} file(s).")
    if missing:
        print("\nMISSING required files:")
        for m in missing:
            print("  -", m)
        sys.exit(1)
    print("Next:  python scripts/check_setup.py")


if __name__ == "__main__":
    main()
