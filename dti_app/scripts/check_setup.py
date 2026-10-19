"""Checks that everything is in place before you start the app.

    python scripts/check_setup.py            # files + imports + loads the checkpoints
    python scripts/check_setup.py --predict  # also runs one real prediction (downloads ESM-2 on first use)
"""
import argparse
import importlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OK, BAD, WARN = "[ OK ]", "[FAIL]", "[warn]"
failures = 0


def report(ok, msg, optional=False):
    global failures
    tag = OK if ok else (WARN if optional else BAD)
    if not ok and not optional:
        failures += 1
    print(f"{tag} {msg}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predict", action="store_true", help="run one end-to-end prediction")
    args = ap.parse_args()

    print("== Python and packages")
    report(sys.version_info >= (3, 10), f"Python {sys.version.split()[0]} (need 3.10+)")
    for mod in ("torch", "torch_geometric", "rdkit", "transformers", "flask", "numpy", "pandas", "requests"):
        try:
            m = importlib.import_module(mod)
            report(True, f"{mod} {getattr(m, '__version__', '')}")
        except Exception as e:                                          # noqa: BLE001
            report(False, f"{mod} not importable: {e}")
    for mod in ("matplotlib", "vina", "meeko"):
        try:
            importlib.import_module(mod); report(True, f"{mod} (optional)")
        except Exception:                                               # noqa: BLE001
            report(False, f"{mod} not installed (optional)", optional=True)
    report(bool(shutil.which("obabel")), "Open Babel 'obabel' on PATH (optional, docking only)", optional=True)

    import config
    print("\n== Files")
    ckpts = sorted(config.CHECKPOINT_DIR.glob("*/*/best.pt"))
    report(len(ckpts) > 0, f"{len(ckpts)} checkpoint(s) in checkpoints/")
    for c in ckpts:
        report((c.parent / "test_metrics.json").exists(), f"  {c.parent.parent.name}/{c.parent.name}: test_metrics.json", optional=True)
    for ep in config.ENDPOINTS:
        report((config.REFERENCE_DIR / f"{ep}_train.csv").exists(), f"data/reference/{ep}_train.csv")
    report((config.RESULTS_DIR / "comparison_table.csv").exists(), "data/results/comparison_table.csv (About page)", optional=True)
    report(config.TRAIN_EMB_PKL.exists(), "cache/protein_embeddings.pkl (saves recomputing ESM-2)", optional=True)
    report(config.ESM_LOCAL_DIR.exists(), "checkpoints/esm2_t6_8M (offline ESM-2; else downloaded on first use)", optional=True)
    report((config.RESULTS_DIR / "applicability.json").exists(),
           "data/results/applicability.json (run scripts/calibrate_applicability.py)", optional=True)

    if failures == 0 and ckpts:
        print("\n== Loading checkpoints")
        try:
            from dti.protein import ESMEmbedder, EmbeddingCache
            from dti.registry import ModelRegistry
            reg = ModelRegistry(EmbeddingCache(ESMEmbedder()))
            n = reg.load_all()
            report(n == len(ckpts), f"loaded {n} of {len(ckpts)} checkpoints with strict=True")
            for m in reg.available():
                print(f"       {m['endpoint']}/{m['split']}/{m['model']}  test RMSE {m['test_rmse']}")
            if args.predict:
                print("\n== One prediction (EGFR / erlotinib)")
                from services import uniprot
                _, seq = uniprot.fetch_sequence("P00533")
                from dti.protein import clean_sequence
                seq = clean_sequence(seq)
                for ep in config.ENDPOINTS:
                    from services import predict_service
                    res = predict_service.predict_endpoint(
                        reg, ep, "COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC", seq)
                    p = res.get("primary")
                    report(p is not None, f"{ep}: " + (f"p{ep} {p['pAffinity']} ({p['conc']}, {p['label']})" if p else res))
        except Exception as e:                                          # noqa: BLE001
            report(False, f"loading failed: {e}")

    print()
    if failures:
        print(f"{failures} problem(s) found - fix the [FAIL] lines above.")
        sys.exit(1)
    print("Everything required is in place.  Start the app:  python app.py")


if __name__ == "__main__":
    main()
