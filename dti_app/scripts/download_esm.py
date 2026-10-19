"""Download ESM-2 (8M) once into checkpoints/esm2_t6_8M so the app can run offline.

    python scripts/download_esm.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


def main():
    from transformers import AutoModel, AutoTokenizer
    config.ESM_LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    AutoTokenizer.from_pretrained(config.ESM_NAME).save_pretrained(config.ESM_LOCAL_DIR)
    AutoModel.from_pretrained(config.ESM_NAME).save_pretrained(config.ESM_LOCAL_DIR)
    print("saved to", config.ESM_LOCAL_DIR)


if __name__ == "__main__":
    main()
