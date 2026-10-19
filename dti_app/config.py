"""Central configuration for the DTI web app.  Most values can be overridden with environment variables
(see .env.example)."""
import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints"      # <endpoint>_<split>/<model>/{best.pt,config.json,test_metrics.json}
CACHE_DIR = BASE_DIR / "cache"
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = BASE_DIR / "uploads"
REFERENCE_DIR = DATA_DIR / "reference"         # Ki_train.csv, IC50_train.csv (+ *_validation.csv)
RESULTS_DIR = DATA_DIR / "results"
EXAMPLES_DIR = DATA_DIR / "examples"

# ---- protein language model ----
ESM_NAME = "facebook/esm2_t6_8M_UR50D"
ESM_LOCAL_DIR = CHECKPOINT_DIR / "esm2_t6_8M"                 # optional offline copy (scripts/download_esm.py)
TRAIN_EMB_PKL = CACHE_DIR / "protein_embeddings.pkl"          # copied from Drive 08_Embeddings (optional)
USER_EMB_PKL = CACHE_DIR / "user_embeddings.pkl"              # embeddings of new proteins computed by the app

# ---- model choice ----
DEFAULT_SPLIT = "cold_drug"        # realistic estimate for NEW compounds
REFERENCE_SPLIT = "random"         # secondary / reference prediction
ENDPOINTS = ("Ki", "IC50")

# ---- limits ----
MAX_SMILES_LEN = 400
MAX_SEQ_LEN = 5000
MAX_UPLOAD_ROWS = int(os.environ.get("DTI_MAX_UPLOAD_ROWS", 5000))
MAX_UPLOAD_MB = int(os.environ.get("DTI_MAX_UPLOAD_MB", 5))
MAX_EXPLAIN_ATOMS = 120
MAX_COMPARE_DRUGS = 4
MAX_SELECTIVITY_TARGETS = 8
SCREEN_THUMBNAILS = 50             # 2D structures drawn for the top rows of a screen
APPLICABILITY_ROWS = 200           # similarity check is computed for at most this many top rows

# ---- potency labels on pAffinity (= 9 - log10 nM) ----
STRONG_MIN = 8.0                   # >= 8  ->  <= 10 nM
MODERATE_MIN = 6.0                 # 6-8   ->  10 nM - 1 uM ; below 6 -> weak (> 1 uM)
MODELS_DISAGREE = 1.0              # |primary - reference| above this is flagged

P_MIN, P_MAX = 2.0, 12.0

# ---- applicability domain (max Tanimoto to the training drugs) ----
SIM_HIGH, SIM_MID = 0.5, 0.3
_calib = RESULTS_DIR / "applicability.json"                   # written by scripts/calibrate_applicability.py
if _calib.exists():
    try:
        _c = json.loads(_calib.read_text())
        SIM_HIGH, SIM_MID = float(_c["sim_high"]), float(_c["sim_mid"])
    except Exception:                                         # keep defaults if the file is malformed
        pass

# ---- docking (optional) ----
DOCK_DIR = CACHE_DIR / "docking"
DOCK_MAX_BOX = 40
DOCK_MAX_EXHAUSTIVENESS = 16
DOCK_DEFAULT_BOX = 24
DOCK_DEFAULT_EXHAUSTIVENESS = 8

# ---- server ----
HOST = os.environ.get("DTI_HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", os.environ.get("DTI_PORT", 5000)))
JOB_WORKERS = int(os.environ.get("DTI_JOB_WORKERS", 2))
