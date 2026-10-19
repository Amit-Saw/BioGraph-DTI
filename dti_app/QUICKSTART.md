# Quickstart Guide — DTI Binding Affinity Prediction App

This project is fully built and ready to run. Pre-trained checkpoints, reference datasets, chemistry modules, and offline ESM-2 embeddings are configured and tested.

---

## 🚀 1. Quick Launch (In 1 Step)

From the project root:
- **Windows:** Double-click or run `run.bat`
  *(Or inside `dti_app/`: run `run_dev.bat` or `python app.py`)*
- **Linux / macOS:**
  ```bash
  cd dti_app
  .venv/Scripts/activate
  python app.py
  ```

Open your browser at: **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 🧪 2. Verify Everything Works (100% Endpoints & ML Models)

To test all 25 web pages, API endpoints, PyTorch GNN models, atom explanations, and background jobs:

- **Windows:** Run `verify.bat` (or `python scripts/verify_endpoints.py` from `dti_app/`)
- **Unit test suite:** `pytest -v`

All tests and endpoints will run and verify that model weights load correctly, RDKit features compute properly, and batch predictions run successfully.

---

## 🌐 3. What Each Page Does

| Page | URL | Purpose |
|---|---|---|
| **Predict** | `/predict` | Single drug SMILES + protein target: predicts $K_i$ and $\text{IC}_{50}$, potency badges, confidence ranges, 2D structure, 3D conformer, atom occlusion sensitivity, and similar training compounds. |
| **Batch** | `/batch` | Upload a CSV/text file of multiple SMILES against a target protein; scores asynchronously and provides downloadable results. |
| **Screen** | `/screen` | Virtual screening: rank a compound library against a target protein and view top candidates with 2D structures. |
| **Compare** | `/compare` | Compare 2–4 drugs against 1 target, or test 1 drug across up to 8 targets (selectivity profiling). |
| **Dock** | `/dock` | Optional AutoDock Vina 3D binding pose visualization using 3Dmol.js (for targets with PDB entries). |
| **About** | `/about` | View loaded models, test set benchmarks, split methodologies, and scientific limitations. |

---

## 📡 4. JSON API Endpoints

All endpoints return JSON (`{"error": "message"}` on error):

- `GET /api/models` — List loaded models and their test RMSE.
- `GET /api/examples` — Example drugs and target proteins.
- `GET /api/uniprot/<accession>` — Fetch FASTA sequence by UniProt ID (e.g. `P00533`).
- `POST /api/validate` — Validate SMILES and generate 2D SVG preview (`{"smiles": "..."}`).
- `POST /api/predict` — Predict $K_i$ and $\text{IC}_{50}$ (`{"smiles": "...", "sequence": "..."}`).
- `POST /api/properties` — Molecular weight, LogP, Lipinski rule-of-5, formula.
- `POST /api/similar` — Find nearest training compounds by Morgan fingerprint Tanimoto similarity.
- `POST /api/conformer` — Generate 3D conformer molblock for 3Dmol.js.
- `POST /api/explain` — Atom-level occlusion attribution heatmap.
- `POST /api/compare` — Compare multiple drugs against 1 protein sequence.
- `POST /api/selectivity` — Compare 1 drug across multiple target sequences.
- `POST /api/batch` & `POST /api/screen` — Asynchronous screening jobs (`GET /api/jobs/<id>` to poll).
- `GET /api/download/<id>.csv` — Download batch/screening results.

---

## 🛠️ 5. Clean Environment Setup (For New Machines)

If setting up on a new computer from scratch:

1. **Create and activate Python 3.10–3.12 virtual environment:**
   ```bash
   python -m venv .venv
   # Windows: .venv\Scripts\activate
   # Linux/macOS: source .venv/bin/activate
   ```
2. **Install PyTorch:**
   ```bash
   # CPU
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   # or GPU (CUDA 12.1)
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```
3. **Install Requirements:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Check Setup:**
   ```bash
   python scripts/check_setup.py --predict
   ```
