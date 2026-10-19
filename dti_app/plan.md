# plan.md — Build Plan for the DTI Web App (Flask)

This file is the build specification. Every function, the page that uses it, and the API endpoint that connects them are listed here. Build in the order of section 9. A function is "done" when its acceptance check passes.

**Project:** predict drug–target binding affinity (pKi and pIC50) for a (drug SMILES, protein sequence) pair, explain it, and show it.
**Stack:** Flask + PyTorch + PyTorch Geometric + RDKit + ESM-2 (8M) + (optional) AutoDock Vina.
**Models available (already trained, no retraining needed):** MPNN+ESM-2, AttentiveFP+ESM-2, GINE+ESM-2 for Ki and IC50, on `random` and `cold_drug` splits.

---

## 1. Goals and non-goals

**Goals**
1. Predict Ki and IC50 independently for one drug–protein pair, with a potency label and an uncertainty range.
2. Show the drug (2D structure, properties, 3D conformer) and explain the prediction (atom-level occlusion).
3. Batch prediction (CSV) and virtual screening (one protein vs. a library, ranked).
4. Optional docking view (Vina + 3Dmol) for targets with a PDB structure.
5. Show honest model information: metrics, limits, applicability-domain warnings.

**Non-goals**
- No retraining or fine-tuning in the app.
- No user accounts, database server, or cloud deployment in v1 (single-machine Flask app).
- Not a clinical or regulatory tool. Output is a research estimate only.

---

## 2. Architecture and request flow

```
Browser (HTML + JS, 3Dmol.js)
   │  fetch() JSON
   ▼
routes/api.py ──► services/* ──► dti/registry.py ──► dti/encoders.py + checkpoints/*.pt
   │                  │                 │
   │                  │                 └─► dti/protein.py (ESM-2 8M, cached embeddings)
   │                  ├─► services/chem.py     (RDKit: drawing, properties, similarity)
   │                  ├─► services/explain.py  (occlusion)
   │                  ├─► services/docking.py  (background job: Vina)
   │                  └─► services/screening.py (background job: batch scoring)
routes/pages.py ──► templates/*.html
```

Rules:
- **Models and ESM-2 are loaded once at startup** (`create_app()`), never per request.
- All torch inference goes through one lock (`registry.infer_lock`), because a single-process Flask app can have several request threads.
- Long tasks (docking, screening over ~200 rows) run as **background jobs**: the POST returns `job_id`, the page polls `GET /api/jobs/<job_id>`.
- `dti/` code is a **verbatim copy** of the training code (`dti_pipeline_v3.py`). Do not rewrite the featurizer or models; a one-feature difference loads wrong weights silently.

---

## 3. Directory layout

```
dti_app/
├── app.py                 create_app(), error handlers, startup model loading
├── config.py              constants (see section 4)
├── requirements.txt
├── README.md   plan.md
├── dti/
│   ├── featurize.py       mol_graph, smiles_to_data
│   ├── encoders.py        MPNNEncoder, GINEEncoder, AttentiveFPEncoder, DTIModel
│   ├── protein.py         clean_sequence, embed_sequence, EmbeddingCache
│   └── registry.py        ModelRegistry
├── services/
│   ├── labels.py          to_nM, potency_label, uncertainty_band
│   ├── chem.py            drawing, descriptors, similarity, applicability
│   ├── explain.py         occlusion (+ optional gradient) attributions
│   ├── uniprot.py         fetch_sequence
│   ├── screening.py       parse_library, run_screen
│   ├── docking.py         dock_job, pose analysis
│   └── jobs.py            JobManager (thread pool + status dict)
├── routes/
│   ├── pages.py           HTML pages
│   └── api.py             JSON endpoints
├── templates/             base, index, predict, batch, screen, dock, compare, about
├── static/                css/, js/ (predict.js, batch.js, screen.js, dock.js, compare.js), img/
├── checkpoints/           <endpoint>_<split>/<model>/{best.pt,config.json}  (+ esm2_t6_8M/ optional)
├── data/
│   ├── reference/         Ki_train.csv, IC50_train.csv   (train split of cold_drug; for similarity search)
│   ├── examples/          drugs.csv, targets.csv
│   └── results/           comparison_table.csv, predicted_vs_actual_*.csv, plots
├── cache/                 protein_embeddings.pkl, docking/<job_id>/
├── uploads/
└── tests/                 test_featurize.py, test_registry.py, test_api.py, test_explain.py
```

Checkpoint source (copy from Drive `13_Models/`): only `best.pt` and `config.json` per model. Do **not** copy `latest.pt` (contains optimizer state, much larger, not needed).

---

## 4. Configuration (`config.py`)

| Constant | Value | Used by |
|---|---|---|
| `CHECKPOINT_DIR` | `checkpoints/` | registry |
| `DEFAULT_SPLIT` | `"cold_drug"` | registry, predict |
| `DEFAULT_MODEL` | `{"Ki": "attentivefp_esm", "IC50": "attentivefp_esm"}` (cold_drug best by val RMSE) | predict |
| `REFERENCE_MODEL` | `{"Ki": ("random", "mpnn_esm"), "IC50": ("random", "attentivefp_esm")}` | predict (second column) |
| `P_MIN, P_MAX` | 2.0, 12.0 (same as training) | validation of outputs |
| `POTENCY_BANDS` | strong ≥ 8.0 (≤ 10 nM); moderate 6.0–8.0; weak < 6.0 (> 1 µM) | labels |
| `BAND_SIGMA` | per model: its **test RMSE on its own split** (read from `test_metrics.json`) | uncertainty |
| `SIM_HIGH, SIM_MID` | 0.5, 0.3 (Tanimoto, Morgan r=2, 2048 bits) — to be **calibrated** in step 6.3 | applicability |
| `MAX_SMILES_LEN` | 400 | validation |
| `MAX_SEQ_LEN` | 5000 | validation |
| `MAX_UPLOAD_ROWS` | 5000 | batch, screening |
| `MAX_UPLOAD_MB` | 5 | Flask `MAX_CONTENT_LENGTH` |
| `DOCK_TIMEOUT_S` | 600 | docking |
| `EXHAUSTIVENESS` | 8 (fast) / 16 (accurate) | docking |

---

## 5. Module specification

### 5.1 `dti/featurize.py`

| Function | Input → Output | What it does | Used by |
|---|---|---|---|
| `validate_smiles(smiles)` | str → `(ok: bool, message: str, canonical: str|None)` | RDKit parse, length check, reject empty/multi-fragment-only. Returns canonical SMILES | every route that takes a drug |
| `mol_graph(smiles)` | str → `(x[N,38], edge_index[2,E], edge_attr[E,7])` | **Copy of training function** (atom: element, degree, charge, hybridization, H count, aromatic, ring = 38; bond: type, conjugated, ring = 7). LRU cached | `smiles_to_data` |
| `smiles_to_data(smiles, protein_emb)` | str, ndarray[320] → torch_geometric `Data` | builds graph + attaches `protein_emb` as shape (1,320) | registry |
| `data_without_atom(data, atom_idx)` | Data, int → Data | deletes one node and its edges (for occlusion) | explain |

**Acceptance:** `mol_graph` output equals the training function on 100 random training SMILES (tensor equality).

### 5.2 `dti/encoders.py`
Copy `MPNNEncoder`, `GINEEncoder`, `AttentiveFPEncoder`, `DTIModel`, and the constants `ATOM_DIM=38, BOND_DIM=7, PROT_DIM=320, HIDDEN=128, DROPOUT=0.2` unchanged. **Acceptance:** `DTIModel(name).load_state_dict(torch.load(best.pt))` succeeds with `strict=True` for all four checkpoints.

### 5.3 `dti/protein.py`

| Function | Input → Output | What it does | Used by |
|---|---|---|---|
| `clean_sequence(seq)` | str → str | uppercase, strip whitespace/FASTA header, reject non-amino-acid letters, length ≤ `MAX_SEQ_LEN` | routes |
| `load_esm()` | → (tokenizer, model) | loads `facebook/esm2_t6_8M_UR50D` (or local folder) once, `.eval()` | startup |
| `embed_sequence(seq)` | str → ndarray[320] | **Copy of training logic**: windows 900 / stride 800, mean over residues excluding `<cls>`/`<eos>`, average across windows | `get_embedding` |
| `get_embedding(seq)` | str → ndarray[320] | looks in `EmbeddingCache` (pickle in `cache/`), otherwise computes and stores | registry, screening |

**Acceptance:** for 5 sequences from the training embedding pickle, `embed_sequence` matches stored vectors (`allclose`, atol 1e-4).

### 5.4 `dti/registry.py` — `ModelRegistry`

| Method | What it does | Used by |
|---|---|---|
| `load_all()` | scans `checkpoints/*/*/`, builds each `DTIModel`, loads `best.pt`, reads `config.json` + test RMSE, `.eval()`, CPU/GPU | startup |
| `available()` | → list of `{endpoint, split, model, test_rmse}` | `/api/models`, About, dropdowns |
| `predict_pair(endpoint, split, model, smiles, seq)` | → float pAffinity | predict, compare |
| `predict_many(endpoint, split, model, smiles_list, seq, batch=256)` | embeds protein once, batches molecules with PyG `DataLoader`, returns array (NaN for invalid SMILES) | batch, screening |
| `predict_both(endpoint, smiles, seq)` | runs the default (cold_drug) and reference (random) models | `/api/predict` |
| `model_graph_forward(...)` | helper used by explain: forward on a prepared `Data` | explain |

All calls hold `infer_lock` and use `torch.inference_mode()`. **Acceptance (`tests/test_registry.py`):** run the first 200 rows of each `predicted_vs_actual`/test set through the registry; predictions match saved `test_predictions.csv` within 1e-3.

### 5.5 `services/labels.py`

| Function | What it does |
|---|---|
| `to_nM(p)` | `10 ** (9 - p)` (inverse of pAffinity = 9 − log10 nM) |
| `format_conc(nM)` | "12 nM", "3.4 µM", "1.2 mM" |
| `potency_label(p)` | "Strong" / "Moderate" / "Weak" per `POTENCY_BANDS`, with a colour key |
| `uncertainty_band(p, sigma)` | `(low_nM, high_nM)` from p ± sigma (≈ 68% interval if errors are roughly Gaussian) |
| `agreement(p1, p2)` | `abs(p1-p2)`; flag "models disagree" if > 1.0 |

### 5.6 `services/chem.py`

| Function | What it does | Used by |
|---|---|---|
| `draw_svg(smiles, highlight=None, size=(450,350))` | RDKit `rdMolDraw2D` SVG of the 2D structure; optional atom colour map | predict, explain, compare |
| `descriptors(smiles)` | MW, cLogP, HBD, HBA, TPSA, rotatable bonds, rings, formula | predict |
| `lipinski(smiles)` | rule-of-5 violations list | predict |
| `conformer_molblock(smiles)` | AddHs → ETKDG embed → MMFF optimise → molblock (for 3Dmol) | predict 3D tab |
| `nearest_neighbours(smiles, endpoint, k=5)` | Tanimoto vs `data/reference/<endpoint>_train.csv` → table of SMILES, similarity, protein id, experimental pAffinity | predict "similar known compounds" |
| `applicability(smiles, endpoint)` | max Tanimoto → `"high"/"medium"/"low"` | predict, batch, screening |

Precompute and cache the reference fingerprints at startup (numpy bit arrays) so a lookup is fast.

### 5.7 `services/explain.py` — atom-level explanation (no retraining)

| Function | What it does |
|---|---|
| `occlusion_scores(endpoint, split, model, smiles, seq)` | baseline prediction `p0`; for each atom `i` build `data_without_atom(i)`, predict in one batch → `score_i = p0 − p_i` (positive = atom increases predicted affinity) |
| `gradient_scores(...)` (optional cross-check) | gradient of prediction w.r.t. atom features, summed per atom (`x.requires_grad_()`) |
| `scores_to_colormap(scores)` | diverging colours (positive = one hue, negative = another), normalised by max abs |
| `explain(...)` | returns `{scores:[...], svg: draw_svg(highlight=...)}` |

Notes: removing an atom can create a disconnected/unphysical fragment, so present results as "model sensitivity", not chemical truth. Cap at 120 heavy atoms. BatchNorm models (GINE) must be in `eval()`.
**Acceptance:** scores sum/shape sanity (length = number of atoms), deterministic across two calls, and for a simple known molecule (e.g. erlotinib vs EGFR) the highlighted atoms are visible in the SVG.

### 5.8 `services/uniprot.py`
`fetch_sequence(accession)` → GET `https://rest.uniprot.org/uniprotkb/<id>.fasta`, parse FASTA, return `(name, sequence)`; timeout 10 s; clear error when offline/not found. Cache results in `cache/uniprot.json`.

### 5.9 `services/jobs.py` — `JobManager`
`submit(fn, *args) -> job_id`, `status(job_id) -> {state: queued|running|done|error, progress: 0-1, result|error}`, a `ThreadPoolExecutor(max_workers=2)`. Jobs are in memory (lost on restart); results larger than a few MB are written to `cache/jobs/<id>.json`.

### 5.10 `services/screening.py`

| Function | What it does |
|---|---|
| `parse_library(file_storage)` | reads CSV (column `smiles`, optional `name`) or SMILES-per-line; validates each; drops invalid with a reason list; enforces `MAX_UPLOAD_ROWS` |
| `run_screen(job, endpoint, seq, smiles_list, names, top_n)` | embeds protein once; scores in batches (updates `job.progress`); adds label, nM, applicability; sorts descending pAffinity |
| `screen_to_csv(result)` | CSV download |

### 5.11 `services/docking.py` (optional module — app still runs if Vina is not installed)

| Function | What it does |
|---|---|
| `docking_available()` | checks `import vina` and `obabel` on PATH; used to enable/disable the tab |
| `fetch_pdb(pdb_id)` | downloads `https://files.rcsb.org/download/<ID>.pdb` |
| `list_ligands(pdb_id)` | lists HETATM residues (code, chain, atom count) so the user can pick the pocket ligand |
| `prepare_receptor(pdb_id, chains, het)` | protein-only PDB → PDBQT (`obabel -xr -p 7.4`); pocket centre = centroid of chosen ligand |
| `prepare_ligand(smiles)` | RDKit embed + Meeko PDBQT (for user SMILES; the earlier demo used the PDB ideal ligand) |
| `dock_job(job, pdb_id, chains, het, smiles, box, exhaustiveness)` | Vina dock, return best pose (PDBQT), score (kcal/mol), 4 Å contact residues, protein PDB text |
| `pose_contacts(protein_pdb, pose)` | residues within 4 Å of the pose |

Output shown in the dock tab with 3Dmol: protein cartoon, contact residues as sticks, docked pose green. Always display "Vina score is not pKi/pIC50".

---

## 6. API specification (`routes/api.py`)

All JSON. Errors: `{"error": "message"}` with HTTP 400 (bad input), 404, 413 (upload too big), 500.

| Method + path | Request | Response | Calls |
|---|---|---|---|
| `GET /api/models` | – | list of available models with test metrics | `registry.available` |
| `POST /api/validate` | `{smiles}` | `{ok, canonical, svg, message}` (live preview while typing) | `validate_smiles`, `draw_svg` |
| `GET /api/uniprot/<acc>` | – | `{name, sequence, length}` | `fetch_sequence` |
| `POST /api/predict` | `{smiles, sequence, endpoints:["Ki","IC50"]}` | per endpoint: primary + reference predictions, label, nM, band, applicability, agreement | `predict_both`, `labels.*`, `chem.applicability` |
| `POST /api/properties` | `{smiles}` | descriptors, Lipinski, SVG, `nearest_neighbours` per endpoint | `chem.*` |
| `POST /api/conformer` | `{smiles}` | `{molblock}` | `conformer_molblock` |
| `POST /api/explain` | `{smiles, sequence, endpoint, split?, model?}` | `{scores, svg, p0}` | `explain.explain` |
| `POST /api/batch` | multipart: CSV + `sequence` + endpoints | `{job_id}` (or direct result if ≤ 100 rows) | `screening.parse_library`, `run_screen` |
| `POST /api/screen` | multipart: library + `sequence` + endpoint + `top_n` | `{job_id}` | `screening.run_screen` |
| `POST /api/compare` | `{smiles_list(2..4), sequence}` | table of predictions for both endpoints | `predict_many` |
| `POST /api/selectivity` | `{smiles, sequences:[{name, seq}...]}` | predictions of one drug vs several targets | `predict_many` (loop over proteins) |
| `POST /api/dock` | `{pdb_id, chains, het, smiles?, box?}` | `{job_id}` | `docking.dock_job` |
| `GET /api/pdb/<id>/ligands` | – | ligand list | `docking.list_ligands` |
| `GET /api/jobs/<job_id>` | – | `{state, progress, result?}` | `JobManager.status` |
| `GET /api/download/<job_id>.csv` | – | CSV file | `screen_to_csv` |

### Example: `POST /api/predict` response
```json
{
  "smiles": "COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC",
  "results": {
    "Ki": {
      "primary":   {"model": "attentivefp_esm", "split": "cold_drug", "pAffinity": 7.84,
                    "nM": 14.5, "label": "Moderate", "band_nM": [1.4, 150.0], "sigma": 0.98},
      "reference": {"model": "mpnn_esm", "split": "random", "pAffinity": 8.11, "nM": 7.8},
      "agreement_abs": 0.27,
      "applicability": {"max_tanimoto": 0.62, "level": "high"}
    },
    "IC50": { "...": "same structure" }
  }
}
```

---

## 7. Pages and what each does

| Page (route) | Purpose | Components | API calls |
|---|---|---|---|
| **Home** `/` | intro, what the tool does, 3 example buttons | hero, example cards (load example drug+target) | `/api/models` |
| **Predict** `/predict` | main page: one drug + one target | SMILES box with live 2D preview; target input (paste sequence / UniProt ID / example list); Ki and IC50 result cards (value, nM, label badge, uncertainty bar, reference-model value, agreement warning); tabs: **2D & properties**, **3D conformer**, **Explain (atoms)**, **Similar known compounds**; applicability warning banner | `/api/validate`, `/api/uniprot`, `/api/predict`, `/api/properties`, `/api/conformer`, `/api/explain` |
| **Batch** `/batch` | many drugs, one target | CSV upload, target input, progress bar, sortable results table (name, pAffinity, nM, label, applicability), download CSV, histogram of predictions | `/api/batch`, `/api/jobs/<id>`, download |
| **Screen** `/screen` | virtual screening: rank library against one target | upload, endpoint choice, top-N, progress, ranked table with 2D thumbnails, click row → open in Predict, download | `/api/screen`, `/api/jobs/<id>` |
| **Compare** `/compare` | 2–4 drugs vs a target, **and** one drug vs several targets (selectivity) | two modes, bar chart (pKi/pIC50), 2D structures side by side | `/api/compare`, `/api/selectivity` |
| **Dock** `/dock` | 3D binding pose for targets with a PDB entry | PDB ID input → ligand list → pick chain/ligand → run → 3Dmol view (protein cartoon, contacts, pose), Vina score, contact list, warning text | `/api/pdb/.../ligands`, `/api/dock`, `/api/jobs/<id>` |
| **About** `/about` | trust and transparency | metrics table (from `data/results/comparison_table.csv`), predicted-vs-actual plots, split explanation, limitations, citations | `/api/models` |

Shared UI rules: every result shows pAffinity **and** nM; every number that comes from a cold_drug model says "for new compounds"; low applicability shows an amber banner ("this molecule is far from the training data; treat the value as rough").

---

## 8. Data contracts and behaviours to get right

1. **Same featurization as training** (section 5.1) — verified by test.
2. **Endpoints are independent**: Ki results never use IC50 models and vice versa.
3. **Uncertainty band** uses the model's own test RMSE from `test_metrics.json` (e.g. 0.98 for Ki AttentiveFP cold_drug); label it "typical error", not a guarantee.
4. **Applicability thresholds** are heuristic until calibrated (step 6.3).
5. **Invalid input** never crashes the server: validation returns a message; batch rows with bad SMILES are listed and skipped.
6. **Sequence characters:** only the 20 standard amino-acid letters (plus `X`, `U`, `B`, `Z`, `O` mapped as ESM expects); anything else → 400.
7. **Docking jobs** are sandboxed to `cache/docking/<job_id>/` and time out after `DOCK_TIMEOUT_S`.

---

## 9. Build order (milestones with acceptance checks)

| # | Milestone | Deliverable | Acceptance |
|---|---|---|---|
| 1 | Skeleton | folders, `app.py`, `config.py`, base template, `/` renders | `flask run` shows the home page |
| 2 | Model package | `dti/featurize.py`, `encoders.py`, `protein.py` (copied from v3) | graph + embedding equality tests pass |
| 3 | Registry | `dti/registry.py` + checkpoints in place | `test_registry.py` reproduces saved test predictions (1e-3) |
| 4 | Predict API | `labels.py`, `/api/predict`, `/api/validate`, `/api/uniprot` | curl returns the JSON in section 6 for erlotinib + EGFR |
| 5 | Predict page | `predict.html` + `predict.js` (inputs, result cards, label badges) | working in browser, error messages for bad SMILES |
| 6 | Chemistry features | `chem.py`: 2D SVG, properties, Lipinski, 3D conformer, nearest neighbours, applicability | 2D/3D tabs work; 6.3 below |
| 6.3 | Calibration | on the cold_drug **validation** set, bin absolute error by max-Tanimoto and set `SIM_HIGH/SIM_MID` from the observed error | table of error vs similarity saved in `data/results/` and quoted on About |
| 7 | Explain | `explain.py` + Explain tab | scores length = atoms; highlighted SVG appears |
| 8 | Jobs + Batch | `jobs.py`, `screening.parse_library`, Batch page | 1000-row CSV completes with progress; bad rows reported |
| 9 | Screening + Compare | `/api/screen`, `/api/compare`, `/api/selectivity`, pages | top-N table + bar charts render |
| 10 | Docking (optional) | `docking.py`, Dock page | 1M17/AQ4 and 1HSG/MK1 produce poses overlapping the crystal ligand |
| 11 | About + polish | About page, responsive layout, loading states | all pages usable on a phone-width screen |
| 12 | Hardening | tests, input limits, error handler, README run steps verified on a clean machine | `pytest` green; fresh venv install works |

---

## 10. Testing plan (`tests/`)

- `test_featurize.py`: graph equality with training function; invalid SMILES handling; single-atom molecule.
- `test_registry.py`: reproduces `test_predictions.csv` for each loaded checkpoint (first 200 rows).
- `test_api.py` (Flask test client): every endpoint — happy path, bad input, oversize upload.
- `test_explain.py`: shape, determinism, atom-cap behaviour.
- Manual: three drugs (erlotinib/EGFR, indinavir/HIV protease, an out-of-domain odd molecule) through the full UI.

---

## 11. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Feature mismatch between app and training | verbatim copy + equality tests (milestones 2, 3) |
| Users over-trust a single number | uncertainty band, applicability banner, About-page limitations |
| Models weaker on new scaffolds (cold_drug R² 0.4–0.5) | default to cold_drug models; show reference model only as secondary |
| Memory use from several models + ESM-2 | load once; `gunicorn --workers 1 --threads 4` |
| Long jobs block the server | background `JobManager`, upload caps, timeouts |
| Docking dependencies hard to install | optional module with `docking_available()`; tab hidden if missing |
| Occlusion creates odd fragments | wording "model sensitivity"; optional gradient cross-check |
| UniProt/RCSB unreachable | friendly error + allow pasting sequence / uploading PDB manually |
| Protein embedding discards pocket detail | stated in limitations; docking tab complements for known structures |

---

## 12. Definition of done

- All milestones 1–9 and 11–12 pass; 10 (docking) passes or is cleanly disabled.
- `README.md` setup steps work on a clean machine.
- The app reproduces the saved test metrics through its own prediction path (milestone 3).
- Limitations are visible in the UI, not only in the docs.


---

## Implementation notes (as built)

- Similar compounds are served by `POST /api/similar` (not inside `/api/properties`); applicability thresholds are read from `data/results/applicability.json` through `config.py`.
- Background jobs use `JobManager.submit(fn, kind)`; there is no per-job timeout (docking size is bounded by box size and exhaustiveness in `config.py`).
- The docking module is pure python for PDB parsing (no Biopython) and uses Vina, Meeko and Open Babel only when installed.
- Atom explanation uses occlusion only (the optional gradient view was not built).
- Run `QUICKSTART.md` for setup, deployment and troubleshooting.
