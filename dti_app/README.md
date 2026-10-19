# Drug–Target Binding Affinity Prediction (Ki / IC50) with Graph Neural Networks and ESM-2

Predicts how strongly a small molecule binds a protein, as **pKi** and **pIC50**, from the drug's SMILES string and the protein's amino-acid sequence. Includes a Flask web app that shows predictions, labels, 2D/3D drug views, atom-level explanations, batch/virtual screening and an optional docking view.

> **To run the app, follow [`QUICKSTART.md`](QUICKSTART.md).** This README explains the theory, models and results.
>
> Research tool only. Predictions are statistical estimates (typical error about 1 log unit on new drugs). They are not a substitute for experiments and must not be used for clinical decisions.

---

## Contents
1. [Problem and theory](#1-problem-and-theory)
2. [Data](#2-data)
3. [Representations](#3-representations)
4. [Model architectures](#4-model-architectures)
5. [Data splitting](#5-data-splitting)
6. [Training procedure](#6-training-procedure)
7. [Evaluation and results](#7-evaluation-and-results)
8. [Explaining and visualising predictions](#8-explaining-and-visualising-predictions)
9. [Web application](#9-web-application)
10. [Setup and installation](#10-setup-and-installation)
11. [Running the app](#11-running-the-app)
12. [Reproducing the training (Colab)](#12-reproducing-the-training-colab)
13. [Repository structure](#13-repository-structure)
14. [Limitations](#14-limitations)
15. [References](#15-references)

---

## 1. Problem and theory

**Drug–target interaction (DTI) affinity prediction** estimates the strength of binding between a compound (ligand) and a protein target. It is a **regression** problem: input (drug, protein) → one real number.

### Ki and IC50
- **Ki (inhibition constant):** concentration of inhibitor at which half of the protein's binding sites are occupied at equilibrium. It reflects binding strength.
- **IC50 (half-maximal inhibitory concentration):** concentration of inhibitor that reduces the measured activity by 50%. It depends on assay conditions (substrate concentration, enzyme concentration), so it is noisier than Ki.
- Both are measured in **nanomolar (nM)**. **Smaller = stronger**: 1 nM is very potent, 1,000 nM weak, 100,000 nM almost inactive.

Ki and IC50 are **different quantities with different noise**, so they are trained and evaluated as two separate tasks with separate models.

### pAffinity (log transform)
Raw nM values span many orders of magnitude (0.01 nM to millions). Models train poorly on that scale, so values are converted to a negative-log scale:

```
pAffinity = 9 − log10(value in nM)         (called pKi or pIC50)
value_nM  = 10 ^ (9 − pAffinity)
```

| nM | pAffinity | Meaning |
|---|---|---|
| 1 | 9 | very strong |
| 10 | 8 | strong |
| 100 | 7 | good |
| 1,000 | 6 | moderate |
| 10,000 | 5 | weak |

A difference of 1.0 in pAffinity is a 10-fold difference in concentration. The app labels predictions as **Strong** (≥ 8, i.e. ≤ 10 nM), **Moderate** (6–8) and **Weak** (< 6, i.e. > 1 µM).

---

## 2. Data

- **Source:** BindingDB, split into a Ki dataset and an IC50 dataset (`BindingDB_Ki_final.csv`, `BindingDB_IC50_final.csv`; columns `canonical_smiles`, `protein_sequence`, `label_nM`). These files come from the earlier cleaning stage (see `deduplication_summary.csv` and the cleaning notebooks).
- **Steps performed by the training pipeline (`dti_pipeline_v3.py`):**
  1. Drop rows with empty SMILES/sequence or a non-positive / non-numeric `label_nM`.
  2. Convert to pAffinity (`9 − log10(nM)`).
  3. Keep only plausible values: **2.0 ≤ pAffinity ≤ 12.0**.
  4. **Aggregate duplicate drug–protein pairs by the median** (`n_obs` records how many measurements were merged). One row per pair avoids the same pair appearing in train and test with slightly different labels.
  5. Encode drugs and proteins as integer ids for fast splitting, then write `train/validation/test.csv` per endpoint and split mode.

---

## 3. Representations

### 3.1 Drug → molecular graph
A molecule is a graph: atoms are nodes, bonds are edges. RDKit parses the SMILES, and each atom/bond gets a feature vector (one-hot encodings, with an "other" bin):

| | Features | Size |
|---|---|---|
| **Atom** | element (H, C, N, O, F, P, S, Cl, Br, I), degree (0–5), formal charge (−2…+2), hybridization (SP, SP2, SP3, SP3D, SP3D2), number of H (0–4), aromatic, in ring | **38** |
| **Bond** | type (single, double, triple, aromatic), conjugated, in ring | **7** |

Bonds are stored in both directions. A single-atom molecule gets a zero-feature self-loop.

### 3.2 Protein → ESM-2 embedding
- Model: **ESM-2, 8M parameters** (`facebook/esm2_t6_8M_UR50D`), a protein language model pre-trained on millions of sequences.
- Each sequence is split into overlapping windows (**window 900, stride 800**, max length 1024 tokens) so long proteins fit.
- The hidden states of the real residues are **mean-pooled** (the `<cls>` and `<eos>` tokens are excluded) and averaged across windows → one **320-dimensional vector per protein**.
- Embeddings are computed once, cached in a pickle and reused. ESM-2 is **frozen** (not fine-tuned).

---

## 4. Model architectures

All models share one design: a **two-tower, late-fusion** network.

```
SMILES ──► molecular graph ──► [GNN encoder] ──► 128-d drug vector ─┐
                                                                     ├─ concat (256) ─► MLP head ─► pAffinity
sequence ─► ESM-2 (frozen) ─► 320-d ─► MLP (320→256→128) ───────────┘
```

- **Protein tower:** LayerNorm → Linear(320, 256) → ReLU → Dropout(0.2) → Linear(256, 128) → ReLU.
- **Head:** Linear(256, 256) → ReLU → Dropout → Linear(256, 64) → ReLU → Linear(64, 1).
- Only the GNN encoder changes between the compared models:

| Model | Encoder | Idea |
|---|---|---|
| **MPNN + ESM-2** (primary) | Gilmer-style message passing: `NNConv` (edge-conditioned messages; the bond features produce a weight matrix per bond), shared over 3 steps, **GRU** state update, **Set2Set** readout; 64-d atoms → 128-d output | Messages depend on bond type; recurrent update; order-invariant readout |
| **GINE + ESM-2** (latest) | 3 × `GINEConv` (GIN with edge features) with residual connections and BatchNorm, mean pooling; 128-d | Maximally expressive sum-aggregation GNN with bond features |
| **AttentiveFP + ESM-2** (comparison) | `AttentiveFP`: 3 attention layers over atoms/bonds + 2 attentive read-out steps; 128-d | Attention lets the model weigh neighbours and focus on important substructures |

Regression is plain **MSE loss** on pAffinity.

---

## 5. Data splitting

How the data is split decides what the score means. Splits are done by **groups**, and an assertion checks that no group appears in two subsets ("group leakage" check). Ratio 80 / 10 / 10.

| Split | Group | Question answered |
|---|---|---|
| **random** | identical drug–protein pair | How well does the model do when similar drugs/proteins were seen in training? (optimistic) |
| **cold_drug** | **Murcko scaffold** of the drug (core ring system) | How well does it predict **new chemical scaffolds**? (realistic for new compounds) |
| **cold_protein** | protein sequence | How well does it predict **new targets**? (implemented, not run yet) |

Take the cold_drug numbers as the honest estimate for new drugs; random-split numbers are higher because related molecules appear in both train and test.

---

## 6. Training procedure

| Setting | Value |
|---|---|
| Optimizer | AdamW, lr 1e-3, weight decay 1e-5 |
| Scheduler | ReduceLROnPlateau on validation MAE (factor 0.5, patience 3) |
| Batch size | 128 |
| Gradient clipping | max-norm 5.0 |
| Dropout | 0.2 |
| **"Epoch"** | a **random chunk of 60,000 training rows** (new rows each epoch) instead of the full set, for speed; validation on a fixed 20,000-row subset |
| Early stopping | patience 8 epochs; an epoch counts as improvement only if validation MAE improves by ≥ 0.002 |
| Max epochs | 50 initially; best models per split continued to 120 epochs |
| Seed | 42 |

Every model has its own folder `13_Models/<endpoint>_<split>/<model>/` with `best.pt` (best-validation weights), `latest.pt` (full state to resume after a Colab disconnect), `history.json`, `config.json`, `test_metrics.json`, `test_predictions.csv`.

**Model selection** uses **validation** RMSE, so the test set is never used to choose a model. Final scores are computed on the **full** validation and test sets with the best-validation weights.

---

## 7. Evaluation and results

Metrics on the held-out test set (pAffinity units):
- **MSE / RMSE** — average squared / root error; penalises large mistakes (lower is better).
- **MAE** — average absolute error (lower is better).
- **R²** — fraction of variance explained (1 = perfect, 0 = predicts the mean).
- **Pearson / Spearman** — linear / rank correlation between predicted and actual (higher is better; Spearman tells how well the model *ranks* compounds).
- **Within 1 log unit** — share of predictions within a factor of 10 of the true value.

### Ki
| Split | Model | Epochs | RMSE | MAE | R² | Pearson | Spearman |
|---|---|---|---|---|---|---|---|
| cold_drug | AttentiveFP | 111 | **0.980** | 0.755 | 0.502 | 0.710 | 0.681 |
| cold_drug | MPNN | 50 | 1.097 | 0.858 | 0.376 | 0.618 | 0.581 |
| cold_drug | GINE | 23 | 1.192 | 0.952 | 0.263 | 0.549 | 0.501 |
| random | MPNN | 97 | **0.919** | 0.709 | 0.600 | 0.775 | 0.753 |
| random | AttentiveFP | 62 | 0.979 | 0.765 | 0.546 | 0.741 | 0.715 |
| random | GINE | 34 | 1.170 | 0.929 | 0.351 | 0.617 | 0.580 |

### IC50
| Split | Model | Epochs | RMSE | MAE | R² | Pearson | Spearman |
|---|---|---|---|---|---|---|---|
| cold_drug | AttentiveFP | 54 | **1.053** | 0.821 | 0.419 | 0.649 | 0.646 |
| cold_drug | MPNN | 30 | 1.117 | 0.883 | 0.345 | 0.589 | 0.586 |
| random | AttentiveFP | 88 | **0.950** | 0.738 | 0.528 | 0.728 | 0.724 |
| random | MPNN | 36 | 1.046 | 0.826 | 0.428 | 0.655 | 0.650 |

(Test sizes: Ki ≈ 25.3k pairs per split, IC50 ≈ 50.4k.)

### How to read these
- **random vs cold_drug gap** (e.g. Ki R² 0.60 vs 0.50) is the "leakage effect": the model does better when it has seen similar drugs.
- **AttentiveFP** is the most robust on new scaffolds; **MPNN** is best on the random Ki split.
- **GINE** stopped early (23–34 epochs) and looks under-trained; BatchNorm with lr 1e-3 is the suspected cause (not yet confirmed). Treat its numbers as provisional. A LayerNorm variant trained in a new folder is the planned fix.
- Models were run for **different numbers of epochs** because of early stopping, so the comparison is not perfectly equal-budget.
- **Published reference points** (different splits, not directly comparable): kNN-DTA reports RMSE ≈ 0.75 (Ki) and ≈ 0.68 (IC50) on BindingDB; DeepDTA-style models report IC50 RMSE ≈ 0.78. Our random-split results (0.92 / 0.95) are in the same region but not better; treat comparisons cautiously.
- **Check before publishing:** BindingDB contains many censored values (e.g. "> 10,000 nM") that become a spike at pAffinity = 5.0. Inspect the label histogram and report metrics with and without that spike.

---

## 8. Explaining and visualising predictions

All of these use the trained models as they are (no retraining).

1. **Atom-level explanation (occlusion).** For every atom, remove it from the graph and re-predict; the change in predicted pAffinity is that atom's score (positive = the atom raises predicted affinity). Drawn as a colour map on the 2D structure. It shows what the *model* is sensitive to, not proven chemistry.
2. **Applicability domain.** The highest Tanimoto similarity (Morgan radius 2, 2048 bits) between the query drug and the training drugs. Low similarity → the prediction is an extrapolation and the app warns the user. Thresholds are calibrated on validation error vs similarity.
3. **Uncertainty band.** The model's test RMSE is turned into an approximate range (e.g. "14 nM, typical range 1.4–150 nM").
4. **Similar known compounds.** Nearest training molecules with their measured values, as a sanity check.
5. **Docking (optional).** AutoDock Vina places the drug in a protein structure from the PDB; the pose is drawn in 3D with 3Dmol. The Vina score (kcal/mol) is a separate physics-based estimate and is **not** comparable to pKi/pIC50.

---

## 9. Web application

Flask app with these pages: **Predict**, **Batch**, **Screen** (virtual screening), **Compare / Selectivity**, **Dock**, **About**. The full function-by-function and endpoint-by-endpoint design is in **[`plan.md`](plan.md)**.

Where the models live: `checkpoints/<endpoint>_<split>/<model>/{best.pt, config.json}` — e.g. `checkpoints/Ki_cold_drug/attentivefp_esm/best.pt`.
Default behaviour: the **cold_drug** model is the main prediction (it is the realistic one for new compounds); the best **random**-split model is shown as a reference.

---

## 10. Setup and installation

### 10.1 Requirements
- Python **3.10 or 3.11** (3.12 also works if all wheels are available)
- ~3 GB free disk (PyTorch, ESM-2, checkpoints), 4 GB+ RAM
- GPU is optional; CPU is fine for single predictions and small screens
- Internet on first run (downloads ESM-2 weights, ~30 MB; UniProt/RCSB lookups)

### 10.2 Create an environment
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
```

### 10.3 Install PyTorch (choose ONE)
```bash
# CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu

# NVIDIA GPU (CUDA 12.1 example; pick yours at https://pytorch.org/get-started/locally/)
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 10.4 Install the rest
```bash
pip install torch_geometric            # GINEConv, NNConv, Set2Set, AttentiveFP need no extra compiled wheels
pip install rdkit                      # chemistry (pip wheel works on Windows/Linux/macOS)
pip install transformers               # ESM-2
pip install flask gunicorn             # web app (gunicorn on Linux/macOS only)
pip install numpy pandas scipy scikit-learn requests
pip install pytest                     # tests
```

### 10.5 Optional: docking tab
```bash
pip install vina meeko py3Dmol biopython
# Open Babel (needed to convert the protein to PDBQT):
sudo apt-get install -y openbabel      # Debian/Ubuntu
# or: conda install -c conda-forge openbabel
# or: pip install openbabel-wheel
```
If these are missing, the app still runs; the Dock tab shows "not available".

### 10.6 `requirements.txt`
```
torch
torch_geometric
rdkit
transformers
flask
gunicorn; platform_system != "Windows"
numpy
pandas
scipy
scikit-learn
requests
pytest
# optional docking
vina
meeko
biopython
py3Dmol
```
Reproducibility tip: record the exact versions used for training and pin them (`pip freeze`), especially `torch`, `torch_geometric` and `rdkit`:
```bash
python -c "import torch, torch_geometric, rdkit, transformers; print(torch.__version__, torch_geometric.__version__, rdkit.__version__, transformers.__version__)"
```

### 10.7 Put the model files in place
Copy from Google Drive `13_Models/` into `checkpoints/` (keep the folder names), **only `best.pt` and `config.json`**:
```
checkpoints/Ki_cold_drug/attentivefp_esm/{best.pt,config.json}
checkpoints/Ki_random/mpnn_esm/{best.pt,config.json}
checkpoints/IC50_cold_drug/attentivefp_esm/{best.pt,config.json}
checkpoints/IC50_random/attentivefp_esm/{best.pt,config.json}
```
Reference data for the similar-compound search: copy `10_Splits_v3/Ki_cold_drug/train.csv` → `data/reference/Ki_train.csv` and the IC50 equivalent. Copy `comparison_table.csv` and `predicted_vs_actual.csv` files to `data/results/` for the About page.

Optional offline ESM-2: download `facebook/esm2_t6_8M_UR50D` once and save it to `checkpoints/esm2_t6_8M/`.

---

## 11. Running the app

Development:
```bash
python app.py                     # or: flask --app app run --debug
# open http://127.0.0.1:5000
```
Production-style (Linux/macOS), **one worker** so the models are loaded once:
```bash
gunicorn -w 1 --threads 4 -b 0.0.0.0:8000 "app:create_app()"
```
Tests:
```bash
pytest -q
```
The first test (`test_registry.py`) checks that the app reproduces the saved test predictions, which confirms that code and checkpoints match.

---

## 12. Reproducing the training (Colab)

1. Open a Colab notebook with a GPU runtime; mount Google Drive.
2. `!pip install -q rdkit transformers torch-geometric scikit-learn scipy`
3. Run the cells of `dti_pipeline_v3.py` in order: setup → splits → embeddings → models/training → run loop.
4. Set `ENDPOINT_LIST` (`["Ki"]` or `["IC50"]`), `SPLIT_MODES` (`["random","cold_drug"]`) and `MODELS`; finished models are skipped and unfinished ones resume from `latest.pt`.
5. Optional cells: hide DataLoader worker warnings and set `EPOCHS = 50`; `compare_models()` to build the comparison table; extend the best model of each split to 120 epochs.
6. Outputs appear under `13_Models/` (see section 6).

---

## 13. Repository structure

```
dti_app/
├── app.py, config.py, requirements.txt, README.md, plan.md
├── dti/         featurize.py, encoders.py, protein.py, registry.py
├── services/    labels.py, chem.py, explain.py, uniprot.py, screening.py, docking.py, jobs.py
├── routes/      pages.py, api.py
├── templates/   base, index, predict, batch, screen, dock, compare, about
├── static/      css/, js/, img/
├── checkpoints/ <endpoint>_<split>/<model>/{best.pt,config.json}
├── data/        reference/, examples/, results/
├── cache/       protein embeddings, docking jobs
├── uploads/
└── tests/
```

---

## 14. Limitations

- **Protein information is coarse.** One mean-pooled 320-d vector per protein ignores which residues form the binding pocket. The model learns "which proteins tend to bind what", not 3D complementarity.
- **No 3D input.** The network never sees a binding pose; the docking tab is a separate tool.
- **Label noise.** BindingDB mixes labs and assay conditions; typical inter-lab differences are several-fold. This puts a floor on achievable error. Censored values may bias results (see section 7).
- **New scaffolds are harder:** cold_drug RMSE ≈ 1.0–1.05 (about ten-fold typical error). Always read the uncertainty band.
- **Generalisation to unseen proteins is untested** (cold_protein split not run).
- **Unequal training budgets** across models; GINE under-trained.
- **Domain:** trained on drug-like small molecules with atoms H, C, N, O, F, P, S, Cl, Br, I (others fall into an "other" bin); peptides, metals and very large molecules are outside the domain.
- **Not medical advice**; not validated for clinical or regulatory use.

---

## 15. References

- Gilmer, J. et al. (2017). *Neural Message Passing for Quantum Chemistry.* ICML. (MPNN)
- Xu, K. et al. (2019). *How Powerful are Graph Neural Networks?* ICLR. (GIN)
- Hu, W. et al. (2020). *Strategies for Pre-training Graph Neural Networks.* ICLR. (GINE-style edge-aware GIN)
- Xiong, Z. et al. (2020). *Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph Attention Mechanism.* J. Med. Chem. (AttentiveFP)
- Lin, Z. et al. (2023). *Evolutionary-scale prediction of atomic-level protein structure with a language model.* Science. (ESM-2)
- Liu, T. et al. (2007) / Gilson, M. K. et al. (2016). *BindingDB.* Nucleic Acids Res.
- Bemis, G. W. & Murcko, M. A. (1996). *The properties of known drugs. 1. Molecular frameworks.* J. Med. Chem. (scaffolds)
- Öztürk, H. et al. (2018). *DeepDTA: deep drug–target binding affinity prediction.* Bioinformatics.
- Pei, Q. et al. (2023). *kNN-DTA* / *Breaking the barriers of data scarcity in drug–target affinity prediction.* Briefings in Bioinformatics.
- Fey, M. & Lenssen, J. E. (2019). *Fast Graph Representation Learning with PyTorch Geometric.*
- Loshchilov, I. & Hutter, F. (2019). *Decoupled Weight Decay Regularization.* ICLR. (AdamW)
- Trott, O. & Olson, A. J. (2010). *AutoDock Vina.* J. Comput. Chem.; Eberhardt, J. et al. (2021). *AutoDock Vina 1.2.0.* J. Chem. Inf. Model.
- Landrum, G. *RDKit: Open-source cheminformatics.* https://www.rdkit.org
- Rego, N. & Koes, D. (2015). *3Dmol.js.* Bioinformatics.

*Check the exact bibliographic details of the references against the original papers before citing them in a report.*
