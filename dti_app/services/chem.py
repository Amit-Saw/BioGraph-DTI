"""RDKit helpers: 2D drawing, properties, 3D conformer, and similarity to the training drugs."""
import threading
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, rdDepictor, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

import config


# ------------------------------------------------------------------ drawing
def draw_svg(smiles, size=(420, 320), highlight=None):
    """2D structure as SVG text.  highlight = {atom_index: (r, g, b)} with values in 0..1 (used by explain)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")
    rdDepictor.Compute2DCoords(mol)
    d = rdMolDraw2D.MolDraw2DSVG(int(size[0]), int(size[1]))
    d.drawOptions().padding = 0.08
    if highlight:
        atoms = [int(i) for i in highlight]
        rdMolDraw2D.PrepareAndDrawMolecule(
            d, mol, highlightAtoms=atoms,
            highlightAtomColors={int(i): tuple(c) for i, c in highlight.items()},
            highlightAtomRadii={i: 0.38 for i in atoms})
    else:
        rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
    d.FinishDrawing()
    return d.GetDrawingText()


# ------------------------------------------------------------------ properties
def descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return {
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "mol_weight": round(Descriptors.MolWt(mol), 2),
        "logp": round(Crippen.MolLogP(mol), 2),
        "h_donors": int(Lipinski.NumHDonors(mol)),
        "h_acceptors": int(Lipinski.NumHAcceptors(mol)),
        "tpsa": round(rdMolDescriptors.CalcTPSA(mol), 1),
        "rotatable_bonds": int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        "rings": int(rdMolDescriptors.CalcNumRings(mol)),
        "heavy_atoms": int(mol.GetNumHeavyAtoms()),
    }


def lipinski(desc):
    """Rule of five on a descriptors() dict. One violation is still considered a pass."""
    v = []
    if desc["mol_weight"] > 500: v.append("molecular weight > 500")
    if desc["logp"] > 5: v.append("logP > 5")
    if desc["h_donors"] > 5: v.append("more than 5 H-bond donors")
    if desc["h_acceptors"] > 10: v.append("more than 10 H-bond acceptors")
    return {"violations": v, "passes": len(v) <= 1}


def conformer_molblock(smiles, seed=42):
    """3D conformer (ETKDG + MMFF) as a MOL block for 3Dmol.js."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) != 0:
            raise ValueError("Could not build a 3D conformer for this molecule.")
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        pass
    return Chem.MolToMolBlock(mol)


# ------------------------------------------------------------------ fingerprints
_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)
_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
_gen_lock = threading.Lock()


def fingerprint(smiles):
    """Packed 2048-bit Morgan (radius 2) fingerprint as 256 bytes, or None for an invalid SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    with _gen_lock:
        bits = _gen.GetFingerprintAsNumPy(mol).astype(np.uint8)
    return np.packbits(bits)


def tanimoto_to_many(q, fps):
    """Tanimoto of one packed fingerprint against an (n, 256) uint8 matrix."""
    inter = _POP[np.bitwise_and(fps, q)].sum(axis=1).astype(np.float32)
    union = float(_POP[q].sum()) + _POP[fps].sum(axis=1).astype(np.float32) - inter
    return inter / np.maximum(union, 1.0)


class _Ref:
    """Training drugs of one endpoint with fingerprints."""
    def __init__(self, path, cache_path):
        df = pd.read_csv(path, usecols=["canonical_smiles", "protein_sequence", "p_affinity"],
                         dtype={"canonical_smiles": "category", "protein_sequence": "category"})
        g = df.groupby("canonical_smiles", observed=True).agg(
            n_meas=("p_affinity", "size"), median_p=("p_affinity", "median"),
            n_targets=("protein_sequence", "nunique"))
        self.smiles = np.array(g.index.astype(str))
        self.n_meas = g["n_meas"].to_numpy()
        self.median_p = g["median_p"].to_numpy()
        self.n_targets = g["n_targets"].to_numpy()
        self.df = df
        fps = None
        if cache_path.exists() and cache_path.stat().st_mtime >= Path(path).stat().st_mtime:
            try:
                arr = np.load(cache_path)
                if arr.shape == (len(self.smiles), 256):
                    fps = arr
            except Exception:
                fps = None
        if fps is None:
            fps = np.zeros((len(self.smiles), 256), dtype=np.uint8)
            for i, s in enumerate(self.smiles):
                fp = fingerprint(s)
                if fp is not None:
                    fps[i] = fp
            try:
                cache_path.parent.mkdir(exist_ok=True)
                np.save(cache_path, fps)
            except OSError:
                pass
        self.fps = fps


class ReferenceIndex:
    """Similarity search / applicability domain against data/reference/<endpoint>_train.csv."""
    def __init__(self, reference_dir=None):
        self.dir = Path(reference_dir or config.REFERENCE_DIR)
        self._refs = {}
        self._lock = threading.RLock()

    def _path(self, endpoint):
        return self.dir / f"{endpoint}_train.csv"

    def available(self, endpoint):
        return self._path(endpoint).exists()

    def _load(self, endpoint):
        with self._lock:
            if endpoint not in self._refs:
                self._refs[endpoint] = _Ref(self._path(endpoint), config.CACHE_DIR / f"ref_{endpoint}_fps.npy")
            return self._refs[endpoint]

    def _sims(self, endpoint, smiles):
        ref = self._load(endpoint)
        q = fingerprint(smiles)
        return ref, (None if q is None else tanimoto_to_many(q, ref.fps))

    def max_similarity(self, endpoint, smiles):
        if not self.available(endpoint):
            return None
        _, sims = self._sims(endpoint, smiles)
        return None if sims is None else float(sims.max())

    def applicability(self, endpoint, smiles):
        """-> {max_tanimoto, level: high|medium|low|unknown, message}"""
        sim = self.max_similarity(endpoint, smiles) if self.available(endpoint) else None
        if sim is None:
            return {"max_tanimoto": None, "level": "unknown",
                    "message": "No reference data, so similarity to the training set is unknown."}
        if sim >= config.SIM_HIGH:
            level, msg = "high", "Similar compounds were in the training data."
        elif sim >= config.SIM_MID:
            level, msg = "medium", "Only moderately similar compounds were in the training data."
        else:
            level, msg = "low", "This molecule is far from the training data; treat the value as a rough estimate."
        return {"max_tanimoto": round(sim, 3), "level": level, "message": msg}

    def nearest(self, endpoint, smiles, k=5, seq=None):
        if not self.available(endpoint):
            return []
        ref, sims = self._sims(endpoint, smiles)
        if sims is None:
            return []
        k = max(1, min(k, len(sims)))
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        same = {}
        if seq:
            names = [str(ref.smiles[i]) for i in idx]
            sub = ref.df[(ref.df.protein_sequence == seq) & ref.df.canonical_smiles.isin(names)]
            if len(sub):
                same = sub.groupby("canonical_smiles", observed=True)["p_affinity"].median().to_dict()
        out = []
        for i in idx:
            s = str(ref.smiles[i])
            out.append({"smiles": s, "similarity": round(float(sims[i]), 3),
                        "median_pAffinity": round(float(ref.median_p[i]), 2),
                        "n_targets": int(ref.n_targets[i]), "n_measurements": int(ref.n_meas[i]),
                        "same_target_pAffinity": round(float(same[s]), 2) if s in same else None})
        return out
