"""Atom-level explanation by occlusion (no retraining needed).

For every atom we delete it (and its bonds) from the molecular graph and predict again:
    score_i = p_full - p_without_atom_i
Positive: the atom pushes the predicted affinity UP (removing it lowers the prediction).
Negative: the atom pushes it down.  This is the model's sensitivity, not proof of chemistry."""
import config
from services import chem
from services.errors import UserError


def _color(score, max_abs):
    """Blue = raises the predicted value, orange = lowers it; stronger colour = larger |score|."""
    a = 0.30 + 0.70 * min(1.0, abs(score) / max_abs)
    base = (26 / 255, 102 / 255, 217 / 255) if score > 0 else (235 / 255, 128 / 255, 20 / 255)
    return tuple(round(1.0 - a * (1.0 - c), 3) for c in base)


def explain(registry, endpoint, split, name, smiles, sequence):
    # lazy imports: these need torch, the rest of the app does not
    from rdkit import Chem
    from dti.featurize import data_without_atom, smiles_to_data

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise UserError("Invalid SMILES.")
    n = mol.GetNumAtoms()
    if n > config.MAX_EXPLAIN_ATOMS:
        raise UserError(f"The molecule has {n} atoms; the explanation is limited to {config.MAX_EXPLAIN_ATOMS}.")
    try:
        registry.get(endpoint, split, name)
    except KeyError as e:
        raise UserError(str(e).strip("'\""), 404)

    base = smiles_to_data(smiles, registry.cache.get(sequence))
    if base.x.shape[0] != n:                                      # atom order of graph and drawing must agree
        raise UserError("Internal error: atom count mismatch.", 500)
    if n == 1:
        p0 = float(registry.predict_data(endpoint, split, name, [base])[0])
        scores = [0.0]
    else:
        datas = [base] + [data_without_atom(base, i) for i in range(n)]
        preds = registry.predict_data(endpoint, split, name, datas)
        p0 = float(preds[0])
        scores = [float(p0 - preds[i + 1]) for i in range(n)]

    max_abs = max((abs(s) for s in scores), default=0.0)
    highlight = {i: _color(s, max_abs) for i, s in enumerate(scores)
                 if max_abs > 0 and abs(s) >= 0.05 * max_abs}
    atoms = [{"index": i, "symbol": mol.GetAtomWithIdx(i).GetSymbol(), "score": round(s, 3)}
             for i, s in enumerate(scores)]
    top = sorted(atoms, key=lambda a: abs(a["score"]), reverse=True)[:8]
    return {
        "pAffinity": round(p0, 3),
        "max_abs": round(max_abs, 3),
        "atoms": atoms,
        "top_atoms": top,
        "svg": chem.draw_svg(smiles, size=(520, 400), highlight=highlight),
    }
