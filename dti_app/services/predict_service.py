"""Chooses which loaded models answer a request and shapes the prediction result."""
import config
from services import labels


def select_models(registry, endpoint):
    """-> (primary, reference) as (split, name) tuples; reference may be None.
    primary  = best cold_drug model (realistic for new compounds); falls back to any loaded split.
    reference = best random-split model, shown as a secondary value."""
    splits = sorted({m.split for m in registry.models.values() if m.endpoint == endpoint})
    if not splits:
        return None, None
    primary_split = config.DEFAULT_SPLIT if config.DEFAULT_SPLIT in splits else splits[0]
    primary = (primary_split, registry.best_model(endpoint, primary_split))
    reference = None
    if config.REFERENCE_SPLIT in splits and config.REFERENCE_SPLIT != primary_split:
        reference = (config.REFERENCE_SPLIT, registry.best_model(endpoint, config.REFERENCE_SPLIT))
    return primary, reference


def _one(registry, endpoint, split, name, smiles, seq):
    lm = registry.get(endpoint, split, name)
    p = registry.predict_pair(endpoint, split, name, smiles, seq)
    d = labels.describe(p, lm.test_rmse)
    d.update(model=name, split=split, test_rmse=lm.test_rmse)
    return d


def predict_endpoint(registry, endpoint, smiles, seq):
    primary, reference = select_models(registry, endpoint)
    if primary is None:
        return {"error": f"No {endpoint} model is loaded."}
    out = {"primary": _one(registry, endpoint, *primary, smiles, seq)}
    if reference:
        out["reference"] = _one(registry, endpoint, *reference, smiles, seq)
        diff = abs(out["primary"]["pAffinity"] - out["reference"]["pAffinity"])
        out["agreement_abs"] = round(diff, 3)
        out["models_disagree"] = diff > config.MODELS_DISAGREE
    return out
