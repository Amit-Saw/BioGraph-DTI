"""Unit conversion, potency labels and uncertainty ranges. Pure python (no torch / rdkit)."""
import math

import config


def to_nM(p):
    """pAffinity = 9 - log10(nM)  ->  nM = 10 ** (9 - pAffinity)."""
    return float(10.0 ** (9.0 - float(p)))


def format_conc(nM):
    nM = float(nM)
    if nM < 1.0:
        return f"{nM * 1000:.3g} pM"
    if nM < 1000.0:
        return f"{nM:.3g} nM"
    if nM < 1e6:
        return f"{nM / 1000:.3g} µM"
    return f"{nM / 1e6:.3g} mM"


def potency_label(p):
    """'Strong' (<= 10 nM), 'Moderate' (10 nM - 1 uM) or 'Weak' (> 1 uM).  The UI uses the text as a CSS class."""
    p = float(p)
    if p >= config.STRONG_MIN:
        return "Strong"
    if p >= config.MODERATE_MIN:
        return "Moderate"
    return "Weak"


def uncertainty_band(p, sigma):
    """Approximate 1-sigma range in nM [low, high]. Higher pAffinity = lower nM, so the bounds swap."""
    if sigma is None or not math.isfinite(sigma):
        return None
    return [to_nM(p + sigma), to_nM(p - sigma)]


def describe(p, sigma=None):
    """Everything the UI needs for one predicted pAffinity value (sigma = the model's test RMSE)."""
    p = float(p)
    nm = to_nM(p)
    band = uncertainty_band(p, sigma)
    return {
        "pAffinity": round(p, 3),
        "nM": nm,
        "conc": format_conc(nm),
        "label": potency_label(p),
        "sigma": None if sigma is None else round(float(sigma), 3),
        "band_nM": band,
        "band_text": None if band is None else f"{format_conc(band[0])} – {format_conc(band[1])}",
    }
