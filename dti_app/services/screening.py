"""Batch scoring / virtual screening: many drugs against one protein."""
import csv
import io
import re

import config
from dti import smiles as smiles_mod
from services import chem, labels, predict_service
from services.errors import UserError


def parse_library(raw, max_rows=None):
    """CSV with a 'smiles' column (optional 'name'), or plain text with one SMILES per line
    (optionally followed by a name).  -> (entries[{name, smiles}], invalid[{line, smiles, reason}])"""
    max_rows = max_rows or config.MAX_UPLOAD_ROWS
    text = raw.decode("utf-8-sig", errors="replace") if isinstance(raw, (bytes, bytearray)) else str(raw)
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        raise UserError("The uploaded file is empty.")

    records = []                                   # (line_no, name, smiles)
    header = [h.strip().lower() for h in re.split(r"[,\t;]", lines[0])]
    if "smiles" in header or "canonical_smiles" in header:
        col = header.index("smiles") if "smiles" in header else header.index("canonical_smiles")
        name_col = next((header.index(c) for c in ("name", "id", "compound", "compound_id") if c in header), None)
        delim = "\t" if "\t" in lines[0] else (";" if ";" in lines[0] and "," not in lines[0] else ",")
        for n, row in enumerate(csv.reader(io.StringIO("\n".join(lines[1:])), delimiter=delim), start=2):
            if len(row) <= col:
                continue
            name = row[name_col].strip() if name_col is not None and len(row) > name_col else ""
            records.append((n, name, row[col].strip()))
    else:
        for n, line in enumerate(lines, start=1):
            parts = line.split(None, 1) if not ("," in line or ";" in line) else re.split(r"[,;]\s*", line, 1)
            records.append((n, parts[1].strip() if len(parts) > 1 else "", parts[0].strip()))

    if len(records) > max_rows:
        raise UserError(f"The file has {len(records)} rows; the limit is {max_rows}.", 413)
    entries, invalid = [], []
    for n, name, smi in records:
        ok, msg, can = smiles_mod.validate_smiles(smi)
        if ok:
            entries.append({"name": name or f"mol_{len(entries) + 1}", "smiles": can})
        else:
            invalid.append({"line": n, "smiles": smi[:80], "reason": msg})
    if not entries:
        raise UserError("No valid SMILES found in the file.")
    return entries, invalid


def run_scoring(job, registry, reference, endpoints, seq, entries, top_n=None, thumbnails=0):
    """Score every entry for each endpoint with the primary (cold_drug) model; rank by the first endpoint."""
    smiles = [e["smiles"] for e in entries]
    n_steps = len(endpoints) * len(smiles)
    done_steps, models = 0, {}
    rows = [{"name": e["name"], "smiles": e["smiles"]} for e in entries]

    for ep in endpoints:
        primary, _ = predict_service.select_models(registry, ep)
        if primary is None:
            raise UserError(f"No {ep} model is loaded.", 503)
        split, name = primary
        models[ep] = {"split": split, "model": name}
        base = done_steps

        def progress(k, total, base=base):
            job.update((base + k) / max(n_steps, 1), f"{ep}: scored {k:,} of {total:,}")

        preds = registry.predict_many(ep, split, name, smiles, seq, progress=progress)
        for row, p in zip(rows, preds):
            row[f"{ep}_p"] = round(float(p), 3)
            row[f"{ep}_nM"] = labels.to_nM(p)
            row[f"{ep}_conc"] = labels.format_conc(labels.to_nM(p))
            row[f"{ep}_label"] = labels.potency_label(p)
        done_steps += len(smiles)

    first = endpoints[0]
    rows.sort(key=lambda r: -r[f"{first}_p"])
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    if top_n:
        rows = rows[:top_n]

    job.update(0.97, "Checking similarity to training data")
    for i, r in enumerate(rows):
        if i < config.APPLICABILITY_ROWS and reference is not None and reference.available(first):
            try:
                r["applicability"] = reference.applicability(first, r["smiles"])["level"]
            except Exception:                                  # noqa: BLE001
                r["applicability"] = None
        if i < thumbnails:
            r["svg"] = chem.draw_svg(r["smiles"], size=(180, 130))
    return {"rows": rows, "endpoints": list(endpoints), "models": models, "n_scored": len(smiles)}


def rows_to_csv(result):
    eps = result["endpoints"]
    cols = ["rank", "name", "smiles"]
    for ep in eps:
        cols += [f"{ep}_p", f"{ep}_nM", f"{ep}_label"]
    cols.append("applicability")
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in result["rows"]:
        w.writerow(r)
    return buf.getvalue()
