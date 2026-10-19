"""JSON API.  All errors are {"error": message}.  See plan.md section 6."""
import csv
import io

import pandas as pd
from flask import Blueprint, Response, current_app, jsonify, request

import config
from dti import protein as protein_mod
from dti import smiles as smiles_mod
from services import chem, docking, explain as explain_svc, labels, predict_service, screening, uniprot
from services.errors import UserError

bp = Blueprint("api", __name__, url_prefix="/api")


# ------------------------------------------------------------------ helpers
def _body():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise UserError("Expected a JSON body.")
    return data


def _smiles(raw):
    ok, msg, canonical = smiles_mod.validate_smiles(raw)
    if not ok:
        raise UserError(msg)
    return canonical


def _seq(raw):
    try:
        return protein_mod.clean_sequence(raw)
    except ValueError as e:
        raise UserError(str(e))


def _registry():
    reg = current_app.registry
    if reg is None or not reg.models:
        raise UserError("No models are loaded. Put checkpoints in checkpoints/ and restart the app.", 503)
    return reg


def _endpoints(raw, default=config.ENDPOINTS):
    if not raw:
        return list(default)
    if isinstance(raw, str):
        raw = [x.strip() for x in raw.split(",") if x.strip()]
    bad = [e for e in raw if e not in config.ENDPOINTS]
    if bad or not raw:
        raise UserError("endpoints must be a subset of: " + ", ".join(config.ENDPOINTS))
    return list(dict.fromkeys(raw))


def _form_sequence():
    """Sequence for upload forms: pasted sequence, or a UniProt accession."""
    seq = (request.form.get("sequence") or "").strip()
    if not seq and request.form.get("uniprot"):
        seq = uniprot.fetch_sequence(request.form["uniprot"])[1]
    return _seq(seq)


# ------------------------------------------------------------------ info
@bp.get("/models")
def models():
    reg = current_app.registry
    return jsonify(reg.available() if reg else [])


@bp.get("/examples")
def examples():
    def read(name):
        p = config.EXAMPLES_DIR / name
        if not p.exists() or p.stat().st_size == 0:
            return []
        try:
            return pd.read_csv(p).fillna("").to_dict("records")
        except Exception:
            return []
    ok, msg = docking.docking_available()
    return jsonify(drugs=read("drugs.csv"), targets=read("targets.csv"),
                   docking={"available": ok, "message": msg})


@bp.get("/uniprot/<accession>")
def uniprot_lookup(accession):
    name, seq = uniprot.fetch_sequence(accession)
    return jsonify(name=name, sequence=seq, length=len(seq))


# ------------------------------------------------------------------ single drug
@bp.post("/validate")
def validate():
    ok, msg, canonical = smiles_mod.validate_smiles(_body().get("smiles"))
    if not ok:
        return jsonify(ok=False, message=msg)
    return jsonify(ok=True, canonical=canonical, svg=chem.draw_svg(canonical, size=(380, 280)))


@bp.post("/predict")
def predict():
    d = _body()
    smiles, seq, eps = _smiles(d.get("smiles")), _seq(d.get("sequence")), _endpoints(d.get("endpoints"))
    reg = _registry()
    return jsonify(smiles=smiles, results={ep: predict_service.predict_endpoint(reg, ep, smiles, seq) for ep in eps})


@bp.post("/properties")
def properties():
    smiles = _smiles(_body().get("smiles"))
    desc = chem.descriptors(smiles)
    return jsonify(smiles=smiles, descriptors=desc, lipinski=chem.lipinski(desc),
                   svg=chem.draw_svg(smiles))


@bp.post("/similar")
def similar():
    d = _body()
    smiles = _smiles(d.get("smiles"))
    seq = _seq(d["sequence"]) if d.get("sequence") else None
    ref = current_app.reference
    out = {}
    for ep in config.ENDPOINTS:
        if ref is None or not ref.available(ep):
            out[ep] = {"available": False}
            continue
        out[ep] = {"available": True, **ref.applicability(ep, smiles),
                   "neighbours": ref.nearest(ep, smiles, k=5, seq=seq)}
    return jsonify(smiles=smiles, results=out,
                   thresholds={"high": config.SIM_HIGH, "medium": config.SIM_MID})


@bp.post("/conformer")
def conformer():
    smiles = _smiles(_body().get("smiles"))
    try:
        return jsonify(molblock=chem.conformer_molblock(smiles))
    except ValueError as e:
        raise UserError(str(e))


@bp.post("/explain")
def explain():
    d = _body()
    smiles, seq = _smiles(d.get("smiles")), _seq(d.get("sequence"))
    ep = _endpoints([d.get("endpoint", "Ki")])[0]
    reg = _registry()
    primary, _ = predict_service.select_models(reg, ep)
    if primary is None:
        raise UserError(f"No {ep} model is loaded.", 503)
    split, name = d.get("split") or primary[0], d.get("model") or primary[1]
    result = explain_svc.explain(reg, ep, split, name, smiles, seq)
    result.update(endpoint=ep, split=split, model=name)
    return jsonify(result)


# ------------------------------------------------------------------ compare / selectivity
@bp.post("/compare")
def compare():
    d = _body()
    raw = d.get("smiles_list") or []
    if not 2 <= len(raw) <= config.MAX_COMPARE_DRUGS:
        raise UserError(f"Give between 2 and {config.MAX_COMPARE_DRUGS} SMILES.")
    smiles = [_smiles(s) for s in raw]
    seq, reg = _seq(d.get("sequence")), _registry()
    items = [{"smiles": s, "svg": chem.draw_svg(s, size=(260, 200)), "results": {}} for s in smiles]
    for ep in config.ENDPOINTS:
        primary, _ = predict_service.select_models(reg, ep)
        if primary is None:
            continue
        lm = reg.get(ep, *primary)
        preds = reg.predict_many(ep, primary[0], primary[1], smiles, seq)
        for item, p in zip(items, preds):
            item["results"][ep] = labels.describe(p, lm.test_rmse)
    return jsonify(items=items)


@bp.post("/selectivity")
def selectivity():
    d = _body()
    smiles = _smiles(d.get("smiles"))
    targets = d.get("targets") or []
    if not 1 <= len(targets) <= config.MAX_SELECTIVITY_TARGETS:
        raise UserError(f"Give between 1 and {config.MAX_SELECTIVITY_TARGETS} targets.")
    reg = _registry()
    rows = []
    for t in targets:
        row = {"name": t.get("name") or t.get("uniprot") or "target", "results": {}}
        try:
            if t.get("sequence"):
                seq = _seq(t["sequence"])
            else:
                name, raw = uniprot.fetch_sequence(t.get("uniprot"))
                row["name"] = t.get("name") or name
                seq = _seq(raw)
            for ep in config.ENDPOINTS:
                primary, _ = predict_service.select_models(reg, ep)
                if primary is None:
                    continue
                lm = reg.get(ep, *primary)
                p = reg.predict_pair(ep, primary[0], primary[1], smiles, seq)
                row["results"][ep] = labels.describe(p, lm.test_rmse)
        except UserError as e:
            row["error"] = str(e)
        rows.append(row)
    return jsonify(smiles=smiles, svg=chem.draw_svg(smiles, size=(300, 220)), rows=rows)


# ------------------------------------------------------------------ batch / screening jobs
def _start_scoring(kind):
    reg = _registry()
    f = request.files.get("file")
    if f is None or not f.filename:
        raise UserError("Upload a CSV or text file of SMILES.")
    seq = _form_sequence()
    eps = _endpoints(request.form.get("endpoints") or request.form.get("endpoint"))
    entries, invalid = screening.parse_library(f.read())
    top_n = None
    if kind == "screen":
        try:
            top_n = max(1, min(int(request.form.get("top_n", 100)), 1000))
        except ValueError:
            raise UserError("top_n must be a number.")
    thumbs = config.SCREEN_THUMBNAILS if kind == "screen" else 0
    reference = current_app.reference

    def work(job):
        out = screening.run_scoring(job, reg, reference, eps, seq, entries, top_n, thumbs)
        out["invalid"] = invalid
        return out

    job_id = current_app.jobs.submit(work, kind)
    return jsonify(job_id=job_id, n_valid=len(entries), n_invalid=len(invalid))


@bp.post("/batch")
def batch():
    return _start_scoring("batch")


@bp.post("/screen")
def screen():
    return _start_scoring("screen")


@bp.get("/jobs/<job_id>")
def job_status(job_id):
    st = current_app.jobs.status(job_id)
    if st is None:
        raise UserError("Unknown or expired job.", 404)
    return jsonify(st)


@bp.get("/download/<job_id>.csv")
def download(job_id):
    job = current_app.jobs.get(job_id)
    if job is None or job.state != "done" or not job.result or "rows" not in job.result:
        raise UserError("No finished results for this job.", 404)
    return Response(screening.rows_to_csv(job.result), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename=dti_{job.kind}_{job_id}.csv"})


# ------------------------------------------------------------------ docking
@bp.get("/pdb/<pdb_id>/ligands")
def pdb_ligands(pdb_id):
    return jsonify(docking.list_ligands(pdb_id))


@bp.post("/dock")
def dock():
    d = _body()
    ok, msg = docking.docking_available()
    if not ok:
        raise UserError(msg, 503)
    lig = d.get("ligand") or {}
    if not all(lig.get(k) for k in ("code", "chain", "resi")):
        raise UserError("Choose the pocket ligand (code, chain, residue).")
    smiles = _smiles(d["smiles"]) if (d.get("smiles") or "").strip() else None
    chains = [c for c in (d.get("chains") or []) if isinstance(c, str) and len(c) == 1]

    def work(job):
        return docking.dock_job(job, d.get("pdb_id"), chains, lig, smiles, d.get("box"), d.get("exhaustiveness"))

    return jsonify(job_id=current_app.jobs.submit(work, "dock"))
