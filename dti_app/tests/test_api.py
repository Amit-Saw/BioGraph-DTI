"""API and page tests using a fake registry (no checkpoints needed). Requires RDKit."""
import io
import time

import pytest

pytest.importorskip("rdkit")

from app import create_app                                     # noqa: E402
from tests.fakes import FakeReference, FakeRegistry            # noqa: E402

ERLOTINIB = "COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC"
GEFITINIB = "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"
SEQ = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVV"


def client():
    app = create_app(load_models=False, registry=FakeRegistry(), reference=FakeReference())
    app.testing = True
    return app.test_client()


def wait(c, job_id, timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = c.get(f"/api/jobs/{job_id}").get_json()
        if st["state"] in ("done", "error"):
            return st
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def test_pages_render():
    c = client()
    for path in ["/", "/predict", "/batch", "/screen", "/compare", "/dock", "/about", "/health"]:
        assert c.get(path).status_code == 200, path


def test_models_and_examples():
    c = client()
    assert len(c.get("/api/models").get_json()) == 4
    assert "drugs" in c.get("/api/examples").get_json()


def test_validate():
    c = client()
    ok = c.post("/api/validate", json={"smiles": ERLOTINIB}).get_json()
    assert ok["ok"] and "<svg" in ok["svg"]
    bad = c.post("/api/validate", json={"smiles": "not a smiles"}).get_json()
    assert not bad["ok"]


def test_predict_shape_and_errors():
    c = client()
    r = c.post("/api/predict", json={"smiles": ERLOTINIB, "sequence": SEQ})
    assert r.status_code == 200
    ki = r.get_json()["results"]["Ki"]
    assert ki["primary"]["split"] == "cold_drug" and ki["reference"]["split"] == "random"
    assert ki["primary"]["label"] in ("Strong", "Moderate", "Weak")
    assert c.post("/api/predict", json={"smiles": "xx", "sequence": SEQ}).status_code == 400
    assert c.post("/api/predict", json={"smiles": ERLOTINIB, "sequence": "123"}).status_code == 400


def test_properties_conformer_similar():
    c = client()
    p = c.post("/api/properties", json={"smiles": ERLOTINIB}).get_json()
    assert p["descriptors"]["mol_weight"] > 300
    assert "molblock" in c.post("/api/conformer", json={"smiles": ERLOTINIB}).get_json()
    s = c.post("/api/similar", json={"smiles": ERLOTINIB, "sequence": SEQ}).get_json()
    assert s["results"]["Ki"]["level"] == "low"


def test_compare():
    c = client()
    r = c.post("/api/compare", json={"smiles_list": [ERLOTINIB, GEFITINIB], "sequence": SEQ}).get_json()
    assert len(r["items"]) == 2 and "Ki" in r["items"][0]["results"]


def test_batch_job_and_csv():
    c = client()
    data = {"file": (io.BytesIO(f"name,smiles\nA,{ERLOTINIB}\nB,{GEFITINIB}\nC,bad\n".encode()), "lib.csv"),
            "sequence": SEQ, "endpoints": "Ki,IC50"}
    start = c.post("/api/batch", data=data, content_type="multipart/form-data").get_json()
    assert start["n_valid"] == 2 and start["n_invalid"] == 1
    st = wait(c, start["job_id"])
    assert st["state"] == "done" and len(st["result"]["rows"]) == 2
    csv_text = c.get(f"/api/download/{start['job_id']}.csv").get_data(as_text=True)
    assert csv_text.startswith("rank,name,smiles,Ki_p")


def test_screen_top_n():
    c = client()
    lines = "\n".join([ERLOTINIB, GEFITINIB, "CCO", "CCN"])
    data = {"file": (io.BytesIO(lines.encode()), "lib.smi"), "sequence": SEQ, "endpoint": "Ki", "top_n": "2"}
    start = c.post("/api/screen", data=data, content_type="multipart/form-data").get_json()
    st = wait(c, start["job_id"])
    assert st["state"] == "done" and len(st["result"]["rows"]) == 2


def test_unknown_job_and_dock_disabled_message():
    c = client()
    assert c.get("/api/jobs/nope").status_code == 404
    r = c.post("/api/dock", json={"pdb_id": "1M17", "ligand": {"code": "AQ4", "chain": "A", "resi": "999"}})
    assert r.status_code in (200, 503)          # 503 when Vina / Open Babel are not installed
