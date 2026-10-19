"""End-to-end verification script for all DTI web app routes and API endpoints."""
import io
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import create_app

ERLOTINIB = "COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC"
GEFITINIB = "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"
EGFR_SHORT = "MRPSGTAGAALLALLAALCPASRALEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLGNLEITYVQRNYDLSFLKTIQEVAGYVLIALNTVER"

def test_all():
    print("=" * 65)
    print("Initializing Flask app with real checkpoints & ESM-2...")
    app = create_app(load_models=True)
    app.testing = True
    c = app.test_client()
    print("[ OK ] Flask application created successfully.")
    print("=" * 65)

    failures = 0
    total = 0

    def check(name, status_ok, details=""):
        nonlocal failures, total
        total += 1
        tag = "[ PASS ]" if status_ok else "[ FAIL ]"
        if not status_ok:
            failures += 1
        print(f"{tag} {name} {details}")

    # 1. Page Routes
    print("\n--- 1. Web Page Routes (HTML) ---")
    pages = ["/", "/predict", "/batch", "/screen", "/compare", "/dock", "/about", "/health"]
    for p in pages:
        res = c.get(p)
        check(f"GET {p}", res.status_code == 200, f"(Status: {res.status_code})")

    # 2. Metadata & Helpers
    print("\n--- 2. Metadata & Info Endpoints ---")
    r = c.get("/api/models")
    data = r.get_json()
    check("GET /api/models", r.status_code == 200 and len(data) >= 4, f"({len(data)} models loaded)")

    r = c.get("/api/examples")
    data = r.get_json()
    check("GET /api/examples", r.status_code == 200 and "drugs" in data and len(data["drugs"]) > 0, 
          f"({len(data.get('drugs', []))} drugs, {len(data.get('targets', []))} targets)")

    # UniProt lookup (network call)
    try:
        r = c.get("/api/uniprot/P00533")
        data = r.get_json()
        check("GET /api/uniprot/P00533", r.status_code == 200 and "sequence" in data, f"(Loaded '{data.get('name')}', {data.get('length')} aa)")
    except Exception as e:
        print(f"[ WARN ] UniProt lookup skipped/failed (network required): {e}")

    # 3. Chemistry & Validation Endpoints
    print("\n--- 3. Chemical Analysis Endpoints ---")
    r = c.post("/api/validate", json={"smiles": ERLOTINIB})
    data = r.get_json()
    check("POST /api/validate (valid SMILES)", r.status_code == 200 and data.get("ok"), "(Valid canonical SMILES generated)")

    r = c.post("/api/validate", json={"smiles": "INVALID_SMILES_STRING"})
    data = r.get_json()
    check("POST /api/validate (invalid SMILES)", r.status_code == 200 and not data.get("ok"), f"(Rejected gracefully: {data.get('message')})")

    r = c.post("/api/properties", json={"smiles": ERLOTINIB})
    data = r.get_json()
    check("POST /api/properties", r.status_code == 200 and "descriptors" in data, f"(MW: {data['descriptors'].get('mol_weight')}, Formula: {data['descriptors'].get('formula')})")

    r = c.post("/api/conformer", json={"smiles": ERLOTINIB})
    data = r.get_json()
    check("POST /api/conformer (3D)", r.status_code == 200 and "molblock" in data and len(data["molblock"]) > 100, "(3D Molblock generated)")

    # 4. Predictions & Explainability (Real Inference)
    print("\n--- 4. Machine Learning Inference & Explanation ---")
    t0 = time.time()
    r = c.post("/api/predict", json={"smiles": ERLOTINIB, "sequence": EGFR_SHORT, "endpoints": ["Ki", "IC50"]})
    t_pred = time.time() - t0
    data = r.get_json()
    ki_p = data.get("results", {}).get("Ki", {}).get("primary", {}).get("pAffinity")
    ic50_p = data.get("results", {}).get("IC50", {}).get("primary", {}).get("pAffinity")
    check("POST /api/predict (Ki & IC50)", r.status_code == 200 and ki_p is not None and ic50_p is not None,
          f"(pKi: {ki_p:.2f}, pIC50: {ic50_p:.2f} in {t_pred:.2f}s)")

    # Similar compounds
    r = c.post("/api/similar", json={"smiles": ERLOTINIB, "sequence": EGFR_SHORT})
    data = r.get_json()
    check("POST /api/similar", r.status_code == 200 and "results" in data, "(Tanimoto nearest neighbors retrieved)")

    # Atom explainability (Occlusion)
    t0 = time.time()
    r = c.post("/api/explain", json={"smiles": ERLOTINIB, "sequence": EGFR_SHORT, "endpoint": "Ki"})
    t_exp = time.time() - t0
    data = r.get_json()
    atoms = data.get("atoms", [])
    check("POST /api/explain", r.status_code == 200 and len(atoms) > 0,
          f"({len(atoms)} atom sensitivity scores in {t_exp:.2f}s, top atom: {data.get('top_atoms', [{}])[0]})")

    # 5. Multi-Compound & Selectivity
    print("\n--- 5. Comparison & Selectivity Endpoints ---")
    r = c.post("/api/compare", json={"smiles_list": [ERLOTINIB, GEFITINIB], "sequence": EGFR_SHORT})
    data = r.get_json()
    items = data.get("items", [])
    check("POST /api/compare", r.status_code == 200 and len(items) == 2, f"(Compared {len(items)} drugs against target)")

    r = c.post("/api/selectivity", json={"smiles": ERLOTINIB, "targets": [{"name": "EGFR_A", "sequence": EGFR_SHORT}, {"name": "EGFR_B", "sequence": EGFR_SHORT}]})
    data = r.get_json()
    rows = data.get("rows", [])
    check("POST /api/selectivity", r.status_code == 200 and len(rows) == 2, f"(Evaluated selectivity across {len(rows)} targets)")

    # 6. Batch & Virtual Screening (Async Jobs)
    print("\n--- 6. Batch & Virtual Screening (Jobs) ---")
    csv_payload = f"name,smiles\nErlotinib,{ERLOTINIB}\nGefitinib,{GEFITINIB}\n"
    r = c.post("/api/batch", data={"file": (io.BytesIO(csv_payload.encode("utf-8")), "batch.csv"), "sequence": EGFR_SHORT, "endpoints": "Ki"})
    job_id = r.get_json().get("job_id")
    check("POST /api/batch (job submission)", r.status_code == 200 and job_id is not None, f"(Job ID: {job_id})")

    # Poll job
    st = None
    for _ in range(40):
        time.sleep(0.5)
        st = c.get(f"/api/jobs/{job_id}").get_json()
        if st.get("state") in ("done", "error"):
            break
    check("GET /api/jobs/<job_id> (batch)", st and st.get("state") == "done", f"(Job completed with {len(st.get('result', {}).get('rows', []))} rows)")

    r = c.get(f"/api/download/{job_id}.csv")
    check("GET /api/download/<job_id>.csv", r.status_code == 200 and "Ki_p" in r.get_data(as_text=True), "(CSV download verified)")

    # Virtual Screening
    r = c.post("/api/screen", data={"file": (io.BytesIO(csv_payload.encode("utf-8")), "screen.csv"), "sequence": EGFR_SHORT, "endpoint": "Ki", "top_n": 10})
    job_id = r.get_json().get("job_id")
    check("POST /api/screen (job submission)", r.status_code == 200 and job_id is not None, f"(Job ID: {job_id})")

    st = None
    for _ in range(40):
        time.sleep(0.5)
        st = c.get(f"/api/jobs/{job_id}").get_json()
        if st.get("state") in ("done", "error"):
            break
    check("GET /api/jobs/<job_id> (screen)", st and st.get("state") == "done", f"(Screening completed with {len(st.get('result', {}).get('rows', []))} ranked candidates)")

    print("\n" + "=" * 65)
    if failures == 0:
        print(f"ALL {total} ENDPOINTS & TESTS PASSED PERFECTLY (100% SUCCESS)!")
    else:
        print(f"COMPLETED with {failures} failures out of {total} checks.")
    print("=" * 65)

if __name__ == "__main__":
    test_all()
