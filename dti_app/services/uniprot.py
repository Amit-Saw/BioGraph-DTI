"""UniProt accession -> (name, sequence) with a small on-disk cache."""
import json
import re
import threading

import requests

import config
from services.errors import UserError

# official UniProt accession pattern (6 or 10 characters)
ACC_RE = re.compile(r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})$")
_lock = threading.Lock()


def _cache_file():
    return config.CACHE_DIR / "uniprot.json"


def looks_like_accession(text):
    return bool(ACC_RE.match((text or "").strip().upper()))


def _read_cache():
    try:
        return json.loads(_cache_file().read_text())
    except Exception:
        return {}


def parse_fasta(text):
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if not lines or not lines[0].startswith(">"):
        raise UserError("Unexpected response from UniProt.", 502)
    header, seq = lines[0], "".join(lines[1:])
    m = re.match(r">\w+\|[^|]+\|\S+\s+(.*?)(?:\s+OS=|$)", header)
    return (m.group(1) if m else header.lstrip(">")), seq


def fetch_sequence(accession, timeout=12):
    """-> (name, sequence).  Raises UserError with a readable message."""
    acc = (accession or "").strip().upper()
    if not looks_like_accession(acc):
        raise UserError(f"'{accession}' is not a valid UniProt accession (example: P00533).")
    with _lock:
        hit = _read_cache().get(acc)
    if hit:
        return hit["name"], hit["sequence"]
    try:
        r = requests.get(f"https://rest.uniprot.org/uniprotkb/{acc}.fasta", timeout=timeout,
                         headers={"User-Agent": "dti-app"})
    except requests.RequestException:
        raise UserError("Could not reach UniProt. Paste the sequence instead, or check the internet connection.", 502)
    if r.status_code == 404 or r.status_code == 400:
        raise UserError(f"UniProt has no entry for {acc}.", 404)
    if r.status_code != 200:
        raise UserError(f"UniProt returned HTTP {r.status_code}.", 502)
    name, seq = parse_fasta(r.text)
    with _lock:
        cache = _read_cache()
        cache[acc] = {"name": name, "sequence": seq}
        try:
            config.CACHE_DIR.mkdir(exist_ok=True)
            _cache_file().write_text(json.dumps(cache))
        except OSError:
            pass
    return name, seq
