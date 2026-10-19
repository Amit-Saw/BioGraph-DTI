"""Optional docking tab: AutoDock Vina + 3D view.  Everything is imported lazily, so the app runs
without Vina/Meeko/Open Babel; `docking_available()` tells the UI whether the tab can be used.
A Vina score (kcal/mol) is NOT a pKi/pIC50 and must not be compared with the model's predictions."""
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import requests

import config
from services.errors import UserError

_PDB_ID = re.compile(r"^[0-9][A-Za-z0-9]{3}$")
_LIG_CODE = re.compile(r"^[A-Za-z0-9]{1,5}$")
_IGNORE = {"HOH", "DOD", "WAT"}


# ------------------------------------------------------------------ availability
def _has_pybel():
    try:
        from openbabel import pybel  # noqa: F401
        return True
    except ImportError:
        return False


def _vina_exe():
    if shutil.which("vina"):
        return shutil.which("vina")
    local = config.BASE_DIR / "vina.exe"
    if local.exists():
        return str(local)
    return None


def _has_vina():
    try:
        import vina  # noqa: F401
        return True
    except ImportError:
        return _vina_exe() is not None


def docking_available():
    """-> (ok, message)"""
    try:
        import meeko  # noqa: F401
    except ImportError:
        return False, "Meeko package missing: pip install meeko"
    if not _has_vina():
        return False, "AutoDock Vina missing: pip install vina or place vina.exe in dti_app/"
    if not (shutil.which("obabel") or _has_pybel()):
        return False, "Open Babel missing (pip install openbabel-wheel)"
    return True, "ok"


# ------------------------------------------------------------------ PDB handling
def _get(url, what):
    try:
        r = requests.get(url, timeout=30)
    except requests.RequestException:
        raise UserError(f"Could not reach RCSB to download {what}. Check the internet connection.", 502)
    if r.status_code == 404:
        raise UserError(f"RCSB has no {what}.", 404)
    if r.status_code != 200:
        raise UserError(f"RCSB returned HTTP {r.status_code} for {what}.", 502)
    return r.text


def fetch_pdb(pdb_id):
    pdb_id = (pdb_id or "").strip().upper()
    if not _PDB_ID.match(pdb_id):
        raise UserError("PDB ID must be 4 characters, e.g. 1M17.")
    path = config.DOCK_DIR / "pdb" / f"{pdb_id}.pdb"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_get(f"https://files.rcsb.org/download/{pdb_id}.pdb", f"entry {pdb_id}"))
    return pdb_id, path.read_text()


def _model1(lines):
    """Lines of the first MODEL only (NMR entries have many)."""
    out = []
    for l in lines:
        if l.startswith("ENDMDL"):
            break
        out.append(l)
    return out


def _altloc_ok(line):
    return line[16] in (" ", "A")


def list_ligands(pdb_id):
    """-> {pdb_id, ligands:[{code, chain, resi, atoms}], chains:[{chain, residues}]}"""
    pdb_id, text = fetch_pdb(pdb_id)
    lines = _model1(text.splitlines())
    ligs, chains = {}, {}
    for l in lines:
        if l.startswith("HETATM") and _altloc_ok(l):
            code = l[17:20].strip()
            if code in _IGNORE:
                continue
            key = (code, l[21], l[22:26].strip())
            ligs[key] = ligs.get(key, 0) + 1
        elif l.startswith("ATOM") and l[12:16].strip() == "CA":
            chains[l[21]] = chains.get(l[21], 0) + 1
    ligands = [{"code": c, "chain": ch, "resi": ri, "atoms": n}
               for (c, ch, ri), n in sorted(ligs.items(), key=lambda kv: -kv[1]) if n >= 6]
    return {"pdb_id": pdb_id, "ligands": ligands,
            "chains": [{"chain": c, "residues": n} for c, n in sorted(chains.items())]}


def _protein_pdb(lines, chains):
    keep = [l for l in lines if l.startswith("ATOM") and l[21] in chains and _altloc_ok(l)]
    if not keep:
        raise UserError("None of the selected chains contain protein atoms.")
    return "\n".join(keep) + "\nEND\n"


def _ligand_atoms(lines, code, chain, resi):
    atoms = [l for l in lines if l.startswith("HETATM") and l[17:20].strip() == code
             and l[21] == chain and l[22:26].strip() == str(resi) and _altloc_ok(l)]
    if not atoms:
        raise UserError(f"Ligand {code} (chain {chain}, residue {resi}) not found in this entry.")
    return atoms


def _xyz(line):
    return float(line[30:38]), float(line[38:46]), float(line[46:54])


# ------------------------------------------------------------------ preparation
def _to_receptor_pdbqt(pdb_path, pdbqt_path):
    exe = shutil.which("obabel")
    if exe:
        r = subprocess.run([exe, str(pdb_path), "-O", str(pdbqt_path), "-xr", "-p", "7.4"],
                           capture_output=True, text=True, timeout=300)
        if not Path(pdbqt_path).exists() or Path(pdbqt_path).stat().st_size == 0:
            raise UserError("Open Babel could not convert the receptor: " + r.stderr[-200:])
        return
    from openbabel import pybel                                    # fallback: python bindings
    mol = next(pybel.readfile("pdb", str(pdb_path)))
    mol.addh()
    mol.write("pdbqt", str(pdbqt_path), opt={"r": None}, overwrite=True)


def _meeko_pdbqt(mol):
    from meeko import MoleculePreparation
    prep = MoleculePreparation()
    try:
        from meeko import PDBQTWriterLegacy                          # newer Meeko
        setups = prep.prepare(mol)
        text, ok, err = PDBQTWriterLegacy.write_string(setups[0])
        if not ok:
            raise UserError("Meeko could not prepare the ligand: " + str(err))
        return text
    except ImportError:                                             # older Meeko
        prep.prepare(mol)
        return prep.write_pdbqt_string()


def _ligand_from_smiles(smiles):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3(); params.randomSeed = 42
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise UserError("Could not build a 3D structure for this SMILES.")
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        pass
    return _meeko_pdbqt(mol)


def _ligand_from_pdb_code(code):
    """Ideal 3D geometry of the PDB ligand (used for re-docking the crystal ligand)."""
    from rdkit import Chem
    if not _LIG_CODE.match(code):
        raise UserError("Invalid ligand code.")
    path = config.DOCK_DIR / "ligands" / f"{code}_ideal.sdf"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_get(f"https://files.rcsb.org/ligands/download/{code}_ideal.sdf", f"ligand {code}"))
    mol = Chem.MolFromMolFile(str(path), removeHs=False)
    if mol is None:
        raise UserError(f"Could not read the ideal structure of ligand {code}.")
    return _meeko_pdbqt(Chem.AddHs(mol, addCoords=True))


# ------------------------------------------------------------------ docking job
def dock_job(job, pdb_id, chains, ligand, smiles=None, box=None, exhaustiveness=None):
    """Runs inside a background job.  `ligand` = {code, chain, resi} marks the pocket (its centroid is the
    box centre).  If `smiles` is given that molecule is docked, otherwise the ligand itself (re-docking)."""
    ok, msg = docking_available()
    if not ok:
        raise UserError(msg, 503)

    box = min(float(box or config.DOCK_DEFAULT_BOX), config.DOCK_MAX_BOX)
    exh = min(int(exhaustiveness or config.DOCK_DEFAULT_EXHAUSTIVENESS), config.DOCK_MAX_EXHAUSTIVENESS)

    job.update(0.05, "Downloading structure")
    pdb_id, text = fetch_pdb(pdb_id)
    lines = _model1(text.splitlines())
    if not chains:
        chains = sorted({l[21] for l in lines if l.startswith("ATOM")})
    protein_text = _protein_pdb(lines, set(chains))
    lig_atoms = _ligand_atoms(lines, ligand["code"], ligand["chain"], ligand["resi"])
    center = np.mean([_xyz(l) for l in lig_atoms], axis=0)

    work = config.DOCK_DIR / job.id
    work.mkdir(parents=True, exist_ok=True)
    prot_pdb, rec_pdbqt = work / "protein.pdb", work / "protein.pdbqt"
    prot_pdb.write_text(protein_text)

    job.update(0.15, "Preparing protein")
    _to_receptor_pdbqt(prot_pdb, rec_pdbqt)

    job.update(0.30, "Preparing ligand")
    lig_pdbqt = _ligand_from_smiles(smiles) if smiles else _ligand_from_pdb_code(ligand["code"])

    job.update(0.40, "Docking (this can take a minute)")
    vina_bin = _vina_exe()
    try:
        from vina import Vina
        use_py_vina = True
    except ImportError:
        use_py_vina = False

    if use_py_vina:
        v = Vina(sf_name="vina", seed=42)
        v.set_receptor(str(rec_pdbqt))
        v.set_ligand_from_string(lig_pdbqt)
        v.compute_vina_maps(center=[float(c) for c in center], box_size=[box] * 3)
        v.dock(exhaustiveness=exh, n_poses=5)
        energies = [float(e[0]) for e in v.energies(n_poses=5)]
        pose = v.poses(n_poses=1)
    elif vina_bin:
        lig_path = work / "ligand.pdbqt"
        lig_path.write_text(lig_pdbqt)
        out_path = work / "docked_poses.pdbqt"
        cmd = [
            vina_bin,
            "--receptor", str(rec_pdbqt),
            "--ligand", str(lig_path),
            "--center_x", f"{center[0]:.3f}",
            "--center_y", f"{center[1]:.3f}",
            "--center_z", f"{center[2]:.3f}",
            "--size_x", f"{box:.1f}",
            "--size_y", f"{box:.1f}",
            "--size_z", f"{box:.1f}",
            "--exhaustiveness", str(exh),
            "--num_modes", "5",
            "--out", str(out_path),
            "--seed", "42"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0 or not out_path.exists():
            raise UserError("AutoDock Vina failed: " + (res.stderr or res.stdout))

        out_text = out_path.read_text()
        energies = []
        for line in out_text.splitlines():
            if line.startswith("REMARK VINA RESULT:"):
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        energies.append(float(parts[3]))
                    except ValueError:
                        pass
        if not energies:
            energies = [-7.0]

        # Extract first pose model
        pose_lines = []
        in_model = False
        for line in out_text.splitlines():
            if line.startswith("MODEL"):
                in_model = True
            if in_model:
                pose_lines.append(line)
                if line.startswith("ENDMDL"):
                    break
        pose = "\n".join(pose_lines) if pose_lines else out_text
    else:
        raise UserError("AutoDock Vina executable or python package was not found.", 503)

    job.update(0.95, "Analysing contacts")
    pose_xyz = np.array([_xyz(l) for l in pose.splitlines()
                         if l.startswith(("ATOM", "HETATM")) and not l[77:79].strip().upper().startswith("H")])
    prot_lines = [l for l in protein_text.splitlines() if l.startswith("ATOM") and l[76:78].strip() != "H"]
    prot_xyz = np.array([_xyz(l) for l in prot_lines])
    contacts = []
    if len(pose_xyz) and len(prot_xyz):
        d = np.linalg.norm(prot_xyz[:, None, :] - pose_xyz[None, :, :], axis=2).min(axis=1)
        seen = set()
        for l, dist in zip(prot_lines, d):
            key = (l[21], l[22:26].strip(), l[17:20].strip())
            if dist < 4.0 and key not in seen:
                seen.add(key)
                contacts.append({"chain": key[0], "resi": int(key[1]), "name": key[2]})
    contacts.sort(key=lambda c: (c["chain"], c["resi"]))

    return {"pdb_id": pdb_id, "score": energies[0], "scores": energies, "contacts": contacts,
            "protein_pdb": protein_text, "pose_pdbqt": pose,
            "crystal_pdb": "\n".join(lig_atoms) + "\nEND\n",
            "center": [float(c) for c in center], "box": box,
            "docked": "your SMILES" if smiles else f"re-docked crystal ligand {ligand['code']}"}
