"""SMILES validation (RDKit only - no torch needed)."""
import config


def validate_smiles(smiles):
    """-> (ok, message, canonical_smiles_or_None).  Never raises."""
    if not isinstance(smiles, str) or not smiles.strip():
        return False, "SMILES is empty.", None
    smiles = smiles.strip()
    if len(smiles) > config.MAX_SMILES_LEN:
        return False, f"SMILES is longer than {config.MAX_SMILES_LEN} characters.", None
    try:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.error")
        mol = Chem.MolFromSmiles(smiles)
    except Exception:                                           # pragma: no cover
        mol = None
    if mol is None:
        return False, "RDKit could not parse this SMILES.", None
    if mol.GetNumAtoms() == 0:
        return False, "The molecule has no atoms.", None
    return True, "ok", Chem.MolToSmiles(mol)
