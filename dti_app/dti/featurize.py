"""SMILES -> molecular graph.  mol_graph() is a VERBATIM copy of the training function
(dti_pipeline_v3.py).  Do not change feature definitions: the trained weights depend on them."""
from functools import lru_cache

import torch
from rdkit import Chem, RDLogger
from torch_geometric.data import Data

from dti.smiles import validate_smiles          # noqa: F401  (re-exported)

RDLogger.DisableLog("rdApp.error")

ATOM_DIM, BOND_DIM = 38, 7


def onehot(v, choices):
    return [int(v == c) for c in choices] + [int(v not in choices)]


@lru_cache(maxsize=30_000)
def mol_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    x = [onehot(a.GetAtomicNum(), [1, 6, 7, 8, 9, 15, 16, 17, 35, 53])
         + onehot(a.GetDegree(), [0, 1, 2, 3, 4, 5])
         + onehot(a.GetFormalCharge(), [-2, -1, 0, 1, 2])
         + onehot(str(a.GetHybridization()), ["SP", "SP2", "SP3", "SP3D", "SP3D2"])
         + onehot(a.GetTotalNumHs(), [0, 1, 2, 3, 4])
         + [int(a.GetIsAromatic()), int(a.IsInRing())] for a in mol.GetAtoms()]       # 38
    ei, ea = [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        f = onehot(str(b.GetBondType()), ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]) \
            + [int(b.GetIsConjugated()), int(b.IsInRing())]                           # 7
        ei += [[i, j], [j, i]]; ea += [f, f]
    if not ei:                                   # single-atom molecule: add a zero-feature self loop
        ei, ea = [[0, 0]], [[0] * BOND_DIM]
    return (torch.tensor(x, dtype=torch.float),
            torch.tensor(ei, dtype=torch.long).t().contiguous(),
            torch.tensor(ea, dtype=torch.float))


def smiles_to_data(smiles, protein_emb):
    """Graph + protein embedding (numpy float32, shape (320,)) as a PyG Data object."""
    x, ei, ea = mol_graph(smiles)
    return Data(x=x, edge_index=ei, edge_attr=ea,
                protein_emb=torch.as_tensor(protein_emb, dtype=torch.float32).unsqueeze(0))


def data_without_atom(data, atom_idx):
    """Copy of `data` with one atom (node) and its bonds removed - used for occlusion explanations."""
    n = data.x.shape[0]
    keep = torch.ones(n, dtype=torch.bool); keep[atom_idx] = False
    new_index = torch.cumsum(keep.long(), 0) - 1
    ei, ea = data.edge_index, data.edge_attr
    mask = keep[ei[0]] & keep[ei[1]]
    ei2 = new_index[ei[:, mask]]
    ea2 = ea[mask]
    if ei2.shape[1] == 0:                        # nothing left to connect: zero-feature self loop
        ei2 = torch.zeros((2, 1), dtype=torch.long)
        ea2 = torch.zeros((1, ea.shape[1]))
    return Data(x=data.x[keep], edge_index=ei2, edge_attr=ea2, protein_emb=data.protein_emb)
