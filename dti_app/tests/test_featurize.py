import torch

from dti.featurize import (ATOM_DIM, BOND_DIM, data_without_atom, mol_graph,
                           smiles_to_data, validate_smiles)
import numpy as np

ERLOTINIB = "COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC"


def test_dimensions():
    x, ei, ea = mol_graph("CCO")
    assert x.shape == (3, ATOM_DIM) and ea.shape[1] == BOND_DIM
    assert ei.shape == (2, 4) and ea.shape[0] == 4          # 2 bonds, both directions


def test_single_atom_gets_self_loop():
    x, ei, ea = mol_graph("[Na+]")
    assert x.shape[0] == 1 and ei.shape == (2, 1) and float(ea.sum()) == 0


def test_validate():
    assert validate_smiles(ERLOTINIB)[0]
    assert not validate_smiles("")[0]
    assert not validate_smiles("not a smiles")[0]
    assert not validate_smiles("C" * 1000)[0]


def test_occlusion_graph():
    d = smiles_to_data("CCO", np.zeros(320, dtype=np.float32))
    d2 = data_without_atom(d, 1)
    assert d2.x.shape[0] == 2
    assert int(d2.edge_index.max()) < 2                      # indices re-numbered
