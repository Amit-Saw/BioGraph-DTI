"""Model definitions - VERBATIM copy of dti_pipeline_v3.py (section 4).
Architecture, layer sizes and constants must match training or checkpoints will not load."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, NNConv, Set2Set, global_mean_pool
from torch_geometric.nn.models import AttentiveFP

ATOM_DIM, BOND_DIM, PROT_DIM, HIDDEN = 38, 7, 320, 128
DROPOUT = 0.2


class GINEEncoder(nn.Module):
    """GINE: GIN message passing with bond features, 3 residual layers, mean pooling."""
    def __init__(self, layers=3):
        super().__init__()
        self.atom_enc = nn.Linear(ATOM_DIM, HIDDEN)
        self.convs = nn.ModuleList([
            GINEConv(nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN)),
                     edge_dim=BOND_DIM) for _ in range(layers)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(HIDDEN) for _ in range(layers)])

    def forward(self, b):
        h = self.atom_enc(b.x)
        for conv, bn in zip(self.convs, self.bns):
            h = h + torch.relu(bn(conv(h, b.edge_index, b.edge_attr)))
        return global_mean_pool(h, b.batch)                        # (B, 128)


class MPNNEncoder(nn.Module):
    """Gilmer-style MPNN: edge-conditioned messages (NNConv) + GRU update, shared over
    T steps, Set2Set readout."""
    def __init__(self, steps=3, dim=64):
        super().__init__()
        self.steps = steps
        self.atom_enc = nn.Linear(ATOM_DIM, dim)
        edge_net = nn.Sequential(nn.Linear(BOND_DIM, 128), nn.ReLU(), nn.Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, edge_net, aggr="mean")
        self.gru = nn.GRU(dim, dim)
        self.readout = Set2Set(dim, processing_steps=3)           # output 2*dim = 128

    def forward(self, b):
        out = F.relu(self.atom_enc(b.x))
        h = out.unsqueeze(0)
        for _ in range(self.steps):
            m = F.relu(self.conv(out, b.edge_index, b.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)
        return self.readout(out, b.batch)                          # (B, 128)


class AttentiveFPEncoder(nn.Module):
    """AttentiveFP (Xiong et al. 2020)."""
    def __init__(self):
        super().__init__()
        self.net = AttentiveFP(in_channels=ATOM_DIM, hidden_channels=HIDDEN, out_channels=HIDDEN,
                               edge_dim=BOND_DIM, num_layers=3, num_timesteps=2, dropout=DROPOUT)

    def forward(self, b):
        return self.net(b.x, b.edge_index, b.edge_attr, b.batch)  # (B, 128)


ENCODERS = {"mpnn_esm": MPNNEncoder, "gine_esm": GINEEncoder, "attentivefp_esm": AttentiveFPEncoder}


class DTIModel(nn.Module):
    """Drug graph encoder + ESM-2 protein embedding -> regression head (pKi or pIC50)."""
    def __init__(self, name):
        super().__init__()
        self.drug_enc = ENCODERS[name]()
        self.prot_enc = nn.Sequential(nn.LayerNorm(PROT_DIM), nn.Linear(PROT_DIM, 256), nn.ReLU(),
                                      nn.Dropout(DROPOUT), nn.Linear(256, HIDDEN), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(2 * HIDDEN, 256), nn.ReLU(), nn.Dropout(DROPOUT),
                                  nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, b):
        z = torch.cat([self.drug_enc(b), self.prot_enc(b.protein_emb)], dim=1)
        return self.head(z).squeeze(-1)
