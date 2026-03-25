"""
baseline.py — Pure GIN baseline. Zero UTS components.

Identical hyperparameters to UTSGraphClassifier so the comparison is fair:
  - Same GIN depth and hidden dim
  - Same readout (sum + mean + max)
  - Same classifier MLP structure
  - Same optimizer, scheduler, dropout

The only differences vs the full model:
  - No EmbeddingUTS concatenated to readout         (no §4.1)
  - No TopoRegLoss or LayerSmoothLoss               (no §4.2)
  - No UTSTopPool                                   (no §4.3)
  - No layer-wise UTS tracking                      (no §4.4)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GINConv, BatchNorm
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool


# ---------------------------------------------------------------------------
# MLP inside GINConv  (identical to backbone.py)
# ---------------------------------------------------------------------------

def _make_mlp(in_dim: int, out_dim: int) -> nn.Sequential:
    hidden = out_dim * 2
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.BatchNorm1d(hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


# ---------------------------------------------------------------------------
# BaselineGIN
# ---------------------------------------------------------------------------

class BaselineGIN(nn.Module):
    """
    Plain GIN graph classifier. No topology, no UTS, no pooling beyond readout.

    Args:
        in_dim      : input node feature dimension
        hidden_dim  : hidden / output embedding dimension  
        num_classes : number of output classes
        num_layers  : number of GIN layers
        dropout     : dropout rate
    """

    def __init__(self,
                 in_dim: int,
                 hidden_dim: int = 128,
                 num_classes: int = 2,
                 num_layers: int = 4,
                 dropout: float = 0.3):
        super().__init__()

        self.num_layers = num_layers
        self.dropout    = dropout

        # GIN layers — identical construction to GINEncoder in backbone.py
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        dims = [in_dim] + [hidden_dim] * num_layers
        for i in range(num_layers):
            mlp = _make_mlp(dims[i], dims[i + 1])
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(BatchNorm(dims[i + 1]))

        # Readout dim = 3 * hidden  (sum + mean + max, same as full model)
        # Full model readout_dim = 3*hidden + EmbeddingUTS.DIM(14)
        # Baseline readout_dim = 3*hidden only — this is the only structural diff
        readout_dim = 3 * hidden_dim

        self.classifier = nn.Sequential(
            nn.Linear(readout_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    # ------------------------------------------------------------------
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Readout: sum + mean + max
        z = torch.cat([
            global_add_pool(x, batch),
            global_mean_pool(x, batch),
            global_max_pool(x, batch),
        ], dim=-1)

        return self.classifier(z)


# ---------------------------------------------------------------------------
# Training helpers (self-contained, no imports from train.py)
# ---------------------------------------------------------------------------

def train_baseline(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        logits = model(data)
        loss   = F.cross_entropy(logits, data.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate_baseline(model, loader, device):
    model.eval()
    correct = total = 0
    for data in loader:
        data    = data.to(device)
        logits  = model(data)
        pred    = logits.argmax(dim=-1)
        correct += (pred == data.y).sum().item()
        total   += data.y.size(0)
    return correct / max(total, 1)