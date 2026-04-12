"""
backbone.py — GIN backbone with layer-wise UTS tracking (Section 4.4).

GINEncoder produces:
  - final node embeddings H^(L)
  - per-layer EmbeddingUTS vectors S^(0..L)  (for analysis + regularization)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GINConv, BatchNorm
from torch_geometric.utils import to_networkx

from diff_uts import DifferentiableEmbeddingUTS


# ---------------------------------------------------------------------------
# MLP helper used inside GINConv
# ---------------------------------------------------------------------------

def _make_mlp(in_dim: int, out_dim: int, hidden: int = None) -> nn.Sequential:
    hidden = hidden or out_dim * 2
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.BatchNorm1d(hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


# ---------------------------------------------------------------------------
# GINEncoder
# ---------------------------------------------------------------------------

class GINEncoder(nn.Module):
    """
    Multi-layer GIN encoder.

    Args:
        in_dim      : input node feature dimension
        hidden_dim  : hidden / output embedding dimension
        num_layers  : number of GIN layers
        dropout     : dropout probability
        track_uts   : if True, compute EmbeddingUTS at every layer (4.4)
        uts_kwargs  : kwargs forwarded to EmbeddingUTS
    """

    def __init__(self,
                 in_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 4,
                 dropout: float = 0.3,
                 track_uts: bool = True,
                 uts_kwargs: dict = None):
        super().__init__()

        self.num_layers = num_layers
        self.track_uts  = track_uts
        self.dropout    = dropout

        uts_kwargs = uts_kwargs or {}
        # DifferentiableEmbeddingUTS — no detach, gradients flow back through
        # uts_list into the GIN encoder for LayerSmoothLoss (§4.2)
        self.uts_computer = DifferentiableEmbeddingUTS(**uts_kwargs) if track_uts else None

        # --- GIN layers ---------------------------------------------------
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        dims = [in_dim] + [hidden_dim] * num_layers
        for i in range(num_layers):
            mlp = _make_mlp(dims[i], dims[i + 1])
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(BatchNorm(dims[i + 1]))

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor,
                batch: torch.Tensor):
        """
        Returns:
            H        : (N, hidden_dim) final node embeddings
            uts_list : list of (num_graphs, EmbeddingUTS.DIM) tensors,
                       one per layer. Empty list if track_uts=False.
        """
        H = x
        uts_list = []

        # Cora-style single-graph datasets set batch=None
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            H = conv(H, edge_index)
            H = bn(H)
            H = F.relu(H)
            H = F.dropout(H, p=self.dropout, training=self.training)

            if self.track_uts:
                # No detach — DifferentiableEmbeddingUTS stays in computation
                # graph so LayerSmoothLoss gradients flow back into the encoder
                sig = self.uts_computer.compute_batch(H, batch)
                uts_list.append(sig)           # (num_graphs, DIM)

        return H, uts_list
