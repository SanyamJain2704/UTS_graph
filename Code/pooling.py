"""
pooling.py — UTS-TopPool: Topology-Guided Pooling (Section 4.3).

Each node receives a local topological importance score derived from
the EmbeddingUTS of its k-hop ego-subgraph embeddings.  A differentiable
soft-selection over these scores drives cluster assignment.

Architecture:
  1. Local UTS score per node   → α_i = MLP(UTS(H_{N(v_i)}))
  2. Soft top-k selection        → S  ∈ R^{N × k_clusters}
  3. Coarsened graph via SRC     → (H', A', batch')
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_geometric.nn import global_add_pool

from diff_uts import DifferentiableEmbeddingUTS


# ---------------------------------------------------------------------------
# Local UTS scorer
# ---------------------------------------------------------------------------

class LocalUTSScorer(nn.Module):
    """
    For each node v, collects the embeddings of its 1-hop neighbourhood
    (including v itself) and computes a DifferentiableEmbeddingUTS vector.
    These vectors are projected to a scalar importance score.

    Uses DifferentiableEmbeddingUTS — no detach, gradients flow through
    the UTS computation into the scorer MLP and back to the encoder.
    """

    def __init__(self, uts_dim: int, hidden: int = 64):
        super().__init__()
        self.uts_computer = DifferentiableEmbeddingUTS(k_neighbors=4)
        self.scorer = nn.Sequential(
            nn.Linear(uts_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            scores : (N,) importance score per node
        """
        N      = x.size(0)
        device = x.device

        # Build adjacency list for neighbourhood lookup
        src, dst = edge_index[0].tolist(), edge_index[1].tolist()
        nbrs = {i: [] for i in range(N)}
        for s, d in zip(src, dst):
            nbrs[s].append(d)
            nbrs[d].append(s)

        uts_vecs = []
        for v in range(N):
            hood_idx = list(set([v] + nbrs[v]))
            H_local  = x[hood_idx]             # NO detach — gradient flows through
            if len(hood_idx) < 3:
                uts_vecs.append(
                    torch.zeros(DifferentiableEmbeddingUTS.DIM,
                                device=device)
                )
            else:
                uts_vecs.append(self.uts_computer.compute(H_local))

        uts_tensor = torch.stack(uts_vecs)              # (N, DIM) — in graph
        scores     = self.scorer(uts_tensor).squeeze(-1) # (N,)
        return scores


# ---------------------------------------------------------------------------
# UTS-TopPool
# ---------------------------------------------------------------------------

class UTSTopPool(nn.Module):
    """
    UTS-guided hierarchical pooling layer.

    Args:
        in_dim       : node embedding dimension
        ratio        : fraction of nodes to keep  (0 < ratio ≤ 1)
        num_clusters : target cluster count (overrides ratio if set)
    """

    def __init__(self,
                 in_dim: int,
                 ratio: float = 0.5,
                 num_clusters: int = None):
        super().__init__()
        self.ratio        = ratio
        self.num_clusters = num_clusters
        self.scorer       = LocalUTSScorer(uts_dim=DifferentiableEmbeddingUTS.DIM)

        # Learnable projection after pooling
        self.proj = nn.Linear(in_dim, in_dim)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor,
                batch: torch.Tensor):
        """
        Returns:
            x_pool      : (N', in_dim) coarsened node embeddings
            edge_index_pool : coarsened edge_index
            batch_pool  : (N',) batch vector for coarsened graph
            perm        : selected node indices (for skip connections)
        """
        scores = self.scorer(x, edge_index)         # (N,)
        scores = torch.sigmoid(scores)

        num_graphs = int(batch.max().item()) + 1
        keep_lists = []

        for g in range(num_graphs):
            mask = (batch == g)
            idx  = mask.nonzero(as_tuple=False).squeeze(1)
            n_g  = idx.size(0)

            k = (self.num_clusters
                 if self.num_clusters is not None
                 else max(1, int(n_g * self.ratio)))
            k = min(k, n_g)

            s_g   = scores[idx]
            topk  = torch.topk(s_g, k).indices
            keep_lists.append(idx[topk])

        perm       = torch.cat(keep_lists, dim=0)        # (N',)
        x_pool     = F.relu(self.proj(x[perm]))
        batch_pool = batch[perm]

        # Re-index edges: keep only edges where both endpoints are retained
        src_t, dst_t = edge_index[0], edge_index[1]
        src_mask = torch.isin(src_t, perm)
        dst_mask = torch.isin(dst_t, perm)
        keep_edges = src_mask & dst_mask

        new_src_t = src_t[keep_edges]
        new_dst_t = dst_t[keep_edges]

        # Remap to new indices
        perm_cpu = perm.cpu()
        remap = torch.full((x.size(0),), -1, dtype=torch.long)
        remap[perm_cpu] = torch.arange(perm.size(0))
        remap = remap.to(x.device)

        edge_index_pool = torch.stack([
            remap[new_src_t], remap[new_dst_t]
        ], dim=0)

        return x_pool, edge_index_pool, batch_pool, perm
