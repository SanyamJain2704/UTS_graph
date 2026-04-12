"""
losses.py — UTS-based loss functions.

  TopoRegLoss       : Section 4.2 — regularize H^(L) to match GraphUTS anchor
  LayerSmoothLoss   : Section 4.2 — smooth topological evolution across layers
  TopoContrastLoss  : Section 4.5 — topology-preserving contrastive objective
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from uts      import EmbeddingUTS, GraphUTS
from diff_uts import DifferentiableEmbeddingUTS

# GraphUTS (GraphSignature-based) is always a fixed constant computed from the
# input graph topology — it never needs to be differentiable in any loss because
# it is not a function of model parameters.
#
# EmbeddingUTS (GUDHI-based) is used only in §4.1 descriptor (detached, no grad).
#
# DifferentiableEmbeddingUTS is used in all loss functions where gradients must
# flow back through the topological signature to the GNN encoder.


# ---------------------------------------------------------------------------
# 4.2a — Topo Regularization: embedding UTS ≈ graph structural UTS
# ---------------------------------------------------------------------------

class TopoRegLoss(nn.Module):
    """
    Encourages the learned embedding space to preserve the structural
    topology of the input graph.

        L_topo = || UTS(H^(L)) - UTS(G) ||^2_2

    Args:
        lambda_reg : weighting coefficient λ
    """

    def __init__(self, lambda_reg: float = 0.01, use_ricci: bool = False):
        super().__init__()
        self.lambda_reg  = lambda_reg
        self.graph_uts   = GraphUTS(use_ricci=use_ricci, use_persistence=True)
        self.diff_uts    = DifferentiableEmbeddingUTS()   # replaces EmbeddingUTS
        # Learnable projection: GraphUTS.DIM → DifferentiableEmbeddingUTS.DIM
        self.align = nn.Linear(GraphUTS.DIM, DifferentiableEmbeddingUTS.DIM)

    def forward(self,
                H: torch.Tensor,
                batch: torch.Tensor,
                nx_graphs: list = None,
                graph_uts_cache: dict = None,
                graph_indices: list = None) -> torch.Tensor:
        """
        Args:
            H               : (total_N, d) final node embeddings
            batch           : (total_N,) graph index
            nx_graphs       : list of nx.Graph — used if cache not provided
            graph_uts_cache : dict {dataset_idx: np.ndarray} precomputed vectors
            graph_indices   : list of dataset indices for graphs in this batch
        Returns:
            scalar loss
        """
        device     = H.device
        num_graphs = int(batch.max().item()) + 1 if batch is not None else 1
        total_loss = torch.tensor(0.0, device=device)

        for g in range(num_graphs):
            mask = (batch == g)
            H_g  = H[mask]

            # Embedding-side — DifferentiableEmbeddingUTS, no detach
            e_uts = self.diff_uts.compute(H_g)

            # Graph-side — use cache if available, else recompute
            if graph_uts_cache is not None and graph_indices is not None:
                idx   = graph_indices[g]
                g_uts = graph_uts_cache.get(
                    idx, np.zeros(GraphUTS.DIM, dtype=np.float32)
                )
            elif nx_graphs is not None:
                g_uts = self.graph_uts.safe_compute(nx_graphs[g])
            else:
                g_uts = np.zeros(GraphUTS.DIM, dtype=np.float32)

            g_uts_t    = torch.tensor(g_uts, dtype=torch.float32, device=device)
            g_uts_proj = self.align(g_uts_t)
            total_loss = total_loss + F.mse_loss(e_uts, g_uts_proj)

        return self.lambda_reg * total_loss / max(num_graphs, 1)


# ---------------------------------------------------------------------------
# 4.2b — Layer-smooth regularization
# ---------------------------------------------------------------------------

class LayerSmoothLoss(nn.Module):
    """
    Penalises abrupt topological changes between consecutive GNN layers.

        L_smooth = sum_l || S^(l) - S^(l-1) ||^2_2

    Args:
        lambda_smooth : weighting coefficient
    """

    def __init__(self, lambda_smooth: float = 0.005):
        super().__init__()
        self.lambda_smooth = lambda_smooth

    def forward(self, uts_list: list) -> torch.Tensor:
        """
        Args:
            uts_list : list of (num_graphs, UTS_DIM) tensors from GINEncoder
        Returns:
            scalar loss
        """
        if len(uts_list) < 2:
            return torch.tensor(0.0)

        device = uts_list[0].device
        loss   = torch.tensor(0.0, device=device)
        for l in range(1, len(uts_list)):
            loss = loss + F.mse_loss(uts_list[l].float(),
                                     uts_list[l - 1].float())
        return self.lambda_smooth * loss / (len(uts_list) - 1)


# ---------------------------------------------------------------------------
# 4.5 — Topology-Preserving Contrastive Loss
# ---------------------------------------------------------------------------

class TopoContrastLoss(nn.Module):
    """
    Given two augmented views of each graph, encourages their embedding
    spaces to share similar topological structure.

        L_topo = || UTS(H_G) - UTS(H_G~) ||^2_2

    Augmentation is handled externally; this loss only receives the two
    embedding matrices.

    Args:
        lambda_contrast : weighting coefficient
        temperature     : temperature for optional NT-Xent term
        use_ntxent      : if True, add standard contrastive NT-Xent loss
    """

    def __init__(self,
                 lambda_contrast: float = 0.1,
                 temperature: float = 0.5,
                 use_ntxent: bool = True):
        super().__init__()
        self.lambda_contrast = lambda_contrast
        self.temperature     = temperature
        self.use_ntxent      = use_ntxent
        self.diff_uts        = DifferentiableEmbeddingUTS()   # replaces EmbeddingUTS

    # ------------------------------------------------------------------
    def _ntxent(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """NT-Xent loss over graph-level representations."""
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        N  = z1.size(0)

        z   = torch.cat([z1, z2], dim=0)             # (2N, d)
        sim = torch.mm(z, z.t()) / self.temperature  # (2N, 2N)

        # Mask out self-similarity
        mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, -1e9)

        labels = torch.cat([
            torch.arange(N, 2 * N, device=z.device),
            torch.arange(0, N,     device=z.device),
        ])
        return F.cross_entropy(sim, labels)

    # ------------------------------------------------------------------
    def forward(self,
                H1: torch.Tensor, batch1: torch.Tensor,
                H2: torch.Tensor, batch2: torch.Tensor,
                z1: torch.Tensor = None,
                z2: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            H1, H2     : (N, d) node embeddings for view 1 and view 2
            batch1/2   : (N,) graph indices
            z1, z2     : (num_graphs, d) graph-level reps for NT-Xent (optional)
        Returns:
            scalar loss
        """
        num_graphs = int(batch1.max().item()) + 1
        device     = H1.device
        topo_loss  = torch.tensor(0.0, device=device)

        for g in range(num_graphs):
            m1 = (batch1 == g)
            m2 = (batch2 == g)
            # No detach — both views stay in computation graph
            s1 = self.diff_uts.compute(H1[m1])
            s2 = self.diff_uts.compute(H2[m2])
            topo_loss = topo_loss + F.mse_loss(s1, s2)

        topo_loss = topo_loss / max(num_graphs, 1)

        if self.use_ntxent and z1 is not None and z2 is not None:
            topo_loss = topo_loss + self._ntxent(z1, z2)

        return self.lambda_contrast * topo_loss
