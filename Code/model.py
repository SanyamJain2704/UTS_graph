"""
model.py — Full UTS-GNN models.

  UTSGraphClassifier  : graph classification  (Sections 4.1, 4.2, 4.3)
  UTSNodeClassifier   : node classification   (Sections 4.2, 4.4)

Both models expose a `compute_uts_analysis()` method for Section 4.4
layer-wise analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from backbone import GINEncoder
from pooling  import UTSTopPool
from uts      import EmbeddingUTS, GraphUTS
from losses   import TopoRegLoss, LayerSmoothLoss


# ---------------------------------------------------------------------------
# Readout: concatenate sum + mean + max  (richer than any single pooling)
# ---------------------------------------------------------------------------

def _readout(H: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    return torch.cat([
        global_add_pool(H, batch),
        global_mean_pool(H, batch),
        global_max_pool(H, batch),
    ], dim=-1)                             # (B, 3 * hidden_dim)


# ---------------------------------------------------------------------------
# Graph Classifier
# ---------------------------------------------------------------------------

class UTSGraphClassifier(nn.Module):
    """
    Graph classification model integrating:
      4.1 — UTS concatenated to graph-level readout
      4.2 — Topological regularization losses
      4.3 — UTS-TopPool hierarchical pooling
      4.4 — Layer-wise UTS tracking (exposed via uts_list)

    Args:
        in_dim                   : input node feature dimension
        hidden_dim               : GIN hidden dimension
        num_classes              : number of output classes
        num_layers               : GIN depth
        dropout                  : dropout rate
        use_toppool              : whether to apply UTS-TopPool after GIN
        pool_ratio               : ratio for UTS-TopPool
        lambda_reg               : coefficient for TopoRegLoss
        lambda_smooth            : coefficient for LayerSmoothLoss
        use_embed_uts_descriptor : concatenate EmbeddingUTS(H) to readout (§4.1)
        use_graph_uts_descriptor : concatenate GraphUTS(G) to readout (§4.1 extension)

    Descriptor / loss combinations:
        use_embed_uts_descriptor=True,  use_reg=False  →  pure §4.1 (V1)
        use_graph_uts_descriptor=True,  use_reg=False  →  input topology as feature only
        use_embed_uts_descriptor=False, use_reg=True   →  §4.2 loss only, no descriptor
        use_embed_uts_descriptor=True,  use_reg=True   →  §4.1 + §4.2 (V2)
        both descriptors=True,          use_reg=True   →  full descriptor + §4.2
    """

    def __init__(self,
                 in_dim: int,
                 hidden_dim: int = 128,
                 num_classes: int = 2,
                 num_layers: int = 4,
                 dropout: float = 0.3,
                 use_toppool: bool = True,
                 pool_ratio: float = 0.5,
                 lambda_reg: float = 0.01,
                 lambda_smooth: float = 0.005,
                 use_embed_uts_descriptor: bool = True,
                 use_graph_uts_descriptor: bool = False,
                 use_ricci: bool = False):
        super().__init__()

        self.use_toppool              = use_toppool
        self.use_embed_uts_descriptor = use_embed_uts_descriptor
        self.use_graph_uts_descriptor = use_graph_uts_descriptor
        self.use_ricci                = use_ricci

        # --- GIN backbone (with UTS layer tracking) -----------------------
        self.encoder = GINEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            track_uts=True,
        )

        # --- Optional UTS-TopPool (4.3) -----------------------------------
        if use_toppool:
            self.toppool = UTSTopPool(in_dim=hidden_dim, ratio=pool_ratio)

        # --- Readout dim depends on which descriptors are active ----------
        readout_dim = 3 * hidden_dim
        if use_embed_uts_descriptor:
            readout_dim += EmbeddingUTS.DIM       # + 14
        if use_graph_uts_descriptor:
            readout_dim += GraphUTS.DIM            # + 27

        self.classifier = nn.Sequential(
            nn.Linear(readout_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        # --- Losses (4.2) -------------------------------------------------
        self.topo_reg_loss     = TopoRegLoss(lambda_reg=lambda_reg, use_ricci=use_ricci)
        self.layer_smooth_loss = LayerSmoothLoss(lambda_smooth=lambda_smooth)

        # --- EmbeddingUTS descriptor (only built if needed) ---------------
        if use_embed_uts_descriptor:
            self.embed_uts    = EmbeddingUTS()
            self.embed_uts_bn = nn.BatchNorm1d(EmbeddingUTS.DIM)

        # --- GraphUTS descriptor (only built if needed) -------------------
        if use_graph_uts_descriptor:
            self.graph_uts    = GraphUTS(use_ricci=use_ricci)
            self.graph_uts_bn = nn.BatchNorm1d(GraphUTS.DIM)

    # ------------------------------------------------------------------
    def forward(self, data, nx_graphs=None):
        """
        Args:
            data       : PyG Data / Batch object
            nx_graphs  : list of nx.Graph (needed for TopoRegLoss and
                         GraphUTS descriptor; can be None if neither is used)
        Returns:
            logits     : (B, num_classes)
            aux        : dict with loss terms and uts_list for analysis
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 1. GIN encoding (produces layer-wise UTS) ----------------------
        H, uts_list = self.encoder(x, edge_index, batch)

        # 2. Optional topology-guided pooling (4.3) ----------------------
        if self.use_toppool:
            H, edge_index, batch, _ = self.toppool(H, edge_index, batch)

        # 3. Graph readout (sum + mean + max) ----------------------------
        z_struct = _readout(H, batch)                        # (B, 3*hidden)
        parts    = [z_struct]

        # 4a. EmbeddingUTS descriptor (§4.1) -----------------------------
        if self.use_embed_uts_descriptor:
            e_sig = self.embed_uts.compute_batch(H.detach(), batch)
            e_sig = e_sig.to(H.device)
            e_sig = torch.clamp(e_sig, -50.0, 50.0)
            e_sig = self.embed_uts_bn(e_sig)                 # (B, 14)
            parts.append(e_sig)

        # 4b. GraphUTS descriptor (§4.1 extension) -----------------------
        if self.use_graph_uts_descriptor:
            if nx_graphs is None:
                raise ValueError(
                    "nx_graphs must be provided when use_graph_uts_descriptor=True"
                )
            g_vecs = []
            for g in nx_graphs:
                try:
                    g_vecs.append(self.graph_uts.safe_compute(g))
                except Exception:
                    g_vecs.append(np.zeros(GraphUTS.DIM, dtype=np.float32))
            g_sig = torch.tensor(
                np.stack(g_vecs), dtype=torch.float32, device=H.device
            )                                                # (B, 27)
            g_sig = torch.clamp(g_sig, -50.0, 50.0)
            g_sig = self.graph_uts_bn(g_sig)                # (B, 27)
            parts.append(g_sig)

        z = torch.cat(parts, dim=-1)                         # (B, readout_dim)

        # 5. Classification head -----------------------------------------
        logits = self.classifier(z)

        # 6. Auxiliary losses (4.2) --------------------------------------
        smooth_loss = self.layer_smooth_loss(uts_list)

        reg_loss = torch.tensor(0.0, device=H.device)
        if nx_graphs is not None:
            reg_loss = self.topo_reg_loss(H, batch, nx_graphs)

        aux = {
            "smooth_loss": smooth_loss,
            "reg_loss":    reg_loss,
            "uts_list":    uts_list,
            "z_struct":    z_struct,
        }

        return logits, aux


# ---------------------------------------------------------------------------
# Node Classifier
# ---------------------------------------------------------------------------

class UTSNodeClassifier(nn.Module):
    """
    Node classification model integrating:
      4.2 — Topological regularization
      4.4 — Layer-wise UTS tracking

    Args:
        in_dim        : input node feature dimension
        hidden_dim    : GIN hidden dimension
        num_classes   : number of node classes
        num_layers    : GIN depth
        dropout       : dropout rate
        lambda_reg    : TopoReg coefficient
        lambda_smooth : LayerSmooth coefficient
    """

    def __init__(self,
                 in_dim: int,
                 hidden_dim: int = 128,
                 num_classes: int = 7,
                 num_layers: int = 4,
                 dropout: float = 0.3,
                 lambda_reg: float = 0.01,
                 lambda_smooth: float = 0.005):
        super().__init__()

        self.encoder = GINEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            track_uts=True,
        )

        self.node_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self.layer_smooth_loss = LayerSmoothLoss(lambda_smooth=lambda_smooth)
        self.topo_reg_loss     = TopoRegLoss(lambda_reg=lambda_reg)

    # ------------------------------------------------------------------
    def forward(self, data, nx_graphs=None):
        """
        Returns:
            logits : (N, num_classes)
            aux    : dict with loss terms and uts_list
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        H, uts_list = self.encoder(x, edge_index, batch)
        logits      = self.node_classifier(H)

        smooth_loss = self.layer_smooth_loss(uts_list)
        reg_loss    = torch.tensor(0.0, device=H.device)
        if nx_graphs is not None:
            reg_loss = self.topo_reg_loss(H, batch, nx_graphs)

        aux = {
            "smooth_loss": smooth_loss,
            "reg_loss":    reg_loss,
            "uts_list":    uts_list,
        }
        return logits, aux


# ---------------------------------------------------------------------------
# Contrastive wrapper (4.5)
# ---------------------------------------------------------------------------

class UTSContrastiveModel(nn.Module):
    """
    Self-supervised pre-training wrapper using topology-preserving
    contrastive learning (Section 4.5).

    Uses the UTSGraphClassifier as encoder; the projection head maps
    graph representations to a contrastive latent space.
    """

    def __init__(self,
                 in_dim: int,
                 hidden_dim: int = 128,
                 proj_dim: int = 64,
                 num_layers: int = 4,
                 dropout: float = 0.3):
        super().__init__()

        self.encoder = GINEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            track_uts=True,
        )

        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        H, uts_list = self.encoder(x, edge_index, batch)

        from torch_geometric.nn import global_mean_pool
        z_graph = global_mean_pool(H, batch)   # (B, hidden)
        z_proj  = self.proj(z_graph)           # (B, proj_dim)

        return H, batch, z_proj, uts_list