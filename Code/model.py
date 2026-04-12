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

        # track_uts only when LayerSmoothLoss is active — computing
        # DifferentiableEmbeddingUTS every layer is expensive and wasted
        # for variants that don't use uts_list (V0, V1a, V1b, V1c)
        self._track_uts = lambda_smooth > 0

        # --- GIN backbone -------------------------------------------------
        self.encoder = GINEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            track_uts=self._track_uts,
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
    def forward(self, data, nx_graphs=None, graph_uts_cache=None):
        """
        Args:
            data            : PyG Data / Batch object
            nx_graphs       : list of nx.Graph — used for TopoRegLoss and
                              GraphUTS descriptor if cache not provided
            graph_uts_cache : dict {graph_idx: np.ndarray (DIM,)} —
                              precomputed GraphUTS vectors. When provided,
                              used instead of recomputing from nx_graphs.
                              Eliminates per-batch GraphUTS computation cost.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 1. GIN encoding ------------------------------------------------
        H, uts_list = self.encoder(x, edge_index, batch)

        # 2. Optional topology-guided pooling (4.3) ----------------------
        if self.use_toppool:
            H, edge_index, batch, _ = self.toppool(H, edge_index, batch)

        # 3. Graph readout (sum + mean + max) ----------------------------
        z_struct = _readout(H, batch)
        parts    = [z_struct]

        # 4a. EmbeddingUTS descriptor (§4.1) -----------------------------
        if self.use_embed_uts_descriptor:
            e_sig = self.embed_uts.compute_batch(H.detach(), batch)
            e_sig = e_sig.to(H.device)
            e_sig = torch.clamp(e_sig, -50.0, 50.0)
            e_sig = self.embed_uts_bn(e_sig)
            parts.append(e_sig)

        # 4b. GraphUTS descriptor (§4.1 extension) -----------------------
        if self.use_graph_uts_descriptor:
            g_vecs = self._get_graph_uts_vecs(
                data, nx_graphs, graph_uts_cache, H.device
            )
            g_sig = torch.clamp(g_vecs, -50.0, 50.0)
            g_sig = self.graph_uts_bn(g_sig)
            parts.append(g_sig)

        z      = torch.cat(parts, dim=-1)
        logits = self.classifier(z)

        # Auxiliary losses (§4.2) ----------------------------------------
        # LayerSmoothLoss uses uts_list from backbone (DifferentiableEmbUTS,
        # in-graph when track_uts=True i.e. lambda_smooth > 0)
        smooth_loss = self.layer_smooth_loss(uts_list)

        # TopoRegLoss uses DifferentiableEmbUTS on H internally (in-graph)
        # and GraphUTS from cache/nx_graphs (fixed constant, no grad needed)
        reg_loss = torch.tensor(0.0, device=H.device)
        if nx_graphs is not None or graph_uts_cache is not None:
            reg_loss = self.topo_reg_loss(
                H, batch, nx_graphs,
                graph_uts_cache=graph_uts_cache,
                graph_indices=self._batch_indices(data),
            )
        elif self.topo_reg_loss.lambda_reg > 0:
            # lambda_reg > 0 but no graph structure provided — warn once
            import warnings
            warnings.warn(
                "TopoRegLoss lambda_reg > 0 but no nx_graphs or "
                "graph_uts_cache provided. reg_loss will be zero. "
                "Pass graph_uts_cache to ablation or set use_reg=True.",
                RuntimeWarning, stacklevel=2
            )

        aux = {
            "smooth_loss": smooth_loss,
            "reg_loss":    reg_loss,
            "uts_list":    uts_list,
            "z_struct":    z_struct,
        }
        return logits, aux

    # ------------------------------------------------------------------
    def _batch_indices(self, data) -> list:
        """Extract per-graph dataset indices from a PyG batch."""
        if hasattr(data, "idx") and data.idx is not None:
            idx = data.idx
            if idx.dim() == 0:
                return [idx.item()]
            return idx.tolist()
        return None

    def _get_graph_uts_vecs(self, data, nx_graphs, cache, device) -> torch.Tensor:
        """
        Return (B, GraphUTS.DIM) tensor of GraphUTS vectors.
        Priority: cache lookup → nx_graphs recompute → zeros fallback.
        """
        num_graphs = int(data.batch.max().item()) + 1 \
                     if data.batch is not None else 1

        # --- cache lookup (fast path) ------------------------------------
        if cache is not None:
            indices = self._batch_indices(data)
            if indices is not None and len(indices) == num_graphs:
                g_vecs = []
                for i in indices:
                    vec = cache.get(int(i),
                                    np.zeros(GraphUTS.DIM, dtype=np.float32))
                    g_vecs.append(vec)
                return torch.tensor(
                    np.stack(g_vecs), dtype=torch.float32, device=device
                )

        # --- nx_graphs fallback (slow path) ------------------------------
        if nx_graphs is not None:
            g_vecs = []
            for g in nx_graphs:
                try:
                    g_vecs.append(self.graph_uts.safe_compute(g))
                except Exception:
                    g_vecs.append(np.zeros(GraphUTS.DIM, dtype=np.float32))
            return torch.tensor(
                np.stack(g_vecs), dtype=torch.float32, device=device
            )

        raise ValueError(
            "Either graph_uts_cache (with data.idx) or nx_graphs must be "
            "provided when use_graph_uts_descriptor=True"
        )


# ---------------------------------------------------------------------------
# Node Classifier
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Node Classifier
# ---------------------------------------------------------------------------

class UTSNodeClassifier(nn.Module):
    """
    Node classification model.

    Integrates:
      §4.1 (node) — local neighbourhood UTS concatenated to node embeddings
      §4.2        — TopologicalRegLoss + LayerSmoothLoss
      §4.4        — layer-wise UTS tracking

    Args:
        in_dim                  : input node feature dimension
        hidden_dim              : GIN hidden dimension
        num_classes             : number of node classes
        num_layers              : GIN depth
        dropout                 : dropout rate
        use_local_uts           : concatenate local neighbourhood UTS (§4.1)
        local_uts_hops          : k-hop for neighbourhood (default 2)
        local_uts_max_nodes     : neighbourhood cap for TDA (default 30)
        lambda_reg              : TopoReg coefficient
        lambda_smooth           : LayerSmooth coefficient
        large_graph             : use chunked encoder for large graphs (Cora)
    """

    def __init__(self,
                 in_dim: int,
                 hidden_dim: int = 128,
                 num_classes: int = 7,
                 num_layers: int = 2,
                 dropout: float = 0.3,
                 use_local_uts: bool = False,
                 local_uts_hops: int = 2,
                 local_uts_max_nodes: int = 30,
                 lambda_reg: float = 0.0,
                 lambda_smooth: float = 0.0,
                 large_graph: bool = True):
        super().__init__()

        self.use_local_uts = use_local_uts

        self.encoder = GINEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            track_uts=lambda_smooth > 0,
        )

        # Local UTS encoder — §4.1 for nodes
        if use_local_uts:
            from node_uts import LocalUTSNodeEncoder, LocalUTSNodeEncoderBatched
            if large_graph:
                self.local_uts = LocalUTSNodeEncoderBatched(
                    k=local_uts_hops,
                    max_nodes=local_uts_max_nodes,
                )
            else:
                self.local_uts = LocalUTSNodeEncoder(
                    k=local_uts_hops,
                    max_nodes=local_uts_max_nodes,
                )
            # BatchNorm to normalise local UTS features across nodes
            from node_uts import LocalUTSNodeEncoder as _L
            self.local_uts_bn = nn.BatchNorm1d(_L.UTS_DIM)
            classifier_in_dim = hidden_dim + _L.UTS_DIM   # 128 + 14
        else:
            classifier_in_dim = hidden_dim

        self.node_classifier = nn.Sequential(
            nn.Linear(classifier_in_dim, hidden_dim),
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

        # 1. GIN encoding
        H, uts_list = self.encoder(x, edge_index, batch)

        # 2. Local UTS descriptor per node (§4.1 for nodes)
        if self.use_local_uts:
            local_sig = self.local_uts(H, edge_index)   # (N, 14), in graph
            local_sig = torch.clamp(local_sig, -50.0, 50.0)
            local_sig = self.local_uts_bn(local_sig)
            H_aug     = torch.cat([H, local_sig], dim=-1)   # (N, 142)
        else:
            H_aug = H

        # 3. Per-node classification
        logits = self.node_classifier(H_aug)

        # 4. Auxiliary losses
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
            track_uts=True,   # contrastive always needs uts_list for TopoContrastLoss
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
