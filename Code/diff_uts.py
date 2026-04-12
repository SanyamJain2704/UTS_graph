"""
diff_uts.py — Differentiable Embedding UTS.

Used in all loss functions and scoring mechanisms where gradients must
flow back through the topological/geometric signature to the GNN encoder:

    §4.2  TopoRegLoss        — EmbeddingUTS(H^L) side only
    §4.2  LayerSmoothLoss    — S^(l) and S^(l-1) both
    §4.3  UTSTopPool scorer  — local neighbourhood UTS per node
    §4.5  TopoContrastLoss   — both augmented views

NOT used for:
    §4.1  Readout descriptor — uses EmbeddingUTS (GUDHI) + GraphUTS instead
                               (detached, full persistence features, no grad needed)
    §4.4  Layer-wise analysis — uses output of this class but detaches for analysis

GraphUTS (GraphSignature-based) never needs to be differentiable in any section
because it is always a fixed structural constant computed from the input graph,
never a function of the model parameters.

Output vector (14-dim, matches EmbeddingUTS.DIM):
    [h0_mean, h0_ent, h1_mean, h1_ent,       # persistence surrogates
     betti0, betti1,                           # soft Betti numbers
     mean_nn, std_nn,                          # local geometry
     global_spread, intrinsic_dim,             # global geometry
     eig1, eig2, eig3,                         # spectral (soft kNN Laplacian)
     spectral_entropy]                         # spectral entropy
"""

import torch
import torch.nn as nn


class DifferentiableEmbeddingUTS(nn.Module):
    """
    Fully differentiable topological and geometric signature for latent spaces.

    All operations stay in the PyTorch computation graph — no GUDHI, no numpy,
    no detach. Gradients flow back through every feature to the input embeddings H.

    H0/H1 persistence features are differentiable surrogates, not exact persistence:
      - H0: sorted upper-triangle pairwise distances (approx. MST-based lifetimes)
      - H1: triangle birth/death from sorted per-triangle edge lengths

    Args:
        k_neighbors   : number of nearest neighbours for local geometry + spectral
        max_n_for_h1  : cap on N for triangle enumeration (O(N³) memory).
                        Graphs larger than this get h1=0, betti1=0.
    """

    DIM = 14

    def __init__(self, k_neighbors: int = 5, max_n_for_h1: int = 50):
        super().__init__()
        self.k           = k_neighbors
        self.max_n_for_h1 = max_n_for_h1

    # ------------------------------------------------------------------
    def _entropy(self, values: torch.Tensor) -> torch.Tensor:
        """Differentiable entropy of a non-negative value distribution."""
        if values.numel() == 0:
            return torch.tensor(0.0, device=values.device)
        norm = values / (values.sum() + 1e-12)
        return -torch.sum(norm * torch.log(norm + 1e-12))

    # ------------------------------------------------------------------
    def compute(self, H: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H : (N, d) float tensor — node embeddings, WITH grad if used in loss.
        Returns:
            (DIM,) float tensor, fully connected to H in the computation graph.
        """
        N      = H.shape[0]
        device = H.device

        if N < 3:
            return torch.zeros(self.DIM, dtype=torch.float32, device=device)

        # ── 1. Pairwise distances ─────────────────────────────────────
        # Use manual sqrt(sum + eps) instead of torch.cdist to guarantee
        # stable gradients when two embeddings are identical (dist = 0).
        diff     = H.unsqueeze(1) - H.unsqueeze(0)          # (N, N, d)
        dist_mat = torch.sqrt((diff ** 2).sum(-1) + 1e-8)   # (N, N)

        # ── 2. Local geometry ─────────────────────────────────────────
        k = min(self.k, N - 1)
        k_dists, _ = torch.topk(dist_mat, k=k + 1, largest=False)
        nn_dists   = k_dists[:, 1:]         # exclude self-distance

        mean_nn      = nn_dists.mean()
        std_nn       = nn_dists.std()
        global_spread = dist_mat.max()

        # Intrinsic dimensionality estimate (log ratio of successive NN distances).
        # Uses .mean() not .median() — median has sparse gradients.
        if k >= 2:
            r1 = k_dists[:, 1]
            r2 = k_dists[:, 2]
            ratio        = torch.log(r2 + 1e-12) / (torch.log(r1 + 1e-12) + 1e-12)
            intrinsic_dim = ratio.mean()
        else:
            intrinsic_dim = torch.tensor(1.0, device=device)

        # ── 3. Persistence surrogates ─────────────────────────────────
        triu_idx     = torch.triu_indices(N, N, offset=1)
        edge_weights = dist_mat[triu_idx[0], triu_idx[1]]   # (E,)

        # H0 surrogate — sorted upper-triangle distances as component lifetimes.
        # Approximates MST-based persistence; not exact but differentiable.
        # gradient of sort is sparse but well-defined (flows to source entry).
        sorted_edges, _ = torch.sort(edge_weights)
        h0_lifetimes    = sorted_edges[:N - 1]

        h0_mean = h0_lifetimes.mean()
        h0_ent  = self._entropy(h0_lifetimes)

        # Soft Betti-0 — count of components (lifetimes above mean NN threshold)
        thresh  = mean_nn
        betti0  = torch.sum(torch.sigmoid((h0_lifetimes - thresh) * 10))

        # H1 surrogate — triangle birth/death approximation.
        # A 1-cycle is born at the 2nd-longest triangle edge (filtration entry)
        # and dies when the longest edge fills the triangle.
        # Capped at max_n_for_h1 to prevent O(N³) memory on large graphs.
        if N <= self.max_n_for_h1:
            tri_idx  = torch.combinations(torch.arange(N, device=device), r=3)
            tri_edges = torch.stack([
                dist_mat[tri_idx[:, 0], tri_idx[:, 1]],
                dist_mat[tri_idx[:, 1], tri_idx[:, 2]],
                dist_mat[tri_idx[:, 0], tri_idx[:, 2]],
            ], dim=1)                                        # (T, 3)

            sorted_tri, _ = torch.sort(tri_edges, dim=1)
            births        = sorted_tri[:, 1]                 # 2nd longest edge
            deaths        = sorted_tri[:, 2]                 # longest edge
            h1_lifetimes  = torch.clamp(deaths - births, min=0)

            h1_mean = h1_lifetimes.mean()
            h1_ent  = self._entropy(h1_lifetimes)
            betti1  = torch.sum(torch.sigmoid((h1_lifetimes - thresh) * 10))
        else:
            # Graph too large — skip H1 to avoid memory explosion
            h1_mean = torch.tensor(0.0, device=device)
            h1_ent  = torch.tensor(0.0, device=device)
            betti1  = torch.tensor(0.0, device=device)

        # ── 4. Spectral features (soft kNN Laplacian) ─────────────────
        # Gaussian kernel adjacency — sigma kept in graph (no detach)
        # so gradients flow back through spectral features to H.
        sigma = mean_nn + 1e-5
        adj   = torch.exp(-(dist_mat ** 2) / (2 * sigma ** 2))
        adj   = adj - torch.diag(torch.diag(adj))            # remove self-loops

        # Deterministic diagonal regularisation to prevent NaN gradients
        # from repeated eigenvalues (replaces random noise perturbation).
        adj = adj + torch.eye(N, device=device) * 1e-6

        deg     = adj.sum(dim=1)
        L       = torch.diag(deg) - adj                      # graph Laplacian

        # eigh is fully differentiable (symmetric matrix, real eigenvalues)
        eigvals, _ = torch.linalg.eigh(L)
        eigvals_c  = torch.clamp(eigvals, min=1e-12)

        eig1 = eigvals_c[1] if N > 1 else torch.tensor(0.0, device=device)
        eig2 = eigvals_c[2] if N > 2 else torch.tensor(0.0, device=device)
        eig3 = eigvals_c[3] if N > 3 else torch.tensor(0.0, device=device)

        spec_entropy = self._entropy(eigvals_c)

        # ── 5. Stack into 14-dim vector ───────────────────────────────
        return torch.stack([
            h0_mean, h0_ent, h1_mean, h1_ent,
            betti0,  betti1,
            mean_nn, std_nn, global_spread, intrinsic_dim,
            eig1,    eig2,   eig3,          spec_entropy,
        ])

    # ------------------------------------------------------------------
    def compute_batch(self, H: torch.Tensor,
                      batch: torch.Tensor) -> torch.Tensor:
        """
        Compute DifferentiableEmbeddingUTS for each graph in a PyG batch.

        Args:
            H     : (total_nodes, d) — node embeddings WITH grad.
            batch : (total_nodes,) graph index per node, or None for single graph.
        Returns:
            (num_graphs, DIM) float tensor, connected to H in computation graph.
        """
        if batch is None:
            return self.compute(H).unsqueeze(0)

        num_graphs = int(batch.max().item()) + 1
        sigs = []
        for g in range(num_graphs):
            H_g = H[batch == g]
            sigs.append(self.compute(H_g))
        return torch.stack(sigs)   # autograd handles per-graph gradients correctly
