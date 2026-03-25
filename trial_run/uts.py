"""
uts.py — Unified Topological Signature (UTS) computation.

Two modes:
  - EmbeddingUTS : TDA over a point cloud of node embeddings H^(l)   [Section 4.1, 4.3, 4.5]
  - GraphUTS     : Thin wrapper around GraphSignature (untouched)     [Section 4.2 anchor]
"""

import numpy as np
import torch
import gudhi as gd

import networkx as nx
from scipy.linalg import eigvalsh

from graph_signature import GraphSignature
import multiprocessing as mp


# ---------------------------------------------------------------------------
# Top-level worker for subprocess-isolated Ricci computation.
# Must be module-level (not nested) so multiprocessing can pickle it.
# ---------------------------------------------------------------------------

def _ricci_worker(nodes, edges, use_ricci, use_persistence, result_queue):
    try:
        import networkx as nx
        from graph_signature import GraphSignature
        import numpy as np
        G_local = nx.Graph()
        G_local.add_nodes_from(nodes)
        G_local.add_edges_from(edges)
        sig = GraphSignature(use_ricci=use_ricci, use_persistence=use_persistence)
        vec = sig.compute(G_local).astype(np.float32)
        result_queue.put(vec)
    except Exception:
        result_queue.put(None)


# ---------------------------------------------------------------------------
# Helpers (used by EmbeddingUTS only)
# ---------------------------------------------------------------------------

def _persistence_stats(diag, dim):
    """Extract mean lifetime and persistence entropy for a given homology dim."""
    lifetimes = [d - b for (d_dim, (b, d)) in diag
                 if d_dim == dim and d != float("inf")]
    if len(lifetimes) == 0:
        return 0.0, 0.0
    arr = np.array(lifetimes)
    norm = arr / (arr.sum() + 1e-12)
    ent = float(-np.sum(norm * np.log(norm + 1e-12)))
    return float(np.mean(arr)), ent


# ---------------------------------------------------------------------------
# EmbeddingUTS — TDA over H^(l) point cloud
# ---------------------------------------------------------------------------

class EmbeddingUTS:
    """
    Computes a topological + geometric signature from a matrix of node
    embeddings treated as a point cloud.

    Output vector (fixed dimension = 14):
        [h0_mean, h0_ent, h1_mean, h1_ent,       # persistent homology (β0, β1)
         betti0, betti1,                           # Betti numbers
         mean_nn_dist, std_nn_dist,                # local geometry
         global_spread, intrinsic_dim_est,         # global geometry
         eig1, eig2, eig3,                         # spectral (kNN graph)
         spectral_entropy]                         # spectral entropy
    """

    DIM = 14

    def __init__(self, k_neighbors: int = 5, max_homology_dim: int = 1):
        self.k = k_neighbors
        self.max_dim = max_homology_dim

    # ------------------------------------------------------------------
    def compute(self, H) -> np.ndarray:
        """
        Args:
            H : (N, d) float tensor or numpy array — node embeddings for one graph.
        Returns:
            np.ndarray of shape (DIM,)
        """
        if isinstance(H, torch.Tensor):
            H_np = H.detach().cpu().float().numpy()
        else:
            H_np = H.astype(np.float32)

        N = H_np.shape[0]

        if N < 3:
            return np.zeros(self.DIM, dtype=np.float32)

        # --- pairwise distances -------------------------------------------
        diff = H_np[:, None, :] - H_np[None, :, :]          # (N,N,d)
        dist_mat = np.sqrt((diff ** 2).sum(-1))              # (N,N)

        # --- nearest-neighbour geometry -----------------------------------
        k = min(self.k, N - 1)
        nn_dists = np.sort(dist_mat, axis=1)[:, 1:k + 1]    # exclude self
        mean_nn = float(nn_dists.mean())
        std_nn  = float(nn_dists.std())

        # --- global spread ------------------------------------------------
        global_spread = float(dist_mat.max())

        # --- intrinsic dimensionality estimate (correlation dim proxy) ----
        if k >= 2:
            r1 = np.sort(dist_mat, axis=1)[:, 1]
            r2 = np.sort(dist_mat, axis=1)[:, 2]
            ratio = np.log(r2 + 1e-12) / (np.log(r1 + 1e-12) + 1e-12)
            intrinsic_dim = float(np.median(ratio))
        else:
            intrinsic_dim = 1.0

        # --- persistent homology (Rips complex) ---------------------------
        max_edge = float(np.percentile(dist_mat[dist_mat > 0], 90))
        rips = gd.RipsComplex(distance_matrix=dist_mat, max_edge_length=max_edge)
        st   = rips.create_simplex_tree(max_dimension=self.max_dim)
        diag = st.persistence()

        h0_mean, h0_ent = _persistence_stats(diag, 0)
        h1_mean, h1_ent = _persistence_stats(diag, 1)

        betti0 = sum(1 for (d, (b, de)) in diag if d == 0 and de == float("inf"))
        betti1 = sum(1 for (d, (b, de)) in diag if d == 1 and de == float("inf"))

        # --- spectral features of kNN graph -------------------------------
        adj = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            nbrs = np.argsort(dist_mat[i])[1:k + 1]
            adj[i, nbrs] = 1.0
            adj[nbrs, i] = 1.0

        deg = adj.sum(axis=1)
        D   = np.diag(deg)
        L   = D - adj
        eigvals = np.sort(eigvalsh(L))
        eigvals_clipped = np.clip(eigvals, 1e-12, None)
        prob = eigvals_clipped / eigvals_clipped.sum()
        spec_entropy = float(-np.sum(prob * np.log(prob + 1e-12)))

        eig1 = float(eigvals[1]) if len(eigvals) > 1 else 0.0
        eig2 = float(eigvals[2]) if len(eigvals) > 2 else 0.0
        eig3 = float(eigvals[3]) if len(eigvals) > 3 else 0.0

        return np.array([
            h0_mean, h0_ent, h1_mean, h1_ent,
            float(betti0), float(betti1),
            mean_nn, std_nn,
            global_spread, intrinsic_dim,
            eig1, eig2, eig3,
            spec_entropy,
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    def compute_batch(self, H: torch.Tensor,
                      batch: torch.Tensor) -> torch.Tensor:
        """
        Compute EmbeddingUTS for each graph in a PyG batch.

        Args:
            H     : (total_nodes, d) — stacked node embeddings.
            batch : (total_nodes,)   — graph index per node.
        Returns:
            (num_graphs, DIM) float tensor.
        """
        # batch=None means single graph (e.g. Cora node classification)
        if batch is None:
            sig = self.compute(H)
            return torch.tensor(sig[None, :], dtype=torch.float32, device=H.device)

        num_graphs = int(batch.max().item()) + 1
        sigs = []
        for g in range(num_graphs):
            mask = (batch == g)
            H_g  = H[mask]
            sigs.append(self.compute(H_g))
        return torch.tensor(np.stack(sigs), dtype=torch.float32,
                            device=H.device)


# ---------------------------------------------------------------------------
# GraphUTS — thin wrapper around GraphSignature (DO NOT MODIFY INTERNALS)
# ---------------------------------------------------------------------------

class GraphUTS:
    """
    Delegates entirely to GraphSignature.compute() — no logic duplicated here.

    GraphSignature output layout (30 features total):
        ricci        (8)  : orc_mean, orc_var, orc_min, orc_neg_frac,
                            frc_mean, frc_var, frc_min, frc_max
        distance     (2)  : mean_path_length, diameter
        spectral     (3)  : spectral_gap, max_eig, spectral_entropy
        persistence  (4)  : h0_mean, h0_ent, h1_mean, h1_ent
        degree       (3)  : mean_deg, var_deg, max_deg
        clustering   (2)  : mean_clust, var_clust
        centrality   (3)  : mean_bet, var_bet, mean_clo
        connectivity (2)  : num_components, largest_cc_ratio
                           ────────────────────────────────
                     27   total  (ricci disabled = 19)

    Used as the structural anchor in TopoRegLoss (Section 4.2).
    """

    # Full output when use_ricci=True.  If use_ricci=False, GraphSignature
    # returns [0]*8 for those slots so the length stays the same either way.
    DIM = 27

    def __init__(self, use_ricci: bool = True, use_persistence: bool = True):
        self._sig = GraphSignature(
            use_ricci=use_ricci,
            use_persistence=use_persistence,
        )

    def compute(self, G: nx.Graph) -> np.ndarray:
        """Returns np.ndarray of shape (DIM,)."""
        return self._sig.compute(G).astype(np.float32)

    def safe_compute(self, G: nx.Graph) -> np.ndarray:
        """
        Computes GraphSignature in an isolated subprocess so that a segfault
        inside GraphRicciCurvature's C++ code kills the child process only,
        not the main training process.

        Falls back to zeros on any failure (segfault, timeout, exception).
        """
        if G.number_of_nodes() < 3 or G.number_of_edges() < 1:
            return np.zeros(self.DIM, dtype=np.float32)

        nodes     = list(G.nodes())
        edges     = list(G.edges())
        use_ricci = self._sig.use_ricci
        use_pers  = self._sig.use_persistence

        q = mp.Queue()
        p = mp.Process(target=_ricci_worker,
                       args=(nodes, edges, use_ricci, use_pers, q))
        p.start()
        p.join(timeout=30)

        if p.exitcode != 0 or p.exitcode is None:
            p.terminate()
            return np.zeros(self.DIM, dtype=np.float32)

        result = q.get() if not q.empty() else None
        return result if result is not None else np.zeros(self.DIM, dtype=np.float32)

    def compute_tensor(self, G: nx.Graph,
                       device: torch.device = torch.device("cpu")) -> torch.Tensor:
        """Returns (1, DIM) float tensor."""
        return torch.tensor(self.compute(G), dtype=torch.float32,
                            device=device).unsqueeze(0)