import networkx as nx
import numpy as np
from scipy.linalg import eigvalsh
from scipy.stats import entropy

from GraphRicciCurvature.OllivierRicci import OllivierRicci
from GraphRicciCurvature.FormanRicci import FormanRicci
import gudhi as gd


class GraphSignature:
    def __init__(self, use_ricci=True, use_persistence=True, verbose=False):
        self.use_ricci = use_ricci
        self.use_persistence = use_persistence
        self.verbose = verbose

    # ---------------------------
    # Utility
    # ---------------------------
    def _safe_stats(self, arr):
        if len(arr) == 0:
            return 0.0, 0.0, 0.0, 0.0
        return np.mean(arr), np.var(arr), np.min(arr), np.max(arr)

    # ---------------------------
    # Ricci Features
    # ---------------------------
    def _ricci_features(self, G):
        if not self.use_ricci:
            return [0]*8

    # Ollivier
        orc = OllivierRicci(G.copy(), alpha=0.5, verbose="ERROR")
        orc.compute_ricci_curvature()
        G_orc = orc.G

        orc_vals = [
            d.get("ricciCurvature", 0.0)
            for _, _, d in G_orc.edges(data=True)
        ]

    # Forman
        frc = FormanRicci(G.copy())
        frc.compute_ricci_curvature()
        G_frc = frc.G

        frc_vals = [
            d.get("formanCurvature", 0.0)
            for _, _, d in G_frc.edges(data=True)
        ]

        orc_mean, orc_var, orc_min, orc_max = self._safe_stats(orc_vals)
        frc_mean, frc_var, frc_min, frc_max = self._safe_stats(frc_vals)

        orc_neg_frac = np.mean([v < 0 for v in orc_vals]) if len(orc_vals) else 0

        return [
            orc_mean, orc_var, orc_min, orc_neg_frac,
            frc_mean, frc_var, frc_min, frc_max
        ]
    # ---------------------------
    # Distance
    # ---------------------------
    def _distance_features(self, G):
        # Explicitly work on largest component if disconnected
        if not nx.is_connected(G):
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()

        lengths = dict(nx.all_pairs_shortest_path_length(G))
        dists = [
            lengths[u][v]
            for u in lengths
            for v in lengths[u]
            if u != v
        ]

        if len(dists) == 0:
            return [0, 0]

        return [
            np.mean(dists),
            nx.diameter(G)  # safe now — G is guaranteed connected
        ]

    # ---------------------------
    # Spectral
    # ---------------------------
    def _spectral_features(self, G):
        L = nx.laplacian_matrix(G).toarray()
        eigvals = eigvalsh(L)

        # Extract Fiedler value before any clipping
        spectral_gap = eigvals[1] if len(eigvals) > 1 else 0
        max_eig = eigvals[-1]

        # Clip only for entropy computation
        eigvals_clipped = np.clip(eigvals, 1e-12, None)
        prob = eigvals_clipped / np.sum(eigvals_clipped)
        spec_entropy = entropy(prob)

        return [spectral_gap, max_eig, spec_entropy]

    # ---------------------------
    # Persistence
    # ---------------------------
    def _persistence_features(self, G):
        if not self.use_persistence:
            return [0]*4

        nodes = list(G.nodes())
        n = len(nodes)

        lengths = dict(nx.all_pairs_shortest_path_length(G))
        dist_matrix = np.zeros((n, n))

        for i, u in enumerate(nodes):
            for j, v in enumerate(nodes):
                if u == v:
                    continue
                dist_matrix[i, j] = lengths[u].get(v, np.inf)

        max_edge = np.max(dist_matrix[np.isfinite(dist_matrix)])

        rips = gd.RipsComplex(distance_matrix=dist_matrix, max_edge_length=max_edge)
        st = rips.create_simplex_tree(max_dimension=1)
        diag = st.persistence()

        h0, h1 = [], []

        for dim, (b, d) in diag:
            if d == float("inf"):
                continue
            if dim == 0:
                h0.append(d - b)
            elif dim == 1:
                h1.append(d - b)

        def stats(arr):
            if len(arr) == 0:
                return 0, 0
            return np.mean(arr), entropy(np.array(arr) + 1e-12)

        h0_mean, h0_ent = stats(h0)
        h1_mean, h1_ent = stats(h1)

        return [h0_mean, h0_ent, h1_mean, h1_ent]

    # ---------------------------
    # Degree
    # ---------------------------
    def _degree_features(self, G):
        deg = [d for _, d in G.degree()]
        return [np.mean(deg), np.var(deg), max(deg) if deg else 0]

    # ---------------------------
    # Clustering
    # ---------------------------
    def _clustering_features(self, G):
        c = list(nx.clustering(G).values())
        return [np.mean(c), np.var(c)]

    # ---------------------------
    # Centrality
    # ---------------------------
    def _centrality_features(self, G):
        bet = list(nx.betweenness_centrality(G).values())
        clo = list(nx.closeness_centrality(G).values())

        return [
            np.mean(bet), np.var(bet),
            np.mean(clo)
        ]

    # ---------------------------
    # Connectivity
    # ---------------------------
    def _connectivity_features(self, G):
        comps = list(nx.connected_components(G))
        largest = max(len(c) for c in comps)

        return [
            len(comps),
            largest / G.number_of_nodes()
        ]

    # ---------------------------
    # PUBLIC API
    # ---------------------------
    def compute(self, G):
        feats = []

        feats += self._ricci_features(G)
        feats += self._distance_features(G)
        feats += self._spectral_features(G)
        feats += self._persistence_features(G)
        feats += self._degree_features(G)
        feats += self._clustering_features(G)
        feats += self._centrality_features(G)
        feats += self._connectivity_features(G)

        return np.array(feats)

    # ---------------------------
    # LOCAL SIGNATURE
    # ---------------------------
    def compute_local(self, G, node, k=2):
        subG = nx.ego_graph(G, node, radius=k)
        return self.compute(subG)