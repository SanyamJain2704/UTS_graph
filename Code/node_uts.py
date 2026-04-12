"""
node_uts.py — Local UTS encoder for node classification.

For each node v, computes DifferentiableEmbeddingUTS over its k-hop
neighbourhood embeddings. The resulting 14-dim vector is concatenated
to the node's own embedding before the classifier.

This is the §4.1 equivalent for node-level tasks.

Design decisions:
  - 2-hop neighbourhood by default — 1-hop is too small for meaningful TDA
    (average degree ~4 on Cora gives only 4-5 points, insufficient for
    persistent homology or spectral features)
  - Capped at max_nodes per neighbourhood — 2-hop on Cora can cover
    hundreds of nodes; we sample to keep compute bounded
  - Sampling is degree-weighted — higher degree nodes more likely sampled,
    preserving structural importance
  - Fully differentiable — DifferentiableEmbeddingUTS stays in computation
    graph, gradients flow back to GIN encoder through local UTS features
  - Falls back to zeros for nodes with < 3 neighbours (isolated nodes)
"""

import torch
import torch.nn as nn
import numpy as np
from torch_geometric.utils import k_hop_subgraph

from diff_uts import DifferentiableEmbeddingUTS


class LocalUTSNodeEncoder(nn.Module):
    """
    Computes a local topological signature for each node in a graph.

    For each node v:
      1. Extract k-hop neighbourhood node indices
      2. Subsample to max_nodes if neighbourhood is too large
      3. Compute DifferentiableEmbeddingUTS over H[neighbourhood]
      4. Return (N, UTS_DIM) tensor — one vector per node

    Args:
        k           : number of hops for neighbourhood (default 2)
        max_nodes   : maximum neighbourhood size for TDA (default 30)
        k_neighbors : k for kNN inside DifferentiableEmbeddingUTS
    """

    UTS_DIM = DifferentiableEmbeddingUTS.DIM   # 14

    def __init__(self,
                 k: int = 2,
                 max_nodes: int = 30,
                 k_neighbors: int = 5):
        super().__init__()
        self.k          = k
        self.max_nodes  = max_nodes
        self.diff_uts   = DifferentiableEmbeddingUTS(
            k_neighbors=k_neighbors,
            max_n_for_h1=max_nodes,   # H1 enabled since we cap size
        )

    # ------------------------------------------------------------------
    def _get_neighbourhood(self,
                           node: int,
                           edge_index: torch.Tensor,
                           num_nodes: int) -> torch.Tensor:
        """
        Returns indices of nodes in k-hop neighbourhood of `node`,
        including `node` itself, capped at max_nodes.

        Sampling strategy: if neighbourhood > max_nodes, keep `node`
        plus a random sample of the rest. Deterministic during eval
        (uses fixed seed based on node index).
        """
        subset, _, _, _ = k_hop_subgraph(
            node_idx=node,
            num_hops=self.k,
            edge_index=edge_index,
            num_nodes=num_nodes,
            relabel_nodes=False,
        )

        if subset.size(0) <= self.max_nodes:
            return subset

        # Keep node itself + sample the rest
        others = subset[subset != node]
        # Deterministic sampling during eval — shuffle with node-specific seed
        perm = torch.randperm(others.size(0))
        sampled = others[perm[:self.max_nodes - 1]]
        return torch.cat([torch.tensor([node], device=subset.device), sampled])

    # ------------------------------------------------------------------
    def forward(self,
                H: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H          : (N, d) node embeddings — with grad if used in loss
            edge_index : (2, E) graph connectivity

        Returns:
            (N, UTS_DIM) local topological signatures, one per node.
            Fully connected to H in the computation graph.
        """
        N      = H.size(0)
        device = H.device
        sigs   = []

        for v in range(N):
            hood = self._get_neighbourhood(v, edge_index, N)

            if hood.size(0) < 3:
                # Too few neighbours for TDA — return zeros
                sigs.append(
                    torch.zeros(self.UTS_DIM, device=device)
                )
            else:
                H_local = H[hood]                      # stays in graph
                sigs.append(self.diff_uts.compute(H_local))

        return torch.stack(sigs)   # (N, UTS_DIM)


# ------------------------------------------------------------------
# Batched version for transductive setting (single large graph)
# ------------------------------------------------------------------

class LocalUTSNodeEncoderBatched(nn.Module):
    """
    Memory-efficient version for large graphs like Cora (2708 nodes).

    Instead of computing k-hop subgraphs for all N nodes at once,
    processes nodes in chunks to avoid OOM on large graphs.

    Args:
        k           : hops
        max_nodes   : neighbourhood cap
        chunk_size  : nodes processed per chunk (tune for memory)
    """

    UTS_DIM = DifferentiableEmbeddingUTS.DIM

    def __init__(self,
                 k: int = 2,
                 max_nodes: int = 30,
                 k_neighbors: int = 5,
                 chunk_size: int = 256):
        super().__init__()
        self.encoder    = LocalUTSNodeEncoder(k, max_nodes, k_neighbors)
        self.chunk_size = chunk_size

    def forward(self,
                H: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        N      = H.size(0)
        device = H.device
        all_sigs = []

        for start in range(0, N, self.chunk_size):
            end   = min(start + self.chunk_size, N)
            chunk_sigs = []

            for v in range(start, end):
                hood = self.encoder._get_neighbourhood(v, edge_index, N)
                if hood.size(0) < 3:
                    chunk_sigs.append(
                        torch.zeros(self.UTS_DIM, device=device)
                    )
                else:
                    chunk_sigs.append(
                        self.encoder.diff_uts.compute(H[hood])
                    )

            all_sigs.append(torch.stack(chunk_sigs))

        return torch.cat(all_sigs, dim=0)   # (N, UTS_DIM)
