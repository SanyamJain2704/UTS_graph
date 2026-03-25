"""
train.py — Training loops for all three tasks.

  train_graph_classifier()    : supervised graph classification
  train_node_classifier()     : supervised node classification
  train_contrastive()         : self-supervised contrastive pre-training (4.5)
  evaluate_graph_classifier() : evaluation with accuracy + loss breakdown
"""

import torch
import torch.nn.functional as F
from torch_geometric.utils import to_networkx

from losses import TopoContrastLoss


# ---------------------------------------------------------------------------
# Utility: convert PyG batch to list of nx.Graph
# ---------------------------------------------------------------------------

def batch_to_nx(data):
    """Convert a PyG Batch into a list of networkx graphs."""
    graphs = []
    num_graphs = data.num_graphs
    for i in range(num_graphs):
        mask = (data.batch == i)
        node_idx = mask.nonzero(as_tuple=False).squeeze(1)

        # Build sub-data for this graph
        sub_edge_mask = mask[data.edge_index[0]] & mask[data.edge_index[1]]
        sub_edges = data.edge_index[:, sub_edge_mask]

        # Remap node indices to 0..n_i-1
        old2new = {int(o): n for n, o in enumerate(node_idx.tolist())}
        src = [old2new[int(s)] for s in sub_edges[0].tolist()]
        dst = [old2new[int(d)] for d in sub_edges[1].tolist()]

        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(range(node_idx.size(0)))
        G.add_edges_from(zip(src, dst))
        graphs.append(G)
    return graphs


# ---------------------------------------------------------------------------
# Graph classification
# ---------------------------------------------------------------------------

def train_graph_classifier(model, loader, optimizer, device,
                            use_reg=True, need_nx_graphs=None):
    """
    One epoch of supervised graph classification training.

    Args:
        use_reg        : compute TopoRegLoss + LayerSmoothLoss
        need_nx_graphs : force nx_graphs to be computed even if use_reg=False
                         (needed when use_graph_uts_descriptor=True on V1)
                         defaults to same value as use_reg if not set

    Returns:
        dict with mean losses: task, smooth, reg, total
    """
    # need_nx_graphs=True whenever either the loss or the descriptor needs it
    if need_nx_graphs is None:
        need_nx_graphs = use_reg

    model.train()
    total_task = total_smooth = total_reg = 0.0
    n_batches = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        nx_graphs = batch_to_nx(data) if need_nx_graphs else None
        logits, aux = model(data, nx_graphs=nx_graphs)

        task_loss   = F.cross_entropy(logits, data.y)
        smooth_loss = aux["smooth_loss"]
        reg_loss    = aux["reg_loss"]

        loss = task_loss + smooth_loss + reg_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        total_task   += task_loss.item()
        total_smooth += smooth_loss.item()
        total_reg    += reg_loss.item()
        n_batches    += 1

    n = max(n_batches, 1)
    return {
        "task":   total_task   / n,
        "smooth": total_smooth / n,
        "reg":    total_reg    / n,
        "total":  (total_task + total_smooth + total_reg) / n,
    }


@torch.no_grad()
def evaluate_graph_classifier(model, loader, device, need_nx_graphs=False):
    model.eval()
    correct = total = 0
    total_loss = 0.0

    for data in loader:
        data      = data.to(device)
        nx_graphs = batch_to_nx(data) if need_nx_graphs else None
        logits, _ = model(data, nx_graphs=nx_graphs)
        pred      = logits.argmax(dim=-1)
        correct   += (pred == data.y).sum().item()
        total     += data.y.size(0)
        total_loss += F.cross_entropy(logits, data.y).item()

    return {
        "accuracy": correct / max(total, 1),
        "loss":     total_loss / max(len(loader), 1),
    }


# ---------------------------------------------------------------------------
# Node classification
# ---------------------------------------------------------------------------

def train_node_classifier(model, data, optimizer, device,
                           train_mask, use_reg=True):
    """
    Single-graph transductive node classification (one step).

    Args:
        data       : PyG Data (single graph)
        train_mask : boolean mask over nodes
    Returns:
        loss dict
    """
    model.train()
    data = data.to(device)
    optimizer.zero_grad()

    nx_graph = [to_networkx(data, to_undirected=True)] if use_reg else None
    logits, aux = model(data, nx_graphs=nx_graph)

    task_loss   = F.cross_entropy(logits[train_mask], data.y[train_mask])
    smooth_loss = aux["smooth_loss"]
    reg_loss    = aux["reg_loss"]

    loss = task_loss + smooth_loss + reg_loss
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
    optimizer.step()

    return {
        "task":   task_loss.item(),
        "smooth": smooth_loss.item(),
        "reg":    reg_loss.item(),
        "total":  loss.item(),
    }


@torch.no_grad()
def evaluate_node_classifier(model, data, device, mask):
    model.eval()
    data   = data.to(device)
    logits, _ = model(data)
    pred   = logits.argmax(dim=-1)
    correct = (pred[mask] == data.y[mask]).sum().item()
    total   = mask.sum().item()
    return {"accuracy": correct / max(total, 1)}


# ---------------------------------------------------------------------------
# Contrastive pre-training (4.5)
# ---------------------------------------------------------------------------

def _augment(data, drop_edge_ratio=0.2, mask_feat_ratio=0.1):
    """Simple graph augmentation: random edge dropping + feature masking."""
    import torch
    data2 = data.clone()

    # Edge drop
    n_edges = data2.edge_index.size(1)
    keep    = torch.rand(n_edges) > drop_edge_ratio
    data2.edge_index = data2.edge_index[:, keep]

    # Feature masking
    mask = torch.rand_like(data2.x) < mask_feat_ratio
    data2.x = data2.x.masked_fill(mask, 0.0)

    return data2


def train_contrastive(model, loader, optimizer, device,
                      lambda_contrast=0.1):
    """
    One epoch of topology-preserving contrastive pre-training.
    """
    model.train()
    topo_loss_fn = TopoContrastLoss(lambda_contrast=lambda_contrast)
    total_loss   = 0.0
    n_batches    = 0

    for data in loader:
        data  = data.to(device)
        data2 = _augment(data).to(device)

        optimizer.zero_grad()

        H1, batch1, z1, _ = model(data)
        H2, batch2, z2, _ = model(data2)

        loss = topo_loss_fn(H1, batch1, H2, batch2, z1=z1, z2=z2)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return {"contrastive": total_loss / max(n_batches, 1)}