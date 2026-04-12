"""
ablation.py — Fine-grained ablation study runner.

Variants:
  0   GIN baseline                     (nothing)
  1a  + EmbeddingUTS descriptor only   (§4.1 partial)
  1b  + GraphUTS descriptor only       (§4.1 partial)
  1c  + Both descriptors               (§4.1 full)
  2a  + LayerSmoothLoss only           (§4.2 partial, no descriptor)
  2b  + TopoRegLoss only               (§4.2 partial, no descriptor)
  2c  + Both losses                    (§4.2 full, no descriptor)
  2d  + Both descriptors + Both losses (§4.1 + §4.2 full)
  3   Full model                       (§4.1 + §4.2 + §4.3 pooling)

All UTS flags default to OFF — opt in explicitly.

Examples:
  # Descriptor ablation only
  python ablation.py --dataset PROTEINS --variants 0 1a 1b 1c

  # Loss ablation only (no descriptor)
  python ablation.py --dataset PROTEINS --variants 0 2a 2b 2c

  # Full ablation
  python ablation.py --dataset PROTEINS --variants 0 1a 1c 2a 2b 2c 2d 3 \\
      --use_embed_uts --use_graph_uts --use_reg --use_toppool

  # Just compare baseline to full model
  python ablation.py --dataset PROTEINS --variants 0 2d \\
      --use_embed_uts --use_graph_uts --use_reg
"""

import argparse
import random
import numpy as np
import torch
import torch.optim as optim

from torch_geometric.datasets import TUDataset
from torch_geometric.loader   import DataLoader

from baseline import BaselineGIN, train_baseline, evaluate_baseline
from model    import UTSGraphClassifier
from train    import (train_graph_classifier, evaluate_graph_classifier)
from uts      import GraphUTS
from torch_geometric.utils import to_networkx
from torch_geometric.data  import Data


# ---------------------------------------------------------------------------
# Indexed dataset wrapper — attaches graph index to each Data object
# so model.forward can look up precomputed GraphUTS from cache
# ---------------------------------------------------------------------------

class IndexedDataset(torch.utils.data.Dataset):
    """
    Wraps a PyG dataset and attaches .idx to every Data object.
    Supports integer indexing only — slice/tensor indexing goes through
    the underlying dataset and returns a new IndexedDataset.
    """
    def __init__(self, dataset, offset: int = 0):
        self.dataset = dataset
        self.offset  = offset   # global index offset for sliced subsets

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Integer index — return single Data with .idx attached
        if isinstance(idx, int):
            data = self.dataset[idx]
            # TUDataset may return a copy or the same object — always clone
            if hasattr(data, 'clone'):
                data = data.clone()
            else:
                # Fallback: reconstruct manually
                from torch_geometric.data import Data as PyGData
                data = PyGData(**{k: v for k, v in data})
            data.idx = torch.tensor(self.offset + idx, dtype=torch.long)
            return data

        # Tensor / slice index — delegate to underlying dataset and re-wrap
        # This handles dataset[perm] in make_splits
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()
        if isinstance(idx, list):
            subset = [self[i] for i in idx]   # recursive integer calls
            return _ListDataset(subset)

        # slice
        indices = range(*idx.indices(len(self)))
        return _ListDataset([self[i] for i in indices])

    @property
    def num_node_features(self):
        return self.dataset.num_node_features

    @property
    def num_classes(self):
        return self.dataset.num_classes


class _ListDataset(torch.utils.data.Dataset):
    """Simple wrapper around a list of Data objects."""
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

    @property
    def num_node_features(self):
        return self.items[0].x.size(-1) if self.items else 0

    @property
    def num_classes(self):
        return int(max(d.y.item() for d in self.items)) + 1


# ---------------------------------------------------------------------------
# GraphUTS precomputation — run once per dataset, cache by index
# ---------------------------------------------------------------------------

def precompute_graph_uts(dataset, use_ricci: bool = False) -> dict:
    """
    Precompute GraphUTS vectors for every graph in the dataset.
    Returns dict {dataset_idx: np.ndarray of shape (GraphUTS.DIM,)}.

    Called once before training starts — eliminates per-batch recomputation
    which is the main CPU bottleneck for V1b/V2b/V2c/V2d/V3 variants.
    """
    from torch_geometric.loader import DataLoader as DL
    import numpy as np

    guts  = GraphUTS(use_ricci=use_ricci)
    cache = {}
    total = len(dataset)

    print(f"  Precomputing GraphUTS for {total} graphs "
          f"(ricci={use_ricci})...")

    loader = DL(dataset, batch_size=1, shuffle=False)
    for i, data in enumerate(loader):
        G = to_networkx(data, to_undirected=True)
        cache[i] = guts.safe_compute(G)
        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"    {i+1}/{total}", end="\r")

    print(f"  ✓ GraphUTS cache built ({total} graphs)\n")
    return cache


# ---------------------------------------------------------------------------
# All variant definitions — name + what each flag is set to
# ---------------------------------------------------------------------------

# Each entry: (display_name, use_embed, use_graph, lambda_reg, lambda_smooth, use_toppool)
VARIANT_CONFIGS = {
    "0":  ("V0:  GIN baseline",                   False, False, 0.0,  0.0,  False),
    "1a": ("V1a: +EmbUTS descriptor",             True,  False, 0.0,  0.0,  False),
    "1b": ("V1b: +GraphUTS descriptor",           False, True,  0.0,  0.0,  False),
    "1c": ("V1c: +Both descriptors",              True,  True,  0.0,  0.0,  False),
    "2a": ("V2a: +LayerSmooth loss only",         False, False, 0.0,  None, False),
    "2b": ("V2b: +TopoReg loss only",             False, False, None, 0.0,  False),
    "2c": ("V2c: +Both losses",                   False, False, None, None, False),
    "2d": ("V2d: +Both descs + Both losses",      True,  True,  None, None, False),
    "3":  ("V3:  Full model (2d + TopPool)",      True,  True,  None, None, True),
}
# None in lambda means "use args.lambda_reg / args.lambda_smooth"
# 0.0 means "disable that loss"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(name):
    dataset = TUDataset(root=f"data/{name}", name=name, use_node_attr=True)
    if dataset.num_node_features == 0:
        from torch_geometric.transforms import OneHotDegree
        dataset.transform = OneHotDegree(max_degree=10)
    # Wrap so every Data object gets a .idx field for cache lookup
    return IndexedDataset(dataset)


def make_splits(dataset, seed):
    set_seed(seed)
    n       = len(dataset)
    perm    = torch.randperm(n).tolist()   # list of ints, not tensor
    n_train = int(0.8 * n)
    n_val   = int(0.1 * n)

    train_idx = perm[:n_train]
    val_idx   = perm[n_train:n_train + n_val]
    test_idx  = perm[n_train + n_val:]

    train_loader = DataLoader([dataset[i] for i in train_idx], batch_size=32, shuffle=True)
    val_loader   = DataLoader([dataset[i] for i in val_idx],   batch_size=32)
    test_loader  = DataLoader([dataset[i] for i in test_idx],  batch_size=32)
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Single-variant training run
# ---------------------------------------------------------------------------

def run_variant(model, train_fn, eval_fn,
                train_loader, val_loader, test_loader,
                optimizer, scheduler, device,
                epochs=200, patience=30):

    best_val   = 0.0
    best_state = None
    patience_cnt = 0

    for epoch in range(1, epochs + 1):
        train_fn(model, train_loader, optimizer, device)
        val_result = eval_fn(model, val_loader, device)
        val_acc    = val_result["accuracy"] if isinstance(val_result, dict) else val_result
        scheduler.step()

        if val_acc > best_val:
            best_val     = val_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1

        if patience_cnt >= patience:
            break

    model.load_state_dict(best_state)
    test_result = eval_fn(model, test_loader, device)
    test_acc    = test_result["accuracy"] if isinstance(test_result, dict) else test_result
    return best_val, test_acc


# ---------------------------------------------------------------------------
# Build one model + train/eval fns from a variant config
# ---------------------------------------------------------------------------

def build_variant(variant_key, args, in_dim, n_cls, device,
                  graph_uts_cache=None):
    """
    Returns (model, train_fn, eval_fn) for the given variant key.
    graph_uts_cache is passed through to train/eval functions if provided.
    """
    name, use_embed, use_graph, lreg, lsmooth, use_pool = VARIANT_CONFIGS[variant_key]

    lreg    = args.lambda_reg    if lreg    is None else lreg
    lsmooth = args.lambda_smooth if lsmooth is None else lsmooth

    # Variant 0 uses BaselineGIN
    if variant_key == "0":
        model = BaselineGIN(
            in_dim=in_dim, hidden_dim=args.hidden_dim,
            num_classes=n_cls, num_layers=args.num_layers,
            dropout=args.dropout,
        ).to(device)

        def train_fn(m, loader, opt, dev):
            return train_baseline(m, loader, opt, dev)

        def eval_fn(m, loader, dev):
            return evaluate_baseline(m, loader, dev)

        return model, train_fn, eval_fn

    use_reg  = (lreg > 0) or (lsmooth > 0)
    # need_nx_graphs only if cache unavailable AND (reg or graph descriptor active)
    need_nx  = (use_reg or use_graph) and (graph_uts_cache is None)

    model = UTSGraphClassifier(
        in_dim=in_dim, hidden_dim=args.hidden_dim,
        num_classes=n_cls, num_layers=args.num_layers,
        dropout=args.dropout,
        use_toppool=use_pool,
        pool_ratio=args.pool_ratio,
        lambda_reg=lreg,
        lambda_smooth=lsmooth,
        use_embed_uts_descriptor=use_embed,
        use_graph_uts_descriptor=use_graph,
        use_ricci=args.use_ricci,
    ).to(device)

    def train_fn(m, loader, opt, dev):
        return train_graph_classifier(
            m, loader, opt, dev,
            use_reg=use_reg,
            need_nx_graphs=need_nx,
            graph_uts_cache=graph_uts_cache,
        )

    def eval_fn(m, loader, dev):
        return evaluate_graph_classifier(
            m, loader, dev,
            need_nx_graphs=need_nx,
            graph_uts_cache=graph_uts_cache,
        )

    return model, train_fn, eval_fn


# ---------------------------------------------------------------------------
# Main ablation loop
# ---------------------------------------------------------------------------

def run_ablation(args, device):
    dataset  = load_dataset(args.dataset)
    in_dim   = dataset.num_node_features or 11
    n_cls    = dataset.num_classes
    seeds    = args.seeds
    variants = args.variants

    # Validate
    for v in variants:
        if v not in VARIANT_CONFIGS:
            raise ValueError(
                f"Unknown variant '{v}'. "
                f"Choose from: {list(VARIANT_CONFIGS.keys())}"
            )

    results = {VARIANT_CONFIGS[v][0]: [] for v in variants}

    # Precompute GraphUTS once for the whole dataset if any variant needs it
    needs_graph_uts = any(
        VARIANT_CONFIGS[v][2]  # use_graph column
        or VARIANT_CONFIGS[v][3] is None  # lambda_reg=None means use args value > 0
        for v in variants if v != "0"
    )
    graph_uts_cache = None
    if needs_graph_uts and (args.use_graph_uts or args.use_reg):
        graph_uts_cache = precompute_graph_uts(dataset, use_ricci=args.use_ricci)

    print(f"\nRunning variants: {variants}")
    print(f"Seeds: {seeds}  |  Dataset: {args.dataset}")
    print(f"Flags: embed_uts={args.use_embed_uts}  graph_uts={args.use_graph_uts}  "
          f"ricci={args.use_ricci}\n")

    for seed in seeds:
        print(f"{'='*60}")
        print(f"  Seed {seed}")
        print(f"{'='*60}")

        train_loader, val_loader, test_loader = make_splits(dataset, seed)

        for v in variants:
            set_seed(seed)
            name = VARIANT_CONFIGS[v][0]

            model, train_fn, eval_fn = build_variant(
                v, args, in_dim, n_cls, device,
                graph_uts_cache=graph_uts_cache,
            )

            optimizer = optim.AdamW(model.parameters(),
                                    lr=args.lr, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs, eta_min=1e-5
            )

            val_acc, test_acc = run_variant(
                model, train_fn, eval_fn,
                train_loader, val_loader, test_loader,
                optimizer, scheduler, device,
                epochs=args.epochs, patience=args.patience,
            )

            results[name].append(test_acc)
            print(f"  {name:<40} | val {val_acc:.4f} | test {test_acc:.4f}")

    # ------------------------------------------------------------------
    # Final comparison table
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  ABLATION RESULTS — {args.dataset}  (seeds: {seeds})")
    print(f"{'='*60}")
    print(f"  {'Variant':<40} {'Mean':>7}  {'Std':>7}  Runs")
    print(f"  {'-'*56}")

    # Use V0 as baseline for delta if it was run
    baseline_mean = None
    v0_name = VARIANT_CONFIGS["0"][0]
    if v0_name in results:
        baseline_mean = np.mean(results[v0_name])

    for name, accs in results.items():
        mean = np.mean(accs)
        std  = np.std(accs)
        runs = "  ".join(f"{a:.4f}" for a in accs)
        if baseline_mean is not None:
            delta     = mean - baseline_mean
            sign      = "+" if delta >= 0 else ""
            delta_str = f"({sign}{delta:.4f})"
            print(f"  {name:<40} {mean:.4f}   {std:.4f}   {runs}  {delta_str}")
        else:
            print(f"  {name:<40} {mean:.4f}   {std:.4f}   {runs}")

    print(f"{'='*60}\n")
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
Ablation study runner. All UTS flags default to OFF.

Variant keys: 0, 1a, 1b, 1c, 2a, 2b, 2c, 2d, 3

Examples:
  # Descriptor ablation
  python ablation.py --dataset PROTEINS --variants 0 1a 1b 1c \\
      --use_embed_uts --use_graph_uts

  # Loss ablation (no descriptor)
  python ablation.py --dataset PROTEINS --variants 0 2a 2b 2c --use_reg

  # Full ablation
  python ablation.py --dataset PROTEINS --variants 0 1a 1c 2c 2d 3 \\
      --use_embed_uts --use_graph_uts --use_reg --use_toppool
        """)

    # Dataset / training
    parser.add_argument("--dataset",       type=str,   default="MUTAG")
    parser.add_argument("--variants",      type=str,   nargs="+",
                        default=["0", "1a", "1c", "2c", "2d"],
                        help="Variant keys to run. e.g. 0 1a 1b 1c 2a 2b 2c 2d 3")
    parser.add_argument("--hidden_dim",    type=int,   default=128)
    parser.add_argument("--num_layers",    type=int,   default=4)
    parser.add_argument("--epochs",        type=int,   default=200)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--dropout",       type=float, default=0.3)
    parser.add_argument("--pool_ratio",    type=float, default=0.5)
    parser.add_argument("--lambda_reg",    type=float, default=0.01)
    parser.add_argument("--lambda_smooth", type=float, default=0.005)
    parser.add_argument("--patience",      type=int,   default=30)
    parser.add_argument("--seeds",         type=int,   nargs="+", default=[42, 43, 44])

    # UTS flags — ALL default False
    parser.add_argument("--use_embed_uts", action="store_true", default=False,
                        help="Enable EmbeddingUTS descriptor (§4.1)")
    parser.add_argument("--use_graph_uts", action="store_true", default=False,
                        help="Enable GraphUTS descriptor (§4.1 extension)")
    parser.add_argument("--use_ricci",     action="store_true", default=False,
                        help="Enable Ricci curvature in GraphUTS")
    parser.add_argument("--use_toppool",   action="store_true", default=False,
                        help="Enable UTSTopPool in V3 (§4.3)")
    parser.add_argument("--use_reg",       action="store_true", default=False,
                        help="Enable reg losses in V2x/V3 variants (§4.2)")

    args   = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    run_ablation(args, device)
