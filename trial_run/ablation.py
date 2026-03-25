"""
ablation.py — Ablation study runner.

Runs selected model variants on the SAME train/val/test split and seeds,
then prints a comparison table.

Variants:
  0  GIN baseline          (no UTS at all)
  1  + §4.1 descriptor     (UTS concat to readout, no reg, no pool)
  2  + §4.2 regularization (UTS concat + topo losses, no pool)
  3  + §4.3 pooling        (full model, all components)

Usage:
    python ablation.py --dataset MUTAG --variants 0 1
    python ablation.py --dataset PROTEINS --variants 0 1 2 3 --seeds 42 43 44
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
    return dataset


def make_splits(dataset, seed):
    set_seed(seed)
    perm    = torch.randperm(len(dataset))
    dataset = dataset[perm]
    n       = len(dataset)
    n_train = int(0.8 * n)
    n_val   = int(0.1 * n)
    train_loader = DataLoader(dataset[:n_train],               batch_size=32, shuffle=True)
    val_loader   = DataLoader(dataset[n_train:n_train+n_val],  batch_size=32)
    test_loader  = DataLoader(dataset[n_train+n_val:],         batch_size=32)
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
# Main ablation loop
# ---------------------------------------------------------------------------

VARIANT_NAMES = {
    0: "V0: GIN baseline",
    1: "V1: +§4.1 descriptor",
    2: "V2: +§4.2 regularization",
    3: "V3: +§4.3 pooling (full)",
}


def run_ablation(args, device):
    dataset  = load_dataset(args.dataset)
    in_dim   = dataset.num_node_features or 11
    n_cls    = dataset.num_classes
    seeds    = args.seeds
    variants = sorted(set(args.variants))

    # Validate
    for v in variants:
        if v not in VARIANT_NAMES:
            raise ValueError(f"Unknown variant {v}. Choose from 0 1 2 3.")

    results = {VARIANT_NAMES[v]: [] for v in variants}

    print(f"\nRunning variants: {[VARIANT_NAMES[v] for v in variants]}")
    print(f"Seeds: {seeds}  |  Dataset: {args.dataset}\n")

    for seed in seeds:
        print(f"{'='*55}")
        print(f"  Seed {seed}")
        print(f"{'='*55}")

        train_loader, val_loader, test_loader = make_splits(dataset, seed)

        # ------------------------------------------------------------------
        # V0 — Pure GIN baseline
        # ------------------------------------------------------------------
        if 0 in variants:
            set_seed(seed)
            model_v0 = BaselineGIN(
                in_dim=in_dim, hidden_dim=args.hidden_dim,
                num_classes=n_cls, num_layers=args.num_layers,
                dropout=args.dropout,
            ).to(device)

            opt_v0 = optim.AdamW(model_v0.parameters(), lr=args.lr, weight_decay=1e-4)
            sch_v0 = optim.lr_scheduler.CosineAnnealingLR(opt_v0, T_max=args.epochs, eta_min=1e-5)

            val0, test0 = run_variant(
                model_v0,
                lambda m, l, o, d: train_baseline(m, l, o, d),
                lambda m, l, d: evaluate_baseline(m, l, d),
                train_loader, val_loader, test_loader,
                opt_v0, sch_v0, device,
                epochs=args.epochs, patience=args.patience,
            )
            results[VARIANT_NAMES[0]].append(test0)
            print(f"  V0 GIN baseline          | val {val0:.4f} | test {test0:.4f}")

        # ------------------------------------------------------------------
        # V1 — GIN + §4.1 UTS descriptor only
        # ------------------------------------------------------------------
        if 1 in variants:
            set_seed(seed)
            model_v1 = UTSGraphClassifier(
                in_dim=in_dim, hidden_dim=args.hidden_dim,
                num_classes=n_cls, num_layers=args.num_layers,
                dropout=args.dropout,
                use_toppool=False,
                lambda_reg=0.0,
                lambda_smooth=0.0,
                use_embed_uts_descriptor=args.use_embed_uts,
                use_graph_uts_descriptor=args.use_graph_uts,
                use_ricci=args.use_ricci,
            ).to(device)

            opt_v1 = optim.AdamW(model_v1.parameters(), lr=args.lr, weight_decay=1e-4)
            sch_v1 = optim.lr_scheduler.CosineAnnealingLR(opt_v1, T_max=args.epochs, eta_min=1e-5)

            val1, test1 = run_variant(
                model_v1,
                lambda m, l, o, d: train_graph_classifier(
                    m, l, o, d,
                    use_reg=False,
                    need_nx_graphs=args.use_graph_uts,   # nx needed for descriptor even without loss
                ),
                lambda m, l, d: evaluate_graph_classifier(m, l, d,
                    need_nx_graphs=args.use_graph_uts),
                train_loader, val_loader, test_loader,
                opt_v1, sch_v1, device,
                epochs=args.epochs, patience=args.patience,
            )
            results[VARIANT_NAMES[1]].append(test1)
            print(f"  V1 +§4.1 descriptor      | val {val1:.4f} | test {test1:.4f}")

        # ------------------------------------------------------------------
        # V2 — GIN + §4.1 + §4.2 regularization
        # ------------------------------------------------------------------
        if 2 in variants:
            set_seed(seed)
            model_v2 = UTSGraphClassifier(
                in_dim=in_dim, hidden_dim=args.hidden_dim,
                num_classes=n_cls, num_layers=args.num_layers,
                dropout=args.dropout,
                use_toppool=False,
                lambda_reg=args.lambda_reg,
                lambda_smooth=args.lambda_smooth,
                use_embed_uts_descriptor=args.use_embed_uts,
                use_graph_uts_descriptor=args.use_graph_uts,
                use_ricci=args.use_ricci,
            ).to(device)

            opt_v2 = optim.AdamW(model_v2.parameters(), lr=args.lr, weight_decay=1e-4)
            sch_v2 = optim.lr_scheduler.CosineAnnealingLR(opt_v2, T_max=args.epochs, eta_min=1e-5)

            val2, test2 = run_variant(
                model_v2,
                lambda m, l, o, d: train_graph_classifier(m, l, o, d,
                    use_reg=True, need_nx_graphs=True),
                lambda m, l, d: evaluate_graph_classifier(m, l, d,
                    need_nx_graphs=True),
                train_loader, val_loader, test_loader,
                opt_v2, sch_v2, device,
                epochs=args.epochs, patience=args.patience,
            )
            results[VARIANT_NAMES[2]].append(test2)
            print(f"  V2 +§4.2 regularization  | val {val2:.4f} | test {test2:.4f}")

        # ------------------------------------------------------------------
        # V3 — Full model
        # ------------------------------------------------------------------
        if 3 in variants:
            set_seed(seed)
            model_v3 = UTSGraphClassifier(
                in_dim=in_dim, hidden_dim=args.hidden_dim,
                num_classes=n_cls, num_layers=args.num_layers,
                dropout=args.dropout,
                use_toppool=True,
                pool_ratio=args.pool_ratio,
                lambda_reg=args.lambda_reg,
                lambda_smooth=args.lambda_smooth,
                use_embed_uts_descriptor=args.use_embed_uts,
                use_graph_uts_descriptor=args.use_graph_uts,
                use_ricci=args.use_ricci,
            ).to(device)

            opt_v3 = optim.AdamW(model_v3.parameters(), lr=args.lr, weight_decay=1e-4)
            sch_v3 = optim.lr_scheduler.CosineAnnealingLR(opt_v3, T_max=args.epochs, eta_min=1e-5)

            val3, test3 = run_variant(
                model_v3,
                lambda m, l, o, d: train_graph_classifier(m, l, o, d,
                    use_reg=True, need_nx_graphs=True),
                lambda m, l, d: evaluate_graph_classifier(m, l, d,
                    need_nx_graphs=True),
                train_loader, val_loader, test_loader,
                opt_v3, sch_v3, device,
                epochs=args.epochs, patience=args.patience,
            )
            results[VARIANT_NAMES[3]].append(test3)
            print(f"  V3 +§4.3 pooling (full)  | val {val3:.4f} | test {test3:.4f}")

    # ------------------------------------------------------------------
    # Final comparison table
    # ------------------------------------------------------------------
    print(f"\n{'='*55}")
    print(f"  ABLATION RESULTS — {args.dataset}  (seeds: {seeds})")
    print(f"{'='*55}")
    print(f"  {'Variant':<30} {'Mean':>7}  {'Std':>7}  Runs")
    print(f"  {'-'*50}")

    baseline_mean = np.mean(results[VARIANT_NAMES[0]]) if 0 in variants else None

    for name, accs in results.items():
        mean = np.mean(accs)
        std  = np.std(accs)
        runs = "  ".join(f"{a:.4f}" for a in accs)
        if baseline_mean is not None:
            delta     = mean - baseline_mean
            delta_str = f"(+{delta:.4f})" if delta >= 0 else f"({delta:.4f})"
            print(f"  {name:<30} {mean:.4f}   {std:.4f}   {runs}  {delta_str}")
        else:
            print(f"  {name:<30} {mean:.4f}   {std:.4f}   {runs}")
    print(f"{'='*55}\n")
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",       type=str,   default="MUTAG")
    parser.add_argument("--variants",      type=int,   nargs="+", default=[0, 1, 2, 3],
                        help="Which variants to run: 0=baseline 1=+desc 2=+reg 3=+pool")
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
    parser.add_argument("--use_embed_uts", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Concatenate EmbeddingUTS(H) to readout (§4.1). Default: on.")
    parser.add_argument("--use_graph_uts", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Concatenate GraphUTS(G) to readout (§4.1 extension). Default: off.")
    parser.add_argument("--use_ricci",     action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Enable Ricci curvature in GraphUTS (slow, may crash on large graphs).")

    args   = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    run_ablation(args, device)