"""
run_experiment.py — End-to-end experiment runner.

Benchmarks: MUTAG, PROTEINS, COLLAB (graph classification)
            Cora, CiteSeer (node classification)

Usage:
    python run_experiment.py --task graph --dataset MUTAG
    python run_experiment.py --task node  --dataset Cora
    python run_experiment.py --task pretrain --dataset PROTEINS
"""

import argparse
import random
import numpy as np
import torch
import torch.optim as optim
import networkx as nx

from torch_geometric.datasets import TUDataset, Planetoid
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx


from model    import UTSGraphClassifier, UTSNodeClassifier, UTSContrastiveModel
from train    import (train_graph_classifier, evaluate_graph_classifier,
                      train_node_classifier, evaluate_node_classifier,
                      train_contrastive)
from analysis import UTSAnalyzer


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Graph Classification
# ---------------------------------------------------------------------------

def run_graph_classification(args, device):
    dataset = TUDataset(root=f"data/{args.dataset}",
                        name=args.dataset,
                        use_node_attr=True)

    # Handle datasets with no node features
    if dataset.num_node_features == 0:
        from torch_geometric.transforms import OneHotDegree
        dataset.transform = OneHotDegree(max_degree=10)

    # Shuffle and split 80/10/10
    perm    = torch.randperm(len(dataset))
    dataset = dataset[perm]
    n       = len(dataset)
    n_train = int(0.8 * n)
    n_val   = int(0.1 * n)

    train_set = dataset[:n_train]
    val_set   = dataset[n_train:n_train + n_val]
    test_set  = dataset[n_train + n_val:]

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size)
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size)

    in_dim = dataset.num_node_features or 11   # fallback for OneHotDegree

    model = UTSGraphClassifier(
        in_dim=in_dim,
        hidden_dim=args.hidden_dim,
        num_classes=dataset.num_classes,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_toppool=args.use_toppool,
        pool_ratio=args.pool_ratio,
        lambda_reg=args.lambda_reg,
        lambda_smooth=args.lambda_smooth,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(),
                            lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5)

    print(f"\n── Graph Classification: {args.dataset} ──────────────────")
    print(f"   Graphs: {n}  |  Classes: {dataset.num_classes}  "
          f"|  Features: {in_dim}")
    print(f"   Model params: {sum(p.numel() for p in model.parameters()):,}\n")

    best_val_acc = 0.0
    patience_cnt = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_graph_classifier(
            model, train_loader, optimizer, device,
            use_reg=args.use_reg
        )
        val_metrics = evaluate_graph_classifier(model, val_loader, device)
        scheduler.step()

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save(model.state_dict(), "best_graph_model.pt")
            patience_cnt = 0
        else:
            patience_cnt += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>4} | "
                  f"Train loss: {train_metrics['total']:.4f} "
                  f"(task {train_metrics['task']:.4f} "
                  f"smooth {train_metrics['smooth']:.4f} "
                  f"reg {train_metrics['reg']:.4f}) | "
                  f"Val acc: {val_metrics['accuracy']:.4f}")

        if patience_cnt >= args.patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    # Final test
    model.load_state_dict(torch.load("best_graph_model.pt"))
    test_metrics = evaluate_graph_classifier(model, test_loader, device)
    print(f"\n  ✓ Test Accuracy: {test_metrics['accuracy']:.4f}  "
          f"(best val: {best_val_acc:.4f})")

    # Section 4.4 — Layer-wise analysis
    if args.analyze:
        analyzer = UTSAnalyzer(model, device)
        analyzer.track(test_loader)
        analyzer.print_summary()
        analyzer.plot_evolution(save_path=f"{args.dataset}_uts_evolution.pdf")
        analyzer.plot_pca(save_path=f"{args.dataset}_uts_pca.pdf")

    return test_metrics


# ---------------------------------------------------------------------------
# Node Classification
# ---------------------------------------------------------------------------

def run_node_classification(args, device):
    dataset = Planetoid(root=f"data/{args.dataset}", name=args.dataset)
    data    = dataset[0].to(device)

    model = UTSNodeClassifier(
        in_dim=dataset.num_node_features,
        hidden_dim=args.hidden_dim,
        num_classes=dataset.num_classes,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lambda_reg=args.lambda_reg,
        lambda_smooth=args.lambda_smooth,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(),
                            lr=args.lr, weight_decay=5e-4)

    print(f"\n── Node Classification: {args.dataset} ───────────────────")
    print(f"   Nodes: {data.num_nodes}  |  "
          f"Classes: {dataset.num_classes}  |  "
          f"Features: {dataset.num_node_features}")
    print(f"   Model params: {sum(p.numel() for p in model.parameters()):,}\n")

    best_val_acc = 0.0
    patience_cnt = 0

    # Pre-compute nx graph once (expensive for Ricci — disable for large graphs)
    nx_graph = [to_networkx(data.cpu(), to_undirected=True)] \
               if (args.use_reg and data.num_nodes < 5000) else None

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_node_classifier(
            model, data, optimizer, device,
            train_mask=data.train_mask,
            use_reg=(nx_graph is not None)
        )

        with torch.no_grad():
            val_acc = evaluate_node_classifier(
                model, data, device, data.val_mask)["accuracy"]

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_node_model.pt")
            patience_cnt = 0
        else:
            patience_cnt += 1

        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>4} | "
                  f"Loss: {train_metrics['total']:.4f} | "
                  f"Val acc: {val_acc:.4f}")

        if patience_cnt >= args.patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    model.load_state_dict(torch.load("best_node_model.pt"))
    test_acc = evaluate_node_classifier(
        model, data, device, data.test_mask)["accuracy"]
    print(f"\n  ✓ Test Accuracy: {test_acc:.4f}  "
          f"(best val: {best_val_acc:.4f})")
    return {"accuracy": test_acc}


# ---------------------------------------------------------------------------
# Contrastive pre-training
# ---------------------------------------------------------------------------

def run_pretrain(args, device):
    dataset = TUDataset(root=f"data/{args.dataset}",
                        name=args.dataset, use_node_attr=True)
    if dataset.num_node_features == 0:
        from torch_geometric.transforms import OneHotDegree
        dataset.transform = OneHotDegree(max_degree=10)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    in_dim = dataset.num_node_features or 11

    model = UTSContrastiveModel(
        in_dim=in_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    print(f"\n── Contrastive Pre-training: {args.dataset} ──────────────")
    for epoch in range(1, args.epochs + 1):
        m = train_contrastive(model, loader, optimizer, device)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>4} | Contrastive loss: {m['contrastive']:.4f}")

    torch.save(model.encoder.state_dict(), f"pretrained_encoder_{args.dataset}.pt")
    print(f"\n  ✓ Saved encoder to pretrained_encoder_{args.dataset}.pt")


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",         type=str,   default="graph",
                        choices=["graph", "node", "pretrain"])
    parser.add_argument("--dataset",      type=str,   default="MUTAG")
    parser.add_argument("--hidden_dim",   type=int,   default=128)
    parser.add_argument("--num_layers",   type=int,   default=4)
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--epochs",       type=int,   default=200)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--dropout",      type=float, default=0.3)
    parser.add_argument("--pool_ratio",   type=float, default=0.5)
    parser.add_argument("--lambda_reg",   type=float, default=0.01)
    parser.add_argument("--lambda_smooth",type=float, default=0.005)
    parser.add_argument("--patience",     type=int,   default=30)
    parser.add_argument("--use_toppool",  action="store_true", default=True)
    parser.add_argument("--use_reg",      action="store_true", default=True)
    parser.add_argument("--analyze",      action="store_true", default=False)
    parser.add_argument("--seed",         type=int,   default=42)

    args   = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(args.seed)
    print(f"\nDevice: {device}")

    if args.task == "graph":
        run_graph_classification(args, device)
    elif args.task == "node":
        run_node_classification(args, device)
    elif args.task == "pretrain":
        run_pretrain(args, device)