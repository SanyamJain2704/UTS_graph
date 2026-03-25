"""
analysis.py — Layer-wise UTS analysis (Section 4.4).

Provides:
  UTSAnalyzer.track()    : collect UTS vectors across layers for a dataset
  UTSAnalyzer.plot_evolution()  : per-feature evolution across layers
  UTSAnalyzer.plot_pca()        : PCA of UTS vectors at each layer
  UTSAnalyzer.oversmoothing_index() : scalar measure of representation collapse
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# UTS feature names (must match EmbeddingUTS.DIM = 14)
UTS_FEATURE_NAMES = [
    "H0 mean lifetime", "H0 persistence entropy",
    "H1 mean lifetime", "H1 persistence entropy",
    "Betti β₀", "Betti β₁",
    "Mean NN dist", "Std NN dist",
    "Global spread", "Intrinsic dim est.",
    "Spectral gap λ₁", "λ₂", "λ₃",
    "Spectral entropy",
]


class UTSAnalyzer:
    """
    Collects and analyses layer-wise UTS vectors from a trained model.

    Usage:
        analyzer = UTSAnalyzer(model, device)
        analyzer.track(loader)           # forward passes, collect uts_list
        analyzer.plot_evolution()
        analyzer.plot_pca()
        osi = analyzer.oversmoothing_index()
    """

    def __init__(self, model, device):
        self.model  = model
        self.device = device
        self.data   = {}          # layer_idx → list of (B, UTS_DIM) arrays

    # ------------------------------------------------------------------
    @torch.no_grad()
    def track(self, loader, nx_graphs_loader=None):
        """
        Run inference over loader, collect uts_list from each batch.

        Args:
            loader           : PyG DataLoader
            nx_graphs_loader : optional, list of nx.Graph lists per batch
                               (pass None to skip reg losses)
        """
        self.model.eval()
        self.data.clear()

        for batch_idx, data in enumerate(loader):
            data = data.to(self.device)
            _, aux = self.model(data)
            uts_list = aux["uts_list"]           # list of (B, UTS_DIM) tensors

            for layer_idx, sig in enumerate(uts_list):
                if layer_idx not in self.data:
                    self.data[layer_idx] = []
                self.data[layer_idx].append(sig.cpu().numpy())

        # Concatenate across batches
        for k in self.data:
            self.data[k] = np.concatenate(self.data[k], axis=0)  # (N_graphs, UTS_DIM)

        print(f"[UTSAnalyzer] Tracked {len(self.data)} layers, "
              f"{next(iter(self.data.values())).shape[0]} graphs each.")

    # ------------------------------------------------------------------
    def plot_evolution(self, feature_indices=None, save_path=None):
        """
        Plot the mean ± std of selected UTS features across GNN layers.

        Args:
            feature_indices : list of int indices into UTS_FEATURE_NAMES.
                              Defaults to [0,2,4,5,6,10,13] (key features).
        """
        if not self.data:
            raise RuntimeError("Call track() first.")

        feat_idx = feature_indices or [0, 2, 4, 5, 6, 10, 13]
        layers   = sorted(self.data.keys())
        n_feats  = len(feat_idx)

        fig, axes = plt.subplots(
            1, n_feats, figsize=(3.5 * n_feats, 4), sharey=False
        )
        if n_feats == 1:
            axes = [axes]

        colors = plt.cm.viridis(np.linspace(0.2, 0.85, n_feats))

        for ax, fi, col in zip(axes, feat_idx, colors):
            means = [self.data[l][:, fi].mean() for l in layers]
            stds  = [self.data[l][:, fi].std()  for l in layers]
            ax.plot(layers, means, marker="o", color=col, linewidth=2)
            ax.fill_between(
                layers,
                np.array(means) - np.array(stds),
                np.array(means) + np.array(stds),
                alpha=0.25, color=col,
            )
            ax.set_title(UTS_FEATURE_NAMES[fi], fontsize=9)
            ax.set_xlabel("GNN Layer")
            ax.set_xticks(layers)

        axes[0].set_ylabel("Feature value")
        fig.suptitle("UTS Feature Evolution Across GNN Layers (§4.4)",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    # ------------------------------------------------------------------
    def plot_pca(self, save_path=None):
        """
        PCA of the UTS signature space at each layer.
        Useful for visualising structural evolution.
        """
        if not self.data:
            raise RuntimeError("Call track() first.")

        layers  = sorted(self.data.keys())
        n_layers = len(layers)
        cols    = min(n_layers, 4)
        rows    = (n_layers + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols,
                                 figsize=(4 * cols, 4 * rows))
        axes = np.array(axes).flatten()

        scaler = StandardScaler()
        pca    = PCA(n_components=2)

        # Fit PCA on all layers stacked
        all_data = np.concatenate([self.data[l] for l in layers], axis=0)
        scaler.fit(all_data)
        pca.fit(scaler.transform(all_data))

        cmap = plt.cm.plasma

        for ax, layer in zip(axes, layers):
            X  = scaler.transform(self.data[layer])
            Z  = pca.transform(X)                   # (N, 2)
            sc = ax.scatter(Z[:, 0], Z[:, 1],
                            c=np.arange(len(Z)),
                            cmap=cmap, s=12, alpha=0.7)
            ax.set_title(f"Layer {layer}", fontsize=10)
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")

        # Hide unused axes
        for ax in axes[len(layers):]:
            ax.set_visible(False)

        fig.suptitle("PCA of UTS Signature Space per Layer (§4.4)",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    # ------------------------------------------------------------------
    def oversmoothing_index(self) -> dict:
        """
        Compute a scalar over-smoothing index (OSI) per layer.

        OSI = 1 - (mean pairwise cosine distance between UTS vectors)
        Higher OSI → more collapse / oversmoothing.

        Returns:
            dict: layer_idx → OSI value
        """
        from sklearn.metrics.pairwise import cosine_distances
        osi = {}
        for layer, X in self.data.items():
            if X.shape[0] < 2:
                osi[layer] = 0.0
                continue
            D = cosine_distances(X)
            # Mean of upper triangle
            idx = np.triu_indices(len(D), k=1)
            mean_dist = D[idx].mean()
            osi[layer] = float(1.0 - mean_dist)
        return osi

    # ------------------------------------------------------------------
    def print_summary(self):
        if not self.data:
            print("No data tracked yet. Call track() first.")
            return

        osi = self.oversmoothing_index()
        print("\n── UTS Layer Analysis Summary ──────────────────────────")
        print(f"  Layers tracked : {sorted(self.data.keys())}")
        print(f"  Graphs         : {next(iter(self.data.values())).shape[0]}")
        print(f"  UTS dim        : {next(iter(self.data.values())).shape[1]}")
        print()
        print(f"  {'Layer':<8} {'OSI':>8}   (higher = more oversmoothing)")
        print("  " + "-" * 30)
        for l in sorted(osi.keys()):
            bar = "█" * int(osi[l] * 20)
            print(f"  {l:<8} {osi[l]:>8.4f}   {bar}")
        print()

        # Key UTS stats at first and last layer
        layers = sorted(self.data.keys())
        print(f"  Feature drift  (layer {layers[0]} → {layers[-1]}):")
        X0 = self.data[layers[0]]
        XL = self.data[layers[-1]]
        for fi, name in enumerate(UTS_FEATURE_NAMES):
            d = abs(XL[:, fi].mean() - X0[:, fi].mean())
            print(f"    {name:<30}  Δμ = {d:.4f}")
        print("────────────────────────────────────────────────────────\n")