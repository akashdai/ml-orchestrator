from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class UnsupervisedVisualizer:
    @staticmethod
    def plot_dimensionality_reduction(results, save_path='experiments/visualizations/dim_reduction.png'):
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        methods = list(results.items())
        n_plots = len(methods)
        fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
        if n_plots == 1:
            axes = [axes]

        for idx, (method, data) in enumerate(methods):
            data = np.asarray(data)
            if data.ndim != 2 or data.shape[1] < 2:
                raise ValueError(f"{method} must contain 2D data")
            axes[idx].scatter(data[:, 0], data[:, 1], alpha=0.5)
            axes[idx].set_title(f'{method}', fontsize=14)
            axes[idx].set_xlabel('Component 1')
            axes[idx].set_ylabel('Component 2')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {save_path}")

    @staticmethod
    def plot_clustering(X_2d, results, save_path='experiments/visualizations/clustering.png'):
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        X_2d = np.asarray(X_2d)
        if X_2d.ndim != 2 or X_2d.shape[1] < 2:
            raise ValueError('X_2d must be a 2D array')

        methods = list(results.items())
        n_plots = len(methods)
        n_cols = 2
        n_rows = int(np.ceil(n_plots / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        axes = np.ravel(axes)

        for idx, (method, result) in enumerate(methods):
            labels = result['labels']
            n_clusters = result['n_clusters']
            sil = result['silhouette']

            scatter = axes[idx].scatter(
                X_2d[:, 0],
                X_2d[:, 1],
                c=labels,
                cmap='viridis',
                alpha=0.6,
            )
            axes[idx].set_title(f'{method}\nClusters: {n_clusters}, Silhouette: {sil:.3f}')
            plt.colorbar(scatter, ax=axes[idx])

        for ax in axes[n_plots:]:
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {save_path}")
