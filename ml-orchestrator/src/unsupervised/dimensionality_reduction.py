import importlib

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.impute import SimpleImputer

try:
    umap = importlib.import_module('umap')
except ImportError:  # pragma: no cover - optional dependency may be absent
    umap = None

try:
    from ..core.logger import ColoredLogger
except ImportError:  # pragma: no cover - fallback for direct script execution
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from core.logger import ColoredLogger


class DimensionalityReductionAgent:
    def __init__(self, config=None):
        self.config = config or {}
        self.logger = ColoredLogger()

    def reduce(self, X, n_components=2):
        self.logger.section("DIMENSIONALITY REDUCTION")

        results = {}

        # Convert to numpy and ensure no NaNs reach any reducer
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = X.astype(np.float64)

        # Final safety net: impute any remaining NaNs column-by-column
        if np.isnan(X).any():
            self.logger.warning("NaNs detected before reduction — applying final imputation")
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)

        # Drop columns that are still all-NaN after imputation (edge case)
        col_mask = ~np.all(np.isnan(X), axis=0)
        X = X[:, col_mask]

        self.logger.info("Running PCA...")
        n_comp = min(n_components, X.shape[1], X.shape[0])
        pca = PCA(n_components=n_comp, random_state=42)
        pca_result = pca.fit_transform(X)
        results['PCA'] = pca_result
        var_explained = pca.explained_variance_ratio_.sum()
        self.logger.success(f"PCA: {var_explained:.3f} variance explained")

        self.logger.info("Running t-SNE...")
        perplexity = min(30, max(5, X.shape[0] // 10))
        tsne = TSNE(n_components=n_comp, random_state=42, perplexity=perplexity)
        tsne_result = tsne.fit_transform(X)
        results['t-SNE'] = tsne_result
        self.logger.success("t-SNE complete")

        if umap is not None:
            self.logger.info("Running UMAP...")
            umap_model = umap.UMAP(n_components=n_comp, random_state=42)
            umap_result = umap_model.fit_transform(X)
            results['UMAP'] = umap_result
            self.logger.success("UMAP complete")
        else:
            self.logger.warning("UMAP not installed; skipping")

        return results
