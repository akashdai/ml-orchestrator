import numpy as np

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

try:
    from ..core.logger import ColoredLogger
except ImportError:  # pragma: no cover - fallback for direct script execution
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from core.logger import ColoredLogger


class ClusteringAgent:
    def __init__(self, config=None):
        self.config = config or {}
        self.logger = ColoredLogger()

    def cluster(self, X):
        self.logger.section("CLUSTERING")

        results = {}

        best_k, _ = self._find_optimal_k(X)

        self.logger.info("Running KMeans...")
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        results['KMeans'] = {
            'labels': labels,
            'n_clusters': best_k,
            'silhouette': self._safe_silhouette(X, labels),
        }
        self.logger.success(f"KMeans: {best_k} clusters, sil={results['KMeans']['silhouette']:.3f}")

        self.logger.info("Running DBSCAN...")
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        labels = dbscan.fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        if n_clusters > 1 and -1 not in labels:
            sil = self._safe_silhouette(X, labels)
        elif n_clusters > 1:
            mask = labels != -1
            sil = self._safe_silhouette(X[mask], labels[mask]) if mask.sum() > 1 else 0.0
        else:
            sil = 0.0

        results['DBSCAN'] = {
            'labels': labels,
            'n_clusters': n_clusters,
            'silhouette': sil,
        }
        self.logger.success(f"DBSCAN: {n_clusters} clusters")

        self.logger.info("Running Agglomerative...")
        agg = AgglomerativeClustering(n_clusters=best_k)
        labels = agg.fit_predict(X)
        results['Agglomerative'] = {
            'labels': labels,
            'n_clusters': best_k,
            'silhouette': self._safe_silhouette(X, labels),
        }
        self.logger.success(f"Agglomerative: {best_k} clusters")

        self.logger.info("Running Gaussian Mixture...")
        gmm = GaussianMixture(n_components=best_k, random_state=42)
        labels = gmm.fit_predict(X)
        results['GaussianMixture'] = {
            'labels': labels,
            'n_clusters': best_k,
            'silhouette': self._safe_silhouette(X, labels),
        }
        self.logger.success(f"Gaussian Mixture: {best_k} clusters")

        return results

    def _find_optimal_k(self, X):
        if len(X) < 4:
            return 1, 0.0

        max_k = min(10, len(X) - 1)
        if max_k < 2:
            return 1, 0.0

        range_k = range(2, max_k + 1)
        silhouettes = []

        for k in range_k:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            silhouettes.append(self._safe_silhouette(X, labels))

        if not silhouettes:
            return 1, 0.0

        best_k = range_k[np.argmax(silhouettes)]
        return best_k, max(silhouettes)

    def _safe_silhouette(self, X, labels):
        if len(np.unique(labels)) < 2 or len(X) < 4:
            return 0.0
        try:
            return float(silhouette_score(X, labels))
        except ValueError:
            return 0.0
