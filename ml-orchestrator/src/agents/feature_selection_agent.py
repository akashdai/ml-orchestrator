import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

try:
    from ..core.logger import ColoredLogger
except ImportError:  # pragma: no cover - fallback for direct script execution
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from core.logger import ColoredLogger


class FeatureSelectionAgent:
    def __init__(self, config):
        self.config = config or {}
        self.logger = ColoredLogger()

    def select_features(self, X, y, task_type):
        if not self.config.get('feature_selection.enabled', True):
            return X.columns.tolist()

        self.logger.section("FEATURE SELECTION")

        max_features = min(
            self.config.get('feature_selection.max_features', 30),
            X.shape[1]
        )

        if task_type == 'classification':
            mi_scores = mutual_info_classif(X, y, random_state=42)
        else:
            mi_scores = mutual_info_regression(X, y, random_state=42)

        mi_scores = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

        if task_type == 'classification':
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

        model.fit(X, y)
        tree_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

        top_mi = set(mi_scores.head(max_features).index)
        top_tree = set(tree_imp.head(max_features).index)
        selected = list(top_mi.union(top_tree))[:max_features]

        self.logger.success(f"Selected {len(selected)} features")
        return selected
