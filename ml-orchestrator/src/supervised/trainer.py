import time

from sklearn.model_selection import cross_val_score

try:
    from ..core.logger import ColoredLogger
    from .model_factory import ModelFactory
except ImportError:  # pragma: no cover - fallback for direct script execution
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from core.logger import ColoredLogger
    from supervised.model_factory import ModelFactory


class ModelTrainer:
    def __init__(self, config=None):
        self.config = config or {}
        self.logger = ColoredLogger()
        self.results = {}

    def _get_config_value(self, key, default=None):
        value = self.config
        for part in key.split('.'):
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        return value

    def train_all(self, X, y, task_type):
        self.logger.section("MODEL TRAINING")
        self.results = {}

        models = ModelFactory.get_models(task_type)
        cv_folds = self._get_config_value('supervised.cv_folds', 5)

        for name, model in models.items():
            self.logger.info(f"Training {name}...")
            start = time.time()

            try:
                if task_type == 'classification':
                    scores = cross_val_score(
                        model,
                        X,
                        y,
                        cv=cv_folds,
                        scoring='accuracy',
                        n_jobs=-1,
                    )
                    metric = 'Accuracy'
                else:
                    scores = cross_val_score(
                        model,
                        X,
                        y,
                        cv=cv_folds,
                        scoring='r2',
                        n_jobs=-1,
                    )
                    metric = 'R2'

                model.fit(X, y)
                elapsed = time.time() - start

                self.results[name] = {
                    'model': model,
                    'cv_mean': scores.mean(),
                    'cv_std': scores.std(),
                    'time': elapsed,
                    'metric': metric,
                }

                self.logger.metric(f"{name} {metric}", scores.mean())

            except Exception as exc:
                self.logger.error(f"{name} failed: {str(exc)}")

        if not self.results:
            self.logger.warning("No models completed successfully")
            return self.results, None

        best_name = max(self.results, key=lambda x: self.results[x]['cv_mean'])
        self.logger.success(f"Best Model: {best_name}")

        return self.results, best_name
