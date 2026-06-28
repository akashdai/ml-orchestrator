import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    classification_report,
)

try:
    from ..core.logger import ColoredLogger
except ImportError:  # pragma: no cover - fallback for direct script execution
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from core.logger import ColoredLogger


class ModelEvaluator:
    def __init__(self, config=None):
        self.config = config or {}
        self.logger = ColoredLogger()

    def evaluate(self, model, X_test, y_test, task_type):
        self.logger.section("MODEL EVALUATION")

        y_pred = model.predict(X_test)

        if task_type == 'classification':
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            }

            for name, value in metrics.items():
                self.logger.metric(name.capitalize(), value)

            print("\n" + classification_report(y_test, y_pred, zero_division=0))
        else:
            metrics = {
                'r2': r2_score(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
            }

            for name, value in metrics.items():
                self.logger.metric(name.upper(), value)

        return metrics, y_pred
