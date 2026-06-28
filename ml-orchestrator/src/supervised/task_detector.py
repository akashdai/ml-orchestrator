import pandas as pd

try:
    from ..core.logger import ColoredLogger
except ImportError:  # pragma: no cover - fallback for direct script execution
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from core.logger import ColoredLogger


class TaskDetector:
    def __init__(self, config=None):
        self.config = config or {}
        self.logger = ColoredLogger()

    def detect(self, df, target_col):
        self.logger.section("TASK DETECTION")

        target = df[target_col]
        is_numeric = pd.api.types.is_numeric_dtype(target)

        if not is_numeric or target.nunique() <= 20:
            task_type = 'classification'
            n_classes = target.nunique()
            self.logger.success(f"Task: CLASSIFICATION ({n_classes} classes)")
        else:
            task_type = 'regression'
            self.logger.success("Task: REGRESSION")

        return task_type
