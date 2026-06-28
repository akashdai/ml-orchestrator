from datetime import datetime

try:
    import mlflow
    import mlflow.sklearn
    from mlflow.exceptions import MlflowException
except ImportError:  # pragma: no cover - optional dependency may be absent
    mlflow = None
    MlflowException = Exception

try:
    from ..core.logger import ColoredLogger
except ImportError:  # pragma: no cover - fallback for direct script execution
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from core.logger import ColoredLogger


class ExperimentTracker:
    def __init__(self, config=None):
        self.config = config or {}
        self.logger = ColoredLogger()

        if mlflow is None:
            self.logger.warning("mlflow is not installed; tracking will be disabled")
            self.mlflow = None
            return

        tracking_uri = self.config.get('mlflow.tracking_uri') or 'sqlite:///mlruns/mlflow.db'
        experiment_name = self.config.get('mlflow.experiment_name') or 'ml-orchestrator'

        try:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
        except MlflowException as exc:
            msg = str(exc).lower()
            if 'filesystem tracking backend' in msg or 'maintenance mode' in msg or 'mlflow runstore' in msg:
                fallback_uri = self.config.get('mlflow.sqlite_uri') or 'sqlite:///mlflow.db'
                self.logger.warning(
                    "MLflow filesystem backend unsupported or in maintenance mode; "
                    f"falling back to SQLite tracking at {fallback_uri}"
                )
                mlflow.set_tracking_uri(fallback_uri)
                mlflow.set_experiment(experiment_name)
            else:
                self.logger.warning(
                    f"MLflow unavailable due to tracking error: {exc}; tracking will be disabled"
                )
                self.mlflow = None
                return
        except Exception as exc:
            self.logger.warning(
                f"MLflow unavailable due to unexpected error: {exc}; tracking will be disabled"
            )
            self.mlflow = None
            return

        self.mlflow = mlflow

    def log_run(self, model_name, model, params, metrics, artifacts=None):
        if self.mlflow is None:
            self.logger.warning(f"MLflow unavailable; skipping log for {model_name}")
            return None

        artifacts = artifacts or {}
        with self.mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%H%M%S')}"):
            self.mlflow.log_params(params)
            self.mlflow.log_metrics(metrics)
            self.mlflow.sklearn.log_model(model, "model")

            for name, path in artifacts.items():
                if path and isinstance(path, str):
                    self.mlflow.log_artifact(path, artifact_path=name)

            self.mlflow.set_tag("model_type", model_name)

        self.logger.success(f"Logged: {model_name}")
        return True
