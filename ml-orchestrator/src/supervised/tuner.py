from sklearn.model_selection import cross_val_score

try:
    import optuna
except ImportError:  # pragma: no cover - optional dependency may be absent
    optuna = None

try:
    from ..core.logger import ColoredLogger
    from .model_factory import ModelFactory
except ImportError:  # pragma: no cover - fallback for direct script execution
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from core.logger import ColoredLogger
    from supervised.model_factory import ModelFactory


class HyperparameterTuner:
    def __init__(self, config=None):
        self.config = config or {}
        self.logger = ColoredLogger()

    def _get_config_value(self, key, default=None):
        value = self.config
        for part in key.split('.'):
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        return value

    def tune(self, model_class, X, y, task_type, model_name):
        self.logger.section("HYPERPARAMETER TUNING")

        if optuna is None:
            self.logger.warning("optuna is not installed; skipping hyperparameter tuning")
            return {}

        param_space = ModelFactory.get_param_space(model_name)

        if not param_space:
            self.logger.warning(f"No tuning config for {model_name}")
            return {}

        n_trials = self._get_config_value('supervised.tuning.n_trials', 50)
        cv_folds = self._get_config_value('supervised.tuning.cv_folds', 3)

        def objective(trial):
            params = {}
            for param, bounds in param_space.items():
                low, high = bounds
                if isinstance(low, int) and isinstance(high, int):
                    params[param] = trial.suggest_int(param, low, high)
                else:
                    params[param] = trial.suggest_float(param, low, high)

            try:
                model = model_class(**params, random_state=42)
            except TypeError:
                model = model_class(**params)

            if task_type == 'classification':
                score = cross_val_score(
                    model,
                    X,
                    y,
                    cv=cv_folds,
                    scoring='accuracy',
                    n_jobs=-1,
                ).mean()
            else:
                score = cross_val_score(
                    model,
                    X,
                    y,
                    cv=cv_folds,
                    scoring='r2',
                    n_jobs=-1,
                ).mean()

            return score

        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42),
        )

        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_params = study.best_params
        self.logger.success(f"Best score: {study.best_value:.4f}")

        for param, value in best_params.items():
            self.logger.info(f"{param}: {value}")

        return best_params
