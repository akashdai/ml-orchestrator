import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent / 'ml-orchestrator'
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / 'src'))

try:
    from src.core.config import Config
    from src.core.logger import ColoredLogger
    from src.agents.data_agent import DataAgent
    from src.agents.task_router import TaskRouter
    from src.agents.preprocessing_agent import PreprocessingAgent
    from src.agents.feature_selection_agent import FeatureSelectionAgent
    from src.supervised.task_detector import TaskDetector
    from src.supervised.trainer import ModelTrainer
    from src.supervised.tuner import HyperparameterTuner
    from src.supervised.evaluator import ModelEvaluator
    from src.unsupervised.dimensionality_reduction import DimensionalityReductionAgent
    from src.unsupervised.clustering import ClusteringAgent
    from src.visualization.supervised_viz import SupervisedVisualizer
    from src.visualization.unsupervised_viz import UnsupervisedVisualizer
    from src.tracking.experiment_tracker import ExperimentTracker
except ImportError:  # pragma: no cover - fallback for direct script execution
    sys.path.insert(0, str(PROJECT_ROOT / 'src'))
    from core.config import Config
    from core.logger import ColoredLogger
    from agents.data_agent import DataAgent
    from agents.task_router import TaskRouter
    from agents.preprocessing_agent import PreprocessingAgent
    from agents.feature_selection_agent import FeatureSelectionAgent
    from supervised.task_detector import TaskDetector
    from supervised.trainer import ModelTrainer
    from supervised.tuner import HyperparameterTuner
    from supervised.evaluator import ModelEvaluator
    from unsupervised.dimensionality_reduction import DimensionalityReductionAgent
    from unsupervised.clustering import ClusteringAgent
    from visualization.supervised_viz import SupervisedVisualizer
    from visualization.unsupervised_viz import UnsupervisedVisualizer
    from tracking.experiment_tracker import ExperimentTracker


class MLOrchestrator:
    """
    Advanced ML Experiment Orchestrator
    Supports both Supervised and Unsupervised Learning
    """

    def __init__(self, config_path='config.yaml'):
        self.config = Config(config_path)
        self.logger = ColoredLogger()

        Path('experiments/visualizations').mkdir(parents=True, exist_ok=True)
        Path('models/supervised').mkdir(parents=True, exist_ok=True)
        Path('models/unsupervised').mkdir(parents=True, exist_ok=True)

    def run(self, file_path, target_column=None, pipeline_type=None):
        """
        Main orchestration method

        Args:
            file_path: Path to dataset
            target_column: Target column name (for supervised)
            pipeline_type: 'supervised' or 'unsupervised' (None for interactive)

        Returns:
            Dictionary with results
        """

        self.logger.section("🚀 ML EXPERIMENT ORCHESTRATOR")

        data_agent = DataAgent(self.config)
        df, data_report = data_agent.load_and_inspect(file_path)

        router = TaskRouter(self.config)
        if pipeline_type is None:
            pipeline_type = router.route()

        if pipeline_type == 'supervised':
            return self._supervised_pipeline(df, target_column)
        return self._unsupervised_pipeline(df)

    def _supervised_pipeline(self, df, target_column):
        """Supervised Learning Pipeline"""

        self.logger.section("SUPERVISED LEARNING PIPELINE")

        if target_column is None:
            target_column = self._select_target_column(df)

        task_detector = TaskDetector(self.config)
        task_type = task_detector.detect(df, target_column)

        preprocessor = PreprocessingAgent(self.config)
        df_clean = preprocessor.preprocess(df, target_column, task_type)

        X = df_clean.drop(columns=[target_column])
        y = df_clean[target_column]

        if task_type == 'classification' and y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
            self.logger.info("Target encoded")

        feature_selector = FeatureSelectionAgent(self.config)
        selected_features = feature_selector.select_features(X, y, task_type)
        X = X[selected_features]

        self.logger.section("TRAIN-TEST SPLIT")
        test_size = self.config.get('supervised.test_size', 0.2)

        if task_type == 'classification':
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

        self.logger.success(f"Train: {X_train.shape}, Test: {X_test.shape}")

        trainer = ModelTrainer(self.config)
        results, best_model_name = trainer.train_all(X_train, y_train, task_type)
        best_model = results[best_model_name]['model']

        tuner = HyperparameterTuner(self.config)
        try:
            best_params = tuner.tune(
                type(best_model), X_train, y_train, task_type, best_model_name
            )
        except Exception as exc:
            self.logger.warning(f"Hyperparameter tuning skipped: {exc}")
            best_params = {}

        if best_params:
            try:
                final_model = type(best_model)(**best_params, random_state=42)
            except TypeError:
                final_model = type(best_model)(**best_params)
            final_model.fit(X_train, y_train)
        else:
            final_model = best_model

        evaluator = ModelEvaluator(self.config)
        metrics, y_pred = evaluator.evaluate(final_model, X_test, y_test, task_type)

        self.logger.section("GENERATING VISUALIZATIONS")

        if task_type == 'classification':
            SupervisedVisualizer.plot_confusion_matrix(y_test, y_pred)
        else:
            SupervisedVisualizer.plot_regression_results(y_test, y_pred)

        if hasattr(final_model, 'feature_importances_'):
            SupervisedVisualizer.plot_feature_importance(
                final_model.feature_importances_,
                X.columns.tolist(),
            )

        tracker = ExperimentTracker(self.config)
        tracker.log_run(
            best_model_name,
            final_model,
            best_params if best_params else {},
            metrics,
        )

        self.logger.section("SAVING MODEL")

        model_path = f'models/supervised/{best_model_name}_best.pkl'
        joblib.dump(final_model, model_path)
        self.logger.success(f"Model saved: {model_path}")

        pipeline_info = {
            'preprocessor': preprocessor.transformers,
            'selected_features': selected_features,
            'task_type': task_type,
            'target_column': target_column,
        }
        joblib.dump(pipeline_info, 'models/supervised/pipeline.pkl')
        self.logger.success("Pipeline saved: models/supervised/pipeline.pkl")

        self.logger.section("✅ SUPERVISED PIPELINE COMPLETE!")

        return {
            'pipeline': 'supervised',
            'task_type': task_type,
            'best_model': best_model_name,
            'best_params': best_params,
            'metrics': metrics,
            'model_path': model_path,
        }

    def _unsupervised_pipeline(self, df):
        """Unsupervised Learning Pipeline"""

        self.logger.section("UNSUPERVISED LEARNING PIPELINE")

        preprocessor = PreprocessingAgent(self.config)
        df_clean = preprocessor.preprocess(df)

        X = df_clean.select_dtypes(include=[np.number]).values
        self.logger.info(f"Working with {X.shape[1]} numeric features")

        dim_reducer = DimensionalityReductionAgent(self.config)
        reduction_results = dim_reducer.reduce(X)

        self.logger.section("GENERATING VISUALIZATIONS")
        UnsupervisedVisualizer.plot_dimensionality_reduction(reduction_results)

        clusterer = ClusteringAgent(self.config)
        clustering_results = clusterer.cluster(X)

        X_pca = reduction_results['PCA']
        UnsupervisedVisualizer.plot_clustering(X_pca, clustering_results)

        self.logger.section("SAVING RESULTS")

        results = {
            'dimensionality_reduction': reduction_results,
            'clustering': clustering_results,
        }

        results_path = 'models/unsupervised/results.pkl'
        joblib.dump(results, results_path)
        self.logger.success(f"Results saved: {results_path}")

        self.logger.section("✅ UNSUPERVISED PIPELINE COMPLETE!")

        return {
            'pipeline': 'unsupervised',
            'dimensionality_reduction': {k: v.shape for k, v in reduction_results.items()},
            'clustering': {k: v['n_clusters'] for k, v in clustering_results.items()},
            'results_path': results_path,
        }

    def _select_target_column(self, df):
        """Interactive target column selection"""

        self.logger.section("SELECT TARGET COLUMN")

        print("\nAvailable columns:")
        for i, col in enumerate(df.columns, 1):
            print(f"{i}. {col}")

        while True:
            try:
                choice = int(input("\nSelect target column number: "))
                if 1 <= choice <= len(df.columns):
                    target_col = df.columns[choice - 1]
                    self.logger.success(f"Selected: {target_col}")
                    return target_col
                print("❌ Invalid number!")
            except ValueError:
                print("❌ Please enter a number!")


if __name__ == '__main__':
    orchestrator = MLOrchestrator()

    print("\n" + "=" * 80)
    print("ML ORCHESTRATOR READY!".center(80))
    print("=" * 80)
    print("\nUsage Examples:")
    print("1. Interactive mode:")
    print("   orchestrator.run('data/your_data.csv')")
    print("\n2. Supervised learning:")
    print("   orchestrator.run('data/your_data.csv', target_column='target', pipeline_type='supervised')")
    print("\n3. Unsupervised learning:")
    print("   orchestrator.run('data/your_data.csv', pipeline_type='unsupervised')")
    print("=" * 80)
