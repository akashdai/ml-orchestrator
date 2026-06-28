from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


class SupervisedVisualizer:
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, save_path='experiments/visualizations/confusion_matrix.png'):
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix', fontsize=16)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {save_path}")

    @staticmethod
    def plot_feature_importance(importances, feature_names, save_path='experiments/visualizations/feature_importance.png'):
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        importances = np.asarray(importances, dtype=float)
        feature_names = list(feature_names)
        indices = np.argsort(importances)[-20:]

        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), importances[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Importance')
        plt.title('Top 20 Feature Importances')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {save_path}")

    @staticmethod
    def plot_regression_results(y_true, y_pred, save_path='experiments/visualizations/regression.png'):
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        plt.figure(figsize=(10, 8))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Predictions vs Actual')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {save_path}")
