# ml-orchestrator
# 🤖 ML Experiment Orchestrator

> **Advanced Automated Machine Learning Platform with Multi-Agent Architecture**

An intelligent, end-to-end automated ML platform that handles the full pipeline — from data ingestion and preprocessing to model training, hyperparameter tuning, experiment tracking, and deployment — through a clean Streamlit web interface.

---

## 📸 Screenshots

### Home — Core Capabilities
<img width="1918" height="908" alt="Image" src="https://github.com/user-attachments/assets/c862fd47-c125-4547-a07b-9d21bc805470" />

### Run Experiment — Data Upload & Preview
<img width="1917" height="915" alt="Image" src="https://github.com/user-attachments/assets/89daa97a-e9db-4aea-9e94-c4418dd6220e" />

### Supervised Learning Results
![Supervised Results](https://github.com/akashdai/ml-orchestrator/blob/main/screenshots/supervised_results.png?raw=true)

### Unsupervised Clustering Results
![Unsupervised Results](https://github.com/akashdai/ml-orchestrator/blob/main/screenshots/unsupervised_results.png?raw=true)

---

## ✨ Features

### 🎯 Supervised Learning
- **12+ Advanced Models** — XGBoost, LightGBM, CatBoost, Random Forest, Ridge, and more
- Auto task detection (Classification vs Regression)
- Stratified train/test split with cross-validation
- Optuna-powered hyperparameter tuning (Bayesian Optimization)
- SHAP feature importance visualizations
- Confusion matrix and regression result plots

### 🔍 Unsupervised Learning
- **4 Clustering Algorithms** — KMeans, DBSCAN, Agglomerative, GaussianMixture
- **3 Dimensionality Reduction** techniques — PCA, t-SNE, UMAP
- Silhouette score evaluation
- 2D cluster scatter plot visualizations

### ⚡ Auto Optimization
- Optuna-powered Bayesian hyperparameter search
- Configurable trial count and timeout via `config.yaml`
- Multi-objective tuning support

### 📊 Experiment Tracking
- MLflow integration for run logging
- Metrics, parameters, and model artifacts stored per experiment
- Configurable tracking URI

---

## 🛠️ Technology Stack

| Layer | Libraries |
|---|---|
| **Core ML** | Scikit-learn, XGBoost, LightGBM, CatBoost |
| **Unsupervised** | UMAP, HDBSCAN, Scikit-learn clustering |
| **Optimization** | Optuna (Bayesian + multi-objective) |
| **Tracking** | MLflow |
| **Visualization** | Matplotlib, Seaborn, Plotly, SHAP |
| **Data** | Pandas, NumPy, imbalanced-learn, category-encoders |
| **UI** | Streamlit |
| **Serialization** | Joblib, PyYAML |

---

## 📁 Project Structure

```
ml-orchestrator/
│
├── app.py                      # Streamlit web application
├── orchestrator.py             # Core MLOrchestrator class
├── config.yaml                 # Global configuration
├── requirements.txt            # Python dependencies
│
├── ml-orchestrator/src/        # Source modules
│   ├── agents/                 # Multi-agent components
│   │   ├── data_agent.py       # Data loading & inspection
│   │   ├── task_router.py      # Supervised/unsupervised routing
│   │   ├── preprocessing_agent.py
│   │   └── feature_selection_agent.py
│   ├── supervised/             # Supervised pipeline
│   │   ├── task_detector.py
│   │   ├── trainer.py
│   │   ├── tuner.py
│   │   └── evaluator.py
│   ├── unsupervised/           # Unsupervised pipeline
│   │   ├── clustering.py
│   │   └── dimensionality_reduction.py
│   ├── visualization/          # Plot generation
│   │   ├── supervised_viz.py
│   │   └── unsupervised_viz.py
│   ├── tracking/               # MLflow experiment tracker
│   └── core/                   # Config & logging utilities
│
├── run_classification.py       # CLI runner — classification
├── run_regression.py           # CLI runner — regression
├── run_unsupervised.py         # CLI runner — unsupervised
├── run_interactive.py          # CLI runner — interactive mode
├── test_orchestrator.py        # Basic test suite
│
├── data/samples/               # Sample datasets
├── experiments/visualizations/ # Generated plots (gitignored)
└── models/                     # Saved models & pipelines (gitignored)
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/akashdai/ml-orchestrator.git
cd ml-orchestrator
```

### 2. Create a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the Streamlit app

```bash
streamlit run app.py
```

---

## 🖥️ Usage

### Via Streamlit UI

1. Open the app in your browser (default: `http://localhost:8501`)
2. Navigate to **Run Experiment**
3. Upload your CSV dataset
4. Select **Supervised** or **Unsupervised** learning
5. For supervised: choose your target column
6. Click **Run Experiment** and explore results under **Results**

### Via Python API

```python
from orchestrator import MLOrchestrator

orchestrator = MLOrchestrator()

# Supervised — auto-detects classification vs regression
result = orchestrator.run(
    'data/your_dataset.csv',
    target_column='target',
    pipeline_type='supervised'
)

# Unsupervised — clustering + dimensionality reduction
result = orchestrator.run(
    'data/your_dataset.csv',
    pipeline_type='unsupervised'
)

# Interactive — prompts you to choose target column and pipeline type
result = orchestrator.run('data/your_dataset.csv')
```

### Via CLI scripts

```bash
python run_classification.py
python run_regression.py
python run_unsupervised.py
python run_interactive.py
```

---

## ⚙️ Configuration

Edit `config.yaml` to control pipeline behaviour:

```yaml
preprocessing:
  missing_threshold: 0.5      # Drop columns with >50% missing values
  outlier_method: "iqr"       # Options: "iqr" or "zscore"
  handle_imbalance: true      # Apply SMOTE for imbalanced classification

feature_selection:
  enabled: true
  max_features: 30

supervised:
  cv_folds: 5
  tuning:
    n_trials: 50              # Optuna hyperparameter trials
    timeout: 1800             # Max tuning time in seconds

unsupervised:
  dimensionality_reduction: [PCA, TSNE, UMAP]
  clustering: [KMeans, DBSCAN, AgglomerativeClustering, GaussianMixture]
  n_clusters_range: [2, 10]

mlflow:
  tracking_uri: "./experiments/mlruns"
  experiment_name: "ML_Orchestrator"
```

---

## 📊 Output

After a successful experiment run:

| Output | Location |
|---|---|
| Best trained model | `models/supervised/<model_name>_best.pkl` |
| Preprocessing pipeline | `models/supervised/pipeline.pkl` |
| Unsupervised results | `models/unsupervised/results.pkl` |
| Visualizations | `experiments/visualizations/` |
| MLflow experiment logs | `experiments/mlruns/` |

---

## 🔬 Supported Models

### Classification & Regression
XGBoost · LightGBM · CatBoost · Random Forest · Extra Trees · Gradient Boosting · Ridge · Lasso · ElasticNet · Logistic Regression · SVM · KNN · Neural Networks (MLP)

### Clustering
KMeans · DBSCAN · Agglomerative Clustering · Gaussian Mixture Model

### Dimensionality Reduction
PCA · t-SNE · UMAP

---

## 📋 Requirements

- Python 3.9+
- See `requirements.txt` for full dependency list

---

## 👤 Author

**Akash Das**
- GitHub: [@akashdai](https://github.com/akashdai)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
