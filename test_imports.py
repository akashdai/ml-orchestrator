import sys
print(f"Python: {sys.version}")

packages = [
    'pandas', 'numpy', 'sklearn', 'xgboost', 'lightgbm', 
    'catboost', 'optuna', 'mlflow', 'matplotlib', 'seaborn',
    'plotly', 'shap', 'imblearn', 'category_encoders',
    'yaml', 'joblib', 'tqdm', 'colorama', 'umap', 'hdbscan'
]

for pkg in packages:
    try:
        __import__(pkg)
        print(f"✅ {pkg}")
    except ImportError:
        print(f"❌ {pkg} - FAILED")
