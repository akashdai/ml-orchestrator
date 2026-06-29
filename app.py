import io
import time
import traceback
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

warnings.filterwarnings('ignore')

from orchestrator import MLOrchestrator

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="ML Experiment Orchestrator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CSS ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    .stApp {
        background: linear-gradient(to right, #1e5128 0%, #1e5128 18%, #d6d6d6 18%, #d6d6d6 100%);
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e5128 0%, #2d6a3e 100%);
    }

    [data-testid="stSidebar"] * { color: white !important; }

    .main .block-container {
        background: rgba(255,255,255,0.97);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
    }

    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        text-align: center;
        color: #1e5128;
        padding: 1rem 0;
        margin-bottom: 0.5rem;
    }

    .sub-header {
        text-align: center;
        color: #444;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    .feature-card {
        background: linear-gradient(135deg, #1e5128 0%, #2d6a3e 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 25px rgba(30,81,40,0.3);
        height: 100%;
    }

    .feature-card h3 { color: #e0e0e0 !important; font-size: 1.3rem; }
    .feature-card p  { color: #d0d0d0; margin: 0.3rem 0; font-size: 0.95rem; }

    .info-box {
        background: #ffffff;
        border-left: 5px solid #1e5128;
        padding: 1.2rem 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }

    .info-box h3 { color: #1e5128 !important; margin-bottom: 0.8rem; font-weight: 600; }
    .info-box p, .info-box li, .info-box ol { color: #222 !important; font-size: 1rem; line-height: 1.7; }

    .success-box {
        background: #f0f9f4;
        border-left: 5px solid #28a745;
        padding: 1.2rem 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }

    .success-box h3 { color: #1e5128 !important; font-weight: 600; }
    .success-box li, .success-box ol { color: #222 !important; font-size: 1rem; }

    .warning-box {
        background: #fffbf0;
        border-left: 5px solid #ffc107;
        padding: 1.2rem 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }

    .warning-box h3 { color: #1e5128 !important; font-weight: 600; }
    .warning-box li, .warning-box ol { color: #222 !important; font-size: 1rem; }

    .stButton > button {
        background: linear-gradient(135deg, #1e5128 0%, #2d6a3e 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 10px;
        padding: 0.65rem 2rem;
        font-size: 1rem;
        box-shadow: 0 4px 15px rgba(30,81,40,0.3);
    }

    [data-testid="stMetricValue"] { font-size: 1.8rem; color: #1e5128; font-weight: 700; }

    .tech-badge {
        display: inline-block;
        background: linear-gradient(135deg, #1e5128 0%, #2d6a3e 100%);
        color: white;
        padding: 0.4rem 1.2rem;
        border-radius: 20px;
        margin: 0.4rem;
        font-weight: 600;
        font-size: 0.9rem;
    }

    .feature-list { list-style: none; padding-left: 0; }
    .feature-list li {
        padding: 0.5rem 0;
        border-bottom: 1px solid #eee;
        color: #222 !important;
        font-size: 0.98rem;
    }
    .feature-list li:before { content: "✓ "; color: #1e5128; font-weight: bold; }

    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        border-top: 2px solid #e0e0e0;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE ====================
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = MLOrchestrator()
if 'results' not in st.session_state:
    st.session_state.results = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'n_experiments' not in st.session_state:
    st.session_state.n_experiments = 0

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("## 🤖 ML Orchestrator")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🏠 Home", "📊 Run Experiment", "📈 Results", "📚 Documentation"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### 📊 Statistics")
    models_path = Path("models/supervised")
    n_models = len(list(models_path.glob("*.pkl"))) if models_path.exists() else 0
    st.metric("Experiments", st.session_state.n_experiments)
    st.metric("Saved Models", n_models)

    st.markdown("---")
    st.markdown("### 🔗 Quick Links")
    st.markdown("🐙 [GitHub](https://github.com/akashdai/ml-orchestrator)")

# ==================== HOME ====================
if page == "🏠 Home":

    st.markdown('<h1 class="main-header">🤖 ML Experiment Orchestrator</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced Automated Machine Learning Platform with Multi-Agent Architecture</p>', unsafe_allow_html=True)

    st.markdown("## ✨ Core Capabilities")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""<div class="feature-card">
            <h2>🎯</h2><h3>Supervised Learning</h3>
            <p>Classification & Regression</p>
            <p><strong>12+ Advanced Models</strong></p>
            <p>XGBoost • LightGBM • Random Forest</p>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("""<div class="feature-card">
            <h2>🔍</h2><h3>Unsupervised Learning</h3>
            <p>Clustering & Dimensionality Reduction</p>
            <p><strong>7+ Algorithms</strong></p>
            <p>KMeans • DBSCAN • PCA • t-SNE</p>
        </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown("""<div class="feature-card">
            <h2>⚡</h2><h3>Auto Optimization</h3>
            <p>Hyperparameter Tuning</p>
            <p><strong>Optuna-Powered</strong></p>
            <p>Bayesian Optimization</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## 🎯 What is ML Experiment Orchestrator?")
    st.markdown("""<div class="info-box">
        <p>The <strong>ML Experiment Orchestrator</strong> is an intelligent, end-to-end automated ML platform
        built on a <strong>multi-agent architecture</strong>. It automates the entire ML pipeline — from data ingestion
        and preprocessing to model training, hyperparameter tuning, and evaluation.</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## 🛠️ Technology Stack")
    st.markdown("""<div style="text-align:center;padding:1rem;">
        <span class="tech-badge">Python 3.9+</span>
        <span class="tech-badge">Scikit-learn</span>
        <span class="tech-badge">XGBoost</span>
        <span class="tech-badge">LightGBM</span>
        <span class="tech-badge">Optuna</span>
        <span class="tech-badge">Streamlit</span>
        <span class="tech-badge">Pandas</span>
        <span class="tech-badge">NumPy</span>
        <span class="tech-badge">Matplotlib</span>
        <span class="tech-badge">Seaborn</span>
        <span class="tech-badge">Plotly</span>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## 🚀 Quick Start")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""<div class="success-box">
            <h3>📊 Supervised Learning</h3>
            <ol>
                <li>Go to <strong>Run Experiment</strong></li>
                <li>Upload your CSV dataset</li>
                <li>Select <strong>Supervised Learning</strong></li>
                <li>Choose your target column</li>
                <li>Click <strong>Run Experiment</strong></li>
                <li>Download trained model</li>
            </ol>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("""<div class="warning-box">
            <h3>🔍 Unsupervised Learning</h3>
            <ol>
                <li>Go to <strong>Run Experiment</strong></li>
                <li>Upload your dataset</li>
                <li>Select <strong>Unsupervised Learning</strong></li>
                <li>System auto-selects numeric features</li>
                <li>Click <strong>Run Experiment</strong></li>
                <li>Explore clustering results</li>
            </ol>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## 📥 Try with Example Datasets")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🌸 Iris Dataset", use_container_width=True):
            from sklearn.datasets import load_iris
            iris = load_iris(as_frame=True)
            Path("data/samples").mkdir(parents=True, exist_ok=True)
            iris.frame.to_csv('data/samples/iris.csv', index=False)
            st.success("✅ iris.csv ready in data/samples/")

    with col2:
        if st.button("💊 Diabetes Dataset", use_container_width=True):
            from sklearn.datasets import load_diabetes
            diabetes = load_diabetes(as_frame=True)
            Path("data/samples").mkdir(parents=True, exist_ok=True)
            diabetes.frame.to_csv('data/samples/diabetes.csv', index=False)
            st.success("✅ diabetes.csv ready in data/samples/")

    with col3:
        if st.button("🍷 Wine Dataset", use_container_width=True):
            from sklearn.datasets import load_wine
            wine = load_wine(as_frame=True)
            Path("data/samples").mkdir(parents=True, exist_ok=True)
            wine.frame.to_csv('data/samples/wine.csv', index=False)
            st.success("✅ wine.csv ready in data/samples/")

# ==================== RUN EXPERIMENT ====================
elif page == "📊 Run Experiment":
    st.markdown('<h1 class="main-header">📊 Run ML Experiment</h1>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📁 Upload Your Dataset (CSV)", type=['csv'])

    if uploaded_file:
        # Read directly into memory to avoid HF 400 upload errors
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        df = None
        for enc in encodings:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(io.BytesIO(uploaded_file.read()), encoding=enc)
                break
            except Exception:
                continue

        if df is None:
            st.error("❌ Could not read the CSV. Try saving it as UTF-8.")
        else:
            st.session_state.df = df
            st.success(f"✅ Dataset Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

            with st.expander("👁️ Preview Data & Statistics"):
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.dataframe(df.head(10))
                with col2:
                    st.metric("Total Rows", df.shape[0])
                    st.metric("Total Columns", df.shape[1])
                    st.metric("Missing Values", int(df.isnull().sum().sum()))
                    st.metric("Duplicates", int(df.duplicated().sum()))

            st.markdown("---")

            learning_type = st.radio(
                "🎯 Select Learning Type:",
                ["Supervised Learning", "Unsupervised Learning"],
                horizontal=True
            )

            # ---- SUPERVISED ----
            if learning_type == "Supervised Learning":
                target = st.selectbox("🎯 Select Target Column:", df.columns)

                if st.button("🚀 Run Supervised Experiment", type="primary", use_container_width=True):
                    with st.spinner("🔄 Running ML Pipeline... this takes 2–4 minutes"):
                        try:
                            temp_path = "temp_data.csv"
                            df.to_csv(temp_path, index=False, encoding='utf-8')

                            progress = st.progress(0)
                            status = st.empty()

                            status.text("📊 Inspecting data...")
                            progress.progress(10)
                            status.text("🔧 Preprocessing...")
                            progress.progress(25)
                            status.text("🎯 Selecting features...")
                            progress.progress(40)
                            status.text("🤖 Training models...")
                            progress.progress(55)

                            results = st.session_state.orchestrator.run(
                                temp_path, target, 'supervised'
                            )

                            progress.progress(100)
                            status.empty()
                            progress.empty()

                            st.session_state.results = results
                            st.session_state.n_experiments += 1

                            st.balloons()
                            st.success("🎉 Experiment Completed Successfully!")

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.markdown(f"""<div style="background:#1e5128;padding:1.2rem;border-radius:10px;">
                                    <p style="color:#ccc;margin:0;font-size:0.85rem;">🏆 Best Model</p>
                                    <p style="color:white;font-size:1.8rem;font-weight:700;margin:0.3rem 0 0 0;">{results['best_model']}</p>
                                </div>""", unsafe_allow_html=True)
                            with col2:
                                st.markdown(f"""<div style="background:#1e5128;padding:1.2rem;border-radius:10px;">
                                    <p style="color:#ccc;margin:0;font-size:0.85rem;">📋 Task Type</p>
                                    <p style="color:white;font-size:1.8rem;font-weight:700;margin:0.3rem 0 0 0;">{results['task_type'].upper()}</p>
                                </div>""", unsafe_allow_html=True)
                            with col3:
                                metric_key = list(results['metrics'].keys())[0]
                                st.markdown(f"""<div style="background:#1e5128;padding:1.2rem;border-radius:10px;">
                                    <p style="color:#ccc;margin:0;font-size:0.85rem;">{metric_key.upper()}</p>
                                    <p style="color:white;font-size:1.8rem;font-weight:700;margin:0.3rem 0 0 0;">{results['metrics'][metric_key]:.4f}</p>
                                </div>""", unsafe_allow_html=True)

                            st.markdown("### 📊 Detailed Metrics")
                            st.dataframe(pd.DataFrame([results['metrics']]), use_container_width=True)

                            st.markdown("### 📈 Visualizations")
                            viz_path = Path("experiments/visualizations")
                            if viz_path.exists():
                                png_files = list(viz_path.glob("*.png"))
                                if png_files:
                                    cols = st.columns(2)
                                    for idx, viz_file in enumerate(png_files):
                                        with cols[idx % 2]:
                                            st.image(
                                                Image.open(viz_file),
                                                caption=viz_file.stem.replace('_', ' ').title(),
                                                use_column_width=True
                                            )

                            st.markdown("### 💾 Download Trained Model")
                            model_path = Path(results['model_path'])
                            if model_path.exists():
                                with open(model_path, 'rb') as f:
                                    st.download_button(
                                        "📥 Download Model (.pkl)",
                                        f,
                                        file_name=f"{results['best_model']}_trained.pkl",
                                        use_container_width=True
                                    )

                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                            st.code(traceback.format_exc())

            # ---- UNSUPERVISED ----
            else:
                if st.button("🚀 Run Unsupervised Experiment", type="primary", use_container_width=True):
                    with st.spinner("🔄 Running Unsupervised Learning... this takes 2–3 minutes"):
                        try:
                            temp_path = "temp_data.csv"
                            df.to_csv(temp_path, index=False, encoding='utf-8')

                            progress = st.progress(0)
                            status = st.empty()

                            status.text("📊 Processing data...")
                            progress.progress(25)
                            status.text("🎨 Running dimensionality reduction...")
                            progress.progress(50)
                            status.text("🔍 Clustering...")
                            progress.progress(75)

                            results = st.session_state.orchestrator.run(
                                temp_path, pipeline_type='unsupervised'
                            )

                            progress.progress(100)
                            status.empty()
                            progress.empty()

                            st.session_state.results = results
                            st.session_state.n_experiments += 1

                            st.balloons()
                            st.success("🎉 Unsupervised Learning Complete!")

                            st.markdown("### 🎯 Clustering Results")
                            clustering_df = pd.DataFrame([
                                {'Algorithm': k, 'Clusters Found': v}
                                for k, v in results['clustering'].items()
                            ])
                            st.dataframe(clustering_df, use_container_width=True)

                            st.markdown("### 📈 Visualizations")
                            viz_path = Path("experiments/visualizations")
                            if viz_path.exists():
                                for viz_file in viz_path.glob("*.png"):
                                    st.image(Image.open(viz_file), use_column_width=True)

                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                            st.code(traceback.format_exc())

# ==================== RESULTS ====================
elif page == "📈 Results":
    st.markdown('<h1 class="main-header">📈 Experiment Results</h1>', unsafe_allow_html=True)

    if st.session_state.results is None:
        st.info("ℹ️ No results yet. Run an experiment first.")
    else:
        results = st.session_state.results

        if results.get('pipeline') == 'supervised':
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""<div style="background:#1e5128;padding:1.2rem;border-radius:10px;">
                    <p style="color:#ccc;margin:0;font-size:0.85rem;">🏆 Best Model</p>
                    <p style="color:white;font-size:1.8rem;font-weight:700;margin:0.3rem 0 0 0;">{results['best_model']}</p>
                </div>""", unsafe_allow_html=True)
            with col2:
                st.metric("Task Type", results['task_type'].upper())
            with col3:
                metric_key = list(results['metrics'].keys())[0]
                st.metric(metric_key.upper(), f"{results['metrics'][metric_key]:.4f}")

            st.markdown("### 📊 Detailed Metrics")
            st.dataframe(pd.DataFrame([results['metrics']]), use_container_width=True)

            viz_path = Path("experiments/visualizations")
            if viz_path.exists():
                st.markdown("### 📈 Visualizations")
                for viz_file in viz_path.glob("*.png"):
                    st.image(Image.open(viz_file), use_column_width=True)

        elif results.get('pipeline') == 'unsupervised':
            st.markdown("### 🎯 Clustering Results")
            clustering_df = pd.DataFrame([
                {'Algorithm': k, 'Clusters Found': v}
                for k, v in results['clustering'].items()
            ])
            st.dataframe(clustering_df, use_container_width=True)

            viz_path = Path("experiments/visualizations")
            if viz_path.exists():
                st.markdown("### 📈 Visualizations")
                for viz_file in viz_path.glob("*.png"):
                    st.image(Image.open(viz_file), use_column_width=True)

# ==================== DOCUMENTATION ====================
elif page == "📚 Documentation":
    st.markdown('<h1 class="main-header">📚 Documentation</h1>', unsafe_allow_html=True)

    st.markdown("""<div class="info-box">
        <h3>📖 How to use</h3>
        <ol>
            <li>Go to <strong>Run Experiment</strong></li>
            <li>Upload a CSV file (max 200MB)</li>
            <li>Choose Supervised or Unsupervised learning</li>
            <li>For supervised: pick your target column</li>
            <li>Hit Run — results appear on this page and in the Results tab</li>
        </ol>
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="info-box">
        <h3>⚠️ Free tier notes</h3>
        <p>This runs on Hugging Face Spaces free CPU. Experiments take 2–5 minutes.
        Results reset when the Space restarts. For persistence, run locally.</p>
    </div>""", unsafe_allow_html=True)

# ==================== FOOTER ====================
st.markdown("""
<div class="footer">
    <p><strong>ML Experiment Orchestrator v1.0</strong></p>
    <p>Built with Python · Streamlit · Scikit-learn · XGBoost · LightGBM · Optuna</p>
</div>
""", unsafe_allow_html=True)