import io
import traceback
import warnings
from pathlib import Path

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

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* App background */
    .stApp {
        background-color: #d6d6d6;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e5128 0%, #2d6a3e 100%) !important;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    [data-testid="stSidebar"] .stMetricLabel,
    [data-testid="stSidebar"] .stMetricValue {
        color: white !important;
    }

    /* Main content white card */
    .main .block-container {
        background: #ffffff;
        border-radius: 16px;
        padding: 2rem 2.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.10);
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
    }

    /* All normal text in main area must be dark */
    .main p, .main li, .main ol, .main ul, .main span, .main label {
        color: #1a1a1a !important;
    }

    /* Headings */
    .main h1, .main h2, .main h3, .main h4 {
        color: #1e5128 !important;
    }

    /* Section headers written as st.markdown("## ...") */
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3 {
        color: #1e5128 !important;
    }

    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] ol,
    [data-testid="stMarkdownContainer"] ul {
        color: #1a1a1a !important;
    }

    /* Feature cards — green background, white text */
    .feature-card {
        background: linear-gradient(135deg, #1e5128 0%, #2d6a3e 100%);
        padding: 2rem 1.5rem;
        border-radius: 14px;
        color: white;
        text-align: center;
        box-shadow: 0 6px 20px rgba(30,81,40,0.25);
        min-height: 200px;
    }
    .feature-card h2 { color: white !important; font-size: 2rem; margin-bottom: 0.3rem; }
    .feature-card h3 { color: #d0f0d0 !important; font-size: 1.2rem; margin-bottom: 0.5rem; }
    .feature-card p  { color: #c8e6c9 !important; font-size: 0.95rem; margin: 0.2rem 0; }
    .feature-card strong { color: white !important; }

    /* Info / success / warning boxes */
    .info-box {
        background: #f8fffe;
        border-left: 5px solid #1e5128;
        padding: 1.2rem 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .info-box h3 { color: #1e5128 !important; margin-bottom: 0.6rem; font-weight: 600; }
    .info-box p, .info-box li, .info-box ol, .info-box ul { color: #1a1a1a !important; font-size: 1rem; line-height: 1.7; }

    .success-box {
        background: #f0faf3;
        border-left: 5px solid #28a745;
        padding: 1.2rem 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .success-box h3 { color: #1e5128 !important; font-weight: 600; margin-bottom: 0.6rem; }
    .success-box p, .success-box li, .success-box ol, .success-box ul { color: #1a1a1a !important; font-size: 1rem; line-height: 1.7; }

    .warning-box {
        background: #fffdf0;
        border-left: 5px solid #ffc107;
        padding: 1.2rem 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .warning-box h3 { color: #856404 !important; font-weight: 600; margin-bottom: 0.6rem; }
    .warning-box p, .warning-box li, .warning-box ol, .warning-box ul { color: #1a1a1a !important; font-size: 1rem; line-height: 1.7; }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #1e5128 0%, #2d6a3e 100%) !important;
        color: white !important;
        font-weight: 600;
        border: none !important;
        border-radius: 10px;
        padding: 0.6rem 1.8rem;
        font-size: 1rem;
        box-shadow: 0 4px 12px rgba(30,81,40,0.3);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2d6a3e 0%, #1e5128 100%) !important;
        box-shadow: 0 6px 16px rgba(30,81,40,0.4);
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #1e5128 !important;
        font-weight: 700;
    }
    [data-testid="stMetricLabel"] {
        color: #555 !important;
    }

    /* Radio buttons — make label text dark */
    .stRadio label { color: #1a1a1a !important; font-weight: 500; }
    .stRadio > div { color: #1a1a1a !important; }

    /* Selectbox label */
    .stSelectbox label { color: #1a1a1a !important; font-weight: 500; }

    /* File uploader */
    .stFileUploader label { color: #1a1a1a !important; font-weight: 500; }

    /* Expander */
    .streamlit-expanderHeader { color: #1e5128 !important; font-weight: 600; }
    details summary { color: #1e5128 !important; }

    /* Tech stack badges */
    .tech-badge {
        display: inline-block;
        background: linear-gradient(135deg, #1e5128 0%, #2d6a3e 100%);
        color: white !important;
        padding: 0.35rem 1.1rem;
        border-radius: 20px;
        margin: 0.3rem;
        font-weight: 600;
        font-size: 0.88rem;
        box-shadow: 0 2px 8px rgba(30,81,40,0.25);
    }

    /* Feature list */
    .feature-list { list-style: none; padding-left: 0; margin: 0; }
    .feature-list li {
        padding: 0.45rem 0;
        border-bottom: 1px solid #e8f5e9;
        color: #1a1a1a !important;
        font-size: 0.97rem;
    }
    .feature-list li::before { content: "✓ "; color: #1e5128; font-weight: bold; }

    /* Result metric cards */
    .result-card {
        background: #1e5128;
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
    }
    .result-card .rc-label { color: #b2dfb8; font-size: 0.85rem; margin-bottom: 0.3rem; }
    .result-card .rc-value { color: white; font-size: 1.7rem; font-weight: 700; }

    /* Footer */
    .footer {
        text-align: center;
        padding: 1.5rem;
        color: #666 !important;
        border-top: 1px solid #e0e0e0;
        margin-top: 2rem;
        font-size: 0.9rem;
    }

    /* Dataframe */
    [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
    }

    /* Success/info/error native streamlit messages */
    .stAlert p { color: #1a1a1a !important; }
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
        "Navigation",
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

    st.markdown("## 🤖 ML Experiment Orchestrator")
    st.markdown("##### Advanced Automated Machine Learning Platform with Multi-Agent Architecture")
    st.markdown("---")

    st.markdown("### ✨ Core Capabilities")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <h2>🎯</h2>
            <h3>Supervised Learning</h3>
            <p>Classification &amp; Regression</p>
            <p><strong>12+ Advanced Models</strong></p>
            <p>XGBoost &bull; LightGBM &bull; Random Forest</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <h2>🔍</h2>
            <h3>Unsupervised Learning</h3>
            <p>Clustering &amp; Dimensionality Reduction</p>
            <p><strong>7+ Algorithms</strong></p>
            <p>KMeans &bull; DBSCAN &bull; PCA &bull; t-SNE</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card">
            <h2>⚡</h2>
            <h3>Auto Optimization</h3>
            <p>Hyperparameter Tuning</p>
            <p><strong>Optuna-Powered</strong></p>
            <p>Bayesian Optimization</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🎯 What is ML Experiment Orchestrator?")
    st.markdown("""
    <div class="info-box">
        <p>The <strong>ML Experiment Orchestrator</strong> is an intelligent, end-to-end automated ML platform
        built on a <strong>multi-agent architecture</strong>. It automates the entire ML pipeline — from data ingestion
        and preprocessing to model training, hyperparameter tuning, and evaluation — with no manual intervention required.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🛠️ Technology Stack")
    st.markdown("""
    <div style="text-align:center; padding: 0.5rem 0 1rem 0;">
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
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🚀 Quick Start Guide")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="success-box">
            <h3>📊 For Supervised Learning</h3>
            <ol>
                <li>Navigate to <strong>Run Experiment</strong></li>
                <li>Upload your CSV dataset</li>
                <li>Select <strong>Supervised Learning</strong></li>
                <li>Choose your target column</li>
                <li>Click <strong>Run Experiment</strong></li>
                <li>Download the trained model</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="warning-box">
            <h3>🔍 For Unsupervised Learning</h3>
            <ol>
                <li>Navigate to <strong>Run Experiment</strong></li>
                <li>Upload your dataset</li>
                <li>Select <strong>Unsupervised Learning</strong></li>
                <li>System auto-selects numeric features</li>
                <li>Click <strong>Run Experiment</strong></li>
                <li>Explore clustering results</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 💼 Use Cases")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="info-box">
            <h3>🏥 Healthcare</h3>
            <ul class="feature-list">
                <li>Disease prediction</li>
                <li>Patient clustering</li>
                <li>Treatment optimization</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-box">
            <h3>💰 Finance</h3>
            <ul class="feature-list">
                <li>Credit risk assessment</li>
                <li>Fraud detection</li>
                <li>Customer segmentation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="info-box">
            <h3>🛒 E-Commerce</h3>
            <ul class="feature-list">
                <li>Churn prediction</li>
                <li>Price optimization</li>
                <li>Recommendation systems</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📥 Try with Example Datasets")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🌸 Iris Dataset", use_container_width=True):
            from sklearn.datasets import load_iris
            iris = load_iris(as_frame=True)
            Path("data/samples").mkdir(parents=True, exist_ok=True)
            iris.frame.to_csv('data/samples/iris.csv', index=False)
            st.success("✅ iris.csv ready — upload it from data/samples/")

    with col2:
        if st.button("💊 Diabetes Dataset", use_container_width=True):
            from sklearn.datasets import load_diabetes
            diabetes = load_diabetes(as_frame=True)
            Path("data/samples").mkdir(parents=True, exist_ok=True)
            diabetes.frame.to_csv('data/samples/diabetes.csv', index=False)
            st.success("✅ diabetes.csv ready — upload it from data/samples/")

    with col3:
        if st.button("🍷 Wine Dataset", use_container_width=True):
            from sklearn.datasets import load_wine
            wine = load_wine(as_frame=True)
            Path("data/samples").mkdir(parents=True, exist_ok=True)
            wine.frame.to_csv('data/samples/wine.csv', index=False)
            st.success("✅ wine.csv ready — upload it from data/samples/")

# ==================== RUN EXPERIMENT ====================
elif page == "📊 Run Experiment":

    st.markdown("## 📊 Run ML Experiment")
    st.markdown("---")

    uploaded_file = st.file_uploader("📁 Upload Your Dataset (CSV)", type=['csv'])

    if uploaded_file:
        uploaded_file.name = uploaded_file.name.replace(' ', '_')
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
            st.success(f"✅ Dataset Loaded: **{df.shape[0]} rows × {df.shape[1]} columns**")

            with st.expander("👁️ Preview Data & Statistics"):
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.dataframe(df.head(10), use_container_width=True)
                with col2:
                    st.metric("Total Rows", df.shape[0])
                    st.metric("Total Columns", df.shape[1])
                    st.metric("Missing Values", int(df.isnull().sum().sum()))
                    st.metric("Duplicates", int(df.duplicated().sum()))

            st.markdown("---")
            st.markdown("#### 🎯 Select Learning Type")
            learning_type = st.radio(
                "Learning Type",
                ["Supervised Learning", "Unsupervised Learning"],
                horizontal=True,
                label_visibility="collapsed"
            )

            # ---- SUPERVISED ----
            if learning_type == "Supervised Learning":
                st.markdown("#### 🎯 Select Target Column")
                target = st.selectbox("Target Column", df.columns, label_visibility="collapsed")

                st.markdown("---")
                if st.button("🚀 Run Supervised Experiment", type="primary", use_container_width=True):
                    with st.spinner("🔄 Running ML Pipeline... this takes 2–4 minutes"):
                        try:
                            temp_path = "temp_data.csv"
                            df.to_csv(temp_path, index=False, encoding='utf-8')

                            progress = st.progress(0)
                            status = st.empty()

                            status.text("📊 Inspecting data...")
                            progress.progress(10)
                            status.text("🔧 Preprocessing data...")
                            progress.progress(25)
                            status.text("🎯 Selecting features...")
                            progress.progress(40)
                            status.text("🤖 Training all models...")
                            progress.progress(60)

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
                            st.markdown("---")

                            # Result summary cards
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.markdown(f"""
                                <div class="result-card">
                                    <div class="rc-label">🏆 Best Model</div>
                                    <div class="rc-value">{results['best_model']}</div>
                                </div>""", unsafe_allow_html=True)
                            with col2:
                                st.markdown(f"""
                                <div class="result-card">
                                    <div class="rc-label">📋 Task Type</div>
                                    <div class="rc-value">{results['task_type'].upper()}</div>
                                </div>""", unsafe_allow_html=True)
                            with col3:
                                metric_key = list(results['metrics'].keys())[0]
                                st.markdown(f"""
                                <div class="result-card">
                                    <div class="rc-label">{metric_key.upper()}</div>
                                    <div class="rc-value">{results['metrics'][metric_key]:.4f}</div>
                                </div>""", unsafe_allow_html=True)

                            st.markdown("---")
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
                st.markdown("---")
                if st.button("🚀 Run Unsupervised Experiment", type="primary", use_container_width=True):
                    with st.spinner("🔄 Running Unsupervised Learning... this takes 2–3 minutes"):
                        try:
                            temp_path = "temp_data.csv"
                            df.to_csv(temp_path, index=False, encoding='utf-8')

                            progress = st.progress(0)
                            status = st.empty()

                            status.text("📊 Processing data...")
                            progress.progress(25)
                            status.text("🎨 Dimensionality reduction...")
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
                            st.markdown("---")

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

    st.markdown("## 📈 Experiment Results")
    st.markdown("---")

    if st.session_state.results is None:
        st.info("ℹ️ No results yet. Run an experiment first from the **Run Experiment** page.")
    else:
        results = st.session_state.results

        if results.get('pipeline') == 'supervised':
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="result-card">
                    <div class="rc-label">🏆 Best Model</div>
                    <div class="rc-value">{results['best_model']}</div>
                </div>""", unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="result-card">
                    <div class="rc-label">📋 Task Type</div>
                    <div class="rc-value">{results['task_type'].upper()}</div>
                </div>""", unsafe_allow_html=True)
            with col3:
                metric_key = list(results['metrics'].keys())[0]
                st.markdown(f"""
                <div class="result-card">
                    <div class="rc-label">{metric_key.upper()}</div>
                    <div class="rc-value">{results['metrics'][metric_key]:.4f}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("---")
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

    st.markdown("## 📚 Documentation")
    st.markdown("---")

    st.markdown("""
    <div class="info-box">
        <h3>📖 How to use this app</h3>
        <ol>
            <li>Go to <strong>Run Experiment</strong> in the sidebar</li>
            <li>Upload a CSV file (max 200MB)</li>
            <li>Choose <strong>Supervised</strong> or <strong>Unsupervised</strong> learning</li>
            <li>For supervised: select your target column from the dropdown</li>
            <li>Click <strong>Run Experiment</strong> — results appear below</li>
            <li>Navigate to <strong>Results</strong> to revisit results anytime</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="warning-box">
        <h3>⚠️ Free Tier Notes</h3>
        <p>This app runs on Hugging Face Spaces free CPU tier (2 vCPU, 16GB RAM).
        Experiments typically complete in <strong>2–5 minutes</strong> depending on dataset size.
        Results and saved models reset when the Space restarts after inactivity.
        For persistent storage, run the app locally.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="success-box">
        <h3>✅ Supported Data Formats</h3>
        <ul>
            <li>CSV files (UTF-8, Latin-1, CP1252 encodings)</li>
            <li>Any number of rows (recommended: under 100k for free tier)</li>
            <li>Mixed numeric and categorical columns supported</li>
            <li>Missing values handled automatically</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ==================== FOOTER ====================
st.markdown("""
<div class="footer">
    <strong>ML Experiment Orchestrator v1.0</strong><br>
    Built with Python &bull; Streamlit &bull; Scikit-learn &bull; XGBoost &bull; LightGBM &bull; Optuna
</div>
""", unsafe_allow_html=True)