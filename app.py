import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from PIL import Image
import time
import warnings
warnings.filterwarnings('ignore')

from orchestrator import MLOrchestrator

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="ML Experiment Orchestrator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS WITH FIXED TEXT VISIBILITY ====================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    /* Main background with your colors */
    .stApp {
        background: linear-gradient(to right, #1e5128 0%, #1e5128 30%, #d6d6d6 30%, #d6d6d6 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e5128 0%, #2d6a3e 100%);
        color: white;
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Main content area */
    .main .block-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        margin: 2rem auto;
    }
    
    /* Headers */
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #1e5128 0%, #2d6a3e 50%, #4a9960 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1.5rem 0;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        text-align: center;
        color: #333;
        font-size: 1.3rem;
        font-weight: 400;
        margin-bottom: 2rem;
    }
    
    /* Feature cards */
    .feature-card {
        background: linear-gradient(135deg, #1e5128 0%, #2d6a3e 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 25px rgba(30,81,40,0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        height: 100%;
    }
    
    .feature-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 15px 35px rgba(30,81,40,0.4);
    }
    
    .feature-card h2 {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        color: white !important;
    }
    
    .feature-card h3 {
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
        color: #e0e0e0 !important;
    }
    
    .feature-card p {
        font-size: 1rem;
        color: #d0d0d0;
        margin: 0.3rem 0;
    }
    
    /* Info boxes - FIXED DARK TEXT */
    .info-box {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-left: 5px solid #1e5128;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1.5rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .info-box h3 {
        color: #1e5128 !important;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    .custom-white-heading {
        color: white !important;
    }
    
    .info-box p {
        color: #222 !important;
        font-size: 1.1rem !important;
        line-height: 1.8;
    }
    
    .info-box ul, .info-box ol {
        color: #222 !important;
    }
    
    .info-box li {
        color: #222 !important;
        font-size: 1.05rem !important;
    }
    
    .success-box {
        background: linear-gradient(135d, #ffffff 0%, #f0f9f4 100%);
        border-left: 5px solid #28a745;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1.5rem 0;
    }
    
    .success-box h3 {
        color: #1e5128 !important;
        font-weight: 600;
    }
    
    .success-box ul, .success-box ol {
        color: #222 !important;
    }
    
    .success-box li {
        color: #222 !important;
        font-size: 1.05rem !important;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fffef8 0%, #fffbf0 100%);
        border-left: 5px solid #ffc107;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1.5rem 0;
    }
    
    .warning-box h3 {
        color: #1e5128 !important;
        font-weight: 600;
    }
    
    .warning-box ul, .warning-box ol {
        color: #222 !important;
    }
    
    .warning-box li {
        color: #222 !important;
        font-size: 1.05rem !important;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #1e5128 0%, #2d6a3e 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(30,81,40,0.3);
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #2d6a3e 0%, #1e5128 100%);
        box-shadow: 0 6px 20px rgba(30,81,40,0.4);
        transform: translateY(-2px);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        color: #1e5128;
        font-weight: 700;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px 10px 0 0;
        padding: 1rem 2rem;
        font-weight: 600;
        color: #1e5128;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1e5128 0%, #2d6a3e 100%);
        color: white !important;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
        font-weight: 600;
        color: #1e5128;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border: 2px dashed #1e5128;
        border-radius: 15px;
        padding: 2rem;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #1e5128 0%, #2d6a3e 100%);
    }
    
    /* Selectbox, Radio */
    .stSelectbox, .stRadio {
        color: #1e5128;
    }
    
    /* Dataframe */
    [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Sidebar elements */
    [data-testid="stSidebar"] .stRadio > label {
        background: rgba(255,255,255,0.1);
        padding: 0.75rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    
    [data-testid="stSidebar"] .stRadio > label:hover {
        background: rgba(255,255,255,0.2);
    }
    
    /* List styling - FIXED DARK TEXT */
    .feature-list {
        list-style: none;
        padding-left: 0;
    }
    
    .feature-list li {
        padding: 0.75rem 0;
        border-bottom: 1px solid #e0e0e0;
        font-size: 1.1rem;
        color: #222 !important;
    }
    
    .feature-list li:before {
        content: "✓";
        color: #1e5128;
        font-weight: bold;
        margin-right: 1rem;
        font-size: 1.3rem;
    }
    
    /* Tech stack badges */
    .tech-badge {
        display: inline-block;
        background: linear-gradient(135deg, #1e5128 0%, #2d6a3e 100%);
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 20px;
        margin: 0.5rem;
        font-weight: 600;
        box-shadow: 0 3px 10px rgba(30,81,40,0.3);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        border-top: 2px solid #e0e0e0;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# ==================== INITIALIZE SESSION STATE ====================
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = MLOrchestrator()
if 'results' not in st.session_state:
    st.session_state.results = None
if 'df' not in st.session_state:
    st.session_state.df = None

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("# 🤖")
    st.markdown("# ML Orchestrator")
    st.markdown("---")
    
    page = st.radio(
        "**Navigation**",
        ["🏠 Home", "📊 Run Experiment", "📈 Results", "📚 Documentation"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    st.markdown("### 📊 Statistics")
    
    # Count experiments
    mlruns_path = Path("experiments/mlruns")
    n_experiments = len(list(mlruns_path.glob("*/*"))) if mlruns_path.exists() else 0
    
    # Count models
    models_path = Path("models/supervised")
    n_models = len(list(models_path.glob("*.pkl"))) if models_path.exists() else 0
    
    st.metric("Experiments", n_experiments)
    st.metric("Saved Models", n_models)
    
    st.markdown("---")
    
    st.markdown("### 🔗 Quick Links")
    st.markdown("📧 [Contact](mailto:your@email.com)")
    st.markdown("🐙 [GitHub](https://github.com)")
    st.markdown("📖 [Docs](https://docs.example.com)")

# ==================== HOME PAGE ====================
if page == "🏠 Home":
    
    # Header
    st.markdown('<h1 class="main-header">🤖 ML Experiment Orchestrator</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced Automated Machine Learning Platform with Multi-Agent Architecture</p>', unsafe_allow_html=True)
    
    # Hero section with feature cards
    st.markdown("## ✨ Core Capabilities")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h2>🎯</h2>
            <h3>Supervised Learning</h3>
            <p>Classification & Regression</p>
            <p><strong>12+ Advanced Models</strong></p>
            <p>XGBoost • LightGBM • CatBoost</p>
            <p>Random Forest • Neural Nets</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h2>🔍</h2>
            <h3>Unsupervised Learning</h3>
            <p>Clustering & Dimensionality</p>
            <p><strong>7+ Algorithms</strong></p>
            <p>KMeans • DBSCAN • PCA</p>
            <p>t-SNE • UMAP • GMM</p>
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
            <p>Multi-Objective Tuning</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # What is this platform? - FIXED TEXT COLOR
    st.markdown("## 🎯 What is ML Experiment Orchestrator?")
    
    st.markdown("""
    <div class="info-box">
        <p style="font-size: 1.2rem; line-height: 1.8; color: #222 !important;">
        The <strong>ML Experiment Orchestrator</strong> is an intelligent, end-to-end automated machine learning platform 
        that revolutionizes how data scientists and machine learning engineers approach model development. 
        Built on a sophisticated <strong>multi-agent architecture</strong>, it automates the entire ML pipeline from 
        data ingestion to model deployment, eliminating manual intervention while maintaining full transparency and control.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key Features
    st.markdown("## 🚀 Key Features & Capabilities")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🤖 AI Agents", "🎓 ML Models", "📊 Workflow", "🔧 Advanced"])
    
    with tab1:
        st.markdown("### Multi-Agent System Architecture")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="info-box">
                <h3>🔍 Data Agent</h3>
                <ul class="feature-list">
                    <li>Automatic data profiling & quality assessment</li>
                    <li>Schema detection & type inference</li>
                    <li>Missing value pattern analysis</li>
                    <li>Outlier & anomaly detection</li>
                    <li>Data distribution visualization</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="info-box">
                <h3>🔧 Preprocessing Agent</h3>
                <ul class="feature-list">
                    <li>Intelligent missing value imputation</li>
                    <li>Automatic encoding (One-Hot, Label, Target)</li>
                    <li>Feature scaling & normalization</li>
                    <li>Class imbalance handling (SMOTE)</li>
                    <li>Outlier treatment (IQR, Z-score)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="info-box">
                <h3>🎯 Feature Selection Agent</h3>
                <ul class="feature-list">
                    <li>Mutual Information scoring</li>
                    <li>F-score statistical testing</li>
                    <li>Recursive Feature Elimination</li>
                    <li>Tree-based importance ranking</li>
                    <li>Correlation analysis</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="info-box">
                <h3>⚡ Hyperparameter Agent</h3>
                <ul class="feature-list">
                    <li>Bayesian optimization (Optuna)</li>
                    <li>Multi-objective tuning</li>
                    <li>Parallel trial execution</li>
                    <li>Early stopping mechanisms</li>
                    <li>Pruning strategies</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### Comprehensive Model Library")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="success-box">
                <h3>🎯 Classification Models (9)</h3>
                <ul class="feature-list">
                    <li><strong>Logistic Regression</strong> - Linear baseline</li>
                    <li><strong>Decision Tree</strong> - Interpretable rules</li>
                    <li><strong>Random Forest</strong> - Ensemble learning</li>
                    <li><strong>Gradient Boosting</strong> - Sequential optimization</li>
                    <li><strong>XGBoost</strong> - Extreme gradient boosting</li>
                    <li><strong>LightGBM</strong> - Fast gradient boosting</li>
                    <li><strong>CatBoost</strong> - Categorical boosting</li>
                    <li><strong>K-Nearest Neighbors</strong> - Distance-based</li>
                    <li><strong>Naive Bayes</strong> - Probabilistic classifier</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="success-box">
                <h3>📈 Regression Models (12)</h3>
                <ul class="feature-list">
                    <li><strong>Linear Regression</strong> - OLS baseline</li>
                    <li><strong>Ridge</strong> - L2 regularization</li>
                    <li><strong>Lasso</strong> - L1 regularization</li>
                    <li><strong>ElasticNet</strong> - L1 + L2 regularization</li>
                    <li><strong>Decision Tree</strong> - Non-linear regression</li>
                    <li><strong>Random Forest</strong> - Ensemble regression</li>
                    <li><strong>Gradient Boosting</strong> - Sequential boosting</li>
                    <li><strong>XGBoost</strong> - Advanced boosting</li>
                    <li><strong>LightGBM</strong> - Efficient gradient boosting</li>
                    <li><strong>CatBoost</strong> - Robust to overfitting</li>
                    <li><strong>SVR</strong> - Support vector regression</li>
                    <li><strong>KNN Regressor</strong> - Distance-based</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="warning-box">
            <h3>🔍 Unsupervised Learning</h3>
            <p style="color: #222 !important; font-size: 1.1rem;">
            <strong>Dimensionality Reduction:</strong> PCA, t-SNE, UMAP<br>
            <strong>Clustering:</strong> KMeans, DBSCAN, Agglomerative Clustering, Gaussian Mixture Models
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("### Complete ML Workflow")
        
        st.markdown("""
        <div class="info-box">
            <h3>📊 Supervised Learning Pipeline</h3>
            <ol style="font-size: 1.1rem; line-height: 2; color: #222 !important;">
                <li><strong>Data Upload</strong> - CSV, Excel, or JSON files</li>
                <li><strong>Automated Inspection</strong> - Data profiling, quality checks, statistical analysis</li>
                <li><strong>Task Detection</strong> - Automatic classification vs regression identification</li>
                <li><strong>Intelligent Preprocessing</strong> - Missing values, encoding, scaling, outliers</li>
                <li><strong>Feature Engineering</strong> - Interaction features, polynomial features, binning</li>
                <li><strong>Feature Selection</strong> - Multiple algorithms to select optimal features</li>
                <li><strong>Train-Test Split</strong> - Stratified splitting with configurable ratio</li>
                <li><strong>Multi-Model Training</strong> - Train 12+ models with cross-validation</li>
                <li><strong>Model Evaluation</strong> - Comprehensive metrics for comparison</li>
                <li><strong>Best Model Selection</strong> - Automatic selection based on CV scores</li>
                <li><strong>Hyperparameter Tuning</strong> - Optuna-based optimization (50-100 trials)</li>
                <li><strong>Final Training</strong> - Retrain with optimal parameters</li>
                <li><strong>Test Set Evaluation</strong> - Final performance metrics</li>
                <li><strong>Visualization</strong> - Confusion matrix, feature importance, ROC curves</li>
                <li><strong>MLflow Tracking</strong> - Log all experiments, parameters, metrics</li>
                <li><strong>Model Persistence</strong> - Save model and preprocessing pipeline</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <h3>🔍 Unsupervised Learning Pipeline</h3>
            <ol style="font-size: 1.1rem; line-height: 2; color: #222 !important;">
                <li><strong>Data Upload</strong> - Any structured dataset</li>
                <li><strong>Preprocessing</strong> - Cleaning, scaling, encoding</li>
                <li><strong>Dimensionality Reduction</strong> - PCA, t-SNE, UMAP for visualization</li>
                <li><strong>Optimal Cluster Detection</strong> - Elbow method, silhouette analysis</li>
                <li><strong>Multi-Algorithm Clustering</strong> - KMeans, DBSCAN, Hierarchical, GMM</li>
                <li><strong>Cluster Evaluation</strong> - Silhouette scores, Davies-Bouldin index</li>
                <li><strong>Interactive Visualization</strong> - 2D/3D scatter plots with clusters</li>
                <li><strong>Results Export</strong> - Save cluster assignments and centroids</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with tab4:
        st.markdown("### Advanced Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="info-box">
                <h3>🎯 Model Optimization</h3>
                <ul class="feature-list">
                    <li><strong>Bayesian Optimization</strong> - Smart hyperparameter search</li>
                    <li><strong>Cross-Validation</strong> - K-fold with stratification</li>
                    <li><strong>Early Stopping</strong> - Prevent overfitting</li>
                    <li><strong>Ensemble Methods</strong> - Combine multiple models</li>
                    <li><strong>Feature Importance</strong> - SHAP, permutation importance</li>
                </ul>
            </div>
            
            <div class="info-box">
                <h3>📊 Experiment Tracking</h3>
                <ul class="feature-list">
                    <li><strong>MLflow Integration</strong> - Track all experiments</li>
                    <li><strong>Parameter Logging</strong> - All hyperparameters saved</li>
                    <li><strong>Metric Tracking</strong> - Comprehensive metrics logging</li>
                    <li><strong>Artifact Storage</strong> - Models, plots, data</li>
                    <li><strong>Version Control</strong> - Model versioning system</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="info-box">
                <h3>📈 Visualization Suite</h3>
                <ul class="feature-list">
                    <li><strong>Confusion Matrix</strong> - Classification performance</li>
                    <li><strong>Feature Importance</strong> - Top contributing features</li>
                    <li><strong>Learning Curves</strong> - Training progression</li>
                    <li><strong>ROC & PR Curves</strong> - Threshold analysis</li>
                    <li><strong>Cluster Plots</strong> - Interactive 2D/3D plots</li>
                </ul>
            </div>
            
            <div class="info-box">
                <h3>💾 Export & Deployment</h3>
                <ul class="feature-list">
                    <li><strong>Model Serialization</strong> - Joblib, Pickle formats</li>
                    <li><strong>Pipeline Export</strong> - Complete preprocessing pipeline</li>
                    <li><strong>Prediction API</strong> - Ready-to-deploy models</li>
                    <li><strong>Docker Support</strong> - Containerized deployment</li>
                    <li><strong>Cloud Ready</strong> - AWS, GCP, Azure compatible</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Technology Stack
    st.markdown("## 🛠️ Technology Stack")
    
    st.markdown("""
    <div style="text-align: center; padding: 2rem;">
        <span class="tech-badge">Python 3.9+</span>
        <span class="tech-badge">Scikit-learn</span>
        <span class="tech-badge">XGBoost</span>
        <span class="tech-badge">LightGBM</span>
        <span class="tech-badge">CatBoost</span>
        <span class="tech-badge">Optuna</span>
        <span class="tech-badge">MLflow</span>
        <span class="tech-badge">Streamlit</span>
        <span class="tech-badge">Pandas</span>
        <span class="tech-badge">NumPy</span>
        <span class="tech-badge">Matplotlib</span>
        <span class="tech-badge">Seaborn</span>
        <span class="tech-badge">Plotly</span>
        <span class="tech-badge">SHAP</span>
        <span class="tech-badge">UMAP</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick Start - FIXED TEXT COLOR
    st.markdown("## 🚀 Quick Start Guide")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="success-box">
            <h3 class="custom-white-heading">📊 For Supervised Learning</h3>
            <ol style="font-size: 1.1rem; line-height: 1.8; color: #222 !important;">
                <li>Navigate to <strong>📊 Run Experiment</strong></li>
                <li>Upload your CSV dataset</li>
                <li>Select <strong>Supervised Learning</strong></li>
                <li>Choose your target column</li>
                <li>Click <strong>Run Experiment</strong></li>
                <li>Wait for training & optimization</li>
                <li>Download trained model</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="warning-box">
            <h3>🔍 For Unsupervised Learning</h3>
            <ol style="font-size: 1.1rem; line-height: 1.8; color: #222 !important;">
                <li>Navigate to <strong>📊 Run Experiment</strong></li>
                <li>Upload your dataset</li>
                <li>Select <strong>Unsupervised Learning</strong></li>
                <li>System auto-selects numeric features</li>
                <li>Click <strong>Run Experiment</strong></li>
                <li>Explore clustering results</li>
                <li>Download cluster assignments</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Use Cases - FIXED TEXT COLOR
    st.markdown("## 💼 Use Cases & Applications")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h3>🏥 Healthcare</h3>
            <ul class="feature-list">
                <li>Disease prediction</li>
                <li>Patient clustering</li>
                <li>Treatment optimization</li>
                <li>Medical imaging analysis</li>
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
                <li>Stock prediction</li>
                <li>Customer segmentation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="info-box">
            <h3>🛒 E-Commerce</h3>
            <ul class="feature-list">
                <li>Recommendation systems</li>
                <li>Churn prediction</li>
                <li>Price optimization</li>
                <li>Customer lifetime value</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Example Datasets
    st.markdown("## 📥 Get Started with Example Datasets")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🌸 Generate Iris Dataset", use_container_width=True):
            from sklearn.datasets import load_iris
            iris = load_iris(as_frame=True)
            Path("data/samples").mkdir(parents=True, exist_ok=True)
            iris.frame.to_csv('data/samples/iris.csv', index=False)
            st.success("✅ Created: data/samples/iris.csv")
    
    with col2:
        if st.button("💊 Generate Diabetes Dataset", use_container_width=True):
            from sklearn.datasets import load_diabetes
            diabetes = load_diabetes(as_frame=True)
            Path("data/samples").mkdir(parents=True, exist_ok=True)
            diabetes.frame.to_csv('data/samples/diabetes.csv', index=False)
            st.success("✅ Created: data/samples/diabetes.csv")
    
    with col3:
        if st.button("🍷 Generate Wine Dataset", use_container_width=True):
            from sklearn.datasets import load_wine
            wine = load_wine(as_frame=True)
            Path("data/samples").mkdir(parents=True, exist_ok=True)
            wine.frame.to_csv('data/samples/wine.csv', index=False)
            st.success("✅ Created: data/samples/wine.csv")

# ==================== RUN EXPERIMENT PAGE ====================
elif page == "📊 Run Experiment":
    st.markdown('<h1 class="main-header">📊 Run ML Experiment</h1>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("📁 Upload Your Dataset (CSV)", type=['csv'])
    
    if uploaded_file:
        # Try multiple encodings to handle different CSV formats
        encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'utf-16']
        df = None
        successful_encoding = None
        
        for encoding in encodings_to_try:
            try:
                uploaded_file.seek(0)  # Reset file pointer to beginning
                df = pd.read_csv(uploaded_file, encoding=encoding)
                successful_encoding = encoding
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception as e:
                st.error(f"❌ Error reading file with {encoding}: {str(e)}")
                break
        
        if df is None:
            st.error("❌ Could not read the CSV file. Please check the file format and encoding.")
        else:
            st.session_state.df = df
            
            if successful_encoding != 'utf-8':
                st.info(f"ℹ️ File loaded with {successful_encoding} encoding")
            
            st.success(f"✅ Dataset Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
            
            with st.expander("👁️ Preview Data & Statistics"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.dataframe(df.head(10))
                
                with col2:
                    st.metric("Total Rows", df.shape[0])
                    st.metric("Total Columns", df.shape[1])
                    st.metric("Missing Values", df.isnull().sum().sum())
                    st.metric("Duplicates", df.duplicated().sum())
            
            st.markdown("---")
            
            learning_type = st.radio(
                "🎯 Select Learning Type:",
                ["Supervised Learning", "Unsupervised Learning"],
                horizontal=True
            )
            
            if learning_type == "Supervised Learning":
                target = st.selectbox("🎯 Select Target Column:", df.columns)
                
                if st.button("🚀 Run Supervised Experiment", type="primary", use_container_width=True):
                    with st.spinner("🔄 Running ML Pipeline..."):
                        try:
                            temp_path = "temp_data.csv"
                            df.to_csv(temp_path, index=False, encoding='utf-8')
                            
                            progress = st.progress(0)
                            status = st.empty()
                            
                            status.text("📊 Data Inspection...")
                            progress.progress(15)
                            time.sleep(0.3)
                            
                            status.text("🔧 Preprocessing...")
                            progress.progress(30)
                            time.sleep(0.3)
                            
                            status.text("🎯 Feature Selection...")
                            progress.progress(50)
                            time.sleep(0.3)
                            
                            status.text("🤖 Training Models...")
                            progress.progress(70)
                            
                            results = st.session_state.orchestrator.run(
                                temp_path, target, 'supervised'
                            )
                            
                            status.text("⚡ Hyperparameter Tuning...")
                            progress.progress(90)
                            
                            st.session_state.results = results
                            
                            progress.progress(100)
                            status.empty()
                            progress.empty()
                            
                            st.balloons()
                            st.success("🎉 Experiment Completed Successfully!")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.markdown(f"""
                                <div style="background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                                    <p style="color: #666; font-size: 0.9rem; margin: 0;">🏆 Best Model</p>
                                    <p style="color: #1e5128; font-size: 2rem; font-weight: 700; margin: 0.5rem 0 0 0;">{results['best_model']}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            with col2:
                                st.markdown(f"""
                                <div style="background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                                    <p style="color: #666; font-size: 0.9rem; margin: 0;">📋 Task Type</p>
                                    <p style="color: #1e5128; font-size: 2rem; font-weight: 700; margin: 0.5rem 0 0 0;">{results['task_type'].upper()}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            with col3:
                                metric_key = list(results['metrics'].keys())[0]
                                st.markdown(f"""
                                <div style="background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                                    <p style="color: #666; font-size: 0.9rem; margin: 0;">{metric_key.upper()}</p>
                                    <p style="color: #1e5128; font-size: 2rem; font-weight: 700; margin: 0.5rem 0 0 0;">{results['metrics'][metric_key]:.4f}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            st.markdown("### 📊 Detailed Metrics")
                            st.dataframe(pd.DataFrame([results['metrics']]), use_container_width=True)
                            
                            st.markdown("### 📈 Visualizations")
                            viz_path = Path("experiments/visualizations")
                            if viz_path.exists():
                                cols = st.columns(2)
                                for idx, viz_file in enumerate(viz_path.glob("*.png")):
                                    with cols[idx % 2]:
                                        img = Image.open(viz_file)
                                        st.image(img, caption=viz_file.stem.replace('_', ' ').title())
                            
                            st.markdown("### 💾 Download Trained Model")
                            if Path(results['model_path']).exists():
                                with open(results['model_path'], 'rb') as f:
                                    st.download_button(
                                        "📥 Download Model (.pkl)",
                                        f,
                                        f"{results['best_model']}_trained.pkl",
                                        use_container_width=True
                                    )
                        
                        except Exception as e:
                            st.error(f"❌ Error occurred: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
            
            else:  # Unsupervised
                if st.button("🚀 Run Unsupervised Experiment", type="primary", use_container_width=True):
                    with st.spinner("🔄 Running Unsupervised Learning..."):
                        try:
                            temp_path = "temp_data.csv"
                            df.to_csv(temp_path, index=False, encoding='utf-8')
                            
                            progress = st.progress(0)
                            status = st.empty()
                            
                            status.text("📊 Processing...")
                            progress.progress(25)
                            
                            status.text("🎨 Dimensionality Reduction...")
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
                                    img = Image.open(viz_file)
                                    st.image(img, use_container_width=True)
                        
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())

# ==================== RESULTS PAGE ====================
elif page == "📈 Results":
    st.markdown('<h1 class="main-header">📈 Experiment Results</h1>', unsafe_allow_html=True)
    
    if st.session_state.results is None:
        st.info("ℹ️ No results available. Please run an experiment first!")
    else:
        results = st.session_state.results
        
        if results['pipeline'] == 'supervised':
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                    <div style='background: #1e5128; border-radius: 1rem; padding: 1rem;'>
                        <div style='font-size: 1rem; color: #d4d4d4; margin-bottom: 0.5rem;'>🏆 Best Model</div>
                        <div style='font-size: 1.75rem; font-weight: 700; color: white;'>{results['best_model']}</div>
                    </div>
                """, unsafe_allow_html=True)
            with col2:
                st.metric("📋 Task Type", results['task_type'].upper())
            with col3:
                metric_key = list(results['metrics'].keys())[0]
                st.metric(metric_key.upper(), f"{results['metrics'][metric_key]:.4f}")
            
            st.dataframe(pd.DataFrame([results['metrics']]), use_container_width=True)
            
            viz_path = Path("experiments/visualizations")
            if viz_path.exists():
                for viz_file in viz_path.glob("*.png"):
                    st.image(Image.open(viz_file))

# ==================== DOCUMENTATION PAGE ====================
elif page == "📚 Documentation":
    st.markdown('<h1 class="main-header">📚 Documentation</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Complete documentation coming soon! For now, explore the app and check the Home page for detailed information.
    """)

# Footer
st.markdown("""
<div class="footer">
    <p style="font-size: 1.1rem;"><strong>ML Experiment Orchestrator v1.0</strong></p>
    <p>Built with ❤️ using Python, Streamlit, and Advanced ML Libraries</p>
    <p>© 2024 All Rights Reserved</p>
</div>
""", unsafe_allow_html=True)