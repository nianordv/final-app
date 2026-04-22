"""
DS4EVERYONE @ NYU — Final Project: ML Prediction App
=====================================================
A professional multi-page Streamlit application demonstrating
end-to-end machine learning: EDA, visualization, prediction
with 6+ models, explainability (SHAP), and hyperparameter tuning.

Author: DS4E Students | Professor Gaëtan Brison
Course: DS-UA 9111 — Data Science for Everyone
"""

import streamlit as st

# ── Page config (must be first Streamlit call) ──────────────────────
st.set_page_config(
    page_title="DS4E — ML Prediction App",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
    /* NYU purple accent */
    :root {
        --nyu-purple: #57068C;
        --nyu-purple-light: #8900E1;
    }
    .stApp > header {background-color: transparent;}
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #57068C 0%, #330042 100%);
    }
    [data-testid="stSidebar"] * {color: white !important;}
    [data-testid="stSidebar"] code {
        background: rgba(255,255,255,0.18) !important;
        color: #ffd966 !important;
        padding: 1px 6px;
        border-radius: 4px;
        font-weight: 600;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stMultiSelect label {color: white !important;}
    .metric-card {
        background: #f8f4fc;
        border-left: 4px solid #57068C;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .metric-card h3 {margin: 0; font-size: 0.85rem; color: #666;}
    .metric-card p {margin: 0; font-size: 1.6rem; font-weight: 700; color: #57068C;}
    .hero-banner {
        background: linear-gradient(135deg, #57068C 0%, #8900E1 50%, #b347ff 100%);
        color: white;
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
    }
    .hero-banner h1 {color: white; font-size: 2.2rem; margin-bottom: 0.3rem;}
    .hero-banner p {color: #e0c8f0; font-size: 1.1rem;}
    div[data-testid="stMetric"] {
        background: #f8f4fc;
        border: 1px solid #e0d0f0;
        border-radius: 10px;
        padding: 12px 16px;
    }
</style>
""", unsafe_allow_html=True)


# ── Navigation ──────────────────────────────────────────────────────
from src import page_intro, page_visualization, page_prediction, page_explainability, page_tuning
from src import wandb_tracker

PAGES = {
    "🏠 Business Case & Data": page_intro,
    "📊 Data Visualization": page_visualization,
    "🤖 Model Prediction": page_prediction,
    "🔍 Explainability (SHAP)": page_explainability,
    "⚙️ Hyperparameter Tuning": page_tuning,
}

with st.sidebar:
    st.markdown("## 🎓 DS4E @ NYU")
    st.markdown("---")
    selected = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
    st.markdown("---")
    wandb_tracker.status_badge()
    st.caption("DS-UA 9111 · Prof. Gaëtan Brison")
    st.caption("© 2026 NYU Data Science for Everyone")

# ── Render selected page ────────────────────────────────────────────
PAGES[selected].render()