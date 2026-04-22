import streamlit as st

# 1. PAGE CONFIG
st.set_page_config(
    page_title="CS:GO Strat-Oracle | DS4E NYU",
    page_icon="🔫",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. IMPORTS
try:
    from src import page_intro, page_visualization, page_prediction, page_explainability, page_tuning
    from src import wandb_tracker
except ImportError:
    pass # Fallback if modules aren't linked yet

# 3. DEFINE PAGES
PAGES = {
    "🏠 Business Case & Data": page_intro,
    "📊 Data Visualization": page_visualization,
    "🤖 Model Prediction": page_prediction,
    "🔍 Explainability (SHAP)": page_explainability,
    "⚙️ Hyperparameter Tuning": page_tuning,
}

# 4. TACTICAL ORANGE & BLUE CSS
st.markdown("""
<style>
    :root {
        --ct-blue: #5d79ae;
        --ct-blue-glow: #67c1f5;
        --t-orange: #de9b35;
        --cs-dark-bg: #0b1119;
        --cs-panel: #1b2838;
    }

    /* Background and Global Text */
    .stApp {
        background-color: var(--cs-dark-bg);
        color: #c7d5e0;
    }

    /* Sidebar - Deep Dark Blue */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1119 0%, #1b2838 100%);
        border-right: 1px solid var(--ct-blue);
    }
    [data-testid="stSidebar"] * { color: #c7d5e0 !important; }

    .hero-banner {
        background: linear-gradient(
            135deg, 
            #1b2838 0%,    /* Dark CS-Panel Blue */
            #2a475e 50%,   /* Mid-tone Steam Blue */
            #5d79ae 100%   /* Bright Tactical Blue */
        );
        /* Add a glowing border so it stands out from the dark background */
        border: 1px solid #67c1f5;
        box-shadow: 0 0 20px rgba(103, 193, 245, 0.2);
        
        padding: 2.5rem;
        border-radius: 8px;
        color: white;
        margin-bottom: 2rem;
    }
            /* Keep the text glowing against the dark background */
    .hero-banner h1 {
        text-shadow: 2px 2px 10px rgba(103, 193, 245, 0.5);
        color: #ffffff;
    }

    /* Metric Cards - HUD Style */
    .metric-card {
        background: var(--cs-panel);
        border: 1px solid rgba(103, 193, 245, 0.2);
        border-left: 5px solid var(--t-orange); /* Orange accent for T side vibes */
        padding: 1.2rem;
        border-radius: 4px;
    }
    .metric-card h3 {
        color: #888;
        font-size: 0.75rem;
        letter-spacing: 1px;
    }
    .metric-card p {
        color: var(--ct-blue-glow) !important;
        
        font-size: 1.8rem;
        text-shadow: 0 0 8px rgba(103, 193, 245, 0.4);
    }

    /* Tabs and Radio Buttons */
    .st-bb { border-bottom: 1px solid var(--ct-blue); }
    .st-at { background-color: var(--ct-blue); }

    /* Highlighting Target Variables */
    code {
        color: var(--t-orange) !important;
        background-color: rgba(222, 155, 53, 0.1) !important;
        border: 1px solid rgba(222, 155, 53, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# 5. SIDEBAR NAVIGATION
with st.sidebar:
    st.markdown(f"<h2 style='color:{'#de9b35'}'>S.T.R.A.T.</h2>", unsafe_allow_html=True)
    st.markdown("### 🛠️ ORACLE V1.0")
    st.markdown("---")
    selected = st.radio("SELECT MISSION:", list(PAGES.keys()))
    st.markdown("---")
    st.caption("DEPT: NYU DATA SCIENCE")
    st.caption("STATUS: OPERATIONAL")

# 6. RENDER
PAGES[selected].render()