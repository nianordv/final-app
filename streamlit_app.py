import streamlit as st

# 1. PAGE CONFIG (Must be first)
st.set_page_config(
    page_title="CS:GO Strat-Oracle | DS4E NYU",
    page_icon="🔫",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. IMPORTS (From your src folder)
# Ensure these files exist in your 'src' directory
try:
    from src import page_intro, page_visualization, page_prediction, page_explainability, page_tuning
    from src import wandb_tracker
except ImportError:
    st.error("Could not find 'src' modules. Ensure your folder structure is correct.")

# 3. DEFINE PAGES DICTIONARY (Must be before the sidebar uses it)
PAGES = {
    "🎮 Business Case & Data": page_intro,
    "📊 Data Visualization": page_visualization,
    "🤖 Model Prediction": page_prediction,
    "🔍 Explainability (SHAP)": page_explainability,
    "⚙️ Hyperparameter Tuning": page_tuning,
}

# 4. TACTICAL CSS
st.markdown("""
<style>
    :root {
        --ct-blue: #5d79ae;
        --t-orange: #de9b35;
        --cs-dark: #1b2838;
        --nyu-purple: #57068C;
    }
    .stApp { background-color: #0b1119; color: #c7d5e0; }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #16202d 0%, #0b1119 100%);
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    
    .metric-card {
        background: rgba(42, 71, 94, 0.3);
        border-left: 5px solid var(--ct-blue);
        padding: 1rem;
        border-radius: 4px;
        font-family: 'Courier New', monospace;
    }
    
    .hero-banner {
        background: linear-gradient(90deg, #57068C 0%, #1b2838 100%);
        border: 1px solid rgba(103, 193, 245, 0.3);
        padding: 2rem;
        border-radius: 4px;
    }
    
    /* HUD-style text */
    h1, h2, h3 { font-family: 'Courier New', monospace; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)

# 5. SIDEBAR NAVIGATION
with st.sidebar:
    st.markdown("### 🛠️ STRAT-ORACLE V1.0")
    st.markdown("---")
    
    # Now PAGES is defined, so this won't throw a NameError
    selected = st.radio("SELECT OPERATION:", list(PAGES.keys()))
    
    st.markdown("---")
    try:
        wandb_tracker.status_badge()
    except:
        pass
    st.caption("DS-UA 9111 · Prof. Gaëtan Brison")

# 6. RENDER THE PAGE
PAGES[selected].render()