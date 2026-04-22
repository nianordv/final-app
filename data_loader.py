"""
Shared data loading and preprocessing utilities for CS:GO Datasets.
Provides cached dataset loading for all pages.
"""

import streamlit as st
import pandas as pd
import numpy as np

# ── Available datasets ──────────────────────────────────────────────
DATASETS = {
    "🎮 CS:GO Match Results": "results",
    "💰 CS:GO Economy": "economy",
}

DATASET_DESCRIPTIONS = {
    "results": {
        "title": "CS:GO Match Results",
        "problem": (
            "**Business Problem:** Professional esports organizations need to predict "
            "match outcomes based on team rankings and map selection to optimize "
            "veto strategies and betting models."
        ),
        "target": "match_winner",
        "target_desc": "Final winner of the match (1 or 2)",
        "source": "HLTV Professional Match Records",
        "rows": "45,773 matches",
        "features_desc": {
            "rank_1": "World ranking of Team 1",
            "rank_2": "World ranking of Team 2",
            "_map": "The specific map being played",
            "starting_ct": "Which team started on the Counter-Terrorist side",
            "ct_1": "Rounds won by Team 1 as CT",
            "t_1": "Rounds won by Team 1 as T",
            "map_winner": "Winner of the individual map"
        },
    },
    "economy": {
        "title": "CS:GO Economy & Round Analysis",
        "problem": (
            "**Business Problem:** Analysts want to understand the correlation between "
            "round-start equipment value and win probability to better manage "
            "in-game finances (Eco vs. Buy rounds)."
        ),
        "target": "1_winner",
        "target_desc": "Winner of the first round (Pistol round)",
        "source": "HLTV Economy Logs",
        "rows": "43,234 entries",
        "features_desc": {
            "1_t1": "Team 1 equipment value in Round 1",
            "1_t2": "Team 2 equipment value in Round 1",
            "t1_start": "Starting side of Team 1 (ct/t)",
            "best_of": "Series format (BO1, BO3, BO5)",
            "_map": "The map being played"
        },
    },
}

@st.cache_data
def load_data(dataset_key: str) -> pd.DataFrame:
    """Load and return a clean DataFrame for the chosen CS:GO dataset."""
    if dataset_key == "results":
        # Load results dataset
        df = pd.read_csv('results.csv')
        # Optional: ensure date is datetime for better sparklines
        df['date'] = pd.to_datetime(df['date'])
    elif dataset_key == "economy":
        # Load economy dataset
        df = pd.read_csv('economy.csv')
        df['date'] = pd.to_datetime(df['date'])
    else:
        raise ValueError(f"Unknown dataset: {dataset_key}")
    return df

def get_target(dataset_key: str) -> str:
    return DATASET_DESCRIPTIONS[dataset_key]["target"]

def get_features(df: pd.DataFrame, target: str) -> list[str]:
    """Excludes non-predictive ID columns and the target itself."""
    exclude = [target, "match_id", "event_id", "date"]
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

def dataset_selector() -> tuple[str, pd.DataFrame, dict]:
    """Render a dataset selector in the sidebar and return (key, df, info)."""
    with st.sidebar:
        st.markdown("### 📂 CS:GO Dataset")
        choice = st.selectbox(
            "Choose a dataset",
            list(DATASETS.keys()),
            label_visibility="collapsed",
        )
    key = DATASETS[choice]
    df = load_data(key)
    info = DATASET_DESCRIPTIONS[key]
    return key, df, info