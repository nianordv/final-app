"""
Shared data loading and preprocessing utilities for the Insurance Dataset.
"""

import streamlit as st
import pandas as pd
import numpy as np

# ── Available datasets ──────────────────────────────────────────────
# We've updated this to focus on the new Insurance dataset
DATASETS = {
    "🏥 Medical Insurance": "insurance",
}

DATASET_DESCRIPTIONS = {
    "insurance": {
        "title": "Medical Insurance Cost Prediction",
        "problem": (
            "**Business Problem:** An insurance company wants to accurately predict "
            "individual medical costs billed by health insurance. This helps in "
            "setting competitive premiums and managing financial risk based on "
            "patient demographics and lifestyle factors."
        ),
        "target": "charges",
        "target_desc": "Individual medical costs billed by health insurance ($)",
        "source": "Medical Cost Personal Datasets (Kaggle)",
        "rows": "1,338 beneficiaries",
        "features_desc": {
            "age": "Age of primary beneficiary",
            "sex": "Insurance contractor gender (female, male)",
            "bmi": "Body mass index (kg/m²)",
            "children": "Number of children/dependents covered",
            "smoker": "Smoking status (yes, no)",
            "region": "Beneficiary's residential area in the US",
        },
    },
}

@st.cache_data
def load_data(dataset_key: str) -> pd.DataFrame:
    """Load, clean, and return the Insurance DataFrame from CSV."""
    if dataset_key == "insurance":
        df = pd.read_csv("insurance.csv")

        # Encode binary categorical variables
        df["smoker"] = df["smoker"].map({"yes": 1, "no": 0})
        df["sex"] = df["sex"].map({"male": 1, "female": 0})

        # One-hot encode region
        df = pd.get_dummies(df, columns=["region"], drop_first=True)

        # Interaction features
        df["smoker_bmi"] = df["smoker"] * df["bmi"]
        df["smoker_age"] = df["smoker"] * df["age"]

        # High-risk feature 
        df["high_risk_smoker"] = ((df["smoker"] == 1) & (df["bmi"] >= 30)).astype(int)

    else:
        raise ValueError(f"Unknown dataset: {dataset_key}")

    return df

def get_target(dataset_key: str) -> str:
    return DATASET_DESCRIPTIONS[dataset_key]["target"]

def get_features(df: pd.DataFrame, target: str) -> list[str]:
    return [c for c in df.columns if c != target]

def dataset_selector() -> tuple[str, pd.DataFrame, dict]:
    """Render a dataset selector in the sidebar and return (key, df, info)."""
    with st.sidebar:
        st.markdown("### 📂 Dataset")
        choice = st.selectbox(
            "Choose a dataset",
            list(DATASETS.keys()),
            label_visibility="collapsed",
        )
    key = DATASETS[choice]
    df = load_data(key)
    info = DATASET_DESCRIPTIONS[key]
    return key, df, info
