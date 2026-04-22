"""
Shared data loading and preprocessing utilities.
Provides cached dataset loading for all pages.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import (
    fetch_california_housing,
    load_wine,
    load_diabetes,
)


# ── Available datasets ──────────────────────────────────────────────
DATASETS = {
    "🏠 California Housing": "california",
    "🍷 Wine Quality": "wine",
    "🩺 Diabetes Progression": "diabetes",
}

DATASET_DESCRIPTIONS = {
    "california": {
        "title": "California Housing Prices",
        "problem": (
            "**Business Problem:** A real-estate investment firm needs to predict "
            "median house values across California districts to identify undervalued "
            "markets and optimize their acquisition strategy."
        ),
        "target": "MedHouseVal",
        "target_desc": "Median house value (in $100k)",
        "source": "StatLib — 1990 U.S. Census",
        "rows": "20,640 districts",
        "features_desc": {
            "MedInc": "Median income in the district ($10k)",
            "HouseAge": "Median house age (years)",
            "AveRooms": "Average rooms per household",
            "AveBedrms": "Average bedrooms per household",
            "Population": "District population",
            "AveOccup": "Average household members",
            "Latitude": "District latitude",
            "Longitude": "District longitude",
        },
    },
    "wine": {
        "title": "Wine Quality Classification",
        "problem": (
            "**Business Problem:** A vineyard cooperative wants to predict wine "
            "quality scores from chemical properties to optimize their production "
            "process and reduce costly lab testing."
        ),
        "target": "quality",
        "target_desc": "Wine quality score",
        "source": "UCI Machine Learning Repository",
        "rows": "178 samples",
        "features_desc": {
            "alcohol": "Alcohol content (%)",
            "malic_acid": "Malic acid concentration",
            "ash": "Ash content",
            "alcalinity_of_ash": "Alkalinity of ash",
            "magnesium": "Magnesium content",
            "total_phenols": "Total phenols",
            "flavanoids": "Flavanoid concentration",
            "nonflavanoid_phenols": "Non-flavanoid phenols",
            "proanthocyanins": "Proanthocyanin concentration",
            "color_intensity": "Color intensity",
            "hue": "Hue",
            "od280/od315_of_diluted_wines": "OD280/OD315 ratio",
            "proline": "Proline content",
        },
    },
    "diabetes": {
        "title": "Diabetes Disease Progression",
        "problem": (
            "**Business Problem:** A healthcare provider wants to predict diabetes "
            "disease progression one year after baseline to enable early intervention "
            "and personalized treatment plans, reducing long-term care costs."
        ),
        "target": "progression",
        "target_desc": "Disease progression measure",
        "source": "Efron et al., 2004 — Annals of Statistics",
        "rows": "442 patients",
        "features_desc": {
            "age": "Age (normalized)",
            "sex": "Sex (normalized)",
            "bmi": "Body mass index (normalized)",
            "bp": "Average blood pressure (normalized)",
            "s1": "Total serum cholesterol",
            "s2": "Low-density lipoproteins",
            "s3": "High-density lipoproteins",
            "s4": "Total cholesterol / HDL ratio",
            "s5": "Log of serum triglycerides",
            "s6": "Blood sugar level",
        },
    },
}


@st.cache_data
def load_data(dataset_key: str) -> pd.DataFrame:
    """Load and return a clean DataFrame for the chosen dataset."""
    if dataset_key == "california":
        data = fetch_california_housing(as_frame=True)
        df = data.frame
    elif dataset_key == "wine":
        data = load_wine(as_frame=True)
        df = data.frame
        df.rename(columns={"target": "quality"}, inplace=True)
    elif dataset_key == "diabetes":
        data = load_diabetes(as_frame=True)
        df = data.frame
        df.rename(columns={"target": "progression"}, inplace=True)
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