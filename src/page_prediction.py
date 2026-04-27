"""
Page 3 — Model Prediction (Fixed for Categorical Data)
===========================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)
from data_loader import dataset_selector, get_target, get_features
from src import wandb_tracker

def _mlp_factory():
    return MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42, early_stopping=True)

MODELS = {
    "Linear Regression": LinearRegression,
    "🧠 MLP (Neural Net)": _mlp_factory,
    "Ridge Regression": Ridge,
    "Lasso Regression": Lasso,
    "Elastic Net": ElasticNet,
    "Decision Tree": DecisionTreeRegressor,
    "Random Forest": RandomForestRegressor,
    "Gradient Boosting": GradientBoostingRegressor,
}

def render():
    ds_key, df, info = dataset_selector()
    target = get_target(ds_key)
    features = get_features(df, target)

    st.markdown("## 🤖 Model Prediction")
    st.caption("Train multiple regression models and compare their performance.")
    st.markdown("---")

    # ── Feature & split config ──────────────────────────────────────
    col_cfg1, col_cfg2 = st.columns([3, 1])
    with col_cfg1:
        selected_features = st.multiselect(
            "Select explanatory variables",
            features,
            default=features,
        )
    with col_cfg2:
        test_size = st.slider("Test size (%)", 10, 40, 20) / 100
        scale_data = st.checkbox("Standardize features", value=True)

    if not selected_features:
        st.warning("Please select at least one feature.")
        return

    # ── Prepare data (FIXED FOR CATEGORICAL DATA) ───────────────────
    # 1. Create a subset of the dataframe with selected features
    X_raw = df[selected_features]
    
    # 2. Convert categories (sex, smoker, region) into numbers
    # drop_first=True prevents the "Dummy Variable Trap" by removing redundant columns
    X_encoded = pd.get_dummies(X_raw, drop_first=True)
    
    X = X_encoded.values
    y = df[target].values

    # Show the user how the data looks now
    with st.expander("🔍 View Processed Features (One-Hot Encoded)"):
        st.write("Original columns were expanded into numeric columns:")
        st.dataframe(X_encoded.head(), use_container_width=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    if scale_data:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    st.markdown(f"**Training set:** {len(X_train):,} samples · "
                f"**Test set:** {len(X_test):,} samples")
    st.markdown("---")

    # ── Model selection ─────────────────────────────────────────────
    st.markdown("### 🏗️ Select Models to Train")
    model_choices = st.multiselect(
        "Choose models",
        list(MODELS.keys()),
        default=["Linear Regression", "Random Forest", "Gradient Boosting"],
        label_visibility="collapsed",
    )

    if len(model_choices) < 1:
        st.warning("Select at least one model.")
        return

    # ── W&B toggle ──────────────────────────────────────────────────
    track_wandb = st.checkbox(
        "📡 Log runs to Weights & Biases",
        value=wandb_tracker.is_available(),
        disabled=not wandb_tracker.is_available(),
    )

    # ── Train all models ────────────────────────────────────────────
    if st.button("🚀 Train Models", type="primary", use_container_width=True):
        results = []
        predictions = {}

        progress = st.progress(0, text="Training models...")
        for i, name in enumerate(model_choices):
            # ... (W&B logic remains the same)
            run = None
            if track_wandb:
                run = wandb_tracker.init_run(
                    run_name=f"{ds_key}-{name}",
                    config={"model": name, "test_size": test_size},
                    job_type="baseline-train",
                )

            model = MODELS[name]()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            predictions[name] = y_pred

            # CV Scores
            cv_scores = cross_val_score(
                MODELS[name](), X_train, y_train, cv=5, scoring="r2"
            )

            metrics = {
                "Model": name,
                "R² Score": r2_score(y_test, y_pred),
                "MAE": mean_absolute_error(y_test, y_pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                "MAPE (%)": mean_absolute_percentage_error(y_test, y_pred) * 100,
                "CV R² (mean)": cv_scores.mean(),
            }
            results.append(metrics)
            
            # ... (Logging logic remains the same)
            wandb_tracker.log_metrics(run, {"test/r2": metrics["R² Score"]})
            wandb_tracker.finish_run(run)

            progress.progress((i + 1) / len(model_choices), text=f"Trained {name} ✓")

        progress.empty()
        st.session_state["pred_results"] = results
        st.session_state["pred_predictions"] = predictions
        st.session_state["pred_y_test"] = y_test
        st.session_state["pred_model_choices"] = model_choices

    # ── Display results (Rest of the code remains the same) ─────────
    if "pred_results" in st.session_state:
        results_df = pd.DataFrame(st.session_state["pred_results"]).set_index("Model")
        st.markdown("### 🏆 Model Leaderboard")
        st.dataframe(results_df.style.background_gradient(subset=["R² Score"], cmap="Purples"))
        # ... (rest of the visualization logic)