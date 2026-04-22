"""
Page 3 — Model Prediction
===========================
Train and compare 6 regression models side-by-side.
Users can select features, adjust train/test split,
and view performance metrics + prediction plots.
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

    # ── Prepare data ────────────────────────────────────────────────
    X = df[selected_features].values
    y = df[target].values
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
        "Choose models (minimum 2 recommended)",
        list(MODELS.keys()),
        default=list(MODELS.keys())[:5],
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
        help="Set WANDB_API_KEY in .env to enable.",
    )

    # ── Train all models ────────────────────────────────────────────
    if st.button("🚀 Train Models", type="primary", use_container_width=True):
        results = []
        predictions = {}

        progress = st.progress(0, text="Training models...")
        for i, name in enumerate(model_choices):
            run = None
            if track_wandb:
                run = wandb_tracker.init_run(
                    run_name=f"{ds_key}-{name}",
                    config={
                        "dataset": ds_key,
                        "model": name,
                        "target": target,
                        "n_features": len(selected_features),
                        "features": selected_features,
                        "test_size": test_size,
                        "scale_data": scale_data,
                        "train_samples": len(X_train),
                        "test_samples": len(X_test),
                    },
                    job_type="baseline-train",
                )

            model = MODELS[name]()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            predictions[name] = y_pred

            cv_scores = cross_val_score(
                MODELS[name](), X_train, y_train, cv=5, scoring="r2"
            )

            metrics = {
                "R² Score": r2_score(y_test, y_pred),
                "MAE": mean_absolute_error(y_test, y_pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                "MAPE (%)": mean_absolute_percentage_error(y_test, y_pred) * 100,
                "CV R² (mean)": cv_scores.mean(),
                "CV R² (std)": cv_scores.std(),
            }
            results.append({"Model": name, **metrics})

            wandb_tracker.log_metrics(run, {
                "test/r2": metrics["R² Score"],
                "test/mae": metrics["MAE"],
                "test/rmse": metrics["RMSE"],
                "test/mape": metrics["MAPE (%)"],
                "cv/r2_mean": metrics["CV R² (mean)"],
                "cv/r2_std": metrics["CV R² (std)"],
            })
            wandb_tracker.finish_run(run)

            progress.progress(
                (i + 1) / len(model_choices),
                text=f"Trained {name} ✓",
            )

        progress.empty()

        # Store in session state
        st.session_state["pred_results"] = results
        st.session_state["pred_predictions"] = predictions
        st.session_state["pred_y_test"] = y_test
        st.session_state["pred_model_choices"] = model_choices

    # ── Display results ─────────────────────────────────────────────
    if "pred_results" not in st.session_state:
        st.info("Click **Train Models** to see results.")
        return

    results = st.session_state["pred_results"]
    predictions = st.session_state["pred_predictions"]
    y_test = st.session_state["pred_y_test"]
    model_choices = st.session_state["pred_model_choices"]

    results_df = pd.DataFrame(results).set_index("Model")

    # ── Leaderboard ─────────────────────────────────────────────────
    st.markdown("### 🏆 Model Leaderboard")
    sorted_df = results_df.sort_values("R² Score", ascending=False)
    best_model = sorted_df.index[0]
    st.success(f"**Best model: {best_model}** with R² = {sorted_df.loc[best_model, 'R² Score']:.4f}")

    st.dataframe(
        sorted_df.style
        .format({
            "R² Score": "{:.4f}",
            "MAE": "{:.4f}",
            "RMSE": "{:.4f}",
            "MAPE (%)": "{:.2f}",
            "CV R² (mean)": "{:.4f}",
            "CV R² (std)": "{:.4f}",
        })
        .background_gradient(subset=["R² Score"], cmap="Purples")
        .background_gradient(subset=["MAE", "RMSE"], cmap="Purples_r"),
        use_container_width=True,
    )

    st.markdown("---")

    # ── Bar chart comparison ────────────────────────────────────────
    st.markdown("### 📊 Performance Comparison")
    metric_choice = st.selectbox(
        "Metric to compare",
        ["R² Score", "MAE", "RMSE", "MAPE (%)", "CV R² (mean)"],
    )
    fig = px.bar(
        sorted_df.reset_index(),
        x="Model", y=metric_choice,
        color=metric_choice,
        color_continuous_scale="Purples",
        title=f"{metric_choice} by Model",
    )
    fig.update_layout(template="plotly_white", height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Actual vs Predicted ─────────────────────────────────────────
    st.markdown("### 🎯 Actual vs Predicted")
    model_to_plot = st.selectbox("Select model to inspect", model_choices)
    y_pred = predictions[model_to_plot]

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y_test, y=y_pred, mode="markers",
            marker=dict(color="#57068C", opacity=0.4, size=5),
            name="Predictions",
        ))
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode="lines", line=dict(color="red", dash="dash", width=2),
            name="Perfect prediction",
        ))
        fig.update_layout(
            template="plotly_white", height=450,
            xaxis_title="Actual", yaxis_title="Predicted",
            title=f"{model_to_plot} — Actual vs Predicted",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_p2:
        residuals = y_test - y_pred
        fig = px.histogram(
            x=residuals, nbins=50, color_discrete_sequence=["#57068C"],
            title=f"{model_to_plot} — Residual Distribution",
            labels={"x": "Residual (Actual − Predicted)", "count": "Count"},
        )
        fig.update_layout(template="plotly_white", height=450)
        st.plotly_chart(fig, use_container_width=True)