"""
Page 4 — Explainability (SHAP)
================================
Feature importance analysis using SHAP values, permutation importance,
and model-specific importances.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from data_loader import dataset_selector, get_target, get_features

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"


def _cache_path(ds_key: str, model_name: str) -> Path:
    safe = model_name.lower().replace(" ", "_")
    return CACHE_DIR / f"importance_{ds_key}_{safe}.pkl"


def load_cached(ds_key: str, model_name: str) -> dict | None:
    path = _cache_path(ds_key, model_name)
    if not path.exists():
        return None
    try:
        with path.open("rb") as fh:
            return pickle.load(fh)
    except Exception:
        return None


def compute_live(ds_key: str, model_name: str, df: pd.DataFrame, target: str, features: list[str]) -> dict:
    # Subsample large datasets
    MAX_TRAIN = 5000
    MAX_TEST = 500

    # --- FIX: Encoding Categorical Data ---
    # We apply get_dummies here so the model receives only numbers
    X_raw = df[features]
    y_full = df[target]
    
    X_encoded = pd.get_dummies(X_raw, drop_first=True)
    # Important: Update the 'features' list to match the new encoded column names
    encoded_features = X_encoded.columns.tolist()
    
    if len(df) > MAX_TRAIN + MAX_TEST:
        X_encoded = X_encoded.sample(n=MAX_TRAIN + MAX_TEST, random_state=42)
        y_full = y_full.loc[X_encoded.index]

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y_full, test_size=min(MAX_TEST / len(X_encoded), 0.2), random_state=42
    )

    if model_name == "Random Forest":
        model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1)
    else:
        model = GradientBoostingRegressor(n_estimators=50, random_state=42)
        
    model.fit(X_train, y_train)

    # Use 'encoded_features' for labeling plots
    imp_df = pd.DataFrame({
        "Feature": encoded_features,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=True)

    perm = permutation_importance(
        model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=1
    )
    perm_df = pd.DataFrame({
        "Feature": encoded_features,
        "Importance": perm.importances_mean,
        "Std": perm.importances_std,
    }).sort_values("Importance", ascending=True)

    payload = {
        "features": encoded_features, # Use encoded names
        "imp_df": imp_df,
        "perm_df": perm_df,
        "X_test": X_test.reset_index(drop=True),
    }

    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        shap_arr = np.asarray(shap_values)
        payload["shap_values"] = shap_arr
        payload["shap_df"] = pd.DataFrame({
            "Feature": encoded_features,
            "Mean |SHAP|": np.abs(shap_arr).mean(axis=0),
        }).sort_values("Mean |SHAP|", ascending=True)
    except Exception as exc:
        payload["shap_error"] = str(exc)

    return payload


def render() -> None:
    ds_key, df, info = dataset_selector()
    target = get_target(ds_key)
    features = get_features(df, target)

    st.markdown("## 🔍 Explainability — Feature Importance")
    st.caption(
        "Understand which features drive predictions using SHAP values, "
        "permutation importance, and built-in model importances."
    )
    st.markdown("---")

    model_name = st.selectbox(
        "Model for explainability analysis",
        ["Random Forest", "Gradient Boosting"],
    )

    cache_key = f"importance_{ds_key}_{model_name}"

    payload = load_cached(ds_key, model_name)
    if payload is None:
        payload = st.session_state.get(cache_key)

    if payload is not None:
        st.success(f"✨ Importances ready for **{ds_key}** · **{model_name}**.")
    else:
        st.info(
            "No precomputed cache available. "
            "Click below to train the model and compute importances live."
        )
        if st.button("🔬 Compute Feature Importance Now", type="primary", use_container_width=True):
            with st.spinner("Training model and computing importances — this may take a moment..."):
                payload = compute_live(ds_key, model_name, df, target, features)
            st.session_state[cache_key] = payload
            st.rerun()
        return

    # Pass the features found in the payload, not the original raw ones
    _render_importance(payload, model_name, payload["features"])


def _render_importance(payload: dict, model_name: str, features: list[str]) -> None:
    imp_df = payload["imp_df"]
    perm_df = payload["perm_df"]
    has_shap = "shap_df" in payload

    tab_names = ["🌲 Model Importance", "🔀 Permutation Importance"]
    if has_shap:
        tab_names.append("💎 SHAP Values")
    tabs = st.tabs(tab_names)

    with tabs[0]:
        st.markdown(f"#### {model_name} — Built-in Feature Importance")
        fig = px.bar(
            imp_df, x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale="Purples",
        )
        fig.update_layout(template="plotly_white", height=max(400, len(features) * 30))
        st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        st.markdown("#### Permutation Importance")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=perm_df["Feature"], x=perm_df["Importance"],
            orientation="h",
            marker_color="#57068C",
            error_x=dict(type="data", array=perm_df["Std"]),
        ))
        fig.update_layout(
            template="plotly_white",
            xaxis_title="Mean importance decrease",
            height=max(400, len(features) * 30),
        )
        st.plotly_chart(fig, use_container_width=True)

    if has_shap:
        with tabs[2]:
            shap_df = payload["shap_df"]
            shap_values = payload["shap_values"]
            X_test_df = payload["X_test"]

            st.markdown("#### SHAP — SHapley Additive exPlanations")
            fig = px.bar(
                shap_df, x="Mean |SHAP|", y="Feature", orientation="h",
                color="Mean |SHAP|", color_continuous_scale="Purples",
                title="Mean |SHAP| Value per Feature",
            )
            fig.update_layout(template="plotly_white", height=max(400, len(features) * 30))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### SHAP Feature Impact (Beeswarm-style)")
            top_n = min(10, len(features))
            top_features = shap_df.tail(top_n)["Feature"].tolist()

            MAX_POINTS = 400
            n_samples = len(shap_values)
            if n_samples > MAX_POINTS:
                rng = np.random.default_rng(42)
                sample_idx = rng.choice(n_samples, size=MAX_POINTS, replace=False)
            else:
                sample_idx = np.arange(n_samples)

            fig = go.Figure()
            for feat in top_features:
                idx = features.index(feat)
                fig.add_trace(go.Scatter(
                    x=shap_values[sample_idx, idx],
                    y=[feat] * len(sample_idx),
                    mode="markers",
                    marker=dict(
                        color=X_test_df[feat].values[sample_idx],
                        colorscale="Purples",
                        size=5,
                        opacity=0.55,
                        colorbar=dict(title="Feature value") if feat == top_features[-1] else None,
                        showscale=(feat == top_features[-1]),
                    ),
                    name=feat,
                    showlegend=False,
                ))
            fig.update_layout(
                template="plotly_white",
                xaxis_title="SHAP value (impact on prediction)",
                height=max(400, top_n * 50),
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("💡 **Interpretation:** Features with high importance strongly influence predictions.")