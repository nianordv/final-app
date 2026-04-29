"""
Page 3 — Model Prediction
=========================
Train and compare regression models, then allow the user to make
an individual insurance cost prediction.
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
    return MLPRegressor(
        hidden_layer_sizes=(64, 32),
        max_iter=500,
        random_state=42,
        early_stopping=True,
    )


MODELS = {
    "Linear Regression": LinearRegression,
    "Ridge Regression": Ridge,
    "Lasso Regression": Lasso,
    "Elastic Net": ElasticNet,
    "Decision Tree": DecisionTreeRegressor,
    "Random Forest": RandomForestRegressor,
    "Gradient Boosting": GradientBoostingRegressor,
    "🧠 MLP (Neural Net)": _mlp_factory,
}


def build_prediction_input(
    age: int,
    sex_input: str,
    bmi: float,
    children: int,
    smoker_input: str,
    region_input: str,
) -> pd.DataFrame:
    """Create one new user row with both raw and engineered insurance features."""

    smoker_numeric = 1 if smoker_input == "yes" else 0
    sex_numeric = 1 if sex_input == "male" else 0

    row = {
        "age": age,
        "sex": sex_input,
        "sex_male": sex_numeric,
        "bmi": bmi,
        "children": children,
        "smoker": smoker_input,
        "smoker_yes": smoker_numeric,
        "region": region_input,
        "region_northwest": 1 if region_input == "northwest" else 0,
        "region_southeast": 1 if region_input == "southeast" else 0,
        "region_southwest": 1 if region_input == "southwest" else 0,
        "smoker_bmi": smoker_numeric * bmi,
        "smoker_age": smoker_numeric * age,
        "high_risk_smoker": int(smoker_numeric == 1 and bmi >= 30),
    }

    return pd.DataFrame([row])


def render():
    ds_key, df, info = dataset_selector()
    target = get_target(ds_key)
    features = get_features(df, target)

    st.markdown("## 🤖 Model Prediction")
    st.caption(
        "Train multiple regression models, compare their performance, "
        "and predict insurance charges for a new person."
    )
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
    X_raw = df[selected_features].copy()
    y = df[target].values

    # Convert categorical variables to numeric dummy columns
    X_encoded = pd.get_dummies(X_raw, drop_first=True)

    with st.expander("🔍 View Processed Features"):
        st.write("The selected features are converted into numeric columns before training.")
        st.dataframe(X_encoded.head(), use_container_width=True)

    X = X_encoded.values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
    )

    scaler = None
    if scale_data:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    st.markdown(
        f"**Training set:** {len(X_train):,} samples · "
        f"**Test set:** {len(X_test):,} samples"
    )
    st.markdown("---")

    # ── Model selection ─────────────────────────────────────────────
    st.markdown("### 🏗️ Select Models to Train")

    model_choices = st.multiselect(
        "Choose models",
        list(MODELS.keys()),
        default=list(MODELS.keys()),
    )

    if len(model_choices) < 5:
        st.warning("For the project requirement, select at least 5 models.")

    if len(model_choices) < 1:
        st.warning("Select at least one model.")
        return

    # ── W&B toggle ─────────────────────────────────────────────────
    track_wandb = st.checkbox(
        "📡 Log runs to Weights & Biases",
        value=wandb_tracker.is_available(),
        disabled=not wandb_tracker.is_available(),
        help="Set WANDB_API_KEY in .env to enable.",
    )

    # ── Train models ────────────────────────────────────────────────
    if st.button("🚀 Train Models", type="primary", use_container_width=True):
        results = []
        predictions = {}
        trained_models = {}

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
                        "features": selected_features,
                        "n_features": len(selected_features),
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
            trained_models[name] = model

            cv_scores = cross_val_score(
                MODELS[name](),
                X_train,
                y_train,
                cv=5,
                scoring="r2",
            )

            metrics = {
                "Model": name,
                "R² Score": r2_score(y_test, y_pred),
                "MAE": mean_absolute_error(y_test, y_pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                "MAPE (%)": mean_absolute_percentage_error(y_test, y_pred) * 100,
                "CV R² (mean)": cv_scores.mean(),
                "CV R² (std)": cv_scores.std(),
            }

            results.append(metrics)

            wandb_tracker.log_metrics(
                run,
                {
                    "test/r2": metrics["R² Score"],
                    "test/mae": metrics["MAE"],
                    "test/rmse": metrics["RMSE"],
                    "test/mape": metrics["MAPE (%)"],
                    "cv/r2_mean": metrics["CV R² (mean)"],
                    "cv/r2_std": metrics["CV R² (std)"],
                },
            )
            wandb_tracker.finish_run(run)

            progress.progress(
                (i + 1) / len(model_choices),
                text=f"Trained {name} ✓",
            )

        progress.empty()

        st.session_state["pred_results"] = results
        st.session_state["pred_predictions"] = predictions
        st.session_state["pred_y_test"] = y_test
        st.session_state["pred_model_choices"] = model_choices
        st.session_state["pred_trained_models"] = trained_models
        st.session_state["pred_scaler"] = scaler
        st.session_state["pred_encoded_columns"] = X_encoded.columns.tolist()
        st.session_state["pred_selected_features"] = selected_features
        st.session_state["pred_scale_data"] = scale_data

    # ── Display results ─────────────────────────────────────────────
    if "pred_results" not in st.session_state:
        st.info("Click **Train Models** to see results and unlock the prediction form.")
        return

    results_df = pd.DataFrame(st.session_state["pred_results"]).set_index("Model")
    sorted_df = results_df.sort_values("R² Score", ascending=False)

    st.markdown("### 🏆 Model Leaderboard")

    best_model = sorted_df.index[0]
    best_r2 = sorted_df.loc[best_model, "R² Score"]

    st.success(f"Best model: **{best_model}** with R² = **{best_r2:.4f}**")

    st.dataframe(
        sorted_df.style.format(
            {
                "R² Score": "{:.4f}",
                "MAE": "{:.2f}",
                "RMSE": "{:.2f}",
                "MAPE (%)": "{:.2f}",
                "CV R² (mean)": "{:.4f}",
                "CV R² (std)": "{:.4f}",
            }
        ).background_gradient(subset=["R² Score"], cmap="Purples"),
        use_container_width=True,
    )

    st.markdown("---")

    # ── Performance comparison ─────────────────────────────────────
    st.markdown("### 📊 Performance Comparison")

    metric_choice = st.selectbox(
        "Metric to compare",
        ["R² Score", "MAE", "RMSE", "MAPE (%)", "CV R² (mean)"],
    )

    fig_bar = px.bar(
        sorted_df.reset_index(),
        x="Model",
        y=metric_choice,
        color=metric_choice,
        color_continuous_scale="Purples",
        title=f"{metric_choice} by Model",
    )
    fig_bar.update_layout(template="plotly_white", height=400)
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    # ── Actual vs predicted and residuals ───────────────────────────
    st.markdown("### 🎯 Actual vs Predicted")

    model_to_plot = st.selectbox(
        "Select model to inspect",
        st.session_state["pred_model_choices"],
    )

    y_test_saved = st.session_state["pred_y_test"]
    y_pred = st.session_state["pred_predictions"][model_to_plot]

    col_p1, col_p2 = st.columns(2)

    with col_p1:
        fig_scatter = go.Figure()

        fig_scatter.add_trace(
            go.Scatter(
                x=y_test_saved,
                y=y_pred,
                mode="markers",
                marker=dict(color="#57068C", opacity=0.45, size=6),
                name="Predictions",
            )
        )

        min_val = min(y_test_saved.min(), y_pred.min())
        max_val = max(y_test_saved.max(), y_pred.max())

        fig_scatter.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                line=dict(color="red", dash="dash", width=2),
                name="Perfect prediction",
            )
        )

        fig_scatter.update_layout(
            template="plotly_white",
            height=450,
            xaxis_title="Actual Charges",
            yaxis_title="Predicted Charges",
            title=f"{model_to_plot} — Actual vs Predicted",
        )

        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_p2:
        residuals = y_test_saved - y_pred

        fig_resid = px.histogram(
            x=residuals,
            nbins=50,
            color_discrete_sequence=["#57068C"],
            title=f"{model_to_plot} — Residual Distribution",
            labels={"x": "Residual (Actual − Predicted)", "count": "Count"},
        )
        fig_resid.update_layout(template="plotly_white", height=450)

        st.plotly_chart(fig_resid, use_container_width=True)

    st.markdown("---")

    # ── Individual prediction form ─────────────────────────────────
    st.markdown("### 💰 Predict Insurance Cost for a New Person")

    st.write(
        "Use the form below to estimate medical insurance charges for a new individual "
        "using one of the trained models."
    )

    chosen_model = st.selectbox(
        "Choose trained model for prediction",
        list(st.session_state["pred_trained_models"].keys()),
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=19, step=1)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)

    with col2:
        children = st.number_input("Number of children", min_value=0, max_value=10, value=0, step=1)
        smoker_input = st.selectbox("Smoker", ["no", "yes"])

    with col3:
        sex_input = st.selectbox("Sex", ["female", "male"])
        region_input = st.selectbox(
            "Region",
            ["northeast", "northwest", "southeast", "southwest"],
        )

    input_raw = build_prediction_input(
        age=age,
        sex_input=sex_input,
        bmi=bmi,
        children=children,
        smoker_input=smoker_input,
        region_input=region_input,
    )

    selected_features_saved = st.session_state["pred_selected_features"]

    input_selected = pd.DataFrame()
    for col in selected_features_saved:
        if col in input_raw.columns:
            input_selected[col] = input_raw[col]
        else:
            input_selected[col] = 0

    input_encoded = pd.get_dummies(input_selected, drop_first=True)

    input_encoded = input_encoded.reindex(
        columns=st.session_state["pred_encoded_columns"],
        fill_value=0,
    )

    input_values = input_encoded.values

    if st.session_state["pred_scaler"] is not None:
        input_values = st.session_state["pred_scaler"].transform(input_values)

    if st.button("🔮 Predict Insurance Cost", use_container_width=True):
        model = st.session_state["pred_trained_models"][chosen_model]
        predicted_cost = model.predict(input_values)[0]

        if predicted_cost < 0:
            predicted_cost = 0

        st.metric(
            label="Predicted Medical Insurance Cost",
            value=f"${predicted_cost:,.2f}",
        )

        high_risk = int(smoker_input == "yes" and bmi >= 30)

        if high_risk == 1:
            st.warning(
                "This person is classified as a high-risk smoker because they smoke "
                "and have BMI ≥ 30."
            )
        else:
            st.info(
                "This prediction uses the same selected features, encoding, scaling, "
                "and trained model from the results above."
            )
