
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from data_loader import dataset_selector, get_target, get_features
from src import wandb_tracker

ALL_MODELS = [
    "KNN (K-Nearest Neighbors)",
    "Random Forest",
    "Decision Tree",
    "Linear Regression",
]

# ── Model descriptions (shown in the UI) ─────────────────────────────────────
MODEL_DESCRIPTIONS = {
    "KNN (K-Nearest Neighbors)": (
        "Predicts by averaging the outcomes of the most similar rows in the training data. "
        "Simple and intuitive, but slows down on large datasets."
    ),
    "Random Forest": (
        "Builds hundreds of decision trees and averages their answers. "
        "Handles non-linear patterns well and is robust to outliers."
    ),
    "Decision Tree": (
        "Learns a series of yes/no rules to split the data. "
        "Easy to visualise but can overfit if allowed to grow too deep."
    ),
    "Linear Regression": (
        "Fits a straight line through the data. "
        "Fast and interpretable, but can only capture linear relationships."
    ),
}

# ── Hyperparameter descriptions (plain English tooltip text) ─────────────────
PARAM_DESCRIPTIONS = {
    "n_neighbors":       "How many nearby data points to consider when making a prediction. Higher = smoother but less precise.",
    "weights":           "Whether all neighbours count equally (uniform) or closer ones count more (distance).",
    "metric":            "How 'distance' between data points is measured.",
    "p":                 "Controls the shape of the Minkowski distance formula. 1 = Manhattan, 2 = Euclidean.",
    "leaf_size":         "Affects the speed of finding neighbours. Doesn't change accuracy much.",
    "n_estimators":      "Number of trees to build. More trees = more stable results, but slower. After ~200 the gains shrink.",
    "max_depth":         "How many levels deep each tree can grow. Shallow = simple rules. Too deep = memorises the training data.",
    "min_samples_split": "Minimum data points needed to split a node. Higher = simpler tree, less overfitting.",
    "min_samples_leaf":  "Minimum data points required in a leaf. Higher = smoother predictions.",
    "max_features":      "How many features each tree considers at each split. Adds randomness, which helps generalisation.",
    "criterion":         "The formula used to measure the quality of each split.",
    "splitter":          "Whether to pick the best possible split (best) or a random good split (random).",
    "alpha":             "Regularisation strength — how hard the model is penalised for complexity.",
    "l1_ratio":          "Balance between L1 and L2 regularisation.",
}

# ── Search spaces ─────────────────────────────────────────────────────────────
SEARCH_SPACES = {
    "KNN (K-Nearest Neighbors)": {
        "n_neighbors": "1 — 50",
        "weights":     "uniform, distance",
        "metric":      "euclidean, manhattan, minkowski",
        "p":           "1 — 5",
        "leaf_size":   "10 — 100",
    },
    "Random Forest": {
        "n_estimators":      "50 — 500",
        "max_depth":         "3 — 30",
        "min_samples_split": "2 — 20",
        "min_samples_leaf":  "1 — 10",
        "max_features":      "sqrt, log2",
    },
    "Decision Tree": {
        "max_depth":         "2 — 30",
        "min_samples_split": "2 — 20",
        "min_samples_leaf":  "1 — 20",
        "max_features":      "sqrt, log2, None",
        "criterion":         "squared_error, friedman_mse, absolute_error",
        "splitter":          "best, random",
    },
    "Linear Regression": {
        "(no hyperparameters)": "Linear Regression is deterministic — no search needed.",
    },
}

# ── Plain-English metric captions ─────────────────────────────────────────────
def _r2_caption(val: float, target: str) -> str:
    pct = round(val * 100, 1)
    if val >= 0.9:
        quality = "excellent fit"
    elif val >= 0.75:
        quality = "good fit"
    elif val >= 0.5:
        quality = "moderate fit"
    else:
        quality = "weak fit"
    return f"{pct}% of the variation in **{target}** is explained — {quality}."

def _mae_caption(val: float, target: str, y: np.ndarray) -> str:
    target_mean = float(np.mean(y))
    pct = round((val / target_mean) * 100, 1) if target_mean != 0 else 0
    cost_kw = {"charges", "price", "cost", "salary", "income", "revenue", "wage"}
    val_str = f"${val:,.0f}" if any(k in target.lower() for k in cost_kw) else f"{val:,.2f}"
    return f"On average, predictions are off by **{val_str}** ({pct}% of the mean {target})."

def _rmse_caption(val: float, target: str, y: np.ndarray) -> str:
    cost_kw = {"charges", "price", "cost", "salary", "income", "revenue", "wage"}
    val_str = f"${val:,.0f}" if any(k in target.lower() for k in cost_kw) else f"{val:,.2f}"
    return f"Penalises large errors more heavily. Typical big miss ≈ **{val_str}**."


# ── Main render ───────────────────────────────────────────────────────────────
def render():
    ds_key, df, info = dataset_selector()
    target   = get_target(ds_key)
    features = get_features(df, target)

    # ── Page header ──────────────────────────────────────────────────
    st.markdown("## ⚙️ Hyperparameter Tuning")
    st.markdown(
        "Find the best settings for each model automatically. "
        "Instead of guessing, **Optuna** tries many combinations and learns "
        "which settings work best — like a smart trial-and-error process."
    )
    st.info(
        f"**Dataset:** {ds_key}  ·  "
        f"**Target:** `{target}`  ·  "
        f"**Features:** {len(features)}  ·  "
        f"**Rows:** {len(df):,}",
        icon="📋",
    )
    st.markdown("---")

    # ── Data prep ────────────────────────────────────────────────────
    df_clean = df.copy()
    X_encoded = pd.get_dummies(df_clean[features], drop_first=True)
    y = df_clean[target].values
    encoded_feature_names = X_encoded.columns.tolist()
    X = X_encoded.values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── Step 1 — Choose a model ───────────────────────────────────────
    st.markdown("### Step 1 — Choose a model")

    model_name = st.selectbox(
        "Which model would you like to tune?",
        ALL_MODELS,
        help="Each model learns patterns differently. Try a few and compare results below.",
    )

    # Model description card
    st.markdown(
        f"""
        <div style="
            background: #f0f4ff;
            border-left: 4px solid #4a6cf7;
            border-radius: 0 8px 8px 0;
            padding: 10px 16px;
            margin: 8px 0 16px 0;
            font-size: 0.92rem;
            color: #1a1a2e;
        ">
        {MODEL_DESCRIPTIONS[model_name]}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Step 2 — Configure the search ────────────────────────────────
    st.markdown("### Step 2 — Configure the search")

    col_trials, col_folds = st.columns(2)
    with col_trials:
        n_trials = st.slider(
            "Number of trials",
            min_value=5, max_value=100, value=20, step=5,
            help=(
                "How many different settings combinations Optuna will try. "
                "More trials = more thorough search, but takes longer. "
                "20–40 is a good starting point."
            ),
        )
        st.caption(f"Optuna will test **{n_trials}** combinations and keep the best one.")

    with col_folds:
        cv_folds = st.slider(
            "Cross-validation folds",
            min_value=3, max_value=10, value=5,
            help=(
                "Each trial splits the training data into this many groups, "
                "trains on all but one, and tests on the last — repeated for each group. "
                "5 is standard."
            ),
        )
        st.caption(f"Each trial is evaluated across **{cv_folds}** data splits for reliability.")

    # ── Step 3 — Search space ─────────────────────────────────────────
    st.markdown("### Step 3 — What Optuna will search")

    if model_name == "Linear Regression":
        st.info(
            "Linear Regression has no hyperparameters to tune — "
            "it always fits the same way. A single cross-validated run will be performed.",
            icon="ℹ️",
        )
    else:
        space = SEARCH_SPACES[model_name]
        with st.expander("View search space", expanded=False):
            for param, rng in space.items():
                desc = PARAM_DESCRIPTIONS.get(param, "")
                st.markdown(
                    f"""
                    <div style="display:flex; justify-content:space-between; align-items:flex-start;
                                padding: 8px 0; border-bottom: 1px solid #e8e8e8;">
                        <div>
                            <code style="font-size:0.9rem">{param}</code>
                            {"<br><span style='font-size:0.8rem;color:#666'>" + desc + "</span>" if desc else ""}
                        </div>
                        <span style="font-size:0.85rem; color:#4a6cf7; white-space:nowrap; margin-left:16px">
                            {rng}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # ── W&B toggle ────────────────────────────────────────────────────
    with st.expander("Advanced options", expanded=False):
        track_wandb = st.checkbox(
            "Log study to Weights & Biases",
            value=wandb_tracker.is_available(),
            disabled=not wandb_tracker.is_available(),
            help="Set WANDB_API_KEY in .env to enable experiment tracking.",
        )
        if not wandb_tracker.is_available():
            st.caption("W&B not connected. Set `WANDB_API_KEY` in your `.env` file to enable.")

    # ── Run button with summary ───────────────────────────────────────
    st.markdown("---")
    st.markdown(
        f"**Ready to run:** {model_name} · {n_trials} trials · {cv_folds}-fold CV"
    )

    run_clicked = st.button(
        "🚀 Start Optimization",
        type="primary",
        use_container_width=True,
    )

    # ── Optimization logic ────────────────────────────────────────────
    if run_clicked:
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            st.error("Install Optuna first: `pip install optuna`")
            return

        wb_run = None
        if track_wandb:
            wb_run = wandb_tracker.init_run(
                run_name=f"{ds_key}-tune-{model_name}",
                config={
                    "dataset": ds_key, "model": model_name,
                    "n_trials": n_trials, "cv_folds": cv_folds,
                    "target": target, "n_features": len(encoded_feature_names),
                },
                job_type="hparam-search",
            )

        # Linear Regression — no Optuna needed
        if model_name == "Linear Regression":
            with st.spinner("Fitting Linear Regression…"):
                lr_model  = LinearRegression()
                cv_scores = cross_val_score(
                    lr_model, X_train_s, y_train, cv=cv_folds, scoring="r2"
                )
                best_cv_r2 = float(cv_scores.mean())
                lr_model.fit(X_train_s, y_train)
                y_pred = lr_model.predict(X_test_s)

            st.session_state["tune_study"]       = None
            st.session_state["tune_trials"]      = pd.DataFrame([{"Trial": 0, "R² (CV)": best_cv_r2}])
            st.session_state["tune_best_params"] = {}
            st.session_state["tune_best_cv_r2"]  = best_cv_r2

        else:
            # Optuna objective
            def objective(trial):
                if model_name == "KNN (K-Nearest Neighbors)":
                    params = {
                        "n_neighbors": trial.suggest_int("n_neighbors", 1, 50),
                        "weights":     trial.suggest_categorical("weights", ["uniform", "distance"]),
                        "metric":      trial.suggest_categorical("metric", ["euclidean", "manhattan", "minkowski"]),
                        "leaf_size":   trial.suggest_int("leaf_size", 10, 100),
                    }
                    if params["metric"] == "minkowski":
                        params["p"] = trial.suggest_int("p", 1, 5)
                    model  = KNeighborsRegressor(**params)
                    X_use, y_use = X_train_s, y_train

                elif model_name == "Random Forest":
                    params = {
                        "n_estimators":      trial.suggest_int("n_estimators", 50, 500),
                        "max_depth":         trial.suggest_int("max_depth", 3, 30),
                        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                        "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
                        "max_features":      trial.suggest_categorical("max_features", ["sqrt", "log2"]),
                        "random_state": 42, "n_jobs": -1,
                    }
                    model  = RandomForestRegressor(**params)
                    X_use, y_use = X_train, y_train

                else:  # Decision Tree
                    params = {
                        "max_depth":         trial.suggest_int("max_depth", 2, 30),
                        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                        "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 20),
                        "max_features":      trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                        "criterion":         trial.suggest_categorical(
                                                 "criterion",
                                                 ["squared_error", "friedman_mse", "absolute_error"],
                                             ),
                        "splitter":          trial.suggest_categorical("splitter", ["best", "random"]),
                        "random_state": 42,
                    }
                    model  = DecisionTreeRegressor(**params)
                    X_use, y_use = X_train, y_train

                return cross_val_score(model, X_use, y_use, cv=cv_folds, scoring="r2").mean()

            progress  = st.progress(0, text="Starting…")
            live_log  = st.empty()
            log_lines: list[str] = []
            study     = optuna.create_study(direction="maximize", study_name=model_name)

            def callback(study, trial):
                pct  = (trial.number + 1) / n_trials
                score = trial.value if trial.value is not None else float("nan")
                progress.progress(
                    pct,
                    text=f"Trial {trial.number + 1} of {n_trials} — best R² so far: {study.best_value:.4f}",
                )
                params_str = ", ".join(f"{k}={v}" for k, v in trial.params.items())
                log_lines.append(
                    f"Trial {trial.number + 1:>3}/{n_trials} │ R²={score:.4f} │ best={study.best_value:.4f} │ {params_str}"
                )
                live_log.code("\n".join(log_lines[-8:]), language="text")
                if track_wandb:
                    wandb_tracker.log_metrics(wb_run, {
                        "trial/r2": score if score == score else 0.0,
                        "trial/best_r2": study.best_value,
                    }, step=trial.number)

            study.optimize(objective, n_trials=n_trials, callbacks=[callback])
            progress.empty()
            live_log.empty()

            # Evaluate best model
            best_params = study.best_params.copy()
            if model_name == "KNN (K-Nearest Neighbors)":
                best_model = KNeighborsRegressor(**best_params)
                best_model.fit(X_train_s, y_train)
                y_pred = best_model.predict(X_test_s)
            elif model_name == "Random Forest":
                best_model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
                best_model.fit(X_train, y_train)
                y_pred = best_model.predict(X_test)
            else:
                best_model = DecisionTreeRegressor(**best_params, random_state=42)
                best_model.fit(X_train, y_train)
                y_pred = best_model.predict(X_test)

            st.session_state["tune_study"]       = study
            st.session_state["tune_trials"]      = pd.DataFrame(
                [{"Trial": t.number, "R² (CV)": t.value, **t.params} for t in study.trials]
            )
            st.session_state["tune_best_params"] = best_params
            st.session_state["tune_best_cv_r2"]  = study.best_value

        # Shared metrics
        test_r2   = r2_score(y_test, y_pred)
        test_mae  = mean_absolute_error(y_test, y_pred)
        test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

        st.session_state.update({
            "tune_test_metrics": {"R²": test_r2, "MAE": test_mae, "RMSE": test_rmse},
            "tune_y_test":       y_test,
            "tune_y_pred":       y_pred,
            "tune_model_name":   model_name,
            "tune_target":       target,
            "tune_y_full":       y,
            "tune_ready":        True,
        })

        # Update comparison table
        comparison = st.session_state.get("tune_comparison", {})
        comparison[model_name] = {
            "Best CV R²": round(st.session_state["tune_best_cv_r2"], 4),
            "Test R²":    round(test_r2, 4),
            "Test MAE":   round(test_mae, 4),
            "Test RMSE":  round(test_rmse, 4),
        }
        st.session_state["tune_comparison"] = comparison

        if wb_run is not None:
            wandb_tracker.finish_run(wb_run)

    # ─────────────────────────────────────────────────────────────────
    # Results section
    # ─────────────────────────────────────────────────────────────────
    if st.session_state.get("tune_ready"):
        test_metrics = st.session_state["tune_test_metrics"]
        best_cv_r2   = st.session_state["tune_best_cv_r2"]
        best_params  = st.session_state.get("tune_best_params", {})
        tuned_model  = st.session_state.get("tune_model_name", model_name)
        _target      = st.session_state.get("tune_target", target)
        _y_full      = st.session_state.get("tune_y_full", y)

        st.markdown("---")
        st.markdown("## Results")

        # ── CV R² hero card ──────────────────────────────────────────
        pct = round(best_cv_r2 * 100, 1)
        if best_cv_r2 >= 0.9:
            bar_color, label = "#1D9E75", "Excellent"
        elif best_cv_r2 >= 0.75:
            bar_color, label = "#4a6cf7", "Good"
        elif best_cv_r2 >= 0.5:
            bar_color, label = "#f5a623", "Moderate"
        else:
            bar_color, label = "#e05252", "Weak"

        st.markdown(
            f"""
            <div style="
                border: 1px solid #e0e0e0;
                border-radius: 12px;
                padding: 20px 24px;
                margin-bottom: 16px;
            ">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px">
                    <div>
                        <div style="font-size:0.8rem; color:#888; text-transform:uppercase; letter-spacing:0.05em">
                            Best CV R² — {tuned_model}
                        </div>
                        <div style="font-size:2rem; font-weight:600; color:{bar_color}; line-height:1.2">
                            {best_cv_r2:.4f}
                        </div>
                    </div>
                    <span style="
                        background:{bar_color}18; color:{bar_color};
                        border:1px solid {bar_color}44;
                        border-radius:20px; padding:4px 14px; font-size:0.85rem; font-weight:500
                    ">{label}</span>
                </div>
                <div style="background:#f0f0f0; border-radius:6px; height:8px; overflow:hidden">
                    <div style="width:{pct}%; background:{bar_color}; height:100%; border-radius:6px;
                                transition:width 0.8s ease"></div>
                </div>
                <div style="font-size:0.85rem; color:#555; margin-top:10px">
                    {_r2_caption(best_cv_r2, _target)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Best params pill row ──────────────────────────────────────
        if best_params:
            pills_html = " ".join(
                f'<span style="background:#f0f4ff; color:#4a6cf7; border:1px solid #c7d4ff; '
                f'border-radius:20px; padding:3px 10px; font-size:0.8rem; '
                f'font-family:monospace; white-space:nowrap">'
                f'{k} = {v}</span>'
                for k, v in best_params.items()
            )
            st.markdown(
                f'<div style="margin-bottom:20px"><div style="font-size:0.8rem;color:#888;'
                f'margin-bottom:6px">Best hyperparameters found</div>'
                f'<div style="display:flex;flex-wrap:wrap;gap:6px">{pills_html}</div></div>',
                unsafe_allow_html=True,
            )

        # ── Three test-set metric cards ───────────────────────────────
        st.markdown("#### How does it perform on unseen data?")
        st.caption(
            "These metrics are measured on the 20% test set — rows the model "
            "never saw during training or tuning."
        )

        col_r2, col_mae, col_rmse = st.columns(3)
        r2_val, mae_val, rmse_val = (
            test_metrics["R²"], test_metrics["MAE"], test_metrics["RMSE"]
        )

        with col_r2:
            st.metric("Test R²", f"{r2_val:.4f}", help="How much of the target variation is explained.")
            st.caption(_r2_caption(r2_val, _target))

        with col_mae:
            cost_kw = {"charges", "price", "cost", "salary", "income", "revenue", "wage"}
            mae_display = (
                f"${mae_val:,.0f}" if any(k in _target.lower() for k in cost_kw)
                else f"{mae_val:,.2f}"
            )
            st.metric("Test MAE", mae_display, help="Average absolute prediction error.")
            st.caption(_mae_caption(mae_val, _target, _y_full))

        with col_rmse:
            rmse_display = (
                f"${rmse_val:,.0f}" if any(k in _target.lower() for k in cost_kw)
                else f"{rmse_val:,.2f}"
            )
            st.metric("Test RMSE", rmse_display, help="Like MAE but penalises large errors more.")
            st.caption(_rmse_caption(rmse_val, _target, _y_full))

        # [Insert your Plotly charts here]

    # ─────────────────────────────────────────────────────────────────
    # Model comparison section
    # ─────────────────────────────────────────────────────────────────
    comparison: dict = st.session_state.get("tune_comparison", {})
    if comparison:
        st.markdown("---")

        n_done  = len(comparison)
        n_total = len(ALL_MODELS)

        st.markdown("## Model Comparison")
        st.caption(
            f"{n_done} of {n_total} models run. "
            "Each row shows the best result found for that model. "
            "Run all four to get the full picture."
        )

        # Progress bar showing how many models done
        st.progress(n_done / n_total, text=f"{n_done}/{n_total} models completed")

        comp_df = (
            pd.DataFrame.from_dict(comparison, orient="index")
            .reset_index()
            .rename(columns={"index": "Model"})
            .sort_values("Best CV R²", ascending=False)
            .reset_index(drop=True)
        )
        comp_df.insert(0, "Rank", [f"#{i+1}" for i in range(len(comp_df))])

        best_model_name = comp_df.iloc[0]["Model"]

        def highlight_best(row):
            if row["Model"] == best_model_name:
                return ["background-color: #e8f5e9; font-weight: bold"] * len(row)
            return [""] * len(row)

        styled = (
            comp_df.style
            .apply(highlight_best, axis=1)
            .format({
                "Best CV R²": "{:.4f}",
                "Test R²":    "{:.4f}",
                "Test MAE":   "{:.4f}",
                "Test RMSE":  "{:.4f}",
            })
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)

        st.caption(
            f"**Winner so far: {best_model_name}** with CV R² = "
            f"{comp_df.iloc[0]['Best CV R²']:.4f}. "
            "Higher R² = better. Highlighted row is the current best."
        )

        # Bar chart
        fig = px.bar(
            comp_df,
            x="Model",
            y="Best CV R²",
            color="Best CV R²",
            color_continuous_scale="Blues",
            text=comp_df["Best CV R²"].map("{:.4f}".format),
            title="Best CV R² by model — higher is better",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            coloraxis_showscale=False,
            yaxis=dict(range=[0, min(1.05, comp_df["Best CV R²"].max() + 0.12)]),
            plot_bgcolor="white",
            paper_bgcolor="white",
            height=360,
            margin=dict(t=48, b=24),
            font=dict(size=13),
        )
        st.plotly_chart(fig, use_container_width=True)

        col_reset, _ = st.columns([1, 4])
        with col_reset:
            if st.button("🗑️ Reset all results"):
                for key in ["tune_comparison", "tune_ready", "tune_study",
                            "tune_trials", "tune_best_params", "tune_best_cv_r2",
                            "tune_test_metrics", "tune_y_test", "tune_y_pred",
                            "tune_model_name", "tune_target", "tune_y_full"]:
                    st.session_state.pop(key, None)
                st.rerun()
