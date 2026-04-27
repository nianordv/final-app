"""
Page 5 — Hyperparameter Tuning
================================
Automated hyperparameter optimization using Optuna,
with experiment tracking and visualization.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from data_loader import dataset_selector, get_target, get_features
from src import wandb_tracker


def render():
    ds_key, df, info = dataset_selector()
    target = get_target(ds_key)
    features = get_features(df, target)

    st.markdown("## ⚙️ Hyperparameter Tuning")
    st.caption(
        "Optimize model hyperparameters using Optuna and track all experiments. "
        "This replaces manual trial-and-error with automated Bayesian search."
    )
    st.markdown("---")

    # ── Data prep (Fixing the String-to-Float Error) ────────────────
    # 1. Handle missing values (simple approach)
    df_clean = df.copy()
    
    # 2. Convert categorical variables to numeric (One-Hot Encoding)
    # This turns strings like 'female'/'male' into 0s and 1s
    X_encoded = pd.get_dummies(df_clean[features], drop_first=True)
    y = df_clean[target].values
    
    # Use the encoded column names as our new feature list
    encoded_feature_names = X_encoded.columns.tolist()
    X = X_encoded.values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scaler is essential for MLP, Ridge, Lasso, and Elastic Net
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ── Config ──────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        model_name = st.selectbox(
            "Model to tune",
            ["🧠 MLP (Neural Network)", "Random Forest", "Gradient Boosting", "Ridge", "Lasso", "Elastic Net"],
        )
    with col2:
        n_trials = st.slider("Number of trials", 5, 100, 5, step=5)
    with col3:
        cv_folds = st.slider("CV folds", 3, 10, 5)

    # ── Hyperparameter search spaces ────────────────────────────────
    st.markdown("### 🔧 Search Space")
    search_spaces = {
        "🧠 MLP (Neural Network)": {
            "n_hidden_layers": "1 — 4",
            "neurons_per_layer": "16 — 256",
            "activation": "relu, tanh, logistic",
            "learning_rate_init": "0.0001 — 0.01",
            "alpha (L2 penalty)": "0.0001 — 0.1",
            "batch_size": "16 — 128",
            "max_iter": "200 — 1000",
        },
        "Random Forest": {
            "n_estimators": "50 — 500",
            "max_depth": "3 — 30",
            "min_samples_split": "2 — 20",
            "min_samples_leaf": "1 — 10",
            "max_features": "sqrt, log2, 0.5—1.0",
        },
        "Gradient Boosting": {
            "n_estimators": "50 — 500",
            "max_depth": "2 — 10",
            "learning_rate": "0.01 — 0.3",
            "subsample": "0.6 — 1.0",
            "min_samples_split": "2 — 20",
        },
        "Ridge": {"alpha": "0.001 — 100"},
        "Lasso": {"alpha": "0.001 — 100"},
        "Elastic Net": {"alpha": "0.001 — 100", "l1_ratio": "0.0 — 1.0"},
    }

    if model_name == "🧠 MLP (Neural Network)":
        st.markdown("### 🏗️ Neural Network Architecture Preview")
        st.markdown(
            "The MLP (Multi-Layer Perceptron) is a fully-connected feedforward "
            "neural network. Optuna will search over architecture and training parameters."
        )

    space_df = pd.DataFrame(
        [{"Parameter": k, "Range": v} for k, v in search_spaces[model_name].items()]
    )
    st.dataframe(space_df, use_container_width=True, hide_index=True)

    # ── W&B toggle ──────────────────────────────────────────────────
    track_wandb = st.checkbox(
        "📡 Log study to Weights & Biases",
        value=wandb_tracker.is_available(),
        disabled=not wandb_tracker.is_available(),
        help="Set WANDB_API_KEY in .env to enable.",
    )

    # ── Run optimization ────────────────────────────────────────────
    if st.button("🚀 Start Optimization", type="primary", use_container_width=True):
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            st.error("Install Optuna: `pip install optuna`")
            return

        wb_run = None
        if track_wandb:
            wb_run = wandb_tracker.init_run(
                run_name=f"{ds_key}-tune-{model_name}",
                config={
                    "dataset": ds_key,
                    "model": model_name,
                    "n_trials": n_trials,
                    "cv_folds": cv_folds,
                    "target": target,
                    "n_features": len(encoded_feature_names),
                },
                job_type="hparam-search",
            )

        def objective(trial):
            if model_name == "🧠 MLP (Neural Network)":
                n_layers = trial.suggest_int("n_hidden_layers", 1, 4)
                hidden_layers = tuple(
                    trial.suggest_int(f"neurons_layer_{i}", 16, 256, log=True)
                    for i in range(n_layers)
                )
                params = {
                    "hidden_layer_sizes": hidden_layers,
                    "activation": trial.suggest_categorical("activation", ["relu", "tanh", "logistic"]),
                    "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
                    "alpha": trial.suggest_float("alpha", 1e-4, 0.1, log=True),
                    "batch_size": trial.suggest_int("batch_size", 16, 128, log=True),
                    "max_iter": trial.suggest_int("max_iter", 200, 1000, step=100),
                    "random_state": 42,
                    "early_stopping": True,
                }
                model = MLPRegressor(**params)
                X_use, y_use = X_train_s, y_train
            elif model_name == "Random Forest":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                    "max_depth": trial.suggest_int("max_depth", 3, 30),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                    "random_state": 42, "n_jobs": -1,
                }
                model = RandomForestRegressor(**params)
                X_use, y_use = X_train, y_train
            elif model_name == "Gradient Boosting":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                    "max_depth": trial.suggest_int("max_depth", 2, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                    "random_state": 42,
                }
                model = GradientBoostingRegressor(**params)
                X_use, y_use = X_train, y_train
            elif model_name == "Ridge":
                alpha = trial.suggest_float("alpha", 0.001, 100, log=True)
                model = Ridge(alpha=alpha)
                X_use, y_use = X_train_s, y_train
            elif model_name == "Lasso":
                alpha = trial.suggest_float("alpha", 0.001, 100, log=True)
                model = Lasso(alpha=alpha)
                X_use, y_use = X_train_s, y_train
            else:
                alpha = trial.suggest_float("alpha", 0.001, 100, log=True)
                l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
                model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
                X_use, y_use = X_train_s, y_train

            scores = cross_val_score(model, X_use, y_use, cv=cv_folds, scoring="r2")
            return scores.mean()

        progress = st.progress(0, text="Optimizing...")
        live_log = st.empty()
        log_lines: list[str] = []
        study = optuna.create_study(direction="maximize", study_name=model_name)

        def callback(study, trial):
            progress.progress(
                (trial.number + 1) / n_trials,
                text=f"Trial {trial.number + 1}/{n_trials} — Best R²: {study.best_value:.4f}",
            )
            score = trial.value if trial.value is not None else float("nan")
            params_str = ", ".join(f"{k}={v}" for k, v in trial.params.items())
            log_lines.append(
                f"Trial {trial.number + 1:>3}/{n_trials} │ R²={score:.4f} │ best={study.best_value:.4f} │ {params_str}"
            )
            live_log.code("\n".join(log_lines[-10:]), language="text") # Show last 10 lines
            
            if track_wandb:
                wandb_tracker.log_metrics(wb_run, {
                    "trial/r2": score if score == score else 0.0,
                    "trial/best_r2": study.best_value,
                }, step=trial.number)

        study.optimize(objective, n_trials=n_trials, callbacks=[callback])
        progress.empty()

        # ── Evaluate best model ─────────────────────────────────────
        best_params = study.best_params.copy()
        if model_name == "🧠 MLP (Neural Network)":
            n_layers = best_params.pop("n_hidden_layers")
            hidden_layers = tuple(
                best_params.pop(f"neurons_layer_{i}") for i in range(n_layers)
            )
            for k in list(best_params.keys()):
                if k.startswith("neurons_layer_"):
                    best_params.pop(k)
            best_model = MLPRegressor(hidden_layer_sizes=hidden_layers, **best_params, random_state=42)
            best_model.fit(X_train_s, y_train)
            y_pred = best_model.predict(X_test_s)
            best_params["architecture"] = " → ".join(str(n) for n in hidden_layers)
        elif model_name == "Random Forest":
            best_model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
            best_model.fit(X_train, y_train)
            y_pred = best_model.predict(X_test)
        elif model_name == "Gradient Boosting":
            best_model = GradientBoostingRegressor(**best_params, random_state=42)
            best_model.fit(X_train, y_train)
            y_pred = best_model.predict(X_test)
        elif model_name in ["Ridge", "Lasso", "Elastic Net"]:
            best_model = globals()[model_name.replace(" ", "")](**best_params)
            best_model.fit(X_train_s, y_train)
            y_pred = best_model.predict(X_test_s)

        # Store results in session state
        st.session_state["tune_study"] = study
        st.session_state["tune_trials"] = pd.DataFrame([{"Trial": t.number, "R² (CV)": t.value, **t.params} for t in study.trials])
        st.session_state["tune_best_params"] = best_params
        st.session_state["tune_test_metrics"] = {
            "R²": r2_score(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        }
        st.session_state["tune_y_test"] = y_test
        st.session_state["tune_y_pred"] = y_pred
        st.session_state["tune_model_name"] = model_name
        st.session_state["tune_ready"] = True

        if wb_run is not None:
            wandb_tracker.finish_run(wb_run)

    # ── Display results ─────────────────────────────────────────────
    if st.session_state.get("tune_ready"):
        # (Rest of your visualization code remains the same as your original)
        trials_df = st.session_state["tune_trials"]
        best_params = st.session_state["tune_best_params"]
        test_metrics = st.session_state["tune_test_metrics"]
        y_test = st.session_state["tune_y_test"]
        y_pred = st.session_state["tune_y_pred"]
        tuned_model = st.session_state["tune_model_name"]

        st.markdown("### 🏆 Best Results")
        st.success(f"Best CV R²: {st.session_state['tune_study'].best_value:.4f}")
        
        m_cols = st.columns(3)
        for i, (metric, val) in enumerate(test_metrics.items()):
            m_cols[i].metric(f"Test {metric}", f"{val:.4f}")

        # [Insert your Plotly charts here as they were in the original]