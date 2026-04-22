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

    # ── Data prep ───────────────────────────────────────────────────
    X = df[features].values
    y = df[target].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
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

    # ── MLP architecture visualizer ─────────────────────────────────
    if model_name == "🧠 MLP (Neural Network)":
        st.markdown("### 🏗️ Neural Network Architecture Preview")
        st.markdown(
            "The MLP (Multi-Layer Perceptron) is a fully-connected feedforward "
            "neural network. Optuna will search over the number of hidden layers, "
            "neurons per layer, activation function, learning rate, and regularization."
        )
        st.markdown(
            "```\n"
            "Input Layer ──▶ Hidden Layer(s) ──▶ Output Layer\n"
            "  (features)    (relu/tanh/logistic)   (prediction)\n"
            "```"
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
                    "n_features": len(features),
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
                    "validation_fraction": 0.1,
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
            live_log.code("\n".join(log_lines), language="text")
            wandb_tracker.log_metrics(wb_run, {
                "trial/r2": score if score == score else 0.0,
                "trial/best_r2": study.best_value,
            }, step=trial.number)

        study.optimize(objective, n_trials=n_trials, callbacks=[callback])
        progress.empty()

        # ── Evaluate best model on test set ─────────────────────────
        best_params = study.best_params
        if model_name == "🧠 MLP (Neural Network)":
            # Reconstruct hidden_layer_sizes from individual layer params
            n_layers = best_params.pop("n_hidden_layers")
            hidden_layers = tuple(
                best_params.pop(f"neurons_layer_{i}") for i in range(n_layers)
            )
            # Remove any extra layer keys from unused layers
            for k in list(best_params.keys()):
                if k.startswith("neurons_layer_"):
                    best_params.pop(k)
            best_model = MLPRegressor(
                hidden_layer_sizes=hidden_layers,
                **best_params, random_state=42,
                early_stopping=True, validation_fraction=0.1,
            )
            best_model.fit(X_train_s, y_train)
            y_pred = best_model.predict(X_test_s)
            # Re-add for display
            best_params["n_hidden_layers"] = n_layers
            best_params["architecture"] = " → ".join(str(n) for n in hidden_layers)
        elif model_name == "Random Forest":
            best_model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
            best_model.fit(X_train, y_train)
            y_pred = best_model.predict(X_test)
        elif model_name == "Gradient Boosting":
            best_model = GradientBoostingRegressor(**best_params, random_state=42)
            best_model.fit(X_train, y_train)
            y_pred = best_model.predict(X_test)
        elif model_name == "Ridge":
            best_model = Ridge(**best_params)
            best_model.fit(X_train_s, y_train)
            y_pred = best_model.predict(X_test_s)
        elif model_name == "Lasso":
            best_model = Lasso(**best_params)
            best_model.fit(X_train_s, y_train)
            y_pred = best_model.predict(X_test_s)
        else:
            best_model = ElasticNet(**best_params)
            best_model.fit(X_train_s, y_train)
            y_pred = best_model.predict(X_test_s)

        # Store results
        trials_data = []
        for t in study.trials:
            row = {"Trial": t.number, "R² (CV)": t.value}
            row.update(t.params)
            trials_data.append(row)

        st.session_state["tune_study"] = study
        st.session_state["tune_trials"] = pd.DataFrame(trials_data)
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
            wandb_tracker.log_metrics(wb_run, {
                "final/best_cv_r2": study.best_value,
                "final/test_r2": st.session_state["tune_test_metrics"]["R²"],
                "final/test_mae": st.session_state["tune_test_metrics"]["MAE"],
                "final/test_rmse": st.session_state["tune_test_metrics"]["RMSE"],
            })
            try:
                wb_run.summary["best_params"] = {
                    k: v for k, v in best_params.items() if isinstance(v, (int, float, str))
                }
            except Exception:
                pass
            wandb_tracker.finish_run(wb_run)

    # ── Display results ─────────────────────────────────────────────
    if not st.session_state.get("tune_ready"):
        st.info("Click **Start Optimization** to begin hyperparameter search.")
        return

    trials_df = st.session_state["tune_trials"]
    best_params = st.session_state["tune_best_params"]
    test_metrics = st.session_state["tune_test_metrics"]
    y_test = st.session_state["tune_y_test"]
    y_pred = st.session_state["tune_y_pred"]
    tuned_model = st.session_state["tune_model_name"]

    st.markdown("---")

    # ── Best parameters ─────────────────────────────────────────────
    st.markdown("### 🏆 Best Hyperparameters")
    st.success(f"**{tuned_model}** — Best CV R²: {st.session_state['tune_study'].best_value:.4f}")

    param_cols = st.columns(len(best_params))
    for i, (k, v) in enumerate(best_params.items()):
        with param_cols[i]:
            display_val = f"{v:.4f}" if isinstance(v, float) else str(v)
            st.metric(k, display_val)

    # ── Test set performance ────────────────────────────────────────
    st.markdown("### 📈 Test Set Performance (Best Model)")
    m_cols = st.columns(3)
    for i, (metric, val) in enumerate(test_metrics.items()):
        m_cols[i].metric(metric, f"{val:.4f}")

    st.markdown("---")

    # ── Optimization history ────────────────────────────────────────
    st.markdown("### 📉 Optimization History")
    col_h1, col_h2 = st.columns(2)

    with col_h1:
        best_so_far = trials_df["R² (CV)"].cummax()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=trials_df["Trial"], y=trials_df["R² (CV)"],
            mode="markers", name="Trial score",
            marker=dict(color="#b347ff", size=6, opacity=0.5),
        ))
        fig.add_trace(go.Scatter(
            x=trials_df["Trial"], y=best_so_far,
            mode="lines", name="Best so far",
            line=dict(color="#57068C", width=3),
        ))
        fig.update_layout(
            template="plotly_white", height=400,
            title="Optimization Progress",
            xaxis_title="Trial", yaxis_title="R² (CV)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_h2:
        fig = px.scatter(
            x=y_test, y=y_pred, opacity=0.4,
            color_discrete_sequence=["#57068C"],
            title=f"Best {tuned_model} — Actual vs Predicted",
            labels={"x": "Actual", "y": "Predicted"},
        )
        mn = min(y_test.min(), y_pred.min())
        mx = max(y_test.max(), y_pred.max())
        fig.add_trace(go.Scatter(
            x=[mn, mx], y=[mn, mx],
            mode="lines", line=dict(color="red", dash="dash"),
            showlegend=False,
        ))
        fig.update_layout(template="plotly_white", height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Parallel coordinates ────────────────────────────────────────
    st.markdown("### 🔀 Hyperparameter Exploration")
    param_names = [c for c in trials_df.columns if c not in ("Trial", "R² (CV)")]
    if len(param_names) >= 2:
        dims = [dict(label="R² (CV)", values=trials_df["R² (CV)"])]
        for p in param_names:
            dims.append(dict(label=p, values=trials_df[p]))
        fig = go.Figure(go.Parcoords(
            line=dict(
                color=trials_df["R² (CV)"],
                colorscale="Purples",
                showscale=True,
                cmin=trials_df["R² (CV)"].min(),
                cmax=trials_df["R² (CV)"].max(),
            ),
            dimensions=dims,
        ))
        fig.update_layout(height=500, title="Parallel Coordinates — All Trials")
        st.plotly_chart(fig, use_container_width=True)

    # ── Experiment log ──────────────────────────────────────────────
    st.markdown("### 📋 Full Experiment Log")
    st.dataframe(
        trials_df.sort_values("R² (CV)", ascending=False).style.format(
            {c: "{:.4f}" for c in trials_df.select_dtypes(include="float").columns}
        ),
        use_container_width=True,
        height=400,
    )