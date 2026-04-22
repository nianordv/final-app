"""
Precompute feature importances for the Explainability page.

Trains Random Forest and Gradient Boosting on each dataset, computes
built-in, permutation, and SHAP importances, and pickles the results
to `cache/importance_<dataset>_<model>.pkl` so the Streamlit page
can render them instantly.

Run once:  python precompute_importance.py
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from data_loader import DATASETS, get_features, get_target, load_data

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

MODELS = {
    "Random Forest": lambda: RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting": lambda: GradientBoostingRegressor(n_estimators=100, random_state=42),
}


def cache_path(dataset_key: str, model_name: str) -> Path:
    safe_model = model_name.lower().replace(" ", "_")
    return CACHE_DIR / f"importance_{dataset_key}_{safe_model}.pkl"


def compute_for(dataset_key: str, model_name: str) -> dict:
    df = load_data(dataset_key)
    target = get_target(dataset_key)
    features = get_features(df, target)
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = MODELS[model_name]()
    model.fit(X_train, y_train)

    imp_df = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=True)

    perm_result = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
    )
    perm_df = pd.DataFrame({
        "Feature": features,
        "Importance": perm_result.importances_mean,
        "Std": perm_result.importances_std,
    }).sort_values("Importance", ascending=True)

    payload = {
        "features": features,
        "imp_df": imp_df,
        "perm_df": perm_df,
        "X_test": X_test.reset_index(drop=True),
    }

    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        payload["shap_values"] = np.asarray(shap_values)
        payload["shap_df"] = pd.DataFrame({
            "Feature": features,
            "Mean |SHAP|": np.abs(shap_values).mean(axis=0),
        }).sort_values("Mean |SHAP|", ascending=True)
    except ImportError:
        print("  (shap not installed — skipping SHAP)")

    return payload


def main() -> None:
    print(f"Cache directory: {CACHE_DIR}")
    for ds_key in DATASETS.values():
        for model_name in MODELS:
            t0 = time.time()
            print(f"→ {ds_key:10s} · {model_name}")
            payload = compute_for(ds_key, model_name)
            out = cache_path(ds_key, model_name)
            with out.open("wb") as fh:
                pickle.dump(payload, fh)
            print(f"  saved {out.name} ({time.time() - t0:.1f}s)")
    print("Done.")


if __name__ == "__main__":
    main()