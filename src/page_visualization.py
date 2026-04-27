"""
Page 2 — Data Visualization (Robust Insurance Edition)
=============================
Custom-tailored visuals for health insurance cost analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import importlib.util
from data_loader import dataset_selector, get_target, get_features

# Check if statsmodels is installed for trendlines
HAS_STATSMODELS = importlib.util.find_spec("statsmodels") is not None

def render():
    ds_key, df, info = dataset_selector()
    target = get_target(ds_key)
    features = get_features(df, target)
    
    # Identify column types
    numeric_cols = df[features].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df[features].select_dtypes(exclude=[np.number]).columns.tolist()

    st.markdown("## 🏥 Insurance Insights")
    st.caption("Investigating how lifestyle and physical metrics drive medical expenses.")
    
    # ── 1. Summary Metrics ──────────────────────────────────────────
    avg_charge = df[target].mean()
    smoker_avg = df[df['smoker'] == 'yes'][target].mean()
    non_smoker_avg = df[df['smoker'] == 'no'][target].mean()

    m1, m2, m3 = st.columns(3)
    m1.metric("Average Charge", f"${avg_charge:,.0f}")
    m2.metric("Smoker Avg", f"${smoker_avg:,.0f}", f"+${smoker_avg-non_smoker_avg:,.0f} vs non")
    m3.metric("Non-Smoker Avg", f"${non_smoker_avg:,.0f}")

    st.markdown("---")

    # ── 2. The Interaction: BMI & Smoking ─────────────────────────
    st.markdown("### ⚖️ The 'Double Whammy': BMI + Smoking")
    st.write(
        "As you noted, high BMI significantly increases charges **only** for smokers. "
        "Non-smokers (blue) stay in a relatively stable cost bracket regardless of BMI."
    )
    
    # Optional Trendline Logic
    trend_type = "ols" if HAS_STATSMODELS else None
    if not HAS_STATSMODELS:
        st.info("💡 Tip: To see trendlines, run `pip install statsmodels` in your terminal.")

    fig_scatter = px.scatter(
        df, x="bmi", y=target, color="smoker",
        size="age", 
        hover_data=['age', 'region', 'children'],
        opacity=0.6,
        color_discrete_map={"yes": "#EF553B", "no": "#636EFA"},
        trendline=trend_type,
        title="BMI vs Charges (Interaction with Smoking Status)"
    )
    fig_scatter.update_layout(template="plotly_white", legend_title="Smoker?")
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")

    # ── 3. Violin Plots: Seeing the Clusters ────────────────────────
    st.markdown("### 🎻 Cost Distributions")
    col_a, col_b = st.columns(2)
    
    with col_a:
        group_feat = st.selectbox("Group costs by:", categorical_cols, index=0) # Default Smoker
    with col_b:
        view_type = st.radio("View Type:", ["Violin", "Box"], horizontal=True)

    if view_type == "Violin":
        fig_dist = px.violin(df, y=target, x=group_feat, color=group_feat, box=True, points="outliers")
    else:
        fig_dist = px.box(df, y=target, x=group_feat, color=group_feat)
        
    fig_dist.update_layout(template="plotly_white", showlegend=False)
    st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown("---")

    # ── 4. Correlation & Age ───────────────────────────────────────
    st.markdown("### 📈 Other Factors")
    
    tab1, tab2 = st.tabs(["Age Progression", "Regional Heatmap"])
    
    with tab1:
        # Age often creates "bands" of costs
        fig_age = px.scatter(df, x="age", y=target, color="smoker", 
                            opacity=0.5, title="Charges by Age")
        fig_age.update_layout(template="plotly_white")
        st.plotly_chart(fig_age, use_container_width=True)
        
    with tab2:
        # Correlation matrix for numerical values
        corr = df.select_dtypes(include=[np.number]).corr()
        fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="Purples")
        st.plotly_chart(fig_corr, use_container_width=True)