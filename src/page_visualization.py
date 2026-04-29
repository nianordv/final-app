"""
Page 2 — Data Visualization (Robust Insurance Edition)
=============================
Optimized for encoded data loader and interaction features.
"""


import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import importlib.util
from data_loader import dataset_selector, get_target, get_features


# Check if statsmodels is installed for trendlines
HAS_STATSMODELS = importlib.util.find_spec("statsmodels") is not None


def render():
   ds_key, df, info = dataset_selector()
   target = get_target(ds_key)
   features = get_features(df, target)
  
   # NEW LOGIC: Identify column types for encoded data
   # Since everything is now numeric, we distinguish by cardinality (unique values)
   binary_cols = [col for col in features if df[col].nunique() == 2]
   continuous_cols = [col for col in features if df[col].nunique() > 10]


   st.markdown("## 🏥 Insurance Insights")
   st.caption("Investigating how lifestyle and physical metrics drive medical expenses.")
  
   # ── 1. Summary Metrics ──────────────────────────────────────────
   # FIX: smoker is now 1 (yes) and 0 (no)
   avg_charge = df[target].mean()
   smoker_avg = df[df['smoker'] == 1][target].mean()
   non_smoker_avg = df[df['smoker'] == 0][target].mean()


   m1, m2, m3 = st.columns(3)
   m1.metric("Average Charge", f"${avg_charge:,.0f}")
   m2.metric("Smoker Avg", f"${smoker_avg:,.0f}", f"+${smoker_avg-non_smoker_avg:,.0f} vs non")
   m3.metric("Non-Smoker Avg", f"${non_smoker_avg:,.0f}")


   st.markdown("---")


   # ── 2. The Interaction: BMI & Smoking ─────────────────────────
   st.markdown("### ⚖️ The 'Double Whammy': BMI + Smoking")
   st.write("High BMI significantly increases charges **only** for smokers.")
  
   trend_type = "ols" if HAS_STATSMODELS else None


   # FIX: Removed 'region' from hover (it no longer exists as a single column)
   # Added your new interaction features to hover instead
   fig_scatter = px.scatter(
       df, x="bmi", y=target,
       color="smoker",
       size="age",
       hover_data=['smoker_bmi', 'high_risk_smoker', 'children'],
       opacity=0.6,
       color_discrete_map={1: "#EF553B", 0: "#636EFA"}, # FIX: Use 1 and 0
       trendline=trend_type,
       title="BMI vs Charges (Interaction with Smoking Status)"
   )
   fig_scatter.update_layout(template="plotly_white", legend_title="Smoker (1=Yes)")
   st.plotly_chart(fig_scatter, use_container_width=True)


   st.markdown("---")


   # ── 3. Violin Plots: Cost Distributions ────────────────────────
   st.markdown("### 🎻 Cost Distributions")
   col_a, col_b = st.columns(2)
  
   with col_a:
       group_feat = st.selectbox("Group costs by:", binary_cols, index=0)
   with col_b:
       view_type = st.radio("View Type:", ["Violin", "Box"], horizontal=True)


   fig_dist = px.violin(df, y=target, x=group_feat, color=group_feat, box=True, points="outliers") if view_type == "Violin" else px.box(df, y=target, x=group_feat, color=group_feat)
   fig_dist.update_layout(template="plotly_white", showlegend=False)
   st.plotly_chart(fig_dist, use_container_width=True)


   # ── 4. Regional Impact (Handling One-Hot Encoded Regions) ───────
   st.markdown("---")
   st.markdown("### 🗺️ Regional Analysis")
   st.write("Since data is one-hot encoded, we compare individual regional indicators against charges.")


   # Reconstruct a temporary 'region' column for visualization purposes only
   region_cols = [c for c in df.columns if "region_" in c]
   if region_cols:
       # We find which region column is 1 for each row
       temp_df = df.copy()
       temp_df['region_name'] = temp_df[region_cols].idxmax(axis=1).str.replace('region_', '')
      
       fig_reg = px.bar(
           temp_df.groupby('region_name')[target].mean().reset_index(),
           x='region_name', y=target, color='region_name',
           title="Mean Charges by Region",
           labels={target: "Avg Charges ($)", "region_name": "Region"}
       )
       st.plotly_chart(fig_reg, use_container_width=True)


   # ── 5. Age & Correlation ───────────────────────────────────────
   st.markdown("---")
   st.markdown("### 📈 Multivariate Analysis")
  
   tab1, tab2 = st.tabs(["Age Progression", "Correlation Heatmap"])
  
   with tab1:
       # Show faceted age scatter by children
       fig_facet = px.scatter(
           df, x="age", y=target, color="smoker",
           facet_col="children", facet_col_wrap=3,
           opacity=0.6,
           color_discrete_map={1: "#EF553B", 0: "#636EFA"},
           title="Age vs. Charges (Segmented by Number of Children)"
       )
       fig_facet.for_each_annotation(lambda a: a.update(text=f"{a.text.split('=')[-1]} Children"))
       st.plotly_chart(fig_facet, use_container_width=True)
      
   with tab2:
       # Correlation heatmap including new features like smoker_bmi
       corr = df.corr()
       fig_corr = px.imshow(
           corr, text_auto=".2f",
           color_continuous_scale="Purples",
           title="Feature Correlation Matrix (Including Engineered Features)"
       )
       st.plotly_chart(fig_corr, use_container_width=True)


   # ── 6. Family Size ──────────────────────────────────────────
   st.markdown("---")
   st.markdown("### 🧒 Family Size vs. Financial Impact")
   child_corr = df['children'].corr(df[target])
   st.info(f"**Correlation Note:** The correlation between children and charges is **{child_corr:.2f}**.")


   fig_children = px.box(
       df.sort_values("children"), x="children", y=target,
       color="children", points="outliers", notched=True,
       color_discrete_sequence=px.colors.sequential.Aggrnyl,
       title="Distribution of Charges by Number of Children"
   )
   st.plotly_chart(fig_children, use_container_width=True)
