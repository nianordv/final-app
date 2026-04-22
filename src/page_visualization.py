"""
Page 2 — Data Visualization
=============================
Interactive charts exploring distributions, correlations,
and relationships in the dataset.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from data_loader import dataset_selector, get_target, get_features


def render():
    ds_key, df, info = dataset_selector()
    target = get_target(ds_key)
    features = get_features(df, target)

    st.markdown("## 📊 Data Visualization")
    st.caption("Explore the dataset through interactive charts to uncover patterns and insights.")
    st.markdown("---")

    # ── 1. Target distribution ──────────────────────────────────────
    st.markdown("### 🎯 Target Variable Distribution")
    col1, col2 = st.columns([2, 1])
    with col1:
        fig = px.histogram(
            df, x=target, nbins=50, color_discrete_sequence=["#57068C"],
            title=f"Distribution of {target}",
        )
        fig.update_layout(
            template="plotly_white",
            xaxis_title=info["target_desc"],
            yaxis_title="Count",
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("")
        st.markdown("")
        stats = df[target].describe()
        for stat_name in ["mean", "std", "min", "25%", "50%", "75%", "max"]:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{stat_name.upper()}</h3>
                <p>{stats[stat_name]:.2f}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── 2. Feature distributions ────────────────────────────────────
    st.markdown("### 📈 Feature Distributions")
    selected_features = st.multiselect(
        "Select features to visualize",
        features,
        default=features[:4],
    )

    if selected_features:
        n_cols = min(len(selected_features), 3)
        n_rows = (len(selected_features) + n_cols - 1) // n_cols
        fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=selected_features)

        colors = px.colors.sequential.Purples_r
        for i, feat in enumerate(selected_features):
            r, c = divmod(i, n_cols)
            fig.add_trace(
                go.Histogram(
                    x=df[feat], name=feat, nbinsx=30,
                    marker_color=colors[i % len(colors)],
                    showlegend=False,
                ),
                row=r + 1, col=c + 1,
            )
        fig.update_layout(height=300 * n_rows, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── 3. Correlation heatmap ──────────────────────────────────────
    st.markdown("### 🔥 Correlation Heatmap")
    corr = df.corr(numeric_only=True)
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdPu",
        aspect="auto",
        title="Pearson Correlation Matrix",
    )
    fig.update_layout(height=600, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # ── Top correlations with target ────────────────────────────────
    target_corr = corr[target].drop(target).abs().sort_values(ascending=False)
    st.markdown(f"**Top features correlated with `{target}`:**")
    for feat, val in target_corr.head(5).items():
        direction = "+" if corr.loc[feat, target] > 0 else "−"
        bar_width = int(val * 100)
        st.markdown(
            f"- **{feat}** → {direction}{val:.3f} "
            f'<span style="display:inline-block;height:10px;width:{bar_width}px;'
            f'background:#57068C;border-radius:4px;"></span>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── 4. Scatter plot explorer ────────────────────────────────────
    st.markdown("### 🔗 Feature vs Target Explorer")
    col_a, col_b = st.columns(2)
    with col_a:
        x_feat = st.selectbox("X-axis feature", features, index=0)
    with col_b:
        color_feat = st.selectbox(
            "Color by (optional)", ["None"] + features, index=0
        )

    color = color_feat if color_feat != "None" else None
    fig = px.scatter(
        df, x=x_feat, y=target, color=color,
        color_continuous_scale="Purples",
        opacity=0.5, title=f"{x_feat} vs {target}",
        trendline="ols",
    )
    fig.update_layout(template="plotly_white", height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── 5. Geographic map ───────────────────────────────────────────
    lat_col = next((c for c in df.columns if c.lower() in ("latitude", "lat")), None)
    lon_col = next((c for c in df.columns if c.lower() in ("longitude", "lon", "lng")), None)

    if lat_col and lon_col:
        st.markdown("### 🗺️ Geographic Map")
        st.caption("Each point is a district coloured by the selected variable.")

        map_col1, map_col2 = st.columns([1, 1])
        with map_col1:
            map_color = st.selectbox(
                "Colour points by",
                [target] + features,
                index=0,
                key="map_color",
            )
        with map_col2:
            map_size = st.selectbox(
                "Size points by",
                ["Uniform"] + [target] + features,
                index=0,
                key="map_size",
            )

        map_df = df.copy()
        size_col = None
        if map_size != "Uniform":
            size_col = map_size
            # Normalise to a nice visual range
            s = map_df[size_col]
            map_df["_size"] = ((s - s.min()) / (s.max() - s.min()) * 12 + 2)
            size_arg = "_size"
        else:
            map_df["_size"] = 4
            size_arg = "_size"

        fig = px.scatter_map(
            map_df,
            lat=lat_col,
            lon=lon_col,
            color=map_color,
            size=size_arg,
            color_continuous_scale="Purples",
            opacity=0.65,
            zoom=4.5,
            center={"lat": map_df[lat_col].mean(), "lon": map_df[lon_col].mean()},
            map_style="carto-positron",
            hover_data={
                lat_col: ":.2f",
                lon_col: ":.2f",
                map_color: ":.2f",
                "_size": False,
            },
            title=f"Map of {info['title']} — coloured by {map_color}",
        )
        fig.update_layout(
            height=620,
            template="plotly_white",
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown("### 🗺️ Geographic Map")
        st.info(
            "This dataset does not contain geographic coordinates (latitude / longitude). "
            "Switch to **California Housing** to explore the map view."
        )

    st.markdown("---")

    # ── 6. Box plots ────────────────────────────────────────────────
    st.markdown("### 📦 Box Plots — Outlier Detection")
    box_feats = st.multiselect("Features for box plots", features, default=features[:3], key="box")
    if box_feats:
        fig = go.Figure()
        for feat in box_feats:
            fig.add_trace(go.Box(y=df[feat], name=feat, marker_color="#57068C"))
        fig.update_layout(template="plotly_white", height=450, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)