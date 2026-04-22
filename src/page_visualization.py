import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from data_loader import dataset_selector, get_target, get_features

def render():
    ds_key, df, info = dataset_selector()
    target = get_target(ds_key)
    
    # Tactical Colors
    CT_BLUE = "#67c1f5"
    T_ORANGE = "#de9b35"
    DARK_BG = "#0b1119"
    PANEL_BG = "#1b2838"

    st.markdown(f"## <span style='color:{CT_BLUE}'>📊</span> Tactical Data Intel", unsafe_allow_html=True)
    st.caption("Analyzing match dynamics and economic advantages.")

    # ── 1. CS:GO SPECIFIC: MAP PERFORMANCE (For Results Dataset) ──
    if ds_key == "results":
        st.markdown("### 🗺️ Win Distribution by Map")
        # Calculate win counts per map
        map_stats = df.groupby(['_map', 'map_winner']).size().unstack().fillna(0)
        map_stats.columns = ['Team 1 Wins', 'Team 2 Wins']
        
        fig = px.bar(
            map_stats, 
            barmode="group",
            color_discrete_map={'Team 1 Wins': CT_BLUE, 'Team 2 Wins': T_ORANGE},
            template="plotly_dark"
        )
        fig.update_layout(paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG)
        st.plotly_chart(fig, use_container_width=True)

    # ── 2. CS:GO SPECIFIC: ECONOMY CURVE (For Economy Dataset) ──
    elif ds_key == "economy":
        st.markdown("### 💰 Average Equipment Value per Round")
        # Extract columns like 1_t1, 2_t1... and calculate means
        t1_cols = [f"{i}_t1" for i in range(1, 16)]
        t2_cols = [f"{i}_t2" for i in range(1, 16)]
        
        avg_val = pd.DataFrame({
            'Round': range(1, 16),
            'Team 1 (Avg Value)': df[t1_cols].mean().values,
            'Team 2 (Avg Value)': df[t2_cols].mean().values
        })
        
        fig = px.line(
            avg_val, x='Round', y=['Team 1 (Avg Value)', 'Team 2 (Avg Value)'],
            color_discrete_map={'Team 1 (Avg Value)': CT_BLUE, 'Team 2 (Avg Value)': T_ORANGE},
            template="plotly_dark"
        )
        fig.update_layout(paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── 3. TARGET DISTRIBUTION ──
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### 🎯 Outcome Frequency")
        fig = px.histogram(
            df, x=target, 
            color_discrete_sequence=[CT_BLUE],
            template="plotly_dark"
        )
        fig.update_layout(paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Stats Log")
        # Displaying target counts as tactical metrics
        counts = df[target].value_counts()
        for val, count in counts.items():
            st.markdown(f"""
            <div class="metric-card">
                <h3>WINNER: {val}</h3>
                <p>{count} Matches</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── 4. CORRELATION (CLEANED) ──
    st.markdown("### 🔥 Strategic Correlation")
    st.caption("Only showing high-impact features to avoid clutter.")
    
    # Calculate correlation and take top 10 most relevant to the target
    full_corr = df.corr(numeric_only=True)
    top_corr_features = full_corr[target].abs().sort_values(ascending=False).head(12).index
    clean_corr = df[top_corr_features].corr()

    fig = px.imshow(
        clean_corr,
        color_continuous_scale=[[0, DARK_BG], [0.5, PANEL_BG], [1, CT_BLUE]],
        text_auto=".2f",
        template="plotly_dark"
    )
    fig.update_layout(paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG)
    st.plotly_chart(fig, use_container_width=True)

    # ── 5. BOX PLOTS (Tactical Outlier Detection) ──
    st.markdown("### 📦 Score/Rank Spread")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Filter out ID columns
    display_cols = [c for c in numeric_cols if "id" not in c.lower()][:5]
    
    selected_box = st.multiselect("Select features for spread analysis", numeric_cols, default=display_cols)
    
    if selected_box:
        fig = go.Figure()
        for col in selected_box:
            fig.add_trace(go.Box(y=df[col], name=col, marker_color=CT_BLUE))
        fig.update_layout(template="plotly_dark", paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG)
        st.plotly_chart(fig, use_container_width=True)