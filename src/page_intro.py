"""
Page 1 — Business Case & Data Presentation
============================================
Presents the problem statement, dataset overview,
descriptive statistics, data quality checks,
and a rich interactive HTML data exploration report.
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from data_loader import dataset_selector, get_target, get_features


# ── Helper: small sparkline SVG ─────────────────────────────────────
def _sparkline_svg(values, width=120, height=32, color="#57068C"):
    """Return an inline SVG sparkline for a numeric series."""
    vals = pd.to_numeric(values.dropna(), errors="coerce").dropna()
    if len(vals) < 2:
        return ""
    mn, mx = vals.min(), vals.max()
    rng = mx - mn if mx != mn else 1
    n = min(len(vals), 80)
    sampled = vals.iloc[:: max(1, len(vals) // n)]
    points = []
    for i, v in enumerate(sampled):
        x = round(i / (len(sampled) - 1) * width, 2)
        y = round(height - (v - mn) / rng * (height - 4) - 2, 2)
        points.append(f"{x},{y}")
    poly = " ".join(points)
    return (
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
        f'<polyline points="{poly}" fill="none" stroke="{color}" stroke-width="1.5" '
        f'stroke-linecap="round" stroke-linejoin="round"/></svg>'
    )


# ── Helper: histogram SVG ──────────────────────────────────────────
def _histogram_svg(values, bins=20, width=220, height=60, color="#57068C"):
    """Return an inline SVG histogram bar chart."""
    vals = pd.to_numeric(values.dropna(), errors="coerce").dropna()
    if len(vals) < 2:
        return "<span style='color:#999'>—</span>"
    counts, edges = np.histogram(vals, bins=bins)
    mx = counts.max() if counts.max() > 0 else 1
    bar_w = width / len(counts)
    bars = []
    for i, c in enumerate(counts):
        bh = max(1, c / mx * (height - 4))
        x = round(i * bar_w, 2)
        y = round(height - bh, 2)
        opacity = 0.5 + 0.5 * (c / mx)
        bars.append(
            f'<rect x="{x}" y="{y}" width="{round(bar_w - 1, 2)}" '
            f'height="{round(bh, 2)}" rx="1" fill="{color}" opacity="{opacity:.2f}"/>'
        )
    return (
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
        + "".join(bars)
        + "</svg>"
    )


# ── Helper: correlation matrix as HTML table ────────────────────────
def _correlation_html(df, features, target):
    """Build a styled HTML correlation matrix with colour-coded cells."""
    cols = features + [target]
    cols = [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    corr = df[cols].corr()

    def _cell_color(v):
        if v >= 0:
            intensity = int(min(abs(v) * 255, 255))
            return f"rgba(87, 6, 140, {abs(v):.2f})"
        else:
            intensity = int(min(abs(v) * 255, 255))
            return f"rgba(220, 50, 50, {abs(v):.2f})"

    header = "".join(
        f'<th style="padding:6px 8px;font-size:0.7rem;writing-mode:vertical-lr;'
        f'transform:rotate(180deg);color:#57068C;font-weight:600;white-space:nowrap;">{c}</th>'
        for c in corr.columns
    )
    rows_html = ""
    for row_name in corr.index:
        cells = f'<td style="padding:6px 8px;font-weight:600;font-size:0.75rem;color:#57068C;white-space:nowrap;">{row_name}</td>'
        for col_name in corr.columns:
            v = corr.loc[row_name, col_name]
            bg = _cell_color(v)
            text_color = "white" if abs(v) > 0.5 else "#333"
            cells += (
                f'<td style="padding:4px 6px;background:{bg};color:{text_color};'
                f'text-align:center;font-size:0.7rem;border-radius:3px;min-width:42px;">'
                f'{v:.2f}</td>'
            )
        rows_html += f"<tr>{cells}</tr>"

    return (
        '<div style="overflow-x:auto;">'
        '<table style="border-collapse:separate;border-spacing:2px;">'
        f"<tr><th></th>{header}</tr>{rows_html}"
        "</table></div>"
    )


# ── Helper: outlier detection summary ──────────────────────────────
def _outlier_summary(df, features):
    """Return a list of (feature, n_outliers, pct) using IQR method."""
    results = []
    for col in features:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((df[col] < lower) | (df[col] > upper)).sum())
        pct = n_out / len(df) * 100
        results.append((col, n_out, pct))
    return results


# ── Main report builder ────────────────────────────────────────────
def _build_data_report(df, features, target, info):
    """Build a complete HTML data exploration report."""

    n_rows, n_cols = df.shape
    mem_kb = df.memory_usage(deep=True).sum() / 1024
    missing_total = df.isnull().sum().sum()
    missing_pct = (missing_total / (n_rows * n_cols)) * 100
    duplicates = df.duplicated().sum()
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    # ── Section 1: Overview cards ──────────────────────────────────
    overview_cards = f"""
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:24px;">
        <div class="report-card">
            <div class="report-card-label">Observations</div>
            <div class="report-card-value">{n_rows:,}</div>
        </div>
        <div class="report-card">
            <div class="report-card-label">Variables</div>
            <div class="report-card-value">{n_cols}</div>
        </div>
        <div class="report-card">
            <div class="report-card-label">Missing Cells</div>
            <div class="report-card-value" style="color:{'#2ea043' if missing_total == 0 else '#d73a49'};">{missing_total:,} <span style="font-size:0.75rem;">({missing_pct:.1f}%)</span></div>
        </div>
        <div class="report-card">
            <div class="report-card-label">Duplicate Rows</div>
            <div class="report-card-value" style="color:{'#2ea043' if duplicates == 0 else '#d73a49'};">{duplicates:,}</div>
        </div>
        <div class="report-card">
            <div class="report-card-label">Memory</div>
            <div class="report-card-value">{mem_kb:.0f} <span style="font-size:0.75rem;">KB</span></div>
        </div>
        <div class="report-card">
            <div class="report-card-label">Numeric</div>
            <div class="report-card-value">{len(numeric_cols)}</div>
        </div>
    </div>
    """

    # ── Section 2: Variable details ────────────────────────────────
    var_rows = ""
    for col in df.columns:
        is_num = pd.api.types.is_numeric_dtype(df[col])
        s = df[col]
        n_miss = s.isnull().sum()
        n_unique = s.nunique()
        miss_bar_pct = n_miss / len(df) * 100

        if is_num:
            dtype_badge = '<span class="dtype-badge dtype-num">Numeric</span>'
            stats_html = (
                f'<span class="stat-pill">μ = {s.mean():.2f}</span>'
                f'<span class="stat-pill">σ = {s.std():.2f}</span>'
                f'<span class="stat-pill">min = {s.min():.2f}</span>'
                f'<span class="stat-pill">max = {s.max():.2f}</span>'
            )
            chart = _histogram_svg(s)
            sparkline = _sparkline_svg(s)
        else:
            dtype_badge = '<span class="dtype-badge dtype-cat">Categorical</span>'
            top = s.mode().iloc[0] if len(s.mode()) > 0 else "—"
            stats_html = (
                f'<span class="stat-pill">top = {top}</span>'
                f'<span class="stat-pill">unique = {n_unique}</span>'
            )
            chart = ""
            sparkline = ""

        # Missing bar
        miss_color = "#2ea043" if n_miss == 0 else "#d73a49"
        miss_bar = (
            f'<div style="display:flex;align-items:center;gap:6px;">'
            f'<div style="flex:1;height:6px;background:#eee;border-radius:3px;overflow:hidden;">'
            f'<div style="width:{miss_bar_pct:.1f}%;height:100%;background:{miss_color};border-radius:3px;"></div>'
            f'</div>'
            f'<span style="font-size:0.7rem;color:#888;">{n_miss} ({miss_bar_pct:.1f}%)</span>'
            f'</div>'
        )

        is_target = col == target
        target_tag = ' <span style="background:#57068C;color:white;font-size:0.6rem;padding:2px 6px;border-radius:10px;vertical-align:middle;">TARGET</span>' if is_target else ""

        var_rows += f"""
        <div class="var-row">
            <div class="var-header">
                <div style="display:flex;align-items:center;gap:8px;">
                    <span class="var-name">{col}</span>{target_tag}
                    {dtype_badge}
                </div>
                <div style="display:flex;align-items:center;gap:6px;font-size:0.75rem;color:#888;">
                    <span>{n_unique} unique</span>
                </div>
            </div>
            <div class="var-body">
                <div class="var-stats">
                    <div style="margin-bottom:6px;">{stats_html}</div>
                    <div style="margin-top:4px;">{miss_bar}</div>
                </div>
                <div class="var-chart">
                    {chart}
                    <div style="margin-top:4px;">{sparkline}</div>
                </div>
            </div>
        </div>
        """

    # ── Section 3: Correlation matrix ──────────────────────────────
    corr_html = _correlation_html(df, features, target)

    # ── Section 4: Outlier analysis ────────────────────────────────
    outlier_data = _outlier_summary(df, features)
    outlier_rows = ""
    for feat, n_out, pct in sorted(outlier_data, key=lambda x: -x[2]):
        bar_color = "#2ea043" if pct < 1 else ("#f0ad4e" if pct < 5 else "#d73a49")
        severity = "Low" if pct < 1 else ("Medium" if pct < 5 else "High")
        sev_color = "#2ea043" if pct < 1 else ("#f0ad4e" if pct < 5 else "#d73a49")
        outlier_rows += f"""
        <tr>
            <td style="padding:8px 12px;font-weight:500;font-size:0.8rem;">{feat}</td>
            <td style="padding:8px 12px;text-align:center;font-size:0.8rem;">{n_out:,}</td>
            <td style="padding:8px 12px;">
                <div style="display:flex;align-items:center;gap:6px;">
                    <div style="flex:1;height:6px;background:#eee;border-radius:3px;overflow:hidden;max-width:100px;">
                        <div style="width:{min(pct, 100):.1f}%;height:100%;background:{bar_color};border-radius:3px;"></div>
                    </div>
                    <span style="font-size:0.75rem;color:#666;">{pct:.1f}%</span>
                </div>
            </td>
            <td style="padding:8px 12px;text-align:center;">
                <span style="font-size:0.7rem;padding:2px 8px;border-radius:10px;background:{sev_color}22;color:{sev_color};font-weight:600;">{severity}</span>
            </td>
        </tr>
        """

    # ── Section 5: Target distribution stats ───────────────────────
    target_s = df[target]
    target_skew = target_s.skew()
    target_kurt = target_s.kurtosis()
    target_hist = _histogram_svg(target_s, bins=30, width=300, height=80, color="#8900E1")
    q1, q2, q3 = target_s.quantile(0.25), target_s.quantile(0.50), target_s.quantile(0.75)

    # ── Assemble full report ───────────────────────────────────────
    report_html = f"""
    <style>
        .data-report {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            color: #333;
        }}
        .report-section {{
            margin-bottom: 28px;
        }}
        .report-section-title {{
            font-size: 1rem;
            font-weight: 700;
            color: #57068C;
            margin-bottom: 12px;
            padding-bottom: 6px;
            border-bottom: 2px solid #f0e6f8;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .report-card {{
            background: #f8f4fc;
            border: 1px solid #e8daf3;
            border-radius: 10px;
            padding: 14px 16px;
            text-align: center;
            transition: transform 0.15s, box-shadow 0.15s;
        }}
        .report-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(87,6,140,0.12);
        }}
        .report-card-label {{
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #888;
            margin-bottom: 4px;
        }}
        .report-card-value {{
            font-size: 1.4rem;
            font-weight: 700;
            color: #57068C;
        }}
        .var-row {{
            background: #fafafa;
            border: 1px solid #eee;
            border-radius: 10px;
            margin-bottom: 8px;
            overflow: hidden;
            transition: box-shadow 0.15s;
        }}
        .var-row:hover {{
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        }}
        .var-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 16px;
            background: white;
            border-bottom: 1px solid #f0f0f0;
        }}
        .var-name {{
            font-weight: 700;
            font-size: 0.85rem;
            color: #222;
        }}
        .dtype-badge {{
            font-size: 0.65rem;
            padding: 2px 8px;
            border-radius: 10px;
            font-weight: 600;
        }}
        .dtype-num {{
            background: #e8f4fd;
            color: #0969da;
        }}
        .dtype-cat {{
            background: #fdf4e8;
            color: #b35900;
        }}
        .var-body {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 16px;
            gap: 16px;
        }}
        .var-stats {{
            flex: 1;
        }}
        .var-chart {{
            flex-shrink: 0;
            text-align: right;
        }}
        .stat-pill {{
            display: inline-block;
            font-size: 0.72rem;
            background: #f0e6f8;
            color: #57068C;
            padding: 2px 8px;
            border-radius: 8px;
            margin: 2px 3px 2px 0;
            font-weight: 500;
        }}
        .target-box {{
            background: linear-gradient(135deg, #f8f4fc, #f0e6f8);
            border: 2px solid #d4b8e8;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
        }}
        .target-stat {{
            display: inline-block;
            margin: 6px 10px;
            text-align: center;
        }}
        .target-stat-label {{
            font-size: 0.68rem;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 0.3px;
        }}
        .target-stat-value {{
            font-size: 1.1rem;
            font-weight: 700;
            color: #57068C;
        }}
    </style>

    <div class="data-report">

        <!-- Overview -->
        <div class="report-section">
            <div class="report-section-title">📋 Overview</div>
            {overview_cards}
        </div>

        <!-- Target Variable Spotlight -->
        <div class="report-section">
            <div class="report-section-title">🎯 Target Variable — <code style="background:#f0e6f8;padding:2px 6px;border-radius:4px;color:#57068C;">{target}</code></div>
            <div class="target-box">
                <div style="margin-bottom:12px;">{target_hist}</div>
                <div>
                    <div class="target-stat">
                        <div class="target-stat-label">Mean</div>
                        <div class="target-stat-value">{target_s.mean():.2f}</div>
                    </div>
                    <div class="target-stat">
                        <div class="target-stat-label">Median</div>
                        <div class="target-stat-value">{q2:.2f}</div>
                    </div>
                    <div class="target-stat">
                        <div class="target-stat-label">Std</div>
                        <div class="target-stat-value">{target_s.std():.2f}</div>
                    </div>
                    <div class="target-stat">
                        <div class="target-stat-label">Q1</div>
                        <div class="target-stat-value">{q1:.2f}</div>
                    </div>
                    <div class="target-stat">
                        <div class="target-stat-label">Q3</div>
                        <div class="target-stat-value">{q3:.2f}</div>
                    </div>
                    <div class="target-stat">
                        <div class="target-stat-label">Skewness</div>
                        <div class="target-stat-value">{target_skew:.2f}</div>
                    </div>
                    <div class="target-stat">
                        <div class="target-stat-label">Kurtosis</div>
                        <div class="target-stat-value">{target_kurt:.2f}</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Variable Explorer -->
        <div class="report-section">
            <div class="report-section-title">🔬 Variable Explorer</div>
            {var_rows}
        </div>

        <!-- Correlations -->
        <div class="report-section">
            <div class="report-section-title">🔗 Correlation Matrix</div>
            {corr_html}
        </div>

        <!-- Outlier Analysis -->
        <div class="report-section">
            <div class="report-section-title">⚠️ Outlier Analysis <span style="font-size:0.7rem;color:#888;font-weight:400;">(IQR method)</span></div>
            <table style="width:100%;border-collapse:collapse;">
                <thead>
                    <tr style="border-bottom:2px solid #f0e6f8;">
                        <th style="padding:8px 12px;text-align:left;font-size:0.75rem;color:#888;text-transform:uppercase;">Feature</th>
                        <th style="padding:8px 12px;text-align:center;font-size:0.75rem;color:#888;text-transform:uppercase;">Count</th>
                        <th style="padding:8px 12px;text-align:left;font-size:0.75rem;color:#888;text-transform:uppercase;">Percentage</th>
                        <th style="padding:8px 12px;text-align:center;font-size:0.75rem;color:#888;text-transform:uppercase;">Severity</th>
                    </tr>
                </thead>
                <tbody>
                    {outlier_rows}
                </tbody>
            </table>
        </div>

    </div>
    """
    return report_html


def render():
    # ── Dataset selection ───────────────────────────────────────────
    ds_key, df, info = dataset_selector()
    target = get_target(ds_key)
    features = get_features(df, target)

    # ── Hero banner ─────────────────────────────────────────────────
    st.markdown(f"""
    <div class="hero-banner">
        <h1> 🎮 {info["title"]}</h1>
        <p>DS4EVERYONE @ NYU — Final Project Demo App</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Business problem ────────────────────────────────────────────
    st.markdown("## 🎯 Business Problem")
    st.markdown(info["problem"])
    st.markdown("---")

    # ── Key metrics ─────────────────────────────────────────────────
    st.markdown("## 📋 Dataset at a Glance")
    c1, c2, c3, c4 = st.columns(4)
    for col, label, value in [
        (c1, "Rows", f"{len(df):,}"),
        (c2, "Features", str(len(features))),
        (c3, "Target", info["target"]),
        (c4, "Source", info["source"]),
    ]:
        col.markdown(f"""
        <div class="metric-card">
            <h3>{label}</h3>
            <p style="font-size:1.3rem;word-break:break-word;">{value}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # ── Data preview ────────────────────────────────────────────────
    tab_head, tab_tail, tab_sample = st.tabs(["First rows", "Last rows", "Random sample"])
    with tab_head:
        st.dataframe(df.head(10), use_container_width=True)
    with tab_tail:
        st.dataframe(df.tail(10), use_container_width=True)
    with tab_sample:
        st.dataframe(df.sample(10, random_state=42), use_container_width=True)

    st.markdown("---")

    # ── Feature dictionary ──────────────────────────────────────────
    st.markdown("## 📖 Feature Dictionary")
    feat_df = pd.DataFrame(
        [
            {"Feature": k, "Description": v, "Type": str(df[k].dtype)}
            for k, v in info["features_desc"].items()
            if k in df.columns
        ]
    )
    st.dataframe(feat_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Descriptive statistics ──────────────────────────────────────
    st.markdown("## 📊 Descriptive Statistics")
    st.dataframe(df.describe().T.style.format("{:.2f}"), use_container_width=True)

    # ── Data quality ────────────────────────────────────────────────
    st.markdown("## ✅ Data Quality Check")
    col_a, col_b = st.columns(2)
    with col_a:
        missing = df.isnull().sum()
        miss_pct = (missing / len(df) * 100).round(2)
        quality_df = pd.DataFrame({"Missing": missing, "% Missing": miss_pct})
        st.dataframe(quality_df, use_container_width=True)
    with col_b:
        completeness = (1 - df.isnull().mean().mean()) * 100
        duplicates = df.duplicated().sum()
        st.markdown(f"""
        <div class="metric-card">
            <h3>Overall Completeness</h3>
            <p>{completeness:.1f}%</p>
        </div>
        <div class="metric-card">
            <h3>Duplicate Rows</h3>
            <p>{duplicates}</p>
        </div>
        <div class="metric-card">
            <h3>Memory Usage</h3>
            <p>{df.memory_usage(deep=True).sum() / 1024:.0f} KB</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── HTML Data Exploration Report ────────────────────────────────
    st.markdown("## 📑 Data Exploration Report")
    st.caption("Interactive profiling report — distributions, correlations, and outlier analysis.")
    report_html = _build_data_report(df, features, target, info)

    # Estimate iframe height from content length (rows + features + target + outliers)
    n_features = len(df.columns)
    report_height = 900 + n_features * 90 + max(0, n_features - 6) * 40

    full_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{
    margin: 0;
    padding: 12px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    color: #333;
    background: white;
  }}
</style>
</head>
<body>
{report_html}
</body>
</html>"""

    components.html(full_html, height=report_height, scrolling=True)