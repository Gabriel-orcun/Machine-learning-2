"""
Electric School Bus Intelligence Platform
Dashboard for Lab 4 – Apriori Association Analysis
Machine Learning II (ADIF84)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from mlxtend.frequent_patterns import apriori, association_rules

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ESB Intelligence — Lab 4",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# DESIGN TOKENS & GLOBAL CSS
# ─────────────────────────────────────────────
COLORS = {
    "bg": "#0F1117",
    "surface": "#161B27",
    "surface_2": "#1E2535",
    "border": "#252D3D",
    "border_hover": "#3A4460",
    "text_primary": "#F0F4FF",
    "text_secondary": "#8B95B0",
    "text_muted": "#555E75",
    "accent_green": "#22C55E",
    "accent_blue": "#3B82F6",
    "accent_amber": "#F59E0B",
    "accent_red": "#EF4444",
    "accent_purple": "#8B5CF6",
    "accent_cyan": "#06B6D4",
    "positive": "#22C55E",
    "negative": "#EF4444",
    "neutral": "#3B82F6",
}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {{
    font-family: 'DM Sans', sans-serif;
    background-color: {COLORS['bg']};
    color: {COLORS['text_primary']};
}}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{
    padding: 2rem 2.5rem 2rem 2.5rem;
    max-width: 100%;
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background-color: {COLORS['surface']};
    border-right: 1px solid {COLORS['border']};
}}
[data-testid="stSidebar"] .block-container {{ padding: 1.5rem; }}

/* ── KPI Card ── */
.kpi-card {{
    background: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    position: relative;
    transition: border-color 0.2s ease;
}}
.kpi-card:hover {{ border-color: {COLORS['border_hover']}; }}
.kpi-label {{
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {COLORS['text_muted']};
    margin-bottom: 0.5rem;
}}
.kpi-value {{
    font-size: 2rem;
    font-weight: 600;
    line-height: 1;
    color: {COLORS['text_primary']};
    letter-spacing: -0.02em;
    margin-bottom: 0.35rem;
}}
.kpi-delta {{
    font-size: 0.78rem;
    font-weight: 500;
    color: {COLORS['accent_green']};
}}
.kpi-delta.neg {{ color: {COLORS['accent_red']}; }}
.kpi-dot {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 6px;
}}

/* ── Section headers ── */
.section-label {{
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {COLORS['text_muted']};
    margin-bottom: 0.75rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid {COLORS['border']};
}}
.section-title {{
    font-size: 1.1rem;
    font-weight: 600;
    color: {COLORS['text_primary']};
    margin-bottom: 0.25rem;
    letter-spacing: -0.01em;
}}
.section-sub {{
    font-size: 0.82rem;
    color: {COLORS['text_secondary']};
    margin-bottom: 1.25rem;
}}

/* ── Page header ── */
.page-header {{
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid {COLORS['border']};
    margin-bottom: 2rem;
}}
.page-title {{
    font-size: 1.5rem;
    font-weight: 600;
    color: {COLORS['text_primary']};
    letter-spacing: -0.025em;
    margin: 0;
    line-height: 1.2;
}}
.page-badge {{
    display: inline-block;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    background: rgba(59, 130, 246, 0.15);
    color: {COLORS['accent_blue']};
    border: 1px solid rgba(59, 130, 246, 0.25);
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
    margin-bottom: 0.4rem;
}}

/* ── Rule card ── */
.rule-card {{
    background: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.5rem;
    transition: border-color 0.2s;
}}
.rule-card:hover {{ border-color: {COLORS['border_hover']}; }}
.rule-antecedent {{
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: {COLORS['accent_blue']};
}}
.rule-consequent {{
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: {COLORS['accent_green']};
}}
.rule-arrow {{
    color: {COLORS['text_muted']};
    margin: 0 0.4rem;
    font-weight: 300;
}}
.rule-badge {{
    display: inline-block;
    font-size: 0.65rem;
    font-weight: 600;
    padding: 0.15rem 0.45rem;
    border-radius: 4px;
    margin-left: 0.3rem;
    font-family: 'DM Mono', monospace;
}}
.badge-lift {{
    background: rgba(139, 92, 246, 0.15);
    color: {COLORS['accent_purple']};
    border: 1px solid rgba(139, 92, 246, 0.2);
}}
.badge-conf {{
    background: rgba(34, 197, 94, 0.1);
    color: {COLORS['accent_green']};
    border: 1px solid rgba(34, 197, 94, 0.2);
}}
.badge-supp {{
    background: rgba(59, 130, 246, 0.1);
    color: {COLORS['accent_blue']};
    border: 1px solid rgba(59, 130, 246, 0.2);
}}

/* ── Divider ── */
.divider {{
    border: none;
    border-top: 1px solid {COLORS['border']};
    margin: 1.5rem 0;
}}

/* ── Profile tile ── */
.profile-tile {{
    background: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 12px;
    padding: 1.25rem;
    height: 100%;
}}
.profile-icon {{
    font-size: 1.5rem;
    margin-bottom: 0.6rem;
}}
.profile-name {{
    font-size: 0.9rem;
    font-weight: 600;
    color: {COLORS['text_primary']};
    margin-bottom: 0.3rem;
}}
.profile-desc {{
    font-size: 0.78rem;
    color: {COLORS['text_secondary']};
    line-height: 1.5;
    margin-bottom: 0.8rem;
}}
.profile-tag {{
    display: inline-block;
    font-size: 0.65rem;
    font-weight: 500;
    padding: 0.15rem 0.45rem;
    border-radius: 4px;
    margin: 0.15rem 0.1rem;
    font-family: 'DM Mono', monospace;
}}

/* ── Streamlit overrides ── */
.stTabs [data-baseweb="tab-list"] {{
    background-color: {COLORS['surface']};
    border-radius: 8px;
    padding: 3px;
    gap: 2px;
    border: 1px solid {COLORS['border']};
}}
.stTabs [data-baseweb="tab"] {{
    color: {COLORS['text_secondary']};
    font-size: 0.82rem;
    font-weight: 500;
    border-radius: 6px;
    padding: 0.45rem 1rem;
    border: none;
}}
.stTabs [aria-selected="true"] {{
    background-color: {COLORS['surface_2']};
    color: {COLORS['text_primary']};
    font-weight: 600;
}}
.stSelectbox label, .stSlider label, .stMultiSelect label {{
    font-size: 0.78rem;
    color: {COLORS['text_secondary']};
    font-weight: 500;
    letter-spacing: 0.04em;
}}

/* ── Plotly chart background fix ── */
.js-plotly-plot .plotly .bg {{ fill: transparent !important; }}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 5px; height: 5px; }}
::-webkit-scrollbar-track {{ background: {COLORS['bg']}; }}
::-webkit-scrollbar-thumb {{ background: {COLORS['border']}; border-radius: 10px; }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA LOADING & CACHING
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_analyze(path: str, min_support: float, min_confidence: float):
    df = pd.read_csv(path)
    engagement_cols = ["committed", "operating", "delivering"]
    df["any_engagement"] = (df[engagement_cols].sum(axis=1) > 0).astype(int)
    df_engaged = df[df["any_engagement"] == 1].copy()
    df_not = df[df["any_engagement"] == 0].copy()

    feature_cols = [c for c in df.columns if c not in ["any_engagement"]]
    df_trans = df_engaged[feature_cols].astype(bool)

    freq_items = apriori(df_trans, min_support=min_support, use_colnames=True)
    freq_sorted = freq_items.sort_values("support", ascending=False).reset_index(drop=True)
    freq_sorted["size"] = freq_sorted["itemsets"].apply(len)
    freq_sorted["label"] = freq_sorted["itemsets"].apply(
        lambda x: " + ".join(sorted(list(x)))
    )
    freq_sorted["count"] = (freq_sorted["support"] * len(df_engaged)).astype(int)

    rules = pd.DataFrame()
    if len(freq_items) >= 2:
        rules = association_rules(freq_items, metric="confidence", min_threshold=min_confidence)
        if len(rules) > 0:
            rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)
            rules["ant_str"] = rules["antecedents"].apply(lambda x: " & ".join(sorted(list(x))))
            rules["con_str"] = rules["consequents"].apply(lambda x: " & ".join(sorted(list(x))))
            rules["count"] = (rules["support"] * len(df_engaged)).astype(int)

    # Comparison stats
    attr_cols = ["high_income", "high_poverty", "high_need", "urban", "rural",
                 "large_district", "small_district", "epa_2023", "pollution_burden"]
    comp = []
    for col in attr_cols:
        e = df_engaged[col].mean() * 100
        n = df_not[col].mean() * 100
        comp.append({"attribute": col, "engaged_pct": e, "not_engaged_pct": n, "diff": e - n})
    comp_df = pd.DataFrame(comp)

    # Engagement distribution per attribute (for heatmap)
    engagement_breakdown = {}
    for col in feature_cols:
        for status in ["committed", "operating", "delivering"]:
            sub = df[df[col] == 1]
            engagement_breakdown[(col, status)] = sub[status].mean() * 100

    return df, df_engaged, df_not, freq_sorted, rules, comp_df, feature_cols


# ─────────────────────────────────────────────
# SIDEBAR CONTROLS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="margin-bottom:1.5rem;">
        <div style="font-size:0.65rem;letter-spacing:0.12em;text-transform:uppercase;
                    color:#555E75;font-weight:600;margin-bottom:0.5rem;">Platform</div>
        <div style="font-size:1.05rem;font-weight:600;color:#F0F4FF;letter-spacing:-0.01em;">
            ⚡ ESB Intelligence
        </div>
        <div style="font-size:0.75rem;color:#8B95B0;margin-top:0.2rem;">Lab 4 · ADIF84</div>
    </div>
    <hr style="border:none;border-top:1px solid #252D3D;margin:1rem 0;">
    """, unsafe_allow_html=True)

    st.markdown("""<div style="font-size:0.65rem;letter-spacing:0.1em;text-transform:uppercase;
                color:#555E75;font-weight:600;margin-bottom:0.8rem;">Model Parameters</div>""",
                unsafe_allow_html=True)

    min_support = st.slider(
        "Min Support",
        min_value=0.02, max_value=0.30, value=0.05, step=0.01,
        help="Minimum fraction of engaged districts sharing an itemset"
    )
    min_confidence = st.slider(
        "Min Confidence",
        min_value=0.10, max_value=0.90, value=0.40, step=0.05,
        help="Minimum conditional probability P(B|A)"
    )
    top_n = st.slider("Top N itemsets / rules", 10, 50, 20, 5)

    st.markdown("""<hr style="border:none;border-top:1px solid #252D3D;margin:1rem 0;">
    <div style="font-size:0.65rem;letter-spacing:0.1em;text-transform:uppercase;
                color:#555E75;font-weight:600;margin-bottom:0.8rem;">Filter Itemsets</div>""",
                unsafe_allow_html=True)

    size_filter = st.multiselect(
        "Itemset size",
        options=[1, 2, 3, 4, 5],
        default=[1, 2, 3],
        help="Filter by number of items in the set"
    )

    st.markdown("""<hr style="border:none;border-top:1px solid #252D3D;margin:1rem 0;">""",
                unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.72rem;color:#555E75;line-height:1.6;">
        <b style="color:#8B95B0;">Dataset</b><br>
        19,517 US school districts<br>
        13 binary attributes<br>
        Algorithm: Apriori (mlxtend)<br><br>
        <b style="color:#8B95B0;">Scope</b><br>
        Engaged districts only<br>
        (committed / operating / delivering)
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
DATA_PATH = "apriori_data_simplified.csv"
# Fallback path if running from another directory
import os
if not os.path.exists(DATA_PATH):
    DATA_PATH = "/mnt/user-data/uploads/apriori_data_simplified.csv"

with st.spinner("Running Apriori analysis…"):
    df, df_engaged, df_not, freq_sorted, rules, comp_df, feature_cols = load_and_analyze(
        DATA_PATH, min_support, min_confidence
    )

n_total = len(df)
n_engaged = len(df_engaged)
n_committed = int(df["committed"].sum())
n_operating = int(df["operating"].sum())
n_delivering = int(df["delivering"].sum())
n_itemsets = len(freq_sorted)
n_rules = len(rules) if len(rules) > 0 else 0

# Apply size filter
freq_filtered = freq_sorted[freq_sorted["size"].isin(size_filter)] if size_filter else freq_sorted


# ─────────────────────────────────────────────
# PLOTLY DEFAULTS
# ─────────────────────────────────────────────
PLOT_BG = "rgba(0,0,0,0)"
PAPER_BG = "rgba(0,0,0,0)"
GRID_COLOR = "#252D3D"
FONT_COLOR = "#8B95B0"
FONT_FAMILY = "DM Sans, sans-serif"

def base_layout(**kwargs):
    layout = dict(
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(
            family=FONT_FAMILY,
            color=FONT_COLOR,
            size=12
        ),
        margin=dict(l=10, r=10, t=30, b=10),
    )

    # overwrite defaults safely
    for key, value in kwargs.items():
        layout[key] = value

    return layout


def style_axes(fig, row=None, col=None):
    axis_style = dict(
        gridcolor=GRID_COLOR,
        zerolinecolor=GRID_COLOR,
        linecolor=GRID_COLOR,
        tickfont=dict(color=FONT_COLOR, size=11),
    )
    if row:
        fig.update_xaxes(axis_style, row=row, col=col)
        fig.update_yaxes(axis_style, row=row, col=col)
    else:
        fig.update_xaxes(axis_style)
        fig.update_yaxes(axis_style)
    return fig


# ─────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────
st.markdown(f"""
<div class="page-header">
  <div>
    <div class="page-badge">Apriori · Association Mining</div>
    <p class="page-title">Electric School Bus — District Engagement Profiles</p>
    <p style="font-size:0.83rem;color:{COLORS['text_secondary']};margin:0;">
      Frequent itemset analysis across {n_total:,} US school districts &nbsp;·&nbsp;
      Engaged cohort: {n_engaged:,} districts ({n_engaged/n_total*100:.1f}%)
    </p>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────
k1, k2, k3, k4, k5, k6 = st.columns(6)

def kpi(col, label, value, delta=None, delta_neg=False, color="#22C55E"):
    delta_class = "neg" if delta_neg else ""
    delta_html = f'<div class="kpi-delta {delta_class}">{delta}</div>' if delta else ""
    dot = f'<span class="kpi-dot" style="background:{color};"></span>'
    col.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{dot}{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>""", unsafe_allow_html=True)

kpi(k1, "Total Districts", f"{n_total:,}", "US national dataset", color=COLORS["accent_blue"])
kpi(k2, "Engaged Districts", f"{n_engaged:,}", f"{n_engaged/n_total*100:.1f}% of total", color=COLORS["accent_green"])
kpi(k3, "Committed", f"{n_committed:,}", f"{n_committed/n_total*100:.1f}% of total", color=COLORS["accent_amber"])
kpi(k4, "Operating", f"{n_operating:,}", f"{n_operating/n_total*100:.1f}% of total", color=COLORS["accent_cyan"])
kpi(k5, "Frequent Itemsets", f"{n_itemsets:,}", f"min_support = {min_support:.0%}", color=COLORS["accent_purple"])
kpi(k6, "Association Rules", f"{n_rules:,}", f"min_conf = {min_confidence:.0%}", color=COLORS["accent_red"])

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "  📊  Exploration  ",
    "  🔗  Frequent Itemsets  ",
    "  ⚙️  Association Rules  ",
    "  🏷️  District Profiles  ",
])


# ════════════════════════════════════════════
# TAB 1 — EXPLORATION
# ════════════════════════════════════════════
with tab1:
    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        st.markdown("""<div class='section-label'>Attribute Comparison</div>
        <div class='section-title'>Engaged vs Non-Engaged Districts</div>
        <div class='section-sub'>Percentage of districts with each attribute, by engagement status</div>""",
                    unsafe_allow_html=True)

        attr_labels = {
            "high_income": "High Income",
            "high_poverty": "High Poverty",
            "high_need": "High Need",
            "urban": "Urban",
            "rural": "Rural",
            "large_district": "Large District",
            "small_district": "Small District",
            "epa_2023": "EPA 2023 Priority",
            "pollution_burden": "Pollution Burden",
        }
        comp_plot = comp_df.copy()
        comp_plot["attr_label"] = comp_plot["attribute"].map(attr_labels)
        comp_plot = comp_plot.sort_values("diff", ascending=True)

        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            name="Not Engaged",
            x=comp_plot["not_engaged_pct"],
            y=comp_plot["attr_label"],
            orientation="h",
            marker_color=COLORS["surface_2"],
            marker_line_color=COLORS["border"],
            marker_line_width=1,
            text=[f"{v:.0f}%" for v in comp_plot["not_engaged_pct"]],
            textposition="outside",
            textfont=dict(color=FONT_COLOR, size=10),
            hovertemplate="<b>%{y}</b><br>Not Engaged: %{x:.1f}%<extra></extra>",
        ))
        fig_comp.add_trace(go.Bar(
            name="Engaged",
            x=comp_plot["engaged_pct"],
            y=comp_plot["attr_label"],
            orientation="h",
            marker_color=COLORS["accent_blue"],
            marker_line_color="rgba(0,0,0,0)",
            text=[f"{v:.0f}%" for v in comp_plot["engaged_pct"]],
            textposition="outside",
            textfont=dict(color=COLORS["text_primary"], size=10),
            hovertemplate="<b>%{y}</b><br>Engaged: %{x:.1f}%<extra></extra>",
        ))
        fig_comp.update_layout(
            **base_layout(height=380, barmode="overlay", showlegend=True,
                         legend=dict(orientation="h", x=0, y=-0.08, font_size=11)),
            xaxis=dict(gridcolor=GRID_COLOR, tickfont_color=FONT_COLOR,
                       title=dict(text="Percentage of districts (%)", font_color=FONT_COLOR, font_size=11)),
            yaxis=dict(gridcolor=GRID_COLOR, tickfont_color=FONT_COLOR),
        )
        style_axes(fig_comp)
        st.plotly_chart(fig_comp, use_container_width=True, config={"displayModeBar": False})

    with col_right:
        st.markdown("""<div class='section-label'>Engagement Breakdown</div>
        <div class='section-title'>Statuses & Distribution</div>
        <div class='section-sub'>Decomposition of the 19,517 districts</div>""",
                    unsafe_allow_html=True)

        labels = ["Not Engaged", "Committed only", "Operating", "Delivering"]
        committed_only = n_committed - n_operating
        values = [n_total - n_engaged, committed_only, n_operating - n_delivering, n_delivering]
        colors_pie = [COLORS["surface_2"], COLORS["accent_amber"], COLORS["accent_blue"], COLORS["accent_green"]]

        fig_pie = go.Figure(go.Pie(
            labels=labels, values=values,
            marker=dict(colors=colors_pie, line=dict(color=COLORS["bg"], width=3)),
            textinfo="percent",
            hovertemplate="<b>%{label}</b><br>%{value:,} districts (%{percent})<extra></extra>",
            hole=0.55,
            pull=[0, 0.02, 0.02, 0.02],
            sort=False,
        ))
        fig_pie.add_annotation(
            text=f"<b style='font-size:18px'>{n_engaged:,}</b><br><span style='font-size:11px'>Engaged</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(color=COLORS["text_primary"], family=FONT_FAMILY, size=13),
        )
        fig_pie.update_layout(
            **base_layout(
                height=240,
                legend=dict(
                    orientation="v",
                    x=0.7,
                    y=0.5,
                    font_size=10,
                    font_color=FONT_COLOR,
                ),
                margin=dict(
                    l=0,
                    r=0,
                    t=10,
                    b=0,
                ),
            )
        )
        st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # Differential bars
        st.markdown("""<div class='section-label'>Differentials</div>
        <div class='section-title'>Engagement lift by attribute</div>""",
                    unsafe_allow_html=True)

        diff_data = comp_plot.sort_values("diff", ascending=False)
        colors_diff = [COLORS["accent_green"] if d > 0 else COLORS["accent_red"]
                       for d in diff_data["diff"]]
        fig_diff = go.Figure(go.Bar(
            x=diff_data["attr_label"],
            y=diff_data["diff"],
            marker_color=colors_diff,
            marker_line_color="rgba(0,0,0,0)",
            text=[f"{d:+.1f}pp" for d in diff_data["diff"]],
            textposition="outside",
            textfont=dict(size=9, color=FONT_COLOR),
            hovertemplate="<b>%{x}</b><br>Δ %{y:+.1f} pp<extra></extra>",
        ))
        fig_diff.add_hline(y=0, line_color=COLORS["border"], line_width=1)
        fig_diff.update_layout(
            **base_layout(height=200),
            xaxis=dict(tickfont_color=FONT_COLOR, tickfont_size=9, tickangle=-35),
            yaxis=dict(gridcolor=GRID_COLOR, tickfont_color=FONT_COLOR, ticksuffix=" pp"),
        )
        st.plotly_chart(fig_diff, use_container_width=True, config={"displayModeBar": False})


# ════════════════════════════════════════════
# TAB 2 — FREQUENT ITEMSETS
# ════════════════════════════════════════════
with tab2:
    col_a, col_b = st.columns([3, 2], gap="large")

    with col_a:
        st.markdown(f"""<div class='section-label'>Frequent Itemsets</div>
        <div class='section-title'>Top {min(top_n, len(freq_filtered))} itemsets — support ≥ {min_support:.0%}</div>
        <div class='section-sub'>Fraction of engaged districts sharing each combination of attributes</div>""",
                    unsafe_allow_html=True)

        if len(freq_filtered) == 0:
            st.warning("No itemsets match current filters. Try lowering min_support or changing size filter.")
        else:
            top = freq_filtered.head(top_n).copy()
            top_plot = top.sort_values("support")

            color_map = {1: COLORS["accent_blue"], 2: COLORS["accent_green"],
                         3: COLORS["accent_amber"], 4: COLORS["accent_red"],
                         5: COLORS["accent_purple"]}
            bar_colors = [color_map.get(s, COLORS["text_secondary"]) for s in top_plot["size"]]

            fig_items = go.Figure(go.Bar(
                x=top_plot["support"],
                y=top_plot["label"],
                orientation="h",
                marker_color=bar_colors,
                marker_line_color="rgba(0,0,0,0)",
                text=[f"{s:.1%} ({c:,})" for s, c in zip(top_plot["support"], top_plot["count"])],
                textposition="outside",
                textfont=dict(size=10, color=FONT_COLOR, family="DM Mono, monospace"),
                hovertemplate="<b>%{y}</b><br>Support: %{x:.1%} (%{customdata:,} districts)<extra></extra>",
                customdata=top_plot["count"],
            ))
            fig_items.update_layout(
                **base_layout(height=max(300, len(top_plot) * 28 + 60)),
                xaxis=dict(gridcolor=GRID_COLOR, tickfont_color=FONT_COLOR,
                           title=dict(text="Support", font_color=FONT_COLOR, font_size=11),
                           tickformat=".0%"),
                yaxis=dict(gridcolor=GRID_COLOR, tickfont_color=COLORS["text_primary"],
                           tickfont_size=10, tickfont_family="DM Mono, monospace"),
            )
            st.plotly_chart(fig_items, use_container_width=True, config={"displayModeBar": False})

    with col_b:
        st.markdown("""<div class='section-label'>Itemset Analysis</div>
        <div class='section-title'>Size Distribution</div>
        <div class='section-sub'>How many items per itemset</div>""",
                    unsafe_allow_html=True)

        size_dist = freq_sorted.groupby("size")["support"].agg(["count", "mean"]).reset_index()
        fig_size = go.Figure()
        fig_size.add_trace(go.Bar(
            x=[f"Size {s}" for s in size_dist["size"]],
            y=size_dist["count"],
            marker_color=[color_map.get(s, COLORS["text_muted"]) for s in size_dist["size"]],
            marker_line_color="rgba(0,0,0,0)",
            text=size_dist["count"],
            textposition="outside",
            textfont=dict(color=FONT_COLOR, size=10),
            hovertemplate="<b>%{x}</b><br>%{y} itemsets<br>Avg support: %{customdata:.1%}<extra></extra>",
            customdata=size_dist["mean"],
        ))
        fig_size.update_layout(
            **base_layout(height=200),
            xaxis=dict(tickfont_color=FONT_COLOR),
            yaxis=dict(gridcolor=GRID_COLOR, tickfont_color=FONT_COLOR, title_text="Count"),
        )
        st.plotly_chart(fig_size, use_container_width=True, config={"displayModeBar": False})

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        st.markdown("""<div class='section-label'>Item Frequency</div>
        <div class='section-title'>Individual attribute prevalence</div>
        <div class='section-sub'>Among engaged districts</div>""",
                    unsafe_allow_html=True)

        single_items = freq_sorted[freq_sorted["size"] == 1].copy()
        single_items["attr"] = single_items["itemsets"].apply(lambda x: list(x)[0])
        single_items = single_items.sort_values("support", ascending=True)

        fig_single = go.Figure(go.Bar(
            x=single_items["support"],
            y=single_items["attr"],
            orientation="h",
            marker=dict(
                color=single_items["support"],
                colorscale=[[0, COLORS["surface_2"]], [0.5, COLORS["accent_blue"]], [1, COLORS["accent_cyan"]]],
                line_color="rgba(0,0,0,0)",
            ),
            text=[f"{s:.1%}" for s in single_items["support"]],
            textposition="outside",
            textfont=dict(size=10, color=FONT_COLOR, family="DM Mono, monospace"),
            hovertemplate="<b>%{y}</b><br>%{x:.1%} of engaged districts<extra></extra>",
        ))
        fig_single.update_layout(
            **base_layout(height=280),
            xaxis=dict(gridcolor=GRID_COLOR, tickformat=".0%", tickfont_color=FONT_COLOR),
            yaxis=dict(tickfont_color=COLORS["text_primary"], tickfont_family="DM Mono, monospace",
                       tickfont_size=10),
        )
        st.plotly_chart(fig_single, use_container_width=True, config={"displayModeBar": False})


# ════════════════════════════════════════════
# TAB 3 — ASSOCIATION RULES
# ════════════════════════════════════════════
with tab3:
    if len(rules) == 0:
        st.warning(f"No association rules found with current parameters. Try lowering min_confidence ({min_confidence:.0%}) or min_support ({min_support:.0%}).")
    else:
        # Scatter: Support vs Confidence colored by Lift
        col_scatter, col_hist = st.columns([3, 2], gap="large")

        with col_scatter:
            st.markdown(f"""<div class='section-label'>Rule Space</div>
            <div class='section-title'>Support × Confidence — colored by Lift</div>
            <div class='section-sub'>{n_rules:,} rules found · Bubble size = count of districts</div>""",
                        unsafe_allow_html=True)

            fig_scatter = go.Figure(go.Scatter(
                x=rules["support"],
                y=rules["confidence"],
                mode="markers",
                marker=dict(
                    size=np.clip(rules["count"] / 10, 5, 25),
                    color=rules["lift"],
                    colorscale="Viridis",
                    colorbar=dict(
                        title=dict(text="Lift", font_color=FONT_COLOR, font_size=11),
                        tickfont=dict(color=FONT_COLOR, size=10),
                        len=0.6,
                    ),
                    opacity=0.7,
                    line=dict(width=0.5, color=COLORS["bg"]),
                ),
                text=[f"{a} → {c}" for a, c in zip(rules["ant_str"], rules["con_str"])],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Support: %{x:.2%}<br>"
                    "Confidence: %{y:.1%}<br>"
                    "Lift: %{marker.color:.2f}<extra></extra>"
                ),
            ))
            fig_scatter.add_hline(y=min_confidence, line_dash="dot",
                                   line_color=COLORS["text_muted"], line_width=1,
                                   annotation_text=f"min_conf={min_confidence:.0%}",
                                   annotation_font_color=COLORS["text_muted"],
                                   annotation_font_size=10)
            fig_scatter.update_layout(
                **base_layout(height=380),
                xaxis=dict(gridcolor=GRID_COLOR, tickformat=".0%", tickfont_color=FONT_COLOR,
                           title=dict(text="Support", font_color=FONT_COLOR)),
                yaxis=dict(gridcolor=GRID_COLOR, tickformat=".0%", tickfont_color=FONT_COLOR,
                           title=dict(text="Confidence", font_color=FONT_COLOR)),
            )
            st.plotly_chart(fig_scatter, use_container_width=True, config={"displayModeBar": False})

        with col_hist:
            st.markdown("""<div class='section-label'>Lift Distribution</div>
            <div class='section-title'>How strong are the associations?</div>
            <div class='section-sub'>Lift > 1 = positive association</div>""",
                        unsafe_allow_html=True)

            fig_lift = go.Figure(go.Histogram(
                x=rules["lift"],
                nbinsx=40,
                marker_color=COLORS["accent_purple"],
                marker_line_color=COLORS["bg"],
                marker_line_width=0.5,
                opacity=0.85,
                hovertemplate="Lift %{x:.2f}: %{y} rules<extra></extra>",
            ))
            fig_lift.add_vline(x=1.0, line_dash="dot", line_color=COLORS["accent_amber"],
                               annotation_text="Lift = 1", annotation_font_color=COLORS["accent_amber"],
                               annotation_font_size=10)
            fig_lift.update_layout(
                **base_layout(height=200),
                xaxis=dict(gridcolor=GRID_COLOR, tickfont_color=FONT_COLOR,
                           title_text="Lift", title_font_color=FONT_COLOR),
                yaxis=dict(gridcolor=GRID_COLOR, tickfont_color=FONT_COLOR, title_text="Rules"),
            )
            st.plotly_chart(fig_lift, use_container_width=True, config={"displayModeBar": False})

            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

            # Metrics
            st.markdown("""<div class='section-label'>Rule Statistics</div>""",
                        unsafe_allow_html=True)
            m1, m2 = st.columns(2)

            def mini_kpi(col, label, value):
                col.markdown(f"""
                <div style="background:{COLORS['surface']};border:1px solid {COLORS['border']};
                            border-radius:8px;padding:0.75rem;text-align:center;">
                    <div style="font-size:0.65rem;color:{COLORS['text_muted']};
                                text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.3rem;">{label}</div>
                    <div style="font-size:1.25rem;font-weight:600;color:{COLORS['text_primary']};
                                letter-spacing:-0.02em;">{value}</div>
                </div>""", unsafe_allow_html=True)

            mini_kpi(m1, "Max Lift", f"{rules['lift'].max():.2f}")
            mini_kpi(m2, "Avg Confidence", f"{rules['confidence'].mean():.0%}")
            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
            mini_kpi(m1, "Avg Lift", f"{rules['lift'].mean():.2f}")
            mini_kpi(m2, "Median Support", f"{rules['support'].median():.1%}")

        # ── Top rules list ──────────────────────────────
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown(f"""<div class='section-label'>Top Rules</div>
        <div class='section-title'>Highest-lift association rules</div>
        <div class='section-sub'>Rules with strongest positive associations among engaged districts</div>""",
                    unsafe_allow_html=True)

        top_rules = rules.head(top_n)

        cols_rules = st.columns(2)
        for i, (_, row) in enumerate(top_rules.iterrows()):
            lift_color = (COLORS["accent_green"] if row["lift"] > 2
                         else COLORS["accent_amber"] if row["lift"] > 1.5
                         else COLORS["text_secondary"])
            html = f"""
            <div class="rule-card">
                <div style="margin-bottom:0.45rem;">
                    <span class="rule-antecedent">{row['ant_str']}</span>
                    <span class="rule-arrow">→</span>
                    <span class="rule-consequent">{row['con_str']}</span>
                </div>
                <div>
                    <span class="rule-badge badge-lift">⬆ Lift {row['lift']:.2f}</span>
                    <span class="rule-badge badge-conf">✓ Conf {row['confidence']:.0%}</span>
                    <span class="rule-badge badge-supp">◦ Supp {row['support']:.1%}</span>
                </div>
            </div>"""
            cols_rules[i % 2].markdown(html, unsafe_allow_html=True)


# ════════════════════════════════════════════
# TAB 4 — DISTRICT PROFILES
# ════════════════════════════════════════════
with tab4:
    st.markdown("""<div class='section-label'>Archetype Profiles</div>
    <div class='section-title'>Four dominant district profiles identified by Apriori</div>
    <div class='section-sub'>Based on frequent itemsets and engagement patterns</div>""",
                unsafe_allow_html=True)

    # Profile definitions derived from analysis
    profiles = [
        {
            "icon": "🏙️",
            "name": "Urban Champion",
            "pct": 22.3,
            "color": COLORS["accent_blue"],
            "bg": "rgba(59,130,246,0.08)",
            "border": "rgba(59,130,246,0.25)",
            "desc": "Urban districts with large enrollment and high income. Early adopters of ESB, driving visible impact in metro areas.",
            "tags": [("urban", COLORS["accent_blue"]), ("high_income", COLORS["accent_green"]),
                     ("large_district", COLORS["accent_cyan"]), ("operating", COLORS["accent_green"])],
            "stats": [("Urban share", "22.3%"), ("High income", "44.3%"), ("Operating", "53.6%")],
        },
        {
            "icon": "⚖️",
            "name": "Equity-Driven",
            "pct": 53.3,
            "color": COLORS["accent_amber"],
            "bg": "rgba(245,158,11,0.08)",
            "border": "rgba(245,158,11,0.25)",
            "desc": "Large districts with high poverty and high-need students. Motivated by environmental equity and EPA 2023 funding.",
            "tags": [("high_poverty", COLORS["accent_amber"]), ("large_district", COLORS["accent_cyan"]),
                     ("epa_2023", COLORS["accent_purple"]), ("high_need", COLORS["accent_red"])],
            "stats": [("High poverty", "53.3%"), ("Large district", "74.1%"), ("EPA 2023", "40.4%")],
        },
        {
            "icon": "🌾",
            "name": "Rural Pioneer",
            "pct": 35.2,
            "color": COLORS["accent_green"],
            "bg": "rgba(34,197,94,0.08)",
            "border": "rgba(34,197,94,0.25)",
            "desc": "Small rural districts with low income and pollution burden. EPA-prioritized communities making early commitments.",
            "tags": [("rural", COLORS["accent_green"]), ("small_district", COLORS["text_muted"]),
                     ("low_income", COLORS["accent_amber"]), ("pollution_burden", COLORS["accent_red"])],
            "stats": [("Rural share", "35.2%"), ("Pollution burden", "31.4%"), ("EPA lift", "+13.2pp")],
        },
        {
            "icon": "📋",
            "name": "Policy Catalyst",
            "pct": 40.4,
            "color": COLORS["accent_purple"],
            "bg": "rgba(139,92,246,0.08)",
            "border": "rgba(139,92,246,0.25)",
            "desc": "Districts activated by federal EPA 2023 prioritization. Public policy as the primary driver — responsive to incentives.",
            "tags": [("epa_2023", COLORS["accent_purple"]), ("committed", COLORS["accent_blue"]),
                     ("pollution_burden", COLORS["accent_red"]), ("high_need", COLORS["accent_amber"])],
            "stats": [("EPA priority", "40.4%"), ("Lift vs avg", "+13.2pp"), ("Committed", "100%")],
        },
    ]

    p_cols = st.columns(4, gap="small")
    for i, p in enumerate(profiles):
        with p_cols[i]:
            tags_html = "".join([
                f'<span class="profile-tag" style="background:{c}22;color:{c};border:1px solid {c}44;">{t}</span>'
                for t, c in p["tags"]
            ])
            stats_html = "".join([
                f'<div style="display:flex;justify-content:space-between;padding:0.3rem 0;'
                f'border-bottom:1px solid {COLORS["border"]};font-size:0.75rem;">'
                f'<span style="color:{COLORS["text_secondary"]};">{s[0]}</span>'
                f'<span style="color:{COLORS["text_primary"]};font-weight:600;font-family:DM Mono,monospace;">{s[1]}</span>'
                f'</div>'
                for s in p["stats"]
            ])
            st.markdown(f"""
            <div style="background:{p['bg']};border:1px solid {p['border']};
                        border-radius:12px;padding:1.25rem;height:100%;">
                <div style="font-size:1.75rem;margin-bottom:0.5rem;">{p['icon']}</div>
                <div style="font-size:0.95rem;font-weight:600;color:{COLORS['text_primary']};
                            margin-bottom:0.3rem;">{p['name']}</div>
                <div style="font-size:0.75rem;color:{COLORS['text_secondary']};
                            line-height:1.55;margin-bottom:0.8rem;">{p['desc']}</div>
                <div style="margin-bottom:0.8rem;">{tags_html}</div>
                <div>{stats_html}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Radar chart for profile comparison
    col_radar, col_table = st.columns([2, 3], gap="large")

    with col_radar:
        st.markdown("""<div class='section-label'>Profile Radar</div>
        <div class='section-title'>Attribute coverage per profile</div>""",
                    unsafe_allow_html=True)

        categories = ["High Income", "Large District", "Urban", "EPA 2023",
                      "High Poverty", "Rural", "Pollution Burden", "High Need"]
        cat_cols = ["high_income", "large_district", "urban", "epa_2023",
                    "high_poverty", "rural", "pollution_burden", "high_need"]

        # Compute radar values for each profile segment
        def get_radar(mask):
            sub = df_engaged[mask]
            return [sub[c].mean() * 100 for c in cat_cols] if len(sub) > 0 else [0] * len(cat_cols)

        urban_mask = df_engaged["urban"] == 1
        equity_mask = (df_engaged["high_poverty"] == 1) & (df_engaged["large_district"] == 1)
        rural_mask = df_engaged["rural"] == 1
        epa_mask = df_engaged["epa_2023"] == 1

        radar_data = [
            ("Urban Champion", urban_mask, COLORS["accent_blue"]),
            ("Equity-Driven", equity_mask, COLORS["accent_amber"]),
            ("Rural Pioneer", rural_mask, COLORS["accent_green"]),
            ("Policy Catalyst", epa_mask, COLORS["accent_purple"]),
        ]

        fig_radar = go.Figure()
        for name, mask, color in radar_data:
            vals = get_radar(mask)
            vals_closed = vals + [vals[0]]
            cats_closed = categories + [categories[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals_closed, theta=cats_closed,
                fill="toself", name=name,
                line=dict(color=color, width=1.5),
                fillcolor=f'rgba({int(color[1:3],16)}, {int(color[3:5],16)}, {int(color[5:7],16)}, 0.1)',
                opacity=0.85,
                hovertemplate=f"<b>{name}</b><br>%{{theta}}: %{{r:.1f}}%<extra></extra>",
            ))
        fig_radar.update_layout(
            **base_layout(height=340),
            polar=dict(
                bgcolor=PLOT_BG,
                radialaxis=dict(visible=True, range=[0, 100], tickfont_color=FONT_COLOR,
                                gridcolor=GRID_COLOR, ticksuffix="%", tickfont_size=9),
                angularaxis=dict(tickfont_color=COLORS["text_secondary"], gridcolor=GRID_COLOR,
                                 tickfont_size=10),
            ),
            legend=dict(font_color=FONT_COLOR, font_size=10, orientation="h",
                        x=0, y=-0.1, bgcolor="rgba(0,0,0,0)"),
            showlegend=True,
        )
        st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False})

    with col_table:
        st.markdown("""<div class='section-label'>Business Recommendations</div>
        <div class='section-title'>Strategic actions by profile</div>""",
                    unsafe_allow_html=True)

        reco_data = {
            "Profile": ["🏙️ Urban Champion", "⚖️ Equity-Driven", "🌾 Rural Pioneer", "📋 Policy Catalyst"],
            "Target": ["Metro districts, high income", "Large, high-poverty districts",
                       "Small rural districts", "EPA-prioritized districts"],
            "Key Driver": ["Economic capacity + visibility", "Equity & environmental justice",
                           "Pollution burden + EPA grant", "Federal policy incentives"],
            "Recommended Action": [
                "Accelerate fleet transition — showcase ROI",
                "Channel EPA/public subsidies — equity messaging",
                "Hybrid models — fleet-sharing programs",
                "Align grant applications — document outcomes",
            ],
            "Priority": ["High", "High", "Medium", "High"],
        }
        reco_df = pd.DataFrame(reco_data)

        # Styled table
        def priority_color(p):
            return COLORS["accent_green"] if p == "High" else COLORS["accent_amber"]

        rows_html = ""
        for _, row in reco_df.iterrows():
            pc = priority_color(row["Priority"])
            rows_html += f"""
            <tr style="border-bottom:1px solid {COLORS['border']};">
                <td style="padding:0.65rem 0.75rem;font-weight:500;white-space:nowrap;
                           color:{COLORS['text_primary']};">{row['Profile']}</td>
                <td style="padding:0.65rem 0.75rem;font-size:0.78rem;color:{COLORS['text_secondary']};">{row['Target']}</td>
                <td style="padding:0.65rem 0.75rem;font-size:0.78rem;color:{COLORS['text_secondary']};">{row['Key Driver']}</td>
                <td style="padding:0.65rem 0.75rem;font-size:0.78rem;color:{COLORS['text_primary']};">{row['Recommended Action']}</td>
                <td style="padding:0.65rem 0.75rem;text-align:center;">
                    <span style="background:{pc}22;color:{pc};border:1px solid {pc}44;
                                 border-radius:4px;padding:0.15rem 0.5rem;font-size:0.65rem;
                                 font-weight:600;letter-spacing:0.05em;">{row['Priority']}</span>
                </td>
            </tr>"""

        st.markdown(f"""
        <div style="overflow-x:auto;margin-top:0.25rem;">
        <table style="width:100%;border-collapse:collapse;font-size:0.82rem;">
            <thead>
                <tr style="border-bottom:2px solid {COLORS['border']};">
                    <th style="padding:0.5rem 0.75rem;text-align:left;font-size:0.68rem;
                               letter-spacing:0.08em;text-transform:uppercase;
                               color:{COLORS['text_muted']};font-weight:600;">Profile</th>
                    <th style="padding:0.5rem 0.75rem;text-align:left;font-size:0.68rem;
                               letter-spacing:0.08em;text-transform:uppercase;
                               color:{COLORS['text_muted']};font-weight:600;">Target Segment</th>
                    <th style="padding:0.5rem 0.75rem;text-align:left;font-size:0.68rem;
                               letter-spacing:0.08em;text-transform:uppercase;
                               color:{COLORS['text_muted']};font-weight:600;">Key Driver</th>
                    <th style="padding:0.5rem 0.75rem;text-align:left;font-size:0.68rem;
                               letter-spacing:0.08em;text-transform:uppercase;
                               color:{COLORS['text_muted']};font-weight:600;">Recommended Action</th>
                    <th style="padding:0.5rem 0.75rem;text-align:center;font-size:0.68rem;
                               letter-spacing:0.08em;text-transform:uppercase;
                               color:{COLORS['text_muted']};font-weight:600;">Priority</th>
                </tr>
            </thead>
            <tbody>{rows_html}</tbody>
        </table>
        </div>
        """, unsafe_allow_html=True)

        # Key findings
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:{COLORS['surface']};border:1px solid {COLORS['border']};
                    border-radius:10px;padding:1rem 1.25rem;">
            <div style="font-size:0.68rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;
                        color:{COLORS['text_muted']};margin-bottom:0.75rem;">Key Findings</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;">
                <div style="font-size:0.78rem;color:{COLORS['text_secondary']};padding:0.4rem 0.6rem;
                            background:{COLORS['surface_2']};border-radius:6px;border-left:2px solid {COLORS['accent_blue']};">
                    <b style="color:{COLORS['text_primary']};">Large districts dominate</b><br>
                    74.1% of engaged vs 45.2% overall (+28.9pp)
                </div>
                <div style="font-size:0.78rem;color:{COLORS['text_secondary']};padding:0.4rem 0.6rem;
                            background:{COLORS['surface_2']};border-radius:6px;border-left:2px solid {COLORS['accent_amber']};">
                    <b style="color:{COLORS['text_primary']};">High poverty overrepresented</b><br>
                    53.3% engaged vs 31.6% baseline (+21.7pp)
                </div>
                <div style="font-size:0.78rem;color:{COLORS['text_secondary']};padding:0.4rem 0.6rem;
                            background:{COLORS['surface_2']};border-radius:6px;border-left:2px solid {COLORS['accent_purple']};">
                    <b style="color:{COLORS['text_primary']};">EPA catalyzes engagement</b><br>
                    40.4% of engaged are EPA-prioritized (+13.2pp)
                </div>
                <div style="font-size:0.78rem;color:{COLORS['text_secondary']};padding:0.4rem 0.6rem;
                            background:{COLORS['surface_2']};border-radius:6px;border-left:2px solid {COLORS['accent_green']};">
                    <b style="color:{COLORS['text_primary']};">Rural districts lag behind</b><br>
                    35.2% engaged vs 42.6% overall (−7.4pp)
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)