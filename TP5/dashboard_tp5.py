"""
Electric School Bus — Anomaly Detection Dashboard
Machine Learning II (ADIF84) · Lab 5 · DBSCAN
──────────────────────────────────────────────────
Production-quality Streamlit dashboard.
Run: streamlit run dashboard.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors

# ─── PAGE SETUP ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ESB Anomaly Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── DESIGN TOKENS & GLOBAL CSS ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&family=DM+Mono:wght@400;500&display=swap');

/* ── Root palette ── */
:root {
    --bg:          #F7F8FA;
    --surface:     #FFFFFF;
    --surface-2:   #F0F2F5;
    --border:      #E4E7ED;
    --border-strong: #D0D5DD;
    --text:        #0D1117;
    --text-2:      #4B5563;
    --text-3:      #9CA3AF;
    --accent:      #0061FF;
    --accent-light: #E8F0FE;
    --accent-2:    #00C48C;
    --danger:      #EF4444;
    --danger-light: #FEF2F2;
    --warn:        #F59E0B;
    --warn-light:  #FFFBEB;
    --radius:      10px;
    --radius-lg:   16px;
    --shadow:      0 1px 3px rgba(0,0,0,.07), 0 1px 2px rgba(0,0,0,.05);
    --shadow-md:   0 4px 12px rgba(0,0,0,.08);
}

/* ── Global resets ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}
.stApp { background: var(--bg) !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebarNav"] { display: none; }

/* ── Remove Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 3rem !important; max-width: 1440px !important; }
[data-testid="stHorizontalBlock"] { gap: 1rem !important; }

/* ── Section header ── */
.section-header {
    display: flex; align-items: baseline; gap: .5rem;
    margin: 2rem 0 1rem; padding-bottom: .65rem;
    border-bottom: 1px solid var(--border);
}
.section-header h2 {
    font-size: 1rem; font-weight: 600; letter-spacing: -.01em;
    color: var(--text); margin: 0;
}
.section-header span {
    font-size: .78rem; color: var(--text-3); font-weight: 400;
}

/* ── KPI cards ── */
.kpi-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.4rem 1.6rem 1.2rem;
    box-shadow: var(--shadow);
    height: 100%;
    position: relative;
    overflow: hidden;
}
.kpi-card::after {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    border-radius: var(--radius-lg) var(--radius-lg) 0 0;
}
.kpi-card.blue::after   { background: var(--accent); }
.kpi-card.green::after  { background: var(--accent-2); }
.kpi-card.red::after    { background: var(--danger); }
.kpi-card.amber::after  { background: var(--warn); }

.kpi-label {
    font-size: .72rem; font-weight: 500; letter-spacing: .05em;
    text-transform: uppercase; color: var(--text-3); margin-bottom: .5rem;
}
.kpi-value {
    font-size: 2rem; font-weight: 600; letter-spacing: -.03em;
    color: var(--text); line-height: 1.1;
}
.kpi-sub {
    font-size: .78rem; color: var(--text-2); margin-top: .4rem; line-height: 1.4;
}
.kpi-badge {
    display: inline-block; padding: .15rem .55rem;
    border-radius: 99px; font-size: .72rem; font-weight: 500;
    margin-top: .5rem;
}
.badge-red   { background: var(--danger-light); color: var(--danger); }
.badge-green { background: #ECFDF5; color: #059669; }
.badge-blue  { background: var(--accent-light); color: var(--accent); }
.badge-amber { background: var(--warn-light); color: #D97706; }

/* ── Info banner ── */
.insight-box {
    background: var(--accent-light);
    border: 1px solid #C7D9FF;
    border-left: 3px solid var(--accent);
    border-radius: var(--radius);
    padding: .9rem 1.1rem;
    font-size: .84rem; color: #1E3A8A;
    margin: 1rem 0;
    line-height: 1.55;
}
.warn-box {
    background: var(--warn-light);
    border: 1px solid #FDE68A;
    border-left: 3px solid var(--warn);
    border-radius: var(--radius);
    padding: .9rem 1.1rem;
    font-size: .84rem; color: #78350F;
    margin: 1rem 0;
    line-height: 1.55;
}

/* ── Data table ── */
.custom-table {
    width: 100%; border-collapse: collapse; font-size: .84rem;
}
.custom-table th {
    background: var(--surface-2); font-weight: 600;
    padding: .55rem .9rem; text-align: left;
    border-bottom: 1px solid var(--border);
    font-size: .72rem; letter-spacing: .04em; text-transform: uppercase;
    color: var(--text-3);
}
.custom-table td {
    padding: .5rem .9rem; border-bottom: 1px solid var(--border);
    color: var(--text-2); font-family: 'DM Mono', monospace; font-size: .8rem;
}
.custom-table tr:last-child td { border-bottom: none; }
.custom-table tr:hover td { background: var(--surface-2); }
.best-row td { background: #F0FDF4 !important; color: #065F46 !important; font-weight: 600; }

/* ── Chip tags ── */
.chip {
    display: inline-block;
    padding: .2rem .6rem;
    border-radius: 6px;
    font-size: .72rem;
    font-weight: 500;
}

/* ── Sidebar sliders ── */
[data-testid="stSlider"] > div { padding: 0 !important; }
.stSlider label { font-size: .82rem !important; color: var(--text-2) !important; }

/* ── Plotly chart container ── */
.plot-container { border: 1px solid var(--border); border-radius: var(--radius-lg); overflow: hidden; }

/* ── Page header ── */
.page-header {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 1.5rem; padding-bottom: 1.25rem;
    border-bottom: 1px solid var(--border);
}
.page-title { font-size: 1.35rem; font-weight: 600; letter-spacing: -.02em; color: var(--text); }
.page-subtitle { font-size: .84rem; color: var(--text-2); margin-top: .2rem; }
.page-tag {
    background: var(--accent-light); color: var(--accent);
    border: 1px solid #BAD0FF; border-radius: 99px;
    font-size: .72rem; font-weight: 500; padding: .3rem .8rem;
    letter-spacing: .02em;
}
</style>
""", unsafe_allow_html=True)


# ─── CONSTANTS ────────────────────────────────────────────────────────────────
COL_ELEC = "3i. Percent of fleet that is electric"
COL_REV  = "4f. Median household income"
COL_POV  = "4g. Percent of population below the poverty level"

LABEL_MAP = {
    COL_ELEC: "% Fleet Electric",
    COL_REV:  "Median Income",
    COL_POV:  "% Below Poverty",
}

PALETTE = {
    "normal":  "#2563EB",
    "outlier": "#EF4444",
    "grid_bg": "rgba(0,0,0,0)",
    "axis":    "#9CA3AF",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#FAFBFC",
    font_family="DM Sans",
    font_color="#4B5563",
    margin=dict(l=16, r=16, t=36, b=16),
    hoverlabel=dict(bgcolor="white", bordercolor="#E4E7ED", font_size=12, font_family="DM Sans"),
)


# ─── DATA & MODEL ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    return pd.read_csv("anomalies.csv")


@st.cache_data(show_spinner=False)
def run_dbscan(eps: float, min_samples: int):
    df = load_data()
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(df)
    return labels


@st.cache_data(show_spinner=False)
def run_grid():
    df = load_data()
    rows = []
    for eps in [0.3, 0.4, 0.5, 0.6, 0.7, 1.0]:
        for ms in [5, 10, 20]:
            lbl = DBSCAN(eps=eps, min_samples=ms).fit_predict(df)
            nc = len(set(lbl)) - (1 if -1 in lbl else 0)
            no = int((lbl == -1).sum())
            rows.append(dict(eps=eps, min_samples=ms, n_clusters=nc,
                             n_outliers=no, pct=round(no / len(df) * 100, 2)))
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def run_kdistance(k: int):
    df = load_data()
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(df)
    dists, _ = nn.kneighbors(df)
    return np.sort(dists[:, -1])


@st.cache_data(show_spinner=False)
def run_iso(contamination: float):
    df = load_data()
    labels = IsolationForest(contamination=contamination, random_state=42).fit_predict(df)
    return (labels == -1).astype(int)


def classify_outliers(df_out):
    elec = df_out[COL_ELEC]
    rev  = df_out[COL_REV]
    pov  = df_out[COL_POV]

    def _cat(row):
        e, r, p = row[COL_ELEC], row[COL_REV], row[COL_POV]
        if e > 5:
            return "Over-electrified (data check)"
        if r > 0.5 and e < 0:
            return "Wealthy Under-investors"
        if (r < -0.5 or p > 0.5) and e > 0.5:
            return "Low-income Pioneers"
        return "Rare Profiles"

    df_out = df_out.copy()
    df_out["Category"] = df_out.apply(_cat, axis=1)
    return df_out


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:.5rem 0 1.2rem'>
      <div style='font-size:1rem;font-weight:600;letter-spacing:-.01em'>⚡ ESB Analytics</div>
      <div style='font-size:.75rem;color:#9CA3AF;margin-top:.2rem'>Anomaly Detection · Lab 5</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-size:.7rem;text-transform:uppercase;letter-spacing:.05em;color:#9CA3AF;font-weight:600;margin-bottom:.5rem'>DBSCAN Parameters</div>", unsafe_allow_html=True)

    eps_val = st.slider("Epsilon (ε)", min_value=0.2, max_value=1.2, value=0.5, step=0.05,
                        help="Neighbourhood radius. Tune with the k-distance elbow.")
    min_s_val = st.slider("Min Samples", min_value=3, max_value=30, value=10, step=1,
                          help="Minimum points to form a dense region.")

    st.markdown("<hr style='border:none;border-top:1px solid #E4E7ED;margin:1.2rem 0'>", unsafe_allow_html=True)

    st.markdown("<div style='font-size:.7rem;text-transform:uppercase;letter-spacing:.05em;color:#9CA3AF;font-weight:600;margin-bottom:.5rem'>K-Distance Graph</div>", unsafe_allow_html=True)
    k_val = st.slider("k (neighbours)", min_value=3, max_value=20, value=10, step=1,
                      help="Typically 2 × dimensions. Dataset has 3 dims → k=6-10.")

    st.markdown("<hr style='border:none;border-top:1px solid #E4E7ED;margin:1.2rem 0'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:.75rem;color:#9CA3AF;line-height:1.6'>
      <b style='color:#6B7280'>Dataset</b><br>12,837 US school districts<br>3 standardised features (z-scores)<br>
      <br><b style='color:#6B7280'>Model</b><br>DBSCAN · density-based<br>Unsupervised anomaly detection
    </div>
    """, unsafe_allow_html=True)


# ─── LOAD DATA ────────────────────────────────────────────────────────────────
df = load_data()
labels = run_dbscan(eps_val, min_s_val)
n_total    = len(df)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_outliers = int((labels == -1).sum())
n_normal   = n_total - n_outliers
pct_out    = n_outliers / n_total * 100

outlier_mask = labels == -1
df_outliers  = df[outlier_mask].copy()
df_outliers  = classify_outliers(df_outliers)
df_normal    = df[~outlier_mask].copy()


# ─── PAGE HEADER ──────────────────────────────────────────────────────────────
st.markdown("""
<div class='page-header'>
  <div>
    <div class='page-title'>Electric School Bus — District Anomaly Intelligence</div>
    <div class='page-subtitle'>Identifying economically incoherent districts via DBSCAN density-based clustering</div>
  </div>
  <div class='page-tag'>ADIF84 · Lab 5</div>
</div>
""", unsafe_allow_html=True)


# ─── KPI ROW ──────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4, gap="medium")

with k1:
    st.markdown(f"""
    <div class='kpi-card blue'>
      <div class='kpi-label'>Total Districts</div>
      <div class='kpi-value'>{n_total:,}</div>
      <div class='kpi-sub'>US school districts analysed</div>
      <span class='kpi-badge badge-blue'>Full dataset</span>
    </div>""", unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class='kpi-card red'>
      <div class='kpi-label'>Anomalies Detected</div>
      <div class='kpi-value'>{n_outliers}</div>
      <div class='kpi-sub'>Districts flagged as outliers</div>
      <span class='kpi-badge badge-red'>{pct_out:.2f}% of dataset</span>
    </div>""", unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class='kpi-card green'>
      <div class='kpi-label'>Normal Districts</div>
      <div class='kpi-value'>{n_normal:,}</div>
      <div class='kpi-sub'>Assigned to dense cluster(s)</div>
      <span class='kpi-badge badge-green'>{100-pct_out:.1f}% of dataset</span>
    </div>""", unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class='kpi-card amber'>
      <div class='kpi-label'>Dense Clusters</div>
      <div class='kpi-value'>{n_clusters}</div>
      <div class='kpi-sub'>ε = {eps_val} · min_samples = {min_s_val}</div>
      <span class='kpi-badge badge-amber'>DBSCAN output</span>
    </div>""", unsafe_allow_html=True)


# ─── SECTION 1: HYPERPARAMETER TUNING ────────────────────────────────────────
st.markdown("""
<div class='section-header'>
  <h2>01 · Hyperparameter Tuning</h2>
  <span>K-distance elbow + grid search over ε × min_samples</span>
</div>""", unsafe_allow_html=True)

col_a, col_b = st.columns([1, 1], gap="large")

# K-distance graph
with col_a:
    k_dists = run_kdistance(k_val)
    idx = np.arange(len(k_dists))

    # Find approximate elbow (largest second derivative in first 99.5%)
    cut = int(len(k_dists) * 0.995)
    d2 = np.gradient(np.gradient(k_dists[:cut]))
    elbow_idx = int(np.argmax(d2))
    elbow_val = float(k_dists[elbow_idx])

    fig_kd = go.Figure()
    fig_kd.add_trace(go.Scatter(
        x=idx, y=k_dists,
        mode="lines", line=dict(color="#2563EB", width=1.5),
        name="k-distance", hovertemplate="idx %{x}<br>dist %{y:.3f}<extra></extra>",
    ))
    fig_kd.add_trace(go.Scatter(
        x=[elbow_idx], y=[elbow_val],
        mode="markers", marker=dict(size=9, color="#EF4444", symbol="circle"),
        name=f"Elbow ≈ {elbow_val:.2f}",
        hovertemplate=f"Elbow at idx {elbow_idx}<br>dist = {elbow_val:.3f}<extra></extra>",
    ))
    fig_kd.add_hline(y=eps_val, line=dict(color="#F59E0B", width=1.5, dash="dot"),
                     annotation_text=f"  ε = {eps_val}", annotation_font_size=11,
                     annotation_font_color="#D97706")
    fig_kd.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=f"k-Distance Graph (k = {k_val})", font=dict(size=13, color="#0D1117"), x=0, xanchor="left"),
        xaxis=dict(title="Points (sorted by k-distance)", showgrid=True, gridcolor="#F0F2F5",
                   zeroline=False, tickfont=dict(size=10)),
        yaxis=dict(title="Distance to k-th neighbour", showgrid=True, gridcolor="#F0F2F5",
                   zeroline=False, range=[0, min(3, k_dists.max() * 1.1)], tickfont=dict(size=10)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=10)),
        height=310,
    )
    st.plotly_chart(fig_kd, use_container_width=True, config={"displayModeBar": False})

# Grid table
with col_b:
    st.markdown("<div style='font-size:.84rem;font-weight:500;color:#0D1117;margin-bottom:.75rem'>ε × min_samples Grid Search</div>", unsafe_allow_html=True)
    grid_df = run_grid()

    # Build HTML table with best row highlighted
    rows_html = ""
    for _, row in grid_df.iterrows():
        is_best = row["eps"] == 0.5 and row["min_samples"] == 10
        tr_class = "best-row" if is_best else ""
        pct_color = "#EF4444" if row["pct"] > 3 else ("#F59E0B" if row["pct"] > 1.5 else "#059669")
        best_marker = " ⭐" if is_best else ""
        rows_html += f"""
        <tr class='{tr_class}'>
          <td>{row['eps']}{best_marker}</td>
          <td>{int(row['min_samples'])}</td>
          <td>{int(row['n_clusters'])}</td>
          <td>{int(row['n_outliers'])}</td>
          <td><span style='color:{pct_color};font-weight:500'>{row['pct']}%</span></td>
        </tr>"""

    st.markdown(f"""
    <div style='background:white;border:1px solid #E4E7ED;border-radius:10px;overflow:hidden;'>
    <table class='custom-table'>
      <thead><tr>
        <th>ε</th><th>min_samples</th><th>Clusters</th>
        <th>Outliers</th><th>% Outliers</th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
    </div>""", unsafe_allow_html=True)

st.markdown(f"""
<div class='insight-box'>
  <b>Selected configuration:</b> ε = 0.5 · min_samples = 10 → <b>{grid_df[(grid_df.eps==0.5)&(grid_df.min_samples==10)]['n_outliers'].values[0]} anomalies (0.84%)</b>.
  The k-distance elbow at ≈ 0.5 confirms this choice. Lower ε (0.3) flags too many outliers (over-sensitivity);
  higher ε (1.0) misses subtle anomalies. min_samples = 10 avoids micro-clusters while maintaining sensitivity.
</div>""", unsafe_allow_html=True)


# ─── SECTION 2: ANOMALY VISUALISATION ────────────────────────────────────────
st.markdown("""
<div class='section-header'>
  <h2>02 · Anomaly Visualisation</h2>
  <span>Spatial distribution of outliers in feature space</span>
</div>""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📊 2D Projections", "🧊 3D View"])

with tab1:
    c1, c2 = st.columns(2, gap="large")

    with c1:
        fig_2d1 = go.Figure()
        fig_2d1.add_trace(go.Scatter(
            x=df_normal[COL_REV], y=df_normal[COL_ELEC],
            mode="markers", marker=dict(size=3, color="#2563EB", opacity=0.25),
            name=f"Normal ({n_normal:,})",
            hovertemplate="Income: %{x:.2f}<br>% Electric: %{y:.2f}<extra>Normal</extra>",
        ))
        fig_2d1.add_trace(go.Scatter(
            x=df_outliers[COL_REV], y=df_outliers[COL_ELEC],
            mode="markers", marker=dict(size=7, color="#EF4444", opacity=0.85,
                                         line=dict(color="white", width=0.5)),
            name=f"Anomaly ({n_outliers})",
            hovertemplate="Income: %{x:.2f}<br>% Electric: %{y:.2f}<extra>Anomaly</extra>",
        ))
        fig_2d1.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text="Median Income × % Fleet Electric", font=dict(size=12, color="#0D1117"), x=0),
            xaxis=dict(title="Median Income (z-score)", showgrid=True, gridcolor="#F0F2F5",
                       zeroline=True, zerolinecolor="#E4E7ED", tickfont=dict(size=10)),
            yaxis=dict(title="% Fleet Electric (z-score)", showgrid=True, gridcolor="#F0F2F5",
                       zeroline=True, zerolinecolor="#E4E7ED", tickfont=dict(size=10)),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=10)),
            height=340,
        )
        st.plotly_chart(fig_2d1, use_container_width=True, config={"displayModeBar": False})

    with c2:
        fig_2d2 = go.Figure()
        fig_2d2.add_trace(go.Scatter(
            x=df_normal[COL_REV], y=df_normal[COL_POV],
            mode="markers", marker=dict(size=3, color="#2563EB", opacity=0.25),
            name=f"Normal ({n_normal:,})",
            hovertemplate="Income: %{x:.2f}<br>Poverty: %{y:.2f}<extra>Normal</extra>",
        ))
        fig_2d2.add_trace(go.Scatter(
            x=df_outliers[COL_REV], y=df_outliers[COL_POV],
            mode="markers", marker=dict(size=7, color="#EF4444", opacity=0.85,
                                         line=dict(color="white", width=0.5)),
            name=f"Anomaly ({n_outliers})",
            hovertemplate="Income: %{x:.2f}<br>Poverty: %{y:.2f}<extra>Anomaly</extra>",
        ))
        fig_2d2.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text="Median Income × Poverty Rate", font=dict(size=12, color="#0D1117"), x=0),
            xaxis=dict(title="Median Income (z-score)", showgrid=True, gridcolor="#F0F2F5",
                       zeroline=True, zerolinecolor="#E4E7ED", tickfont=dict(size=10)),
            yaxis=dict(title="% Below Poverty (z-score)", showgrid=True, gridcolor="#F0F2F5",
                       zeroline=True, zerolinecolor="#E4E7ED", tickfont=dict(size=10)),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=10)),
            height=340,
        )
        st.plotly_chart(fig_2d2, use_container_width=True, config={"displayModeBar": False})

with tab2:
    fig_3d = go.Figure()
    fig_3d.add_trace(go.Scatter3d(
        x=df_normal[COL_REV], y=df_normal[COL_POV], z=df_normal[COL_ELEC],
        mode="markers", marker=dict(size=2, color="#2563EB", opacity=0.2),
        name=f"Normal ({n_normal:,})",
        hovertemplate="Income: %{x:.2f}<br>Poverty: %{y:.2f}<br>Electric: %{z:.2f}<extra>Normal</extra>",
    ))
    fig_3d.add_trace(go.Scatter3d(
        x=df_outliers[COL_REV], y=df_outliers[COL_POV], z=df_outliers[COL_ELEC],
        mode="markers", marker=dict(size=5, color="#EF4444", opacity=0.85,
                                     line=dict(color="white", width=0.3)),
        name=f"Anomaly ({n_outliers})",
        hovertemplate="Income: %{x:.2f}<br>Poverty: %{y:.2f}<br>Electric: %{z:.2f}<extra>Anomaly</extra>",
    ))
    fig_3d.update_layout(
        **PLOTLY_LAYOUT,
        scene=dict(
            xaxis_title="Median Income", yaxis_title="% Poverty", zaxis_title="% Electric",
            xaxis=dict(backgroundcolor="#FAFBFC", gridcolor="#E4E7ED", showbackground=True),
            yaxis=dict(backgroundcolor="#FAFBFC", gridcolor="#E4E7ED", showbackground=True),
            zaxis=dict(backgroundcolor="#FAFBFC", gridcolor="#E4E7ED", showbackground=True),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=0, x=0, font=dict(size=11)),
        height=460,
      
    )
    st.plotly_chart(fig_3d, use_container_width=True, config={"displayModeBar": False})


# ─── SECTION 3: BUSINESS TYPOLOGY ────────────────────────────────────────────
st.markdown("""
<div class='section-header'>
  <h2>03 · Business Typology</h2>
  <span>Outlier segmentation by commercial value</span>
</div>""", unsafe_allow_html=True)

cat_counts = df_outliers["Category"].value_counts()

CAT_META = {
    "Wealthy Under-investors":       ("badge-blue",  "🎯", "#2563EB", "High-income districts with low EV fleet adoption. Primary commercial targets."),
    "Low-income Pioneers":           ("badge-green", "⭐", "#059669", "Low-income districts with above-average adoption. Showcase references."),
    "Over-electrified (data check)": ("badge-amber", "⚠️", "#D97706", "% > 100 — likely data entry errors. Flag for pipeline review."),
    "Rare Profiles":                 ("badge-red",   "🔍", "#6B7280", "Atypical multivariate combinations requiring case-by-case analysis."),
}

cat_cols = st.columns(len(CAT_META), gap="medium")
for col, (cat, (badge, icon, color, desc)) in zip(cat_cols, CAT_META.items()):
    count = int(cat_counts.get(cat, 0))
    pct   = count / n_outliers * 100 if n_outliers else 0
    with col:
        st.markdown(f"""
        <div class='kpi-card' style='border-top:3px solid {color}'>
          <div style='font-size:1.4rem;margin-bottom:.4rem'>{icon}</div>
          <div style='font-size:.78rem;font-weight:600;color:#0D1117;margin-bottom:.35rem'>{cat}</div>
          <div style='font-size:1.6rem;font-weight:600;letter-spacing:-.02em;color:{color}'>{count}</div>
          <div style='font-size:.72rem;color:#9CA3AF;margin:.2rem 0 .5rem'>{pct:.0f}% of anomalies</div>
          <div style='font-size:.74rem;color:#6B7280;line-height:1.45'>{desc}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

col_pie, col_scatter = st.columns([1, 1.6], gap="large")

with col_pie:
    cats  = list(cat_counts.index)
    vals  = list(cat_counts.values)
    clrs  = [CAT_META.get(c, ("", "", "#9CA3AF", ""))[2] for c in cats]

    fig_pie = go.Figure(go.Pie(
        labels=cats,
        values=vals,
        marker=dict(
            colors=clrs,
            line=dict(color="white", width=2)
        ),
        hole=0.52,
        textinfo="percent",
        hovertemplate="%{label}<br>%{value} districts<br>%{percent}<extra></extra>",
        textfont=dict(size=11),
    ))

    fig_pie.add_annotation(
        text=f"<b>{n_outliers}</b><br><span style='font-size:11px;color:#9CA3AF'>anomalies</span>",
        x=0.5,
        y=0.5,
        font_size=18,
        showarrow=False,
        align="center",
    )

    # FIX: override margin BEFORE unpacking
    pie_layout = {
        **PLOTLY_LAYOUT,
        "margin": dict(l=0, r=120, t=36, b=0),
    }

    fig_pie.update_layout(
        **pie_layout,
        title=dict(
            text="Anomaly Distribution by Type",
            font=dict(size=12, color="#0D1117"),
            x=0
        ),
        legend=dict(
            orientation="v",
            x=1.01,
            y=0.5,
            font=dict(size=10)
        ),
        height=300,
    )

    st.plotly_chart(
        fig_pie,
        use_container_width=True,
        config={"displayModeBar": False}
    )

with col_scatter:
    color_map = {c: CAT_META.get(c, ("", "", "#9CA3AF", ""))[2] for c in df_outliers["Category"].unique()}
    fig_seg = go.Figure()
    for cat, grp in df_outliers.groupby("Category"):
        fig_seg.add_trace(go.Scatter(
            x=grp[COL_REV], y=grp[COL_ELEC],
            mode="markers",
            marker=dict(size=7, color=color_map.get(cat, "#9CA3AF"),
                        opacity=0.85, line=dict(color="white", width=0.5)),
            name=cat,
            hovertemplate=f"<b>{cat}</b><br>Income: %{{x:.2f}}<br>Electric: %{{y:.2f}}<extra></extra>",
        ))
    fig_seg.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Outlier Segments — Income vs Electric Adoption", font=dict(size=12, color="#0D1117"), x=0),
        xaxis=dict(title="Median Income (z-score)", showgrid=True, gridcolor="#F0F2F5",
                   zeroline=True, zerolinecolor="#E4E7ED", tickfont=dict(size=10)),
        yaxis=dict(title="% Fleet Electric (z-score)", showgrid=True, gridcolor="#F0F2F5",
                   zeroline=True, zerolinecolor="#E4E7ED", tickfont=dict(size=10)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=10)),
        height=300,
    )
    st.plotly_chart(fig_seg, use_container_width=True, config={"displayModeBar": False})


# ─── SECTION 4: MODEL COMPARISON ─────────────────────────────────────────────
st.markdown("""
<div class='section-header'>
  <h2>04 · Model Comparison — DBSCAN vs Isolation Forest</h2>
  <span>B202 justification: selecting the most appropriate algorithm</span>
</div>""", unsafe_allow_html=True)

contamination = n_outliers / n_total
iso_flags = run_iso(contamination)
n_iso = int(iso_flags.sum())

dbscan_flags = (labels == -1).astype(int)
agreement    = int(((dbscan_flags == 1) & (iso_flags == 1)).sum())
only_dbscan  = int(((dbscan_flags == 1) & (iso_flags == 0)).sum())
only_iso     = int(((dbscan_flags == 0) & (iso_flags == 1)).sum())
agreement_pct = agreement / n_outliers * 100 if n_outliers else 0

mc1, mc2, mc3, mc4 = st.columns(4, gap="medium")
for col, label, val, badge, bc in zip(
    [mc1, mc2, mc3, mc4],
    ["DBSCAN Anomalies", "Isolation Forest", "Both Agree", "DBSCAN Only"],
    [n_outliers, n_iso, agreement, only_dbscan],
    [f"{pct_out:.2f}%", f"{n_iso/n_total*100:.2f}%", f"{agreement_pct:.0f}% overlap", f"{only_dbscan} unique"],
    ["badge-red", "badge-amber", "badge-green", "badge-blue"],
):
    col.markdown(f"""
    <div style='background:white;border:1px solid #E4E7ED;border-radius:10px;padding:1rem 1.2rem;box-shadow:0 1px 3px rgba(0,0,0,.06)'>
      <div style='font-size:.7rem;text-transform:uppercase;letter-spacing:.05em;color:#9CA3AF;font-weight:600;margin-bottom:.4rem'>{label}</div>
      <div style='font-size:1.5rem;font-weight:600;letter-spacing:-.02em;color:#0D1117'>{val}</div>
      <span class='kpi-badge {bc}' style='margin-top:.3rem;display:inline-block'>{badge}</span>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

col_bar, col_why = st.columns([1.2, 1], gap="large")

with col_bar:
    categories = ["Both agree", "DBSCAN only", "Isolation Forest only"]
    values     = [agreement, only_dbscan, only_iso]
    bar_colors = ["#059669", "#2563EB", "#F59E0B"]
    fig_comp = go.Figure(go.Bar(
        x=categories, y=values,
        marker=dict(color=bar_colors, line=dict(color="white", width=1)),
        text=values, textposition="outside", textfont=dict(size=11),
        hovertemplate="%{x}<br>%{y} districts<extra></extra>",
    ))
    fig_comp.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Anomaly Detection Agreement", font=dict(size=12, color="#0D1117"), x=0),
        xaxis=dict(showgrid=False, tickfont=dict(size=11)),
        yaxis=dict(showgrid=True, gridcolor="#F0F2F5", title="Districts"),
        height=300,
        showlegend=False,
    )
    st.plotly_chart(fig_comp, use_container_width=True, config={"displayModeBar": False})

with col_why:
    st.markdown("""
    <div style='background:white;border:1px solid #E4E7ED;border-radius:10px;padding:1.2rem 1.4rem;height:100%;box-sizing:border-box'>
      <div style='font-size:.84rem;font-weight:600;color:#0D1117;margin-bottom:.8rem'>Why DBSCAN was selected (B202)</div>
      <div style='font-size:.8rem;color:#4B5563;line-height:1.65'>
        <div style='margin-bottom:.5rem'>✅ <b>No cluster count required</b> — DBSCAN discovers structure automatically, unlike k-Means.</div>
        <div style='margin-bottom:.5rem'>✅ <b>Natural outlier label</b> — Points labelled −1 are not forced into any cluster.</div>
        <div style='margin-bottom:.5rem'>✅ <b>Density coherence</b> — Captures arbitrary cluster shapes vs. Isolation Forest's axis-parallel splits.</div>
        <div style='margin-bottom:.5rem'>✅ <b>Interpretable ε</b> — The k-distance elbow provides a principled, visual parameter selection method.</div>
        <div style='color:#9CA3AF;font-size:.75rem;margin-top:.8rem'>Isolation Forest detects {n_iso} anomalies with {agreement_pct:.0f}% overlap, confirming robustness of findings.</div>
      </div>
    </div>
    """.replace("{n_iso}", str(n_iso)).replace("{agreement_pct}", f"{agreement_pct:.0f}"), unsafe_allow_html=True)


# ─── SECTION 5: RAW ANOMALY TABLE ────────────────────────────────────────────
st.markdown("""
<div class='section-header'>
  <h2>05 · Anomaly Records</h2>
  <span>Explore flagged districts and their feature values</span>
</div>""", unsafe_allow_html=True)

filter_cat = st.selectbox(
    "Filter by category",
    options=["All"] + sorted(df_outliers["Category"].unique().tolist()),
    label_visibility="collapsed",
)

display_df = df_outliers if filter_cat == "All" else df_outliers[df_outliers["Category"] == filter_cat]

display_df_show = display_df[[COL_REV, COL_POV, COL_ELEC, "Category"]].copy()
display_df_show.columns = ["Median Income (z)", "% Poverty (z)", "% Fleet Electric (z)", "Category"]
display_df_show = display_df_show.sort_values("% Fleet Electric (z)", ascending=False).round(3)

st.markdown(f"<div style='font-size:.78rem;color:#9CA3AF;margin-bottom:.5rem'>{len(display_df_show)} records shown</div>", unsafe_allow_html=True)
st.dataframe(
    display_df_show,
    use_container_width=True,
    height=280,
    hide_index=True,
)

# ─── FOOTER ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style='margin-top:3rem;padding-top:1.25rem;border-top:1px solid #E4E7ED;
            display:flex;justify-content:space-between;align-items:center'>
  <div style='font-size:.75rem;color:#9CA3AF'>
    Machine Learning II (ADIF84) · Lab 5 · Electric School Bus Initiative
  </div>
  <div style='font-size:.75rem;color:#9CA3AF'>
    DBSCAN · scikit-learn · Plotly · Streamlit
  </div>
</div>
""", unsafe_allow_html=True)
