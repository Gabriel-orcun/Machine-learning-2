import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="ESB Clustering Dashboard",
    page_icon="🎯",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* APP BACKGROUND */
.stApp {
    background:
        radial-gradient(circle at top left, #182235 0%, #0B1220 45%),
        #0B1220;
    color: white;
}

/* REMOVE STREAMLIT TOP BAR */
header {
    visibility: hidden;
}

[data-testid="stToolbar"] {
    display: none;
}

[data-testid="stDecoration"] {
    display: none;
}

[data-testid="stHeader"] {
    background: transparent;
}

main .block-container {
    padding-top: 1rem;
    max-width: 1450px;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: #0F1726;
    border-right: 1px solid rgba(255,255,255,0.05);
}

/* TITLES */
.main-title {
    font-size: 48px;
    font-weight: 700;
    color: white;
    letter-spacing: -1px;
    margin-bottom: 0;
}

.subtitle {
    color: #8EA0B5;
    font-size: 15px;
    margin-top: 6px;
    margin-bottom: 30px;
}

/* METRIC CARDS */
.metric-card {
    background: rgba(20,30,48,0.85);
    border: 1px solid rgba(255,255,255,0.06);
    padding: 24px;
    border-radius: 22px;
    backdrop-filter: blur(12px);
}

.metric-label {
    color: #8FA2B8;
    font-size: 13px;
    margin-bottom: 10px;
}

.metric-value {
    color: white;
    font-size: 34px;
    font-weight: 700;
}

/* SECTION TITLE */
.section-title {
    color: white;
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 18px;
}

/* EXPANDER */
.streamlit-expanderHeader {
    color: white !important;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# COLORS
# =========================================================
BG = "#0B1220"
GRID = "#233044"
TEXT = "#B7C4D4"

COLORS = [
    "#A371F7",
    "#00C896",
    "#F4B740",
    "#4EA1FF",
    "#FF6B81",
    "#8DD96F"
]

NAMES = [
    "Green Leaders",
    "Affluent Areas",
    "Mega Districts",
    "Support Priority",
    "Growth Markets",
    "Emerging Zones"
]

# =========================================================
# HELPERS
# =========================================================
def rgba(hex_color, alpha=0.2):

    hex_color = hex_color.replace("#", "")

    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    return f"rgba({r},{g},{b},{alpha})"


def chart_layout(height=400):

    return dict(
        height=height,
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color=TEXT),
        margin=dict(l=20, r=20, t=30, b=20)
    )

# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data
def load_data(path):

    df = pd.read_csv(path)

    df.columns = [
        "electric_pct",
        "students",
        "lunch_pct",
        "poverty_pct"
    ]

    return df

# =========================================================
# COMPUTE MODEL
# =========================================================
@st.cache_data
def compute_model(X, k):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X)

    centers = model.cluster_centers_

    sil = silhouette_score(X, labels)

    pca = PCA(n_components=2)

    coords = pca.fit_transform(X)

    return (
        labels,
        centers,
        sil,
        coords,
        pca.explained_variance_ratio_
    )

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:

    st.markdown("## 🎯 ESB Clustering")

    file_path = st.text_input(
        "Dataset",
        value="clustering.csv"
    )

    k = st.slider(
        "Clusters (k)",
        2,
        8,
        4
    )

    st.markdown("---")

    st.markdown("""
    ### Dashboard Features

    - K-Means clustering
    - PCA visualization
    - Cluster profiling
    - Distribution analysis
    """)

# =========================================================
# LOAD
# =========================================================
try:
    df_raw = load_data(file_path)

except FileNotFoundError:

    st.error(f"File not found: {file_path}")
    st.stop()

X = df_raw.values

# =========================================================
# MODEL
# =========================================================
labels, centers, sil, coords, pca_var = compute_model(X, k)

df = df_raw.copy()

df["cluster"] = labels
df["pc1"] = coords[:, 0]
df["pc2"] = coords[:, 1]

cluster_sizes = df["cluster"].value_counts().sort_index()

# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div class='main-title'>
    ESB District Segmentation
</div>

<div class='subtitle'>
    Interactive K-Means dashboard for electric school bus analysis
</div>
""", unsafe_allow_html=True)

# =========================================================
# METRICS
# =========================================================
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-label'>Districts</div>
        <div class='metric-value'>{len(df):,}</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-label'>Clusters</div>
        <div class='metric-value'>{k}</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-label'>Silhouette</div>
        <div class='metric-value'>{sil:.3f}</div>
    </div>
    """, unsafe_allow_html=True)

with m4:

    variance = (pca_var[0] + pca_var[1]) * 100

    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-label'>PCA Explained</div>
        <div class='metric-value'>{variance:.0f}%</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =========================================================
# CHARTS
# =========================================================
c1, c2 = st.columns([1.4, 1])

# =========================================================
# PCA SCATTER
# =========================================================
with c1:

    st.markdown(
        "<div class='section-title'>PCA Cluster Projection</div>",
        unsafe_allow_html=True
    )

    fig = go.Figure()

    for i in range(k):

        mask = df["cluster"] == i

        fig.add_trace(go.Scatter(
            x=df.loc[mask, "pc1"],
            y=df.loc[mask, "pc2"],
            mode="markers",
            name=NAMES[i],
            marker=dict(
                size=7,
                color=COLORS[i],
                opacity=0.75
            ),
            hovertemplate=
            "<b>%{text}</b><br>"
            "PC1: %{x:.2f}<br>"
            "PC2: %{y:.2f}<extra></extra>",
            text=[NAMES[i]] * mask.sum()
        ))

    fig.update_layout(
        **chart_layout(450),

        xaxis=dict(
            title=f"PC1 ({pca_var[0]*100:.0f}%)",
            gridcolor=GRID
        ),

        yaxis=dict(
            title=f"PC2 ({pca_var[1]*100:.0f}%)",
            gridcolor=GRID
        ),

        legend=dict(
            bgcolor="rgba(0,0,0,0)"
        )
    )

    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# RADAR CHART
# =========================================================
with c2:

    st.markdown(
        "<div class='section-title'>Cluster Profiles</div>",
        unsafe_allow_html=True
    )

    radar_labels = [
        "Electric %",
        "Students",
        "Free Lunch",
        "Poverty"
    ]

    radar = go.Figure()

    for i in range(k):

        values = list(centers[i]) + [centers[i][0]]

        radar.add_trace(go.Scatterpolar(
            r=values,
            theta=radar_labels + [radar_labels[0]],
            fill="toself",
            name=NAMES[i],
            line=dict(
                color=COLORS[i],
                width=2
            ),
            fillcolor=rgba(COLORS[i], 0.15)
        ))

    radar.update_layout(
        **chart_layout(450),

        polar=dict(
            bgcolor=BG,

            radialaxis=dict(
                visible=True,
                gridcolor=GRID
            ),

            angularaxis=dict(
                gridcolor=GRID
            )
        ),

        legend=dict(
            orientation="h",
            y=-0.12
        )
    )

    st.plotly_chart(radar, use_container_width=True)

# =========================================================
# CLUSTER INSIGHTS
# =========================================================
st.markdown("<br>", unsafe_allow_html=True)

st.markdown(
    "<div class='section-title'>Cluster Insights</div>",
    unsafe_allow_html=True
)

cols = st.columns(min(k, 4))

for i in range(min(k, 4)):

    color = COLORS[i]
    bg = rgba(color, 0.10)
    size = cluster_sizes[i]

    with cols[i]:

        st.markdown(
            f"""
<div style="
background:{bg};
border-left:4px solid {color};
padding:20px;
border-radius:18px;
height:280px;
box-shadow:0 4px 20px rgba(0,0,0,0.25);
">

<div style="
display:flex;
justify-content:space-between;
align-items:center;
margin-bottom:18px;
">

<div style="
font-size:20px;
font-weight:700;
color:{color};
">
{NAMES[i]}
</div>

<div style="
background:{rgba(color,0.18)};
padding:6px 12px;
border-radius:999px;
font-size:12px;
font-weight:600;
color:{color};
">
{size} districts
</div>

</div>

<div style="
height:1px;
background:rgba(255,255,255,0.08);
margin-bottom:18px;
">
</div>

<div style="
display:flex;
justify-content:space-between;
margin-bottom:14px;
">
<span style="color:#8EA0B5;">Electric score</span>
<span style="color:white;font-weight:700;">
{centers[i][0]:+.2f}
</span>
</div>

<div style="
display:flex;
justify-content:space-between;
margin-bottom:14px;
">
<span style="color:#8EA0B5;">Students</span>
<span style="color:white;font-weight:700;">
{centers[i][1]:+.2f}
</span>
</div>

<div style="
display:flex;
justify-content:space-between;
margin-bottom:14px;
">
<span style="color:#8EA0B5;">Free lunch</span>
<span style="color:white;font-weight:700;">
{centers[i][2]:+.2f}
</span>
</div>

<div style="
display:flex;
justify-content:space-between;
">
<span style="color:#8EA0B5;">Poverty</span>
<span style="color:white;font-weight:700;">
{centers[i][3]:+.2f}
</span>
</div>

</div>
""",
            unsafe_allow_html=True
        )

# =========================================================
# VARIABLE DISTRIBUTION
# =========================================================
st.markdown("<br>", unsafe_allow_html=True)

with st.expander("📊 Variable Distribution"):

    features = {
        "electric_pct": "Electric %",
        "students": "Students",
        "lunch_pct": "Free Lunch %",
        "poverty_pct": "Poverty %"
    }

    selected = st.selectbox(
        "Select variable",
        list(features.keys()),
        format_func=lambda x: features[x]
    )

    box = go.Figure()

    for i in range(k):

        mask = df["cluster"] == i

        box.add_trace(go.Box(
            y=df.loc[mask, selected],
            name=NAMES[i],
            marker_color=COLORS[i],
            line_color=COLORS[i],
            fillcolor=rgba(COLORS[i], 0.18),
            boxmean=True
        ))

    box.update_layout(
        **chart_layout(420),

        yaxis=dict(
            title="Standardized value",
            gridcolor=GRID
        )
    )

    st.plotly_chart(box, use_container_width=True)

# =========================================================
# DATA TABLE
# =========================================================
with st.expander("📋 Full Dataset"):

    st.dataframe(
        df,
        use_container_width=True,
        height=350
    )

# =========================================================
# FOOTER
# =========================================================
st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center;
            color:#73859B;
            font-size:13px;
            padding-bottom:10px;'>

Built with Streamlit · Plotly · Scikit-Learn

</div>
""", unsafe_allow_html=True)