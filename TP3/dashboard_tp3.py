import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG & STYLING
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="E-Bus Initiative | Clustering",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium CSS styling
st.markdown("""
<style>
    * {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    .main {
        background: #fafbfc;
    }
    
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    
    .metric-label {
        color: #6b7280;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }
    
    .metric-value {
        color: #111827;
        font-size: 28px;
        font-weight: 700;
        line-height: 1.2;
    }
    
    .metric-subtext {
        color: #9ca3af;
        font-size: 13px;
        margin-top: 6px;
    }
    
    .section-header {
        color: #111827;
        font-size: 18px;
        font-weight: 700;
        margin-top: 28px;
        margin-bottom: 16px;
        padding-bottom: 12px;
        border-bottom: 2px solid #e5e7eb;
    }
    
    .sidebar-section {
        background: rgba(255,255,255,0.4);
        padding: 12px;
        border-radius: 6px;
        margin: 12px 0;
        border-left: 3px solid #3b82f6;
    }
    
    h1 {
        color: #0f172a;
        font-size: 32px !important;
        font-weight: 800 !important;
        margin-bottom: 4px !important;
        letter-spacing: -0.5px;
    }
    
    h2 {
        color: #1f2937;
        font-size: 20px !important;
        font-weight: 700 !important;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('clustering.csv')
    return df

@st.cache_data
def create_feature_names():
    return {
        '3i. Percent of fleet that is electric': 'E-Bus Fleet %',
        '4b. Number of students in district': 'Student Count',
        '4e. Percentage of students in district eligible for free or reduced price lunch': 'Free/Reduced Meals %',
        '4g. Percent of population below the poverty level': 'Poverty Level %'
    }

df = load_data()
feature_map = create_feature_names()
df_display = df.rename(columns=feature_map)
features = list(feature_map.values())

RANDOM_STATE = 42
K_FINAL = 4

# ─────────────────────────────────────────────────────────────
# COMPUTATIONS (cached)
# ─────────────────────────────────────────────────────────────
@st.cache_data
def compute_hierarchical_clustering(k=4):
    """Compute hierarchical clustering with Ward linkage."""
    agg = AgglomerativeClustering(n_clusters=k, linkage='ward')
    labels = agg.fit_predict(df)
    sil = silhouette_score(df, labels)
    db = davies_bouldin_score(df, labels)
    return labels, sil, db

@st.cache_data
def compute_kmeans_clustering(k=4):
    """Compute K-means clustering for comparison."""
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(df)
    sil = silhouette_score(df, labels)
    db = davies_bouldin_score(df, labels)
    return labels, sil, db

@st.cache_data
def compute_tuning_results():
    """Compare linkage methods and cluster counts."""
    resultados = []
    for linkage_method in ['ward', 'complete', 'average', 'single']:
        for k in [2, 3, 4, 5, 6]:
            agg = AgglomerativeClustering(n_clusters=k, linkage=linkage_method).fit(df)
            sil = silhouette_score(df, agg.labels_)
            db = davies_bouldin_score(df, agg.labels_)
            resultados.append({
                'Linkage': linkage_method.capitalize(),
                'Clusters': k,
                'Silhouette': sil,
                'Davies-Bouldin': db,
            })
    return pd.DataFrame(resultados)

@st.cache_data
def compute_pca_projection():
    """Compute 2D PCA projection."""
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    return pca.fit_transform(df), pca.explained_variance_ratio_

@st.cache_data
def compute_dendrogram_data():
    """Compute linkage for dendrogram."""
    return linkage(df, method='ward')

def get_cluster_profiles(labels):
    """Generate cluster profiles with business labels."""
    profiles_df = df_display.copy()
    profiles_df['Cluster'] = labels
    
    cluster_info = []
    for c in sorted(profiles_df['Cluster'].unique()):
        cluster_data = profiles_df[profiles_df['Cluster'] == c]
        means = cluster_data[features].mean()
        size = len(cluster_data)
        
        # Business label
        if means['Student Count'] > 1.0:
            label = "Large Districts"
        elif means['E-Bus Fleet %'] > 1.5:
            label = "E-Bus Pioneers"
        elif means['Free/Reduced Meals %'] > 0.5 or means['Poverty Level %'] > 0.5:
            label = "High-Need Areas"
        else:
            label = "Mid-Market"
        
        cluster_info.append({
            'Cluster': int(c),
            'Label': label,
            'Size': size,
            'E-Bus Fleet %': means['E-Bus Fleet %'],
            'Student Count': means['Student Count'],
            'Free/Reduced Meals %': means['Free/Reduced Meals %'],
            'Poverty Level %': means['Poverty Level %']
        })
    
    return pd.DataFrame(cluster_info)

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 5])
with col1:
    st.markdown("⚡")
with col2:
    st.markdown("""
    # E-Bus Initiative Analytics
    **District Clustering & Market Segmentation**
    """)

st.markdown("""
Hierarchical clustering analysis to identify district groups for targeted electric school bus adoption strategies.
""")

# ─────────────────────────────────────────────────────────────
# SIDEBAR CONTROLS
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    with st.container():
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        tab_main = st.radio(
            "View",
            ["Overview", "Model Tuning", "Comparison", "Cluster Details"],
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 📊 Model Parameters")
    with st.container():
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        k_select = st.slider(
            "Number of Clusters (K)",
            min_value=2,
            max_value=8,
            value=4,
            step=1,
            help="Adjust cluster count to explore different segmentations"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📈 About")
    st.markdown("""
    **Dataset**: 1,524 U.S. school districts
    
    **Features** (standardized):
    - E-Bus fleet adoption %
    - Student enrollment
    - Free/reduced lunch eligibility
    - Population poverty level
    
    **Method**: Hierarchical clustering (Ward linkage) with silhouette validation
    """, help="This dataset comes from the Electric School Bus Initiative")

# ─────────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────────

if tab_main == "Overview":
    # Compute final model
    labels_final, sil_final, db_final = compute_hierarchical_clustering(k_select)
    profiles = get_cluster_profiles(labels_final)
    X_pca, var_ratio = compute_pca_projection()
    
    # KPIs
    st.markdown('<div class="section-header">Model Performance</div>', unsafe_allow_html=True)
    
    kpi_cols = st.columns(4)
    
    with kpi_cols[0]:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Clusters</div>
            <div class="metric-value">""" + str(k_select) + """</div>
            <div class="metric-subtext">Active segmentation groups</div>
        </div>
        """, unsafe_allow_html=True)
    
    with kpi_cols[1]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Silhouette Score</div>
            <div class="metric-value">{sil_final:.3f}</div>
            <div class="metric-subtext">Cluster cohesion & separation</div>
        </div>
        """, unsafe_allow_html=True)
    
    with kpi_cols[2]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Davies-Bouldin Index</div>
            <div class="metric-value">{db_final:.3f}</div>
            <div class="metric-subtext">Lower is better (optimal < 1.5)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with kpi_cols[3]:
        largest_cluster = profiles['Size'].max()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Largest Cluster</div>
            <div class="metric-value">{largest_cluster}</div>
            <div class="metric-subtext">Districts in primary segment</div>
        </div>
        """, unsafe_allow_html=True)
    
    # PCA Visualization
    st.markdown('<div class="section-header">Cluster Distribution (PCA Projection)</div>', unsafe_allow_html=True)
    
    fig_pca = go.Figure()
    
    colors = px.colors.qualitative.Set2[:k_select]
    for cluster_id in range(k_select):
        mask = labels_final == cluster_id
        label_text = profiles[profiles['Cluster'] == cluster_id]['Label'].values[0]
        size_text = profiles[profiles['Cluster'] == cluster_id]['Size'].values[0]
        
        fig_pca.add_trace(go.Scatter(
            x=X_pca[mask, 0],
            y=X_pca[mask, 1],
            mode='markers',
            name=f"Cluster {cluster_id}: {label_text}",
            marker=dict(
                size=8,
                color=colors[cluster_id],
                opacity=0.7,
                line=dict(width=1, color='white')
            ),
            text=[f"<b>{label_text}</b><br>Cluster {cluster_id}" for _ in range(mask.sum())],
            hovertemplate='%{text}<extra></extra>'
        ))
    
    fig_pca.update_layout(
        title={
            'text': f"District Clusters in 2D Space",
            'x': 0,
            'xanchor': 'left',
            'font': {'size': 14}
        },
        xaxis_title=f"PC1 ({var_ratio[0]:.1%} variance)",
        yaxis_title=f"PC2 ({var_ratio[1]:.1%} variance)",
        hovermode='closest',
        height=500,
        plot_bgcolor='rgba(250, 251, 252, 1)',
        paper_bgcolor='white',
        font=dict(family="sans-serif", size=11, color="#6b7280"),
        xaxis=dict(gridcolor='#e5e7eb', showgrid=True, zeroline=False),
        yaxis=dict(gridcolor='#e5e7eb', showgrid=True, zeroline=False),
        margin=dict(t=40, b=40, l=60, r=40),
    )
    
    st.plotly_chart(fig_pca, use_container_width=True, config={'displayModeBar': False})
    
    # Cluster Profile Table
    st.markdown('<div class="section-header">Cluster Profiles</div>', unsafe_allow_html=True)
    
    profile_display = profiles.copy()
    profile_display = profile_display.sort_values('Size', ascending=False)
    profile_display = profile_display.rename(columns={
        'E-Bus Fleet %': 'E-Bus %',
        'Student Count': 'Students',
        'Free/Reduced Meals %': 'Free Meals %',
        'Poverty Level %': 'Poverty %'
    })
    
    st.dataframe(
        profile_display[['Cluster', 'Label', 'Size', 'E-Bus %', 'Students', 'Free Meals %', 'Poverty %']],
        use_container_width=True,
        hide_index=True,
        column_config={
            'Cluster': st.column_config.NumberColumn(format="%d"),
            'Size': st.column_config.NumberColumn(format="%d"),
            'E-Bus %': st.column_config.NumberColumn(format="%.2f"),
            'Students': st.column_config.NumberColumn(format="%.2f"),
            'Free Meals %': st.column_config.NumberColumn(format="%.2f"),
            'Poverty %': st.column_config.NumberColumn(format="%.2f"),
        }
    )
    
    # Feature Distribution by Cluster
    st.markdown('<div class="section-header">Feature Analysis by Cluster</div>', unsafe_allow_html=True)
    
    feature_cols = st.columns(2)
    
    for idx, feature in enumerate(features[:2]):
        with feature_cols[idx % 2]:
            fig_box = go.Figure()
            
            cluster_names = [profiles[profiles['Cluster'] == i]['Label'].values[0] for i in range(k_select)]
            
            for cluster_id in range(k_select):
                mask = labels_final == cluster_id
                fig_box.add_trace(go.Box(
                    y=df_display[feature][mask],
                    name=cluster_names[cluster_id],
                    marker_color=colors[cluster_id],
                    showlegend=False,
                    boxmean='sd'
                ))
            
            fig_box.update_layout(
                title=feature,
                yaxis_title="Standardized Value",
                height=350,
                plot_bgcolor='rgba(250, 251, 252, 1)',
                paper_bgcolor='white',
                font=dict(family="sans-serif", size=10, color="#6b7280"),
                showlegend=False,
                margin=dict(t=40, b=30, l=50, r=20)
            )
            
            st.plotly_chart(fig_box, use_container_width=True, config={'displayModeBar': False})

elif tab_main == "Model Tuning":
    st.markdown('<div class="section-header">Hyperparameter Exploration</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Compare different linkage methods and cluster counts using validation metrics.
    **Silhouette Score**: Higher is better (range: -1 to 1)
    **Davies-Bouldin Index**: Lower is better (optimal < 1.5)
    """)
    
    tuning_results = compute_tuning_results()
    
    # Silhouette comparison
    fig_sil = px.line(
        tuning_results,
        x='Clusters',
        y='Silhouette',
        color='Linkage',
        markers=True,
        title='Silhouette Score by Linkage Method',
        height=400
    )
    
    fig_sil.update_layout(
        plot_bgcolor='rgba(250, 251, 252, 1)',
        paper_bgcolor='white',
        hovermode='x unified',
        font=dict(family="sans-serif", size=11, color="#6b7280"),
        yaxis_title="Silhouette Score",
        xaxis_title="Number of Clusters",
        margin=dict(t=40, b=30, l=60, r=20),
    )
    
    st.plotly_chart(fig_sil, use_container_width=True, config={'displayModeBar': False})
    
    # Davies-Bouldin comparison
    fig_db = px.line(
        tuning_results,
        x='Clusters',
        y='Davies-Bouldin',
        color='Linkage',
        markers=True,
        title='Davies-Bouldin Index by Linkage Method',
        height=400
    )
    
    fig_db.update_layout(
        plot_bgcolor='rgba(250, 251, 252, 1)',
        paper_bgcolor='white',
        hovermode='x unified',
        font=dict(family="sans-serif", size=11, color="#6b7280"),
        yaxis_title="Davies-Bouldin Index (lower is better)",
        xaxis_title="Number of Clusters",
        margin=dict(t=40, b=30, l=60, r=20),
    )
    
    st.plotly_chart(fig_db, use_container_width=True, config={'displayModeBar': False})
    
    # Detailed results table
    st.markdown('<div class="section-header">Detailed Results</div>', unsafe_allow_html=True)
    
    st.dataframe(
        tuning_results.round(4),
        use_container_width=True,
        hide_index=True,
        column_config={
            'Silhouette': st.column_config.NumberColumn(format="%.4f"),
            'Davies-Bouldin': st.column_config.NumberColumn(format="%.4f"),
        }
    )
    
    st.info("""
    **Recommendation**: Ward linkage with K=4 clusters offers:
    - Strong silhouette score (0.45+)
    - Balanced Davies-Bouldin index
    - Business interpretability
    - Practical market segments
    """)

elif tab_main == "Comparison":
    st.markdown('<div class="section-header">Hierarchical vs K-Means Clustering</div>', unsafe_allow_html=True)
    
    st.markdown("Compare Ward hierarchical clustering with K-Means on the same dataset.")
    
    labels_hier, sil_hier, db_hier = compute_hierarchical_clustering(k_select)
    labels_kmeans, sil_kmeans, db_kmeans = compute_kmeans_clustering(k_select)
    
    # Comparison metrics
    comp_cols = st.columns(3)
    
    with comp_cols[0]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Method</div>
            <div class="metric-value" style="font-size: 18px;">Hierarchical</div>
            <div class="metric-subtext">Ward Linkage</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Silhouette</div>
            <div class="metric-value">{sil_hier:.3f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with comp_cols[1]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Method</div>
            <div class="metric-value" style="font-size: 18px;">K-Means</div>
            <div class="metric-subtext">Lloyd Algorithm</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Silhouette</div>
            <div class="metric-value">{sil_kmeans:.3f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with comp_cols[2]:
        ari = adjusted_rand_score(labels_hier, labels_kmeans)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Adjusted Rand Index</div>
            <div class="metric-value">{ari:.3f}</div>
            <div class="metric-subtext">Agreement between methods</div>
        </div>
        """, unsafe_allow_html=True)
    
    # PCA comparison
    X_pca, var_ratio = compute_pca_projection()
    
    fig_comp = go.Figure()
    colors = px.colors.qualitative.Set2[:k_select]
    
    for cluster_id in range(k_select):
        mask_hier = labels_hier == cluster_id
        
        fig_comp.add_trace(go.Scatter(
            x=X_pca[mask_hier, 0],
            y=X_pca[mask_hier, 1],
            mode='markers',
            name=f"Hier. C{cluster_id}",
            marker=dict(size=8, color=colors[cluster_id], symbol='circle', opacity=0.6),
            showlegend=True
        ))
    
    fig_comp.update_layout(
        title="Hierarchical Clustering Results",
        xaxis_title=f"PC1 ({var_ratio[0]:.1%})",
        yaxis_title=f"PC2 ({var_ratio[1]:.1%})",
        height=500,
        plot_bgcolor='rgba(250, 251, 252, 1)',
        paper_bgcolor='white',
        font=dict(family="sans-serif", size=10, color="#6b7280"),
        margin=dict(t=40, b=30, l=60, r=40),
    )
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_comp, use_container_width=True, config={'displayModeBar': False})
    
    with col2:
        # K-Means visualization
        fig_kmeans = go.Figure()
        
        for cluster_id in range(k_select):
            mask_km = labels_kmeans == cluster_id
            
            fig_kmeans.add_trace(go.Scatter(
                x=X_pca[mask_km, 0],
                y=X_pca[mask_km, 1],
                mode='markers',
                name=f"K-Means C{cluster_id}",
                marker=dict(size=8, color=colors[cluster_id], symbol='square', opacity=0.6),
                showlegend=False
            ))
        
        fig_kmeans.update_layout(
            title="K-Means Clustering Results",
            xaxis_title=f"PC1 ({var_ratio[0]:.1%})",
            yaxis_title=f"PC2 ({var_ratio[1]:.1%})",
            height=500,
            plot_bgcolor='rgba(250, 251, 252, 1)',
            paper_bgcolor='white',
            font=dict(family="sans-serif", size=10, color="#6b7280"),
            margin=dict(t=40, b=30, l=60, r=40),
        )
        
        st.plotly_chart(fig_kmeans, use_container_width=True, config={'displayModeBar': False})
    
    st.markdown('<div class="section-header">Method Comparison</div>', unsafe_allow_html=True)
    
    comparison_text = f"""
    | Metric | Hierarchical | K-Means | Winner |
    |--------|-------------|---------|--------|
    | Silhouette Score | {sil_hier:.4f} | {sil_kmeans:.4f} | {"Hierarchical" if sil_hier > sil_kmeans else "K-Means"} |
    | Davies-Bouldin | {db_hier:.4f} | {db_kmeans:.4f} | {"Hierarchical" if db_hier < db_kmeans else "K-Means"} |
    | Agreement (ARI) | — | — | {ari:.4f} |
    
    **Insights**:
    - Both methods show moderate agreement (ARI), indicating different cluster boundaries
    - Hierarchical clustering preserves dendrogram structure for interpretability
    - K-Means is faster but requires pre-specifying K
    - Hierarchical dendrogram reveals natural cut-off heights
    """
    
    st.markdown(comparison_text)

elif tab_main == "Cluster Details":
    st.markdown('<div class="section-header">Cluster Membership Analysis</div>', unsafe_allow_html=True)
    
    labels_final, _, _ = compute_hierarchical_clustering(k_select)
    profiles = get_cluster_profiles(labels_final)
    
    # Select cluster to explore
    cluster_options = {row['Label']: row['Cluster'] for _, row in profiles.iterrows()}
    selected_label = st.selectbox(
        "Select a cluster to explore",
        options=list(cluster_options.keys()),
        format_func=lambda x: f"{x} ({profiles[profiles['Label'] == x]['Size'].values[0]} districts)"
    )
    
    selected_cluster = cluster_options[selected_label]
    cluster_mask = labels_final == selected_cluster
    cluster_data = df_display[cluster_mask].copy()
    cluster_profile = profiles[profiles['Cluster'] == selected_cluster].iloc[0]
    
    # Cluster summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Districts in Cluster", int(cluster_profile['Size']))
    with col2:
        st.metric("% of Total", f"{100*cluster_profile['Size']/len(df):.1f}%")
    with col3:
        st.metric("Avg E-Bus %", f"{cluster_profile['E-Bus Fleet %']:.2f}")
    with col4:
        st.metric("Avg Poverty %", f"{cluster_profile['Poverty Level %']:.2f}")
    
    st.markdown('<div class="section-header">Feature Distribution</div>', unsafe_allow_html=True)
    
    # Distribution plots for selected cluster
    for feature in features:
        fig = go.Figure()
        
        # Overall distribution
        fig.add_trace(go.Histogram(
            x=df_display[feature],
            name='All Districts',
            opacity=0.5,
            nbinsx=30,
            marker_color='#d1d5db'
        ))
        
        # Cluster distribution
        fig.add_trace(go.Histogram(
            x=cluster_data[feature],
            name=selected_label,
            opacity=0.7,
            nbinsx=20,
            marker_color='#3b82f6'
        ))
        
        fig.update_layout(
            title=feature,
            xaxis_title='Standardized Value',
            yaxis_title='Frequency',
            height=300,
            plot_bgcolor='rgba(250, 251, 252, 1)',
            paper_bgcolor='white',
            font=dict(family="sans-serif", size=10, color="#6b7280"),
            barmode='overlay',
            margin=dict(t=40, b=30, l=50, r=20),
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #9ca3af; font-size: 12px; margin-top: 20px;">
    <p>Electric School Bus Initiative Analytics Dashboard | Lab 3: Hierarchical Clustering</p>
    <p>Data: 1,524 U.S. school districts | Method: Ward linkage hierarchical clustering</p>
</div>
""", unsafe_allow_html=True)
