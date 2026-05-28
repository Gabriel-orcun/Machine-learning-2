import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="ESB Intelligence", page_icon="⚡", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #080C14; }

[data-testid="stSidebar"] { background: #0D1320 !important; border-right: 1px solid #1C2333; }
[data-testid="stSidebar"] * { color: #8A95A3 !important; }

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 2rem 2rem; max-width: 1400px; }

.page-title {
    font-family: 'Syne', sans-serif; font-size: 26px; font-weight: 800;
    color: #E6EDF3; margin: 0; line-height: 1.2;
}
.page-sub { font-size: 13px; color: #4A5568; margin: 6px 0 24px; }
.accent { color: #00C896; }

.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 24px; }
.kpi-card {
    background: #0D1320; border: 1px solid #1C2333;
    border-radius: 12px; padding: 20px 22px; position: relative; overflow: hidden;
}
.kpi-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; }
.kpi-card.green::before  { background: #00C896; }
.kpi-card.blue::before   { background: #388BFD; }
.kpi-card.amber::before  { background: #E3B341; }
.kpi-card.purple::before { background: #A371F7; }

.kpi-tag { font-size: 10px; text-transform: uppercase; letter-spacing: 0.1em; color: #4A5568; margin: 0 0 8px; }
.kpi-value {
    font-family: 'Syne', sans-serif; font-size: 32px; font-weight: 700;
    color: #E6EDF3; margin: 0; line-height: 1.1;
}
.kpi-sub { font-size: 11px; color: #4A5568; margin: 6px 0 0; }
.kpi-badge {
    display: inline-block; padding: 2px 9px; border-radius: 20px;
    font-size: 10px; font-weight: 500; margin-top: 8px;
}
.badge-green  { background: #0D2E22; color: #00C896; }
.badge-blue   { background: #0D1E35; color: #388BFD; }
.badge-amber  { background: #2E2005; color: #E3B341; }
.badge-purple { background: #1E1030; color: #A371F7; }

.section-title {
    font-family: 'Syne', sans-serif; font-size: 11px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.12em; color: #4A5568; margin: 0 0 12px;
}

.insight {
    background: linear-gradient(90deg, #0D2E22 0%, #0D1320 100%);
    border: 1px solid #0E3D2A; border-left: 3px solid #00C896;
    border-radius: 8px; padding: 14px 18px;
    font-size: 13px; color: #7A8E86; margin-top: 20px; line-height: 1.7;
}
.insight b { color: #00C896; }
</style>
""", unsafe_allow_html=True)

# ── Plot theme ────────────────────────────────────────────────────────────────
BG     = "#0D1320"
GRID   = "#1C2333"
TXT    = "#8A95A3"
FONT   = "Inter, sans-serif"
GREEN  = "#00C896"
BLUE   = "#388BFD"
AMBER  = "#E3B341"
GRAY   = "#2A3347"

def blayout(h=260, margin=None):
    m = margin or dict(l=10, r=24, t=20, b=30)
    return dict(height=h, paper_bgcolor=BG, plot_bgcolor=BG,
                font=dict(family=FONT, color=TXT, size=11), margin=m)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:12px 0 20px;'>
      <p style='font-family:Syne,sans-serif;font-size:17px;font-weight:700;
                color:#E6EDF3!important;margin:0;'>⚡ ESB Intelligence</p>
      <p style='font-size:11px;color:#2C3A4A!important;margin:4px 0 0;'>
        Machine Learning II · Lab 1</p>
    </div>
    """, unsafe_allow_html=True)

    data_path = st.text_input("", value="dataset.csv",
                              placeholder="Chemin vers dataset.csv")
    st.markdown("<p style='font-size:10px;color:#2C3A4A!important;margin:-8px 0 16px;'>📂 Chemin du fichier</p>",
                unsafe_allow_html=True)
    st.markdown("<hr style='border-color:#1C2333;margin:0 0 16px;'>", unsafe_allow_html=True)

    try:
        df_all = pd.read_csv(data_path)
        df_all.columns = [
            "locale","region","epa_priority","total_buses","students",
            "schools","median_income","poverty_pct","pm25","ozone","nonwhite_pct","committed"
        ]
        df_all["committed"]    = df_all["committed"].astype(int)
        df_all["poverty_pct"]  = df_all["poverty_pct"] * 100
        df_all["nonwhite_pct"] = df_all["nonwhite_pct"] * 100
    except FileNotFoundError:
        st.error(f"Introuvable : `{data_path}`")
        st.stop()

    regions = st.multiselect("Région", sorted(df_all["region"].unique()), default=sorted(df_all["region"].unique()))
    locales = st.multiselect("Zone",   sorted(df_all["locale"].unique()), default=sorted(df_all["locale"].unique()))
    st.markdown("<hr style='border-color:#1C2333;margin:16px 0;'>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:10px;color:#2C3A4A!important;'>Source : electricschoolbusinitiative.org</p>",
                unsafe_allow_html=True)

df = df_all[df_all["region"].isin(regions) & df_all["locale"].isin(locales)]

total    = len(df)
n_comm   = int(df["committed"].sum())
rate     = n_comm / total * 100 if total else 0
avg_stu  = df[df["committed"]==1]["students"].mean()
avg_stu0 = df[df["committed"]==0]["students"].mean()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<p class='page-title'>Bus Scolaires <span class='accent'>Électriques</span><br>Analyse Exploratoire</p>
<p class='page-sub'>États-Unis · Niveau district · 19 516 observations</p>
""", unsafe_allow_html=True)

# ── KPI cards ─────────────────────────────────────────────────────────────────
total_fmt   = f"{total:,}".replace(",", "\u202f")
comm_fmt    = f"{n_comm:,}".replace(",", "\u202f")
stu_fmt     = f"{avg_stu:,.0f}".replace(",", "\u202f")
stu0_fmt    = f"{avg_stu0:,.0f}".replace(",", "\u202f")

st.markdown(f"""
<div class='kpi-grid'>
  <div class='kpi-card green'>
    <p class='kpi-tag'>Districts analysés</p>
    <p class='kpi-value'>{total_fmt}</p>
    <span class='kpi-badge badge-green'>Dataset complet</span>
  </div>
  <div class='kpi-card blue'>
    <p class='kpi-tag'>Districts engagés ESB</p>
    <p class='kpi-value'>{comm_fmt}</p>
    <span class='kpi-badge badge-blue'>Label = 1</span>
  </div>
  <div class='kpi-card amber'>
    <p class='kpi-tag'>Taux d'adoption</p>
    <p class='kpi-value'>{rate:.1f}<span style='font-size:18px;color:#4A5568'>%</span></p>
    <p class='kpi-sub'>Classe minoritaire · déséquilibre à traiter</p>
  </div>
  <div class='kpi-card purple'>
    <p class='kpi-tag'>Élèves moy. (engagés)</p>
    <p class='kpi-value'>{stu_fmt}</p>
    <p class='kpi-sub'>vs {stu0_fmt} non-engagés</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Row 1 : Région + Zone ─────────────────────────────────────────────────────
c1, c2 = st.columns(2, gap="medium")

with c1:
    st.markdown("<p class='section-title'>Adoption par région</p>", unsafe_allow_html=True)
    reg = (df.groupby("region")["committed"]
             .agg(["sum","count"])
             .assign(rate=lambda x: x["sum"]/x["count"]*100)
             .sort_values("rate").reset_index())
    colors = [GREEN if i==len(reg)-1 else BLUE if i>=len(reg)-2 else GRAY for i in range(len(reg))]
    fig1 = go.Figure(go.Bar(
        x=reg["rate"], y=reg["region"], orientation="h",
        marker=dict(color=colors, line_width=0),
        text=reg["rate"].apply(lambda v: f"{v:.1f}%"),
        textposition="outside", textfont=dict(color=TXT, size=11),
        hovertemplate="%{y}: <b>%{x:.1f}%</b><extra></extra>",
    ))
    fig1.update_layout(**blayout(240),
        xaxis=dict(range=[0, reg["rate"].max()*1.35], showgrid=True,
                   gridcolor=GRID, zeroline=False, ticksuffix="%"),
        yaxis=dict(showgrid=False, tickfont=dict(size=12, color="#C0C8D4")),
    )
    st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})

with c2:
    st.markdown("<p class='section-title'>Adoption par type de zone</p>", unsafe_allow_html=True)
    loc = (df.groupby("locale")["committed"]
             .agg(["sum","count"])
             .assign(rate=lambda x: x["sum"]/x["count"]*100)
             .sort_values("rate").reset_index())
    fig2 = go.Figure(go.Bar(
        x=loc["rate"], y=loc["locale"], orientation="h",
        marker=dict(color=loc["rate"],
                    colorscale=[[0,"#1C2333"],[0.5,BLUE],[1,GREEN]],
                    line_width=0),
        text=loc["rate"].apply(lambda v: f"{v:.1f}%"),
        textposition="outside", textfont=dict(color=TXT, size=11),
        hovertemplate="%{y}: <b>%{x:.1f}%</b><extra></extra>",
    ))
    fig2.update_layout(**blayout(240),
        xaxis=dict(range=[0, loc["rate"].max()*1.35], showgrid=True,
                   gridcolor=GRID, zeroline=False, ticksuffix="%"),
        yaxis=dict(showgrid=False, tickfont=dict(size=12, color="#C0C8D4")),
    )
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

# ── Row 2 : Comparaison + EPA + Donut ─────────────────────────────────────────
c3, c4, c5 = st.columns([2.2, 1, 1], gap="medium")

with c3:
    st.markdown("<p class='section-title'>Profil socio-économique — Engagés vs Non-engagés</p>",
                unsafe_allow_html=True)
    comm  = df[df["committed"]==1]
    ncomm = df[df["committed"]==0]
    lbls  = ["Élèves ÷100", "Pauvreté %", "Non-blanc %", "PM2.5 ×10"]
    vc = [comm["students"].mean()/100, comm["poverty_pct"].mean(),
          comm["nonwhite_pct"].mean(), comm["pm25"].mean()*10]
    vn = [ncomm["students"].mean()/100, ncomm["poverty_pct"].mean(),
          ncomm["nonwhite_pct"].mean(), ncomm["pm25"].mean()*10]
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(name="Engagés ESB", x=lbls, y=vc, marker_color=GREEN, marker_line_width=0,
        text=[f"{v:.1f}" for v in vc], textposition="outside", textfont=dict(color=GREEN, size=10)))
    fig3.add_trace(go.Bar(name="Non-engagés",  x=lbls, y=vn, marker_color=GRAY, marker_line_width=0,
        text=[f"{v:.1f}" for v in vn], textposition="outside", textfont=dict(color=TXT, size=10)))
    fig3.update_layout(**blayout(260), barmode="group",
        xaxis=dict(showgrid=False, tickfont=dict(size=11, color="#C0C8D4")),
        yaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False),
        legend=dict(orientation="h", y=1.1, x=0, font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

with c4:
    st.markdown("<p class='section-title'>Priorité EPA 2022</p>", unsafe_allow_html=True)
    epa = (df.groupby("epa_priority")["committed"]
             .agg(["sum","count"])
             .assign(rate=lambda x: x["sum"]/x["count"]*100)
             .reset_index())
    fig4 = go.Figure(go.Bar(
        x=epa["epa_priority"], y=epa["rate"],
        marker=dict(color=[AMBER if v=="Yes" else GRAY for v in epa["epa_priority"]], line_width=0),
        text=epa["rate"].apply(lambda v: f"{v:.1f}%"),
        textposition="outside", textfont=dict(size=11, color=TXT),
        hovertemplate="%{x}: <b>%{y:.1f}%</b><extra></extra>",
    ))
    fig4.update_layout(**blayout(260),
        xaxis=dict(showgrid=False, tickfont=dict(size=12, color="#C0C8D4")),
        yaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False,
                   range=[0, epa["rate"].max()*1.35], ticksuffix="%"),
    )
    st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})

with c5:
    st.markdown("<p class='section-title'>Répartition globale</p>", unsafe_allow_html=True)
    fig5 = go.Figure(go.Pie(
        labels=["Engagés", "Non-engagés"],
        values=[n_comm, total - n_comm],
        hole=0.72,
        marker=dict(colors=[GREEN, "#1C2333"], line=dict(color=BG, width=3)),
        textinfo="none",
        hovertemplate="%{label}: <b>%{value:,}</b> (%{percent})<extra></extra>",
    ))
    fig5.add_annotation(text=f"<b>{rate:.1f}%</b>", x=0.5, y=0.56,
        font=dict(size=22, color=GREEN, family="Syne, sans-serif"), showarrow=False)
    fig5.add_annotation(text="adoption", x=0.5, y=0.4,
        font=dict(size=11, color=TXT), showarrow=False)
    fig5.update_layout(**blayout(260, margin=dict(l=10,r=10,t=20,b=10)),
        showlegend=True,
        legend=dict(orientation="h", y=-0.05, x=0.1,
                    font=dict(size=10, color=TXT), bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig5, use_container_width=True, config={"displayModeBar": False})

# ── Insight ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class='insight'>
  <b>Insights clés</b> — La région <b>Ouest</b> mène l'adoption à <b>11 %</b> (vs 5,1 % Midwest).
  Les districts engagés sont <b>5× plus grands</b> en nb d'élèves — signal fort pour le Random Forest.
  La proportion <b>non-blanche</b> est 63 % plus élevée chez les engagés.
  Être <b>EPA-prioritaire</b> corrèle avec un taux d'adoption 47 % supérieur.
  Dataset <b>déséquilibré</b> (7,8 % positifs) — penser à <b>class_weight="balanced"</b> ou SMOTE.
</div>
""", unsafe_allow_html=True)

# ── Raw data ──────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("📋  Données brutes"):
    st.dataframe(df, use_container_width=True, height=280)
    st.caption(f"{len(df):,} lignes · {df.shape[1]} colonnes")