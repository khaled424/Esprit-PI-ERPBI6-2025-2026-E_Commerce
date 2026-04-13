import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import os
from datetime import datetime

# --- APP CONFIGURATION ---
st.set_page_config(page_title="E-CORP Premium Intelligence", layout="wide")

# IRON-CLAD VISIBILITY CSS (FORCING ALL COLORS)
NAVY = "#1E293B"     # Ardoise foncée
BLUE_MAIN = "#1E40AF" # Bleu Royal
TEXT_STD = "#334155"  # Ardoise medium

st.markdown(f"""
    <style>
    /* Global Styles */
    .stApp {{ background-color: #F8FAFC !important; color: {TEXT_STD} !important; }}
    
    /* Force headings */
    h1, h2, h3, .main-title {{ color: {NAVY} !important; font-weight: 800 !important; }}
    
    /* Section Headers */
    .section-header {{ 
        color: {BLUE_MAIN} !important; 
        font-weight: 700 !important; 
        text-transform: uppercase; 
        letter-spacing: 0.1em;
        margin-top: 2rem !important; 
    }}

    /* METRIC CARDS - ABSOLUTE VISIBILITY */
    div[data-testid="metric-container"] {{
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        padding: 20px !important;
        border-radius: 12px !important;
    }}
    /* Force Label (the small text above value) */
    div[data-testid="stMetricLabel"] p {{
        color: {TEXT_STD} !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
    }}
    /* Force Value */
    div[data-testid="stMetricValue"] div {{
        color: {NAVY} !important;
        font-weight: 800 !important;
    }}

    /* Sidebar Branding */
    [data-testid="stSidebar"] {{ background-color: #0F172A !important; }}
    [data-testid="stSidebar"] * {{ color: #F1F5F9 !important; font-weight: 600 !important; }}

    /* Form Labels & Selectboxes */
    label, p, span {{ color: {TEXT_STD} !important; font-weight: 500 !important; }}
    .stSelectbox label, .stNumberInput label {{ color: {NAVY} !important; font-weight: 700 !important; }}
    
    /* Alerts Fixes */
    .stAlert {{ border-left: 5px solid {BLUE_MAIN} !important; background-color: white !important; }}
    .stAlert p {{ color: {NAVY} !important; font-weight: 600 !important; }}
    </style>
""", unsafe_allow_html=True)

# --- LOADING ASSETS ---
@st.cache_resource
def load_assets():
    assets = {}
    m_dir = '02_ML_Engineering/models/'
    paths = {
        'reg': m_dir + 'regression_revenue_model.joblib',
        'cls': m_dir + 'classification_model.joblib',
        'clu': m_dir + 'clustering_model.joblib',
        'clu_sc': m_dir + 'clustering_scaler.joblib',
        'ui_m': m_dir + 'user_item_matrix.joblib',
        'sim': m_dir + 'user_similarity_df.joblib'
    }
    for k, p in paths.items():
        if os.path.exists(p): assets[k] = joblib.load(p)
    return assets

@st.cache_data
def load_data():
    df = pd.read_csv('02_ML_Engineering/data/processed/dataset_ml_features.csv')
    df['FK_Date'] = df['FK_Date'].replace([-1, 0], pd.NA)
    df['Date'] = pd.to_datetime(df['FK_Date'].astype(str), format='%Y%m%d', errors='coerce')
    return df

assets = load_assets()
data = load_data()
data_v = data.dropna(subset=['Date']).sort_values('Date').copy()

# Data Engineering Engine
def prepare_input(c_id, p_id, q, r, clu):
    n = datetime.now()
    avg_ca = data['client_ca_moyen'].mean()
    if c_id in data['FK_Client'].values:
        avg_ca = data[data['FK_Client'] == c_id]['client_ca_moyen'].iloc[0]
    
    feat = pd.DataFrame([{
        'Mois_sin': np.sin(2 * np.pi * n.month / 12),
        'Segment_Client': clu,
        'client_ca_moyen': avg_ca,
        'client_nb_achats': 1.0,
        'log_Likes': 0.0,
        'FK_Geographie': -1,
        'log_Prix_unitaire': data['log_Prix_unitaire'].mean(),
        'Est_Ete': 1 if n.month in [6, 7, 8] else 0,
        'Semaine': n.isocalendar()[1],
        'Jour': n.day,
        'log_Quantite': np.log1p(q),
        'A_Remise': r / 100,
        'Est_Debut_Mois': 1 if n.day <= 5 else 0,
        'Jour_sem_sin': np.sin(2 * np.pi * n.weekday() / 7),
        'FK_Client': c_id,
        'Likes_roll4w': 0.0,
        'Annee': n.year,
        'Jour_semaine': n.weekday(),
        'Jour_sem_cos': np.cos(2 * np.pi * n.weekday() / 7),
        'FK_Produit': p_id,
        'Tranche_Remise': 1 if r > 10 else 0,
        'Est_Q4': 1 if n.month >= 10 else 0,
        'Mois_cos': np.cos(2 * np.pi * n.month / 12),
        'Mois': n.month,
        'log_Prix_concurrent': 4.0
    }])
    cols = ['Mois_sin', 'Segment_Client', 'client_ca_moyen', 'client_nb_achats', 'log_Likes', 'FK_Geographie', 'log_Prix_unitaire', 'Est_Ete', 'Semaine', 'Jour', 'log_Quantite', 'A_Remise', 'Est_Debut_Mois', 'Jour_sem_sin', 'FK_Client', 'Likes_roll4w', 'Annee', 'Jour_semaine', 'Jour_sem_cos', 'FK_Produit', 'Tranche_Remise', 'Est_Q4', 'Mois_cos', 'Mois', 'log_Prix_concurrent']
    return feat[cols]

def style_fig(fig):
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color=TEXT_STD,
        xaxis=dict(gridcolor='#E2E8F0', title_font_color=TEXT_STD, tickfont_color=TEXT_STD),
        yaxis=dict(gridcolor='#E2E8F0', title_font_color=TEXT_STD, tickfont_color=TEXT_STD)
    )
    return fig

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("<h2 style='color:white;'>E-CORP</h2>", unsafe_allow_html=True)
    st.write("---")
    menu = st.radio("Pilotage", ["📊 Dashboard Stratégique", "👤 Intelligence CRM", "🔮 Simulateur Prédictif"])
    st.write("---")
    st.caption("v6.1 - Iron Visibility")

# --- DASHBOARD ---
if menu == "📊 Dashboard Stratégique":
    st.markdown("<h1 class='main-title'>Dashboard Stratégique</h1>", unsafe_allow_html=True)
    
    # KPIs Row
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Chiffre d'Affaires", f"{data['Montant_TTC'].sum():,.0f} €")
    k2.metric("Clients", f"{data['FK_Client'].nunique()}")
    k3.metric("Panier Moyen", f"{data['Montant_TTC'].mean():,.1f} €")
    k4.metric("Articles", f"{data['FK_Produit'].nunique()}")

    st.markdown("<p class='section-header'>Analyse de Performance</p>", unsafe_allow_html=True)
    
    # Evolution Chart
    daily = data_v.groupby('Date')['Montant_TTC'].sum().reset_index()
    fig_l = px.line(daily, x='Date', y='Montant_TTC', color_discrete_sequence=[BLUE_MAIN], title="Évolution du CA Journalier")
    st.plotly_chart(style_fig(fig_l), use_container_width=True)
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("<p class='section-header'>Segmentation Clients</p>", unsafe_allow_html=True)
        if 'clu' in assets:
            c_f = ['client_ca_moyen', 'client_nb_achats', 'log_Quantite', 'A_Remise']
            data['Cluster'] = assets['clu'].predict(assets['clu_sc'].transform(data[c_f]))
            pie_df = data.groupby('Cluster')['Montant_TTC'].sum().reset_index()
            fig_p = px.pie(pie_df, values='Montant_TTC', names='Cluster', color_discrete_sequence=px.colors.sequential.Plotly3)
            st.plotly_chart(style_fig(fig_p), use_container_width=True)

    with col_b:
        st.markdown("<p class='section-header'>Top 5 Produits</p>", unsafe_allow_html=True)
        top_df = data.groupby('FK_Produit')['log_Quantite'].sum().sort_values(ascending=False).head(5).reset_index()
        fig_b = px.bar(top_df, x='FK_Produit', y='log_Quantite', color_discrete_sequence=[BLUE_MAIN])
        st.plotly_chart(style_fig(fig_b), use_container_width=True)

# --- CRM ---
elif menu == "👤 Intelligence CRM":
    st.markdown("<h1 class='main-title'>Intelligence CRM</h1>", unsafe_allow_html=True)
    
    cid = st.selectbox("Rechercher un ID Client", sorted(data['FK_Client'].unique()))
    
    if cid:
        hist = data[data['FK_Client'] == cid].sort_values(by='Date', ascending=False)
        l, r = st.columns(2)
        
        with l:
            st.markdown("<p class='section-header'>Profil IA</p>", unsafe_allow_html=True)
            try:
                row = hist.iloc[0]
                c_f = ['client_ca_moyen', 'client_nb_achats', 'log_Quantite', 'A_Remise']
                seg = assets['clu'].predict(assets['clu_sc'].transform([row[c_f].values]))[0]
                xin = prepare_input(cid, row['FK_Produit'], 1, 0, seg)
                score = assets['cls'].predict_proba(xin)[0][1]
                
                st.metric("Segment", f"Cluster {seg}")
                st.metric("Score de Fidélité", f"{score:.1%}")
            except: st.info("Profilage indisponible.")

        with r:
            st.markdown("<p class='section-header'>Recommandations</p>", unsafe_allow_html=True)
            cint = int(cid)
            reco_ok = False
            if 'sim' in assets and cint in assets['sim'].index:
                sims = assets['sim'][cint].sort_values(ascending=False)[1:6].index
                owned = assets['ui_m'].loc[cint]
                recos = assets['ui_m'].loc[sims, owned[owned==0].index].mean().sort_values(ascending=False).head(3).index.tolist()
                for i in recos: st.success(f"Suggest: Produit {i}"); reco_ok = True
            if not reco_ok:
                tops = data.groupby('FK_Produit')['log_Quantite'].sum().sort_values(ascending=False).head(3).index.tolist()
                for i in tops: st.info(f"Top: Produit {i}")

        st.markdown("<p class='section-header'>Historique des Ventes</p>", unsafe_allow_html=True)
        st.dataframe(hist[['Date', 'FK_Produit', 'Montant_TTC']].head(10), use_container_width=True)

# --- SIMULATEUR ---
else:
    st.markdown("<h1 class='main-title'>Simulateur Prédictif</h1>", unsafe_allow_html=True)
    st.write("Estimez la valeur d'une transaction future via l'intelligence artificielle.")
    
    sl1, sl2 = st.columns(2)
    with sl1:
        pr = st.selectbox("Produit Cible", sorted(data['FK_Produit'].unique()))
        qt = st.number_input("Quantité", min_value=1, value=1)
    with sl2:
        re = st.slider("Remise (%)", 0, 100, 5)
        se = st.selectbox("Cluster Cible", range(5))

    if st.button("Calculer le Potentiel"):
        if 'reg' in assets:
            try:
                xi = prepare_input(-1, pr, qt, re, se)
                val = np.expm1(assets['reg'].predict(xi)[0])
                st.write("---")
                st.metric("Prevision Revenu", f"{val:,.2f} €")
                st.success("Analyse prédictive complète.")
            except Exception as e: st.error(f"Inférence impossible: {e}")
        else: st.warning("Modèle non chargé.")