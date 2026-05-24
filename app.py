import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION PAGE
# ══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="TP Yves ZOGO",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════
# CSS PERSONNALISÉ
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600;700&family=Fira+Mono:wght@400;500&display=swap');

:root {
    --navy:   #1B4F8A;
    --teal:   #0E6655;
    --gold:   #D4AC0D;
    --red:    #C0392B;
    --green:  #1E8449;
    --lblue:  #D6E4F0;
    --lgray:  #F4F6F7;
    --dgray:  #2C3E50;
    --white:  #FFFFFF;
    --accent: #2471A3;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: var(--dgray);
}

/* Header principal */
.main-header {
    background: linear-gradient(135deg, #1B4F8A 0%, #0E6655 60%, #1B4F8A 100%);
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 28px;
    box-shadow: 0 8px 32px rgba(27,79,138,0.25);
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 200px; height: 200px;
    border-radius: 50%;
    background: rgba(255,255,255,0.06);
}
.main-header::after {
    content: '';
    position: absolute;
    bottom: -60px; left: 30%;
    width: 300px; height: 300px;
    border-radius: 50%;
    background: rgba(255,255,255,0.04);
}
.main-header h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    color: white;
    margin: 0 0 6px 0;
    letter-spacing: -0.5px;
}
.main-header p {
    color: rgba(255,255,255,0.82);
    font-size: 1rem;
    margin: 0;
    font-weight: 300;
}

/* Section cards */
.section-card {
    background: white;
    border-radius: 14px;
    padding: 24px 28px;
    margin-bottom: 20px;
    border: 1px solid #E8ECF0;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}
.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.2rem;
    color: var(--navy);
    margin-bottom: 16px;
    padding-bottom: 10px;
    border-bottom: 2px solid var(--lblue);
    display: flex;
    align-items: center;
    gap: 10px;
}

/* Prédiction box */
.pred-facility {
    background: linear-gradient(135deg, #0E6655, #1A8A6F);
    border-radius: 16px;
    padding: 28px 32px;
    text-align: center;
    color: white;
    box-shadow: 0 8px 24px rgba(14,102,85,0.35);
    margin: 12px 0;
}
.pred-home {
    background: linear-gradient(135deg, #C0392B, #E74C3C);
    border-radius: 16px;
    padding: 28px 32px;
    text-align: center;
    color: white;
    box-shadow: 0 8px 24px rgba(192,57,43,0.35);
    margin: 12px 0;
}
.pred-result {
    font-family: 'DM Serif Display', serif;
    font-size: 1.8rem;
    margin: 0 0 6px 0;
}
.pred-prob {
    font-size: 3rem;
    font-weight: 700;
    font-family: 'Fira Mono', monospace;
    margin: 8px 0;
}
.pred-sub {
    font-size: 0.9rem;
    opacity: 0.85;
}

/* Metric cards */
.metric-card {
    background: var(--lgray);
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
    border-left: 4px solid var(--navy);
}
.metric-val {
    font-family: 'Fira Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    color: var(--navy);
}
.metric-lbl {
    font-size: 0.78rem;
    color: #6B7280;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Error & warning boxes */
.error-box {
    background: #FDEDEC;
    border: 1.5px solid #E74C3C;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 0.88rem;
    color: #C0392B;
    margin: 4px 0 10px 0;
    display: flex;
    align-items: flex-start;
    gap: 8px;
}
.warning-box {
    background: #FEFCE8;
    border: 1.5px solid #D4AC0D;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 0.88rem;
    color: #7D6608;
    margin: 4px 0 10px 0;
}
.success-box {
    background: #EAFAF1;
    border: 1.5px solid #1E8449;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 0.88rem;
    color: #1E8449;
    margin: 4px 0 10px 0;
}
.info-box {
    background: #EBF5FB;
    border: 1.5px solid #1B4F8A;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 0.9rem;
    color: #1B4F8A;
    margin: 8px 0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1B4F8A 0%, #0E6655 100%) !important;
}
[data-testid="stSidebar"] * {
    color: white !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label {
    color: rgba(255,255,255,0.85) !important;
    font-size: 0.85rem !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: var(--lgray);
    border-radius: 10px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    font-weight: 500;
    font-size: 0.9rem;
}
.stTabs [aria-selected="true"] {
    background: var(--navy) !important;
    color: white !important;
}

/* Bouton principal */
.stButton > button {
    background: linear-gradient(135deg, #1B4F8A, #2471A3) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 32px !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    font-family: 'DM Sans', sans-serif !important;
    letter-spacing: 0.3px;
    box-shadow: 0 4px 14px rgba(27,79,138,0.35) !important;
    transition: all 0.2s ease !important;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(27,79,138,0.45) !important;
}

/* Inputs */
.stSelectbox > div > div,
.stNumberInput > div > div {
    border-radius: 8px !important;
    border-color: #CBD5E1 !important;
}
.stSelectbox > div > div:focus-within,
.stNumberInput > div > div:focus-within {
    border-color: var(--navy) !important;
    box-shadow: 0 0 0 3px rgba(27,79,138,0.15) !important;
}

/* Footer */
.footer {
    text-align: center;
    font-size: 0.78rem;
    color: #94A3B8;
    padding: 20px 0 8px 0;
    border-top: 1px solid #E2E8F0;
    margin-top: 32px;
}

/* Badge */
.badge {
    display: inline-block;
    background: var(--lblue);
    color: var(--navy);
    font-size: 0.72rem;
    font-weight: 600;
    padding: 3px 9px;
    border-radius: 20px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.badge-green {
    background: #D5F5E3;
    color: var(--green);
}
.badge-red {
    background: #FADBD8;
    color: var(--red);
}

/* Progress bar custom */
.progress-bar-container {
    background: #E8ECF0;
    border-radius: 8px;
    height: 12px;
    margin: 8px 0;
    overflow: hidden;
}
.progress-bar-fill {
    height: 100%;
    border-radius: 8px;
    transition: width 0.8s ease;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# CHARGEMENT DU MODÈLE
# ══════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_models():
    """Charge les modèles ML entraînés sur l'EDS Cameroun 2018."""
    model_path = 'ml_models.pkl'
    if not os.path.exists(model_path):
        # Chemin alternatif
        for alt in ['./ml_models.pkl', '../ml_models.pkl',
                    os.path.join(os.path.dirname(__file__), 'ml_models.pkl')]:
            if os.path.exists(alt):
                model_path = alt
                break
    with open(model_path, 'rb') as f:
        return pickle.load(f)

try:
    model_data = load_models()
    MODELS_LOADED = True
except Exception as e:
    MODELS_LOADED = False
    MODEL_ERROR = str(e)


# ══════════════════════════════════════════════════════════════════════
# FONCTIONS UTILITAIRES
# ══════════════════════════════════════════════════════════════════════
def validate_inputs(age, poids, taille, nb_cpn, nb_enfants):
    """Valide toutes les entrées numériques. Retourne (is_valid, errors, warnings)."""
    errors = {}
    warnings_list = {}

    # ── Âge ─────────────────────────────────────────────────────────
    if age is None or age == 0:
        errors['age'] = "L'âge est obligatoire."
    elif age < 15:
        errors['age'] = f"Âge invalide ({age} ans). Les femmes éligibles ont ≥ 15 ans selon l'EDS."
    elif age > 49:
        errors['age'] = f"Âge invalide ({age} ans). Cette application concerne les femmes de 15 à 49 ans."
    elif age < 18:
        warnings_list['age'] = f"Grossesse adolescente ({age} ans) — groupe à risque élevé."
    elif age > 40:
        warnings_list['age'] = f"ℹ Grossesse tardive ({age} ans) — surveillance renforcée recommandée."

    # ── Poids ────────────────────────────────────────────────────────
    if poids is None or poids == 0:
        errors['poids'] = "Le poids est obligatoire."
    elif poids < 30:
        errors['poids'] = f"Poids invalide ({poids} kg). Valeur minimale attendue : 30 kg."
    elif poids > 180:
        errors['poids'] = f"Poids invalide ({poids} kg). Valeur maximale attendue : 180 kg."
    elif poids < 40:
        warnings_list['poids'] = f"Poids très faible ({poids} kg) — vérifier la saisie."

    # ── Taille ───────────────────────────────────────────────────────
    if taille is None or taille == 0:
        errors['taille'] = "La taille est obligatoire."
    elif taille < 120:
        errors['taille'] = f"Taille invalide ({taille} cm). Valeur minimale : 120 cm."
    elif taille > 220:
        errors['taille'] = f"Taille invalide ({taille} cm). Valeur maximale : 220 cm."

    # ── Nombre de visites CPN ────────────────────────────────────────
    if nb_cpn is None or nb_cpn < 0:
        errors['cpn'] = "Le nombre de visites CPN ne peut pas être négatif."
    elif nb_cpn > 20:
        errors['cpn'] = f"Nombre de visites CPN irréaliste ({nb_cpn}). Maximum attendu : 20."
    elif nb_cpn == 0:
        warnings_list['cpn'] = "Aucune visite CPN — facteur de risque majeur d'accouchement à domicile."
    elif nb_cpn < 4:
        warnings_list['cpn'] = f"Seulement {nb_cpn} visite(s) CPN — l'OMS recommande ≥ 4 visites."

    # ── Parité ───────────────────────────────────────────────────────
    if nb_enfants is None or nb_enfants < 0:
        errors['enfants'] = "Le nombre d'enfants ne peut pas être négatif."
    elif nb_enfants > 20:
        errors['enfants'] = f"Nombre d'enfants irréaliste ({nb_enfants}). Maximum attendu : 20."
    elif nb_enfants >= 5:
        warnings_list['enfants'] = f"Parité élevée ({nb_enfants} enfants) — facteur associé à l'accouchement à domicile."

    return len(errors) == 0, errors, warnings_list


def compute_bmi(poids, taille):
    """Calcule l'IMC."""
    if poids and taille and taille > 0:
        return poids / ((taille / 100) ** 2)
    return None


def encode_features(age, residence, educ_femme, travail, bmi_cat,
                    cpn, parite_cat, avortement, decision, tv,
                    educ_mari, occup_mari, richesse, eau, toilette):
    """Encode les variables catégorielles en valeurs numériques pour le modèle."""
    return np.array([[
        int(residence),
        int(age),
        int(educ_femme),
        int(travail),
        int(bmi_cat),
        int(cpn),
        int(parite_cat),
        int(avortement),
        int(decision),
        int(tv),
        int(educ_mari),
        int(occup_mari),
        int(richesse),
        int(eau),
        int(toilette),
    ]])


def get_risk_level(prob):
    """Retourne le niveau de risque et sa couleur."""
    if prob >= 0.80:
        return "Très élevé", "#1E8449", "✅"
    elif prob >= 0.60:
        return "Élevé", "#148F77", "✔"
    elif prob >= 0.40:
        return "Modéré", "#D4AC0D", "⚠"
    elif prob >= 0.20:
        return "Faible", "#E67E22", "⚠"
    else:
        return "Très faible", "#C0392B", "⛔"


def make_gauge(prob, title="Probabilité"):
    """Crée un graphique gauge avec Plotly."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob * 100,
        number={'suffix': '%', 'font': {'size': 42, 'family': 'DM Sans', 'color': '#1B4F8A'}},
        delta={'reference': 71.5, 'increasing': {'color': '#1E8449'}, 'decreasing': {'color': '#C0392B'}},
        title={'text': title, 'font': {'size': 14, 'family': 'DM Sans', 'color': '#2C3E50'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#94A3B8',
                     'tickfont': {'size': 11}},
            'bar': {'color': '#1B4F8A', 'thickness': 0.28},
            'bgcolor': 'white',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 20],  'color': '#FADBD8'},
                {'range': [20, 40], 'color': '#FAE5D3'},
                {'range': [40, 60], 'color': '#FEFCE8'},
                {'range': [60, 80], 'color': '#D5F5E3'},
                {'range': [80, 100],'color': '#A9DFBF'},
            ],
            'threshold': {
                'line': {'color': '#C0392B', 'width': 3},
                'thickness': 0.85,
                'value': 71.5
            }
        }
    ))
    fig.update_layout(
        height=260, margin=dict(t=40, b=10, l=20, r=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'DM Sans'},
    )
    return fig


def make_feature_importance_chart(model_data, model_key):
    """Crée un graphique d'importance des variables."""
    if model_key == 'random_forest':
        fi = model_data['feature_importance']
        fi_df = pd.DataFrame({'Variable': list(fi.keys()), 'Importance': list(fi.values())})
        fi_df['Variable'] = fi_df['Variable'].map(model_data['feature_names_fr'])
        fi_df = fi_df.sort_values('Importance', ascending=True).tail(12)
        colors_fi = ['#1B4F8A' if v >= 0.05 else '#AED6F1' for v in fi_df['Importance']]
        fig = go.Figure(go.Bar(
            x=fi_df['Importance'], y=fi_df['Variable'],
            orientation='h',
            marker_color=colors_fi,
            text=[f"{v:.1%}" for v in fi_df['Importance']],
            textposition='outside',
            textfont={'size': 10},
        ))
    else:
        model = model_data[model_key]
        coefs = np.abs(model.coef_[0])
        fi_df = pd.DataFrame({'Variable': model_data['features'], 'Importance': coefs})
        fi_df['Variable'] = fi_df['Variable'].map(model_data['feature_names_fr'])
        fi_df = fi_df.sort_values('Importance', ascending=True).tail(12)
        colors_fi = ['#1B4F8A' if v >= np.percentile(coefs, 70) else '#AED6F1' for v in fi_df['Importance']]
        fig = go.Figure(go.Bar(
            x=fi_df['Importance'], y=fi_df['Variable'],
            orientation='h', marker_color=colors_fi,
            text=[f"{v:.2f}" for v in fi_df['Importance']],
            textposition='outside', textfont={'size': 10},
        ))

    fig.update_layout(
        title=dict(text="Importance des variables prédictives", font=dict(size=13, color='#1B4F8A')),
        xaxis_title="Importance",
        height=400, margin=dict(t=50, b=30, l=10, r=60),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'family': 'DM Sans', 'color': '#2C3E50'},
        xaxis=dict(showgrid=True, gridcolor='#E8ECF0'),
    )
    return fig


def make_comparison_chart(probs_dict):
    """Graphique comparatif des 3 modèles."""
    models = list(probs_dict.keys())
    probs  = [probs_dict[m] * 100 for m in models]
    colors_c = ['#1B4F8A' if p >= 50 else '#C0392B' for p in probs]
    fig = go.Figure(go.Bar(
        x=models, y=probs,
        marker_color=colors_c,
        text=[f"{p:.1f}%" for p in probs],
        textposition='outside', textfont={'size': 13, 'color': '#2C3E50'},
    ))
    fig.add_hline(y=71.5, line_dash="dot", line_color="#D4AC0D",
                  annotation_text="Prévalence nationale (71,5%)",
                  annotation_font_color="#7D6608", annotation_font_size=10)
    fig.update_layout(
        yaxis=dict(range=[0, 110], title="Probabilité (%)"),
        height=320, margin=dict(t=30, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'family': 'DM Sans'},
        xaxis=dict(showgrid=False),
        yaxis_showgrid=True, yaxis_gridcolor='#E8ECF0',
    )
    return fig


# ══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 10px 0 20px 0;">
        <div style="font-family:'DM Serif Display',serif; font-size:1.3rem;
                    font-weight:700; margin:8px 0 4px 0;">Yves ZOGO</div>
        <div style="font-size:0.78rem; opacity:0.75;">EDS Cameroun 2018</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("<div style='font-size:0.85rem; font-weight:600; opacity:0.9; margin-bottom:8px;'>⚙ MODÈLE ML</div>",
                unsafe_allow_html=True)
    model_choice_label = st.selectbox(
        "Choisir un algorithme de prédiction",
        ["Régression Logistique", "Forêt Aléatoire", "Gradient Boosting"],
        help="Choisissez le modèle ML pour la prédiction."
    )
    model_key_map = {
        "Régression Logistique": "logistic",
        "Forêt Aléatoire": "random_forest",
        "Gradient Boosting": "gradient_boosting",
    }
    model_key = model_key_map[model_choice_label]

    st.markdown("---")

    if MODELS_LOADED:
        auc_dict = model_data['auc_cv']
        st.markdown("<div style='font-size:0.85rem; font-weight:600; opacity:0.9; margin-bottom:10px;'>PERFORMANCE (AUC CV)</div>",
                    unsafe_allow_html=True)
        for name, auc_val in auc_dict.items():
            bar_w = int(auc_val * 100)
            is_sel = (name == model_choice_label)
            bg = "rgba(255,255,255,0.25)" if is_sel else "rgba(255,255,255,0.1)"
            border = "2px solid rgba(255,255,255,0.5)" if is_sel else "none"
            st.markdown(f"""
            <div style="background:{bg}; border:{border}; border-radius:8px;
                        padding:8px 12px; margin-bottom:8px;">
                <div style="font-size:0.78rem; font-weight:{'700' if is_sel else '400'};
                            margin-bottom:4px;">{'▶ ' if is_sel else ''}{name}</div>
                <div style="background:rgba(255,255,255,0.15); border-radius:6px;
                            height:8px; overflow:hidden;">
                    <div style="background:{'#D4AC0D' if is_sel else 'rgba(255,255,255,0.5)'};
                                width:{bar_w}%; height:100%; border-radius:6px;"></div>
                </div>
                <div style="font-size:0.9rem; font-weight:700; margin-top:4px;
                            font-family:'Fira Mono',monospace;">{auc_val:.3f}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"""
    <div style="font-size:0.78rem; opacity:0.7; line-height:1.6;">
        <b>Données :</b> EDS Cameroun 2018<br>
        <b>N :</b> {model_data['N'] if MODELS_LOADED else '6 463'} femmes<br>
        <b>Outcome :</b> Lieu accouchement<br>
        <b>TP Stat. Multivariée</b>  2024-2025
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# CONTENU PRINCIPAL
# ══════════════════════════════════════════════════════════════════════
# ── Header ───────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>Prédiction du Lieu d'Accouchement</h1>
    <p>Application de Machine Learning  Enquête Démographique et de Santé du Cameroun 2018 (EDS-C 2018)</p>
    <p style="margin-top:8px; font-size:0.85rem; opacity:0.7;">
        Renseignez le profil d'une femme pour prédire la probabilité d'accouchement en établissement de santé
    </p>
</div>
""", unsafe_allow_html=True)

if not MODELS_LOADED:
    st.markdown(f"""
    <div class="error-box">
        <span style="font-size:1.2rem;"></span>
        <div>
            <b>Modèle non chargé.</b> Assurez-vous que le fichier <code>ml_models.pkl</code>
            est dans le même répertoire que l'application.<br>
            <small>Erreur : {MODEL_ERROR}</small>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Onglets principaux ────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Prédiction individuelle",
    "Analyse du modèle",
    "À propos de l'étude",
    "Guide d'utilisation"
])


# ════════════════════════════════════════════════════════════════════
# ONGLET 1 : PRÉDICTION INDIVIDUELLE
# ════════════════════════════════════════════════════════════════════
with tab1:
    col_form, col_result = st.columns([1.15, 1], gap="large")

    with col_form:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Caractéristiques de la femme</div>', unsafe_allow_html=True)

        # ── Sous-section : Données biométriques ──────────────────────
        st.markdown("**Données biométriques**")
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Âge (ans) *", min_value=0, max_value=100,
                                   value=0, step=1,
                                   help="Âge de la femme en années révolues (15-49 ans)")
        with c2:
            poids = st.number_input("Poids (kg) *", min_value=0.0, max_value=300.0,
                                     value=0.0, step=0.5, format="%.1f",
                                     help="Poids en kilogrammes (30-180 kg)")
        with c3:
            taille = st.number_input("Taille (cm) *", min_value=0.0, max_value=250.0,
                                      value=0.0, step=0.5, format="%.1f",
                                      help="Taille en centimètres (120-220 cm)")

        # Calcul et affichage IMC en temps réel
        bmi_val = None
        if poids > 0 and taille > 0:
            bmi_val = compute_bmi(poids, taille)
            if bmi_val < 18.5:
                bmi_cat_num = 0; bmi_label = "Insuffisant"; bmi_color = "#C0392B"
            elif bmi_val < 25:
                bmi_cat_num = 1; bmi_label = "Normal"; bmi_color = "#1E8449"
            else:
                bmi_cat_num = 2; bmi_label = "Surpoids/Obèse"; bmi_color = "#E67E22"
            st.markdown(f"""
            <div class="info-box">
                <b>IMC calculé :</b> {bmi_val:.1f} kg/m²
                &nbsp;→&nbsp; <span style="font-weight:600; color:{bmi_color};">{bmi_label}</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            bmi_cat_num = 1

        st.markdown("---")
        st.markdown("**Contexte socioéconomique**")
        c4, c5 = st.columns(2)
        with c4:
            residence = st.selectbox("Milieu de résidence *",
                                      ["Rural", "Urbain"],
                                      index=0,
                                      help="Zone de résidence de la femme")
            residence_num = 1 if residence == "Urbain" else 0

            educ_femme = st.selectbox("Instruction de la femme *",
                                       ["Aucune", "Primaire", "Secondaire/Supérieur"],
                                       index=0,
                                       help="Niveau d'instruction le plus élevé atteint")
            educ_femme_map = {"Aucune": 0, "Primaire": 1, "Secondaire/Supérieur": 2}
            educ_femme_num = educ_femme_map[educ_femme]

            richesse = st.selectbox("Quintile de richesse du ménage *",
                                     ["Pauvre (Q1-Q2)", "Moyen (Q3)", "Riche (Q4-Q5)"],
                                     index=0,
                                     help="Niveau économique du ménage")
            richesse_map = {"Pauvre (Q1-Q2)": 0, "Moyen (Q3)": 1, "Riche (Q4-Q5)": 2}
            richesse_num = richesse_map[richesse]

            eau = st.selectbox("Source d'eau de boisson",
                                ["Non améliorée", "Améliorée"],
                                index=1,
                                help="Source principale d'eau de boisson du ménage")
            eau_num = 1 if eau == "Améliorée" else 0

        with c5:
            toilette = st.selectbox("Type de toilettes",
                                     ["Non hygiénique", "Hygiénique"],
                                     index=0,
                                     help="Type d'installation sanitaire du ménage")
            toilette_num = 1 if toilette == "Hygiénique" else 0

            tv = st.selectbox("Exposition à la télévision",
                               ["Non (jamais/moins d'1×/sem.)", "Oui (≥ 1× par semaine)"],
                               index=0,
                               help="Fréquence de visionnage de la TV")
            tv_num = 1 if "Oui" in tv else 0

            travail = st.selectbox("Occupation de la femme",
                                    ["Ne travaille pas", "Travaille"],
                                    index=1,
                                    help="Statut d'activité professionnelle de la femme")
            travail_num = 1 if travail == "Travaille" else 0

            decision = st.selectbox("Décision sur les soins de santé",
                                     ["Autre personne", "Soi-même",
                                      "Les deux (conjoint + femme)", "Mari/conjoint seul"],
                                     index=2,
                                     help="Qui décide des soins de santé de la femme ?")
            decision_map = {"Autre personne": 0, "Soi-même": 1,
                            "Les deux (conjoint + femme)": 2, "Mari/conjoint seul": 3}
            decision_num = decision_map[decision]

        st.markdown("---")
        st.markdown("**Données obstétricales**")
        c6, c7, c8 = st.columns(3)
        with c6:
            nb_cpn = st.number_input("Visites (Consultations Pré-Natales) *",
                                      min_value=0, max_value=30, value=0, step=1,
                                      help="Nombre de consultations prénatales effectuées (0 à 20)")
            cpn_num = 0 if nb_cpn == 0 else (1 if nb_cpn <= 3 else 2)

        with c7:
            nb_enfants = st.number_input("Nombre d'enfants nés vivants *",
                                          min_value=0, max_value=20, value=1, step=1,
                                          help="Nombre total d'enfants nés vivants (parité)")
            parite_num = 0 if nb_enfants <= 2 else (1 if nb_enfants <= 4 else 2)

        with c8:
            avortement = st.selectbox("Antécédent de grossesse interrompue",
                                       ["Non", "Oui"],
                                       index=0,
                                       help="La femme a-t-elle eu une fausse couche ou avortement ?")
            avortement_num = 1 if avortement == "Oui" else 0

        st.markdown("---")
        st.markdown("**Profil du mari / conjoint**")
        c9, c10 = st.columns(2)
        with c9:
            educ_mari = st.selectbox("Instruction du mari/conjoint",
                                      ["Aucune", "Primaire", "Secondaire/Supérieur"],
                                      index=0,
                                      help="Niveau d'instruction du mari ou partenaire")
            educ_mari_num = educ_femme_map[educ_mari]

        with c10:
            occup_mari = st.selectbox("Occupation du mari/conjoint",
                                       ["Agriculture/Élevage", "Professionnel/Services", "Autre"],
                                       index=0,
                                       help="Secteur d'activité principal du mari")
            occup_mari_map = {"Agriculture/Élevage": 0, "Professionnel/Services": 1, "Autre": 2}
            occup_mari_num = occup_mari_map[occup_mari]

        st.markdown('</div>', unsafe_allow_html=True)

        # ── Validation et bouton ───────────────────────────────────────
        st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
        is_valid, errors, warnings_dict = validate_inputs(age, poids, taille, nb_cpn, nb_enfants)

        # Affichage des erreurs
        if errors:
            for field, msg in errors.items():
                st.markdown(f'<div class="error-box"><span></span><div><b>{msg}</b></div></div>',
                            unsafe_allow_html=True)

        # Affichage des avertissements
        if warnings_dict:
            for field, msg in warnings_dict.items():
                st.markdown(f'<div class="warning-box">{msg}</div>', unsafe_allow_html=True)

        predict_btn = st.button("Lancer la prédiction", disabled=not is_valid,
                                 use_container_width=True)

    # ── Colonne résultats ─────────────────────────────────────────────
    with col_result:
        if not is_valid and any(age > 0 or poids > 0 or taille > 0 for _ in [1]):
            st.markdown("""
            <div class="section-card" style="text-align:center; padding:40px 20px;">
                <div style="font-family:'DM Serif Display',serif; font-size:1.1rem; color:#1B4F8A;">
                    Corrigez les erreurs pour obtenir une prédiction
                </div>
                <div style="color:#94A3B8; font-size:0.85rem; margin-top:8px;">
                    Les champs marqués * sont obligatoires
                </div>
            </div>
            """, unsafe_allow_html=True)

        elif not predict_btn and not st.session_state.get('last_prediction'):
            st.markdown("""
            <div class="section-card" style="text-align:center; padding:50px 20px;">
                <div style="font-family:'DM Serif Display',serif; font-size:1.2rem; color:#1B4F8A; margin-bottom:8px;">
                    Résultat de la prédiction
                </div>
                <div style="color:#94A3B8; font-size:0.9rem; line-height:1.6;">
                    Renseignez le profil de la femme<br>dans le formulaire, puis cliquez sur<br>
                    <b style="color:#1B4F8A;">Lancer la prédiction</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

        if predict_btn and is_valid:
            X_input = encode_features(
                age, residence_num, educ_femme_num, travail_num, bmi_cat_num,
                cpn_num, parite_num, avortement_num, decision_num, tv_num,
                educ_mari_num, occup_mari_num, richesse_num, eau_num, toilette_num
            )

            # Prédictions des 3 modèles
            probs = {}
            for mk, ml in [('logistic','Régression Logistique'),
                            ('random_forest','Forêt Aléatoire'),
                            ('gradient_boosting','Gradient Boosting')]:
                p_etab = model_data[mk].predict_proba(X_input)[0][1]
                probs[ml] = p_etab

            selected_prob = probs[model_choice_label]
            pred_class = 1 if selected_prob >= 0.5 else 0
            risk_level, risk_color, risk_icon = get_risk_level(selected_prob)

            # Stocker en session
            st.session_state['last_prediction'] = {
                'prob': selected_prob, 'class': pred_class, 'probs': probs,
                'model': model_choice_label
            }

            # ── Résultat principal ────────────────────────────────────
            if pred_class == 1:
                st.markdown(f"""
                <div class="pred-facility">
                    <div class="pred-result"> Établissement de santé</div>
                    <div class="pred-prob">{selected_prob*100:.1f}%</div>
                    <div class="pred-sub">Probabilité d'accouchement en établissement</div>
                    <div style="margin-top:12px; font-size:0.85rem; background:rgba(255,255,255,0.2);
                                border-radius:8px; padding:6px 12px;">
                        Modèle : {model_choice_label}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="pred-home">
                    <div class="pred-result">Risque d'accouchement à domicile</div>
                    <div class="pred-prob">{selected_prob*100:.1f}%</div>
                    <div class="pred-sub">Probabilité d'accouchement en établissement</div>
                    <div style="margin-top:12px; font-size:0.85rem; background:rgba(255,255,255,0.2);
                                border-radius:8px; padding:6px 12px;">
                        Modèle : {model_choice_label}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ── Gauge ─────────────────────────────────────────────────
            st.plotly_chart(make_gauge(selected_prob, "Probabilité d'accouchement en établissement"),
                            use_container_width=True, config={'displayModeBar': False})

            # ── Comparaison des 3 modèles ─────────────────────────────
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Comparaison des 3 modèles</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(make_comparison_chart(probs),
                            use_container_width=True, config={'displayModeBar': False})
            st.markdown('</div>', unsafe_allow_html=True)

            # ── Résumé du profil ──────────────────────────────────────
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Résumé des facteurs de risque</div>',
                        unsafe_allow_html=True)
            risk_factors = []
            protective = []

            if residence_num == 0:
                risk_factors.append("Résidence rurale")
            else:
                protective.append("Résidence urbaine")
            if educ_femme_num == 0:
                risk_factors.append("Aucune instruction")
            elif educ_femme_num == 2:
                protective.append("Instruction secondaire/supérieure")
            if cpn_num == 0:
                risk_factors.append("Aucune visite CPN")
            elif cpn_num == 2:
                protective.append(" ≥ 4 visites CPN")
            if richesse_num == 0:
                risk_factors.append("Ménage pauvre")
            elif richesse_num == 2:
                protective.append("Ménage riche")
            if tv_num == 1:
                protective.append("Exposition à la TV")
            if parite_num == 2:
                risk_factors.append("Parité élevée (≥ 5 enfants)")
            if bmi_val and bmi_val < 18.5:
                risk_factors.append("Insuffisance pondérale")
            if bmi_val and bmi_val >= 25:
                protective.append("Surpoids/obésité (paradoxalement protecteur)")

            col_rf, col_pf = st.columns(2)
            with col_rf:
                if risk_factors:
                    st.markdown("**Facteurs de risque**")
                    for rf_ in risk_factors:
                        st.markdown(f'<div class="error-box">{rf_}</div>',
                                    unsafe_allow_html=True)
                else:
                    st.markdown('<div class="success-box"> Aucun facteur de risque majeur identifié</div>',
                                unsafe_allow_html=True)
            with col_pf:
                if protective:
                    st.markdown("** Facteurs protecteurs**")
                    for pf_ in protective:
                        st.markdown(f'<div class="success-box">{pf_}</div>',
                                    unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

            # ── Recommandations ───────────────────────────────────────
            if selected_prob < 0.5:
                st.markdown("""
                <div class="section-card" style="border-left:4px solid #C0392B;">
                    <div class="section-title" style="color:#C0392B;">
                        Recommandations prioritaires
                    </div>
                    <p>Ce profil présente un <b>risque élevé d'accouchement à domicile</b>.
                    Les interventions suivantes sont recommandées :</p>
                    <ul style="line-height:2;">
                        <li><b>Planifier immédiatement les visites CPN</b> (objectif ≥ 4 visites)</li>
                        <li><b>Identifier le centre de santé le plus proche</b> et organiser le transport</li>
                        <li><b>Sensibiliser aux risques</b> de l'accouchement à domicile</li>
                        <li><b>Impliquer le mari</b> dans la décision du lieu d'accouchement</li>
                        <li><b>Informer sur les subventions</b> disponibles pour l'accouchement</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="section-card" style="border-left:4px solid #1E8449;">
                    <div class="section-title" style="color:#1E8449;"> Profil favorable</div>
                    <p>Ce profil présente une <b>probabilité élevée d'accouchement en établissement
                    ({selected_prob*100:.1f} %)</b>. Continuer à encourager :</p>
                    <ul style="line-height:2;">
                        <li>Les visites CPN jusqu'à l'accouchement</li>
                        <li>L'accouchement dans un établissement qualifié</li>
                        <li>Les consultations post-natales</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# ONGLET 2 : ANALYSE DU MODÈLE
# ════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Performance des modèles (Validation croisée 5-fold)</div>',
                unsafe_allow_html=True)

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    auc_dict = model_data['auc_cv']
    metrics_display = [
        ("Rég. Logistique", auc_dict['Régression Logistique'], "#1B4F8A"),
        ("Forêt Aléatoire", auc_dict['Forêt Aléatoire'], "#0E6655"),
        ("Gradient Boosting", auc_dict['Gradient Boosting'], "#D4AC0D"),
        ("N observations", model_data['N'], "#2C3E50"),
    ]
    for col, (label, value, color) in zip([col_m1, col_m2, col_m3, col_m4], metrics_display):
        with col:
            if label == "N observations":
                st.markdown(f"""
                <div class="metric-card" style="border-left-color:{color};">
                    <div class="metric-val" style="color:{color};">{int(value):,}</div>
                    <div class="metric-lbl">{label}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                sel ="" if label.replace("Rég. ","Régression ") in model_choice_label or label in model_choice_label else ""
                st.markdown(f"""
                <div class="metric-card" style="border-left-color:{color};">
                    <div class="metric-val" style="color:{color};">{value:.3f}</div>
                    <div class="metric-lbl">{sel}AUC — {label}</div>
                </div>
                """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    col_fi, col_stats = st.columns([1.2, 1], gap="large")

    with col_fi:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Importance des variables</div>',
                    unsafe_allow_html=True)
        fig_fi = make_feature_importance_chart(model_data, model_key)
        st.plotly_chart(fig_fi, use_container_width=True, config={'displayModeBar': False})
        st.markdown("""
        <div class="info-box" style="font-size:0.82rem;">
            <b>Note :</b> Pour la Forêt Aléatoire, l'importance est mesurée par la réduction
            moyenne de l'impureté de Gini. Pour la Régression Logistique, ce sont les valeurs
            absolues des coefficients standardisés.
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_stats:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Statistiques de l\'échantillon</div>',
                    unsafe_allow_html=True)
        prev = model_data['prevalence']

        # Donut chart
        fig_donut = go.Figure(go.Pie(
            labels=['Établissement de santé', 'Domicile'],
            values=[prev, 1-prev],
            hole=0.62,
            marker_colors=['#1B4F8A', '#AED6F1'],
            textinfo='percent+label',
            textfont_size=11,
            showlegend=False,
        ))
        fig_donut.add_annotation(text=f"{prev*100:.1f}%", x=0.5, y=0.55,
                                  font=dict(size=26, color='#1B4F8A', family='DM Serif Display'),
                                  showarrow=False)
        fig_donut.add_annotation(text="établissement", x=0.5, y=0.42,
                                  font=dict(size=11, color='#6B7280'), showarrow=False)
        fig_donut.update_layout(height=260, margin=dict(t=10, b=10, l=10, r=10),
                                paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_donut, use_container_width=True, config={'displayModeBar': False})

        # Tableau OR ajustés clés
        st.markdown("**Principaux prédicteurs (OR ajustés)**")
        or_data = pd.DataFrame({
            'Variable': ['CPN ≥4 visites', 'TV : Oui', 'Instr. Sec./Sup.', 'Résidence urbaine',
                         'Richesse riche', 'Instr. Primaire', 'Parité ≥5'],
            'aOR': [21.40, 2.87, 3.93, 1.75, 2.23, 2.51, 0.64],
            'p': ['<0,001','<0,001','<0,001','<0,001','0,004','<0,001','0,029'],
        })
        for _, row in or_data.iterrows():
            color = "#1E8449" if row['aOR'] > 1 else "#C0392B"
            dir_arrow = "▲" if row['aOR'] > 1 else "▼"
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; align-items:center;
                        padding:5px 8px; border-radius:6px; margin-bottom:3px;
                        background:#F8FAFC; font-size:0.83rem;">
                <span>{row['Variable']}</span>
                <span style="font-weight:700; color:{color}; font-family:'Fira Mono',monospace;">
                    {dir_arrow} {row['aOR']:.2f}
                </span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Tableau de qualité du modèle
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Indicateurs de qualité du modèle</div>',
                unsafe_allow_html=True)
    col_q1, col_q2, col_q3, col_q4, col_q5 = st.columns(5)
    quality_metrics = [
        ("Pseudo R² (McFadden)", "0,4501", "Excellente adéquation"),
        ("AUC — Rég. Logistique", "0,888", "Très bonne discrimination"),
        ("AUC — Forêt Aléatoire", "0,886", "Très bonne discrimination"),
        ("Test Hosmer-Lemeshow", "p = 0,342", "Bonne calibration (p > 0,05)"),
        ("N (sous-échantillon)", "2 613*", "*Obs. complètes (sans NA)"),
    ]
    for col, (label, val, interp) in zip([col_q1,col_q2,col_q3,col_q4,col_q5], quality_metrics):
        with col:
            st.markdown(f"""
            <div style="background:#F4F6F7; border-radius:10px; padding:14px 12px;
                        text-align:center; border-top:3px solid #1B4F8A;">
                <div style="font-size:1.15rem; font-weight:700; color:#1B4F8A;
                            font-family:'Fira Mono',monospace;">{val}</div>
                <div style="font-size:0.75rem; font-weight:600; color:#2C3E50;
                            margin:4px 0 2px 0; text-transform:uppercase; letter-spacing:0.3px;">{label}</div>
                <div style="font-size:0.71rem; color:#94A3B8; line-height:1.4;">{interp}</div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# ONGLET 3 : À PROPOS
# ════════════════════════════════════════════════════════════════════
with tab3:
    col_a1, col_a2 = st.columns([1.1, 1], gap="large")

    with col_a1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Contexte de l\'étude</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <p>L'<b>Enquête Démographique et de Santé du Cameroun 2018</b> (EDS-C 2018) est une
        enquête nationale représentative réalisée par l'Institut National de la Statistique (INS)
        avec l'appui technique du programme DHS (ICF International).</p>
        <p>Cette application ML est développée dans le cadre d'un <b>Travail Pratique de
        Statistique Multivariée</b> et vise à prédire la probabilité qu'une femme camerounaise
        accouche dans un établissement de santé plutôt qu'à domicile.</p>
        """, unsafe_allow_html=True)

        for label, val in [
            ("Pays", "Cameroun"), ("Année EDS", "2018"),
            ("Fichier utilisé", "IR (Individual Recode — Femmes)"),
            ("Population cible", "Femmes de 15 à 49 ans"),
            ("N final (avec accouchement documenté)", "6 463 femmes"),
            ("Prévalence d'accouchement en établissement", "71,5 %"),
            ("Variable dépendante (Y)", "Lieu d'accouchement (0=Domicile / 1=Établissement)"),
            ("Modèle de référence", "Régression logistique binaire"),
        ]:
            st.markdown(f"""
            <div style="display:flex; gap:12px; padding:6px 0; border-bottom:1px solid #E8ECF0;
                        font-size:0.88rem;">
                <span style="color:#6B7280; min-width:200px;">{label} :</span>
                <span style="font-weight:600; color:#2C3E50;">{val}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Algorithmes utilisés</div>',
                    unsafe_allow_html=True)
        for algo, desc, auc_v in [
            ("Régression Logistique", "Modèle linéaire généralisé avec fonction logit. Standard en épidémiologie. Coefficients interprétables comme Odds Ratios.", f"AUC = {auc_dict['Régression Logistique']:.3f}"),
            ("Forêt Aléatoire", "Ensemble de 200 arbres de décision entraînés par bagging. Robuste aux non-linéarités et interactions. Fournit une mesure d'importance des variables.", f"AUC = {auc_dict['Forêt Aléatoire']:.3f}"),
            ("Gradient Boosting", "Ensemble d'arbres entraînés séquentiellement par boosting. Minimise l'erreur résiduelle à chaque étape. Généralement très performant.", f"AUC = {auc_dict['Gradient Boosting']:.3f}"),
        ]:
            st.markdown(f"""
            <div style="background:#F8FAFC; border-radius:10px; padding:14px 16px;
                        margin-bottom:10px; border-left:3px solid #1B4F8A;">
                <div style="font-weight:700; color:#1B4F8A; margin-bottom:4px;">{algo}
                    <span style="float:right; font-family:'Fira Mono',monospace;
                                 font-size:0.85rem; color:#0E6655;">{auc_v}</span>
                </div>
                <div style="font-size:0.85rem; color:#4B5563; line-height:1.5;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_a2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Variables du modèle</div>',
                    unsafe_allow_html=True)
        vars_info = [
            ("M15$1", "Lieu d'accouchement (Y)", "Domicile / Établissement"),
            ("V025", "Milieu de résidence", "Urbain / Rural"),
            ("V012", "Âge de la femme", "15-49 ans"),
            ("V106", "Instruction de la femme", "Aucun→Supérieur"),
            ("M14$1", "Visites CPN", "0 / 1-3 / ≥4"),
            ("V190", "Quintile de richesse", "Pauvre/Moyen/Riche"),
            ("V159", "Exposition TV", "Oui / Non"),
            ("V701", "Instruction du mari", "Aucun→Supérieur"),
            ("V705", "Occupation du mari", "Agri./Prof./Autre"),
            ("V201", "Parité", "1-2 / 3-4 / ≥5"),
            ("V445", "IMC de la femme", "kg/m² catégorisé"),
            ("V743A", "Décision sur soins", "4 modalités"),
            ("V113", "Source d'eau", "Améliorée/Non améliorée"),
            ("V116", "Toilettes", "Hygiénique/Non"),
            ("V228", "Antécédent avortement", "Oui / Non"),
        ]
        for code, label, modalities in vars_info:
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; align-items:center;
                        padding:5px 8px; border-radius:6px; margin-bottom:2px;
                        background:#F8FAFC; font-size:0.82rem;">
                <span style="font-family:'Fira Mono',monospace; color:#1B4F8A;
                             font-size:0.78rem; background:#EBF5FB; padding:2px 6px;
                             border-radius:4px; min-width:60px;">{code}</span>
                <span style="flex:1; padding: 0 10px; color:#2C3E50;">{label}</span>
                <span style="color:#6B7280; font-size:0.78rem;">{modalities}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Limites et précautions</div>',
                    unsafe_allow_html=True)
        for limit in [
            "Le modèle est entraîné sur des données transversales (EDS 2018) — il ne peut pas inférer de causalité.",
            "La variable 'Instruction supérieur' présente une quasi-séparation parfaite (99,2 % en établissement) → fusionnée avec Secondaire.",
            "L'application est à des fins pédagogiques. Ne pas utiliser pour des décisions cliniques réelles.",
            "Les performances peuvent varier selon les régions du Cameroun et selon l'année.",
            "Les données de 2018 peuvent ne plus refléter la situation actuelle (2024-2025).",
        ]:
            st.markdown(f"""
            <div style="padding:6px 0 6px 12px; border-left:3px solid #D4AC0D;
                        font-size:0.85rem; color:#4B5563; margin-bottom:6px;
                        line-height:1.5;">{limit}</div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# ONGLET 4 : GUIDE D'UTILISATION
# ════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Guide d\'utilisation de l\'application</div>',
                unsafe_allow_html=True)

    for i, (title, content) in enumerate([
        ("Saisie des données",
         "Accédez à l'onglet <b>Prédiction individuelle</b>. Renseignez tous les champs marqués d'un astérisque (*). L'application valide automatiquement chaque saisie en temps réel."),
        ("Validation des champs",
         "Les erreurs de saisie sont signalées en <span style='color:#C0392B;font-weight:600;'>rouge</span> (erreurs bloquantes) ou en <span style='color:#7D6608;font-weight:600;'>jaune</span> (avertissements). Le bouton de prédiction reste désactivé tant qu'il y a des erreurs."),
        ("Calcul automatique de l'IMC",
         "L'IMC est calculé automatiquement à partir du poids et de la taille. La catégorie (Insuffisant/Normal/Surpoids) est affichée en temps réel et utilisée dans le modèle ML."),
        ("Lancement de la prédiction",
         "Cliquez sur <b>Lancer la prédiction</b>. Les 3 modèles (Régression Logistique, Forêt Aléatoire, Gradient Boosting) sont exécutés simultanément. La jauge indique la probabilité du modèle sélectionné."),
        ("Interprétation des résultats",
         "<b>Probabilité ≥ 50 %</b> → prédiction d'accouchement en établissement <br><b>Probabilité &lt; 50 %</b> → risque d'accouchement à domicile <br>La ligne rouge sur la jauge indique la prévalence nationale (71,5 %)."),
        ("Comparaison des modèles",
         "Le graphique comparatif montre les probabilités prédites par les 3 algorithmes. Des résultats cohérents entre modèles indiquent une prédiction plus fiable."),
        ("Choix du modèle",
         "Changez l'algorithme dans la <b>barre latérale gauche</b>. La Forêt Aléatoire offre le meilleur AUC (0,886). La Régression Logistique est plus interprétable (OR)."),
    ], 1):
        st.markdown(f"""
        <div style="display:flex; gap:16px; padding:14px; background:#F8FAFC;
                    border-radius:10px; margin-bottom:10px; align-items:flex-start;">
            <div style="background:#1B4F8A; color:white; border-radius:50%; width:30px;
                        height:30px; display:flex; align-items:center; justify-content:center;
                        font-weight:700; font-size:0.9rem; flex-shrink:0;">{i}</div>
            <div>
                <div style="font-weight:600; color:#1B4F8A; margin-bottom:4px;">{title}</div>
                <div style="font-size:0.88rem; color:#4B5563; line-height:1.6;">{content}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Règles de validation
    st.markdown("---")
    st.markdown("**Règles de validation des champs**")
    val_rules = [
        ("Âge", "15-49 ans", "Entier", "< 15 ou > 49 → erreur"),
        ("Poids", "30-180 kg", "Décimal", "< 30 ou > 180 → erreur"),
        ("Taille", "120-220 cm", "Décimal", "< 120 ou > 220 → erreur"),
        ("Visites CPN", "0-20", "Entier", "> 20 → erreur ; < 4 → avertissement"),
        ("Nb enfants", "0-20", "Entier", "> 20 → erreur ; ≥ 5 → avertissement"),
    ]
    header_cols = st.columns([1.5, 1.5, 1, 2])
    for col, h in zip(header_cols, ["Champ", "Plage valide", "Type", "Comportement"]):
        col.markdown(f"**{h}**")
    for row in val_rules:
        cols = st.columns([1.5, 1.5, 1, 2])
        for col, val in zip(cols, row):
            col.markdown(f"<span style='font-size:0.88rem;'>{val}</span>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <b>Yves ZOGO</b> Application ML<br>
    <span style="color:#CBD5E1;">
        Modèles : Régression Logistique | Forêt Aléatoire | Gradient Boosting 
        AUC : 0,882 - 0,888  N = 6 463 femmes
    </span>
</div>
""", unsafe_allow_html=True)