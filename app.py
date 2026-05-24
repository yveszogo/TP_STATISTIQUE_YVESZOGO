import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

import plotly.graph_objects as go

st.set_page_config(
    page_title="TP-Yves ZOGO",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito+Sans:wght@300;400;500;600;700;800&display=swap');
:root {
    --rausch:#FF5A5F; --rausch-dk:#E00007; --babu:#00A699;
    --arches:#FC642D; --hof:#484848; --foggy:#767676;
    --border:#DDDDDD; --bg:#F7F7F7; --white:#FFFFFF; --dark:#222222;
}
html,body,[class*="css"],.stApp{font-family:'Nunito Sans',-apple-system,sans-serif!important;background:var(--white)!important;color:var(--dark)!important;}
[data-testid="stSidebar"]{display:none!important;}
[data-testid="collapsedControl"]{display:none!important;}
.block-container{max-width:1100px!important;padding:0 24px 60px!important;margin:0 auto!important;}
header[data-testid="stHeader"]{background:var(--white)!important;}
[data-testid="stDecoration"]{display:none!important;}
.nav-bar{border-bottom:1px solid var(--border);padding:18px 0;margin-bottom:36px;}
.nav-inner{max-width:1100px;margin:0 auto;padding:0 24px;display:flex;align-items:center;justify-content:space-between;}
.nav-logo{font-size:1.45rem;font-weight:800;color:var(--rausch);letter-spacing:-0.5px;}
.nav-meta{font-size:0.8rem;color:var(--foggy);text-align:right;line-height:1.5;}
.sec-lbl{font-size:0.68rem;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:var(--foggy);margin-bottom:8px;}
.divider{border:none;border-top:1px solid var(--border);margin:28px 0;}
.card{background:var(--white);border:1px solid var(--border);border-radius:12px;padding:24px 28px;margin-bottom:18px;}
.card-title{font-size:0.95rem;font-weight:700;color:var(--dark);margin-bottom:18px;padding-bottom:12px;border-bottom:1px solid var(--bg);}
.stSelectbox>div>div,.stNumberInput>div>div{border-radius:8px!important;border:1.5px solid var(--border)!important;background:var(--white)!important;}
.stSelectbox>div>div:focus-within,.stNumberInput>div>div:focus-within{border-color:var(--dark)!important;box-shadow:none!important;}
.stSelectbox label,.stNumberInput label{font-size:0.82rem!important;font-weight:600!important;color:var(--dark)!important;}
.msg-error{font-size:0.82rem;color:var(--rausch);font-weight:500;margin:3px 0 8px 1px;line-height:1.4;}
.msg-warn{font-size:0.82rem;color:var(--arches);font-weight:500;margin:3px 0 8px 1px;line-height:1.4;}
.stButton>button{background-color:var(--rausch)!important;color:var(--white)!important;border:none!important;border-radius:8px!important;padding:14px 28px!important;font-size:0.95rem!important;font-weight:700!important;font-family:'Nunito Sans',sans-serif!important;width:100%!important;transition:background-color 0.15s!important;}
.stButton>button:hover:not(:disabled){background-color:var(--rausch-dk)!important;}
.stButton>button:disabled{background-color:#EBEBEB!important;color:#AAAAAA!important;}
.result-block{border-radius:12px;padding:32px 36px;text-align:center;margin:8px 0 20px 0;}
.result-block.pos{background:var(--bg);border:1px solid var(--border);}
.result-block.neg{background:#FFF5F5;border:1px solid #FFCDD2;}
.result-lbl{font-size:0.68rem;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:var(--foggy);margin-bottom:10px;}
.result-prob{font-size:3.6rem;font-weight:800;line-height:1;margin-bottom:8px;letter-spacing:-2px;}
.result-prob.pos{color:var(--babu);}
.result-prob.neg{color:var(--rausch);}
.result-verdict{font-size:1rem;font-weight:700;color:var(--dark);margin-bottom:4px;}
.result-sub{font-size:0.85rem;color:var(--foggy);}
.stat-row{display:flex;border:1px solid var(--border);border-radius:12px;overflow:hidden;margin-bottom:28px;}
.stat-cell{flex:1;padding:18px 20px;border-right:1px solid var(--border);text-align:center;}
.stat-cell:last-child{border-right:none;}
.stat-val{font-size:1.5rem;font-weight:800;color:var(--dark);line-height:1;margin-bottom:4px;letter-spacing:-0.5px;}
.stat-lbl{font-size:0.72rem;color:var(--foggy);font-weight:600;text-transform:uppercase;letter-spacing:0.5px;}
.prog-wrap{background:#EBEBEB;border-radius:4px;height:5px;margin:6px 0;overflow:hidden;}
.prog-fill{height:100%;border-radius:4px;background:var(--rausch);}
.prog-fill.teal{background:var(--babu);}
.compare-row{display:flex;align-items:center;gap:14px;padding:12px 0;border-bottom:1px solid var(--bg);font-size:0.88rem;}
.compare-row:last-child{border-bottom:none;}
.compare-name{width:175px;color:var(--hof);font-weight:500;flex-shrink:0;}
.compare-bar{flex:1;}
.compare-val{width:48px;text-align:right;font-weight:700;color:var(--dark);}
.factor-item{display:flex;align-items:flex-start;gap:10px;padding:10px 0;border-bottom:1px solid var(--bg);font-size:0.875rem;color:var(--hof);line-height:1.4;}
.factor-item:last-child{border-bottom:none;}
.dot{width:7px;height:7px;border-radius:50%;flex-shrink:0;margin-top:5px;}
.dot.r{background:var(--rausch);}
.dot.g{background:var(--babu);}
.var-row{display:flex;align-items:center;gap:12px;padding:9px 0;border-bottom:1px solid var(--bg);font-size:0.86rem;}
.var-row:last-child{border-bottom:none;}
.var-rank{width:24px;color:var(--foggy);font-weight:700;text-align:center;flex-shrink:0;}
.var-name{flex:1;color:var(--hof);}
.var-bar{width:110px;flex-shrink:0;}
.var-pct{width:44px;text-align:right;font-weight:700;color:var(--dark);}
.reco-block{border-radius:12px;padding:20px 24px;margin-top:14px;border:1px solid var(--border);}
.reco-block.urg{border-color:#FFCDD2;background:#FFF5F5;}
.reco-block.ok{border-color:#B2DFDB;background:#F0FFFE;}
.reco-title{font-size:0.9rem;font-weight:700;color:var(--dark);margin-bottom:12px;}
.reco-item{font-size:0.85rem;color:var(--hof);padding:5px 0 5px 14px;position:relative;line-height:1.5;}
.reco-item::before{content:'';position:absolute;left:0;top:12px;width:4px;height:4px;border-radius:50%;background:var(--rausch);}
.reco-block.ok .reco-item::before{background:var(--babu);}
.desc-tbl{width:100%;font-size:0.85rem;border-collapse:collapse;}
.desc-tbl th{text-align:left;font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;color:var(--foggy);padding:9px 10px;border-bottom:1px solid var(--border);}
.desc-tbl td{padding:9px 10px;color:var(--hof);border-bottom:1px solid var(--bg);vertical-align:top;}
.desc-tbl tr:last-child td{border-bottom:none;}
.footer{border-top:1px solid var(--border);padding:24px 0 0 0;margin-top:48px;display:flex;justify-content:space-between;flex-wrap:wrap;gap:6px;}
.ft{font-size:0.78rem;color:var(--foggy);}
.stTabs [data-baseweb="tab-list"]{gap:0!important;background:transparent!important;border-bottom:1px solid var(--border)!important;border-radius:0!important;padding:0!important;}
.stTabs [data-baseweb="tab"]{border-radius:0!important;background:transparent!important;color:var(--foggy)!important;font-size:0.9rem!important;font-weight:500!important;padding:14px 20px!important;border-bottom:2px solid transparent!important;}
.stTabs [aria-selected="true"]{color:var(--dark)!important;border-bottom-color:var(--dark)!important;font-weight:700!important;background:transparent!important;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    for path in ['ml_models.pkl', './ml_models.pkl',
                 os.path.join(os.path.dirname(__file__), 'ml_models.pkl')]:
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)
    raise FileNotFoundError("ml_models.pkl introuvable.")

try:
    model_data = load_models()
    LOADED = True
except Exception as e:
    LOADED = False
    LOAD_ERR = str(e)

def validate(age, poids, taille, nb_cpn, nb_enfants):
    errors, warnings = {}, {}
    if age == 0:            errors['age']   = "L'âge est requis."
    elif age < 15:          errors['age']   = "Âge hors plage - minimum 15 ans."
    elif age > 49:          errors['age']   = "Âge hors plage - maximum 49 ans."
    elif age < 18:          warnings['age'] = "Grossesse adolescente - groupe à risque élevé."
    if poids == 0.0:        errors['poids'] = "Le poids est requis."
    elif poids < 30:        errors['poids'] = "Poids invalide - minimum 30 kg."
    elif poids > 180:       errors['poids'] = "Poids invalide - maximum 180 kg."
    if taille == 0.0:       errors['taille']= "La taille est requise."
    elif taille < 120:      errors['taille']= "Taille invalide - minimum 120 cm."
    elif taille > 220:      errors['taille']= "Taille invalide - maximum 220 cm."
    if nb_cpn < 0:          errors['cpn']   = "Valeur négative non acceptée."
    elif nb_cpn > 20:       errors['cpn']   = "Valeur supérieure au maximum attendu."
    elif nb_cpn == 0:       warnings['cpn'] = "Aucune visite CPN - facteur de risque majeur."
    elif nb_cpn < 4:        warnings['cpn'] = f"{nb_cpn} visite(s) - l'OMS recommande au moins 4."
    if nb_enfants < 0:      errors['enf']   = "Valeur négative non acceptée."
    elif nb_enfants > 20:   errors['enf']   = "Valeur supérieure au maximum attendu."
    elif nb_enfants >= 5:   warnings['enf'] = f"Parité élevée ({nb_enfants} enfants) - facteur limitant."
    return len(errors) == 0, errors, warnings

def bmi_calc(poids, taille):
    if poids > 0 and taille > 0: return poids / (taille/100)**2
    return None

def bmi_cat_n(bmi):
    if bmi is None: return 1
    if bmi < 18.5:  return 0
    if bmi < 25:    return 1
    return 2

def build_X(age, res, educ_f, travail, bmi_c, cpn_c, par_c,
            avort, dec, tv, educ_m, occ_m, rich, eau, toil):
    em = {'Aucune':0,'Primaire':1,'Secondaire / Supérieur':2}
    dm = {'Autre':0,'Moi-même':1,'Conjoint et moi':2,'Conjoint seul':3}
    rm = {'Faible':0,'Moyen':1,'Élevé':2}
    om = {'Agriculture':0,'Secteur formel':1,'Autre':2}
    cpn_e = 0 if cpn_c=='0' else (1 if cpn_c=='1-3' else 2)
    par_e = 0 if par_c=='1-2' else (1 if par_c=='3-4' else 2)
    return np.array([[
        1 if res=='Urbain' else 0, age, em.get(educ_f,0),
        1 if travail else 0, bmi_c, cpn_e, par_e,
        1 if avort else 0, dm.get(dec,0), 1 if tv else 0,
        em.get(educ_m,0), om.get(occ_m,0), rm.get(rich,0),
        1 if eau else 0, 1 if toil else 0,
    ]], dtype=float)

def gauge_chart(prob):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob*100,1),
        number={'suffix':'%','font':{'size':36,'family':'Nunito Sans','color':'#222222'}},
        gauge={
            'axis':{'range':[0,100],'tickfont':{'size':10,'color':'#767676'},'tickwidth':1,'tickcolor':'#DDDDDD'},
            'bar':{'color':'#FF5A5F' if prob<0.5 else '#00A699','thickness':0.22},
            'bgcolor':'white','borderwidth':0,
            'steps':[{'range':[0,50],'color':'#FFF5F5'},{'range':[50,100],'color':'#F0FFFE'}],
            'threshold':{'line':{'color':'#484848','width':2},'thickness':0.8,'value':71.5},
        },
    ))
    fig.update_layout(height=210,margin=dict(t=28,b=0,l=14,r=14),
                      paper_bgcolor='rgba(0,0,0,0)',font={'family':'Nunito Sans'})
    return fig

st.markdown("""
<div class="nav-bar" style="margin-top:50px;">
  <div class="nav-inner">
    <span class="nav-logo">Machine Learning : Prédiction du Lieu d'Accouchement - Yves ZOGO</span>
  </div>
</div>
""", unsafe_allow_html=True)

if not LOADED:
    st.markdown(f"""
    <div style="max-width:560px;margin:80px auto;text-align:center;">
      <div style="font-size:1.1rem;font-weight:700;color:#222;margin-bottom:10px;">Fichier modèle introuvable</div>
      <div style="font-size:0.88rem;color:#767676;">
        Placez <code>ml_models.pkl</code> dans le même répertoire.<br>
        <span style="font-size:0.78rem;color:#aaa;">{LOAD_ERR}</span>
      </div>
    </div>""", unsafe_allow_html=True)
    st.stop()

auc_dict = model_data['auc_cv']
tab1, tab2, tab3 = st.tabs(["Prédiction", "Analyse du modèle", "À propos"])

# ═══════════ ONGLET 1 ═══════════
with tab1:
    st.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)

    model_options = {
        "Régression Logistique": "logistic",
        "Forêt Aléatoire":       "random_forest",
        "Gradient Boosting":     "gradient_boosting",
    }
    st.markdown('<p class="sec-lbl">Choisir l\'algorithme de prédiction</p>', unsafe_allow_html=True)
    selected_name = st.radio("", list(model_options.keys()),
                              horizontal=True, index=0, label_visibility="collapsed")
    selected_key = model_options[selected_name]
    st.markdown(f"""
    <div style="font-size:0.8rem;color:#767676;margin:4px 0 24px 0;">
      AUC (CV 5-fold) : <strong style="color:#222;">{auc_dict[selected_name]:.3f}</strong>
      &nbsp;·&nbsp; Prévalence nationale : <strong style="color:#222;">71,5 %</strong>
    </div>
        <p style="margin-top:8px; font-size:0.85rem; opacity:0.7;">
        Renseignez le profil d'une femme pour prédire la probabilité d'accouchement en établissement de santé
    </p>""", unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    col_form, col_res = st.columns([1,1], gap="large")

    with col_form:
        st.markdown('<p class="sec-lbl">Données biométriques</p>', unsafe_allow_html=True)
        c1,c2,c3 = st.columns(3)
        with c1: age = st.number_input("Âge", min_value=0, max_value=100, value=0, step=1)
        with c2: poids = st.number_input("Poids (kg)", min_value=0.0, max_value=300.0, value=0.0, step=0.5, format="%.1f")
        with c3: taille = st.number_input("Taille (cm)", min_value=0.0, max_value=250.0, value=0.0, step=0.5, format="%.1f")

        bmi = bmi_calc(poids, taille)
        if bmi and poids > 0 and taille > 0:
            if bmi<18.5:   bl,bc,bt = "Insuffisant","#FFF5F5","#E00007"
            elif bmi<25:   bl,bc,bt = "Normal","#F0FFFE","#00A699"
            else:          bl,bc,bt = "Surpoids","#FFF8F0","#FC642D"
            st.markdown(f"""
            <div style="margin:4px 0 14px;font-size:0.84rem;color:#484848;">
              IMC&nbsp;<strong style="color:{bt};">{bmi:.1f} kg/m²</strong>&nbsp;
              <span style="background:{bc};color:{bt};font-weight:700;font-size:0.76rem;
                           padding:2px 10px;border-radius:12px;">{bl}</span>
            </div>""", unsafe_allow_html=True)
        bmi_c = bmi_cat_n(bmi)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<p class="sec-lbl">Situation socioéconomique</p>', unsafe_allow_html=True)
        c4,c5 = st.columns(2)
        with c4:
            residence = st.selectbox("Milieu de résidence", ["Rural","Urbain"])
            educ_femme = st.selectbox("Niveau d'instruction", ["Aucune","Primaire","Secondaire / Supérieur"])
            richesse = st.selectbox("Niveau économique", ["Faible","Moyen","Élevé"])
        with c5:
            tv_sel = st.selectbox("Accès à la télévision", ["Non","Oui (au moins 1×/semaine)"])
            tv_num = "Oui" in tv_sel
            travail_sel = st.selectbox("Occupation", ["Sans emploi","En activité"])
            travail_num = travail_sel == "En activité"
            eau_sel = st.selectbox("Eau de boisson", ["Non améliorée","Améliorée"])
            eau_num = eau_sel == "Améliorée"

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<p class="sec-lbl">Parcours obstétrical</p>', unsafe_allow_html=True)
        c6,c7,c8 = st.columns(3)
        with c6:
            nb_cpn = st.number_input("Visites CPN", min_value=0, max_value=30, value=0, step=1)
            cpn_cat = '0' if nb_cpn==0 else ('1-3' if nb_cpn<=3 else '≥4')
        with c7:
            nb_enfants = st.number_input("Enfants nés vivants", min_value=0, max_value=20, value=1, step=1)
            par_cat = '1-2' if nb_enfants<=2 else ('3-4' if nb_enfants<=4 else '≥5')
        with c8:
            toilette = st.selectbox("Toilettes", ["Non hygiénique","Hygiénique"])
            toil_num = toilette == "Hygiénique"
        avortement = st.checkbox("Antécédent de grossesse interrompue")

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<p class="sec-lbl">Profil du conjoint</p>', unsafe_allow_html=True)
        c9,c10,c11 = st.columns(3)
        with c9:  educ_mari = st.selectbox("Instruction du conjoint", ["Aucune","Primaire","Secondaire / Supérieur"])
        with c10: occup_mari = st.selectbox("Activité du conjoint", ["Agriculture","Secteur formel","Autre"])
        with c11: decision = st.selectbox("Décision sur les soins", ["Autre","Moi-même","Conjoint et moi","Conjoint seul"])

        st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
        is_valid, errors, warns = validate(age, poids, taille, nb_cpn, nb_enfants)
        for msg in errors.values():
            st.markdown(f'<div class="msg-error">{msg}</div>', unsafe_allow_html=True)
        for msg in warns.values():
            st.markdown(f'<div class="msg-warn">{msg}</div>', unsafe_allow_html=True)

        btn = st.button("Calculer la probabilité", disabled=not is_valid, use_container_width=True)

    with col_res:
        if not btn and 'pred' not in st.session_state:
            st.markdown("""
            <div style="height:100px"></div>
            <div style="text-align:center;padding:48px 20px;">
              <div style="font-size:1.05rem;font-weight:700;color:#222;margin-bottom:10px;">Résultat de l'analyse</div>
              <div style="font-size:0.88rem;color:#767676;line-height:1.6;">
                Renseignez le profil dans le formulaire,<br>
                puis cliquez sur <strong>Calculer la probabilité</strong>.
              </div>
            </div>""", unsafe_allow_html=True)

        if btn and is_valid:
            X = build_X(age, residence, educ_femme, travail_num, bmi_c,
                        cpn_cat, par_cat, avortement, decision, tv_num,
                        educ_mari, occup_mari, richesse, eau_num, toil_num)
            probs_all = {n: float(model_data[k].predict_proba(X)[0][1])
                         for n,k in model_options.items()}
            st.session_state['pred'] = {
                'prob': probs_all[selected_name],
                'pred': probs_all[selected_name] >= 0.5,
                'probs_all': probs_all,
                'model': selected_name,
            }

        if 'pred' in st.session_state:
            s = st.session_state['pred']
            prob = s['prob']; pred = s['pred']
            probs_all = s['probs_all']

            bc = "pos" if pred else "neg"
            pc = "pos" if pred else "neg"
            verdict = "Accouchement en établissement" if pred else "Risque d'accouchement à domicile"
            subtext = "Profil favorable - suivi de routine" if pred else "Intervention prioritaire recommandée"

            st.markdown(f"""
            <div class="result-block {bc}">
              <div class="result-lbl">Probabilité estimée</div>
              <div class="result-prob {pc}">{prob*100:.1f}<span style="font-size:1.7rem;">%</span></div>
              <div class="result-verdict">{verdict}</div>
              <div class="result-sub">{subtext}</div>
            </div>""", unsafe_allow_html=True)

            st.plotly_chart(gauge_chart(prob), use_container_width=True,
                            config={'displayModeBar': False})
            st.markdown("""
            <div style="font-size:0.76rem;color:#767676;text-align:center;margin-top:-6px;margin-bottom:22px;">
              La ligne noire indique la moyenne nationale (71,5 %)
            </div>""", unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Comparaison des algorithmes</div>', unsafe_allow_html=True)
            for name, pv in probs_all.items():
                is_sel = name == s['model']
                bar_c = "teal" if pv >= 0.5 else ""
                st.markdown(f"""
                <div class="compare-row">
                  <div class="compare-name" style="{'font-weight:700;color:#222;' if is_sel else ''}">{name}</div>
                  <div class="compare-bar">
                    <div class="prog-wrap" style="margin:0;">
                      <div class="prog-fill {bar_c}" style="width:{pv*100:.1f}%;"></div>
                    </div>
                  </div>
                  <div class="compare-val">{pv*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            risk_f, prot_f = [], []
            if residence=="Rural":             risk_f.append("Résidence rurale")
            else:                              prot_f.append("Résidence urbaine")
            if educ_femme=="Aucune":           risk_f.append("Aucune instruction")
            elif "Secondaire" in educ_femme:   prot_f.append("Instruction secondaire ou supérieure")
            if cpn_cat=='0':                   risk_f.append("Aucune visite CPN")
            elif cpn_cat=='≥4':               prot_f.append("4 visites CPN ou plus")
            if richesse=="Faible":             risk_f.append("Ménage à faible revenu")
            elif richesse=="Élevé":            prot_f.append("Ménage à revenu élevé")
            if tv_num:                         prot_f.append("Accès à la télévision")
            if par_cat=='≥5':                 risk_f.append("Parité élevée (5 enfants et plus)")
            if bmi and bmi<18.5:              risk_f.append("Insuffisance pondérale")

            if risk_f or prot_f:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="card-title">Facteurs identifiés</div>', unsafe_allow_html=True)
                for f in risk_f:
                    st.markdown(f'<div class="factor-item"><span class="dot r"></span><span>{f}</span></div>', unsafe_allow_html=True)
                for f in prot_f:
                    st.markdown(f'<div class="factor-item"><span class="dot g"></span><span>{f}</span></div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            cls = "urg" if not pred else "ok"
            if not pred:
                st.markdown("""
                <div class="reco-block urg">
                  <div class="reco-title">Recommandations prioritaires</div>
                  <div class="reco-item">Planifier les visites CPN dès que possible (objectif : 4 visites)</div>
                  <div class="reco-item">Identifier le centre de santé le plus proche et organiser le transport</div>
                  <div class="reco-item">Informer sur les dispositifs de prise en charge financière disponibles</div>
                  <div class="reco-item">Impliquer le conjoint dans la planification du lieu d'accouchement</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="reco-block ok">
                  <div class="reco-title">Suivi recommandé</div>
                  <div class="reco-item">Maintenir les visites CPN jusqu'à l'accouchement</div>
                  <div class="reco-item">Confirmer le choix de l'établissement de santé</div>
                  <div class="reco-item">Planifier les consultations post-natales</div>
                </div>""", unsafe_allow_html=True)

# ═══════════ ONGLET 2 ═══════════
with tab2:
    st.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="stat-row">
      <div class="stat-cell"><div class="stat-val">6 463</div><div class="stat-lbl">Observations</div></div>
      <div class="stat-cell"><div class="stat-val">71,5 %</div><div class="stat-lbl">Établissement</div></div>
      <div class="stat-cell"><div class="stat-val">15</div><div class="stat-lbl">Variables</div></div>
      <div class="stat-cell"><div class="stat-val">0,888</div><div class="stat-lbl">Meilleur AUC</div></div>
      <div class="stat-cell"><div class="stat-val">0,45</div><div class="stat-lbl">Pseudo R²</div></div>
    </div>""", unsafe_allow_html=True)

    col_a, col_b = st.columns([1,1], gap="large")

    with col_a:
        st.markdown('<p class="sec-lbl">Performance des algorithmes</p>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        perf = [("Régression Logistique", auc_dict.get("Régression Logistique", 0.888), 0.851, 0.820),
                ("Forêt Aléatoire",       auc_dict.get("Forêt Aléatoire",       0.886), 0.851, 0.826),
                ("Gradient Boosting",     auc_dict.get("Gradient Boosting",     0.882), 0.849, 0.824)]
        best_auc = max(p[1] for p in perf)
        rows_html = ""
        for name, auc_v, f1, acc in perf:
            best = auc_v == best_auc
            rows_html += f"""<tr>
              <td style="font-weight:700;{'color:#FF5A5F;' if best else ''}">{name}</td>
              <td style="text-align:center;font-weight:700;">{auc_v:.3f}</td>
              <td style="text-align:center;">{f1:.1%}</td>
              <td style="text-align:center;">{acc:.1%}</td>
            </tr>"""
        st.markdown(f"""<table class="desc-tbl"><thead><tr>
          <th>Modèle</th><th style="text-align:center;">AUC</th>
          <th style="text-align:center;">F1</th><th style="text-align:center;">Accuracy</th>
        </tr></thead><tbody>{rows_html}</tbody></table>
        <div style="font-size:0.76rem;color:#767676;margin-top:10px;">Validation croisée 5-fold.</div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<p class="sec-lbl" style="margin-top:24px;">Indicateurs de qualité</p>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        for lbl,val,interp in [("Pseudo R² (McFadden)","0,4501","Excellente adéquation (> 0,20)"),
                                ("AIC","1 828,6","Critère de sélection du modèle"),
                                ("Test Hosmer-Lemeshow","p = 0,342","Bonne calibration (p > 0,05)"),
                                ("AUC courbe ROC","0,888","Très bonne discrimination")]:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:baseline;
                        padding:9px 0;border-bottom:1px solid #F7F7F7;font-size:0.875rem;">
              <span style="color:#484848;">{lbl}</span>
              <span style="font-weight:700;color:#222;">{val}</span>
            </div>
            <div style="font-size:0.76rem;color:#767676;padding:2px 0 7px;">{interp}</div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<p class="sec-lbl">Importance des variables - Forêt Aléatoire</p>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        fi = [("Visites CPN",0.266),("Instruction de la femme",0.179),
              ("Niveau économique",0.147),("Exposition TV",0.098),
              ("Activité du conjoint",0.064),("Résidence",0.060),
              ("Âge",0.042),("Instruction du conjoint",0.037),
              ("Toilettes",0.036),("IMC",0.017)]
        for i,(name,imp) in enumerate(fi):
            bw = int(imp/0.266*100)
            bc_c = "#FF5A5F" if imp>=0.05 else "#DDDDDD"
            st.markdown(f"""
            <div class="var-row">
              <div class="var-rank">{i+1}</div>
              <div class="var-name">{name}</div>
              <div class="var-bar">
                <div class="prog-wrap" style="margin:0;">
                  <div class="prog-fill" style="width:{bw}%;background:{bc_c};"></div>
                </div>
              </div>
              <div class="var-pct">{imp:.1%}</div>
            </div>""", unsafe_allow_html=True)
        st.markdown('<div style="font-size:0.76rem;color:#767676;margin-top:10px;">Réduction moyenne de l\'impureté de Gini. Variables en rouge : contribution ≥ 5 %.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ═══════════ ONGLET 3 ═══════════
with tab3:
    st.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)
    col_c1, col_c2 = st.columns([1,1], gap="large")

    with col_c1:
        st.markdown('<p class="sec-lbl">Source des données</p>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        for k,v in [("Enquête","EDS Cameroun 2018 (ECAM 2018)"),
                    ("Organisme","Institut National de la Statistique / ICF"),
                    ("Fichier","IR - Individual Recode (femmes)"),
                    ("Population","Femmes de 15 à 49 ans"),
                    ("Échantillon","6 463 femmes avec accouchement documenté"),
                    ("Variable dépendante","Lieu d'accouchement (0 = Domicile / 1 = Établissement)"),
                    ("Prévalence","71,5 % en établissement de santé"),
                    ("Données","<a href=\"https://www.dhsprogram.com/data/available-datasets.cfm\" target=\"_blank\">dhsprogram.com</a> (accès gratuit sur inscription)")]:
            st.markdown(f"""
            <div style="display:flex;gap:14px;padding:8px 0;border-bottom:1px solid #F7F7F7;font-size:0.875rem;">
              <span style="color:#767676;min-width:160px;flex-shrink:0;">{k}</span>
              <span style="font-weight:600;color:#222;">{v}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<p class="sec-lbl" style="margin-top:24px;">Variables</p>', unsafe_allow_html=True)
        variables = [
            ("M15",   "Lieu d'accouchement (Y)", "0 / 1"),
            ("V025",  "Résidence",               "Urbain / Rural"),
            ("V012",  "Âge",                     "15-49 ans"),
            ("V106",  "Instruction femme",        "4 niveaux"),
            ("M14",   "Visites CPN",              "0 / 1-3 / ≥4"),
            ("V190",  "Niveau économique",        "3 quintiles"),
            ("V159",  "Exposition TV",            "Oui / Non"),
            ("V701",  "Instruction conjoint",     "4 niveaux"),
            ("V705",  "Activité conjoint",        "3 catégories"),
            ("V201",  "Parité",                   "1-2 / 3-4 / ≥5"),
            ("V445",  "IMC",                      "3 catégories"),
            ("V743A", "Décision soins",           "4 modalités"),
            ("V113",  "Eau de boisson",           "Améliorée / Non"),
            ("V116",  "Toilettes",                "Hygiénique / Non"),
            ("V228",  "Antécédent avortement",    "Oui / Non"),
        ]
        var_rows = "".join(f"""<tr>
              <td style="font-family:monospace;font-size:0.78rem;color:#767676;">{code}</td>
              <td>{lbl}</td>
              <td style="color:#767676;font-size:0.82rem;">{rec}</td>
            </tr>""" for code, lbl, rec in variables)
        st.markdown(f"""<div class="card">
        <table class="desc-tbl"><thead><tr>
          <th>Code</th><th>Variable</th><th>Recodage</th>
        </tr></thead><tbody>{var_rows}</tbody></table>
        </div>""", unsafe_allow_html=True)

    with col_c2:
        st.markdown('<p class="sec-lbl">Algorithmes</p>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        for name,desc,auc_v in [
            ("Régression Logistique","Modèle linéaire généralisé (logit). Baseline interprétatif. Coefficients = Odds Ratios. Régularisation C = 1.","0,888"),
            ("Forêt Aléatoire","200 arbres par bagging. Robuste aux non-linéarités. Fournit une importance des variables. max_depth = 8.","0,886"),
            ("Gradient Boosting","150 arbres séquentiels. learning_rate = 0,1. Minimise le résidu à chaque étape. max_depth = 4.","0,882"),
        ]:
            st.markdown(f"""
            <div style="margin-bottom:18px;padding-bottom:18px;border-bottom:1px solid #F7F7F7;">
              <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:5px;">
                <span style="font-size:0.9rem;font-weight:700;color:#222;">{name}</span>
                <span style="font-size:0.82rem;font-weight:700;color:#FF5A5F;">AUC {auc_v}</span>
              </div>
              <div style="font-size:0.85rem;color:#767676;line-height:1.5;">{desc}</div>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<p class="sec-lbl" style="margin-top:40px;">Limites</p>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        for l in ["Données transversales (EDS 2018) - associations, non causalité.",
                  "Variables structurelles absentes : distance aux soins, qualité perçue, facteurs culturels.",
                  "Niveau supérieur fusionné avec Secondaire (quasi-séparation parfaite).",
                  "Application à usage académique - ne se substitue pas à un avis médical.",
                  "Performances estimées sur données 2018, susceptibles d'évoluer."]:
            st.markdown(f"""
            <div style="font-size:0.875rem;color:#484848;padding:7px 0 7px 14px;
                        position:relative;border-bottom:1px solid #F7F7F7;line-height:1.5;">
              <span style="position:absolute;left:0;top:14px;width:4px;height:4px;
                           background:#DDDDDD;border-radius:50%;"></span>{l}
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="footer">
  <div class="ft">TP-Yves ZOGO &nbsp;·&nbsp; EDS Cameroun 2018 &nbsp;·&nbsp; TP Statistique Multivariée 2025-2026</div>
  <div class="ft">Données : <a href="https://www.dhsprogram.com/data/available-datasets.cfm" target="_blank">dhsprogram.com</a> &nbsp;·&nbsp; Modèles : scikit-learn</div>
</div>""", unsafe_allow_html=True)