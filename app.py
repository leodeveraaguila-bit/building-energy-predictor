import os
import base64
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from supabase import create_client

# ArchiLoad AI — Streamlit application
# Keep these three model files beside app.py in your GitHub repository:
# heating_model.pkl, cooling_model.pkl, feature_columns.pkl

st.set_page_config(page_title="ArchiLoad AI", page_icon="🏢", layout="wide")

NAVY = "#142B3D"
NAVY_LIGHT = "#213E51"
TEAL = "#00858B"
TEAL_LIGHT = "#E9F6F5"
BG = "#F4F6F8"
TEXT = "#183244"
MUTED = "#637583"
BORDER = "#DCE4E9"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root {{ --archi-navy:{NAVY}; --archi-teal:{TEAL}; --archi-muted:{MUTED}; --archi-border:{BORDER}; }}
html, body, [class*="css"] {{ font-family:Inter, sans-serif; }}
.stApp {{ background:{BG}; color:{TEXT}; }}
[data-testid="stHeader"] {{ background:rgba(244,246,248,.92); }}
[data-testid="stSidebar"] {{ background:{NAVY}; border-right:1px solid #263f50; }}
[data-testid="stSidebar"] * {{ color:#F5F8FA; }}
[data-testid="stSidebar"] .stRadio label {{ color:#D9E3E9 !important; }}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{ color:#B7C8D3; }}
[data-testid="stSidebar"] [data-testid="stRadio"] > div {{ gap:.28rem; }}
[data-testid="stSidebar"] [data-testid="stRadio"] label {{ padding:.55rem .65rem; border-radius:7px; }}
.brand {{ color:white; font-size:1.28rem; font-weight:700; letter-spacing:-.03em; margin-bottom:.2rem; }}
.sidebar-kicker {{ color:#B7C8D3; font-size:.68rem; font-weight:700; letter-spacing:.1em; margin:1.25rem 0 .5rem; }}
.sidebar-promo {{ background:{NAVY_LIGHT}; border:1px solid #345064; border-radius:10px; padding:1rem; margin-top:2rem; }}
.sidebar-promo h4 {{ color:white; font-size:.9rem; margin:.4rem 0; }}
.sidebar-promo p {{ color:#B7C8D3; font-size:.78rem; line-height:1.55; margin:0; }}
.page-title {{ font-size:1.9rem; font-weight:600; color:{TEXT}; letter-spacing:-.035em; margin:.55rem 0 .2rem; }}
.page-subtitle,.muted {{ color:{MUTED}; font-size:.89rem; line-height:1.6; }}
.card {{ background:white; border:1px solid {BORDER}; border-radius:11px; padding:1.2rem 1.3rem; margin-bottom:1rem; box-shadow:0 2px 8px rgba(20,43,61,.025); }}
.card-title {{ color:{TEXT}; font-weight:650; font-size:1.03rem; margin-bottom:.4rem; }}
.eyebrow {{ color:{TEAL}; text-transform:uppercase; letter-spacing:.075em; font-size:.7rem; font-weight:700; }}
.metric-value {{ color:{TEXT}; font-size:1.85rem; font-weight:700; line-height:1.25; }}
.metric-label {{ color:{MUTED}; font-size:.8rem; margin-top:.3rem; }}
.notice {{ background:{TEAL_LIGHT}; border:1px solid #D5EFED; border-radius:9px; padding:1.05rem 1.15rem; }}
.footer {{ color:{MUTED}; font-size:.73rem; border-top:1px solid {BORDER}; padding-top:1rem; margin-top:2rem; }}
div.stButton > button[kind="primary"],button[kind="primary"] {{ background:{TEAL}; border-color:{TEAL}; color:white; border-radius:7px; font-weight:600; min-height:2.65rem; }}
div.stButton > button {{ border-radius:7px; min-height:2.45rem; }}
[data-testid="stForm"] {{ background:white; border:1px solid {BORDER}; padding:1.15rem; border-radius:11px; }}
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input {{ border-radius:7px; }}
[data-testid="stSelectbox"] [data-baseweb="select"] > div {{ border-radius:7px; }}
[data-testid="stTabs"] button {{ font-weight:600; }}
hr {{ border-color:{BORDER}; }}
@media (max-width: 800px) {{ .page-title {{ font-size:1.55rem; }} }}
</style>
""", unsafe_allow_html=True)

INPUT_LABELS = {
    "relative_compactness": "Relative compactness",
    "surface_area": "Surface area",
    "wall_area": "Wall area",
    "roof_area": "Roof area",
    "overall_height": "Overall height",
    "orientation": "Orientation",
    "glazing_area": "Glazing area",
    "glazing_distribution": "Glazing distribution",
}
DEFAULT_PREFERENCES = {
    "language": "English",
    "default_page": "Overview",
    "show_guidance": True,
    "product_updates": False,
    "getting_started_tips": True,
}
VALID_PAGES = ["Overview", "Prediction", "About", "Settings", "Account"]

for key, default in {
    "access_token": None,
    "refresh_token": None,
    "user": None,
    "page": "Overview",
    "auth_view": "Sign in",
    "last_prediction": None,
    "prediction_inputs": None,
    "_default_page_applied": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default
