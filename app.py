import os
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from supabase import create_client

# ArchiLoad AI — PDF-inspired Streamlit interface
st.set_page_config(page_title="ArchiLoad AI", page_icon="🏗️", layout="wide", initial_sidebar_state="expanded")

ROOT = Path(__file__).resolve().parent
NAVY = "#142B3D"
NAVY_2 = "#1D3B50"
TEAL = "#00838F"
TEAL_DARK = "#006F75"
INK = "#213547"
MUTED = "#647789"
BG = "#F4F7F9"
BORDER = "#DCE5EA"
WHITE = "#FFFFFF"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root {{ --navy:{NAVY}; --teal:{TEAL}; --ink:{INK}; --muted:{MUTED}; --border:{BORDER}; }}
html, body, [class*="css"] {{ font-family: Inter, sans-serif; }}
.stApp {{ background:{BG}; color:{INK}; }}
[data-testid="stHeader"] {{ background:rgba(244,247,249,.96); }}
[data-testid="stSidebar"] {{ background:{NAVY}; border-right:1px solid #25465B; padding-top:1rem; }}
[data-testid="stSidebar"] * {{ color:#F7FAFC; }}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{ color:#C9D8E1; }}
[data-testid="stSidebar"] hr {{ border-color:#365367; }}
[data-testid="stSidebar"] [data-testid="stRadio"] label {{ padding:.5rem .75rem; border-radius:8px; font-weight:600; font-size:.9rem; }}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {{ background:#234357; }}
h1,h2,h3 {{ color:{NAVY}; letter-spacing:-.025em; }}
h1 {{ font-weight:800; }}
.eyebrow {{ color:{TEAL}; font-size:.74rem; font-weight:800; text-transform:uppercase; letter-spacing:.13em; margin-bottom:.25rem; }}
.hero-title {{ color:{NAVY}; font-size:clamp(2rem,3vw,2.75rem); line-height:1.12; font-weight:800; letter-spacing:-.045em; margin:.15rem 0 .55rem; }}
.subtle {{ color:{MUTED}; line-height:1.65; margin-top:0; }}
.sidebar-brand {{ font-size:1.25rem; font-weight:800; color:white; margin:.35rem 0 .15rem; letter-spacing:-.03em; }}
.sidebar-caption {{ font-size:.78rem; color:#C9D8E1; line-height:1.5; }}
.metric-card {{ background:{WHITE}; border:1px solid {BORDER}; border-radius:16px; padding:1.25rem 1.4rem; box-shadow:0 4px 18px rgba(20,43,61,.025); }}
.metric-label {{ color:{MUTED}; font-size:.8rem; font-weight:700; text-transform:uppercase; letter-spacing:.08em; }}
.metric-value {{ color:{NAVY}; font-size:2.25rem; font-weight:800; margin:.3rem 0 .15rem; letter-spacing:-.04em; }}
.pill {{ display:inline-block; background:#E3F4F3; color:{TEAL_DARK}; padding:.32rem .68rem; border-radius:99px; font-size:.76rem; font-weight:750; }}
.note {{ border-left:4px solid {TEAL}; background:#EAF5F5; border-radius:9px; padding:.85rem 1rem; color:#31545D; }}
.warning {{ border-left:4px solid #D59B35; background:#FFF7E7; border-radius:9px; padding:.85rem 1rem; color:#6B5123; }}
.small-muted {{ color:{MUTED}; font-size:.8rem; }}
.section-label {{ font-size:.78rem; color:{MUTED}; text-transform:uppercase; letter-spacing:.09em; font-weight:750; }}
.topbar {{ display:flex; align-items:center; justify-content:space-between; gap:1rem; padding:.2rem 0 1.2rem; margin-bottom:1.2rem; border-bottom:1px solid {BORDER}; }}
.topbar-label {{ color:{MUTED}; font-size:.85rem; font-weight:600; }}
.profile-chip {{ display:flex; align-items:center; gap:.55rem; color:{INK}; font-size:.85rem; font-weight:700; background:white; border:1px solid {BORDER}; padding:.35rem .75rem; border-radius:99px; }}
.profile-initials {{ width:1.75rem; height:1.75rem; border-radius:50%; background:#E3F4F3; color:{TEAL_DARK}; display:inline-flex; align-items:center; justify-content:center; font-weight:800; font-size:.75rem; }}
.stButton > button, .stDownloadButton > button {{ border-radius:10px; font-weight:650; min-height:2.65rem; border:1px solid {BORDER}; }}
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {{ background:{TEAL}; border-color:{TEAL}; color:white; }}
.stButton > button[kind="primary"]:hover {{ background:{TEAL_DARK}; border-color:{TEAL_DARK}; }}
div[data-testid="stForm"] {{ background:white; border:1px solid {BORDER}; border-radius:16px; padding:1.1rem 1.2rem; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ background:white; border-radius:16px; border:1px solid {BORDER}; box-shadow:0 4px 18px rgba(20,43,61,.025); }}
footer {{ visibility:hidden; }}
#MainMenu {{ visibility:hidden; }}
</style>
""", unsafe_allow_html=True)

FEATURES = [
    ("relative_compactness", "Relative compactness"),
    ("surface_area", "Surface area (m²)"),
    ("wall_area", "Wall area (m²)"),
    ("roof_area", "Roof area (m²)"),
    ("overall_height", "Overall height (m)"),
    ("orientation", "Orientation"),
    ("glazing_area", "Glazing area (fraction)"),
    ("glazing_distribution", "Glazing distribution"),
]
DEFAULT_INPUTS = {
    "relative_compactness": 0.80, "surface_area": 661.50, "wall_area": 318.50,
    "roof_area": 147.00, "overall_height": 7.0, "orientation": 2,
    "glazing_area": 0.25, "glazing_distribution": 0,
}
REPORTED_METRICS = {
    "Heating load": {"MAE": 0.35235, "RMSE": 0.489673218, "R²": 0.997699565},
    "Cooling load": {"MAE": 1.079106926, "RMSE": 1.750736111, "R²": 0.966920264},
}


def secret_value(name, default=None):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    try:
        return st.secrets.get("supabase", {}).get(name, default)
    except Exception:
        return default


@st.cache_resource(show_spinner=False)
def get_supabase_client(url, key):
    return create_client(url, key)


def value_of(obj, key, default=None):
    if obj is None:
        return default
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def user_metadata(user=None):
    user = user if user is not None else st.session_state.get("user")
    meta = value_of(user, "user_metadata", {}) or {}
    return dict(meta) if isinstance(meta, dict) else {}


def current_email():
    return value_of(st.session_state.get("user"), "email", "") or ""


def display_name():
    meta = user_metadata()
    full = f"{str(meta.get('first_name', '')).strip()} {str(meta.get('last_name', '')).strip()}".strip()
    if full:
        return full
    if current_email():
        return current_email().split("@")[0].replace(".", " ").replace("_", " ").title()
    return "Alex"


def get_user_id():
    return value_of(st.session_state.get("user"), "id")


def set_user_from_auth_response(response):
    session = value_of(response, "session")
    user = value_of(response, "user") or value_of(session, "user")
    if user is not None:
        st.session_state.user = user
        st.session_state.authenticated = True
    return user


def sign_out():
    try:
        client = st.session_state.get("supabase")
        if client:
            client.auth.sign_out()
    except Exception:
        pass
    for key in ["user", "authenticated", "last_prediction", "auth_message", "sidebar_page", "auth_mode"]:
        st.session_state.pop(key, None)
    st.session_state.page = "Overview"
    st.rerun()


@st.cache_resource(show_spinner=False)
def load_models():
    paths = [ROOT / "heating_model.pkl", ROOT / "cooling_model.pkl", ROOT / "feature_columns.pkl"]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing model file(s): " + ", ".join(missing) + ". Upload them to the root of your GitHub repository.")
    heating_model, cooling_model, feature_columns = [joblib.load(p) for p in paths]
    if isinstance(feature_columns, (pd.Index, np.ndarray)):
        feature_columns = list(feature_columns)
    return heating_model, cooling_model, list(feature_columns)


def normalize_name(value):
    return "".join(ch.lower() for ch in str(value) if ch.isalnum())


ALIASES = {
    "relative_compactness": ["relativecompactness", "relative_compactness", "x1"],
    "surface_area": ["surfacearea", "surface_area", "x2"],
    "wall_area": ["wallarea", "wall_area", "x3"],
    "roof_area": ["roofarea", "roof_area", "x4"],
    "overall_height": ["overallheight", "overall_height", "x5"],
    "orientation": ["orientation", "x6"],
    "glazing_area": ["glazingarea", "glazing_area", "x7"],
    "glazing_distribution": ["glazingareadistribution", "glazingdistribution", "glazing_area_distribution", "glazing_distribution", "x8"],
}


def make_model_frame(values, columns):
    lookup = {normalize_name(alias): key for key, aliases in ALIASES.items() for alias in aliases}
    row = {}
    for column in columns:
        key = lookup.get(normalize_name(column))
        if key is None:
            raise ValueError(f"Unrecognized feature column '{column}' in feature_columns.pkl. Check the training feature names.")
        row[column] = values[key]
    return pd.DataFrame([row], columns=columns)


def predict_loads(values):
    heating_model, cooling_model, columns = load_models()
    frame = make_model_frame(values, columns)
    heating = float(np.asarray(heating_model.predict(frame)).reshape(-1)[0])
    cooling = float(np.asarray(cooling_model.predict(frame)).reshape(-1)[0])
    return heating, cooling


def save_prediction(values, heating, cooling):
    user_id = get_user_id()
    if not user_id:
        raise RuntimeError("Your session has expired. Please sign in again.")
    record = {"user_id": str(user_id)}
    for key, _ in FEATURES:
        record[key] = int(values[key]) if key in ("orientation", "glazing_distribution") else float(values[key])
    record.update({"heating_load": float(heating), "cooling_load": float(cooling)})
    st.session_state.supabase.table("predictions").insert(record).execute()


def fetch_predictions(limit=100):
    user_id = get_user_id()
    if not user_id:
        return []
    response = (st.session_state.supabase.table("predictions").select("*")
                .eq("user_id", str(user_id)).order("created_at", desc=True).limit(limit).execute())
    return response.data or []


def preferences():
    saved = user_metadata().get("archiload_preferences", {})
    if not isinstance(saved, dict):
        saved = {}
    defaults = {"language": "English", "default_page": "Overview", "parameter_guidance": True,
                "product_updates": False, "getting_started_tips": True}
    return {**defaults, **saved}


def update_user_metadata(updates):
    client = st.session_state.supabase
    meta = user_metadata()
    meta.update(updates)
    response = client.auth.update_user({"data": meta})
    updated_user = value_of(response, "user")
    if updated_user is not None:
        st.session_state.user = updated_user
    return updated_user


def render_auth(client):
    st.markdown(f"""<style>
    [data-testid="stSidebar"] {{display:none;}}
    .auth-kicker {{color:{TEAL};font-size:.74rem;font-weight:800;letter-spacing:.13em;text-transform:uppercase;}}
    .auth-art {{
        background: linear-gradient(180deg, rgba(20,43,61,0.85) 0%, rgba(20,43,61,0.95) 100%), 
                    radial-gradient(circle at top right, rgba(0,131,143,0.3), transparent 60%);
        border-radius:18px;
        padding:2.5rem;
        color:white;
        min-height:580px;
        display:flex;
        flex-direction:column;
        justify-content:space-between;
    }}
    .auth-art h2,.auth-art h3,.auth-art p {{color:white;}}
    .auth-art p {{color:#C9D8E1;line-height:1.65;}}
    .auth-
