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
    .auth-title {{font-size:2rem;font-weight:800;line-height:1.15;letter-spacing:-.035em;color:{NAVY};margin-bottom:.35rem;}}
    </style>""", unsafe_allow_html=True)

    if "auth_mode" not in st.session_state:
        st.session_state.auth_mode = "Sign in"

    left, right = st.columns([1.05, 0.95], gap="large", vertical_alignment="center")
    
    with left:
        st.markdown(f"""
        <div class="auth-art">
            <div>
                <div style="display:flex; align-items:center; gap:.5rem; font-weight:800; font-size:1.15rem; color:white; letter-spacing:-.02em;">
                    <span style="background:{TEAL}; width:24px; height:24px; border-radius:6px; display:inline-flex; align-items:center; justify-content:center; font-size:12px;">🏗️</span> ArchiLoad AI
                </div>
                <div style="margin-top:4rem;">
                    <div class="auth-kicker" style="color:#67D0D0;">EARLY-STAGE DESIGN EXPLORATION</div>
                    <h2 class="hero-title" style="color:white; font-size:2.3rem; margin:.4rem 0 1rem 0;">AI-Based Heating and<br>Cooling Load Predictor</h2>
                    <p>Explore how early-stage building design parameters influence predicted heating and cooling loads.</p>
                </div>
            </div>
            <div style="font-size:.78rem; color:#A2B6C3; border-top:1px solid rgba(255,255,255,0.12); padding-top:1.2rem; margin-top:2rem;">
                For preliminary exploration. Not a substitute for detailed energy simulation or professional engineering assessment.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with right:
        mode = st.session_state.auth_mode
        col_lbl, col_btn = st.columns([1.3, 0.9])
        with col_lbl:
            if mode == "Sign in":
                st.markdown('<span style="font-size:.84rem; color:#647789;">New to ArchiLoad AI?</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span style="font-size:.84rem; color:#647789;">Already have an account?</span>', unsafe_allow_html=True)
        with col_btn:
            if mode == "Sign in":
                if st.button("Create an account", use_container_width=True):
                    st.session_state.auth_mode = "Create account"
                    st.rerun()
            else:
                if st.button("Sign in", use_container_width=True):
                    st.session_state.auth_mode = "Sign in"
                    st.rerun()

        st.write("")
        if mode == "Sign in":
            st.markdown('<div class="auth-title">Welcome back</div>', unsafe_allow_html=True)
            st.markdown('<p class="subtle">Sign in to explore building configurations and access your saved predictions.</p>', unsafe_allow_html=True)
            
            with st.form("login_form"):
                email = st.text_input("Email address", placeholder="alex.lee@example.com")
                password = st.text_input("Password", type="password", placeholder="••••••••••••")
                
                col_forgot, col_dummy = st.columns([1, 1])
                with col_forgot:
                    if st.form_submit_button("Forgot password?", type="secondary"):
                        st.session_state.auth_mode = "Reset password"
                        st.rerun()
                
                submitted = st.form_submit_button("Sign in", type="primary", use_container_width=True)
            
            if submitted:
                if not email.strip() or not password:
                    st.warning("Enter your email address and password.")
                else:
                    try:
                        response = client.auth.sign_in_with_password({"email": email.strip(), "password": password})
                        user = set_user_from_auth_response(response)
                        if user is not None:
                            default_page = preferences().get("default_page", "Overview")
                            st.session_state.page = default_page if default_page in ["Overview", "Prediction", "About", "Settings", "Account"] else "Overview"
                            st.rerun()
                        st.error("Sign-in did not return a user. Check your Supabase Auth settings.")
                    except Exception as exc:
                        st.error(f"Could not sign in. Check your credentials. Details: {exc}")
            
            st.markdown('<p class="small-muted" style="margin-top:1rem;">Your predictions stay connected to your account.</p>', unsafe_allow_html=True)

        elif mode == "Create account":
            st.markdown('<div class="auth-title">Create an account</div>', unsafe_allow_html=True)
            st.markdown('<p class="subtle">Get started with ArchiLoad AI workspace.</p>', unsafe_allow_html=True)
            
            with st.form("signup_form"):
                first = st.text_input("First name", placeholder="Alex")
                last = st.text_input("Last name", placeholder="Lee")
                email = st.text_input("Email address", placeholder="alex.lee@example.com")
                password = st.text_input("Create password", type="password", placeholder="At least 6 characters")
                submitted = st.form_submit_button("Create account", type="primary", use_container_width=True)
            
            if submitted:
                if not all([first.strip(), last.strip(), email.strip(), password]):
                    st.warning("Complete all fields.")
                elif len(password) < 6:
                    st.warning("Use a password with at least 6 characters.")
                else:
                    try:
                        response = client.auth.sign_up({
                            "email": email.strip(), 
                            "password": password,
                            "options": {"data": {"first_name": first.strip(), "last_name": last.strip()}}
                        })
                        user = set_user_from_auth_response(response)
                        if user is not None and value_of(response, "session") is not None:
                            st.success("Account created. You are signed in.")
                            st.rerun()
                        st.success("Account request submitted. Confirm your email before signing in.")
                    except Exception as exc:
                        st.error(f"Could not create account. Details: {exc}")

        else:
            st.markdown('<div class="auth-title">Reset password</div>', unsafe_allow_html=True)
            st.markdown('<p class="subtle">We will send a password-reset link to your email address.</p>', unsafe_allow_html=True)
            
            with st.form("reset_form"):
                email = st.text_input("Email address", placeholder="alex.lee@example.com")
                col_back, col_submit = st.columns([1, 1])
                with col_back:
                    if st.form_submit_button("Back to sign in"):
                        st.session_state.auth_mode = "Sign in"
                        st.rerun()
                submitted = st.form_submit_button("Send reset link", type="primary", use_container_width=True)
            
            if submitted:
                if not email.strip():
                    st.warning("Enter your email address.")
                else:
                    try:
                        client.auth.reset_password_for_email(email.strip())
                        st.success("If the address belongs to an account, a reset email should arrive shortly.")
                    except Exception as exc:
                        st.error(f"Could not request a reset email. Details: {exc}")

        st.markdown(
            f'<div style="display:flex; justify-content:space-between; align-items:center; margin-top:2rem; font-size:.78rem; color:{MUTED}; border-top:1px solid {BORDER}; padding-top:1rem;">'
            f'<span>© 2026 ArchiLoad AI</span><span>Privacy · Terms</span>'
            f'</div>',
            unsafe_allow_html=True
        )


def render_sidebar():
    pages = ["Overview", "Prediction", "About"]
    current = st.session_state.get("page", "Overview")
    if current not in pages and current not in ["Settings", "Account"]:
        current = "Overview"
    
    with st.sidebar:
        logo = ROOT / "archiloadai.png"
        if logo.exists():
            st.image(str(logo), width=125)
        else:
            st.markdown('<div style="display:flex; align-items:center; gap:.5rem; font-weight:800; font-size:1.15rem; color:white; letter-spacing:-.02em;"><span style="background:#00838F; width:24px; height:24px; border-radius:6px; display:inline-flex; align-items:center; justify-content:center; font-size:12px;">🏗️</span> ArchiLoad AI</div>', unsafe_allow_html=True)
        
        st.write("")
        st.markdown('<div class="sidebar-caption" style="letter-spacing:.12em;font-weight:800;margin-top:.5rem">PERSONAL WORKSPACE</div>', unsafe_allow_html=True)
        
        # Main radio navigation matching image_3
        selected = st.radio("Workspace navigation", pages, index=pages.index(current) if current in pages else 0, label_visibility="collapsed", key="sidebar_page")
        if selected != current and selected in pages:
            st.session_state.page = selected
            st.rerun()

        st.markdown("<br><br><br>", unsafe_allow_html=True)
        
        # Bottom sidebar card matching image_3 ("Better early-stage decisions")
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.08); border-radius:14px; padding:1.1rem; margin-bottom:1rem;">
            <div style="font-size:1.1rem; margin-bottom:.3rem;">🌱</div>
            <div style="font-weight:700; font-size:.85rem; color:white; margin-bottom:.3rem;">Better early-stage decisions</div>
            <div style="font-size:.75rem; color:#A2B6C3; line-height:1.5;">Explore building form, orientation, and glazing before detailed analysis.</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Footer nav items in sidebar (Settings, Account, Sign out)
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            if st.button("⚙️ Settings", use_container_width=True):
                st.session_state.page = "Settings"
                st.rerun()
        with col_s2:
            if st.button("👤 Account", use_container_width=True):
                st.session_state.page = "Account"
                st.rerun()
        with col_s3:
            if st.button("🚪 Sign out", use_container_width=True):
                sign_out()


def render_overview():
    # Top bar matching image_3
    st.markdown(f"""
    <div class="topbar">
        <div class="topbar-label">Workspace &nbsp;&nbsp;›&nbsp;&nbsp; <b>Overview</b></div>
        <div style="display:flex; align-items:center; gap:1rem;">
            <div class="profile-chip">
                <span class="profile-initials">AL</span>
                <span>Alex Lee</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Action button row using columns for "+ New prediction" button aligned right
    col_title, col_action = st.columns([3, 1])
    with col_title:
        st.markdown(f'<div class="hero-title" style="margin-top:0;">Overview</div>', unsafe_allow_html=True)
        st.markdown(f'<p class="subtle" style="margin-bottom:1.5rem;">Welcome, Alex. Start exploring the energy implications of your design.</p>', unsafe_allow_html=True)
    with col_action:
        st.write("")
        if st.button("+ New prediction", type="primary", use_container_width=True):
            go_to("Prediction")

    # Top row: 3 Metric Cards matching image_3
    m1, m2, m3 = st.columns(3, gap="medium")
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                <div class="metric-label">Dataset records</div>
                <span style="font-size:1.1rem; color:#647789;">🗄️</span>
            </div>
            <div class="metric-value">768</div>
            <div class="small-muted">Simulated building configurations</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                <div class="metric-label">Prediction outputs</div>
                <span style="font-size:1.1rem; color:#647789;">📈</span>
            </div>
            <div class="metric-value">2</div>
            <div class="small-muted">Heating load and cooling load</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                <div class="metric-label">Building inputs</div>
                <span style="font-size:1.1rem; color:#647789;">🎛️</span>
            </div>
            <div class="metric-value">8</div>
            <div class="small-muted">Form, dimensions, orientation &amp; glazing</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # Middle row: Explore alternatives & Getting started
    left, right = st.columns([1.22, .78], gap="large")
    with left:
        with st.container(border=True):
            st.markdown('<div class="eyebrow" style="margin-bottom:.3rem;">FROM BUILDING PARAMETERS TO EARLY INSIGHT</div>', unsafe_allow_html=True)
            st.markdown('<h3 style="margin-top:0; font-size:1.45rem;">Explore alternatives before detailed energy analysis.</h3>', unsafe_allow_html=True)
            st.markdown('<p class="subtle" style="font-size:.9rem;">ArchiLoad AI uses machine learning to estimate heating and cooling loads from selected building design parameters. Compare configurations to support informed early-stage architectural decisions.</p>', unsafe_allow_html=True)
            
            # Flow diagram steps
            st.markdown(f"""
            <div style="display:flex; align-items:center; justify-content:space-between; background:#F8FAFB; border:1px solid {BORDER}; border-radius:12px; padding:1rem 1.25rem; margin:1.25rem 0;">
                <div style="text-align:center;">
                    <div style="font-size:1.1rem; margin-bottom:.2rem;">🏢</div>
                    <div style="font-size:.8rem; font-weight:700; color:{NAVY};">Building inputs</div>
                </div>
                <div style="color:{MUTED}; font-weight:600;">→</div>
                <div style="text-align:center;">
                    <div style="font-size:1.1rem; margin-bottom:.2rem;">⚙️</div>
                    <div style="font-size:.8rem; font-weight:700; color:{NAVY};">Regression models</div>
                </div>
                <div style="color:{MUTED}; font-weight:600;">→</div>
                <div style="text-align:center;">
                    <div style="font-size:1.1rem; margin-bottom:.2rem;">🌡️</div>
                    <div style="font-size:.8rem; font-weight:700; color:{NAVY};">Load estimates</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("→ Explore a building configuration", type="primary", use_container_width=True):
                go_to("Prediction")

    with right:
        with st.container(border=True):
            st.markdown('<h3 style="margin-top:0; font-size:1.25rem; margin-bottom:1rem;">Getting started</h3>', unsafe_allow_html=True)
            
            steps_data = [
                ("01", "Define your building", "Enter all eight parameters for the configuration you want to explore."),
                ("02", "Generate estimates", "Review the estimated heating and cooling loads together."),
                ("03", "Save and revisit", "Registered users can save and review their own prediction history.")
            ]
            for num, title, desc in steps_data:
                st.markdown(f"""
                <div style="display:flex; gap:.85rem; margin-bottom:.9rem; align-items:flex-start;">
                    <span style="background:#E3F4F3; color:{TEAL_DARK}; font-weight:800; font-size:.75rem; padding:.2rem .45rem; border-radius:6px; min-width:24px; text-align:center;">{num}</span>
                    <div>
                        <div style="font-weight:700; font-size:.9rem; color:{NAVY};">{title}</div>
                        <div style="font-size:.8rem; color:{MUTED}; line-height:1.4;">{desc}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    st.write("")

    # Bottom row: Dataset info & Prediction history
    b_left, b_right = st.columns(2, gap="large")
    with b_left:
        with st.container(border=True):
            st.markdown('<h3 style="margin-top:0; font-size:1.2rem; margin-bottom:.5rem;">Built on the UCI Energy Efficiency dataset</h3>', unsafe_allow_html=True)
            st.markdown('<p class="subtle" style="font-size:.85rem;">768 simulated configurations describe building form, dimensions, orientation, and glazing—not measured building consumption.</p>', unsafe_allow_html=True)
            if st.button("About the dataset and models →", use_container_width=False):
                go_to("About")
    with b_right:
        with st.container(border=True):
            st.markdown('<h3 style="margin-top:0; font-size:1.2rem; margin-bottom:.5rem;">Your prediction history</h3>', unsafe_allow_html=True)
            st.markdown('<p class="subtle" style="font-size:.85rem;">No saved predictions yet. Start with a configuration, then return to review your saved estimates.</p>', unsafe_allow_html=True)
            if st.button("Go to prediction →", use_container_width=False):
                go_to("Prediction")

    # Bottom banner notice matching image_3
    st.markdown(f"""
    <div style="background:#EBF8F9; border:1px solid #BCE3E5; border-radius:14px; padding:1.15rem 1.4rem; margin-top:1.5rem; display:flex; align-items:center; gap:1rem;">
        <span style="font-size:1.25rem;">ℹ️</span>
        <div>
            <div style="font-weight:800; font-size:.85rem; color:{NAVY};">Preliminary estimates only</div>
            <div style="font-size:.8rem; color:#31545D; margin-top:.1rem;">Model-based estimates for early-stage design exploration, not final engineering calculations or building energy certification.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Footer matching image_3
    st.markdown(f"""
    <div style="display:flex; justify-content:space-between; align-items:center; margin-top:2rem; padding-top:1rem; border-top:1px solid {BORDER}; font-size:.8rem; color:{MUTED};">
        <span>ArchiLoad AI - Early-stage building energy exploration</span>
        <span>UCI Energy Efficiency dataset</span>
    </div>
    """, unsafe_allow_html=True)


def reset_prediction_inputs():
    for key, value in DEFAULT_INPUTS.items():
        st.session_state[f"prediction_{key}"] = value


def render_prediction():
    for key, value in DEFAULT_INPUTS.items():
        st.session_state.setdefault(f"prediction_{key}", value)
    
    st.markdown(f"""
    <div class="topbar">
        <div class="topbar-label">Workspace &nbsp;&nbsp;›&nbsp;&nbsp; <b>Prediction</b></div>
        <div class="profile-chip"><span class="profile-initials">AL</span><span>Alex Lee</span></div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="hero-title" style="margin-top:0;">Prediction</div>', unsafe_allow_html=True)
    st.markdown('<p class="subtle">Adjust the eight building parameters to explore a configuration.</p>', unsafe_allow_html=True)
    st.markdown('<div class="note">Use categories represented in the UCI Energy Efficiency dataset. Results are estimates for comparison, not final engineering calculations.</div>', unsafe_allow_html=True)
    st.write("")
    
    left, right = st.columns([1.08, .92], gap="large")
    with left:
        with st.container(border=True):
            st.markdown('<h3 style="margin-top:0;">Building Design Parameters</h3>', unsafe_allow_html=True)
            st.markdown('<p class="subtle">Example values are ready to edit. All eight inputs are required.</p>', unsafe_allow_html=True)
            with st.form("prediction_form"):
                x, y = st.columns(2, gap="medium")
                with x:
                    st.markdown(f'<div class="section-label">Form &amp; dimensions</div>', unsafe_allow_html=True)
                    relative_compactness = st.number_input("Relative Compactness", .60, 1.00, step=.01, format="%.2f", key="prediction_relative_compactness")
                    surface_area = st.number_input("Surface Area (m²)", 500.0, 820.0, step=1.5, format="%.2f", key="prediction_surface_area")
                    wall_area = st.number_input("Wall Area (m²)", 240.0, 420.0, step=1.0, format="%.2f", key="prediction_wall_area")
                    roof_area = st.number_input("Roof Area (m²)", 100.0, 225.0, step=1.0, format="%.2f", key="prediction_roof_area")
                with y:
                    st.markdown(f'<div class="section-label">Height, orientation &amp; glazing</div>', unsafe_allow_html=True)
                    overall_height = st.selectbox("Overall Height (m)", [3.5, 7.0], key="prediction_overall_height")
                    orientation = st.selectbox("Orientation", [2, 3, 4, 5], key="prediction_orientation")
                    glazing_area = st.selectbox("Glazing Area", [0.00, 0.10, 0.25, 0.40], format_func=lambda v: f"{v:.2f}", key="prediction_glazing_area")
                    glazing_distribution = st.selectbox("Glazing Area Distribution", [0, 1, 2, 3, 4, 5], key="prediction_glazing_distribution")
                submitted = st.form_submit_button("Predict Heating and Cooling Loads", type="primary", use_container_width=True)
            
            reset_col, hint_col = st.columns([.7, 1.3])
            with reset_col:
                st.button("Reset inputs", on_click=reset_prediction_inputs, use_container_width=True)
            with hint_col:
                st.caption("Use one change at a time to compare configurations.")
            
            if submitted:
                values = {"relative_compactness": relative_compactness, "surface_area": surface_area, "wall_area": wall_area,
                          "roof_area": roof_area, "overall_height": overall_height, "orientation": orientation,
                          "glazing_area": glazing_area, "glazing_distribution": glazing_distribution}
                try:
                    with st.spinner("Calculating heating and cooling load estimates…"):
                        heating, cooling = predict_loads(values)
                        save_prediction(values, heating, cooling)
                    st.session_state.last_prediction = {"values": values, "heating": heating, "cooling": cooling,
                                                        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                    st.success("Prediction generated and saved to your account.")
                except Exception as exc:
                    st.error("The prediction could not be completed.")
                    st.exception(exc)

    with right:
        with st.container(border=True):
            st.markdown('<h3 style="margin-top:0;">Load estimates</h3>', unsafe_allow_html=True)
            result = st.session_state.get("last_prediction")
            if result:
                r1, r2 = st.columns(2)
                r1.metric("Heating load", f"{result['heating']:.2f}")
                r2.metric("Cooling load", f"{result['cooling']:.2f}")
                st.caption(f"Generated: {result.get('created_at', 'recently')}")
                st.markdown('<div class="warning" style="margin-top:1rem;">Model estimates are for relative comparison. They do not represent every site, weather, or HVAC condition.</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="text-align:center;padding:1.5rem .5rem"><div class="section-label">Awaiting prediction</div><h3 style="margin-top:1rem">Your estimates will appear here</h3><p class="subtle">Review the building parameters, then run a prediction to estimate heating and cooling loads.</p></div>', unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<h3 style="margin-top:0;">My prediction history</h3>', unsafe_allow_html=True)
        try:
            rows = fetch_predictions(100)
            if rows:
                history = pd.DataFrame(rows)
                if "created_at" in history:
                    history["created_at"] = pd.to_datetime(history["created_at"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
                preferred = ["created_at", "relative_compactness", "surface_area", "wall_area", "roof_area", "overall_height", "orientation", "glazing_area", "glazing_distribution", "heating_load", "cooling_load"]
                cols = [c for c in preferred if c in history.columns]
                st.dataframe(history[cols], use_container_width=True, hide_index=True)
                st.download_button("Download my history as CSV", history[cols].to_csv(index=False).encode("utf-8"), "archiload_prediction_history.csv", "text/csv")
            else:
                st.caption("No saved predictions yet. Generate a prediction to start your history.")
        except Exception as exc:
            st.warning("Prediction history could not be loaded.")
            st.caption(str(exc))
    
    st.markdown(f"""
    <div style="display:flex; justify-content:space-between; align-items:center; margin-top:2rem; padding-top:1rem; border-top:1px solid {BORDER}; font-size:.8rem; color:{MUTED};">
        <span>ArchiLoad AI - Early-stage building energy exploration</span>
        <span>UCI Energy Efficiency dataset</span>
    </div>
    """, unsafe_allow_html=True)


def render_about():
    st.markdown(f"""
    <div class="topbar">
        <div class="topbar-label">Workspace &nbsp;&nbsp;›&nbsp;&nbsp; <b>About</b></div>
        <div class="profile-chip"><span class="profile-initials">AL</span><span>Alex Lee</span></div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="hero-title" style="margin-top:0;">About ArchiLoad AI</div>', unsafe_allow_html=True)
    st.markdown('<p class="subtle">Understand the data, models, and limitations behind every estimate.</p>', unsafe_allow_html=True)
    
    tabs = st.tabs(["Dataset & models", "Model evaluation", "Design relevance", "Responsible use"])
    with tabs[0]:
        a, b = st.columns(2, gap="large")
        with a:
            with st.container(border=True):
                st.markdown('<h3 style="margin-top:0;">Dataset</h3>', unsafe_allow_html=True)
                st.markdown("**UCI Energy Efficiency dataset**")
                st.markdown("- 768 simulated building configurations\n- 8 building-form and glazing inputs\n- 2 targets: heating load and cooling load")
        with b:
            with st.container(border=True):
                st.markdown('<h3 style="margin-top:0;">Trained regressors</h3>', unsafe_allow_html=True)
                st.markdown("**Scikit-learn pipeline**")
                st.markdown("- `heating_model.pkl`\n- `cooling_model.pkl`\n- `feature_columns.pkl`")
    with tabs[1]:
        with st.container(border=True):
            st.markdown('<h3 style="margin-top:0;">Model evaluation metrics</h3>', unsafe_allow_html=True)
            metrics_data = []
            for target, mvals in REPORTED_METRICS.items():
                metrics_data.append({"Target": target, "MAE": mvals["MAE"], "RMSE": mvals["RMSE"], "R²": mvals["R²"]})
            st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)
    with tabs[2]:
        with st.container(border=True):
            st.markdown('<h3 style="margin-top:0;">Design relevance & exploration</h3>', unsafe_allow_html=True)
            st.markdown("ArchiLoad AI is designed for early-stage conceptual exploration, helping designers evaluate compactness, surface area, and glazing fractions quickly.")
    with tabs[3]:
        with st.container(border=True):
            st.markdown('<h3 style="margin-top:0;">Responsible use & limitations</h3>', unsafe_allow_html=True)
            st.markdown("Estimates are for preliminary exploration and educational purposes only. They are not a substitute for professional engineering reviews or dynamic thermal simulations.")
    
    st.markdown(f"""
    <div style="display:flex; justify-content:space-between; align-items:center; margin-top:2rem; padding-top:1rem; border-top:1px solid {BORDER}; font-size:.8rem; color:{MUTED};">
        <span>ArchiLoad AI - Early-stage building energy exploration</span>
        <span>UCI Energy Efficiency dataset</span>
    </div>
    """, unsafe_allow_html=True)


def render_settings():
    st.markdown(f"""
    <div class="topbar">
        <div class="topbar-label">Workspace &nbsp;&nbsp;›&nbsp;&nbsp; <b>Settings</b></div>
        <div class="profile-chip"><span class="profile-initials">AL</span><span>Alex Lee</span></div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="hero-title" style="margin-top:0;">Settings</div>', unsafe_allow_html=True)
    st.markdown('<p class="subtle">Manage your workspace preferences and display options.</p>', unsafe_allow_html=True)
    
    prefs = preferences()
    with st.container(border=True):
        st.markdown('<h3 style="margin-top:0;">Workspace preferences</h3>', unsafe_allow_html=True)
        with st.form("settings_form"):
            default_page = st.selectbox("Default landing page", ["Overview", "Prediction", "About"],
                                         index=["Overview", "Prediction", "About"].index(prefs.get("default_page", "Overview")) if prefs.get("default_page") in ["Overview", "Prediction", "About"] else 0)
            parameter_guidance = st.checkbox("Show parameter hints and design tips", value=prefs.get("parameter_guidance", True))
            product_updates = st.checkbox("Receive occasional product updates", value=prefs.get("product_updates", False))
            saved = st.form_submit_button("Save preferences", type="primary")
        if saved:
            try:
                new_prefs = {**prefs, "default_page": default_page, "parameter_guidance": parameter_guidance, "product_updates": product_updates}
                update_user_metadata({"archiload_preferences": new_prefs})
                st.success("Preferences updated successfully.")
            except Exception as exc:
                st.error(f"Could not save preferences: {exc}")


def render_account():
    st.markdown(f"""
    <div class="topbar">
        <div class="topbar-label">Workspace &nbsp;&nbsp;›&nbsp;&nbsp; <b>Account</b></div>
        <div class="profile-chip"><span class="profile-initials">AL</span><span>Alex Lee</span></div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="hero-title" style="margin-top:0;">Account</div>', unsafe_allow_html=True)
    st.markdown('<p class="subtle">Manage your personal profile and account credentials.</p>', unsafe_allow_html=True)
    
    meta = user_metadata()
    with st.container(border=True):
        st.markdown('<h3 style="margin-top:0;">Profile details</h3>', unsafe_allow_html=True)
        with st.form("profile_form"):
            first_name = st.text_input("First name", value=meta.get("first_name", "Alex"))
            last_name = st.text_input("Last name", value=meta.get("last_name", "Lee"))
            email = st.text_input("Email address", value=current_email() or "alex.lee@example.com", disabled=True)
            submitted = st.form_submit_button("Update profile", type="primary")
        if submitted:
            try:
                update_user_metadata({"first_name": first_name.strip(), "last_name": last_name.strip()})
                st.success("Profile updated successfully.")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not update profile: {exc}")
    
    with st.container(border=True):
        st.markdown('<h3 style="margin-top:0;">Security</h3>', unsafe_allow_html=True)
        with st.form("password_form"):
            new_password = st.text_input("New password", type="password")
            confirm_password = st.text_input("Confirm new password", type="password")
            pass_submitted = st.form_submit_button("Update password")
        if pass_submitted:
            if not new_password:
                st.warning("Enter a new password.")
            elif new_password != confirm_password:
                st.warning("The passwords do not match.")
            elif len(new_password) < 6:
                st.warning("Use a password with at least 6 characters.")
            else:
                try:
                    client = st.session_state.get("supabase")
                    client.auth.update_user({"password": new_password})
                    st.success("Password updated successfully.")
                except Exception as exc:
                    st.error(f"Could not update password: {exc}")


def main():
    supabase_url = secret_value("SUPABASE_URL")
    supabase_key = secret_value("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        st.error("Missing Supabase configuration. Set SUPABASE_URL and SUPABASE_KEY in your Streamlit secrets.")
        return

    client = get_supabase_client(supabase_url, supabase_key)
    st.session_state.supabase = client

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        render_auth(client)
        return

    render_sidebar()
    page = st.session_state.get("page", "Overview")
    if page == "Overview":
        render_overview()
    elif page == "Prediction":
        render_prediction()
    elif page == "About":
        render_about()
    elif page == "Settings":
        render_settings()
    elif page == "Account":
        render_account()
    else:
        render_overview()


if __name__ == "__main__":
    main()
