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
TEAL = "#00858B"
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
[data-testid="stSidebar"] {{ background:{NAVY}; border-right:1px solid #25465B; }}
[data-testid="stSidebar"] * {{ color:#F7FAFC; }}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{ color:#C9D8E1; }}
[data-testid="stSidebar"] hr {{ border-color:#365367; }}
[data-testid="stSidebar"] [data-testid="stRadio"] label {{ padding:.45rem .55rem; border-radius:9px; }}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {{ background:#234357; }}
h1,h2,h3 {{ color:{NAVY}; letter-spacing:-.025em; }}
h1 {{ font-weight:800; }}
.eyebrow {{ color:{TEAL}; font-size:.74rem; font-weight:800; text-transform:uppercase; letter-spacing:.13em; margin-bottom:.25rem; }}
.hero-title {{ color:{NAVY}; font-size:clamp(2rem,3vw,2.75rem); line-height:1.12; font-weight:800; letter-spacing:-.045em; margin:.15rem 0 .55rem; }}
.subtle {{ color:{MUTED}; line-height:1.65; margin-top:0; }}
.sidebar-brand {{ font-size:1.35rem; font-weight:800; color:white; margin:.35rem 0 .15rem; letter-spacing:-.03em; }}
.sidebar-caption {{ font-size:.78rem; color:#C9D8E1; line-height:1.5; }}
.card {{ background:{WHITE}; border:1px solid {BORDER}; border-radius:18px; padding:1.15rem 1.25rem; box-shadow:0 5px 22px rgba(20,43,61,.035); }}
.metric-card {{ background:{WHITE}; border:1px solid {BORDER}; border-radius:16px; padding:1.15rem 1.25rem; min-height:122px; box-shadow:0 4px 18px rgba(20,43,61,.025); }}
.metric-label {{ color:{MUTED}; font-size:.84rem; font-weight:600; }}
.metric-value {{ color:{NAVY}; font-size:1.9rem; font-weight:800; margin:.25rem 0 .15rem; letter-spacing:-.04em; }}
.pill {{ display:inline-block; background:#E3F4F3; color:{TEAL_DARK}; padding:.32rem .68rem; border-radius:99px; font-size:.76rem; font-weight:750; }}
.note {{ border-left:4px solid {TEAL}; background:#EAF5F5; border-radius:9px; padding:.85rem 1rem; color:#31545D; }}
.warning {{ border-left:4px solid #D59B35; background:#FFF7E7; border-radius:9px; padding:.85rem 1rem; color:#6B5123; }}
.small-muted {{ color:{MUTED}; font-size:.8rem; }}
.section-label {{ font-size:.78rem; color:{MUTED}; text-transform:uppercase; letter-spacing:.09em; font-weight:750; }}
.topbar {{ display:flex; align-items:center; justify-content:space-between; gap:1rem; padding:.15rem 0 .9rem; margin-bottom:1rem; border-bottom:1px solid {BORDER}; }}
.topbar-label {{ color:{MUTED}; font-size:.83rem; font-weight:600; }}
.profile-chip {{ display:flex; align-items:center; gap:.55rem; color:{INK}; font-size:.84rem; font-weight:700; }}
.profile-initials {{ width:2rem; height:2rem; border-radius:50%; background:#E3F4F3; color:{TEAL_DARK}; display:inline-flex; align-items:center; justify-content:center; font-weight:800; }}
.stButton > button, .stDownloadButton > button {{ border-radius:10px; font-weight:650; min-height:2.65rem; border:1px solid {BORDER}; }}
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {{ background:{TEAL}; border-color:{TEAL}; color:white; }}
.stButton > button[kind="primary"]:hover {{ background:{TEAL_DARK}; border-color:{TEAL_DARK}; }}
[data-testid="stMetric"] {{ background:white; border:1px solid {BORDER}; padding:1rem; border-radius:14px; }}
div[data-testid="stForm"] {{ background:white; border:1px solid {BORDER}; border-radius:16px; padding:1.1rem 1.2rem; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ background:white; border-radius:17px; }}
[data-testid="stTabs"] button {{ font-weight:650; }}
[data-testid="stDataFrame"] {{ border:1px solid {BORDER}; border-radius:12px; overflow:hidden; }}
footer {{ visibility:hidden; }}
#MainMenu {{ visibility:hidden; }}
@media (max-width: 760px) {{ .hero-title {{ font-size:2rem; }} }}
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
# Verify these figures against evaluation of the exact deployed model files before formal reporting.
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
    return "Your workspace"


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
    for key in ["user", "authenticated", "last_prediction", "auth_message", "sidebar_page"]:
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


def page_header(kicker, title, subtitle):
    """Render the workspace breadcrumb/profile row and the page heading."""
    meta = user_metadata()
    first = str(meta.get("first_name", "")).strip()
    last = str(meta.get("last_name", "")).strip()
    initials = ((first[:1] + last[:1]) or current_email()[:1] or "A").upper()
    name = display_name()
    page_name = kicker.split("/")[-1].strip().title() if "/" in kicker else kicker
    st.markdown(
        f'<div class="topbar"><div class="topbar-label">Workspace&nbsp;&nbsp; / &nbsp;&nbsp;{page_name}</div>'
        f'<div class="profile-chip"><span class="profile-initials">{initials}</span><span>{name}</span></div></div>'
        f'<div class="eyebrow">{kicker}</div><div class="hero-title">{title}</div>'
        f'<p class="subtle">{subtitle}</p>',
        unsafe_allow_html=True,
    )


def card_title(title, subtitle=None):
    st.markdown(f"### {title}")
    if subtitle:
        st.markdown(f'<p class="subtle">{subtitle}</p>', unsafe_allow_html=True)


def footer():
    st.markdown(f'<hr style="border:0;border-top:1px solid {BORDER};margin-top:1.7rem"><p class="small-muted">ArchiLoad AI · Early-stage design support, not a substitute for detailed energy simulation or professional review.</p>', unsafe_allow_html=True)


def go_to(page):
    st.session_state.page = page
    st.rerun()


def render_auth(client):
    # Streamlit widgets are placed alongside the custom HTML panel to keep them functional.
    st.markdown(f"""<style>
    [data-testid="stSidebar"] {{display:none;}}
    .auth-kicker {{color:{TEAL};font-size:.75rem;font-weight:800;letter-spacing:.13em;text-transform:uppercase;}}
    .auth-title {{font-size:2.15rem;font-weight:800;line-height:1.12;letter-spacing:-.04em;color:{NAVY};}}
    .auth-art {{background:linear-gradient(145deg,{NAVY} 0%,{NAVY_2} 100%);border-radius:22px;padding:2rem;color:white;min-height:490px;}}
    .auth-art h2,.auth-art h3,.auth-art p {{color:white;}}
    .auth-art p {{color:#D7E4EB;line-height:1.7;}}
    .auth-architecture {{margin-top:2rem;padding:1rem;border:1px solid #496477;border-radius:14px;background:rgba(255,255,255,.04);}}
    </style>""", unsafe_allow_html=True)
    left, right = st.columns([1.02, .98], gap="large", vertical_alignment="center")
    with left:
        st.markdown(f'<div class="auth-art"><div class="auth-kicker" style="color:#67D0D0">BUILDING PERFORMANCE · AI</div><h2 style="font-size:2.25rem;line-height:1.12;margin-top:1.4rem">Design smarter,<br>earlier.</h2><h3 style="font-size:1.05rem;font-weight:600">AI-Based Heating and Cooling Load Predictor</h3><p>Explore how building form, orientation, and glazing choices may influence estimated heating and cooling loads.</p><div class="auth-architecture"><div style="font-size:.76rem;letter-spacing:.1em;color:#8FE0DE;font-weight:700">DESIGN EXPLORATION</div><p style="margin:.5rem 0 0">Compare early concepts. Understand potential energy implications. Make more informed design decisions.</p></div><p style="font-size:.8rem;margin-top:1.6rem">Educational use only · Validate important decisions with detailed simulation.</p></div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="auth-kicker">ARCHILOAD AI WORKSPACE</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-title">Welcome to your<br>design workspace.</div>', unsafe_allow_html=True)
        st.markdown('<p class="subtle">Sign in or create an account to save and revisit your building load estimates.</p>', unsafe_allow_html=True)
        mode = st.radio("Account options", ["Sign in", "Create account", "Reset password"], horizontal=True, label_visibility="collapsed", key="auth_mode")
        if mode == "Sign in":
            with st.form("login_form"):
                email = st.text_input("Email address", placeholder="you@example.com")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
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
                        st.error(f"Could not sign in. Check your credentials and Supabase Auth settings. Details: {exc}")
        elif mode == "Create account":
            with st.form("signup_form"):
                first = st.text_input("First name")
                last = st.text_input("Last name")
                email = st.text_input("Email address", placeholder="you@example.com")
                password = st.text_input("Create password", type="password")
                confirm = st.text_input("Confirm password", type="password")
                submitted = st.form_submit_button("Create account", type="primary", use_container_width=True)
            if submitted:
                if not all([first.strip(), last.strip(), email.strip(), password]):
                    st.warning("Complete all fields.")
                elif password != confirm:
                    st.warning("The passwords do not match.")
                elif len(password) < 6:
                    st.warning("Use a password with at least 6 characters.")
                else:
                    try:
                        response = client.auth.sign_up({"email": email.strip(), "password": password,
                            "options": {"data": {"first_name": first.strip(), "last_name": last.strip()}}})
                        user = set_user_from_auth_response(response)
                        if user is not None and value_of(response, "session") is not None:
                            st.success("Account created. You are signed in.")
                            st.rerun()
                        st.success("Account request submitted. If email confirmation is enabled, confirm your email before signing in.")
                    except Exception as exc:
                        st.error(f"Could not create account. Details: {exc}")
        else:
            st.write("We'll send a password-reset link to your email address.")
            with st.form("reset_form"):
                email = st.text_input("Email address", placeholder="you@example.com")
                submitted = st.form_submit_button("Send reset link", type="primary", use_container_width=True)
            if submitted:
                if not email.strip():
                    st.warning("Enter your email address.")
                else:
                    try:
                        client.auth.reset_password_for_email(email.strip())
                        st.success("If the address belongs to an account, a reset email should arrive shortly. Check your inbox and spam folder.")
                    except Exception as exc:
                        st.error(f"Could not request a reset email. Details: {exc}")
        st.markdown('<p class="small-muted">By continuing, you acknowledge that ArchiLoad AI provides preliminary educational estimates only.</p>', unsafe_allow_html=True)


def render_sidebar():
    pages = ["Overview", "Prediction", "About", "Settings", "Account"]
    current = st.session_state.get("page", "Overview")
    if current not in pages:
        current = "Overview"
    with st.sidebar:
        logo = ROOT / "archiloadai.png"
        if logo.exists():
            st.image(str(logo), width=125)
        st.markdown('<div class="sidebar-brand">ArchiLoad AI</div><div class="sidebar-caption">Better early-stage building decisions.</div>', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown('<div class="sidebar-caption" style="letter-spacing:.12em;font-weight:800">WORKSPACE</div>', unsafe_allow_html=True)
        selected = st.radio("Workspace navigation", pages, index=pages.index(current), label_visibility="collapsed", key="sidebar_page")
        if selected != current:
            st.session_state.page = selected
            st.rerun()
        st.markdown("---")
        st.markdown(f"**{display_name()}**")
        st.caption(current_email())
        if st.button("↪  Sign out", use_container_width=True):
            sign_out()
        st.markdown("---")
        st.markdown('<div class="sidebar-caption">Use estimates to compare options—not to certify a design.</div>', unsafe_allow_html=True)


def render_overview():
    page_header("WORKSPACE / OVERVIEW", "Overview", "Welcome to ArchiLoad AI. Start exploring the energy implications of your design.")
    a, b, c = st.columns(3, gap="medium")
    cards = [("DATASET RECORDS", "768", "Simulated building configurations"), ("PREDICTION OUTPUTS", "02", "Heating load and cooling load"), ("BUILDING INPUTS", "08", "Form, orientation, and glazing")]
    for col, (label, number, caption) in zip([a, b, c], cards):
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{number}</div><div class="small-muted">{caption}</div></div>', unsafe_allow_html=True)
    st.write("")
    left, right = st.columns([1.22, .78], gap="large")
    with left:
        with st.container(border=True):
            card_title("From building parameters to early insight", "Explore alternatives before detailed energy analysis.")
            cols = st.columns(3)
            steps = [("01", "Set inputs", "Define building form, orientation, and glazing."), ("02", "Run the model", "The trained regressors evaluate your parameters."), ("03", "Review results", "Compare heating and cooling load estimates.")]
            for col, (num, title, desc) in zip(cols, steps):
                with col:
                    st.markdown(f'<div style="color:{TEAL};font-weight:800;font-size:.78rem">STEP {num}</div><h4 style="margin:.25rem 0">{title}</h4><p class="small-muted">{desc}</p>', unsafe_allow_html=True)
            st.markdown('<div class="note">Tip: change one parameter at a time to explore how design choices affect the estimates.</div>', unsafe_allow_html=True)
            st.write("")
            if st.button("Start a new prediction  →", type="primary", use_container_width=True):
                go_to("Prediction")
        with st.container(border=True):
            card_title("Getting started")
            st.markdown("**01** Open Prediction and enter the building parameters.")
            st.markdown("**02** Generate heating and cooling load estimates.")
            st.markdown("**03** Save and revisit your personal prediction history.")
    with right:
        with st.container(border=True):
            card_title("About the dataset")
            st.markdown("The **UCI Energy Efficiency dataset** contains 768 simulated building configurations, eight input features, and two load targets.")
            st.markdown('<span class="pill">Educational prototype</span>', unsafe_allow_html=True)
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("**Model outputs**")
            st.caption("Heating load · Cooling load")
        with st.container(border=True):
            card_title("Your prediction history")
            try:
                rows = fetch_predictions(5)
                if rows:
                    recent = pd.DataFrame(rows)
                    cols = [x for x in ["created_at", "heating_load", "cooling_load"] if x in recent.columns]
                    if "created_at" in recent:
                        recent["created_at"] = pd.to_datetime(recent["created_at"], errors="coerce").dt.strftime("%b %d, %Y · %H:%M")
                    st.dataframe(recent[cols], use_container_width=True, hide_index=True)
                    if st.button("View all history", use_container_width=True):
                        go_to("Prediction")
                else:
                    st.caption("No saved predictions yet. Your estimates will appear here after your first prediction.")
            except Exception as exc:
                st.warning("Could not load saved predictions. Check your Supabase table and Row Level Security policies.")
                st.caption(str(exc))
    footer()


def reset_prediction_inputs():
    """Reset prediction widgets to the example values shown in the design."""
    for key, value in DEFAULT_INPUTS.items():
        st.session_state[f"prediction_{key}"] = value


def render_prediction():
    # Initialize keyed widget state so the Reset inputs button can reliably restore defaults.
    for key, value in DEFAULT_INPUTS.items():
        st.session_state.setdefault(f"prediction_{key}", value)
    page_header("WORKSPACE / PREDICTION", "Prediction", "Adjust the eight building parameters to explore a configuration.")
    st.markdown('<div class="note">Use categories represented in the UCI Energy Efficiency dataset. Results are estimates for comparison, not final engineering calculations.</div>', unsafe_allow_html=True)
    st.write("")
    left, right = st.columns([1.08, .92], gap="large")
    with left:
        with st.container(border=True):
            card_title("Building Design Parameters", "Example values are ready to edit. All eight inputs are required.")
            with st.form("prediction_form"):
                x, y = st.columns(2, gap="medium")
                with x:
                    st.markdown(f'<div class="section-label">Form &amp; dimensions</div>', unsafe_allow_html=True)
                    relative_compactness = st.number_input("Relative Compactness", .60, 1.00, step=.01, format="%.2f", help="Building form compactness", key="prediction_relative_compactness")
                    st.caption("Building form compactness")
                    surface_area = st.number_input("Surface Area (m²)", 500.0, 820.0, step=1.5, format="%.2f", key="prediction_surface_area")
                    st.caption("Total building surface area")
                    wall_area = st.number_input("Wall Area (m²)", 240.0, 420.0, step=1.0, format="%.2f", key="prediction_wall_area")
                    st.caption("Exterior wall area")
                    roof_area = st.number_input("Roof Area (m²)", 100.0, 225.0, step=1.0, format="%.2f", key="prediction_roof_area")
                    st.caption("Total roof area")
                with y:
                    st.markdown(f'<div class="section-label">Height, orientation &amp; glazing</div>', unsafe_allow_html=True)
                    overall_height = st.selectbox("Overall Height (m)", [3.5, 7.0], help="Building height", key="prediction_overall_height")
                    st.caption("Building height")
                    orientation = st.selectbox("Orientation", [2, 3, 4, 5], help="Dataset orientation code", key="prediction_orientation")
                    st.caption("Dataset orientation code")
                    glazing_area = st.selectbox("Glazing Area", [0.00, 0.10, 0.25, 0.40], format_func=lambda v: f"{v:.2f}", help="Glazing area proportion", key="prediction_glazing_area")
                    st.caption("Glazing area proportion")
                    glazing_distribution = st.selectbox("Glazing Area Distribution", [0, 1, 2, 3, 4, 5], help="Dataset distribution code", key="prediction_glazing_distribution")
                    st.caption("Dataset distribution code")
                if preferences().get("parameter_guidance", True):
                    st.caption("Design tip: compare alternatives by changing one input at a time.")
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
            card_title("Load estimates", "Your most recent successful prediction.")
            result = st.session_state.get("last_prediction")
            if result:
                r1, r2 = st.columns(2)
                r1.metric("Heating load", f"{result['heating']:.2f}")
                r2.metric("Cooling load", f"{result['cooling']:.2f}")
                st.caption(f"Generated: {result.get('created_at', 'recently')}")
                st.markdown('<div class="warning">Model estimates are for relative comparison. They do not represent every site, weather, occupancy, construction, or HVAC condition.</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="text-align:center;padding:1.5rem .5rem"><div class="section-label">Awaiting prediction</div><h3 style="margin-top:1rem">Your estimates will appear here</h3><p class="subtle">Review the building parameters, then run a prediction to estimate heating and cooling loads for this configuration.</p></div>', unsafe_allow_html=True)
            st.markdown("---")
            st.markdown("**Explore one change at a time**")
            st.caption("Compare forms, orientations, or glazing settings while holding other inputs constant.")
            if st.button("Understand the model", use_container_width=True):
                go_to("About")
    with st.container(border=True):
        card_title("My prediction history", "Only records associated with your signed-in account are shown.")
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
            st.warning("Prediction history could not be loaded. Confirm the predictions table and RLS policies.")
            st.caption(str(exc))
    footer()


def render_about():
    page_header("WORKSPACE / ABOUT", "About ArchiLoad AI", "Understand the data, models, and limitations behind every estimate.")
    tabs = st.tabs(["Dataset & models", "Model evaluation", "Design relevance", "Responsible use"])
    with tabs[0]:
        a, b = st.columns(2, gap="large")
        with a:
            with st.container(border=True):
                card_title("Dataset")
                st.markdown("**UCI Energy Efficiency dataset**")
                st.markdown("- 768 simulated building configurations\n- 8 building-form and glazing inputs\n- 2 targets: heating load and cooling load")
                st.caption("Simulated records do not represent every real building, climate, or operating condition.")
        with b:
            with st.container(border=True):
                card_title("Regression models")
                st.markdown("The application loads `heating_model.pkl` and `cooling_model.pkl`. `feature_columns.pkl` supplies the feature names and order expected by the models.")
                st.markdown("Training and evaluation are completed separately from the deployed Streamlit interface.")
    with tabs[1]:
        with st.container(border=True):
            card_title("Reported evaluation results", "Verify these figures against the exact deployed model files before using them in a formal report.")
            st.dataframe(pd.DataFrame([{"Target": target, **metrics} for target, metrics in REPORTED_METRICS.items()]), use_container_width=True, hide_index=True)
        a, b, c = st.columns(3)
        with a:
            st.markdown("**MAE**")
            st.caption("Mean Absolute Error. Lower values indicate smaller average absolute errors.")
        with b:
            st.markdown("**RMSE**")
            st.caption("Root Mean Squared Error. Larger errors have a greater influence.")
        with c:
            st.markdown("**R²**")
            st.caption("Coefficient of determination. Values closer to 1 generally indicate a stronger fit.")
    with tabs[2]:
        with st.container(border=True):
            card_title("Relevance to architectural design")
            st.markdown("ArchiLoad AI supports early-stage comparison of building compactness, surface areas, height, orientation, and glazing choices.")
            st.markdown("- Compare alternative parameter combinations.\n- Treat changes in predicted loads as prompts for further investigation.\n- Use detailed energy simulation and professional review for consequential design decisions.")
    with tabs[3]:
        with st.container(border=True):
            card_title("Limitations and responsible use")
            st.markdown("- The model is based on a finite set of simulated building configurations.\n- Climate, occupancy, schedules, HVAC systems, construction details, and other factors are not fully represented.\n- Results are not measured building performance, utility-cost estimates, code-compliance checks, certification, or final engineering calculations.\n- Validate important design decisions with appropriate tools and qualified professionals.")
            st.markdown('<div class="warning">Use the model to explore—not to certify.</div>', unsafe_allow_html=True)
    footer()


def render_settings():
    page_header("WORKSPACE / SETTINGS", "Settings", "Personalize your workspace. These preferences do not change the prediction models.")
    current = preferences()
    left, right = st.columns([1.15, .85], gap="large")
    with left:
        with st.container(border=True):
            card_title("Application preferences")
            with st.form("settings_form"):
                language = st.selectbox("Language", ["English"], index=0, help="English is currently the available interface language.")
                choices = ["Overview", "Prediction", "About"]
                default_page = st.selectbox("Default page after sign-in", choices, index=choices.index(current.get("default_page", "Overview")) if current.get("default_page") in choices else 0)
                parameter_guidance = st.toggle("Show parameter guidance", value=bool(current.get("parameter_guidance", True)))
                st.markdown("---")
                card_title("Communication preferences")
                product_updates = st.toggle("Product updates", value=bool(current.get("product_updates", False)))
                getting_started_tips = st.toggle("Getting-started tips", value=bool(current.get("getting_started_tips", True)))
                st.caption("Essential account and security emails may still be sent when needed.")
                submitted = st.form_submit_button("Save preferences", type="primary", use_container_width=True)
            if submitted:
                prefs = {"language": language, "default_page": default_page, "parameter_guidance": parameter_guidance,
                         "product_updates": product_updates, "getting_started_tips": getting_started_tips}
                try:
                    update_user_metadata({"archiload_preferences": prefs})
                    st.success("Preferences saved to your account.")
                    st.rerun()
                except Exception as exc:
                    st.error("Preferences could not be saved. Check Supabase Auth configuration.")
                    st.caption(str(exc))
    with right:
        with st.container(border=True):
            card_title("Account settings")
            st.markdown("Manage your profile, password, and saved predictions from your account page.")
            if st.button("Go to account  →", use_container_width=True):
                go_to("Account")
        with st.container(border=True):
            card_title("Preferences, not model controls")
            st.markdown("These settings personalize the workspace only. They do not tune the trained models or change previously saved predictions.")
    footer()


def render_account():
    page_header("WORKSPACE / ACCOUNT", "My account", "Manage your profile, security, and saved predictions.")
    meta = user_metadata()
    email = current_email()
    first_default, last_default = str(meta.get("first_name", "")), str(meta.get("last_name", ""))
    initials = ((first_default[:1] + last_default[:1]) or email[:1] or "A").upper()
    try:
        # This is the number of records fetched up to the limit, not an unlimited database count.
        count = len(fetch_predictions(10000))
        count_text = f"{count}+" if count == 10000 else str(count)
    except Exception:
        count_text = "—"
    left, right = st.columns([1.08, .92], gap="large")
    with left:
        with st.container(border=True):
            st.markdown(f'<div style="display:flex;align-items:center;gap:1rem"><div style="background:#E3F4F3;color:{TEAL};border-radius:50%;width:58px;height:58px;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:1.2rem">{initials}</div><div><h2 style="margin:0">{display_name()}</h2><span class="pill">Signed-in account</span></div></div>', unsafe_allow_html=True)
            st.write("")
            st.markdown(f"**Email address**  \n{email}")
            st.caption("Your saved predictions are associated with this account.")
        with st.container(border=True):
            card_title("Personal profile")
            with st.form("profile_form"):
                first_name = st.text_input("First name", value=first_default)
                last_name = st.text_input("Last name", value=last_default)
                st.text_input("Email address", value=email, disabled=True)
                save_profile = st.form_submit_button("Save profile", type="primary", use_container_width=True)
            if save_profile:
                try:
                    update_user_metadata({"first_name": first_name.strip(), "last_name": last_name.strip()})
                    st.success("Profile updated.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not update profile. Details: {exc}")
    with right:
        with st.container(border=True):
            card_title("Security & sign-in")
            st.markdown("**Password management**")
            st.caption("Your current password is never displayed.")
            with st.form("password_form"):
                new_password = st.text_input("New password", type="password")
                confirm_password = st.text_input("Confirm new password", type="password")
                change_password = st.form_submit_button("Change password", type="primary", use_container_width=True)
            if change_password:
                if len(new_password) < 6:
                    st.warning("Use a password with at least 6 characters.")
                elif new_password != confirm_password:
                    st.warning("The passwords do not match.")
                else:
                    try:
                        st.session_state.supabase.auth.update_user({"password": new_password})
                        st.success("Password updated.")
                    except Exception as exc:
                        st.error(f"Could not update password. Details: {exc}")
            st.markdown("---")
            if st.button("Sign out", use_container_width=True):
                sign_out()
        with st.container(border=True):
            card_title("Saved predictions")
            st.markdown(f'<div class="metric-value">{count_text}</div>', unsafe_allow_html=True)
            st.caption("Predictions saved to your account")
            if st.button("My prediction history", use_container_width=True):
                go_to("Prediction")
    with st.container(border=True):
        card_title("Make the workspace yours")
        st.write("Adjust workspace preferences without changing the underlying prediction models.")
        if st.button("Open settings", use_container_width=True):
            go_to("Settings")
    footer()


def main():
    st.session_state.setdefault("page", "Overview")
    st.session_state.setdefault("authenticated", False)
    url = secret_value("SUPABASE_URL")
    key = secret_value("SUPABASE_KEY")
    if not url or not key:
        st.error("Supabase connection settings are missing.")
        st.markdown("Add `SUPABASE_URL` and `SUPABASE_KEY` under **Streamlit Cloud → App settings → Secrets**.")
        st.code('SUPABASE_URL = "https://YOUR-PROJECT.supabase.co"\nSUPABASE_KEY = "YOUR_SUPABASE_PUBLISHABLE_OR_ANON_KEY"', language="toml")
        st.info("Use your project's publishable/anon key. Never place a service-role key in a public app.")
        st.stop()
    try:
        client = get_supabase_client(str(url), str(key))
        st.session_state.supabase = client
    except Exception as exc:
        st.error("Could not initialize Supabase.")
        st.exception(exc)
        st.stop()
    if not st.session_state.get("authenticated") or st.session_state.get("user") is None:
        render_auth(client)
        return
    render_sidebar()
    page = st.session_state.get("page", "Overview")
    renderers = {"Overview": render_overview, "Prediction": render_prediction, "About": render_about, "Settings": render_settings, "Account": render_account}
    renderers.get(page, render_overview)()


if __name__ == "__main__":
    main()
