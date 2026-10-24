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


def supabase_config():
    try:
        url = st.secrets.get("SUPABASE_URL", "")
        key = st.secrets.get("SUPABASE_KEY", "")
    except Exception:
        url = os.getenv("SUPABASE_URL", "")
        key = os.getenv("SUPABASE_KEY", "")
    return str(url).strip(), str(key).strip()


def make_client():
    url, key = supabase_config()
    if not url or not key:
        return None
    try:
        return create_client(url, key)
    except Exception as exc:
        st.error(f"Unable to initialize Supabase: {exc}")
        return None


def restore_session(client):
    if not client or not st.session_state.access_token or not st.session_state.refresh_token:
        return
    try:
        response = client.auth.set_session(
            st.session_state.access_token, st.session_state.refresh_token
        )
        session = getattr(response, "session", None)
        user = getattr(response, "user", None)
        if session:
            st.session_state.access_token = session.access_token
            st.session_state.refresh_token = session.refresh_token
        if user:
            st.session_state.user = user
    except Exception:
        st.session_state.access_token = None
        st.session_state.refresh_token = None
        st.session_state.user = None


def store_auth_response(response):
    session = getattr(response, "session", None)
    user = getattr(response, "user", None)
    if session:
        st.session_state.access_token = session.access_token
        st.session_state.refresh_token = session.refresh_token
    if user:
        st.session_state.user = user


def user_obj():
    return st.session_state.get("user")


def user_id():
    user = user_obj()
    if user is None:
        return None
    return getattr(user, "id", None) or (user.get("id") if isinstance(user, dict) else None)


def user_email():
    user = user_obj()
    if user is None:
        return ""
    return getattr(user, "email", None) or (user.get("email", "") if isinstance(user, dict) else "")


def metadata():
    user = user_obj()
    if user is None:
        return {}
    data = getattr(user, "user_metadata", None)
    if data is None and isinstance(user, dict):
        data = user.get("user_metadata")
    return data or {}


def display_name():
    data = metadata()
    full_name = f"{data.get('first_name', '')} {data.get('last_name', '')}".strip()
    return full_name or (user_email().split("@")[0] if user_email() else "Your account")


def initials(name):
    parts = str(name).split()
    return "".join(p[0].upper() for p in parts[:2]) or "AL"


def sign_out(client):
    try:
        if client:
            client.auth.sign_out()
    except Exception:
        pass
    for key in ["access_token", "refresh_token", "user", "last_prediction", "prediction_inputs"]:
        st.session_state[key] = None
    st.session_state.page = "Overview"
    st.session_state.auth_view = "Sign in"
    st.rerun()


@st.cache_resource(show_spinner="Loading trained models...")
def load_models():
    heating_model = joblib.load("heating_model.pkl")
    cooling_model = joblib.load("cooling_model.pkl")
    columns_obj = joblib.load("feature_columns.pkl")
    if isinstance(columns_obj, pd.DataFrame):
        columns = list(columns_obj.columns)
    elif isinstance(columns_obj, dict):
        columns = list(columns_obj.keys())
    elif hasattr(columns_obj, "tolist"):
        columns = columns_obj.tolist()
    else:
         columns = list(columns_obj)
    if len(columns) != 8:
        raise ValueError("feature_columns.pkl must contain the eight model input column names.")
    return heating_model, cooling_model, columns


def predict(values):
    heating_model, cooling_model, columns = load_models()
    ordered_values = [
        float(values["relative_compactness"]),
        float(values["surface_area"]),
        float(values["wall_area"]),
        float(values["roof_area"]),
        float(values["overall_height"]),
        int(values["orientation"]),
        float(values["glazing_area"]),
        int(values["glazing_distribution"]),
    ]
    # Use the exact column names and order saved during model training.
    frame = pd.DataFrame([ordered_values], columns=columns)
    heating = float(np.asarray(heating_model.predict(frame)).reshape(-1)[0])
    cooling = float(np.asarray(cooling_model.predict(frame)).reshape(-1)[0])
    return heating, cooling


def get_history(client, limit=100):
    uid = user_id()
    if not uid:
        return []
    try:
        response = (
            client.table("predictions")
            .select("id,user_id,relative_compactness,surface_area,wall_area,roof_area,overall_height,orientation,glazing_area,glazing_distribution,heating_load,cooling_load,created_at")
            .eq("user_id", uid)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return response.data or []
    except Exception as exc:
        st.warning(f"Could not load prediction history. Check your Supabase table and policies. Details: {exc}")
        return []


def save_prediction(client, values, heating, cooling):
    uid = user_id()
    if not uid:
        raise RuntimeError("Your session has expired. Please sign in again.")
    row = {
        "user_id": uid,
        "relative_compactness": float(values["relative_compactness"]),
        "surface_area": float(values["surface_area"]),
        "wall_area": float(values["wall_area"]),
        "roof_area": float(values["roof_area"]),
        "overall_height": float(values["overall_height"]),
        "orientation": int(values["orientation"]),
        "glazing_area": float(values["glazing_area"]),
        "glazing_distribution": int(values["glazing_distribution"]),
        "heating_load": float(heating),
        "cooling_load": float(cooling),
    }
    client.table("predictions").insert(row).execute()


def get_preferences():
    prefs = DEFAULT_PREFERENCES.copy()
    saved = metadata().get("archiload_preferences", {})
    if isinstance(saved, dict):
        prefs.update(saved)
    return prefs


def save_preferences(client, prefs):
    # Stores preferences in Supabase Auth user metadata; no new table is required.
    client.auth.update_user({"data": {**metadata(), "archiload_preferences": prefs}})
    try:
        response = client.auth.get_user()
        fresh_user = getattr(response, "user", None)
        if fresh_user:
            st.session_state.user = fresh_user
    except Exception:
        pass


def header(title, subtitle, page):
    left, right = st.columns([3, 1])
    with left:
        st.markdown(f'<div class="muted">Workspace &nbsp; › &nbsp; <b>{page}</b></div>', unsafe_allow_html=True)
    with right:
        st.markdown(
            f'<div style="text-align:right;font-size:.84rem;color:{TEXT};">'
            f'<span style="display:inline-flex;align-items:center;justify-content:center;background:{TEAL_LIGHT};color:{TEAL};border-radius:50%;width:32px;height:32px;font-weight:700;margin-right:8px;">{initials(display_name())}</span>{display_name()}</div>',
            unsafe_allow_html=True,
        )
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def footer():
    st.markdown('<div class="footer">ArchiLoad AI · Early-stage building energy exploration <span style="float:right">UCI Energy Efficiency dataset</span></div>', unsafe_allow_html=True)


def asset_data_uri(filename):
    """Read an optional image asset stored beside app.py."""
    path = Path(__file__).resolve().parent / filename
    if not path.is_file():
        return None
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/svg+xml" if suffix == ".svg" else "application/octet-stream"
    try:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"
    except OSError:
        return None


def logo_html(height=38):
    logo_uri = asset_data_uri("supplied-archiload-logo.png")
    if logo_uri:
        return f'<img src="{logo_uri}" alt="ArchiLoad AI logo" style="height:{height}px;max-width:100%;object-fit:contain;object-position:left center;" />'
    return '<span style="font-size:1.3rem;margin-right:.45rem;">▥</span><span>ArchiLoad AI</span>'


def sidebar(client):
    with st.sidebar:
        st.markdown(f'<div class="brand">{logo_html(38)}</div><div class="sidebar-kicker">PERSONAL WORKSPACE</div>', unsafe_allow_html=True)
        page = st.radio("Navigation", VALID_PAGES, index=VALID_PAGES.index(st.session_state.page) if st.session_state.page in VALID_PAGES else 0, label_visibility="collapsed", key="nav_radio")
        if page != st.session_state.page:
            st.session_state.page = page
            st.rerun()
        st.markdown('<div class="sidebar-promo"><div style="font-size:1.2rem;color:#66D0D2;">✦</div><h4>Better early-stage decisions</h4><p>Explore building form, orientation, and glazing before detailed analysis.</p></div>', unsafe_allow_html=True)
        st.write("")
        if st.button("↪  Sign out", use_container_width=True):
            sign_out(client)


def auth_screen(client):
    left, right = st.columns([1.02, 1], gap="large", vertical_alignment="center")
    with left:
        building_uri = asset_data_uri("building-envelope-study.png")
        logo = logo_html(42)
        if building_uri:
            left_background = f'background-image:linear-gradient(135deg,rgba(20,43,61,.94),rgba(20,43,61,.73)),url("{building_uri}");background-size:cover;background-position:center;'
        else:
            left_background = f'background:linear-gradient(145deg,{NAVY} 0%,#1B3B50 65%,#285969 100%);'
        st.markdown(
            f'''<div style="{left_background}padding:2.25rem 2rem;border-radius:14px;min-height:570px;color:white;display:flex;flex-direction:column;justify-content:space-between;">
              <div>{logo}</div>
              <div style="padding-top:2rem;padding-bottom:2rem;">
                <div style="font-size:.72rem;letter-spacing:.13em;font-weight:700;color:#73D5D4;">EARLY-STAGE DESIGN SUPPORT</div>
                <h1 style="color:white;font-size:2.35rem;line-height:1.13;letter-spacing:-.035em;margin:1rem 0;">AI-Based Heating and Cooling Load Predictor</h1>
                <p style="color:#D5E1E7;font-size:1rem;line-height:1.75;max-width:33rem;">Explore how early building design parameters relate to estimated heating and cooling loads.</p>
              </div>
              <div style="background:rgba(9,27,40,.42);border:1px solid rgba(255,255,255,.16);padding:1rem 1.1rem;border-radius:9px;">
                <div style="color:white;font-weight:650;margin-bottom:.35rem;">A preliminary design aid</div>
                <p style="color:#D5E1E7;font-size:.82rem;line-height:1.6;margin:0;">Use estimates to support early exploration—not as a substitute for detailed energy simulation or professional engineering review.</p>
              </div>
            </div>''',
            unsafe_allow_html=True,
        )
    with right:
        st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
        st.markdown('<div class="eyebrow">YOUR PERSONAL WORKSPACE</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-title" style="font-size:1.85rem;margin-top:.55rem;">Welcome back</div><div class="page-subtitle">Sign in to continue to your ArchiLoad AI workspace.</div>', unsafe_allow_html=True)
        auth_choices = ["Sign in", "Create account", "Reset password"]
        st.session_state.auth_view = st.radio("Account action", auth_choices, index=auth_choices.index(st.session_state.auth_view) if st.session_state.auth_view in auth_choices else 0, horizontal=True, label_visibility="collapsed", key="auth_choice")
        if st.session_state.auth_view == "Sign in":
            with st.form("signin_form"):
                email = st.text_input("Email address", placeholder="you@example.com")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                submitted = st.form_submit_button("Sign in", type="primary", use_container_width=True)
            if submitted:
                if not email.strip() or not password:
                    st.error("Enter your email address and password.")
                else:
                    try:
                        response = client.auth.sign_in_with_password({"email": email.strip(), "password": password})
                        store_auth_response(response)
                        if user_obj():
                            st.session_state.page = "Overview"
                            st.rerun()
                        else:
                            st.error("No active session was returned. Check whether email confirmation is required in Supabase.")
                    except Exception as exc:
                        st.error(f"Sign-in failed. Check your credentials. Details: {exc}")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Create an account", use_container_width=True):
                    st.session_state.auth_view = "Create account"
                    st.rerun()
            with c2:
                if st.button("Forgot password?", use_container_width=True):
                    st.session_state.auth_view = "Reset password"
                    st.rerun()
        elif st.session_state.auth_view == "Create account":
            with st.form("signup_form"):
                first = st.text_input("First name")
                last = st.text_input("Last name")
                email = st.text_input("Email address", placeholder="you@example.com")
                password = st.text_input("Password", type="password")
                confirm = st.text_input("Confirm password", type="password")
                submitted = st.form_submit_button("Create account", type="primary", use_container_width=True)
            if submitted:
                if not first.strip() or not last.strip() or not email.strip() or not password:
                    st.error("Complete all fields.")
                elif password != confirm:
                    st.error("The passwords do not match.")
                elif len(password) < 6:
                    st.error("Use a password with at least 6 characters.")
                else:
                    try:
                        response = client.auth.sign_up({"email": email.strip(), "password": password, "options": {"data": {"first_name": first.strip(), "last_name": last.strip()}}})
                        store_auth_response(response)
                        if user_obj() and st.session_state.access_token:
                            st.session_state.page = "Overview"
                            st.rerun()
                        else:
                            st.success("Account created. Check your email for a confirmation link, then sign in.")
                    except Exception as exc:
                        st.error(f"Could not create the account. Details: {exc}")
            if st.button("Back to sign in"):
                st.session_state.auth_view = "Sign in"
                st.rerun()
        else:
            with st.form("reset_form"):
                email = st.text_input("Account email", placeholder="you@example.com")
                submitted = st.form_submit_button("Send reset email", type="primary", use_container_width=True)
            if submitted:
                if not email.strip():
                    st.error("Enter the email address for your account.")
                else:
                    try:
                        client.auth.reset_password_for_email(email.strip())
                        st.success("If the account exists, a password-reset email will be sent.")
                    except Exception as exc:
                        st.error(f"Could not request a reset email. Details: {exc}")
            if st.button("Back to sign in"):
                st.session_state.auth_view = "Sign in"
                st.rerun()
        st.markdown('<p class="muted" style="text-align:center;margin-top:1.6rem;font-size:.76rem;">ArchiLoad AI · UCI Energy Efficiency dataset</p>', unsafe_allow_html=True)


def overview(client):
    header("Welcome to your workspace", "Explore the relationship between early building design choices and estimated heating and cooling loads.", "Overview")
    history = get_history(client)
    cols = st.columns(3)
    for col, value, label, note in zip(cols, ["768", "2", "8"], ["Simulated building records", "Prediction outputs", "Building inputs"], ["UCI Energy Efficiency dataset", "Heating and cooling load", "Form, orientation, and glazing"]):
        with col:
            st.markdown(f'<div class="card"><div class="metric-value">{value}</div><b>{label}</b><div class="metric-label">{note}</div></div>', unsafe_allow_html=True)
    left, right = st.columns([1.25, 1], gap="large")
    with left:
        st.markdown('<div class="card"><div class="card-title">From building inputs to load estimates</div><div class="muted">ArchiLoad AI applies trained regression models to a proposed building configuration.</div></div>', unsafe_allow_html=True)
        stages = st.columns(3)
        for col, num, title, desc in zip(stages, ["01", "02", "03"], ["Building inputs", "Regression models", "Load estimates"], ["Set dimensions, form, orientation, and glazing.", "The trained models process selected parameters.", "Review estimated heating and cooling loads."]):
            with col:
                st.markdown(f'<div class="card" style="min-height:155px;"><div class="eyebrow">{num}</div><b>{title}</b><div class="muted" style="margin-top:.5rem;">{desc}</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="card"><div class="card-title">Getting started</div><div class="muted">1. Open Prediction and enter a building configuration.<br>2. Run the models to estimate heating and cooling loads.<br>3. Save and compare configurations in prediction history.</div></div>', unsafe_allow_html=True)
        if st.button("Start a prediction →", type="primary"):
            st.session_state.page = "Prediction"
            st.rerun()
    with right:
        st.markdown('<div class="card"><div class="card-title">About the dataset</div><div class="muted">The UCI Energy Efficiency dataset contains 768 simulated building configurations with eight input variables and two load targets.</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="notice"><b>Preliminary estimates</b><p class="muted">Use predictions to explore early-stage design alternatives. They are not measured energy consumption or a substitute for detailed simulation and professional review.</p></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="card"><div class="card-title">Your prediction history</div><div class="metric-value">{len(history)}</div><div class="muted">Saved predictions in your account</div></div>', unsafe_allow_html=True)
        if history:
            st.caption("Most recent saved prediction")
            st.write(f"Heating: **{float(history[0]['heating_load']):.2f}** · Cooling: **{float(history[0]['cooling_load']):.2f}**")
        else:
            st.caption("No saved predictions yet. Try your first configuration.")
        if st.button("View prediction history"):
            st.session_state.page = "Prediction"
            st.rerun()
    footer()


def prediction_page(client):
    header("Prediction workspace", "Enter a building configuration to estimate heating and cooling loads.", "Prediction")
    form_col, result_col = st.columns([1.1, .9], gap="large")
    with form_col:
        st.markdown('<div class="card"><div class="card-title">Building parameters</div><div class="muted">Adjust the values to explore one design configuration at a time.</div></div>', unsafe_allow_html=True)
        with st.form("prediction_form"):
            st.markdown("**Building form and dimensions**")
            a, b = st.columns(2)
            with a:
                compact = st.number_input("Relative compactness", min_value=0.62, max_value=0.98, value=0.80, step=0.01, format="%.2f")
                surface = st.number_input("Surface area", min_value=514.5, max_value=808.5, value=661.5, step=1.0, format="%.2f")
                wall = st.number_input("Wall area", min_value=245.0, max_value=416.5, value=318.5, step=1.0, format="%.2f")
                roof = st.number_input("Roof area", min_value=110.25, max_value=220.5, value=147.0, step=1.0, format="%.2f")
            with b:
                height = st.selectbox("Overall height", [3.5, 7.0], index=1, help="Values represented in the dataset.")
                orientation = st.selectbox("Orientation", [2, 3, 4, 5], index=0, help="Orientation categories in the dataset are coded 2–5.")
                glazing = st.selectbox("Glazing area", [0.0, 0.10, 0.25, 0.40], index=2, format_func=lambda x: f"{x:.2f}")
                distribution = st.selectbox("Glazing distribution", [0, 1, 2, 3, 4, 5], index=0)
            submitted = st.form_submit_button("Calculate load estimates", type="primary", use_container_width=True)
        if submitted:
            values = {"relative_compactness": compact, "surface_area": surface, "wall_area": wall, "roof_area": roof, "overall_height": height, "orientation": orientation, "glazing_area": glazing, "glazing_distribution": distribution}
            try:
                 with st.spinner("Calculating and saving estimates..."):
                    heating, cooling = predict(values)
                    save_prediction(client, values, heating, cooling)
                st.session_state.last_prediction = {"heating_load": heating, "cooling_load": cooling, "created_at": datetime.now().isoformat(timespec="seconds")}
                st.session_state.prediction_inputs = values
                st.success("Prediction calculated and saved to your account.")
                st.rerun()
            except FileNotFoundError as exc:
                st.error(f"A required model file is missing. Put heating_model.pkl, cooling_model.pkl, and feature_columns.pkl beside app.py in GitHub. Details: {exc}")
            except Exception as exc:
                st.error(f"Could not calculate or save the prediction. Check your model files and Supabase connection. Details: {exc}")
    with result_col:
        st.markdown('<div class="card"><div class="card-title">Load estimates</div><div class="muted">Predicted values from the trained regression models.</div>', unsafe_allow_html=True)
        result = st.session_state.last_prediction
        if result:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<div class="eyebrow">HEATING LOAD</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{result["heating_load"]:.2f}</div>', unsafe_allow_html=True)
                st.caption("Dataset target units")
            with c2:
                st.markdown('<div class="eyebrow">COOLING LOAD</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{result["cooling_load"]:.2f}</div>', unsafe_allow_html=True)
                st.caption("Dataset target units")
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown('<div class="notice"><b>Explore one change at a time</b><p class="muted">Change one parameter and compare the new estimates with a previous saved configuration. A predicted change is not a guarantee of real-world performance.</p></div>', unsafe_allow_html=True)
        else:
            st.info("Enter the parameters and select Calculate load estimates to see your results.")
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown('<div class="card"><div class="card-title">What happens next?</div><div class="muted">The eight inputs are sent to two trained regression models. The estimates are saved to your account for later comparison.</div></div>', unsafe_allow_html=True)
        if st.button("Reset displayed result"):
            st.session_state.last_prediction = None
            st.session_state.prediction_inputs = None
            st.rerun()
    st.markdown('<div class="card"><div class="card-title">My prediction history</div><div class="muted">Only records belonging to your signed-in account are shown here.</div></div>', unsafe_allow_html=True)
    history = get_history(client, limit=100)
    if history:
        frame = pd.DataFrame(history)
        display = {"created_at":"Created at", "relative_compactness":"Compactness", "surface_area":"Surface area", "wall_area":"Wall area", "roof_area":"Roof area", "overall_height":"Height", "orientation":"Orientation", "glazing_area":"Glazing area", "glazing_distribution":"Glazing distribution", "heating_load":"Heating load", "cooling_load":"Cooling load"}
        cols = [c for c in display if c in frame.columns]
        st.dataframe(frame[cols].rename(columns=display), use_container_width=True, hide_index=True)
    else:
        st.caption("No saved predictions yet.")
    st.caption("Responsible use: estimates support early-stage exploration only and should be validated with detailed building-energy analysis.")
    footer()


def about_page():
    header("About ArchiLoad AI", "Understand the data, the models, and the limits behind every estimate.", "About")
    tab1, tab2, tab3, tab4 = st.tabs(["Dataset & models", "Model evaluation", "Design relevance", "Responsible use"])
    with tab1:
        a, b = st.columns(2, gap="large")
        with a:
            st.markdown('<div class="card"><div class="card-title">The dataset</div><div class="muted">The UCI Energy Efficiency dataset contains 768 simulated building configurations. Eight input variables describe building form, dimensions, orientation, and glazing.</div><p style="color:#00858B;font-size:.8rem;font-weight:600;">768 records · 8 inputs · 2 outputs</p></div>', unsafe_allow_html=True)
        with b:
            st.markdown('<div class="card"><div class="card-title">Two trained regression models</div><div class="muted">One model estimates heating load and a second model estimates cooling load. The models were trained in Google Colab and exported for this application.</div><p class="muted">Training environment: Google Colab</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="card"><div class="card-title">Eight model inputs</div></div>', unsafe_allow_html=True)
        st.write("Relative Compactness, Surface Area, Wall Area, Roof Area, Overall Height, Orientation, Glazing Area, Glazing Area Distribution")
    with tab2:
        st.markdown('<div class="card"><div class="card-title">Model evaluation</div><div class="muted">These are the metrics supplied for the interface. Verify that they match the final trained models and evaluation split before presenting them as final results.</div></div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([{"Target":"Heating load", "MAE":0.35235, "RMSE":0.489673218, "R²":0.997699565}, {"Target":"Cooling load", "MAE":1.079106926, "RMSE":1.750736111, "R²":0.966920264}]), use_container_width=True, hide_index=True)
        cols = st.columns(3)
        for col, name, desc in zip(cols, ["MAE", "RMSE", "R²"], ["Mean absolute prediction error.", "Penalizes larger errors more strongly.", "How much variation in the target is explained by the model."]):
            with col:
                st.markdown(f'<div class="card"><div class="eyebrow">{name}</div><div class="muted">{desc}</div></div>', unsafe_allow_html=True)
    with tab3:
        st.markdown('<div class="card"><div class="card-title">Architecture relevance & intended design use</div><div class="muted">Architects and design students can use the app to explore how alternative building configurations relate to estimated heating and cooling loads during early-stage design. It supports preliminary comparisons of building form, orientation, and glazing choices, but does not replace detailed energy simulation or professional engineering assessment.</div></div>', unsafe_allow_html=True)
        st.write("**A practical workflow**")
        st.write("1. Define an early building configuration.\n2. Review the estimated heating and cooling loads.\n3. Change one design parameter at a time.\n4. Compare saved configurations and validate promising options with appropriate simulation tools.")
    with tab4:
        for title, desc in [
            ("Finite simulated dataset", "The models use a finite dataset of simulated designs. Estimates may be less reliable for configurations outside the training data."),
            ("Not every real-world factor is modeled", "The app does not fully account for local weather, occupancy, operating schedules, HVAC equipment, or all construction details."),
            ("Estimates are not measured performance", "Predictions are not actual energy consumption, utility costs, proof of compliance, certification, or final engineering calculations."),
            ("Professional validation is essential", "Validate important decisions using appropriate building-energy simulation tools and qualified professionals."),
        ]:
            st.markdown(f'<div class="card"><div class="card-title">{title}</div><div class="muted">{desc}</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="notice"><b>Use the model to explore—not to certify</b><p class="muted">Use estimates to frame early design questions, then validate decisions through detailed analysis and professional review.</p></div>', unsafe_allow_html=True)
    footer()


def settings_page(client):
    header("Settings", "Personalize your workspace. These preferences do not change prediction inputs or models.", "Settings")
    prefs = get_preferences()
    with st.form("settings_form"):
        st.markdown('<div class="card-title">Application preferences</div><div class="muted">These preferences are stored with your Supabase user metadata.</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            language = st.selectbox("Language", ["English"], index=0, help="Only English is currently available.")
        with c2:
            default_page = st.selectbox("Default page after sign in", VALID_PAGES, index=VALID_PAGES.index(prefs.get("default_page", "Overview")) if prefs.get("default_page", "Overview") in VALID_PAGES else 0)
        guidance = st.checkbox("Show parameter guidance", value=bool(prefs.get("show_guidance", True)))
        st.markdown("---")
        st.markdown('<div class="card-title">Communication preferences</div>', unsafe_allow_html=True)
        updates = st.checkbox("Product updates", value=bool(prefs.get("product_updates", False)), help="The app does not currently send marketing emails.")
        tips = st.checkbox("Getting-started tips", value=bool(prefs.get("getting_started_tips", True)))
        st.caption("Essential account and security emails are not affected by these preferences.")
        save = st.form_submit_button("Save preferences", type="primary")
    if save:
        new_prefs = {"language": language, "default_page": default_page, "show_guidance": guidance, "product_updates": updates, "getting_started_tips": tips}
        try:
            save_preferences(client, new_prefs)
            st.success("Preferences saved to your account.")
        except Exception as exc:
            st.error(f"Could not save preferences. Details: {exc}")
    a, b = st.columns(2)
    with a:
        st.markdown('<div class="card"><div class="card-title">Account settings</div><div class="muted">Manage your profile, email address, and password from your personal account.</div></div>', unsafe_allow_html=True)
        if st.button("Go to account →"):
            st.session_state.page = "Account"
            st.rerun()
    with b:
        st.markdown('<div class="notice"><b>Preferences, not model controls</b><p class="muted">These settings do not tune the models, change training data, or alter saved predictions.</p></div>', unsafe_allow_html=True)
    footer()


def account_page(client):
    header("My account", "Manage your personal profile, security, and saved predictions.", "Account")
    name, email = display_name(), user_email()
    data = metadata()
    history = get_history(client, limit=500)
    st.markdown(f'<div class="card"><div style="display:flex;align-items:center;gap:1rem;"><div style="width:60px;height:60px;flex-shrink:0;background:{TEAL_LIGHT};border-radius:50%;display:flex;align-items:center;justify-content:center;color:{TEAL};font-size:1.2rem;font-weight:700;">{initials(name)}</div><div><div style="font-size:1.2rem;font-weight:650;color:{TEXT};">{name}</div><div class="muted">{email}</div></div></div></div>', unsafe_allow_html=True)
    left, right = st.columns([1.4, 1], gap="large")
    with left:
        st.markdown('<div class="card-title">Personal profile</div><div class="muted">Your name is used throughout your workspace.</div>', unsafe_allow_html=True)
        with st.form("profile_form"):
            first = st.text_input("First name", value=str(data.get("first_name", "")))
            last = st.text_input("Last name", value=str(data.get("last_name", "")))
            st.text_input("Email address", value=email, disabled=True, help="Email changes require a separate confirmation process.")
            save_profile = st.form_submit_button("Save profile", type="primary")
        if save_profile:
            if not first.strip() or not last.strip():
                st.error("Enter both your first and last name.")
            else:
                try:
                    client.auth.update_user({"data": {**data, "first_name": first.strip(), "last_name": last.strip()}})
                    fresh = client.auth.get_user()
                    fresh_user = getattr(fresh, "user", None)
                    if fresh_user:
                        st.session_state.user = fresh_user
                    st.success("Profile updated.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not update your profile. Details: {exc}")
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        st.markdown('<div class="card-title">Security & sign-in</div><div class="muted">Request a secure password-reset email for your account.</div>', unsafe_allow_html=True)
        if st.button("Send password reset email"):
            try:
                client.auth.reset_password_for_email(email)
                st.success("If the account exists, a password-reset email will be sent.")
            except Exception as exc:
                st.error(f"Could not request a reset email. Details: {exc}")
        if st.button("Sign out of current session"):
            sign_out(client)
    with right:
        st.markdown(f'<div class="card"><div class="card-title">Saved predictions</div><div class="metric-value">{len(history)}</div><div class="muted">Predictions saved to your account</div></div>', unsafe_allow_html=True)
        if history:
            st.caption("Most recent saved estimate")
            st.write(f"Heating load: **{float(history[0]['heating_load']):.2f}**")
            st.write(f"Cooling load: **{float(history[0]['cooling_load']):.2f}**")
        else:
            st.caption("No saved predictions yet. Try your first configuration on the Prediction page.")
        if st.button("My Prediction History", use_container_width=True):
            st.session_state.page = "Prediction"
            st.rerun()
        st.markdown('<div class="card"><div class="card-title">Make the workspace yours</div><div class="muted">Manage application guidance and preferences in Settings.</div></div>', unsafe_allow_html=True)
        if st.button("Open settings →", use_container_width=True):
            st.session_state.page = "Settings"
            st.rerun()
    footer()


client = make_client()
if client is None:
    st.markdown('<div class="page-title">ArchiLoad AI setup</div>', unsafe_allow_html=True)
    st.error("Supabase connection details are missing.")
    st.markdown('''Add these values in **Streamlit Cloud → your app → Settings → Secrets**:
```toml
SUPABASE_URL = "https://YOUR-PROJECT.supabase.co"
SUPABASE_KEY = "YOUR-SUPABASE-ANON-OR-PUBLISHABLE-KEY"
```
Do not put a secret/service-role key in `app.py` or commit it to GitHub.''')
    st.stop()

restore_session(client)
if not user_obj():
    auth_screen(client)
    st.stop()

if st.session_state.get("_default_page_applied") != user_id():
    preferred = get_preferences().get("default_page", "Overview")
    st.session_state.page = preferred if preferred in VALID_PAGES else "Overview"
    st.session_state._default_page_applied = user_id()

sidebar(client)
page = st.session_state.page
if page == "Overview":
    overview(client)
elif page == "Prediction":
    prediction_page(client)
elif page == "About":
    about_page()
elif page == "Settings":
    settings_page(client)
elif page == "Account":
    account_page(client)
else:
    st.session_state.page = "Overview"
    st.rerun()
