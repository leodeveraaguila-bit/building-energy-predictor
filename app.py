import os
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from supabase import create_client

# ---------------------------------------------------------------------
# ArchiLoad AI — Streamlit adaptation of the supplied interface designs
# Required files in the repository root:
#   app.py, requirements.txt, heating_model.pkl, cooling_model.pkl,
#   feature_columns.pkl
# Optional image files:
#   archiloadai.png, building-envelope-study.png
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="ArchiLoad AI",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAVY = "#142B3D"
TEAL = "#00858B"
INK = "#213547"
MUTED = "#647789"
PALE = "#F3F7F9"
BORDER = "#DCE5EA"
WHITE = "#FFFFFF"

st.markdown(
    f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
      html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
      .stApp {{ background: #F6F8FA; color: {INK}; }}
      [data-testid="stHeader"] {{ background: rgba(246,248,250,0.92); }}
      [data-testid="stSidebar"] {{ background: {NAVY}; }}
      [data-testid="stSidebar"] * {{ color: #F7FAFC; }}
      [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{ color: #DCE8EE; }}
      [data-testid="stSidebar"] button {{ border-radius: 10px; }}
      h1, h2, h3 {{ color: {NAVY}; letter-spacing: -0.025em; }}
      .eyebrow {{ color: {TEAL}; font-size: 0.76rem; font-weight: 800; text-transform: uppercase; letter-spacing: .12em; }}
      .hero-title {{ color: {NAVY}; font-size: clamp(2rem, 3vw, 3rem); line-height: 1.12; font-weight: 800; margin: .2rem 0 .6rem 0; }}
      .subtle {{ color: {MUTED}; line-height: 1.65; }}
      .card {{ background: {WHITE}; border: 1px solid {BORDER}; border-radius: 18px; padding: 1.15rem 1.25rem; box-shadow: 0 5px 20px rgba(20,43,61,.035); }}
      .metric-card {{ background: {WHITE}; border: 1px solid {BORDER}; border-radius: 16px; padding: 1.1rem 1.2rem; min-height: 120px; }}
      .metric-label {{ color: {MUTED}; font-size: .88rem; font-weight: 600; }}
      .metric-value {{ color: {NAVY}; font-size: 1.85rem; font-weight: 800; margin-top: .3rem; }}
      .pill {{ display: inline-block; background: #E3F4F3; color: {TEAL}; padding: .3rem .65rem; border-radius: 99px; font-size: .78rem; font-weight: 700; }}
      .divider {{ border-top: 1px solid {BORDER}; margin: 1.2rem 0; }}
      .sidebar-brand {{ font-size: 1.3rem; font-weight: 800; color: white; margin: .3rem 0 .1rem 0; }}
      .sidebar-caption {{ font-size: .78rem; color: #C4D5DF; line-height: 1.45; }}
      .note {{ border-left: 4px solid {TEAL}; background: #EAF5F5; border-radius: 8px; padding: .85rem 1rem; color: #31545D; }}
      .warning {{ border-left: 4px solid #D59B35; background: #FFF7E7; border-radius: 8px; padding: .85rem 1rem; color: #6B5123; }}
      .small-muted {{ color: {MUTED}; font-size: .82rem; }}
      .stButton > button {{ border-radius: 10px; font-weight: 650; min-height: 2.65rem; }}
      .stButton > button[kind="primary"] {{ background: {TEAL}; border-color: {TEAL}; }}
      div[data-testid="stForm"] {{ background: {WHITE}; border: 1px solid {BORDER}; border-radius: 16px; padding: 1.2rem; }}
      [data-testid="stMetric"] {{ background: {WHITE}; border: 1px solid {BORDER}; padding: 1rem; border-radius: 14px; }}
      footer {{ visibility: hidden; }}
    </style>
    """,
    unsafe_allow_html=True,
)

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
    "relative_compactness": 0.80,
    "surface_area": 661.50,
    "wall_area": 318.50,
    "roof_area": 147.00,
    "overall_height": 7.0,
    "orientation": 2,
    "glazing_area": 0.25,
    "glazing_distribution": 0,
}

# The values below were included in the interface draft. Confirm them against
# the final evaluation output before using them in a formal report.
REPORTED_METRICS = {
    "Heating load": {"MAE": 0.35235, "RMSE": 0.489673218, "R²": 0.997699565},
    "Cooling load": {"MAE": 1.079106926, "RMSE": 1.750736111, "R²": 0.966920264},
}

def secret_value(name, default=None):
    """Read either top-level Streamlit secrets or a [supabase] section."""
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

def get_user_value(user, attr, default=None):
    if user is None:
        return default
    if isinstance(user, dict):
        return user.get(attr, default)
    return getattr(user, attr, default)

def user_metadata(user):
    meta = get_user_value(user, "user_metadata", {}) or {}
    return dict(meta) if isinstance(meta, dict) else {}

def current_email():
    return get_user_value(st.session_state.get("user"), "email", "") or ""

def display_name():
    user = st.session_state.get("user")
    meta = user_metadata(user)
    first = str(meta.get("first_name", "")).strip()
    last = str(meta.get("last_name", "")).strip()
    combined = f"{first} {last}".strip()
    if combined:
        return combined
    email = current_email()
    if email:
        return email.split("@")[0].replace(".", " ").replace("_", " ").title()
    return "Your workspace"

def get_user_id():
    return get_user_value(st.session_state.get("user"), "id", None)

def set_user_from_auth_response(response):
    session = get_user_value(response, "session")
    user = get_user_value(response, "user")
    if user is None and session is not None:
        user = get_user_value(session, "user")
    if user is not None:
        st.session_state.user = user
        st.session_state.authenticated = True
    return user

def sign_out():
    try:
        client = st.session_state.get("supabase")
        if client is not None:
            client.auth.sign_out()
    except Exception:
        pass
    for key in ["user", "authenticated", "last_prediction", "auth_message"]:
        st.session_state.pop(key, None)
    st.session_state.page = "Overview"
    st.rerun()

@st.cache_resource(show_spinner=False)
def load_models():
    root = Path(__file__).resolve().parent
    heating_path = root / "heating_model.pkl"
    cooling_path = root / "cooling_model.pkl"
    columns_path = root / "feature_columns.pkl"
    missing = [p.name for p in [heating_path, cooling_path, columns_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing model file(s): " + ", ".join(missing)
            + ". Upload these files to the root of your GitHub repository."
        )
    heating_model = joblib.load(heating_path)
    cooling_model = joblib.load(cooling_path)
    feature_columns = joblib.load(columns_path)
    if isinstance(feature_columns, pd.Index):
        feature_columns = list(feature_columns)
    elif isinstance(feature_columns, np.ndarray):
        feature_columns = feature_columns.tolist()
    return heating_model, cooling_model, list(feature_columns)

def normalize_name(value):
    return "".join(ch.lower() for ch in str(value) if ch.isalnum())

FEATURE_ALIASES = {
    "relative_compactness": ["relativecompactness", "relative_compactness", "x1"],
    "surface_area": ["surfacearea", "surface_area", "x2"],
    "wall_area": ["wallarea", "wall_area", "x3"],
    "roof_area": ["roofarea", "roof_area", "x4"],
    "overall_height": ["overallheight", "overall_height", "x5"],
    "orientation": ["orientation", "x6"],
    "glazing_area": ["glazingarea", "glazing_area", "x7"],
    "glazing_distribution": [
        "glazingareadistribution", "glazingdistribution", "glazing_area_distribution",
        "glazing_distribution", "x8"
    ],
}

def make_model_frame(values, feature_columns):
    """Build a one-row frame in the exact feature order saved with the model."""
    normalized_to_key = {}
    for key, aliases in FEATURE_ALIASES.items():
        for alias in aliases:
            normalized_to_key[normalize_name(alias)] = key

    row = {}
    for col in feature_columns:
        key = normalized_to_key.get(normalize_name(col))
        if key is not None:
            row[col] = values[key]
        else:
            # Preserve model input order but fail loudly rather than silently
            # feeding incorrect values to an unknown feature name.
            raise ValueError(
                f"Unrecognized model feature column '{col}' in feature_columns.pkl. "
                "Check that it matches the feature names used when training the model."
            )
    return pd.DataFrame([row], columns=feature_columns)

def predict_loads(values):
    heating_model, cooling_model, feature_columns = load_models()
    X = make_model_frame(values, feature_columns)
    heating = float(np.asarray(heating_model.predict(X)).reshape(-1)[0])
    cooling = float(np.asarray(cooling_model.predict(X)).reshape(-1)[0])
    return heating, cooling

def save_prediction(values, heating, cooling):
    client = st.session_state.supabase
    user_id = get_user_id()
    if not user_id:
        raise RuntimeError("Your session has expired. Please sign in again.")
    record = {
        "user_id": str(user_id),
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
    client.table("predictions").insert(record).execute()

def fetch_predictions(limit=100):
    client = st.session_state.supabase
    user_id = get_user_id()
    if not user_id:
        return []
    # This filter complements the table's existing Supabase RLS policies.
    response = (
        client.table("predictions")
        .select("*")
        .eq("user_id", str(user_id))
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return response.data or []

def preferences():
    meta = user_metadata(st.session_state.get("user"))
    saved = meta.get("archiload_preferences", {})
    if not isinstance(saved, dict):
        saved = {}
    defaults = {
        "language": "English",
        "default_page": "Overview",
        "parameter_guidance": True,
        "product_updates": False,
        "getting_started_tips": True,
    }
    return {**defaults, **saved}

def update_user_metadata(updates):
    client = st.session_state.supabase
    user = st.session_state.user
    meta = user_metadata(user)
    meta.update(updates)
    response = client.auth.update_user({"data": meta})
    updated_user = get_user_value(response, "user")
    if updated_user is not None:
        st.session_state.user = updated_user
    return updated_user

def page_header(kicker, title, subtitle):
    st.markdown(f'<div class="eyebrow">{kicker}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="hero-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<p class="subtle">{subtitle}</p>', unsafe_allow_html=True)

def card_title(title, subtitle=None):
    st.markdown(f"### {title}")
    if subtitle:
        st.markdown(f'<p class="subtle">{subtitle}</p>', unsafe_allow_html=True)

def render_footer():
    st.markdown(
        f'<div class="divider"></div><p class="small-muted">ArchiLoad AI · Early-stage design support, not a substitute for detailed energy simulation or professional review.</p>',
        unsafe_allow_html=True,
    )

def render_auth(client):
    st.markdown(
        f"""
        <style>
        [data-testid="stSidebar"] {{ display: none; }}
        .auth-panel {{ background: {NAVY}; color: white; border-radius: 22px; padding: 2rem; min-height: 530px; }}
        .auth-panel h1, .auth-panel h2, .auth-panel p {{ color: white; }}
        .auth-panel .auth-muted {{ color: #D7E4EB; line-height: 1.7; }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.02, 0.98], gap="large")
    with left:
        st.markdown('<div class="auth-panel">', unsafe_allow_html=True)
        logo = Path(__file__).resolve().parent / "archiloadai.png"
        building = Path(__file__).resolve().parent / "building-envelope-study.png"
        if logo.exists():
            st.image(str(logo), width=135)
        elif building.exists():
            st.image(str(building), use_container_width=True)
        st.markdown("## Design smarter, earlier.")
        st.markdown("### AI-Based Heating and Cooling Load Predictor")
        st.markdown(
            '<p class="auth-muted">Explore how early-stage building form and glazing choices may influence estimated heating and cooling loads.</p>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown(
            '<p class="auth-muted">For educational exploration and preliminary comparisons only. Results are not a substitute for detailed energy simulation or professional review.</p>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="eyebrow">ARCHILOAD AI WORKSPACE</div>', unsafe_allow_html=True)
        mode = st.radio("Choose an option", ["Sign in", "Create account", "Reset password"], horizontal=True, label_visibility="collapsed")
        if mode == "Sign in":
            st.header("Welcome back")
            st.write("Sign in to continue to your design workspace.")
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
                            st.session_state.auth_message = "Signed in successfully."
                            st.session_state.page = preferences().get("default_page", "Overview")
                            st.rerun()
                        st.error("Sign-in did not return a user. Check your Supabase authentication settings.")
                    except Exception as exc:
                        st.error(f"Could not sign in. Check your email/password and Supabase Auth settings. Details: {exc}")
            st.caption("Use the account registered with your Supabase project.")
        elif mode == "Create account":
            st.header("Create your account")
            st.write("Your saved predictions will be linked to your account.")
            with st.form("signup_form"):
                first = st.text_input("First name")
                last = st.text_input("Last name")
                email = st.text_input("Email address", placeholder="you@example.com")
                password = st.text_input("Create password", type="password")
                confirm = st.text_input("Confirm password", type="password")
                submitted = st.form_submit_button("Create account", type="primary", use_container_width=True)
            if submitted:
                if not first.strip() or not last.strip() or not email.strip() or not password:
                    st.warning("Complete all fields.")
                elif password != confirm:
                    st.warning("The passwords do not match.")
                elif len(password) < 6:
                    st.warning("Use a password with at least 6 characters.")
                else:
                    try:
                        response = client.auth.sign_up({
                            "email": email.strip(),
                            "password": password,
                            "options": {"data": {"first_name": first.strip(), "last_name": last.strip()}},
                        })
                        user = set_user_from_auth_response(response)
                        if user is not None and get_user_value(response, "session") is not None:
                            st.success("Account created. You are signed in.")
                            st.rerun()
                        else:
                            st.success("Account request submitted. If email confirmation is enabled in Supabase, open the confirmation email before signing in.")
                    except Exception as exc:
                        st.error(f"Could not create account. Details: {exc}")
        else:
            st.header("Reset your password")
            st.write("We’ll send a password-reset link to your email address.")
            with st.form("reset_form"):
                email = st.text_input("Email address", placeholder="you@example.com")
                submitted = st.form_submit_button("Send reset link", type="primary", use_container_width=True)
            if submitted:
                if not email.strip():
                    st.warning("Enter your email address.")
                else:
                    try:
                        client.auth.reset_password_for_email(email.strip())
                        st.success("If the address belongs to an account, a password-reset email should arrive shortly. Check your inbox and spam folder.")
                    except Exception as exc:
                        st.error(f"Could not request a reset email. Details: {exc}")
        st.markdown('<p class="small-muted">ArchiLoad AI is an educational prototype for early-stage building design.</p>', unsafe_allow_html=True)

def render_sidebar():
    with st.sidebar:
        logo = Path(__file__).resolve().parent / "archiloadai.png"
        if logo.exists():
            st.image(str(logo), width=130)
        st.markdown('<div class="sidebar-brand">ArchiLoad AI</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-caption">Better early-stage building decisions.</div>', unsafe_allow_html=True)
        st.markdown("---")
        pages = ["Overview", "Prediction", "About", "Settings", "Account"]
        current = st.session_state.get("page", "Overview")
        if current not in pages:
            current = "Overview"
        selected = st.radio(
            "Workspace navigation",
            pages,
            index=pages.index(current),
            label_visibility="collapsed",
            key="sidebar_page",
        )
        if selected != current:
            st.session_state.page = selected
            st.rerun()
        st.markdown("---")
        st.markdown(f"**{display_name()}**")
        st.caption(current_email())
        if st.button("Sign out", use_container_width=True):
            sign_out()
        st.markdown("---")
        st.markdown(
            '<div class="sidebar-caption">Use estimates to compare options—not to certify a design.</div>',
            unsafe_allow_html=True,
        )

def render_overview():
    page_header("WORKSPACE / OVERVIEW", "Design with better insight.", "Explore early-stage building parameters and review your saved heating and cooling load estimates.")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="metric-card"><div class="metric-label">Dataset records</div><div class="metric-value">768</div><div class="small-muted">Simulated building configurations</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card"><div class="metric-label">Prediction outputs</div><div class="metric-value">2</div><div class="small-muted">Heating load and cooling load</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card"><div class="metric-label">Building inputs</div><div class="metric-value">8</div><div class="small-muted">Form, orientation, and glazing</div></div>', unsafe_allow_html=True)
    st.write("")
    left, right = st.columns([1.25, 0.75], gap="large")
    with left:
        with st.container(border=True):
            card_title("From design inputs to load estimates", "A simple workflow for preliminary design exploration.")
            stages = st.columns(3)
            for col, title, detail in zip(
                stages,
                ["01 · Inputs", "02 · AI model", "03 · Estimates"],
                ["Set building form and glazing parameters.", "Trained regression models evaluate the inputs.", "Review heating and cooling load estimates."],
            ):
                with col:
                    st.markdown(f"**{title}**")
                    st.caption(detail)
            st.markdown('<div class="note">Try changing one parameter at a time to understand how the predicted loads respond.</div>', unsafe_allow_html=True)
            if st.button("Start a new prediction", type="primary", use_container_width=True):
                st.session_state.page = "Prediction"
                st.rerun()
        with st.container(border=True):
            card_title("Getting started")
            st.markdown("1. Open **Prediction** and enter the building parameters.")
            st.markdown("2. Generate heating and cooling load estimates.")
            st.markdown("3. Save and revisit your own prediction history.")
    with right:
        with st.container(border=True):
            card_title("About the dataset")
            st.markdown("The app is based on the **UCI Energy Efficiency dataset**, which contains 768 simulated building configurations and eight input features.")
            st.markdown('<span class="pill">Educational prototype</span>', unsafe_allow_html=True)
        with st.container(border=True):
            card_title("Your recent predictions")
            try:
                rows = fetch_predictions(limit=5)
                if rows:
                    recent = pd.DataFrame(rows)
                    show_cols = [c for c in ["created_at", "heating_load", "cooling_load"] if c in recent.columns]
                    if "created_at" in recent.columns:
                        recent["created_at"] = pd.to_datetime(recent["created_at"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
                    st.dataframe(recent[show_cols], use_container_width=True, hide_index=True)
                    if st.button("View prediction history", use_container_width=True):
                        st.session_state.page = "Prediction"
                        st.rerun()
                else:
                    st.caption("No saved predictions yet. Your saved estimates will appear here.")
            except Exception as exc:
                st.warning("Could not load saved predictions. Check the Supabase table and Row Level Security policies.")
                st.caption(str(exc))
    render_footer()

def render_prediction():
    page_header("WORKSPACE / PREDICTION", "Prediction workspace", "Adjust the building parameters below to estimate heating and cooling loads.")
    st.markdown('<div class="note">For valid inputs, use the categories represented in the UCI Energy Efficiency dataset. Predictions are estimates for comparison, not final engineering calculations.</div>', unsafe_allow_html=True)
    st.write("")
    left, right = st.columns([1.08, 0.92], gap="large")
    with left:
        with st.container(border=True):
            card_title("Building parameters", "Enter values for the building configuration.")
            with st.form("prediction_form"):
                a, b = st.columns(2)
                with a:
                    relative_compactness = st.number_input("Relative compactness", min_value=0.60, max_value=1.00, value=float(DEFAULT_INPUTS["relative_compactness"]), step=0.01, format="%.2f", help="A dimensionless measure of building compactness.")
                    surface_area = st.number_input("Surface area (m²)", min_value=500.0, max_value=820.0, value=float(DEFAULT_INPUTS["surface_area"]), step=1.5, format="%.2f")
                    wall_area = st.number_input("Wall area (m²)", min_value=240.0, max_value=420.0, value=float(DEFAULT_INPUTS["wall_area"]), step=1.0, format="%.2f")
                    roof_area = st.number_input("Roof area (m²)", min_value=100.0, max_value=225.0, value=float(DEFAULT_INPUTS["roof_area"]), step=1.0, format="%.2f")
                with b:
                    overall_height = st.selectbox("Overall height (m)", [3.5, 7.0], index=1, help="The dataset includes two overall-height categories.")
                    orientation = st.selectbox("Orientation code", [2, 3, 4, 5], index=0, help="Use the dataset's orientation codes.")
                    glazing_area = st.selectbox("Glazing area (fraction)", [0.00, 0.10, 0.25, 0.40], index=2, format_func=lambda x: f"{x:.2f}", help="Fraction of the floor area represented by glazing in the dataset.")
                    glazing_distribution = st.selectbox("Glazing distribution code", [0, 1, 2, 3, 4, 5], index=0, help="Use the dataset's glazing-distribution codes.")
                guidance = preferences().get("parameter_guidance", True)
                if guidance:
                    st.caption("Tip: change one input at a time when comparing design alternatives.")
                submitted = st.form_submit_button("Generate load estimates", type="primary", use_container_width=True)
            if submitted:
                values = {
                    "relative_compactness": relative_compactness,
                    "surface_area": surface_area,
                    "wall_area": wall_area,
                    "roof_area": roof_area,
                    "overall_height": overall_height,
                    "orientation": orientation,
                    "glazing_area": glazing_area,
                    "glazing_distribution": glazing_distribution,
                }
                try:
                    with st.spinner("Calculating heating and cooling load estimates..."):
                        heating, cooling = predict_loads(values)
                        save_prediction(values, heating, cooling)
                    st.session_state.last_prediction = {
                        "values": values,
                        "heating": heating,
                        "cooling": cooling,
                        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
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
                r1.metric("Heating load", f"{result['heating']:.2f}", help="Model output; use the units associated with your training dataset.")
                r2.metric("Cooling load", f"{result['cooling']:.2f}", help="Model output; use the units associated with your training dataset.")
                st.caption(f"Generated: {result.get('created_at', 'recently')}")
                st.markdown('<div class="warning">These are model estimates for relative comparison. They do not account for every site, weather, occupancy, construction, or HVAC condition.</div>', unsafe_allow_html=True)
            else:
                st.info("Enter the building parameters and select **Generate load estimates** to see results here.")
            st.markdown("---")
            st.markdown("**Explore one change at a time**")
            st.caption("Compare alternative forms, orientations, or glazing settings while holding the other inputs constant.")
            if st.button("Understand the model", use_container_width=True):
                st.session_state.page = "About"
                st.rerun()
    st.write("")
    with st.container(border=True):
        card_title("My prediction history", "Only records associated with your signed-in account are shown.")
        try:
            rows = fetch_predictions(limit=100)
            if rows:
                history = pd.DataFrame(rows)
                if "created_at" in history.columns:
                    history["created_at"] = pd.to_datetime(history["created_at"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
                preferred = [
                    "created_at", "relative_compactness", "surface_area", "wall_area",
                    "roof_area", "overall_height", "orientation", "glazing_area",
                    "glazing_distribution", "heating_load", "cooling_load",
                ]
                cols = [c for c in preferred if c in history.columns]
                st.dataframe(history[cols], use_container_width=True, hide_index=True)
                csv = history[cols].to_csv(index=False).encode("utf-8")
                st.download_button("Download my history as CSV", data=csv, file_name="archiload_prediction_history.csv", mime="text/csv")
            else:
                st.caption("No saved predictions yet. Generate a prediction to start your history.")
        except Exception as exc:
            st.warning("Prediction history could not be loaded. Confirm that the `predictions` table and RLS policies are configured.")
            st.caption(str(exc))
    render_footer()

def render_about():
    page_header("WORKSPACE / ABOUT", "About ArchiLoad AI", "Understand the data, the models, and the limits behind every estimate.")
    tab1, tab2, tab3, tab4 = st.tabs(["Dataset & models", "Model evaluation", "Design relevance", "Responsible use"])
    with tab1:
        a, b = st.columns(2, gap="large")
        with a:
            with st.container(border=True):
                card_title("Dataset")
                st.markdown("**UCI Energy Efficiency dataset**")
                st.markdown("- 768 simulated building configurations")
                st.markdown("- 8 input parameters describing building form and glazing")
                st.markdown("- 2 targets: heating load and cooling load")
                st.caption("The dataset is simulated and does not represent every real building, climate, or operating condition.")
        with b:
            with st.container(border=True):
                card_title("Regression models")
                st.markdown("The app uses trained regression models saved as `heating_model.pkl` and `cooling_model.pkl`. `feature_columns.pkl` defines the feature names and order expected by the models.")
                st.markdown("Model training and evaluation are completed separately from the deployed Streamlit interface.")
    with tab2:
        st.markdown("### Reported evaluation results")
        st.caption("These values were supplied in the interface draft. Verify that they match the final evaluation of the exact model files deployed with this app before including them in a formal report.")
        metrics_df = pd.DataFrame([
            {"Target": target, **values}
            for target, values in REPORTED_METRICS.items()
        ])
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        st.markdown("**MAE — Mean Absolute Error:** average absolute difference between predictions and reference values; lower is better.")
        st.markdown("**RMSE — Root Mean Squared Error:** gives greater weight to larger errors; lower is better.")
        st.markdown("**R² — Coefficient of Determination:** indicates how much variation in the reference values is explained by the model; values closer to 1 generally indicate a better fit.")
    with tab3:
        with st.container(border=True):
            card_title("Relevance to architectural design")
            st.markdown("ArchiLoad AI can support early-stage comparison of building compactness, surface areas, overall height, orientation, and glazing choices.")
            st.markdown("- Use it to compare alternative parameter combinations.")
            st.markdown("- Treat changes in predicted load as prompts for further investigation.")
            st.markdown("- Use detailed building-energy simulation and professional review for design decisions.")
    with tab4:
        with st.container(border=True):
            card_title("Limitations and responsible use")
            st.markdown("- The model is based on a finite set of simulated building configurations.")
            st.markdown("- Site climate, weather files, occupancy, schedules, HVAC systems, construction details, and many other factors are not fully represented.")
            st.markdown("- Results are not measured building performance, utility-cost estimates, code-compliance checks, certification, or final engineering calculations.")
            st.markdown("- Validate important design decisions with appropriate simulation tools and qualified professionals.")
            st.markdown('<div class="warning">Use the model to explore—not to certify.</div>', unsafe_allow_html=True)
    render_footer()

def render_settings():
    page_header("WORKSPACE / SETTINGS", "Settings", "Personalize your workspace. These preferences do not change prediction inputs or models.")
    current = preferences()
    left, right = st.columns([1.2, 0.8], gap="large")
    with left:
        with st.container(border=True):
            card_title("Application preferences")
            with st.form("settings_form"):
                language = st.selectbox("Language", ["English"], index=0, help="English is currently the available interface language.")
                default_page = st.selectbox("Default page after sign-in", ["Overview", "Prediction", "About"], index=["Overview", "Prediction", "About"].index(current.get("default_page", "Overview")) if current.get("default_page", "Overview") in ["Overview", "Prediction", "About"] else 0)
                parameter_guidance = st.toggle("Show parameter guidance", value=bool(current.get("parameter_guidance", True)))
                st.markdown("---")
                card_title("Communication preferences")
                product_updates = st.toggle("Product updates", value=bool(current.get("product_updates", False)))
                getting_started_tips = st.toggle("Getting-started tips", value=bool(current.get("getting_started_tips", True)))
                st.caption("Essential account and security emails may still be sent when needed.")
                save = st.form_submit_button("Save preferences", type="primary")
            if save:
                new_preferences = {
                    "language": language,
                    "default_page": default_page,
                    "parameter_guidance": parameter_guidance,
                    "product_updates": product_updates,
                    "getting_started_tips": getting_started_tips,
                }
                try:
                    update_user_metadata({"archiload_preferences": new_preferences})
                    st.success("Preferences saved to your Supabase user metadata.")
                    st.rerun()
                except Exception as exc:
                    st.error("Preferences could not be saved. Check the Supabase Auth configuration.")
                    st.caption(str(exc))
    with right:
        with st.container(border=True):
            card_title("Account settings")
            st.markdown("Manage your profile, password, and saved predictions from your account page.")
            if st.button("Go to account", use_container_width=True):
                st.session_state.page = "Account"
                st.rerun()
        with st.container(border=True):
            card_title("Preferences, not model controls")
            st.markdown("These settings personalize your workspace only. They do not tune the trained models or change previously saved predictions.")
    render_footer()

def render_account():
    page_header("WORKSPACE / ACCOUNT", "My account", "Manage your personal profile, security, and saved predictions.")
    user = st.session_state.user
    meta = user_metadata(user)
    email = current_email()
    first_default = str(meta.get("first_name", ""))
    last_default = str(meta.get("last_name", ""))
    initials = ((first_default[:1] + last_default[:1]) or email[:1] or "A").upper()
    try:
        count = len(fetch_predictions(limit=10000))
        count_text = str(count)
    except Exception:
        count_text = "—"
    top1, top2 = st.columns([1.1, 0.9], gap="large")
    with top1:
        with st.container(border=True):
            st.markdown(f"## {display_name()}")
            st.markdown(f"<span class='pill'>Signed-in account</span>", unsafe_allow_html=True)
            st.write(email)
            st.caption(f"Profile initials: {initials}")
            st.caption("Your saved predictions are associated with this account.")
        with st.container(border=True):
            card_title("Personal profile")
            with st.form("profile_form"):
                first_name = st.text_input("First name", value=first_default)
                last_name = st.text_input("Last name", value=last_default)
                st.text_input("Email address", value=email, disabled=True)
                save_profile = st.form_submit_button("Save profile", type="primary")
            if save_profile:
                try:
                    update_user_metadata({"first_name": first_name.strip(), "last_name": last_name.strip()})
                    st.success("Profile updated.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not update profile. Details: {exc}")
    with top2:
        with st.container(border=True):
            card_title("Security & sign-in")
            st.markdown("**Password**")
            st.caption("For security, your current password is never displayed.")
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
                        response = st.session_state.supabase.auth.update_user({"password": new_password})
                        st.success("Password updated.")
                    except Exception as exc:
                        st.error(f"Could not update password. Details: {exc}")
            st.markdown("---")
            st.markdown("**Current session**")
            if st.button("Sign out", use_container_width=True):
                sign_out()
        with st.container(border=True):
            card_title("Saved predictions")
            st.markdown(f"<div class='metric-value'>{count_text}</div>", unsafe_allow_html=True)
            st.caption("Predictions saved to your account")
            if st.button("My prediction history", use_container_width=True):
                st.session_state.page = "Prediction"
                st.rerun()
    with st.container(border=True):
        card_title("Make the workspace yours")
        st.write("Adjust your workspace preferences without changing the underlying prediction models.")
        if st.button("Open settings", use_container_width=True):
            st.session_state.page = "Settings"
            st.rerun()
    render_footer()

def main():
    if "page" not in st.session_state:
        st.session_state.page = "Overview"
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    supabase_url = secret_value("SUPABASE_URL")
    supabase_key = secret_value("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        st.error("Supabase connection settings are missing.")
        st.markdown("Add `SUPABASE_URL` and `SUPABASE_KEY` under **Streamlit Cloud → App settings → Secrets**.")
        st.code('SUPABASE_URL = "https://YOUR-PROJECT.supabase.co"\\nSUPABASE_KEY = "YOUR_SUPABASE_PUBLISHABLE_OR_ANON_KEY"', language="toml")
        st.info("Use your project's publishable/anon key. Never place a service-role key in a public app.")
        st.stop()

    try:
        client = get_supabase_client(str(supabase_url), str(supabase_key))
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
        st.session_state.page = "Overview"
        st.rerun()

if __name__ == "__main__":
    main()
