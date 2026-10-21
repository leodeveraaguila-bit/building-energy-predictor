
import streamlit as st
import pandas as pd
import joblib
from supabase import create_client

# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="Building Energy Predictor",
    page_icon="🏢",
    layout="wide"
)

# ==================================================
# 2. LOAD TRAINED MODELS
# ==================================================
@st.cache_resource
def load_models():
    heating_model = joblib.load("heating_model.pkl")
    cooling_model = joblib.load("cooling_model.pkl")
    feature_columns = joblib.load("feature_columns.pkl")

    if isinstance(feature_columns, pd.DataFrame):
        feature_columns = list(feature_columns.columns)
    else:
        feature_columns = list(feature_columns)

    return heating_model, cooling_model, feature_columns


try:
    heating_model, cooling_model, feature_columns = load_models()

    if len(feature_columns) != 8:
        st.error(
            "feature_columns.pkl should contain the eight input features "
            "used to train your models."
        )
        st.stop()

except Exception as error:
    st.error(
        "Could not load the trained models. Check that these files are "
        "in the root of your GitHub repository: heating_model.pkl, "
        "cooling_model.pkl, and feature_columns.pkl."
    )
    st.stop()


# ==================================================
# 3. SUPABASE CONNECTION
# ==================================================
try:
    supabase_url = st.secrets["SUPABASE_URL"]
    supabase_key = st.secrets["SUPABASE_KEY"]

    if "supabase_client" not in st.session_state:
        st.session_state.supabase_client = create_client(
            supabase_url,
            supabase_key
        )

    supabase = st.session_state.supabase_client

except Exception:
    st.error(
        "Supabase credentials are missing. Add SUPABASE_URL and "
        "SUPABASE_KEY in your Streamlit app's Secrets settings."
    )
    st.stop()


# ==================================================
# 4. MODEL EVALUATION METRICS
# Replace None with your actual Colab test-set values.
# ==================================================
METRICS = {
    "Heating Load": {
        "MAE": None,
        "RMSE": None,
        "R2": None
    },
    "Cooling Load": {
        "MAE": None,
        "RMSE": None,
        "R2": None
    }
}


# ==================================================
# 5. HOMEPAGE / PROJECT OVERVIEW
# ==================================================
st.title("🏢 AI-Based Heating and Cooling Load Predictor")

st.write(
    """
    This application uses machine learning to estimate the heating
    and cooling loads of a building based on selected design parameters.

    It is intended to support early-stage architectural design exploration
    by helping users compare building configurations before detailed
    energy analysis.
    """
)

st.info(
    "These are model-based estimates for preliminary design exploration, "
    "not final engineering calculations or building energy certification."
)

st.subheader("Project Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Dataset Records", "768")
col2.metric("Prediction Outputs", "2")
col3.metric("Building Inputs", "8")

st.write(
    """
    The application uses the UCI Energy Efficiency dataset, which contains
    simulated building configurations. Users enter eight building
    parameters to obtain estimated heating and cooling loads.
    Registered users can save and review their own prediction history.
    """
)


# ==================================================
# 6. LOGIN AND REGISTRATION
# ==================================================
st.sidebar.title("User Account")

if "user_id" not in st.session_state:
    st.session_state.user_id = None

if st.session_state.user_id is None:

    auth_option = st.sidebar.radio(
        "Choose an option",
        ["Log in", "Register"]
    )

    email = st.sidebar.text_input("Email address")
    password = st.sidebar.text_input(
        "Password",
        type="password"
    )

    if st.sidebar.button(auth_option, use_container_width=True):

        if not email.strip() or not password:
            st.sidebar.warning("Enter your email and password.")

        else:
            try:
                if auth_option == "Register":

                    response = supabase.auth.sign_up({
                        "email": email.strip(),
                        "password": password
                    })

                    if response.session is not None:
                        st.session_state.user_id = response.user.id
                        st.rerun()
                    else:
                        st.sidebar.success(
                            "Registration submitted. If email confirmation "
                            "is enabled, confirm your email before logging in."
                        )

                else:
                    response = supabase.auth.sign_in_with_password({
                        "email": email.strip(),
                        "password": password
                    })

                    st.session_state.user_id = response.user.id
                    st.rerun()

            except Exception:
                st.sidebar.error(
                    "Authentication failed. Check your email, password, "
                    "and Supabase Authentication settings."
                )

    st.warning(
        "Please log in or register to use the predictor and access "
        "your saved prediction history."
    )
    st.stop()


# Log out
if st.sidebar.button("Log out", use_container_width=True):
    try:
        supabase.auth.sign_out()
    finally:
        st.session_state.user_id = None
        st.rerun()

st.sidebar.success("Logged in successfully.")


# ==================================================
# 7. BUILDING INPUT FORM
# ==================================================
st.header("Building Design Parameters")

st.write(
    "Adjust the parameters below to explore a building configuration."
)

with st.form("building_input_form"):

    left, right = st.columns(2)

    with left:
        relative_compactness = st.slider(
            "Relative Compactness",
            min_value=0.62,
            max_value=0.98,
            value=0.80,
            step=0.01
        )

        surface_area = st.slider(
            "Surface Area",
            min_value=514.5,
            max_value=808.5,
            value=661.5,
            step=1.0
        )

        wall_area = st.slider(
            "Wall Area",
            min_value=245.0,
            max_value=416.5,
            value=318.5,
            step=1.0
        )

        roof_area = st.slider(
            "Roof Area",
            min_value=110.25,
            max_value=220.5,
            value=147.0,
            step=1.0
        )

    with right:
        overall_height = st.selectbox(
            "Overall Height",
            options=[3.5, 7.0],
            index=1
        )

        orientation = st.selectbox(
            "Orientation",
            options=[2, 3, 4, 5]
        )

        glazing_area = st.selectbox(
            "Glazing Area",
            options=[0.00, 0.10, 0.25, 0.40],
            index=2
        )

        glazing_distribution = st.selectbox(
            "Glazing Area Distribution",
            options=[0, 1, 2, 3, 4, 5]
        )

    predict_button = st.form_submit_button(
        "Predict Heating and Cooling Loads",
        use_container_width=True
    )


# ==================================================
# 8. MAKE PREDICTIONS AND SAVE THEM
# ==================================================
if predict_button:

    input_values = [
        relative_compactness,
        surface_area,
        wall_area,
        roof_area,
        overall_height,
        orientation,
        glazing_area,
        glazing_distribution
    ]

    try:
        # Use the same feature order as during model training.
        input_data = pd.DataFrame(
            [input_values],
            columns=feature_columns
        )

        heating_prediction = float(
            heating_model.predict(input_data)[0]
        )

        cooling_prediction = float(
            cooling_model.predict(input_data)[0]
        )

        # Show the prediction results.
        st.subheader("Prediction Results")

        result_left, result_right = st.columns(2)

        result_left.metric(
            "Predicted Heating Load",
            f"{heating_prediction:.2f}"
        )

        result_right.metric(
            "Predicted Cooling Load",
            f"{cooling_prediction:.2f}"
        )

        st.caption(
            "The values are estimates. Check the dataset documentation "
            "for the target units and definitions."
        )

        # ------------------------------------------
        # ARCHITECTURAL INTERPRETATION
        # ------------------------------------------
        st.subheader("Architectural Interpretation")

        if heating_prediction > cooling_prediction:
            st.write(
                """
                The model estimates a higher heating load than cooling
                load for this configuration. During early design,
                consider investigating envelope heat loss, insulation,
                orientation, and the heating requirements of the building.
                """
            )

        elif cooling_prediction > heating_prediction:
            st.write(
                """
                The model estimates a higher cooling load than heating
                load for this configuration. Consider investigating solar
                exposure, glazing, shading, envelope performance, and
                cooling-system requirements.
                """
            )

        else:
            st.write(
                """
                The model estimates similar heating and cooling loads.
                Both seasonal design conditions should be considered.
                """
            )

        # ------------------------------------------
        # SAVE TO SUPABASE
        # Column names match your existing SQL table.
        # ------------------------------------------
        prediction_record = {
            "user_id": st.session_state.user_id,
            "relative_compactness": relative_compactness,
            "surface_area": surface_area,
            "wall_area": wall_area,
            "roof_area": roof_area,
            "overall_height": overall_height,
            "orientation": orientation,
            "glazing_area": glazing_area,
            "glazing_distribution": glazing_distribution,
            "heating_load": heating_prediction,
            "cooling_load": cooling_prediction
        }

        try:
            supabase.table("predictions").insert(
                prediction_record
            ).execute()

            st.success("Your prediction was saved to your account.")

        except Exception:
            st.error(
                "The prediction was calculated, but saving it to Supabase "
                "failed. Check your authentication session, table columns, "
                "and Row Level Security policies."
            )

    except Exception:
        st.error(
            "The model could not make a prediction. Check that the model "
            "files and feature_columns.pkl match your Colab training setup."
        )


# ==================================================
# 9. MY SAVED PREDICTION HISTORY
# ==================================================
st.header("My Prediction History")

st.write(
    "This section displays saved predictions associated with your account."
)

try:
    history_response = (
        supabase.table("predictions")
        .select(
            "created_at, relative_compactness, surface_area, "
            "wall_area, roof_area, overall_height, orientation, "
            "glazing_area, glazing_distribution, "
            "heating_load, cooling_load"
        )
        .eq("user_id", st.session_state.user_id)
        .order("created_at", desc=True)
        .limit(100)
        .execute()
    )

    history = history_response.data

    if history:
        history_df = pd.DataFrame(history)

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

    else:
        st.info(
            "No saved predictions yet. Submit your first prediction above."
        )

except Exception:
    st.warning(
        "Your prediction history could not be loaded. Verify the table "
        "columns, login session, and Supabase security policies."
    )


# ==================================================
# 10. DATASET AND MODEL INFORMATION
# ==================================================
with st.expander("About the Dataset and Machine Learning Model"):

    st.markdown("### Dataset")

    st.write(
        """
        The UCI Energy Efficiency dataset contains 768 simulated building
        configurations. Its eight input parameters describe building form,
        dimensions, orientation, and glazing.
        """
    )

    st.markdown("### Machine Learning Model")

    st.write(
        """
        The application loads two trained regression models: one estimates
        heating load and the other estimates cooling load. The models were
        trained in Google Colab and saved for use in this web application.
        """
    )

    st.markdown("### Model Evaluation")

    st.write(
        """
        MAE is the mean absolute prediction error. RMSE measures prediction
        error while penalizing larger errors more heavily. R² indicates how
        well the model explains variation in the target values.
        """
    )

    for target_name, metric_values in METRICS.items():

        st.markdown(f"**{target_name}**")

        if all(value is not None for value in metric_values.values()):

            st.write(
                f"MAE: {metric_values['MAE']}  \n"
                f"RMSE: {metric_values['RMSE']}  \n"
                f"R²: {metric_values['R2']}"
            )

        else:
            st.warning(
                "The metrics have not been entered yet. Add the actual "
                "Colab test-set MAE, RMSE, and R² values in the METRICS "
                "section near the top of app.py."
            )


# ==================================================
# 11. ARCHITECTURAL RELEVANCE
# ==================================================
with st.expander("Architecture Relevance and Intended Design Use"):

    st.write(
        """
        Architects and design students can use the application to explore
        how alternative building configurations relate to estimated
        heating and cooling loads during early-stage design.

        The tool can support preliminary comparisons of building form,
        orientation, and glazing choices. It should be used as a
        supplementary design aid rather than a replacement for detailed
        energy simulation or professional engineering assessment.
        """
    )


# ==================================================
# 12. LIMITATIONS AND RESPONSIBLE USE
# ==================================================
with st.expander("Limitations and Responsible Use"):

    st.markdown(
        """
        - The models use a finite dataset of simulated building designs.
        - Estimates may be less reliable for configurations outside the
          range and characteristics represented in the training data.
        - The application does not fully model every real-world factor,
          including local weather, occupancy, operating schedules,
          HVAC equipment, and construction details.
        - Predictions should not be treated as actual energy consumption,
          utility costs, or proof of code compliance.
        - Detailed design decisions should be checked using appropriate
          building-energy simulation tools and qualified professionals.
        """
    )
