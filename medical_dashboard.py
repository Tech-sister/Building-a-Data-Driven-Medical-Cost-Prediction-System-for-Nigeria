import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Medical Cost Prediction System",
    page_icon="⚕️",
    layout="wide"
)


# ============================================================
# LOAD MODEL ARTIFACTS
# ============================================================

@st.cache_resource
def load_model_artifact():

    try:
        model = joblib.load("random_forest_model.pkl")
        scaler = joblib.load("scaler.pkl")
        label_encoder = joblib.load("label_encoder.pkl")
        feature_columns = joblib.load("feature_columns.pkl")

        return model, scaler, label_encoder, feature_columns

    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        st.stop()

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.stop()


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    try:
        df = pd.read_csv("data/clean_nigeria_medical_insurance.csv")

        return df

    except FileNotFoundError as e:
        st.error(f"Dataset not found: {e}")
        st.stop()

    except Exception as e:
        st.error(f"An error occurred while loading the dataset: {e}")
        st.stop()


# ============================================================
# SIDEBAR FILTERS
# ============================================================

def sidebar_filters(df):

    st.sidebar.header("🔎 Filters")

    state = st.sidebar.multiselect(
        "Select State",
        options=sorted(df["State"].unique()),
        default=sorted(df["State"].unique())
    )

    smoker = st.sidebar.radio(
        "Smoker?",
        options=["All", "Yes", "No"],
        index=0
    )

    return state, smoker


# ============================================================
# FILTER DATA
# ============================================================

def filter_data(df, state, smoker):

    filtered_df = df[df["State"].isin(state)]

    if smoker != "All":
        filtered_df = filtered_df[
            filtered_df["Smoker"] == smoker
        ]

    return filtered_df


# ============================================================
# DISPLAY MAIN METRICS
# ============================================================

def display_metrics(filtered_df):

    col1, col2, col3, col4, col5 = st.columns(5)

    # Total Patients
    with col1:
        st.metric(
            "👥 Total Patients",
            len(filtered_df)
        )

    # Average Hospital Bill
    with col2:

        avg_bill = (
            filtered_df["Hospital Bill"].mean()
            if len(filtered_df) > 0
            else 0
        )

        st.metric(
            "💰 Average Hospital Bill",
            f"₦{avg_bill:,.0f}"
        )

    # Total Bills
    with col3:

        total_bill = (
            filtered_df["Hospital Bill"].sum()
            if len(filtered_df) > 0
            else 0
        )

        st.metric(
            "💵 Total Bills Paid",
            f"₦{total_bill:,.0f}"
        )

    # Top State
    with col4:

        if len(filtered_df) > 0:
            top_state = filtered_df["State"].mode()[0]
        else:
            top_state = "N/A"

        st.metric(
            "📍 Top State",
            top_state
        )

    # Highest Bill
    with col5:

        highest_bill = (
            filtered_df["Hospital Bill"].max()
            if len(filtered_df) > 0
            else 0
        )

        st.metric(
            "🏥 Highest Bill",
            f"₦{highest_bill:,.0f}"
        )


# ============================================================
# HOME PAGE
# ============================================================

def homepage(filtered_df):

    st.header("🏠 Home Page")

    st.subheader(
        "Nigerian Healthcare Expenditure Analytics Platform"
    )

    col1, col2 = st.columns(2)

    # Average Age
    with col1:

        avg_age = (
            filtered_df["Age"].mean()
            if len(filtered_df) > 0
            else 0
        )

        st.metric(
            "👤 Average Age",
            f"{avg_age:,.2f}"
        )

    # Average BMI
    with col2:

        avg_bmi = (
            filtered_df["Bmi"].mean()
            if len(filtered_df) > 0
            else 0
        )

        st.metric(
            "⚖️ Average BMI",
            f"{avg_bmi:,.2f}"
        )

    # --------------------------------------------------------
    # TOP STATES
    # --------------------------------------------------------

    st.subheader("🏆 Top States by Hospital Bills")

    if len(filtered_df) > 0:

        top_states = (
            filtered_df
            .groupby("State")["Hospital Bill"]
            .sum()
            .sort_values(ascending=True)
        )

        fig = px.bar(
            x=top_states.index,
            y=top_states.values,
            title="Total Hospital Bills by State",
            labels={
                "x": "State",
                "y": "Hospital Bills"
            }
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------------
    # SMOKER DISTRIBUTION
    # --------------------------------------------------------

    st.subheader("🚬 Smoker Distribution")

    if len(filtered_df) > 0:

        smokers = filtered_df["Smoker"].value_counts()

        fig = px.pie(
            values=smokers.values,
            names=smokers.index,
            hole=0.4,
            title="Smoker Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# EDA DASHBOARD
# ============================================================

def eda_dashboard(filtered_df):

    st.header("📊 Exploratory Data Analysis")

    if len(filtered_df) == 0:

        st.warning(
            "No data available for the selected filters."
        )

        return

    # --------------------------------------------------------
    # UNIVARIATE ANALYSIS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    # Age Distribution
    with col1:

        st.subheader("Age Distribution")

        fig = px.histogram(
            filtered_df,
            x="Age",
            nbins=15,
            title="Distribution of Age"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # BMI Distribution
    with col2:

        st.subheader("BMI Distribution")

        fig = px.histogram(
            filtered_df,
            x="Bmi",
            nbins=15,
            title="Distribution of BMI"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------------
    # HOSPITAL BILL + SMOKER
    # --------------------------------------------------------

    col3, col4 = st.columns(2)

    with col3:

        st.subheader("Medical Bill Distribution")

        fig = px.histogram(
            filtered_df,
            x="Hospital Bill",
            nbins=15,
            title="Distribution of Hospital Bills"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col4:

        st.subheader("Smoker Distribution")

        smokers = filtered_df["Smoker"].value_counts()

        fig = px.pie(
            values=smokers.values,
            names=smokers.index,
            hole=0.4,
            title="Smoker Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------------
    # BIVARIATE ANALYSIS
    # --------------------------------------------------------

    st.header("📈 Bivariate Analysis")

    col5, col6 = st.columns(2)

    # Age vs Hospital Bill
    with col5:

        st.subheader("Age vs Hospital Bill")

        fig = px.scatter(
            filtered_df,
            x="Age",
            y="Hospital Bill",
            title="Age vs Hospital Bill",
            labels={
                "Age": "Age",
                "Hospital Bill": "Hospital Bill"
            }
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # State vs Hospital Bill
    with col6:

        st.subheader("🏥 Hospital Bills by State")

        top_states = (
            filtered_df
            .groupby("State")["Hospital Bill"]
            .sum()
            .sort_values(ascending=True)
        )

        fig = px.bar(
            x=top_states.index,
            y=top_states.values,
            title="Hospital Bills by State",
            labels={
                "x": "State",
                "y": "Hospital Bills"
            }
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# DISPLAY DATA TABLE
# ============================================================

def table_data(filtered_df):

    if len(filtered_df) > 0:

        st.dataframe(
            filtered_df,
            use_container_width=True,
            height=300
        )

    else:

        st.warning(
            "No patient data to display."
        )


# ============================================================
# GET FILTERED OPTIONS
# ============================================================

def get_filtered_options(
    df,
    age=None,
    gender=None,
    bmi=None,
    children=None,
    smoker=None
):

    filtered_df = df.copy()

    if age is not None:
        filtered_df = filtered_df[
            filtered_df["Age"] == age
        ]

    if gender is not None:
        filtered_df = filtered_df[
            filtered_df["Gender"] == gender
        ]

    if bmi is not None:
        filtered_df = filtered_df[
            filtered_df["Bmi"] == bmi
        ]

    if children is not None:
        filtered_df = filtered_df[
            filtered_df["Children"] == children
        ]

    if smoker is not None:
        filtered_df = filtered_df[
            filtered_df["Smoker"] == smoker
        ]

    options = {

        "Age": sorted(
            filtered_df["Age"].unique().tolist()
        ),

        "Gender": sorted(
            filtered_df["Gender"].unique().tolist()
        ),

        "BMI": sorted(
            filtered_df["Bmi"].unique().tolist()
        ),

        "Children": sorted(
            filtered_df["Children"].unique().tolist()
        ),

        "Smoker": sorted(
            filtered_df["Smoker"].unique().tolist()
        ),

        "State": sorted(
            filtered_df["State"].unique().tolist()
        )
    }

    return options


# ============================================================
# PREDICT MEDICAL BILL
# ============================================================

def predict_bill(
    patient_data,
    model,
    scaler,
    label_encoder,
    feature_columns
):

    try:

        input_df = pd.DataFrame(
            [patient_data]
        )

        # Encode categorical variables
        categorical_columns = [
            "Gender",
            "Smoker",
            "State"
        ]

        for col in categorical_columns:

            input_df[col + "_encoded"] = (
                label_encoder[col].transform(
                    input_df[col]
                )
            )

        # Prepare features
        feature_dict = {

            "Age": patient_data["Age"],

            "Gender_encoded":
                input_df["Gender_encoded"].values[0],

            "Bmi":
                patient_data["Bmi"],

            "Children":
                patient_data["Children"],

            "Smoker_encoded":
                input_df["Smoker_encoded"].values[0],

            "State_encoded":
                input_df["State_encoded"].values[0]
        }

        # Create feature array
        features = np.array([
            [
                feature_dict[col]
                for col in feature_columns
            ]
        ])

        # Scale features
        scaled_features = scaler.transform(
            features
        )

        # Make prediction
        predicted_bill = model.predict(
            scaled_features
        )[0]

        # Prediction range
        margin_percentage = 0.15

        min_bill = predicted_bill * (
            1 - margin_percentage
        )

        max_bill = predicted_bill * (
            1 + margin_percentage
        )

        return {

            "predicted_bill": predicted_bill,

            "min_predicted_bill": min_bill,

            "max_predicted_bill": max_bill
        }

    except Exception as e:

        st.error(
            f"An error occurred during prediction: {e}"
        )

        return None


# ============================================================
# PREDICTION INTERFACE
# ============================================================

def prediction_interface(df):

    st.header(
        "🏥 Medical Insurance Evaluation System"
    )

    st.write(
        "Fill in all the required fields to get an estimated medical bill."
    )

    # Load model
    model, scaler, label_encoder, feature_columns = (
        load_model_artifact()
    )

    # --------------------------------------------------------
    # INPUT FIELDS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        gender = st.selectbox(
            "Gender *",
            options=df["Gender"].unique()
        )

        bmi = st.number_input(
            "Body Mass Index (BMI) *",
            min_value=6.0,
            max_value=53.1,
            value=25.0
        )

        children = st.number_input(
            "Number of Children *",
            min_value=0,
            max_value=5,
            value=0,
            step=1
        )

    with col2:

        smoker = st.selectbox(
            "Smoker *",
            options=df["Smoker"].unique()
        )

        age = st.number_input(
            "Age *",
            min_value=18,
            max_value=64,
            value=30,
            step=1
        )

        state = st.selectbox(
            "State *",
            options=sorted(df["State"].unique())
        )

    # --------------------------------------------------------
    # PREDICTION BUTTON
    # --------------------------------------------------------

    if st.button(
        "🔮 Get Medical Bill Prediction",
        use_container_width=True
    ):

        patient_data = {

            "Age": age,

            "Gender": gender,

            "Bmi": bmi,

            "Children": children,

            "Smoker": smoker,

            "State": state
        }

        with st.spinner(
            "Calculating estimated medical bill..."
        ):

            result = predict_bill(
                patient_data,
                model,
                scaler,
                label_encoder,
                feature_columns
            )

        if result is not None:

            st.markdown("---")

            st.subheader(
                "💰 Estimated Medical Bill"
            )

            st.success(
                f"₦{result['predicted_bill']:,.0f}"
            )

            st.write(
                f"**Estimated Bill Range:** "
                f"₦{result['min_predicted_bill']:,.0f} "
                f"– "
                f"₦{result['max_predicted_bill']:,.0f}"
            )

            st.subheader(
                "👤 Patient Information"
            )

            st.write(
                f"**Age:** {age}"
            )

            st.write(
                f"**Gender:** {gender}"
            )

            st.write(
                f"**BMI:** {bmi}"
            )

            st.write(
                f"**Children:** {children}"
            )

            st.write(
                f"**Smoker:** {smoker}"
            )

            st.write(
                f"**State:** {state}"
            )


# ============================================================
# MODEL PERFORMANCE PAGE
# ============================================================

def model_performance_page():

    st.header("🤖 Model Performance")

    st.write(
        "Performance evaluation of the Random Forest Regression model."
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Train R²",
            "0.65"
        )

    with col2:
        st.metric(
            "Test R²",
            "0.05"
        )

    with col3:
        st.metric(
            "Train RMSE",
            "₦6,419,626.59"
        )

    with col4:
        st.metric(
            "Test RMSE",
            "₦8,941,840.89"
        )


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    # Load dataset
    df = load_data()

    if df is None:
        return

    # --------------------------------------------------------
    # SIDEBAR NAVIGATION
    # --------------------------------------------------------

    st.sidebar.title("📌 Dashboard Navigation")

    page = st.sidebar.radio(
        "Select Page",
        [
            "Home Page",
            "EDA Dashboard",
            "Prediction Interface",
            "Model Performance"
        ]
    )

    # --------------------------------------------------------
    # SIDEBAR FILTERS
    # --------------------------------------------------------

    state, smoker_filter = sidebar_filters(df)

    filtered_df = filter_data(
        df,
        state,
        smoker_filter
    )

    # --------------------------------------------------------
    # MAIN TITLE
    # --------------------------------------------------------

    st.title(
        "⚕️ Medical Cost Prediction System"
    )

    st.markdown("---")

    # --------------------------------------------------------
    # DISPLAY SELECTED PAGE
    # --------------------------------------------------------

    if page == "Home Page":

        display_metrics(filtered_df)

        homepage(filtered_df)

        st.markdown("---")

        table_data(filtered_df)

    elif page == "EDA Dashboard":

        display_metrics(filtered_df)

        eda_dashboard(filtered_df)

        st.markdown("---")

        table_data(filtered_df)

    elif page == "Prediction Interface":

        prediction_interface(df)

    elif page == "Model Performance":

        model_performance_page()


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    main()

