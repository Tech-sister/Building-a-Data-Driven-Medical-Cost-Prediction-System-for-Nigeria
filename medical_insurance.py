import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

#set page config
st.set_page_config(
    page_title="Medical Insurance",
    page_icon= "🏥",
    layout="centered"
)


@st.cache_data
def load_data():
    return pd.read_csv("data/clean_nigeria_medical_insurance.csv")


data = load_data()



@st.cache_resource
def load_model():

    model = joblib.load("model.pkl")
    return model


model = load_model()



st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Home Page",
        "EDA Dashboard",
        "Prediction Page",
        "Model Performance Page"
    ]
)



if page == "Home Page":

    st.title("Medical Cost Prediction Dashboard")

    st.write(
        """
        This dashboard predicts medical costs using machine learning.
        It allows users to explore healthcare cost patterns and estimate
        expected hospital bills based on patient information.
        """
    )


    # Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
            st.metric(
        label="Total Patients",
        value=f"{len(data):,}"
    )

    avg_bill = data["Hospital Bill"].mean() / 1_000_000
    max_bill = data["Hospital Bill"].max() / 1_000_000
    min_bill = data["Hospital Bill"].min() / 1_000_000

    with col2:
        st.metric(label="Average Hospital Bill", value=f"₦{avg_bill:,.2f}M")

    with col3:
        st.metric(label="Maximum Hospital Bill", value=f"₦{max_bill:,.2f}M")

    with col4:
        st.metric(label="Minimum Hospital Bill", value=f"₦{min_bill:,.2f}M")


elif page == "EDA Dashboard":

    st.title("Exploratory Data Analysis Dashboard")


    # Cost Distribution
    fig1 = px.histogram(
        data,
        x="Hospital Bill",
        title="Hospital Bill Distribution",
        nbins=30
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )


    # Smoker Analysis
    fig2 = px.box(
        data,
        x="Smoker",
        y="Hospital Bill",
        title="Hospital Bill by Smoker Status"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )


    # BMI Relationship
    fig3 = px.scatter(
        data,
        x="Bmi",
        y="Hospital Bill",
        color="Smoker",
        title="BMI vs Hospital Bill"
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )



elif page == "Prediction Page":

    st.title("Medical Cost Prediction")


    age = st.number_input(
        "Age",
        min_value=18,
        max_value=64
    )


    gender = st.selectbox(
        "Gender",
        data["Gender"].unique()
    )
    gender = 1 if gender == 'Male' else 0


    bmi = st.number_input(
        "BMI",
        min_value=16.0,
        max_value=53.1
    )


    children = st.number_input(
        "Number Of Children",
        min_value=0,
        max_value=2
    )


    smoker = st.selectbox(
        "Smoker",
        data["Smoker"].unique()
    )
    smoker = 1 if smoker == "Yes" else 0


    state = st.selectbox(
        "State",
        data["State"].unique()
    )


    if st.button("Predict Medical Cost"):


        input_data = pd.DataFrame({

            "Age":[age],
            "Gender":[gender],
            "BMI":[bmi],
            "Number Of Children":[children],
            "Smoker Status":[smoker],
            "State":[state]

        })




        prediction = model.predict(input_data)[0]


        st.success(
            f"Predicted Medical Cost: ₦{prediction:,.2f}"
        )


        # Risk Indicator

        if prediction < 500000:

            st.success("Risk Level: Low")

        elif prediction < 1500000:

            st.warning("Risk Level: Medium")

        else:

            st.error("Risk Level: High")




elif page == "Model Performance Page":

    st.title("Model Performance")


    st.write(
        """
        Model evaluation metrics.
        """
    )


    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "R² Score",
            "0.10"
        )

    with col2:
        st.metric(
            "MAE",
            "₦12953810.63"
        )

    with col3:
        st.metric(
            "RMSE",
            "₦16475160.17"
        )
    with col4:
        st.metric(
            "MSE",
            "₦271430902601922.09"
        )


    # Feature Importance

    if hasattr(model, "feature_importances_"):

        importance = pd.DataFrame({

            "Feature": data.drop(
                "Hospital Bill",
                axis=1
            ).columns,

            "Importance": model.feature_importances_

        })


        importance = importance.sort_values(
            "Importance",
            ascending=False
        )


        fig = px.bar(
            importance,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Feature Importance"
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )
def main():
    if __name__ == "__main__":
        main()