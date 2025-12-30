import streamlit as st
import pandas as pd
import numpy as np
import pickle
# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------


st.set_page_config(
    page_title="California Housing Price Predictor",
    layout="centered"
)

st.title("California Housing Price Predictor")

# --------------------------------------------------
# LOAD MODEL (ONCE PER SESSION)
# --------------------------------------------------

def column_ratio(X):
    
    return X[:, [0]] / X[:, [1]]


@st.cache_resource
def load_model():
    with open("my_california_housing_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# --------------------------------------------------
# USER INPUTS
# --------------------------------------------------
st.subheader("Enter House Details")

longitude = st.number_input(
    "Longitude",
    value=-122.42,
    help="Example: -122.42"
)

latitude = st.number_input(
    "Latitude",
    value=37.80,
    help="Example: 37.80"
)

housing_median_age = st.number_input(
    "Housing Median Age",
    value=52.0,
    help="Example: 52"
)

total_rooms = st.number_input(
    "Total Rooms",
    value=3321.0,
    help="Example: 3321"
)

total_bedrooms = st.number_input(
    "Total Bedrooms",
    value=1115.0,
    help="Example: 1115"
)

population = st.number_input(
    "Population",
    value=1576.0,
    help="Example: 1576"
)

households = st.number_input(
    "Households",
    value=1034.0,
    help="Example: 1034"
)

median_income = st.number_input(
    "Median Income",
    value=2.0987,
    help="Example: 2.0987"
)

ocean_proximity = st.selectbox(
    "Ocean Proximity",
    (
        "<1H OCEAN",
        "INLAND",
        "NEAR OCEAN",
        "NEAR BAY",
        "ISLAND"
    )
)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
if st.button("Predict House Value"):
    input_df = pd.DataFrame([{
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "ocean_proximity": ocean_proximity
    }])

    try:
        prediction = model.predict(input_df)
        st.success(f"Predicted House Value: {prediction[0]:,.2f}")
    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)
