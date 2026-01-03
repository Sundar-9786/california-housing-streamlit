import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --------------------------------------------------
# CRITICAL:  Define ALL custom functions used during training
# These MUST match the exact names and signatures from training
# --------------------------------------------------

def ratio_name(function_transformer, feature_names_in):
    """
    Custom function for naming ratio features in sklearn pipeline. 
    This function was used during model training. 
    """
    return ["ratio"]  # or return appropriate feature names


def column_ratio(X):
    """
    Custom transformer function for sklearn pipeline. 
    Computes ratio between two columns.
    """
    return X[: , [0]] / X[:, [1]]


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="California Housing Price Predictor",
    layout="centered"
)

st.title("🏠 California Housing Price Predictor")

# --------------------------------------------------
# LOAD MODEL (ONCE PER SESSION)
# --------------------------------------------------
@st.cache_resource
def load_model():
    try:
        with open("my_california_housing_model.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("❌ Model file not found!  Please ensure 'my_california_housing_model.pkl' exists.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.exception(e)
        st.stop()

model = load_model()

# Display success message
st. success("✅ Model loaded successfully!")

# --------------------------------------------------
# USER INPUTS
# --------------------------------------------------
st. subheader("Enter House Details")

col1, col2 = st.columns(2)

with col1:
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

    housing_median_age = st. number_input(
        "Housing Median Age",
        value=52.0,
        min_value=0.0,
        help="Example: 52"
    )

    total_rooms = st. number_input(
        "Total Rooms",
        value=3321.0,
        min_value=1.0,
        help="Example: 3321"
    )

    total_bedrooms = st.number_input(
        "Total Bedrooms",
        value=1115.0,
        min_value=1.0,
        help="Example: 1115"
    )

with col2:
    population = st.number_input(
        "Population",
        value=1576.0,
        min_value=1.0,
        help="Example: 1576"
    )

    households = st.number_input(
        "Households",
        value=1034.0,
        min_value=1.0,
        help="Example: 1034"
    )

    median_income = st.number_input(
        "Median Income (in tens of thousands)",
        value=2.0987,
        min_value=0.0,
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
if st. button("🔮 Predict House Value", type="primary"):
    # Create input dataframe with exact column names expected by model
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

    # Display input data
    with st.expander("📋 View Input Data"):
        st.dataframe(input_df)

    # Make prediction
    try:
        with st.spinner("Making prediction..."):
            prediction = model.predict(input_df)
        
        # Display result
        st.success(f"### Predicted House Value: ${prediction[0]: ,.2f}")
        
        # Additional info
        st.info(f"""
        **Prediction Details:**
        - Estimated Value: ${prediction[0]:,.2f}
        - Annual Income: ${median_income * 10000:,.2f}
        - Location: {ocean_proximity}
        """)
        
    except Exception as e:
        st.error("❌ Prediction failed!")
        st.error(f"Error details: {str(e)}")
        
        # Debug information
        with st.expander("🔍 Debug Information"):
            st.write("Input DataFrame:")
            st.write(input_df)
            st.write("\nInput DataFrame dtypes:")
            st.write(input_df.dtypes)
            st.write("\nInput shape:")
            st.write(input_df.shape)
            st.exception(e)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.markdown("Built with Streamlit 🎈 | Model:  California Housing Price Prediction")
