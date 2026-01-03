import streamlit as st
import pandas as pd
import numpy as np
import dill  # Changed from pickle
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn. metrics. pairwise import rbf_kernel

# --------------------------------------------------
# CUSTOM TRANSFORMERS (MUST MATCH TRAINING CODE)
# --------------------------------------------------

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    """
    Custom transformer that computes similarity to cluster centers.
    This MUST match the exact implementation used during training. 
    """
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self. gamma = gamma
        self.random_state = random_state
        self.kmeans_ = None

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None):
        return [f"Cluster_{i}_similarity" for i in range(self. n_clusters)]


def ratio_name(X):
    """
    Custom function for computing ratios. 
    This MUST be defined with the exact same name as in training. 
    """
    return X[: , [0]] / X[:, [1]]


def column_ratio(X):
    """
    Alternative ratio function - keeping for compatibility. 
    """
    return X[: , [0]] / X[:, [1]]


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="California Housing Price Predictor",
    layout="centered",
    page_icon="🏠"
)

st.title("🏠 California Housing Price Predictor")

# --------------------------------------------------
# LOAD MODEL (ONCE PER SESSION)
# --------------------------------------------------
@st.cache_resource
def load_model():
    try:
        # Try with dill first
        with open("my_california_housing_model.pkl", "rb") as f:
            model = dill.load(f)
        return model
    except Exception as e:
        st. error(f"❌ Error with dill: {str(e)}")
        
        # Fallback to pickle
        try:
            import pickle
            with open("my_california_housing_model.pkl", "rb") as f:
                model = pickle.load(f)
            return model
        except FileNotFoundError:
            st. error("❌ Model file not found!  Please ensure 'my_california_housing_model.pkl' exists.")
            st.stop()
        except Exception as e2:
            st.error(f"❌ Error loading model:  {str(e2)}")
            st.error("**Possible causes:**")
            st.error("- Python version mismatch between training and deployment")
            st.error("- scikit-learn version mismatch")
            st.error("- Model needs to be re-saved with proper module structure")
            st.stop()

model = load_model()

# Display success message
st.success("✅ Model loaded successfully!")

# --------------------------------------------------
# USER INPUTS
# --------------------------------------------------
st.subheader("Enter House Details")

col1, col2 = st.columns(2)

with col1:
    longitude = st.number_input(
        "Longitude",
        value=-122.42,
        format="%.2f",
        help="Example: -122.42"
    )

    latitude = st.number_input(
        "Latitude",
        value=37.80,
        format="%.2f",
        help="Example: 37.80"
    )

    housing_median_age = st. number_input(
        "Housing Median Age",
        value=52.0,
        min_value=0.0,
        max_value=100.0,
        help="Example: 52"
    )

    total_rooms = st.number_input(
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
        "Median Income (in $10,000s)",
        value=2.0987,
        min_value=0.0,
        format="%. 4f",
        help="Example:  2.0987 = $20,987"
    )

    ocean_proximity = st.selectbox(
        "Ocean Proximity",
        options=[
            "<1H OCEAN",
            "INLAND",
            "NEAR OCEAN",
            "NEAR BAY",
            "ISLAND"
        ],
        help="Select the proximity to ocean"
    )

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
st. markdown("---")

if st.button("🔮 Predict House Value", type="primary", use_container_width=True):
    # Create input dataframe with exact column names from training
    input_df = pd.DataFrame([{
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households":  households,
        "median_income": median_income,
        "ocean_proximity": ocean_proximity
    }])

    # Display input data
    with st.expander("📋 View Input Data"):
        st.dataframe(input_df, use_container_width=True)

    # Make prediction
    try:
        with st.spinner("🔄 Making prediction..."):
            prediction = model.predict(input_df)
        
        # Display result with styling
        st.success("### ✨ Prediction Complete!")
        
        # Main prediction display
        st.markdown(f"""
        <div style='padding: 20px; background-color: #f0f2f6; border-radius: 10px; text-align: center;'>
            <h2 style='color:  #1f77b4; margin: 0;'>Predicted House Value</h2>
            <h1 style='color: #2ca02c; margin: 10px 0;'>${prediction[0]:,.2f}</h1>
        </div>
        """, unsafe_allow_html=True)
        
        # Additional information
        st.markdown("---")
        st.subheader("📊 Input Summary")
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a: 
            st.metric("Location", f"({latitude:. 2f}, {longitude:.2f})")
            st.metric("Median Income", f"${median_income * 10000:,. 0f}")
        
        with col_b: 
            st.metric("Total Rooms", f"{total_rooms: ,. 0f}")
            st.metric("Bedrooms", f"{total_bedrooms:,.0f}")
        
        with col_c:
            st.metric("Population", f"{population:,.0f}")
            st.metric("Ocean Proximity", ocean_proximity)
        
    except Exception as e: 
        st.error("❌ Prediction failed!")
        st.error(f"**Error details:** {str(e)}")
        
        # Debug information
        with st.expander("🔍 Debug Information (Click to expand)"):
            st.write("**Input DataFrame:**")
            st.write(input_df)
            st.write("\n**DataFrame Data Types:**")
            st.write(input_df.dtypes)
            st.write("\n**Model Information:**")
            st.write(f"Model type: {type(model)}")
            st.write("\n**Full Error Traceback:**")
            st.exception(e)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with ❤️ using Streamlit | California Housing Price Prediction Model</p>
    <p><small>Based on California Housing Dataset</small></p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SIDEBAR INFO
# --------------------------------------------------
with st.sidebar:
    st. header("ℹ️ About")
    st.info("""
    This app predicts California housing prices based on:
    - 📍 Geographic location (latitude/longitude)
    - 🏘️ Housing characteristics
    - 👥 Demographics
    - 🌊 Proximity to ocean
    """)
    
    st.header("📖 How to Use")
    st.markdown("""
    1. Enter the house details in the form
    2. Click **Predict House Value**
    3. View the predicted price
    """)
    
    st.header("💡 Tips")
    st.markdown("""
    - **Median Income** is measured in $10,000s
    - **Longitude** should be negative (West coast)
    - **Latitude** ranges from 32-42 for California
    """)
