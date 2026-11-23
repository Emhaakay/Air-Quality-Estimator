
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from tensorflow import keras

# Page configuration
st.set_page_config(
    page_title="Air Quality Estimator",
    page_icon="🌍",
    layout="centered"
)

# Load saved model and preprocessing tools
BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_resources():
    model = keras.models.load_model(
        BASE_DIR / "models" / "air_quality_nn.keras"
    )
    preprocessing = joblib.load(
        BASE_DIR / "models" / "preprocessing.joblib"
    )
    return model, preprocessing

model, preprocessing = load_resources()

# Application title
st.title("🌍 Air Quality Estimator")
st.write(
    "Estimate nitrogen dioxide (NO₂) concentration "
    "using weather conditions and sensor measurements."
)

st.divider()

st.subheader("Weather Conditions")

temperature = st.number_input(
    "Temperature (°C)",
    value=20.0
)

humidity = st.number_input(
    "Relative Humidity (%)",
    min_value=0.0,
    max_value=100.0,
    value=50.0
)

absolute_humidity = st.number_input(
    "Absolute Humidity",
    min_value=0.0,
    value=1.0,
    format="%.4f"
)

st.subheader("Sensor Measurements")

sensor_co = st.number_input(
    "PT08.S1(CO)",
    value=1200.0
)

sensor_nmhc = st.number_input(
    "PT08.S2(NMHC)",
    value=900.0
)

sensor_nox = st.number_input(
    "PT08.S3(NOx)",
    value=1000.0
)

sensor_no2 = st.number_input(
    "PT08.S4(NO2)",
    value=1500.0
)

sensor_o3 = st.number_input(
    "PT08.S5(O3)",
    value=1000.0
)

st.subheader("Time Information")

hour = st.slider(
    "Hour of Day",
    min_value=0,
    max_value=23,
    value=12
)

weekday = st.selectbox(
    "Day of the Week",
    [
        "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday"
    ]
)

month = st.selectbox(
    "Month",
    list(range(1, 13)),
    index=5
)

st.divider()

# Make prediction
if st.button("Estimate NO₂ Concentration", type="primary"):

    # Create input dataframe
    input_data = pd.DataFrame([{
        "PT08.S1(CO)": sensor_co,
        "PT08.S2(NMHC)": sensor_nmhc,
        "PT08.S3(NOx)": sensor_nox,
        "PT08.S4(NO2)": sensor_no2,
        "PT08.S5(O3)": sensor_o3,
        "T": temperature,
        "RH": humidity,
        "AH": absolute_humidity,
        "Hour": hour,
        "Weekday": [
            "Monday", "Tuesday", "Wednesday",
            "Thursday", "Friday", "Saturday", "Sunday"
        ].index(weekday),
        "Month": month
    }])

    # Match training feature order
    input_data = input_data[preprocessing["features"]]

    # Apply the same preprocessing used during training
    input_clean = preprocessing["imputer"].transform(input_data)
    input_scaled = preprocessing["scaler"].transform(input_clean)

    # Predict concentration
    prediction = model.predict(
        input_scaled,
        verbose=0
    ).flatten()[0]

    st.success("Prediction completed!")

    st.metric(
        "Estimated NO₂ Concentration",
        f"{prediction:.2f} µg/m³"
    )

    st.caption(
        "This is a machine learning estimate, "
        "not a certified air-quality measurement."
    )
