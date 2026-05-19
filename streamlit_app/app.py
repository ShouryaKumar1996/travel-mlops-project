import streamlit as st
import requests
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="Flight Price Prediction",
    page_icon="✈️",
    layout="centered"
)

# Load Dataset
df = pd.read_csv("../data/flights.csv")

# Title
st.title("✈️ Flight Price Prediction System")

st.write("Enter flight details to predict ticket price.")

# Dynamic Dropdown Inputs
from_city = st.selectbox(
    "From City",
    sorted(df['from'].unique())
)

to_city = st.selectbox(
    "To City",
    sorted(df['to'].unique())
)

flight_type = st.selectbox(
    "Flight Type",
    sorted(df['flightType'].unique())
)

agency = st.selectbox(
    "Agency",
    sorted(df['agency'].unique())
)

# Numerical Inputs
flight_time = st.number_input(
    "Flight Time (hours)",
    min_value=0.0,
    value=1.5
)

distance = st.number_input(
    "Distance",
    min_value=0.0,
    value=500.0
)

year = st.number_input(
    "Year",
    min_value=2019,
    max_value=2030,
    value=2019
)

month = st.number_input(
    "Month",
    min_value=1,
    max_value=12,
    value=9
)

day = st.number_input(
    "Day",
    min_value=1,
    max_value=31,
    value=26
)

weekday = st.number_input(
    "Weekday",
    min_value=0,
    max_value=6,
    value=3
)

# Prediction Button
if st.button("Predict Flight Price"):

    url = "http://localhost:5001/predict"

    payload = {
        "from": from_city,
        "to": to_city,
        "flightType": flight_type,
        "agency": agency,
        "time": flight_time,
        "distance": distance,
        "year": year,
        "month": month,
        "day": day,
        "weekday": weekday
    }

    try:
        response = requests.post(url, json=payload)

        prediction = response.json()

        if "predicted_price" in prediction:

            st.success(
                f"Predicted Flight Price: ₹ {round(prediction['predicted_price'], 2)}"
            )

        else:
            st.error(prediction)

    except Exception as e:

        st.error(f"Error: {e}")