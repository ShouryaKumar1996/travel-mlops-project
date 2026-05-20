import streamlit as st
import requests
import pandas as pd

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Flight Price Prediction",
    page_icon="✈️",
    layout="centered"
)

# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv("../data/flights.csv")

# =========================================================
# TITLE
# =========================================================

st.title("✈️ Flight Price Prediction System")

st.write("Enter flight details to predict ticket price.")

# =========================================================
# INPUTS
# =========================================================

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

# =========================================================
# NUMERICAL INPUTS
# =========================================================

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

# =========================================================
# PREDICTION BUTTON
# =========================================================

if st.button("Predict Flight Price"):

    url = "http://localhost:5001/predict"

    # Current Date Features
    today = pd.Timestamp.today()

    weekday = today.dayofweek

    month = today.month

    # Duration Category
    if flight_time <= 2:

        duration_category = 'short'

    elif flight_time <= 5:

        duration_category = 'medium'

    else:

        duration_category = 'long'

    # Payload
    payload = pd.DataFrame([{

        'from': from_city,

        'to': to_city,

        'flightType': flight_type,

        'agency': agency,

        'time': flight_time,

        'distance': distance,

        'year': today.year,

        'month': month,

        'day': today.day,

        'weekday': weekday,

        'is_weekend': 1 if weekday >= 5 else 0,

        'is_peak_month': 1 if month in [6, 7, 12] else 0,

        'duration_category': duration_category

    }])

    try:

        response = requests.post(
            url,
            json=payload.to_dict(orient='records')[0]
        )

        prediction = response.json()

        if "predicted_price" in prediction:

            st.success(
                f"Predicted Flight Price: ₹ {round(prediction['predicted_price'], 2)}"
            )

        else:

            st.error(prediction)

    except Exception as e:

        st.error(f"Error: {e}")
   