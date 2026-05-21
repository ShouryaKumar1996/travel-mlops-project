import streamlit as st
import pandas as pd
import joblib

# ========================================================
# LOAD MODEL
# ========================================================

model = joblib.load("../models/gender_model.pkl")

# ========================================================
# PAGE CONFIG
# ========================================================

st.set_page_config(
    page_title="Gender Classification App",
    page_icon="🧑",
    layout="centered"
)

# ========================================================
# TITLE
# ========================================================

st.title("🧑 Gender Classification System")

st.markdown(
    """
    Predict user gender based on travel and booking behaviour.
    """
)

st.markdown("---")

# ========================================================
# USER INPUTS
# ========================================================

age = st.number_input(
    "Age",
    min_value=18,
    max_value=80,
    value=30
)

total_flights = st.number_input(
    "Total Flights",
    min_value=0,
    value=10
)

avg_flight_price = st.number_input(
    "Average Flight Price",
    min_value=0.0,
    value=15000.0
)

avg_distance = st.number_input(
    "Average Distance",
    min_value=0.0,
    value=800.0
)

# ========================================================
# PREDICTION
# ========================================================

if st.button("Predict Gender"):

    input_df = pd.DataFrame({

        "company": [0],
        "age": [age],
        "total_flights": [total_flights],
        "avg_flight_price": [avg_flight_price],
        "total_flight_spend": [avg_flight_price * total_flights],
        "avg_flight_time": [2.5],
        "avg_distance": [avg_distance],
        "max_flight_price": [avg_flight_price * 1.5],
        "total_hotel_bookings": [5],
        "avg_hotel_price": [6000],
        "total_hotel_spend": [25000],
        "avg_stay_days": [3],
        "max_hotel_spend": [12000],
        "preferred_flight_type": [0],
        "preferred_agency": [0]

    })

    prediction = model.predict(input_df)

    # ========================================================
    # OUTPUT LABEL MAPPING
    # ========================================================

    gender_map = {
        0: "female",
        1: "male",
        2: "none"
    }

    predicted_gender = gender_map.get(
        prediction[0],
        "Unknown"
    )

    st.markdown("---")

    st.success(
        f"Predicted Gender: {predicted_gender}"
    )

    st.balloons()

# ========================================================
# FOOTER
# ========================================================

st.markdown("---")

st.caption(
    "MLOps Capstone Project — Gender Classification Model"
)