import pickle
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu

# Load the trained SVM model
model = pickle.load(open('Weather Model.sav', 'rb'))

# Mapping numerical predictions to labels
weather_map = {0: "Rainy", 1: "Sunny", 2: "Overcast", 3: "Cloudy", 4: "Snowy"}

# Streamlit UI
st.set_page_config(page_title="Weather Predictor", layout="centered")
st.title('🌤️ Weather Type Prediction App')

# Input Form
with st.form("weather_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        temperature = st.number_input("Temperature (°C)", value=25.0)
        season = st.selectbox("Season", ["Winter", "Spring", "Autumn", "Summer"])
        cloud_cover = st.selectbox("Cloud Cover", ["Overcast", "Partly Cloudy", "Clear", "Cloudy"])
        rainfall = st.selectbox("Rainfall", ["No", "Yes"])

    with col2:
        humidity = st.number_input("Humidity (%)", value=60.0)
        pressure = st.number_input("Pressure (hPa)", value=1010.0)
        location = st.selectbox("Location", ["Inland", "Mountain", "Coastal"])

    with col3:
        wind_speed = st.number_input("Wind Speed (km/h)", value=10.0)
        visibility = st.number_input("Visibility (km)", value=50.0)
        dew_point = st.number_input("Dew Point (°C)", value=10.0)

    submitted = st.form_submit_button("Predict")

    if submitted:
        # Encode categorical values
        season_map = {"Winter": 0, "Spring": 1, "Autumn": 2, "Summer": 3}
        cloud_map = {"Overcast": 0, "Partly Cloudy": 1, "Clear": 2, "Cloudy": 3}
        location_map = {"Inland": 0, "Mountain": 1, "Coastal": 2}
        rainfall_map = {"No": 0, "Yes": 1}

        # Create input sample
        sample = np.array([
            temperature,
            humidity,
            wind_speed,
            visibility,
            season_map[season],
            pressure,
            cloud_map[cloud_cover],
            location_map[location],
            dew_point,
            rainfall_map[rainfall]
        ]).reshape(1, -1)

        # Predict using the loaded model
        prediction = model.predict(sample)[0]
        predicted_weather = weather_map.get(prediction, "Unknown")

        # Display result
        st.success(f"🌦️ Predicted Weather Type: **{predicted_weather}**")

