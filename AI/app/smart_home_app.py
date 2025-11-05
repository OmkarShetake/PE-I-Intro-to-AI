import streamlit as st
import pandas as pd
import pickle
import numpy as np
from src.energy_tips import energy_tip

st.set_page_config(page_title="Smart Home Energy Predictor", page_icon="🏠", layout="centered")
st.title("🏠 Smart Home Energy Predictor")
st.write("Predict your daily energy usage (kWh) based on Temperature, Humidity and Hour. Includes energy-saving tips & a small usage graph.")

# Load model
with open("models/energy_model.pkl", "rb") as f:
    model = pickle.load(f)

# Inputs
col1, col2, col3 = st.columns(3)
with col1:
    temp = st.number_input("🌡️ Temperature (°C):", min_value=0.0, max_value=50.0, value=30.0, step=0.5)
with col2:
    hum = st.number_input("💧 Humidity (%):", min_value=0.0, max_value=100.0, value=65.0, step=1.0)
with col3:
    hour = st.number_input("🕒 Hour of Day (0–23):", min_value=0, max_value=23, value=12, step=1)

if st.button("🔍 Predict Energy Usage"):
    input_data = np.array([[temp, hum, hour]])
    pred = model.predict(input_data)[0]
    st.success(f"Predicted Energy Usage: {pred:.2f} kWh")
    st.info(energy_tip(pred))

    # Small usage graph: predict for every hour of the day with same temp & hum
    hours = np.arange(0,24)
    X = np.column_stack([np.full_like(hours, temp), np.full_like(hours, hum), hours])
    preds = model.predict(X)

    st.write("### 📊 Predicted Energy Usage for 24 hours (same temp & humidity)")
    st.line_chart(pd.DataFrame({'Hour': hours, 'Predicted_kWh': preds}).set_index('Hour'))
    st.write("### 🔎 Summary Report")
    st.write("- Input Temperature (°C):", temp)
    st.write("- Input Humidity (%):", hum)
    st.write("- Predicted (selected hour): {:.2f} kWh".format(pred))
    st.write("- Tip:", energy_tip(pred))

st.caption("Made by Omkar Shetake | Smart Home Energy Predictor - AI Mini Project")
