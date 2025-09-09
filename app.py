import streamlit as st
import joblib
import pandas as pd

# Load model
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

st.title("🎓 Student Performance Predictor")

# Inputs
study_hours = st.number_input("Study Hours", min_value=0, max_value=24, step=1)
attendance = st.number_input("Attendance (%)", min_value=0, max_value=100, step=1)

if st.button("Predict"):
    features = [[study_hours, attendance]]
    prediction = model.predict(features)
    st.success(f"Predicted Performance: {prediction[0]}")
