import streamlit as st
import pandas as pd
import joblib

# Load model
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

st.title("🎓 Student Performance Predictor")

# Take inputs
hours_studied = st.number_input("Hours Studied", min_value=0, max_value=24, step=1)
attendance = st.number_input("Attendance (%)", min_value=0, max_value=100, step=1)
internal_score = st.number_input("Internal Score", min_value=0, max_value=100, step=1)

# Prediction
if st.button("Predict Performance"):
    features = [[hours_studied, attendance, internal_score]]  # Must match training order
    prediction = model.predict(features)

    if prediction[0] == 1:
        st.success("✅ The student is likely to Pass!")
    else:
        st.error("❌ The student is likely to Fail.")
