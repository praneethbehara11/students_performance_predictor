import streamlit as st
import pandas as pd
import joblib

# Load model
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

st.title("🎓 Student Performance Predictor")

# Inputs
hours_studied = st.number_input("Hours Studied", min_value=0, max_value=24, step=1, value=0)
attendance = st.number_input("Attendance (%)", min_value=0, max_value=100, step=1, value=0)
internal_score = st.number_input("Internal Score", min_value=0, max_value=100, step=1, value=0)

if st.button("Predict Performance"):
    # Always ensure correct dataframe shape
    features = pd.DataFrame([[hours_studied, attendance, internal_score]],
                            columns=['Hours_Studied', 'Attendance', 'Internal_Score'])

    st.write("🔍 Features passed to model:", features)  # Debugging helper

    prediction = model.predict(features)

    if prediction[0] == 1:
        st.success("✅ The student is likely to Pass!")
    else:
        st.error("❌ The student is likely to Fail.")
