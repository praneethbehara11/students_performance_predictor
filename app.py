import streamlit as st
import pickle

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

st.title("🎓 Student Performance Predictor")

hours = st.slider("Hours Studied", 0, 10)
attendance = st.slider("Attendance (%)", 0, 100)
internal = st.slider("Internal Score", 0, 20)

if st.button("Predict"):
    result = model.predict([[hours, attendance, internal]])
    if result[0] == 1:
        st.success("✅ Prediction: Pass 🎉")
    else:
        st.error("❌ Prediction: Fail 😟")
