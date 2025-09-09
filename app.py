import streamlit as st
import pandas as pd
import joblib

# Load model
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

st.title("🎓 Student Performance Predictor")

# Initialize history in session state
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=['Hours_Studied', 'Attendance', 'Internal_Score', 'Prediction'])

# Inputs
hours_studied = st.number_input("Hours Studied", min_value=0, max_value=24, step=1, value=0)
attendance = st.number_input("Attendance (%)", min_value=0, max_value=100, step=1, value=0)
internal_score = st.number_input("Internal Score", min_value=0, max_value=100, step=1, value=0)

if st.button("Predict Performance"):
    features = pd.DataFrame([[hours_studied, attendance, internal_score]],
                            columns=['Hours_Studied', 'Attendance', 'Internal_Score'])

    prediction = model.predict(features)[0]

    # Save to history
    st.session_state.history = pd.concat([
        st.session_state.history,
        pd.DataFrame([[hours_studied, attendance, internal_score, "Pass" if prediction == 1 else "Fail"]],
                     columns=['Hours_Studied', 'Attendance', 'Internal_Score', 'Prediction'])
    ], ignore_index=True)

    # Show result
    if prediction == 1:
        st.success("✅ The student is likely to Pass!")
    else:
        st.error("❌ The student is likely to Fail.")

# Show history table
st.subheader("📊 Prediction History")
st.dataframe(st.session_state.history)

# Download button
if not st.session_state.history.empty:
    csv = st.session_state.history.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download History as CSV",
        data=csv,
        file_name="student_performance_history.csv",
        mime="text/csv"
    )

# Footer
st.markdown(
    """
    <hr>
    <p style="text-align: center; color: gray;">
    Designed and Developed by <b>Praneeth Behara</b>
    </p>
    """,
    unsafe_allow_html=True
)
