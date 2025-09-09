import streamlit as st
import pandas as pd
import joblib

# --- Custom CSS for white-themed cards ---
st.markdown(
    """
    <style>
    /* App background */
    .stApp {
        background-color: #f5f5f5;
        color: black;
    }

    /* Card style for sections (title, inputs, tables) */
    .card {
        background-color: white;
        padding: 2rem;
        margin: 1rem 0;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }

    /* Input boxes and number inputs */
    div[data-baseweb="input"], .stNumberInput > div > div > input {
        background-color: #fafafa !important;
        color: black !important;
        border-radius: 5px !important;
        border: 1px solid #ddd !important;
        padding: 0.4rem !important;
    }

    /* Buttons */
    div.stButton > button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 0.6rem 1.2rem;
        border-radius: 0.5rem;
        font-weight: bold;
    }
    div.stButton > button:hover {
        background-color: #45a049;
    }

    /* Dataframe table */
    .stDataFrame div[data-testid="stDataFrame"] {
        background-color: white;
        color: black;
        border-radius: 10px;
        padding: 0.5rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }

    /* Footer text */
    p {
        color: gray;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load model
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# Title card
st.markdown('<div class="card"><h1 style="text-align:center;">🎓 Student Performance Predictor</h1></div>', unsafe_allow_html=True)

# Initialize history in session state
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(
        columns=['Name', 'Hours_Studied', 'Attendance', 'Internal_Score', 'Prediction']
    )

# Input card
st.markdown('<div class="card">', unsafe_allow_html=True)
student_name = st.text_input("Student Name")
hours_studied = st.number_input("Hours Studied", min_value=0, max_value=24, step=1, value=0)
attendance = st.number_input("Attendance (%)", min_value=0, max_value=100, step=1, value=0)
internal_score = st.number_input("Internal Score", min_value=0, max_value=100, step=1, value=0)

if st.button("Predict Performance"):
    if student_name.strip() == "":
        st.warning("⚠️ Please enter the student's name.")
    else:
        features = pd.DataFrame([[hours_studied, attendance, internal_score]],
                                columns=['Hours_Studied', 'Attendance', 'Internal_Score'])
        prediction = model.predict(features)[0]

        # Save to history with name
        st.session_state.history = pd.concat([
            st.session_state.history,
            pd.DataFrame([[student_name, hours_studied, attendance, internal_score,
                           "Pass" if prediction == 1 else "Fail"]],
                         columns=['Name', 'Hours_Studied', 'Attendance', 'Internal_Score', 'Prediction'])
        ], ignore_index=True)

        # Show result in a card-like box
        if prediction == 1:
            st.success(f"✅ {student_name} is likely to Pass!")
        else:
            st.error(f"❌ {student_name} is likely to Fail.")
st.markdown('</div>', unsafe_allow_html=True)

# History table card
st.markdown('<div class="card">', unsafe_allow_html=True)
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
st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown(
    """
    <hr>
    <p style="text-align: center;">
    Designed and Developed by <b>Praneeth Behara</b>
    </p>
    """,
    unsafe_allow_html=True
)
