import streamlit as st
import pandas as pd
import joblib

# 👉 NEW: database imports
from database import create_db, insert_data, fetch_data

# 👉 NEW: initialize database (run once)
create_db()

# Load model and scaler
model, scaler = joblib.load("heart_models.pkl")

st.set_page_config(
    page_title="Heart Disease Risk Analysis",
    page_icon="❤️",
    layout="centered"
)

st.markdown(
    """
    <style>
    .main {background-color: #0E1117;}
    h1, h2, h3 {color: #FAFAFA;}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("❤️ Heart Disease Risk Analysis System")
st.caption("Enter patient medical details to assess heart disease risk")

st.divider()

# ===================== PATIENT DETAILS =====================
st.subheader("👤 Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age (years)", 20, 80, 35)
    sex_label = st.radio("Sex", ["Male", "Female"])
    sex = 1 if sex_label == "Male" else 0

    cp_label = st.selectbox(
        "Chest Pain Type",
        [
            "Typical Angina",
            "Atypical Angina",
            "Non-anginal Pain",
            "Asymptomatic"
        ]
    )
    cp = {
        "Typical Angina": 0,
        "Atypical Angina": 1,
        "Non-anginal Pain": 2,
        "Asymptomatic": 3
    }[cp_label]

with col2:
    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, 110)
    chol = st.slider("Serum Cholesterol (mg/dl)", 120, 350, 180)
    thalach = st.slider("Maximum Heart Rate Achieved", 70, 210, 170)

st.divider()

# ===================== CLINICAL DETAILS =====================
st.subheader("🧪 Clinical Measurements")

col3, col4 = st.columns(2)

with col3:
    fbs_label = st.radio("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
    fbs = 1 if fbs_label == "Yes" else 0

    restecg_label = st.selectbox(
        "Resting ECG",
        ["Normal", "ST-T abnormality", "Left Ventricular Hypertrophy"]
    )
    restecg = {
        "Normal": 0,
        "ST-T abnormality": 1,
        "Left Ventricular Hypertrophy": 2
    }[restecg_label]

    exang_label = st.radio("Exercise Induced Angina", ["No", "Yes"])
    exang = 1 if exang_label == "Yes" else 0

with col4:
    oldpeak = st.slider("ST Depression", 0.0, 6.0, 0.0, 0.1)
    slope = st.selectbox("Slope of ST Segment", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels (0–4)", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia", [1, 2, 3])

st.divider()

# ===================== PREDICTION =====================
if st.button("🔍 Predict Heart Disease Risk", use_container_width=True):

    input_data = pd.DataFrame(
        [[age, sex, cp, trestbps, chol, fbs,
          restecg, thalach, exang, oldpeak,
          slope, ca, thal]],
        columns=[
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
            'restecg', 'thalach', 'exang', 'oldpeak',
            'slope', 'ca', 'thal'
        ]
    )

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    st.divider()

    # 👉 result text
    result_text = "High Risk" if prediction[0] == 0 else "Low Risk"

    # 👉 NEW: store prediction in database
    insert_data(age, sex_label, trestbps, chol, result_text)

    if prediction[0] == 0:
        st.error("🚨 **High Risk of Heart Disease**")
    else:
        st.success("✅ **Low Risk of Heart Disease**")

    st.caption("⚠️ This tool is for educational purposes only, not a medical diagnosis.")

# ===================== HISTORY =====================
st.divider()
st.subheader("📊 Prediction History")

history = fetch_data()

if history:
    history_df = pd.DataFrame(
        history,
        columns=["ID", "Age", "Sex", "BP", "Cholesterol", "Result", "Timestamp"]
    )
    st.dataframe(history_df, use_container_width=True)
else:
    st.info("No prediction history available yet.")
