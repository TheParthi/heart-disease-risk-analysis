import streamlit as st
import pandas as pd
import joblib
from database import create_db, insert_data, fetch_data

# ===================== AUTH CONFIG =====================
USERS = {
    "admin": {"password": "admin123", "role": "admin"},
    "doctor": {"password": "doctor123", "role": "doctor"},
    "user": {"password": "user123", "role": "user"}
}

# ===================== SESSION STATE =====================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.role = None
    st.session_state.username = None

# ===================== LOGIN PAGE =====================
if not st.session_state.logged_in:
    st.set_page_config(page_title="Login | Heart AI", page_icon="❤️")

    st.title("🔐 Login – Heart Disease Risk Analysis")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login", use_container_width=True):
        if username in USERS and USERS[username]["password"] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.role = USERS[username]["role"]
            st.success("Login successful")
            st.rerun()
        else:
            st.error("Invalid username or password")

    st.caption("Demo accounts: admin / doctor / user")
    st.stop()

# ===================== INITIAL SETUP =====================
create_db()
model, scaler = joblib.load("heart_models.pkl")

st.set_page_config(
    page_title="Heart Disease Risk Analysis",
    page_icon="❤️",
    layout="centered"
)

# ===================== STYLING =====================
st.markdown(
    """
    <style>
    .main {background-color: #0E1117;}
    h1, h2, h3, h4 {color: #FAFAFA;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

# ===================== SIDEBAR =====================
with st.sidebar:
    st.title("❤️ Heart AI")
    st.markdown(f"**User:** {st.session_state.username}")
    st.markdown(f"**Role:** {st.session_state.role}")
    st.divider()

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.role = None
        st.session_state.username = None
        st.rerun()

# ===================== MAIN HEADER =====================
st.title("Heart Disease Risk Analysis")
st.caption("AI-powered clinical risk assessment system")
st.divider()

# ===================== INPUT FORM =====================
with st.form("heart_form"):
    st.subheader("👤 Patient Information")
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age (years)", 20, 80, 35)
        sex_label = st.radio("Sex", ["Male", "Female"], horizontal=True)
        sex = 1 if sex_label == "Male" else 0

        cp_label = st.selectbox(
            "Chest Pain Type",
            ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
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
    st.subheader("🧪 Clinical Measurements")

    col3, col4 = st.columns(2)

    with col3:
        fbs = 1 if st.radio("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"], horizontal=True) == "Yes" else 0
        restecg = st.selectbox("Resting ECG", [0, 1, 2])
        exang = 1 if st.radio("Exercise Induced Angina", ["No", "Yes"], horizontal=True) == "Yes" else 0

    with col4:
        oldpeak = st.slider("ST Depression", 0.0, 6.0, 0.0, 0.1)
        slope = st.selectbox("Slope", [0, 1, 2])
        ca = st.selectbox("Major Vessels", [0, 1, 2, 3, 4])
        thal = st.selectbox("Thalassemia", [1, 2, 3])

    submit = st.form_submit_button("🔍 Predict Risk", use_container_width=True)

# ===================== PREDICTION =====================
if submit:
    input_df = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs,
                              restecg, thalach, exang, oldpeak,
                              slope, ca, thal]],
                            columns=[
                                'age','sex','cp','trestbps','chol','fbs',
                                'restecg','thalach','exang','oldpeak',
                                'slope','ca','thal'
                            ])

    prediction = model.predict(scaler.transform(input_df))
    result = "High Risk" if prediction[0] == 0 else "Low Risk"

    insert_data(age, sex_label, trestbps, chol, result)

    st.divider()
    st.subheader("📌 Prediction Result")

    if result == "High Risk":
        st.error("🚨 HIGH RISK OF HEART DISEASE")
    else:
        st.success("✅ LOW RISK OF HEART DISEASE")

# ===================== ADMIN / DOCTOR VIEW =====================
if st.session_state.role in ["admin", "doctor"]:
    st.divider()
    st.subheader("📊 Risk Analytics Dashboard")

    history = fetch_data()

    if history:
        df = pd.DataFrame(
            history,
            columns=["ID", "Age", "Sex", "BP", "Cholesterol", "Result", "Timestamp"]
        )

        # ---- Charts ----
        col_c1, col_c2 = st.columns(2)

        with col_c1:
            st.markdown("**Risk Distribution**")
            st.bar_chart(df["Result"].value_counts())

        with col_c2:
            st.markdown("**Age vs Risk**")
            st.line_chart(df.groupby("Age")["Result"].count())

        st.divider()
        st.markdown("**Prediction History**")
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Report (CSV)",
            csv,
            "heart_risk_history.csv",
            "text/csv"
        )
    else:
        st.info("No data available for analysis.")

# ===================== FOOTER =====================
st.divider()
st.caption("© 2026 Heart AI | ML-based Clinical Decision Support System")
