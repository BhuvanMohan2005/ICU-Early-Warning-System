import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page Config
# -----------------------------

st.set_page_config(
    page_title="ICU Patient Deterioration Prediction",
    page_icon="🏥",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
from pathlib import Path

BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "models"/"best_model.pth"
model = joblib.load(MODEL_PATH)
SCALER_PATH = BASE_DIR / "models"/"scaler.pkl"
scaler = joblib.load(SCALER_PATH)
FEATURES_PATH = BASE_DIR / "models"/"features.pkl"
features = joblib.load(FEATURES_PATH)

# -----------------------------
# Header
# -----------------------------

st.title("🏥 ICU Patient Deterioration Prediction System")

st.markdown(
"""
Early warning system that predicts whether an ICU patient
is likely to deteriorate within the next **12 hours** using
Machine Learning (XGBoost).
"""
)

st.divider()

# -----------------------------
# Layout
# -----------------------------

col1, col2, col3 = st.columns(3)

inputs = {}

# -----------------------------
# Column 1
# -----------------------------

with col1:

    st.subheader("👤 Patient Information")

    inputs["age"] = st.number_input(
        "Age",
        18,
        100,
        60
    )

    gender = st.selectbox(
        "Gender",
        ["Female","Male"]
    )

    admission = st.selectbox(
        "Admission Type",
        ["ED","Elective","Transfer"]
    )

    inputs["hour_from_admission"] = st.slider(
        "Hours Since Admission",
        0,
        72,
        12
    )

# -----------------------------
# Column 2
# -----------------------------

with col2:

    st.subheader("❤️ Vital Signs")

    inputs["heart_rate"] = st.number_input(
        "Heart Rate",
        40,
        180,
        90
    )

    inputs["respiratory_rate"] = st.number_input(
        "Respiratory Rate",
        8,
        45,
        20
    )

    inputs["spo2_pct"] = st.slider(
        "SpO₂ (%)",
        70,
        100,
        96
    )

    inputs["temperature_c"] = st.number_input(
        "Temperature (°C)",
        35.0,
        41.0,
        37.0
    )

    inputs["systolic_bp"] = st.number_input(
        "Systolic BP",
        70,
        200,
        120
    )

    inputs["diastolic_bp"] = st.number_input(
        "Diastolic BP",
        40,
        120,
        80
    )

# -----------------------------
# Column 3
# -----------------------------

with col3:

    st.subheader("🧪 Laboratory Values")

    inputs["wbc_count"] = st.number_input(
        "WBC Count",
        value=8.0
    )

    inputs["lactate"] = st.number_input(
        "Lactate",
        value=1.5
    )

    inputs["creatinine"] = st.number_input(
        "Creatinine",
        value=1.0
    )

    inputs["crp_level"] = st.number_input(
        "CRP",
        value=15.0
    )

    inputs["hemoglobin"] = st.number_input(
        "Hemoglobin",
        value=13.0
    )

    inputs["sepsis_risk_score"] = st.slider(
        "Sepsis Risk",
        0.0,
        1.0,
        0.20
    )

st.divider()

# -----------------------------
# Extra Inputs
# -----------------------------

col4, col5, col6 = st.columns(3)

with col4:

    oxygen = st.selectbox(
        "Oxygen Device",
        ["none","nasal","mask","hfnc","niv"]
    )

with col5:

    inputs["oxygen_flow"] = st.number_input(
        "Oxygen Flow",
        value=0.0
    )

    inputs["mobility_score"] = st.slider(
        "Mobility Score",
        0,
        4,
        3
    )

with col6:

    inputs["nurse_alert"] = st.selectbox(
        "Nurse Alert",
        [0,1]
    )

    inputs["comorbidity_index"] = st.slider(
        "Comorbidity Index",
        0,
        8,
        2
    )

# -----------------------------
# Encode Categories
# -----------------------------

inputs["gender_M"] = 1 if gender=="Male" else 0

inputs["oxygen_device_nasal"] = 0
inputs["oxygen_device_mask"] = 0
inputs["oxygen_device_hfnc"] = 0
inputs["oxygen_device_niv"] = 0

if oxygen != "none":
    inputs[f"oxygen_device_{oxygen}"] = 1

inputs["admission_type_Elective"] = 0
inputs["admission_type_Transfer"] = 0

if admission=="Elective":
    inputs["admission_type_Elective"]=1

if admission=="Transfer":
    inputs["admission_type_Transfer"]=1

# -----------------------------
# Predict
# -----------------------------

if st.button("🔍 Predict Deterioration", use_container_width=True):

    sample = pd.DataFrame([inputs])

    sample = sample.reindex(
        columns=features,
        fill_value=0
    )

    sample = scaler.transform(sample)

    prediction = model.predict(sample)[0]

    probability = model.predict_proba(sample)[0][1]

    st.divider()

    st.subheader("Prediction Result")

    st.progress(float(probability))

    if probability < 0.30:

        st.success(f"🟢 LOW RISK ({probability*100:.2f}%)")

        st.info("Continue routine monitoring.")

    elif probability < 0.70:

        st.warning(f"🟠 MODERATE RISK ({probability*100:.2f}%)")

        st.warning("Increase observation frequency.")

    else:

        st.error(f"🔴 HIGH RISK ({probability*100:.2f}%)")

        st.error("Immediate physician review recommended.")

    st.subheader("Patient Summary")

    st.dataframe(pd.DataFrame([inputs]))
