import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from model import MultimodalICUModel
from captum.attr import IntegratedGradients

# -----------------------
# CONFIG
# -----------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SEQ_LEN = 72

TEMPORAL_FEATURES = [
    "heart_rate", "respiratory_rate", "spo2_pct", "temperature_c",
    "systolic_bp", "diastolic_bp", "oxygen_device", "oxygen_flow",
    "mobility_score", "nurse_alert", "wbc_count", "lactate",
    "creatinine", "crp_level", "hemoglobin", "sepsis_risk_score"
]

STATIC_FEATURES = [
    "age", "gender", "comorbidity_index",
    "admission_type", "baseline_risk_score", "los_hours"
]

# -----------------------
# DATA LOAD
# -----------------------
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

@st.cache_data
def load_data():
    csv_path = BASE_DIR / "Dataset" / "processed_data" / "test.csv"
    st.write("Looking for:", csv_path)
    st.write("Exists:", csv_path.exists())
    return pd.read_csv(csv_path)

df = load_data()

# -----------------------
# MODEL LOAD
# -----------------------
@st.cache_resource
def load_model():
    model = MultimodalICUModel().to(DEVICE)
    model.load_state_dict(torch.load("models/best_model.pth", map_location=DEVICE))
    model.eval()
    return model

model = load_model()

# -----------------------
# CAPTUM WRAPPER (IMPORTANT FIX)
# -----------------------
class CaptumWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, temporal, static):
        seq_len = torch.full(
            (temporal.size(0),),
            temporal.size(1),
            dtype=torch.long,
            device=temporal.device
        )
        out, _ = self.model(temporal, static, seq_len)
        return out

wrapped_model = CaptumWrapper(model)
ig = IntegratedGradients(wrapped_model)

# -----------------------
# SIDEBAR: RISK RANKING
# -----------------------
st.sidebar.title("🔥 ICU Risk Ranking")

risk_list = []

for pid in df.patient_id.unique():
    p = df[df.patient_id == pid].sort_values("hour_from_admission")

    if len(p) < 2:
        continue

    temporal = p[TEMPORAL_FEATURES].values.astype(np.float32)
    static = p.iloc[0][STATIC_FEATURES].values.astype(np.float32)

    seq_len = len(temporal)

    if seq_len < MAX_SEQ_LEN:
        pad = np.zeros((MAX_SEQ_LEN - seq_len, temporal.shape[1]), dtype=np.float32)
        temporal_padded = np.vstack([temporal, pad])
    else:
        temporal_padded = temporal[:MAX_SEQ_LEN]
        seq_len = MAX_SEQ_LEN

    x = torch.tensor(temporal_padded, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    s = torch.tensor(static, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    l = torch.tensor([seq_len], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        out, _ = model(x, s, l)
        prob = torch.sigmoid(out).item()

    risk_list.append((pid, prob))

risk_list = sorted(risk_list, key=lambda x: x[1], reverse=True)

st.sidebar.subheader("Top High Risk Patients")
st.sidebar.dataframe(pd.DataFrame(risk_list[:10], columns=["Patient", "Risk"]))

# -----------------------
# MAIN UI
# -----------------------
st.title("🧠 ICU Risk Dashboard")

pid = st.selectbox("Select Patient ID", df.patient_id.unique())

patient = df[df.patient_id == pid].sort_values("hour_from_admission")

st.write(f"Hours available: {len(patient)}")

# -----------------------
# BUILD SEQUENCE
# -----------------------
temporal = patient[TEMPORAL_FEATURES].values.astype(np.float32)
static = patient.iloc[0][STATIC_FEATURES].values.astype(np.float32)

seq_len = len(temporal)

if seq_len < MAX_SEQ_LEN:
    pad = np.zeros((MAX_SEQ_LEN - seq_len, temporal.shape[1]), dtype=np.float32)
    temporal_padded = np.vstack([temporal, pad])
else:
    temporal_padded = temporal[:MAX_SEQ_LEN]
    seq_len = MAX_SEQ_LEN

# -----------------------
# PREDICT FUNCTION
# -----------------------
def predict(x, s, l):
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    s = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    l = torch.tensor([l], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        out, attn = model(x, s, l)
        prob = torch.sigmoid(out).item()

    return prob, attn

# -----------------------
# RUN MODEL
# -----------------------
if st.button("Run ICU Analysis"):

    prob, attn = predict(temporal_padded, static, seq_len)

    # -----------------------
    # RISK CATEGORY
    # -----------------------
    if prob < 0.3:
        status = "🟢 LOW RISK"
    elif prob < 0.7:
        status = "🟡 MODERATE RISK"
    else:
        status = "🔴 HIGH RISK"

    st.metric("ICU Risk Score", f"{prob:.4f}")
    st.subheader(status)

    # -----------------------
    # ATTENTION PLOT
    # -----------------------
    st.subheader("🔍 Attention over Time")

    attn = attn.squeeze().cpu().detach().numpy()[:len(patient)]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=np.arange(len(patient)),
        y=patient["heart_rate"],
        name="Heart Rate"
    ))

    fig.add_trace(go.Bar(
        x=np.arange(len(patient)),
        y=attn * 100,
        name="Attention (%)"
    ))

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------
    # CAPTUM EXPLANATION
    # -----------------------
    st.subheader("🧠 Feature Importance (Static)")

    x = torch.tensor(temporal_padded, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    s = torch.tensor(static, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    attr = ig.attribute((x, s))
    static_attr = attr[1].squeeze().cpu().detach().numpy()

    importance = pd.DataFrame({
        "Feature": STATIC_FEATURES,
        "Importance": static_attr
    }).sort_values("Importance", ascending=False)

    st.dataframe(importance)

    # -----------------------
    # SUMMARY
    # -----------------------
    st.subheader("📋 Clinical Summary")

    st.write(f"""
    - Patient ID: {pid}
    - ICU Risk: {prob:.4f}
    - Status: {status}
    - Hours observed: {len(patient)}
    - Top factor: {importance.iloc[0]['Feature']}
    """)
