# 🏥 ICU Early Warning System
### AI-Powered ICU Deterioration Prediction using Machine Learning & Deep Learning


---

# 🚀 Live Demonstrations

## 🧠 Deep Learning Dashboard

🔗 https://icu-dl-demo.streamlit.app/

---

## 🤖 Machine Learning Dashboard

🔗 https://icu-ml-demo.streamlit.app/

---

# 📌 Overview

Early prediction of clinical deterioration is one of the most important challenges inside an Intensive Care Unit (ICU). Delayed intervention can significantly increase mortality risk.

This project presents **two independent AI-based solutions** for predicting ICU deterioration within the next **12 hours**:

- Machine Learning Pipeline
- Deep Learning Pipeline

Both models are deployed as interactive Streamlit applications to demonstrate real-time clinical risk prediction.

---

# ✨ Features

✔ ICU deterioration prediction

✔ Machine Learning implementation

✔ Deep Learning implementation

✔ Interactive Streamlit dashboards

✔ Explainable AI (Attention + Integrated Gradients)

✔ Real-time patient risk estimation

✔ Attention visualization

✔ Clinical summary generation

---

# 📂 Repository Structure

```text
ICU-Early-Warning-System
│
├── Machine_Learning
│   ├── app.py
│   ├── Code.ipynb
│   ├── models
│   ├── Dataset
│   └── requirements.txt
│
├── Deep_Learning
│   ├── app.py
│   ├── model.py
│   ├── Code.ipynb
│   ├── models
│   ├── Dataset
│   └── requirements.txt
│
└── README.md
```

---

# 📊 Dataset

Hospital Deterioration Dataset

- 417,866 observations
- 10,000 ICU patients
- 28 clinical variables
- Sequence length:
  - 12–72 hours

Target:

```
deterioration_next_12h
```

---

# 🤖 Machine Learning Pipeline

The classical machine learning workflow consists of

- Data preprocessing
- Missing value handling
- Feature scaling
- Label encoding
- Feature engineering
- Model training
- Risk prediction
- Streamlit deployment

The trained model is serialized using Joblib and deployed for inference.

---

# 🧠 Deep Learning Pipeline

The deep learning pipeline uses a multimodal Attention-BiLSTM architecture.

## Temporal branch

- Hourly ICU vital signs
- Laboratory measurements
- BiLSTM encoder
- Attention mechanism

## Static branch

- Demographics
- Admission information
- Baseline patient characteristics

## Fusion

Temporal embedding

+

Static embedding

↓

Fully Connected Network

↓

Probability of ICU deterioration

---

# 🔍 Explainable AI

Unlike traditional black-box models, the deep learning pipeline provides explainability through

- Attention Heatmaps
- Captum Integrated Gradients

This enables clinicians to understand

- which ICU hours influenced prediction
- which patient features contributed most

---

# 💻 Streamlit Applications

Both implementations are available as interactive dashboards.

## Machine Learning

Features

- Risk prediction
- Patient selection
- Clinical summary

---

## Deep Learning

Features

- Risk prediction
- Attention visualization
- Integrated Gradients
- ICU risk ranking
- Clinical interpretation

---

# 🛠 Technologies Used

## Languages

- Python

## Machine Learning

- Scikit-learn

## Deep Learning

- PyTorch

## Explainability

- Captum
- Attention Mechanism

## Visualization

- Plotly
- Matplotlib

## Deployment

- Streamlit

---

# 📈 Workflow

```
Raw ICU Dataset
        │
        ▼
Data Cleaning
        │
        ▼
Feature Engineering
        │
        ▼
Train/Test Split
        │
        ├───────────────┐
        ▼               ▼
Machine Learning    Deep Learning
        │               │
        ▼               ▼
Prediction        Attention + XAI
        │               │
        └──────┬────────┘
               ▼
      Streamlit Dashboard
```

---

# 📸 Screenshots

## Machine Learning Dashboard

(Add Screenshot)

---

## Deep Learning Dashboard

(Add Screenshot)

---

# ⚙ Installation

Clone the repository

```bash
git clone https://github.com/BhuvanMohan2005/ICU-Early-Warning-System.git
```

---

## Machine Learning

```bash
cd Machine_Learning

pip install -r requirements.txt

streamlit run app.py
```

---

## Deep Learning

```bash
cd Deep_Learning

pip install -r requirements.txt

streamlit run app.py
```

---

# 🔮 Future Improvements

- Multi-hospital validation
- Transformer-based temporal models
- FHIR integration
- Real-time EHR connectivity
- Docker deployment
- Cloud inference APIs

---

# 👨‍💻 Author

**Gunupati Venkata Bhuvana Mohan**

Machine Learning • Deep Learning • Computer Vision

GitHub

https://github.com/BhuvanMohan2005

LinkedIn

https://www.linkedin.com/in/gunupatibhuvan

---

⭐ If you found this repository useful, consider giving it a star!
