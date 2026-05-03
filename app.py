import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import warnings
import joblib
import os

warnings.filterwarnings("ignore")

# ─── Page Config ────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Readmission Predictor", page_icon="🏥", layout="wide"
)


# ─── Load & Train Model ─────────────────────────────────
@st.cache_resource
def load_model():
    model_path = "xgb_model.pkl"

    if os.path.exists(model_path):
        model = joblib.load(model_path)
        df = pd.read_csv("diabetic_ml_ready.csv")
        feature_names = df.drop(columns=["high_risk"]).columns.tolist()
        return model, feature_names

    df = pd.read_csv("diabetic_ml_ready.csv")
    X = df.drop(columns=["high_risk"])
    y = df["high_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.01,
        max_depth=6,
        subsample=0.6,
        colsample_bytree=0.6,
        scale_pos_weight=8,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    )
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    print("Model trained and saved!")
    return model, X.columns.tolist()


model, feature_names = load_model()

# ─── Header ─────────────────────────────────────────────
st.title("🏥 Diabetes Hospital Readmission Predictor")
st.markdown(
    "Predict whether a diabetic patient is at **high risk** of being readmitted within 30 days."
)
st.markdown("---")

# ─── Sidebar Inputs ─────────────────────────────────────
st.sidebar.header("👤 Patient Information")
st.sidebar.markdown("Enter patient details:")

age = st.sidebar.slider("Age (midpoint)", 5, 95, 65, step=10)
time_in_hospital = st.sidebar.slider("Days in Hospital", 1, 14, 4)
num_lab_procedures = st.sidebar.slider("Number of Lab Procedures", 1, 132, 40)
num_medications = st.sidebar.slider("Number of Medications", 1, 81, 15)
num_procedures = st.sidebar.slider("Number of Procedures", 0, 6, 1)
number_diagnoses = st.sidebar.slider("Number of Diagnoses", 1, 16, 7)
number_inpatient = st.sidebar.slider("Prior Inpatient Visits (past year)", 0, 21, 0)
number_emergency = st.sidebar.slider("Prior Emergency Visits (past year)", 0, 76, 0)
number_outpatient = st.sidebar.slider("Prior Outpatient Visits (past year)", 0, 42, 0)

st.sidebar.markdown("---")
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
diabetesMed = st.sidebar.selectbox("On Diabetes Medication?", ["Yes", "No"])
change = st.sidebar.selectbox("Medication Changed This Visit?", ["Yes", "No"])
insulin = st.sidebar.selectbox("Insulin Given?", ["Yes", "No"])

# ─── Build input vector ─────────────────────────────────
input_df = pd.DataFrame([np.zeros(len(feature_names))], columns=feature_names)

input_df["age"] = age
input_df["time_in_hospital"] = time_in_hospital
input_df["num_lab_procedures"] = num_lab_procedures
input_df["num_medications"] = num_medications
input_df["num_procedures"] = num_procedures
input_df["number_diagnoses"] = number_diagnoses
input_df["number_inpatient"] = number_inpatient
input_df["number_emergency"] = number_emergency
input_df["number_outpatient"] = number_outpatient
input_df["gender"] = 1 if gender == "Male" else 0
input_df["diabetesMed"] = 1 if diabetesMed == "Yes" else 0
input_df["change"] = 1 if change == "Yes" else 0
input_df["insulin"] = 1 if insulin == "Yes" else 0

# ─── Prediction ─────────────────────────────────────────
prediction = model.predict(input_df)[0]
probability = model.predict_proba(input_df)[0][1]

# ─── Results ────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    if prediction == 1:
        st.error("🔴 HIGH RISK")
        st.markdown("Patient is predicted to be **readmitted within 30 days.**")
    else:
        st.success("🟢 LOW RISK")
        st.markdown("Patient is predicted **NOT** to be readmitted within 30 days.")

with col2:
    st.metric("Readmission Risk", f"{probability*100:.1f}%")
    st.progress(float(probability))

with col3:
    st.metric("Most Important Factor", "Prior Inpatient Visits")
    st.metric("Patient Value", f"{number_inpatient} visits")

st.markdown("---")

# ─── Risk Factors ───────────────────────────────────────
st.subheader("⚠️ Key Risk Factors for This Patient")

col1, col2, col3 = st.columns(3)

with col1:
    inpatient_risk = (
        "🔴 High"
        if number_inpatient > 2
        else "🟡 Medium" if number_inpatient > 0 else "🟢 Low"
    )
    st.metric("Prior Hospitalizations", f"{number_inpatient}", delta=inpatient_risk)

with col2:
    emergency_risk = (
        "🔴 High"
        if number_emergency > 2
        else "🟡 Medium" if number_emergency > 0 else "🟢 Low"
    )
    st.metric("Emergency Visits", f"{number_emergency}", delta=emergency_risk)

with col3:
    age_risk = "🔴 High" if age >= 75 else "🟡 Medium" if age >= 55 else "🟢 Low"
    st.metric("Age Risk", f"{age} yrs", delta=age_risk)

st.markdown("---")

# ─── Model Info ─────────────────────────────────────────
st.subheader("📊 Model Performance")

perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
perf_col1.metric("ROC-AUC", "0.689")
perf_col2.metric("Recall", "61.4%")
perf_col3.metric("F1 Score", "0.281")
perf_col4.metric("Training Data", "101,766 patients")

st.markdown("---")
st.markdown("""
**⚠️ Medical Disclaimer:** This tool is for educational purposes only and should not 
be used for actual clinical decisions. Always consult qualified medical professionals.

**Data Source:** UCI ML Repository — Diabetes 130-US Hospitals (1999-2008) | 
**Model:** XGBoost with RandomizedSearchCV HPT | ROC-AUC: 0.689
""")
