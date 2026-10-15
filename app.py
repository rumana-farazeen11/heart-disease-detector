import streamlit as st
import pandas as pd
import joblib
st.set_page_config(page_title="Heart Disease Risk Detector", page_icon="❤️")

@st.cache_resource
def load_model():
    return joblib.load("heart_model.joblib")

model = load_model()

st.title("❤️ Heart Disease Risk Detector")
st.write("Enter a patient's measurements below to get an estimated risk, "
         "based on a model trained on the Cleveland Heart Disease dataset.")
st.warning("Educational demo only. This is NOT a medical diagnosis. "
           "Talk to a doctor about any health concerns.")

col1, col2 = st.columns(2)

with col1: 
    age = st.slider("Age", 20, 90, 55)
    sex_label = st.selectbox("Sex", ["Female", "Male"])
    cp_label = st.selectbox("Chest pain type", [
        "Typical angina", "Atypical angina",
        "Non-anginal pain", "Asymptomatic"])
    trestbps = st.slider("Resting blood pressure (mm Hg)", 80, 200, 130)
    chol = st.slider("Cholestrol (mg/dl)", 100, 600, 240)
    fbs_label = st.selectbox("Fasting blood sugar above 120 mg/dl?", ["No", "Yes"])
    restecg_label = st.selectbox("Resting ECG result", [
        "Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"])

with col2:
    thalach = st.slider("Maximum heart rate achieved", 60, 220, 150)
    exang_label = st.selectbox("Exercise-induced chest pain?", ["No", "Yes"])
    oldpeak = st.slider("ST depression (oldpeak)", 0.0, 7.0, 1.0, step=0.1)
    slope_label = st.selectbox("Slope of peak exercise ST segment", [
        "Upsloping", "Flat", "Downsloping"])
    ca = st.selectbox("Major vessels colored by fluroscopy (0-3)", [0, 1, 2, 3])
    thal_label = st.selectbox("Thalassemia", [
        "Normal", "Fixed defect", "Reversible defect"])

sex = {"Female": 0, "Male": 1}[sex_label]
cp = {"Typical angina": 1, "Atypical angina": 2,
      "Non-anginal pain": 3, "Asymptomatic": 4}[cp_label]
fbs = {"No": 0, "Yes": 1}[fbs_label]
restecg = {"Normal": 0, "ST-T wave abnormality": 1,
           "Left ventricular hypertrophy": 2}[restecg_label]
exang = {"No": 0, "Yes": 1}[exang_label]
slope = {"Upsloping": 1, "Flat": 2, "Downsloping": 3}[slope_label]
thal = {"Normal": 3, "Fixed defect": 6, "Reversible defect": 7}[thal_label]

if st.button("Check risk"):
    patient = pd.DataFrame([{
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
        "chol": chol, "fbs": fbs, "restecg": restecg, 
        "thalach": thalach, "exang": exang, "oldpeak": oldpeak, 
        "slope": slope, "ca": ca, "thal":thal
    }])

    probability = model.predict_proba(patient)[0][1]

    st.metric("Estimated probability of heart disease", f"{probability:.0%}")

    if probability >= 0.5:
        st.error("The model estimates a HIGHER risk of heart disease.")
    else:
        st.success("The model estimates a LOWER risk of heart disease.")

    st.caption("This estimate comes from a small research dataset (303 patients)"
               "and is not medical advice.")