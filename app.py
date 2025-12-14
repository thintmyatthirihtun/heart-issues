import streamlit as st
import pandas as pd
import joblib

# -------------------------
# Load trained model and features
# -------------------------
model = joblib.load("heart_disease_model.pkl")
features = joblib.load("model_features.pkl")

# -------------------------
# Page setup
# -------------------------
st.set_page_config(page_title="Heart Disease Prediction", layout="centered")
st.title("🫀 Heart Disease Prediction App")
st.write("Predict the likelihood of heart disease based on important medical features.")

# -------------------------
# User Inputs
# -------------------------
def get_user_inputs():
    age = st.slider("Age", 20, 100, 50, help="Age of the patient in years")
    cholesterol = st.slider("Cholesterol", 100, 400, 200, help="Blood cholesterol level (mg/dL)")
    max_hr = st.slider("Max Heart Rate", 60, 220, 150, help="Maximum heart rate achieved during exercise")
    st_depression = st.slider(
        "ST Depression", 0.0, 6.0, 1.0, step=0.1,
        help="ST depression induced by exercise relative to rest, measured in mm. Indicates stress on the heart."
    )

    chest_pain = st.selectbox(
        "Chest Pain Type",
        {
            "Typical Angina": 0,
            "Atypical Angina": 1,
            "Non-anginal Pain": 2,
            "Asymptomatic": 3
        }.items(),
        help="Type of chest pain experienced by the patient."
    )[1]

    thallium = st.select_slider(
        "Thallium Test Result",
        options=[0, 1, 2, 3],
        value=0,
        help="Result of the Thallium stress test. Indicates areas of poor blood flow in the heart."
    )

    vessels = st.select_slider(
        "Number of Vessels (Fluoroscopy)",
        options=[0, 1, 2, 3],
        value=0,
        help="Number of major heart blood vessels narrowed or blocked, determined via imaging."
    )

    return age, cholesterol, max_hr, st_depression, chest_pain, thallium, vessels

# -------------------------
# Build input dataframe
# -------------------------
def build_input_df(age, cholesterol, max_hr, st_depression, chest_pain, thallium, vessels):
    # Default features
    bp_default = 120
    sex_default = 1
    slope_default = 1
    exercise_angina_default = 0
    ekg_default = 1
    fbs_default = 0

    return pd.DataFrame([[
        age,
        sex_default,
        chest_pain,
        bp_default,
        cholesterol,
        fbs_default,
        ekg_default,
        max_hr,
        exercise_angina_default,
        st_depression,
        slope_default,
        vessels,
        thallium
    ]], columns=features)

# -------------------------
# Prediction function
# -------------------------
def predict_risk(input_df):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == "Presence":
        st.error(f"⚠️ High risk of heart disease\nProbability: {probability:.2%}")
    else:
        st.success(f"✅ Low risk of heart disease\nProbability: {probability:.2%}")

# -------------------------
# Main
# -------------------------
age, cholesterol, max_hr, st_depression, chest_pain, thallium, vessels = get_user_inputs()
input_df = build_input_df(age, cholesterol, max_hr, st_depression, chest_pain, thallium, vessels)

with st.expander("What do these features mean?"):
    st.markdown("""
    **ST Depression:** A measure of the deviation of the ST segment on an ECG during stress test. High values indicate possible heart problems.

    **Thallium Test:** A nuclear imaging test showing blood flow to the heart. Values indicate areas with reduced blood flow.

    **Number of Vessels (Fluoroscopy):** The count of major coronary arteries with significant blockage.

    **Chest Pain Type:**  
    - Typical Angina: Chest pain due to heart ischemia.  
    - Atypical Angina: Chest pain not following the typical pattern.  
    - Non-anginal Pain: Chest pain from other causes.  
    - Asymptomatic: No chest pain.
    """)

if st.button("Predict"):
    predict_risk(input_df)