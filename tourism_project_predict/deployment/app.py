import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# --- 1. SET UP PAGE ---
st.set_page_config(page_title="Wellness Package Predictor", layout="centered")
st.title("🚢 Visit with Us: Wellness Package Predictor")
st.markdown("Predict if a customer is likely to purchase the new Wellness Tourism Package.")

# --- 2. LOAD MODEL FROM HUGGING FACE ---
@st.cache_resource
def load_model():
    # MODEL_REPO_ID from previous step
    REPO_ID = "Tamilvelan/tourism-wellness-model"
    model_path = hf_hub_download(repo_id=REPO_ID, filename="model.joblib")
    return joblib.load(model_path)

model = load_model()

# --- 3. USER INPUTS ---
st.sidebar.header("Customer Demographics")
age = st.sidebar.slider("Age", 18, 80, 30)
income = st.sidebar.number_input("Monthly Income", 0, 100000, 25000)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced"])
occupation = st.sidebar.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])

st.sidebar.header("Interaction Data")
pitch_duration = st.sidebar.slider("Duration of Pitch", 5, 120, 15)
followups = st.sidebar.slider("Number of Follow-ups", 1, 10, 3)
passport = st.sidebar.selectbox("Has Passport?", [0, 1])

# --- 4. PREDICTION LOGIC ---
# Create a dataframe from inputs (Must match training features)
input_data = pd.DataFrame({
    'Age': [age],
    'MonthlyIncome': [income],
    'DurationOfPitch': [pitch_duration],
    'NumberOfFollowups': [followups],
    'Passport': [passport]
})



if st.button("Predict Purchase Likelihood"):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    if prediction[0] == 1:
        st.success(f"High Potential! This customer is likely to purchase. (Confidence: {probability:.2%})")
    else:
        st.warning(f"Low Potential. Customer unlikely to purchase. (Confidence: {1-probability:.2%})")
