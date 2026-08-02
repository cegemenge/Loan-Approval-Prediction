import streamlit as st
import pandas as pd
import joblib

# Load model files
model = joblib.load("loan_approval_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

st.set_page_config(page_title="Loan Approval Prediction", page_icon="💰", layout="wide")

st.title("💰 Loan Approval Prediction System")

st.write("Fill in the applicant details below and click **Predict**.")

# Sidebar
st.sidebar.title("About")
st.sidebar.info(
    """
    Machine Learning Project

    Model: Random Forest

    Dataset: Loan Approval Dataset
    """
)

# Create two columns
col1, col2 = st.columns(2)

inputs = []

for i, feature in enumerate(feature_names):
    if i % 2 == 0:
        with col1:
            value = st.number_input(feature, value=0.0)
    else:
        with col2:
            value = st.number_input(feature, value=0.0)

    inputs.append(value)

if st.button("Predict"):

    input_df = pd.DataFrame([inputs], columns=feature_names)

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]

    probability = model.predict_proba(input_scaled)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Not Approved")

    st.subheader("Prediction Probability")

    st.write(f"Approved: **{probability[1]*100:.2f}%**")

    st.write(f"Not Approved: **{probability[0]*100:.2f}%**")