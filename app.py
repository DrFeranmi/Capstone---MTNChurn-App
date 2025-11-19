import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load artifacts
model = joblib.load('best_churn_model_calibrated.pkl')
threshold = joblib.load('optimal_threshold.pkl')
feature_names = joblib.load('feature_names.pkl')

st.title("MTN Churn Prediction Capstone Project")

# Input form (adapt to your 35 features; use sliders/dropdowns for key ones)
inputs = {}
for feature in feature_names[:5]:  # Example: first 5 for demo; expand as needed
    inputs[feature] = st.number_input(f"Enter {feature}", value=0.0)

if st.button("Predict Churn"):
    input_df = pd.DataFrame([inputs])
    proba = model.predict_proba(input_df)[0][1]
    risk = "High Risk (Churn Likely)" if proba >= threshold else "Low Risk"
    st.write(f"Churn Probability: {proba:.2%}")
    st.write(f"Risk Level: {risk} (Threshold: {threshold:.3f})")