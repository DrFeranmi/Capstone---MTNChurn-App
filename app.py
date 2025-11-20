import streamlit as st
import joblib
import pandas as pd
import numpy as np
import sys, traceback

# ========================== Page Config ==========================
st.set_page_config(
    page_title="MTN Nigeria Churn Predictor",
    page_icon="🇳🇬",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ========================== Load Artifacts ==========================
@st.cache_resource
def load_artifacts():
    model = joblib.load("artifacts/best_churn_model_calibrated.pkl")
    threshold = joblib.load("artifacts/optimal_threshold.pkl")
    features = joblib.load("artifacts/feature_names.pkl")
    scaler = joblib.load("artifacts/scaler_verified.pkl")  # ← This was the missing piece!
    return model, threshold, features, scaler

model, THRESHOLD, feature_names, scaler = load_artifacts()

# ========================== Sidebar ==========================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/a/af/MTN_Logo.svg", width=180)
    st.markdown("### MTN Nigeria Churn Prediction")
    st.markdown("**Model**: Calibrated Logistic Regression")
    st.markdown("**AUC**: 0.839  •  **Precision @70% Recall**: 69.5%")
    st.markdown(f"**Threshold**: {THRESHOLD:.3f}")
    st.markdown("—")
    st.markdown("Built by **Kehinde Balogun** • Nov 2025")


# ========================== Main App ==========================
st.title("Capstone Project - MTN Nigeria Customer Churn Prediction")
st.markdown("### Know in seconds if a customer is about to leave — and save them.")

# ========================== Input Form ==========================
c1, c2 = st.columns(2)
with c1:
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges (₦)", 0, 12000, 5000, step=100)
    total_charges = st.number_input("Total Charges (₦)", 0, 900000, 50000, step=500)
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen?", ["No", "Yes"])

with c2:
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], index=1)  # One year
    payment_method = st.selectbox("Payment Method", 
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], 
    index=2)  # Bank transfer (lowest risk)
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    paperless_billing = st.selectbox("Paperless Billing?", ["Yes", "No"])

st.markdown("#### Add-on Services")
c3, c4, c5 = st.columns(3)
with c3:
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"], index=1)  # Yes
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"], index=1)  # Yes
with c4:
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"], index=1)  # Yes
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"], index=1)  # Yes
with c5:
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

st.markdown("#### Other Details")
c6, c7, c8 = st.columns(3)
with c6:
    phone_service = st.selectbox("Phone Service?", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines?", ["No", "Yes", "No phone service"])
with c7:
    partner = st.selectbox("Has Partner?", ["No", "Yes"])
with c8:
    dependents = st.selectbox("Has Dependents?", ["No", "Yes"])

# ========================== Prediction ==========================
if st.button("Check Churn Risk", type="primary", use_container_width=True):
    
    # [Keep your existing input_dict building code here — unchanged]
    input_dict = {col: 0 for col in feature_names}
    raw_tenure = tenure
    raw_monthly = monthly_charges
    raw_total = total_charges if total_charges > 0 else raw_monthly * max(raw_tenure, 1)
    scaled = scaler.transform([[raw_tenure, raw_monthly, raw_total]])[0]
    input_dict['tenure'] = scaled[0]
    input_dict['MonthlyCharges'] = scaled[1]
    input_dict['TotalCharges'] = scaled[2]
    input_dict['AvgMonthlySpend'] = raw_total / max(raw_tenure, 1)
    input_dict['Tenure_MonthlyCharge'] = raw_tenure * raw_monthly
    input_dict['ContractStrength'] = {"Month-to-month":1, "One year":2, "Two year":3}[contract]
    input_dict['AutoPay'] = 1 if "automatic" in payment_method else 0
    add_ons = [online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies]
    input_dict['ServicesCount'] = sum(1 for x in add_ons if x == "Yes")
    input_dict['gender_Male'] = 1 if gender == "Male" else 0
    input_dict['SeniorCitizen'] = 1 if senior_citizen == "Yes" else 0
    input_dict['Partner_Yes'] = 1 if partner == "Yes" else 0
    input_dict['Dependents_Yes'] = 1 if dependents == "Yes" else 0
    input_dict['PhoneService_Yes'] = 1 if phone_service == "Yes" else 0
    input_dict['PaperlessBilling_Yes'] = 1 if paperless_billing == "Yes" else 0
    if contract != "Month-to-month": input_dict[f'Contract_{contract}'] = 1
    if payment_method != "Bank transfer (automatic)": input_dict[f'PaymentMethod_{payment_method}'] = 1
    if internet_service != "DSL": input_dict[f'InternetService_{internet_service}'] = 1
    for feat, val in [
        ('OnlineSecurity', online_security), ('OnlineBackup', online_backup),
        ('DeviceProtection', device_protection), ('TechSupport', tech_support),
        ('StreamingTV', streaming_tv), ('StreamingMovies', streaming_movies)
    ]:
        if val == "Yes": input_dict[f'{feat}_Yes'] = 1
        elif val == "No internet service": input_dict[f'{feat}_No internet service'] = 1
    if multiple_lines == "Yes": input_dict['MultipleLines_Yes'] = 1
    elif multiple_lines == "No phone service": input_dict['MultipleLines_No phone service'] = 1

    input_df = pd.DataFrame([input_dict]).reindex(columns=feature_names, fill_value=0)
    prob = model.predict_proba(input_df)[0][1]
    will_churn = prob >= THRESHOLD

    # ========================== RESULT: YES or NO ==========================
    st.markdown("<br>", unsafe_allow_html=True)

    if will_churn:
        st.error("**HIGH — This customer is likely to churn**")
        st.markdown("""
        ### Recommended Immediate Actions:
        - **Call within 24 hours** with personalized retention offer
        - Offer **30–50% discount** on next bill or free add-ons
        - Propose **contract upgrade** (One/Two year)
        - Assign to **High-Value Retention Team**
        """)
    else:
        st.success("**LOW — This customer is safe**")
        st.markdown("""
        ### Recommended Actions:
        - Monitor monthly
        - Send loyalty reward on next tenure milestone
        - Upsell premium services (streaming, security bundle)
        """)

    # Optional: show probability only in expander (for analysts)
    with st.expander("View detailed probability (for analysts)"):
        st.write(f"Raw churn probability: **{prob:.1%}**")
        st.write(f"Decision threshold: **{THRESHOLD:.3f}**")

st.caption("Trained on 7,043 customers • Precision-optimized for real retention campaigns • © Kehinde Balogun 2025")