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
    scaler = joblib.load("artifacts/scaler.pkl")  # ← This was the missing piece!
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

    # Debug toggle (only you see this)
    debug = st.checkbox("🔧 Show Debug Info (Dev Mode)", value=False)

# ========================== Main App ==========================
st.title("Capstone Project - MTN Nigeria Customer Churn Prediction")
st.markdown("### Know in seconds if a customer is about to leave — and save them.")

st.info("👈 Use the form below or the sidebar sliders to test any customer profile")

# ========================== Input Form ==========================
c1, c2 = st.columns(2)
with c1:
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges (₦)", 0, 12000, 5000, step=100)
    total_charges = st.number_input("Total Charges (₦)", 0, 100000, 18000, step=500)
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen?", ["No", "Yes"])

with c2:
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    payment_method = st.selectbox("Payment Method", 
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    paperless_billing = st.selectbox("Paperless Billing?", ["Yes", "No"])

st.markdown("#### Add-on Services")
c3, c4, c5 = st.columns(3)
with c3:
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
with c4:
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
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
if st.button("Predict Churn Risk", type="primary", use_container_width=True):
    input_dict = {col: 0 for col in feature_names}

    # Raw values
    raw_tenure = tenure
    raw_monthly = monthly_charges
    raw_total = total_charges if total_charges > 0 else raw_monthly * max(raw_tenure, 1)

    # Scale numeric features
    scaled = scaler.transform([[raw_tenure, raw_monthly, raw_total]])[0]
    input_dict['tenure'] = scaled[0]
    input_dict['MonthlyCharges'] = scaled[1]
    input_dict['TotalCharges'] = scaled[2]

    # Engineered features
    input_dict['AvgMonthlySpend'] = raw_total / max(raw_tenure, 1)
    input_dict['Tenure_MonthlyCharge'] = raw_tenure * raw_monthly
    input_dict['ContractStrength'] = {"Month-to-month":1, "One year":2, "Two year":3}[contract]
    input_dict['AutoPay'] = 1 if "automatic" in payment_method else 0

    # Count of "Yes" add-ons (StreamingMovies now included!)
    add_ons = [online_security, online_backup, device_protection,
               tech_support, streaming_tv, streaming_movies]
    input_dict['ServicesCount'] = sum(1 for x in add_ons if x == "Yes")

    # One-hot encoding
    input_dict['gender_Male'] = 1 if gender == "Male" else 0
    input_dict['SeniorCitizen'] = 1 if senior_citizen == "Yes" else 0
    input_dict['Partner_Yes'] = 1 if partner == "Yes" else 0
    input_dict['Dependents_Yes'] = 1 if dependents == "Yes" else 0
    input_dict['PhoneService_Yes'] = 1 if phone_service == "Yes" else 0
    input_dict['PaperlessBilling_Yes'] = 1 if paperless_billing == "Yes" else 0

    if contract != "Month-to-month":
        input_dict[f'Contract_{contract}'] = 1
    if payment_method != "Bank transfer (automatic)":
        input_dict[f'PaymentMethod_{payment_method}'] = 1
    if internet_service != "DSL":
        input_dict[f'InternetService_{internet_service}'] = 1

    # Add-ons (including StreamingMovies)
    for feat, val in [
        ('OnlineSecurity', online_security), ('OnlineBackup', online_backup),
        ('DeviceProtection', device_protection), ('TechSupport', tech_support),
        ('StreamingTV', streaming_tv), ('StreamingMovies', streaming_movies)
    ]:
        if val == "Yes":
            input_dict[f'{feat}_Yes'] = 1
        elif val == "No internet service":
            input_dict[f'{feat}_No internet service'] = 1

    if multiple_lines == "Yes":
        input_dict['MultipleLines_Yes'] = 1
    elif multiple_lines == "No phone service":
        input_dict['MultipleLines_No phone service'] = 1

    # Final input
    input_df = pd.DataFrame([input_dict]).reindex(columns=feature_names, fill_value=0)

    # Prediction
    prob = model.predict_proba(input_df)[0][1]

    # Risk tier & confidence (solves the 0%/100% feeling)
    if prob < 0.3:
        tier = "Low Risk"
        confidence = "Very High"
    elif prob < 0.6:
        tier = "Medium Risk"
        confidence = "Moderate"
    else:
        tier = "High Risk"
        confidence = "High" if prob < 0.95 else "Very High"

    # Display results
    st.markdown("<br>", unsafe_allow_html=True)
    color = "red" if prob >= THRESHOLD else "green"
    st.markdown(f"<h1 style='text-align: center; color: {color};'>{prob:.1%}</h1>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='text-align: center;'>{'HIGH RISK – WILL CHURN' if prob >= THRESHOLD else 'SAFE – WILL STAY'}</h3>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Churn Probability", f"{prob:.1%}")
    with col2:
        st.metric("Risk Tier", tier)
    with col3:
        st.metric("Confidence", confidence)

    # Clean, wrapped action boxes
    if prob >= THRESHOLD:
        st.error(
            """
            **High Churn Risk — Act Now!**\n
            • Offer 20–50% discount on next bill\n
            • Add free tech support + bonus data\n
            • Propose 12-month contract upgrade\n
            • Make proactive care call within 24 hours
            """,
            icon="⚠️"
        )
    else:
        st.success(
            """
            **Low Risk — Stable Customer**\n
            • Upsell premium add-ons\n
            • Send loyalty reward\n
            • Collect feedback to prevent future churn
            """,
            icon="✅"
        )

    # Debug mode (only shows if you tick the box in sidebar)
    if debug:
        with st.expander("🔍 Debug: Input Vector (for developers)"):
            st.write("Scaled values:", {"tenure": scaled[0], "MonthlyCharges": scaled[1], "TotalCharges": scaled[2]})
            st.dataframe(input_df.style.format("{:.4f}"))

st.caption("Trained on 7,043 customers • Precision-optimized • © Kehinde Balogun 2025")