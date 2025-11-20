import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ========================== Page Config ==========================
st.set_page_config(
    page_title="MTN Nigeria Churn Predictor",
    page_icon="🇳🇬",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ========================== Load Model ==========================
@st.cache_resource
def load_artifacts():
    model = joblib.load("artifacts/best_churn_model_calibrated.pkl")
    threshold = joblib.load("artifacts/optimal_threshold.pkl")
    features = joblib.load("artifacts/feature_names.pkl")
    return model, threshold, features

model, THRESHOLD, feature_names = load_artifacts()

# ========================== Sidebar ==========================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/a/af/MTN_Logo.svg", width=200)
    st.markdown("### **MTN Nigeria Churn Prediction**")
    st.markdown("**Model**: Calibrated Logistic Regression") 
    st.markdown("**AUC**: 0.839 | **Precision**: 69.5% | **Recall**: 70.1%")
    st.markdown(f"**Decision Threshold**: {THRESHOLD:.3f}")
    st.markdown("— Built by **Kehinde Balogun** | Nov 2025")

# ========================== Main App ==========================
st.title("Capstone Project - MTN Nigeria Customer Churn Prediction")
st.markdown("### Know in seconds if a customer is about to leave — and save them.")

st.info("👈 Use the form below or the sidebar sliders to test any customer profile")

# ========================== Input Form ==========================
col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("How many months has the customer been with MTN?", 0, 72, 24, help="New customers churn more")
    monthly_charges = st.number_input("Monthly Charges (₦)", 0, 20000, 5000, step=500, help="Higher bills = higher risk")
    total_charges = st.number_input("Total Charges to date (₦)", 0, 100000, 30000, step=1000)
    
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    payment_method = st.selectbox("Payment Method", 
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

with col2:
    internet_service = st.selectbox("Internet Service Type", ["Fiber optic", "DSL", "No"])
    online_security = st.selectbox("Online Security Add-on?", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup Add-on?", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support Add-on?", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV Add-on?", ["No", "Yes", "No internet service"])
    paperless_billing = st.selectbox("Paperless Billing?", ["Yes", "No"])

# Additional important features
st.markdown("#### Other Services")
col3, col4, col5 = st.columns(3)
with col3:
    device_protection = st.selectbox("Device Protection?", ["No", "Yes", "No internet service"])
    multiple_lines = st.selectbox("Multiple Lines?", ["No", "Yes", "No phone service"])
with col4:
    phone_service = st.selectbox("Phone Service?", ["Yes", "No"])
    partner = st.selectbox("Has Partner?", ["No", "Yes"])
with col5:
    dependents = st.selectbox("Has Dependents?", ["No", "Yes"])
    senior_citizen = st.selectbox("Senior Citizen?", ["No", "Yes"])
    gender = st.selectbox("Gender", ["Male", "Female"])

# ========================== Prediction ==========================
if st.button("🔮 Predict Churn Risk", type="primary", use_container_width=True):
    input_dict = {col: 0 for col in feature_names}

    # Numeric
    input_dict['tenure'] = tenure
    input_dict['MonthlyCharges'] = monthly_charges
    input_dict['TotalCharges'] = total_charges
    input_dict['SeniorCitizen'] = 1 if senior_citizen == "Yes" else 0

    # Main categoricals
    input_dict[f'Contract_{contract}'] = 1
    input_dict[f'PaymentMethod_{payment_method}'] = 1
    input_dict[f'InternetService_{internet_service}'] = 1
    input_dict[f'gender_{gender}'] = 1  # NEW: Gender one-hot

    # All Yes/No + special cases (fixed for phone/internet)
    for feat, val in [
        ('OnlineSecurity', online_security),
        ('OnlineBackup', online_backup),
        ('DeviceProtection', device_protection),
        ('TechSupport', tech_support),
        ('StreamingTV', streaming_tv),
        ('PaperlessBilling', paperless_billing),
        ('PhoneService', phone_service),
        ('MultipleLines', multiple_lines),
        ('Partner', partner),
        ('Dependents', dependents),
    ]:
        if val == "No internet service":
            input_dict[f'{feat}_No internet service'] = 1
        elif val == "No phone service":
            input_dict[f'{MultipleLines}_No phone service'] = 1  # FIXED: Specific to MultipleLines
        else:
            input_dict[f'{feat}_{val}'] = 1

    # Final DataFrame (enforce order, fill misses)
    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

    # DEBUG: Peek at input (remove in prod)
    with st.expander("Debug: Raw Input Vector (first 10 cols)"):
        st.write(input_df.iloc[0].head(10))

    # Predict
    prob = model.predict_proba(input_df)[0][1]
    prediction = "HIGH RISK – WILL CHURN" if prob >= THRESHOLD else "SAFE – WILL STAY"
    color = "red" if prob >= THRESHOLD else "green"

    # Results (your existing display code here...)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<h1 style='text-align: center; color: {color}; font-size: 52px; font-weight: bold;'>{prob:.1%}</h1>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='text-align: center;'>{prediction}</h3>", unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Churn Probability", f"{prob:.1%}")
    with col_b:
        st.metric("Customers like this usually...", "CHURN" if prob >= THRESHOLD else "STAY")
    with col_c:
        st.metric("Recommended Action", "URGENT RETENTION" if prob >= THRESHOLD else "MONITOR")

    if prob >= THRESHOLD:
        st.error("⚠️ Immediate retention action recommended: discount, bonus data, contract upgrade, or proactive care call.")
    else:
        st.success("✅ Customer is stable. Consider upselling premium services.")

st.markdown("—")
st.caption("Model trained on 7,043 customers | Precision-optimized for real-world retention campaigns | © Kehinde Balogun 2025")