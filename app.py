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
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/83/MTN_Group_Logo.svg/1280px-MTN_Group_Logo.svg.png", width=200)
    st.markdown("### **MTN Nigeria Churn Prediction**")
    st.markdown("**Model**: Calibrated Logistic Regression") 
    st.markdown("**AUC**: 0.839 | **Precision**: 69.5% | **Recall**: 70.1%")
    st.markdown(f"**Decision Threshold**: {THRESHOLD:.3f}")
    st.markdown("— Built by **Kehinde Balogun** | Nov 2025")

# ========================== Main App ==========================
st.title("🇳🇬 MTN Nigeria Customer Churn Prediction")
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

# ========================== Prediction ==========================
if st.button("🔮 Predict Churn Risk", type="primary", use_container_width=True):
    # Build input dictionary
    input_dict = {col: 0 for col in feature_names}
    
    # Fill known values
    input_dict['tenure'] = tenure
    input_dict['MonthlyCharges'] = monthly_charges
    input_dict['TotalCharges'] = total_charges
    input_dict[f'Contract_{contract}'] = 1
    input_dict[f'PaymentMethod_{payment_method}'] = 1
    input_dict[f'InternetService_{internet_service}'] = 1
    input_dict[f'OnlineSecurity_{online_security}'] = 1
    input_dict[f'OnlineBackup_{online_backup}'] = 1
    input_dict[f'TechSupport_{tech_support}'] = 1
    input_dict[f'StreamingTV_{streaming_tv}'] = 1
    input_dict['PaperlessBilling_' + paperless_billing] = 1
    input_dict[f'DeviceProtection_{device_protection}'] = 1
    input_dict[f'MultipleLines_{multiple_lines}'] = 1
    input_dict['PhoneService_' + phone_service] = 1
    input_dict['Partner_' + partner] = 1
    input_dict['Dependents_' + dependents] = 1
    input_dict['SeniorCitizen'] = 1 if senior_citizen == "Yes" else 0

    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])[feature_names]

    # Predict
    prob = model.predict_proba(input_df)[0][1]
    prediction = "HIGH RISK – WILL CHURN" if prob >= THRESHOLD else "SAFE – WILL STAY"
    color = "red" if prob >= THRESHOLD else "green"

    # ========================== Results Display ==========================
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