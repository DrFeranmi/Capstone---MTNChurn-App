# app.py — FINAL PRODUCTION VERSION — MTN Nigeria Churn Alert System
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ========================== Page Config ==========================
st.set_page_config(
    page_title="MTN Nigeria Churn Alert",
    page_icon="NG",
    layout="wide"
)

# ========================== Load Artifacts ==========================
@st.cache_resource
def load_artifacts():
    model = joblib.load("artifacts/best_churn_model_calibrated.pkl")
    threshold = joblib.load("artifacts/optimal_threshold.pkl")
    features = joblib.load("artifacts/feature_names.pkl")
    scaler = joblib.load("artifacts/scaler.pkl")
    return model, threshold, features, scaler

model, THRESHOLD, feature_names, scaler = load_artifacts()

# MTN Colours
MTN_YELLOW = "#FFC107"
MTN_BLACK = "#212121"

# after load_artifacts() and model/scaler/features loaded
import os
cleaned_path = "Telco_Customer_Churn_Cleaned.csv"
if os.path.exists(cleaned_path):
    _df = pd.read_csv(cleaned_path)
    MEAN_TENURE = float(_df["tenure"].mean())
    MEAN_MONTHLY = float(_df["MonthlyCharges"].mean())
else:
    # fallback reasonable defaults (only for POC; replace with real values later)
    MEAN_TENURE = 24.0
    MEAN_MONTHLY = 50.0

# ========================== Header ==========================
col1, col2 = st.columns([1, 5])
with col1:
    st.image("https://upload.wikimedia.org/wikipedia/commons/a/af/MTN_Logo.svg", width=100)
with col2:
    st.markdown("<h1 style='color:#FFC107; margin:0;'>MTN Nigeria Churn Alert</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#666; font-size:18px;'>Predict churn in seconds — retain customers before they leave</p>", unsafe_allow_html=True)

# Currency conversion
NGN_TO_USD = 220

# ========================== Compact Input Form ==========================
with st.container():
    c1, c2, c3 = st.columns([2, 2, 1])

    with c1:
        st.markdown("**Customer Profile**")
        tenure = st.slider("Tenure (months)", 1, 72, 6)
        monthly_charges = st.number_input("Monthly Charges (₦)", 1000, 500000, 25000, step=1000)
        total_charges = st.number_input("Total Charges (₦)", 1000, 10000000, 180000, step=10000)
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["Yes", "No"])

    with c2:
        st.markdown("**Services & Behaviour**")
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], index=1)
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        payment_method = st.selectbox("Payment Method", ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

    with c3:
        st.markdown("**Add-ons/Other Details**")
        online_security = "Yes" if st.checkbox("Online Security", value=True) else "No"
        streaming_tv = "No" if st.checkbox("Streaming TV", value=False) else "Yes"
        streaming_movies = "No" if st.checkbox("Streaming Movies", value=False) else "Yes"
        online_backup = "Yes" if st.checkbox("Online Backup", value=True) else "No"
        device_protection = "Yes" if st.checkbox("Device Protection", value=True) else "No"
        phone_service = "Yes" if st.checkbox("Phone Service?", value=True) else "No"
        partner = "No" if st.checkbox("Has Partner?", value=False) else "Yes"
        dependents = "No" if st.checkbox("Has Dependents?", value=False) else "Yes"
        multiple_lines = "No" if st.checkbox("Multiple_lines?", value=False) else "Yes"

# ========================== Prediction ==========================
if st.button("Check Churn Risk", type="primary", use_container_width=True):
    input_dict = {col: 0 for col in feature_names}
    # raw values (as before)
    raw_tenure = tenure
    # Precompute raw values in NGN->USD
    raw_monthly_usd = float(monthly_charges) / NGN_TO_USD
    raw_total_usd   = (float(total_charges) / NGN_TO_USD) if total_charges > 0 else raw_monthly_usd * raw_tenure

    # CENTERED approach for interaction (POC safe fix)
    tenure_c = raw_tenure - MEAN_TENURE
    monthly_c = raw_monthly_usd - MEAN_MONTHLY
    avg_monthly = raw_total_usd / max(raw_tenure, 1)

    # Build numeric mapping using centered interaction
    numeric_values_map = {
        "tenure": raw_tenure,                    # keep raw tenure (scaler expects same as training)
        "MonthlyCharges": raw_monthly_usd,
        "TotalCharges": raw_total_usd,
        # use centered product for the interaction feature the model used
        "AvgMonthlySpend": avg_monthly,
        "Tenure_MonthlyCharge": tenure_c * monthly_c
    }

    # Derived numeric features
    avg_monthly = raw_total_usd / max(raw_tenure, 1)
    tenure_monthly = raw_tenure * raw_monthly_usd

    # Build numeric vector matching scaler.feature_names_in_
    numeric_cols = getattr(scaler, "feature_names_in_", None)
    if numeric_cols is None:
        numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges", "AvgMonthlySpend", "Tenure_MonthlyCharge"]
    else:
        numeric_cols = list(numeric_cols)

    # Compute standardized interaction (matching training)
    TMC_scaled = (tenure_monthly - 0.24743262656841747) / 0.9995588904508939

    numeric_values_map = {
        "tenure": raw_tenure,
        "MonthlyCharges": raw_monthly_usd,
        "TotalCharges": raw_total_usd,
        "AvgMonthlySpend": avg_monthly,            
        "Tenure_MonthlyCharge": TMC_scaled         
    }

    numeric_row = [numeric_values_map.get(c, 0) for c in numeric_cols]
    numeric_df = pd.DataFrame([numeric_row], columns=numeric_cols)

    # Scale properly
    scaled_array = scaler.transform(numeric_df)[0]
    scaled_array = np.clip(scaled_array, -3.0, 3.0)

    # Assign scaled values back into input_dict
    for col, val in zip(numeric_cols, scaled_array):
        input_dict[col] = val

    # Ensure derived numeric features exist if not scaled
    if "AvgMonthlySpend" not in numeric_cols:
        input_dict["AvgMonthlySpend"] = avg_monthly
    
    # Categorical and other features
   # 1) ContractStrength — use .get() with a fallback string
    contract_key = contract or "Month-to-month"   # ensures a str
    contract_map = {"Month-to-month": 1, "One year": 2, "Two year": 3}
    input_dict['ContractStrength'] = contract_map.get(contract_key, 1)

    # 2) AutoPay — guard against None, check lowercase for safety
    pm = (payment_method or "").lower()
    input_dict['AutoPay'] = 1 if "automatic" in pm else 0
    add_ons = [online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies]
    input_dict['ServicesCount'] = sum(1 for x in add_ons if x == "Yes")
    input_dict['gender_Male'] = 1 if gender == "Male" else 0
    input_dict['SeniorCitizen'] = 1 if senior_citizen == "Yes" else 0
    input_dict['Partner_Yes'] = 1 if partner == "Yes" else 0
    input_dict['Dependents_Yes'] = 1 if dependents == "Yes" else 0
    input_dict['PhoneService_Yes'] = 1 if phone_service == "Yes" else 0
    input_dict['PaperlessBilling_Yes'] = 1 if paperless_billing == "Yes" else 0

    # One-hot encoding safe assignments
    fn_set = set(feature_names)
    if contract != "Month-to-month":
        name = f'Contract_{contract}'
        if name in fn_set: input_dict[name] = 1
    pm_name = f'PaymentMethod_{payment_method}'
    if pm_name in fn_set: input_dict[pm_name] = 1
    is_name = f'InternetService_{internet_service}'
    if is_name in fn_set: input_dict[is_name] = 1

    for feat, val in [
    ('OnlineSecurity', online_security), ('OnlineBackup', online_backup),
    ('DeviceProtection', device_protection), ('TechSupport', tech_support),
    ('StreamingTV', streaming_tv), ('StreamingMovies', streaming_movies)
    ]:
        if val == "Yes":
            name = f'{feat}_Yes'
            if name in fn_set: input_dict[name] = 1
        elif val == "No internet service":
            name = f'{feat}_No internet service'
            if name in fn_set: input_dict[name] = 1

    if multiple_lines == "Yes":
        if 'MultipleLines_Yes' in fn_set: input_dict['MultipleLines_Yes'] = 1
    elif multiple_lines == "No phone service":
        if 'MultipleLines_No phone service' in fn_set: input_dict['MultipleLines_No phone service'] = 1

    input_df = pd.DataFrame([input_dict]).reindex(columns=feature_names, fill_value=0)

    # Predict
    prob = float(model.predict_proba(input_df)[0][1])
    
    # Determine 3-level risk: high, mid, low
    # Compute customer value (USD) using converted total
    customer_total_usd = raw_total_usd
    # Simple value tiers (USD): low <300, mid 300-700, high >700
    if customer_total_usd > 1500:
        value_tier = "High-Value"
    elif customer_total_usd >= 800:
        value_tier = "Mid-Value"
    else:
        value_tier = "Low-Value"
    
    # Map prob -> band using calibrated prob
    band = "UNKNOWN"
    if np.isnan(prob):
        band = "UNKNOWN"
    elif prob >= THRESHOLD * 1.6:
        band = "high"
    elif prob >= THRESHOLD:
        band = "mid"
    else:
        band = "low"

    # ========================== Results + Charts ==========================
    col_a, col_b = st.columns([1, 2], gap="large")
    with col_a:
        st.markdown("<br><br>", unsafe_allow_html=True)
        actions = []
        if band == "high":
            if value_tier == "High-Value":
                actions = [
                    "Call within 6 hours by Senior Retention Team",
                    "Offer bespoke financial incentive (bill credit or multi-month discount)",
                    "Propose immediate contract extension (1-2 years) with bundled add-ons",
                    "Assign a dedicated account manager and follow-up within 48 hours"
                ]
            elif value_tier == "Mid-Value":
                actions = [
                    "Call within 24 hours with targeted offer",
                    "Offer 20-40% bill discount or free premium add-on for 3 months",
                    "Send SMS + email reminder about loyalty benefits"
                ]
            else: # Low-Value
                actions = [
                    "Automated outreach: SMS + email with a tailored discount code",
                    "Offer 1-month free add-on to improve stickiness",
                    "Monitor for next billing cycle"
                ]
        elif band == "mid":
            if value_tier == "High-Value":
                actions = [
                    "Proactive outreach by operations team within 48 hours",
                    "Offer non-financial perks (priority support, free add-on) to reduce churn risk",
                    "Monitor usage patterns for upsell opportunities"
                ]
            elif value_tier == "Mid-Value":
                actions = [
                    "Send targeted promotional offer via SMS/email",
                    "Offer a smaller discount or bundle trial",
                    "Enroll in loyalty nudges (reminders at key tenure milestones)"
                ]
            else:
                actions = [
                    "Monitor and send value communication (tips, package benefits)",
                    "Offer small non-financial incentives (e.g., free month trial)"
                ]
        elif band == "low":
            actions = [
                "Low priority: continue standard communications",
                "Offer loyalty milestones and occasional rewards",
                "Monitor for feature uptake and upsell opportunities"
            ]
        else:
            actions = ["Unable to compute risk band (check artifacts)"]
        # Render result and actions
        st.markdown("<br>", unsafe_allow_html=True)
        if band == "high":
            st.error("**HIGH — This customer is likely to churn**")
        elif band == "mid":
            st.warning("**MEDIUM — This customer has elevated churn risk**")
        elif band == "low":
            st.success("**LOW — This customer is low risk**")
        else:
            st.info("Risk unknown")

        st.markdown("### Recommended actions:")
        for a in actions:
            st.markdown(f"- {a}")
    with col_b:
        # Chart: Risk Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={'suffix': "%"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': 'teal'},
                'steps': [
                    {'range': [0, 40], 'color': 'lightgreen'},
                    {'range': [40, 70], 'color': 'orange'},
                    {'range': [70, 100], 'color': 'red'}
                ],
                'threshold': {'line': {'color': "red", 'width': 4}, 'value': THRESHOLD * 100}
            },
            title={'text': "Churn Risk Level"}
        ))
        fig_gauge.update_layout(height=250, margin=dict(t=50, b=0, l=0, r=0))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Optional: show probability only in expander (for analysts)
        with st.expander("View detailed probability (for analysts)"):
            st.write(f"Raw churn probability: **{prob:.1%}**")
            st.write(f"Decision threshold: **{THRESHOLD:.3f}**/**{(THRESHOLD*1.6):.3f}**")


st.caption("© Kehinde Balogun 2025 — Capstone Project | Precision-optimized for real retention campaigns")
