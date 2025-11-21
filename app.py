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
    scaler = joblib.load("artifacts/scaler.pkl") 
    return model, threshold, features, scaler


model, THRESHOLD, feature_names, scaler = load_artifacts()


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
# use statutory NGN->USD rate
NGN_TO_USD = 220.0

c1, c2 = st.columns(2)
with c1:
    tenure = st.slider("Tenure (months)", 1, 72, 6)
    # NOTE: For POC we accept USD values to match model training units
    monthly_charges = st.number_input("Monthly Charges (₦)", 0.0, 500000.0, 25000.0, step=1000.0, format="%.2f")
    total_charges = st.number_input("Total Charges (₦)", 0.0, 20000000.0, 180000.0, step=10000.0, format="%.2f")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen?", ["No", "Yes"])

with c2:
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], index=1) # One year
    payment_method = st.selectbox("Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    index=2)
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    paperless_billing = st.selectbox("Paperless Billing?", ["Yes", "No"])

st.markdown("#### Add-on Services")
c3, c4, c5 = st.columns(3)
with c3:
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"], index=1)
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"], index=1)
with c4:
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"], index=1)
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"], index=1)
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

    # ========================== ADMIN DEBUG (robust) ==========================
    with st.expander("ADMIN: Debug assembled model input (do not expose)"):
        # show reindexed input
        input_df = pd.DataFrame([input_dict]).reindex(columns=feature_names, fill_value=0)
        st.write("Reindexed input_df shape:", input_df.shape)
        st.write("Key derived features and values:")
        st.write(input_df.loc[:, ["AvgMonthlySpend", "Tenure_MonthlyCharge", "ServicesCount", "ContractStrength", "AutoPay"]])
        st.write("First 40 features (for quick scan):")
        st.dataframe(input_df.iloc[:, :40])

        # numeric/scaler debug
        st.write("scaler.feature_names_in_:", getattr(scaler, "feature_names_in_", None))
        st.write("n_features_in_:", getattr(scaler, "n_features_in_", None))
        st.write("numeric cols used for scaler:", numeric_cols)
        st.write("numeric row values (pre-scale):", numeric_row)
        st.write("scaled_array:", scaled_array.tolist())

        # ---------- Explainability ----------
        import os, json, numpy as np

        coef_path = os.path.join("artifacts", "feature_coefs.json")
        intercept_path = os.path.join("artifacts", "feature_intercept.json")

        contributions = None
        intercept_val = None

        # 1) Preferred: load pre-exported JSON coefficients (no heavy deps)
        if os.path.exists(coef_path) and os.path.exists(intercept_path):
            try:
                with open(coef_path) as f:
                    coef_map = json.load(f)
                with open(intercept_path) as f:
                    intercept_val = json.load(f).get("intercept", 0.0)
                row = input_df.iloc[0]
                contributions = {f: float(row.get(f, 0.0)) * float(coef_map.get(f, 0.0)) for f in feature_names}
                top_pos = sorted(contributions.items(), key=lambda x: -x[1])[:6]
                top_neg = sorted(contributions.items(), key=lambda x: x[1])[:6]
                st.write("EXPLAIN (from JSON):")
                st.write("Intercept:", intercept_val)
                st.write("Top positive contributors (feature, contribution):", top_pos)
                st.write("Top negative contributors (feature, contribution):", top_neg)
                st.write("Logit (from contributions):", intercept_val + sum(contributions.values()))
                st.write("Probability (from contributions):", 1/(1+np.exp(-(intercept_val + sum(contributions.values())))))
            except Exception as e:
                st.write("Could not compute contributions from JSON coefs:", str(e))

        # 2) Fallback: try unpickling model and inspect (may fail if deps missing)
        if contributions is None:
            try:
                import joblib
                M = joblib.load(os.path.join("artifacts", "best_churn_model_calibrated.pkl"))
                est = M
                if hasattr(M, "base_estimator_"):
                    est = M.base_estimator_
                if hasattr(est, "named_steps"):
                    est = list(est.named_steps.values())[-1]
                if hasattr(est, "coef_"):
                    coefs = est.coef_.ravel()
                    intercept_val = float(est.intercept_.ravel()[0])
                    coef_map = dict(zip(feature_names, coefs))
                    row = input_df.iloc[0]
                    contributions = {f: float(row.get(f, 0.0)) * float(coef_map.get(f, 0.0)) for f in feature_names}
                    top_pos = sorted(contributions.items(), key=lambda x: -x[1])[:6]
                    top_neg = sorted(contributions.items(), key=lambda x: x[1])[:6]
                    st.write("EXPLAIN (from model object):")
                    st.write("Intercept:", intercept_val)
                    st.write("Top positive contributors (feature, contribution):", top_pos)
                    st.write("Top negative contributors (feature, contribution):", top_neg)
                    st.write("Logit (from contributions):", intercept_val + sum(contributions.values()))
                    st.write("Probability (from contributions):", 1/(1+np.exp(-(intercept_val + sum(contributions.values())))))
                else:
                    st.write("Model loaded but no coef_ found on final estimator.")
            except Exception as e:
                st.write("Could not load full model for contributions (expected if imblearn missing):", str(e))

        # 3) If model.predict_proba may not be available, compute fallback prob from JSON/model coefs if present
        prob_from_contribs = None
        if contributions is not None and intercept_val is not None:
            logit = intercept_val + sum(contributions.values())
            prob_from_contribs = 1.0 / (1.0 + np.exp(-logit))
            st.write("Probability (fallback from contributions):", prob_from_contribs)

    # END ADMIN expander
    # ---------- end debug ----------
    # Now compute final prob (safe)
    # Compute probability (safe fallback to contributions if model fails)
    try:
        prob = float(model.predict_proba(input_df)[0][1])
    except Exception:
        prob = float(prob_from_contribs) if ('prob_from_contribs' in locals() and prob_from_contribs is not None) else float("nan")

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

    # ========================== RESULT: YES , NO, or MAYBE ==========================
    st.markdown("<br>", unsafe_allow_html=True)

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


    # Optional: show probability only in expander (for analysts)
    with st.expander("View detailed probability (for analysts)"):
        st.write(f"Raw churn probability: **{prob:.1%}**")
        st.write(f"Decision threshold: **{THRESHOLD:.3f}**/**{(THRESHOLD*1.6):.3f}**")

st.caption("Trained on 7,043 customers • Precision-optimized for real retention campaigns • © Kehinde Balogun 2025")


