import streamlit as st
import joblib
import pandas as pd
import numpy as np


# 1. Load Model + Columns + Scaler

@st.cache_resource
def load_assets():
    model = joblib.load("fraud_model.pkl")
    model_columns = joblib.load("fraud_model_columns.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, model_columns, scaler

model, model_columns, scaler = load_assets()


# 2. Page Setup

st.set_page_config(page_title="Fraud Detection AI", layout="centered")

st.title("💳 Fraud Detection System")
st.markdown("Detect whether a transaction is **Fraudulent or Genuine**")

st.divider()


# 3. Input Section

# ==============================
# 3. Input Section
# ==============================

st.subheader("📥 Transaction Details")

# Basic Inputs
amount = st.number_input(
    "Transaction Amount ($)",
    0.0,
    100000.0,
    100.0
)

time = st.number_input(
    "Transaction Time",
    0.0,
    200000.0,
    10000.0
)

st.divider()

st.subheader("🧠 PCA Features (V1 - V28)")

# Create 4 columns layout
col1, col2, col3, col4 = st.columns(4)

# Store feature values
feature_values = {}

# Generate V1 to V28 sliders
for i in range(1, 29):

    with [col1, col2, col3, col4][(i - 1) % 4]:

        feature_values[f"V{i}"] = st.number_input(
            f"V{i}",
            -50.0,
            50.0,
            0.0
        )

st.divider()


# 4. Predict

if st.button("🔍 Detect Fraud"):

    # Create empty dataframe with all 30 features as zeros
    input_df = pd.DataFrame(
        np.zeros((1, len(model_columns))),
        columns=model_columns
    )

    # Fill user inputs
    input_df["Amount"] = amount
    input_df["Time"] = time
    # Fill all V1-V28 values
    for feature, value in feature_values.items():

        if feature in model_columns:
           input_df[feature] = value

    
    # 5. Apply Scaler
    
    # Scaler transforms 'Amount' and 'Time' columns based on training data
    input_df[['Amount', 'Time']] = scaler.transform(input_df[['Amount', 'Time']])

    
    # 6. Prediction
    
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    
    # 7. Output Result
    
    st.subheader("📊 Result")

    if prediction == 0:
        st.success("✅ Genuine Transaction")
    else:
        st.error("🚨 Fraudulent Transaction Detected")

    # Metrics
    st.metric("Fraud Probability", f"{probability:.2%}")
    st.progress(float(probability))

    
    # 8. Risk Interpretation
    
    st.subheader("🧠 Risk Level")

    if probability < 0.2:
        st.success("🟢 Low Risk")
    elif probability < 0.6:
        st.warning("🟡 Medium Risk")
    else:
        st.error("🔴 High Risk")

    
    # 9. Transparency
    
    with st.expander("📋 Show Processed Input Data"):
        st.write(input_df)

