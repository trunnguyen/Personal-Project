import streamlit as st
import joblib
from src.processing import prepare_input

# Load model
model = joblib.load("models/fraud_detection_pipeline.pkl")

st.title("🚫 Financial Fraud Detection")

# Inputs
t_type = st.selectbox("Transaction Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "DEPOSIT"])
amt = st.number_input("Amount", min_value=0.0)
oldbalanceOrg=st.number_input("Old Balance (Sender)",min_value=0.0,value=10000.0)
newbalanceOrig=st.number_input("New Balance (Sender)",min_value=0.0,value=9000.0)
oldbalanceDest=st.number_input("Old Balance (Receiver)",min_value=0.0,value=0.0)
newbalanceDest=st.number_input("New Balance (Receiver)",min_value=0.0,value=0.0)

if st.button("Predict"):
    # Match the variable names to your st.number_input definitions
    input_df = prepare_input(
        t_type,
        amt,
        oldbalanceOrg,
        newbalanceOrig,
        oldbalanceDest,
        newbalanceDest
    )

    prediction = model.predict(input_df)

    if prediction[0] == 1:
        st.error("⚠️ Warning: This transaction is flagged as FRAUD.")
    else:
        st.success("✅ Transaction appears to be LEGITIMATE.")