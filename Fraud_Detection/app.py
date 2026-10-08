import os

import joblib
import streamlit as st

from src.processing import prepare_input


# Load model
base_path = os.path.dirname(__file__)
model_path = os.path.join(
    base_path,
    "models",
    "fraud_detection_pipeline.pkl"
)

model = joblib.load(model_path)


# Dataset-based limits
MAX_AMOUNT = 92_445_516.64
MAX_OLD_BALANCE_ORG = 59_585_040.37
MAX_NEW_BALANCE_ORIG = 49_585_040.37
MAX_OLD_BALANCE_DEST = 356_015_889.35
MAX_NEW_BALANCE_DEST = 356_179_278.92


st.title("Financial Fraud Detection")


# Inputs
t_type = st.selectbox(
    "Transaction Type",
    ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"]
)

amt = st.number_input(
    "Amount",
    min_value=0.0,
    max_value=MAX_AMOUNT,
    value=1000.0,
    step=100.0
)

oldbalanceOrg = st.number_input(
    "Old Balance (Sender)",
    min_value=0.0,
    max_value=MAX_OLD_BALANCE_ORG,
    value=10000.0,
    step=100.0
)

newbalanceOrig = st.number_input(
    "New Balance (Sender)",
    min_value=0.0,
    max_value=MAX_NEW_BALANCE_ORIG,
    value=9000.0,
    step=100.0
)

oldbalanceDest = st.number_input(
    "Old Balance (Receiver)",
    min_value=0.0,
    max_value=MAX_OLD_BALANCE_DEST,
    value=0.0,
    step=100.0
)

newbalanceDest = st.number_input(
    "New Balance (Receiver)",
    min_value=0.0,
    max_value=MAX_NEW_BALANCE_DEST,
    value=0.0,
    step=100.0
)


if st.button("Predict"):

    # Basic consistency checks
    validation_errors = []

    if amt > oldbalanceOrg and t_type in ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT"]:
        validation_errors.append(
            "The transaction amount is greater than the sender's old balance."
        )

    if t_type == "CASH_IN" and newbalanceOrig < oldbalanceOrg:
        validation_errors.append(
            "For a CASH_IN transaction, the sender's new balance "
            "would normally be expected to increase."
        )

    if validation_errors:
        for error in validation_errors:
            st.warning(error)

        st.info(
            "Please verify the transaction values before making a prediction."
        )

    else:
        input_df = prepare_input(
            t_type,
            amt,
            oldbalanceOrg,
            newbalanceOrig,
            oldbalanceDest,
            newbalanceDest
        )

        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)[0][1]

        if prediction[0] == 1:
            st.error(
                f"Warning: This transaction is flagged as FRAUD "
                f"(probability: {probability:.2%})"
            )

        else:
            st.success(
                f"Transaction appears to be LEGITIMATE "
                f"(fraud probability: {probability:.2%})"
            )