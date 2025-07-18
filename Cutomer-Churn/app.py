import streamlit as st
import joblib
import numpy as np

model = joblib.load(
    r"C:\Users\nguye\OneDrive\Documents\Data_Science\Customer_Churn_Prediction\model.pkl")
scaler = joblib.load(
    r"C:\Users\nguye\OneDrive\Documents\Data_Science\Customer_Churn_Prediction\scaler.pkl")

st.title("Churn Prediction App")

st.divider()

st.write("Enter the values and hit the predict button for the prediction.")

st.divider()

age = st.number_input("Enter age", min_value=10, max_value=100, value=30)

tenure = st.number_input("Enter Tenure", min_value=0, max_value=130, value=10)

monthlyCharges = st.number_input(
    "Enter Monthly Charge", min_value=30, max_value=150)

gender = st.selectbox("Enter the Gender", ["Male", "Female"])

st.divider()


predictionButton = st.button("Predict")

if predictionButton:
    gender_selected = 1 if gender == "Female" else 0

    x = [age, gender_selected, tenure, monthlyCharges]

    X1 = np.array[x]

    x_array = scaler.transform({X1})

    prediction = model.predict(x_array)[0]

    predicted = "Yes" if prediction == 1 else "No"

    st.write(f"Predicted: {predicted}")

else:
    st.write("Please enter the values and use the button")
