import streamlit as st
import mlflow.pyfunc
import numpy as np

# ================== Load Production Model ==================
model_name = "FraudDetectionModel"
prod_model_uri = f"models:/{model_name}/Production"
model = mlflow.pyfunc.load_model(prod_model_uri)

# ================== Meaningful Feature Names ==================
feature_names = [
    "Transaction Amount ($)",
    "Transaction Hour (0-23)",
    "Account Age (months)",
    "Num Previous Transactions",
    "Avg Transaction Amount ($)",
    "Is International (0=No, 1=Yes)",
    "Merchant Risk Score (0-1)",
    "Device Type (0=PC, 1=Mobile)",
    "Num Chargebacks",
    "Is High-Risk Country (0=No, 1=Yes)",
    "Feature 11",
    "Feature 12",
    "Feature 13",
    "Feature 14",
    "Feature 15",
    "Feature 16",
    "Feature 17",
    "Feature 18",
    "Feature 19",
    "Feature 20"
]

st.title("Fraud Detection App")
st.write("Enter transaction details to predict if it is fraud or not.")

# ================== Collect User Inputs ==================
user_input = []
for feature in feature_names:
    if "0-1" in feature:
        val = st.selectbox(feature, [0, 1])
    elif "Hour" in feature:
        val = st.number_input(feature, min_value=0, max_value=23, value=12)
    elif "Amount" in feature or "Avg" in feature:
        val = st.number_input(feature, min_value=0.0, value=100.0, step=10.0)
    elif "Account Age" in feature or "Num" in feature:
        val = st.number_input(feature, min_value=0, value=1, step=1)
    else:
        val = st.number_input(feature, value=0.0)
    user_input.append(val)

# ================== Make Prediction ==================
if st.button("Predict"):
    input_array = np.array(user_input).reshape(1, -1)
    pred = model.predict(input_array)[0]
    st.write("**Fraud Prediction:**", "Fraud" if pred == 1 else "Not Fraud")
