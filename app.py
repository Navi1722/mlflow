import streamlit as st
import numpy as np
import joblib
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score

st.title("Fraud Detection App")

# Step 1: Train and save model (if not already saved)
MODEL_PATH = "fraud_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
    st.info("Loaded existing model.")
except:
    st.info("Training models... Please wait!")

    # Prepare dataset
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                               n_redundant=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define models
    models = {
        "LogisticRegression": LogisticRegression(max_iter=500),
        "RandomForest": RandomForestClassifier(n_estimators=100),
        "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    # Train and pick best model
    best_acc = 0
    best_model = None

    for name, m in models.items():
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        acc = accuracy_score(y_test, preds)
        st.write(f"{name} Accuracy: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_model = m

    # Save best model
    joblib.dump(best_model, MODEL_PATH)
    model = best_model
    st.success(f"Best model trained and saved with accuracy {best_acc:.4f}")

# Step 2: Input fields for prediction
st.subheader("Enter 20 features for prediction")
input_data = []
for i in range(20):
    val = st.number_input(f"Feature {i+1}", value=0.0)
    input_data.append(val)

if st.button("Predict"):
    pred = model.predict(np.array([input_data]))
    st.success(f"Fraud Prediction: {pred[0]}")
