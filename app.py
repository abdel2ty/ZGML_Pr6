import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

# ---------------- Sidebar ----------------
st.sidebar.title("How to Use")
st.sidebar.info("""
1. Upload the advertising dataset (CSV) — *optional*.
2. The app will train a Logistic Regression model and show evaluation metrics.
3. Scroll down to **Manual Input** to enter feature values for prediction.
4. Press **Predict** to see whether the user is likely to click the ad.
""")

# ---------------- Title ----------------
st.title("Advertising — Click Prediction")
st.write("---")

# ---------------- File upload or default ----------------
uploaded_file = st.file_uploader("Upload advertising.csv (optional)", type=["csv"])

default_path = "advertising.csv"
df = None
model = None
scaler = None

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded dataset preview")
    st.write(df.head(10))
elif os.path.exists(default_path):
    df = pd.read_csv(default_path)
    st.subheader("Default dataset preview (advertising.csv)")
    st.write(df.head(10))
else:
    st.warning("No dataset uploaded and default dataset not found. The app will use a temporary dummy model for predictions.")

# ---------------- Train if df available ----------------
if df is not None:
    # expected columns based on the notebook: 
    # ['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Male', 'Clicked on Ad']
    expected_features = ['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Male']
    if not set(expected_features).issubset(df.columns):
        st.error("The dataset does not contain the expected columns. Found columns: " + ", ".join(df.columns))
    else:
        X = df[expected_features]
        y = df['Clicked on Ad']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]

        # ---------------- Metrics ----------------
        st.subheader("Model Performance Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.4f}")
        with col2:
            st.write(f"**Precision:** {precision_score(y_test, y_pred):.4f}")
        with col3:
            st.write(f"**Recall:** {recall_score(y_test, y_pred):.4f}")

        col4, col5 = st.columns(2)
        with col4:
            st.write(f"**F1 Score:** {f1_score(y_test, y_pred):.4f}")
        with col5:
            st.write(f"**ROC AUC:** {roc_auc_score(y_test, y_proba):.4f}")

        st.subheader("Confusion Matrix")
        st.write(confusion_matrix(y_test, y_pred))

        st.success("Model trained successfully! Scroll down for manual prediction.")

st.write("---")

# ---------------- Manual input ----------------
st.subheader("Manual Input for Click Prediction")

feature_columns = [
    'Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Male'
]

user_inputs = {}
st.write("### Enter feature values:")

# Arrange inputs 2 per row (Male will be a selectbox)
for i in range(0, len(feature_columns), 2):
    cols = st.columns(2)
    f1 = feature_columns[i]
    if f1 == 'Male':
        user_inputs[f1] = cols[0].selectbox(f1, options=[0,1], index=0)
    else:
        user_inputs[f1] = cols[0].number_input(f1, value=float(0.0))

    if i+1 < len(feature_columns):
        f2 = feature_columns[i+1]
        if f2 == 'Male':
            user_inputs[f2] = cols[1].selectbox(f2, options=[0,1], index=0)
        else:
            user_inputs[f2] = cols[1].number_input(f2, value=float(0.0))

inputs_df = pd.DataFrame([user_inputs])

# ---------------- Predict ----------------
if st.button("Predict"):
    if model is None or scaler is None:
        st.warning("No real dataset available — training a temporary model on random data for prediction.")

        # create dummy data with realistic-ish ranges
        rng = np.random.RandomState(42)
        dummy_X = np.column_stack([
            rng.normal(loc=65, scale=10, size=500),   # Daily Time Spent on Site
            rng.randint(18, 70, size=500),            # Age
            rng.normal(loc=60000, scale=10000, size=500), # Area Income
            rng.normal(loc=180, scale=50, size=500),  # Daily Internet Usage
            rng.randint(0,2,size=500)                 # Male
        ])
        dummy_y = rng.binomial(1, 0.2, size=500)

        scaler = StandardScaler()
        dummy_scaled = scaler.fit_transform(dummy_X)

        model = LogisticRegression(max_iter=1000)
        model.fit(dummy_scaled, dummy_y)

    user_scaled = scaler.transform(inputs_df[feature_columns])

    pred = model.predict(user_scaled)[0]
    proba = model.predict_proba(user_scaled)[0][1]

    if pred == 1:
        st.success(f"Likely to Click the Ad (Probability = {proba:.2f})")
    else:
        st.info(f"Unlikely to Click (Probability = {proba:.2f})")