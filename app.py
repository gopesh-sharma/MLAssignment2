import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, 
    confusion_matrix, classification_report
)

st.set_page_config(
    page_title="Credit Default Evaluation App", 
    page_icon="💳", 
    layout="wide"
)

st.title("💳 Credit Card Default Classification Portal")
st.write("Upload the evaluation test CSV dataset to inspect performance metrics across trained models.")

# Sidebar Controls
st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload Test CSV Data", type=["csv"])

model_choice = st.sidebar.selectbox(
    "Select Classification Model",
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Support Vector Machine"]
)

@st.cache_resource
def load_artifacts():
    scaler = joblib.load("model/scaler.joblib")
    models = {
        "Logistic Regression": joblib.load("model/logistic_regression.joblib"),
        "Decision Tree": joblib.load("model/decision_tree.joblib"),
        "KNN": joblib.load("model/knn.joblib"),
        "Naive Bayes": joblib.load("model/naive_bayes.joblib"),
        "Support Vector Machine": joblib.load("model/support_vector_machine.joblib")
    }
    return scaler, models

try:
    scaler, models = load_artifacts()
except Exception as e:
    st.error(f"Error loading models. Please verify the /model directory structure: {e}")
    st.stop()

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    
    if 'target' not in test_df.columns:
        st.error("Uploaded CSV must contain the ground-truth target column named 'target'.")
        st.stop()
        
    X_test = test_df.drop(columns=['target'])
    y_test = test_df['target']
    
    # Scale features for relevant models
    if model_choice in ["Logistic Regression", "KNN", "Support Vector Machine", "Naive Bayes"]:
        X_eval = scaler.transform(X_test)
    else:
        X_eval = X_test.values
        
    selected_model = models[model_choice]
    
    # Make Inference
    y_pred = selected_model.predict(X_eval)
    
    if hasattr(selected_model, "predict_proba"):
        y_proba = selected_model.predict_proba(X_eval)[:, 1]
    else:
        y_proba = y_pred

    # Compute Evaluation Metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    st.subheader(f"📊 Performance Metrics: {model_choice}")
    
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Accuracy", f"{acc:.4f}")
    m2.metric("AUC Score", f"{auc:.4f}")
    m3.metric("Precision", f"{prec:.4f}")
    m4.metric("Recall", f"{rec:.4f}")
    m5.metric("F1 Score", f"{f1:.4f}")
    m6.metric("MCC Score", f"{mcc:.4f}")
    
    st.divider()

    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        st.pyplot(fig)

    with col2:
        st.subheader("Classification Report")
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        st.dataframe(report_df.style.format("{:.4f}"))

else:
    st.info("👈 Please upload your `test_data.csv` file via the sidebar to view evaluation results.")