import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef
)

# 1. Load Data
# Download dataset from UCI or Kaggle (UCI Default of Credit Card Clients)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
df = pd.read_excel(url, header=1)

# Clean column names
df.rename(columns={'default payment next month': 'target'}, inplace=True)
if 'ID' in df.columns:
    df.drop(columns=['ID'], inplace=True)

X = df.drop(columns=['target'])
y = df['target']

# 2. Stratified Train-Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save test dataset for Streamlit testing requirement
test_df = X_test.copy()
test_df['target'] = y_test
test_df.to_csv("test_data.csv", index=False)
print("Saved test_data.csv successfully!")

# 3. Preprocessing (Standard Scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

os.makedirs("model", exist_ok=True)
joblib.dump(scaler, "model/scaler.joblib")

# 4. Initialize Classifiers
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=15),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": CalibratedClassifierCV(SVC(C=1.0, kernel='rbf', random_state=42), ensemble=False)
}

# 5. Train, Evaluate & Save Artifacts
metrics_list = []

for name, model in models.items():
    print(f"Training {name}...")
    # Distance and gradient-based models use scaled features
    if name in ["Logistic Regression", "KNN", "Support Vector Machine", "Naive Bayes"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else: # Decision Tree doesn't require scaling
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    metrics_list.append({
        "ML Model Name": name,
        "Accuracy": round(acc, 4),
        "AUC": round(auc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4),
        "MCC": round(mcc, 4)
    })
    
    # Save model
    filename = name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(model, f"model/{filename}")

# Display Summary Table
results_df = pd.DataFrame(metrics_list)
print("\n=== Model Comparison Table ===")
print(results_df.to_string(index=False))