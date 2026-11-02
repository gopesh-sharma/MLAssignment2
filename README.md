## Credit Card Default Classification Portal

#### a. Problem statement

Financial institutions face huge financial risk when credit card clients default on their due payment. Early prediction of potential defaults allows lenders to implement proactive risk-mitigation strategies, such as credit line adjustments or targeted payment reminders. This project aims to build, evaluate, and compare five supervised machine learning models to classify whether a credit card client will default on their next month's payment based on historical payment, billing, and demographic records. The trained models are deployed via an interactive Streamlit web interface for model evaluation.

#### b. Dataset Description

- Dataset Name: Default of Credit Card Clients Dataset   
- Source: UCI Machine Learning Repository / Kaggle   
- Instance Count: 30,000 instances   
- Feature Count: 23 predictive input features   
- Problem Type: Binary Classification (0 = Non-Default, 1 = Default)   
- Feature Breakdown:
   - Demographics: LIMIT_BAL (Credit Limit), SEX, EDUCATION, MARRIAGE, AGE   
   - Repayment Status (PAY_0 to PAY_6): Past monthly payment status tracking delays from April to September   
   - Bill Amounts (BILL_AMT1 to BILL_AMT6): Monthly bill statement figures   
   - Payment Amounts (PAY_AMT1 to PAY_AMT6): Previous monthly payment figures 

#### c. Github Repository Link

[https://github.com/gopesh-sharma/MLAssignment2](https://github.com/gopesh-sharma/MLAssignment2)  

#### d. Models Used

###### Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.8077 | 0.7076 | 0.6868 | 0.2396 | 0.3553 | 0.3244 |
| **Decision Tree** | 0.8163 | 0.7450 | 0.6543 | 0.3595 | 0.4640 | 0.3881 |
| **KNN** | 0.8063 | 0.7327 | 0.6147 | 0.3331 | 0.4321 | 0.3499 |
| **Naive Bayes** | 0.7525 | 0.7249 | 0.4515 | 0.5539 | 0.4975 | 0.3386 |
| **Support Vector Machine** | 0.8178 | 0.7210 | 0.6912 | 0.3188 | 0.4363 | 0.3817 |

###### Model Observations

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieves strong precision (0.6868) and solid accuracy (0.8077) as a linear baseline, but suffers from low recall (0.2396) due to class imbalance (~22% positive default cases). |
| **Decision Tree** | Demonstrates excellent capability in capturing non-linear interactions across repayment status histories (`PAY_0` to `PAY_6`), achieving the highest AUC (0.7450), F1 score (0.4640), and MCC score (0.3881). |
| **KNN** | Benefits significantly from standard feature scaling (`StandardScaler`) across continuous dollar amounts, achieving balanced accuracy (0.8063), though high feature dimensionality slightly degrades distance calculations. |
| **Naive Bayes** | Obtains the highest overall **Recall (0.5539)** and **F1 Score (0.4975)** across all 5 algorithms, capturing the most actual default cases, though it incurs lower overall precision (0.4515). |
| **Support Vector Machine** | Yields the highest **Overall Precision (0.6912)** and overall accuracy (0.8178), separating non-defaulters from defaulters effectively using an RBF kernel in scaled feature space. |
| **Overall, Winner for your dataset?** | **Decision Tree Classifier** — It produces the best overall trade-off across AUC (0.7450), F1 Score (0.4640), and Matthews Correlation Coefficient (0.3881) while naturally handling both continuous bill amounts and discrete payment histories without sensitive scaling bounds. |