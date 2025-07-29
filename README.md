# Fraud Detection Using Machine Learning

This project implements a machine learning pipeline to detect fraudulent financial transactions using a large-scale synthetic dataset of over 6 million rows. We focused on careful feature engineering, class imbalance handling, and training a robust Random Forest classifier to maximize detection accuracy.

## Problem Statement

Fraudulent transactions cause major losses in financial sectors. Detecting fraud in real-time is crucial but challenging due to the imbalanced nature of the data and evolving fraud patterns. Our goal is to build a model that can distinguish fraudulent transactions accurately without disrupting legitimate activity.

## Dataset Overview

- **Records:** 6,362,620 transactions
- **Format:** CSV file
- **Source:** Simulated dataset ([Download](https://drive.google.com/file/d/1p1NuDcKhUI0ZrDS4N1mGn5NPmqPkA9X_/view?usp=sharing))
- **Data Dictionary:** [View](https://drive.google.com/file/d/1wQJYlgNK-ZlbR5CAFXOYxc_rpffDeI7g/view?usp=sharing)

**Key Fields:**
- `step`: Time step (1 = 1 hour)
- `type`: Type of transaction
- `amount`: Transaction amount
- `oldbalanceOrg` / `newbalanceOrig`
- `oldbalanceDest` / `newbalanceDest`
- `isFraud`: Target label

## Project Objectives

- Perform thorough EDA and data cleaning
- Engineer meaningful domain-specific features
- Handle class imbalance with SMOTE
- Train and evaluate a Random Forest model
- Interpret important fraud indicators
- Provide actionable prevention recommendations

## Approach

### 1. EDA & Cleaning
- Checked data types, missing values, and outliers
- Dropped high-cardinality fields (`nameOrig`, `nameDest`)
- Removed irrelevant columns like `isFlaggedFraud`

### 2. Feature Engineering
- Created `balance_diff_of_orig`, `ratio_of_amount_to_orig`, etc.
- Flagged zero balance situations
- Added temporal feature: `day` from `step`
- Applied one-hot encoding to `type`

### 3. Class Balancing
- Used **SMOTE oversampling** after train-test split to resolve imbalance in fraud labels

### 4. Model Selection
- Chose **Random Forest Classifier**
  - Handles non-linear patterns well
  - Robust to outliers
  - Supports feature importance

## Model Performance

| Metric     | Class 0 (Non-Fraud) | Class 1 (Fraud)  |
|------------|---------------------|------------------|
| Precision  | 1.00                | 0.91             |
| Recall     | 1.00                | 1.00             |
| F1-Score   | 1.00                | 0.95             |
| AUC-ROC    | -                   | 0.9996           |

The model shows excellent recall and separability for fraud detection.

## Key Predictive Features

| Feature                       | Description                           | Importance  |
|-------------------------------|---------------------------------------|-------------|
| `ratio_of_amount_to_orig`     | Amount relative to origin balance     |    0.33     |
| `balance_diff_of_orig`        | Drop in origin balance                |    0.31     |
| `ratio_of_amount_to_dest`     | Amount relative to dest balance       |    0.07     |
| `type_TRANSFER`, `log_amount` | Risky transaction indicators          |   ~0.05     |

These features align with real-world fraud patterns such as account draining and transfers.

## Fraud Prevention Suggestions

- Block or review transactions with extreme amount-to-balance ratios
- Monitor for sudden zero balances
- Apply 2FA on new recipients
- Use rate limiting and anomaly detection

## Post-Deployment Monitoring

| Metric              | Expected Outcome      |
|---------------------|-----------------------|
| Fraud Cases Caught  | Increase              |
| False Positives     | Minimal Increase      |
| F1-Score            | Maintain or Improve   |
| Financial Loss      | Significant Reduction |


## Conclusion

This project demonstrates a complete fraud detection pipeline from EDA to deployment-ready model, achieving:
- **AUC-ROC of 0.9996**
- **Recall of 1.00 for fraud**
- Scalable insights for real-world financial fraud prevention

## Files Included

- `fraud_detection_notebook.ipynb` – Full Jupyter notebook with code
- `Fraud Detection Documentation.pdf` – Project report with explanations and metrics
- `README.md` – Project summary and outline

## Author

**Ishaaq MM** – Data Science Enthusiast  
[LinkedIn Profile ([view](https://www.linkedin.com/in/ishaaq-m-m/))]  
[GitHub Portfolio ([view](https://github.com/Ishaaq09))]


