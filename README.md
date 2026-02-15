# Bank Marketing Classification --- ML Assignment 2

## 1. Problem Statement

The objective of this project is to implement and compare multiple
machine learning classification models to predict whether a customer
will subscribe to a term deposit based on marketing campaign data.

The project also includes deployment of an interactive Streamlit web
application that allows model selection, dataset upload, and evaluation
metric visualization.

------------------------------------------------------------------------

## 2. Dataset Description

**Dataset:** Bank Marketing Dataset (UCI Machine Learning Repository)\
**File Used:** `bank-full.csv`

### Dataset Details:

-   Total Instances: 45,211
-   Total Features: 16 input features
-   Target Variable: `y`
    -   0 → No subscription
    -   1 → Subscription

### Key Characteristics:

-   Mix of numerical and categorical features
-   Imbalanced target distribution
-   Real-world marketing dataset

------------------------------------------------------------------------

## 3. Data Preprocessing

-   Target column mapped from {yes, no} → {1, 0}
-   Categorical features encoded using `OneHotEncoder`
-   Numerical features scaled using `StandardScaler`
-   Preprocessing handled using `ColumnTransformer`
-   Full preprocessing + model combined using `Pipeline`

------------------------------------------------------------------------

## 4. Train-Test Split

-   80% Training Data
-   20% Testing Data
-   Stratified sampling used (`stratify=y`)
-   `random_state=42` for reproducibility

------------------------------------------------------------------------

## 5. Cross-Validation

-   5-Fold Stratified Cross-Validation
-   Implemented using `StratifiedKFold`
-   Accuracy used as scoring metric

------------------------------------------------------------------------

## 6. Machine Learning Models Implemented

1.  Logistic Regression\
2.  Decision Tree Classifier\
3.  K-Nearest Neighbors (KNN)\
4.  Gaussian Naive Bayes\
5.  Random Forest (Ensemble)\
6.  XGBoost (Ensemble Boosting)

------------------------------------------------------------------------

## 7. Evaluation Metrics

Each model was evaluated using:

-   Accuracy
-   AUC Score
-   Precision
-   Recall
-   F1 Score
-   Matthews Correlation Coefficient (MCC)

------------------------------------------------------------------------

## 8. Model Performance (Test Set)

  Model                 Accuracy   AUC
  --------------------- ---------- --------
  Logistic Regression   \~0.90     \~0.91
  Decision Tree         \~0.87     \~0.70
  KNN                   \~0.89     \~0.82
  Naive Bayes           \~0.85     \~0.81
  Random Forest         \~0.91     \~0.93
  XGBoost               \~0.90     \~0.93

Observation:\
Ensemble models (Random Forest and XGBoost) achieved the highest
performance. Due to dataset imbalance, F1-score and MCC provide more
insight than accuracy alone.

------------------------------------------------------------------------

## 9. Project Structure

    ml-app/
    │
    ├── app.py
    ├── requirements.txt
    ├── bank-full.csv
    ├── README.md
    │
    └── model/
        ├── train_models.py
        ├── evaluation.py
        ├── *.pkl (trained models)
        ├── test_data.csv
        └── model_metrics.csv

------------------------------------------------------------------------

## 10. Streamlit Application Features

-   CSV dataset upload (must contain target column `y`)
-   Model selection dropdown
-   Real-time evaluation metrics display
-   Confusion matrix visualization
-   Clean and interactive user interface

------------------------------------------------------------------------

## 11. Deployment

The application is deployed using **Streamlit Community Cloud**.

Live Application Link:\
`<Add Your Streamlit App Link Here>`

GitHub Repository:\
`<Add Your GitHub Repository Link Here>`

------------------------------------------------------------------------

## 12. Conclusion

This project demonstrates a complete end-to-end machine learning
workflow:

-   Data preprocessing
-   Feature engineering
-   Model comparison
-   Cross-validation
-   Evaluation with multiple metrics
-   Deployment using Streamlit

The implementation follows best practices using Pipelines, Stratified
Cross-Validation, and proper model evaluation techniques suitable for
real-world classification problems.
