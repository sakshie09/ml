# Bank Marketing Classification — ML Assignment 2

---

## 1. Problem Statement

The objective of this project is to implement and compare multiple
machine learning classification models to predict whether a customer
will subscribe to a term deposit based on marketing campaign data.

The project also includes deployment of an interactive Streamlit web
application that allows model selection, dataset upload, and evaluation
metric visualization.

---

## 2. Dataset Description

**Dataset:** Bank Marketing Dataset (UCI Machine Learning Repository)  
**File Used:** `bank-full.csv`

### Dataset Details:

- Total Instances: 45,211
- Total Features: 16 input features
- Target Variable: `y`
  - 0 → No subscription
  - 1 → Subscription
The dataset contains significantly more "No" responses than "Yes" responses, making metrics such as F1-score and MCC more informative than accuracy alone.


### Key Characteristics:

- Mix of numerical and categorical features
- Imbalanced target distribution
- Real-world marketing campaign dataset

---

## 3. Data Preprocessing

- Target column mapped from {yes, no} → {1, 0}
- Categorical features encoded using `OneHotEncoder`
- Numerical features scaled using `StandardScaler`
- Preprocessing handled using `ColumnTransformer`
- Full preprocessing + model combined using `Pipeline`

---

## 4. Train-Test Split

- 80% Training Data
- 20% Testing Data
- Stratified sampling used (`stratify=y`)
- `random_state=42` for reproducibility

---

## 5. Cross-Validation

- 5-Fold Stratified Cross-Validation
- Implemented using `StratifiedKFold`
- Accuracy used as scoring metric

### Cross-Validation Results

| Model | CV Mean Accuracy |
|--------|------------------|
| Logistic Regression | 0.9016 |
| Decision Tree | 0.8752 |
| KNN | 0.8969 |
| Naive Bayes | 0.8503 |
| Random Forest | 0.8986 |
| XGBoost | 0.9083 |

XGBoost achieved the highest cross-validation accuracy (0.9083), indicating strong generalization capability across folds.

---

## 6. Machine Learning Models Implemented

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN)  
4. Gaussian Naive Bayes  
5. Random Forest (Ensemble)  
6. XGBoost (Boosting Ensemble)

---

## 7. Evaluation Metrics

Each model was evaluated using:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

---

## 8. Model Performance on Test Set

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---------------|----------|------|----------|--------|----------|------|
| Logistic Regression | 0.9013 | 0.9056 | 0.6445 | 0.3478 | 0.4518 | 0.4261 |
| Decision Tree | 0.8746 | 0.7015 | 0.4649 | 0.4754 | 0.4701 | 0.3990 |
| KNN | 0.8962 | 0.8277 | 0.5990 | 0.3403 | 0.4340 | 0.4001 |
| Naive Bayes | 0.8548 | 0.8101 | 0.4059 | 0.5198 | 0.4559 | 0.3774 |
| Random Forest | 0.8979 | 0.9201 | 0.7228 | 0.2070 | 0.3218 | 0.3509 |
| XGBoost | 0.9055 | 0.9287 | 0.6267 | 0.4745 | 0.5401 | 0.4944 |

---

## 9. Observations

| ML Model Name | Observation |
|---------------|------------|
| Logistic Regression | Strong overall performance with good AUC but relatively low recall, indicating missed positive cases. |
| Decision Tree | Moderate recall but lower AUC, suggesting limited generalization capability. |
| KNN | Good accuracy but lower recall; performance affected by class imbalance. |
| Naive Bayes | Higher recall compared to some models but lower precision, leading to more false positives. |
| Random Forest | Achieved high precision and AUC but very low recall, indicating conservative positive predictions. |
| XGBoost | Achieved highest Accuracy, AUC, and MCC, demonstrating the most balanced and robust performance among all models. |

---

## 10. Project Structure

ml/
│── app.py  
│── requirements.txt  
│── bank-full.csv  
│── README.md  
│  
└── model/  
    │── train_models.py  
    │── evaluation.py  
    │── model_metrics.csv  
    │── test_data.csv  
    │── Logistic_Regression.pkl  
    │── Decision_Tree.pkl  
    │── KNN.pkl  
    │── Naive_Bayes.pkl  
    │── Random_Forest.pkl  
    │── XGBoost.pkl  

---

## 11. Streamlit Application Features

- CSV dataset upload (must contain target column `y`)
- Model selection dropdown
- Real-time evaluation metrics display
- Confusion matrix visualization
- Clean and interactive UI

---

## 12. Deployment

The application is deployed using **Streamlit Community Cloud**.

Streamlit App Link:  
https://bank-marketing-ml-app.streamlit.app/

GitHub Repository:  
https://github.com/sakshie09/ml

---

## 13. Conclusion

This project demonstrates a complete end-to-end machine learning workflow:

- Data preprocessing
- Feature engineering
- Model comparison
- Cross-validation
- Evaluation using multiple metrics
- Deployment using Streamlit

The implementation follows best practices using Pipelines,
Stratified Cross-Validation, and robust evaluation techniques
suitable for real-world classification problems.

Among all models, XGBoost demonstrated the best balance between precision and recall, making it the most reliable model for this classification task.
