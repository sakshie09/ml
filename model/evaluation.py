# Model evaluation module

import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score,
    f1_score, matthews_corrcoef
)

df = pd.read_csv("model/test_data.csv")

X = df.drop("y", axis=1)
y = df["y"]

model_files = {
    "Logistic Regression": "model/Logistic_Regression.pkl",
    "Decision Tree": "model/Decision_Tree.pkl",
    "KNN": "model/KNN.pkl",
    "Naive Bayes": "model/Naive_Bayes.pkl",
    "Random Forest": "model/Random_Forest.pkl",
    "XGBoost": "model/XGBoost.pkl"
}

results = []

for name, path in model_files.items():
    model = joblib.load(path)

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    results.append({
        "ML Model Name": name,
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_prob),
        "Precision": precision_score(y, y_pred),
        "Recall": recall_score(y, y_pred),
        "F1 Score": f1_score(y, y_pred),
        "MCC": matthews_corrcoef(y, y_pred)
    })

results_df = pd.DataFrame(results)
results_df.to_csv("model/model_metrics.csv", index=False)
print(results_df)
