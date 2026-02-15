# Heart Disease ML App
# Main application file
import streamlit as st
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score,
    f1_score, matthews_corrcoef,
    confusion_matrix
)
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Bank Marketing ML App", layout="wide")
st.title("Bank Marketing Classification – ML Assignment 2")

# Load models
models = {
    "Logistic Regression": joblib.load("model/Logistic_Regression.pkl"),
    "Decision Tree": joblib.load("model/Decision_Tree.pkl"),
    "KNN": joblib.load("model/KNN.pkl"),
    "Naive Bayes": joblib.load("model/Naive_Bayes.pkl"),
    "Random Forest": joblib.load("model/Random_Forest.pkl"),
    "XGBoost": joblib.load("model/XGBoost.pkl")
}

uploaded_file = st.file_uploader(
    "Upload Test Dataset (CSV only)", type=["csv"]
)

model_name = st.selectbox("Select ML Model", list(models.keys()))

if uploaded_file:
    data = pd.read_csv(uploaded_file)


    if "y" not in data.columns:
        st.error("Uploaded file must contain target column 'y'.")
        st.stop()

    X = data.drop("y", axis=1)
    y = data["y"]
    model = models[model_name]

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    st.subheader("Evaluation Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", round(accuracy_score(y, y_pred), 3))
    col1.metric("AUC", round(roc_auc_score(y, y_prob), 3))

    col2.metric("Precision", round(precision_score(y, y_pred), 3))
    col2.metric("Recall", round(recall_score(y, y_pred), 3))

    col3.metric("F1 Score", round(f1_score(y, y_pred), 3))
    col3.metric("MCC", round(matthews_corrcoef(y, y_pred), 3))

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)
