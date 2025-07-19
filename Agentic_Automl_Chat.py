# ✅ Full Agentic + Multi-Agent AutoML System with Chat + EDA Email Notifications

import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle
import smtplib
import seaborn as sns
import matplotlib.pyplot as plt
import os

from email.message import EmailMessage
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, mean_absolute_error, confusion_matrix, classification_report, precision_score, recall_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb

from langchain.llms import Together
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# === Together AI ===
together_api_keys = [
    "tgp_v1_ecSsk1__FlO2mB_gAaaP2i-Affa6Dv8OCVngkWzBJUY",
    "tgp_v1_4hJBRX0XDlwnw_hhUnhP0e_lpI-u92Xhnqny2QIDAIM"
]

client_email = st.sidebar.text_input("Enter Client Email")

# === AI Prompt Functions ===
def ask_agent(prompt, model, key=0):
    response = requests.post(
        "https://api.together.xyz/v1/chat/completions",
        headers={"Authorization": f"Bearer {together_api_keys[key]}"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
    )
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    return f"Error: {response.text}"

def ask_data_scientist_agent(prompt):
    return ask_agent(f"[DATA SCIENTIST] {prompt}", "mistralai/Mistral-7B-Instruct-v0.1", key=0)

def ask_ml_engineer_agent(prompt):
    return ask_agent(f"[ML ENGINEER] {prompt}", "mistralai/Mistral-7B-Instruct-v0.1", key=1)

def ask_langchain_agent(prompt):
    llm = Together(model="mistralai/Mixtral-8x7B-Instruct-v0.1", api_key=together_api_keys[0])
    chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate(input_variables=["query"], template="You are a smart agent. {query}")
    )
    return chain.run(prompt)

# === Email Notification ===
def send_email_report(subject, body, to, attachment_paths=None):
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = st.secrets["EMAIL_ADDRESS"]
    msg['To'] = to
    msg.set_content(body)

    if attachment_paths:
        for path in attachment_paths:
            with open(path, 'rb') as f:
                img_data = f.read()
            msg.add_attachment(img_data, maintype='image', subtype='png', filename=os.path.basename(path))

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(st.secrets["EMAIL_ADDRESS"], st.secrets["EMAIL_PASSWORD"])
        smtp.send_message(msg)

# === Metrics Helpers ===
def regression_metrics(y_true, y_pred, X_test):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    adj_r2 = 1 - (1 - r2) * (len(y_true) - 1) / (len(y_true) - X_test.shape[1] - 1)
    return f"R²: {r2:.4f}, Adjusted R²: {adj_r2:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}"

def classification_metrics(y_true, y_pred):
    report = classification_report(y_true, y_pred, zero_division=0)
    conf = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return f"\nPrecision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}\n\nClassification Report:\n{report}\nConfusion Matrix:\n{conf}"

# === Task Detection ===
def detect_task_type(y):
    return "regression" if y.dtype in [np.float64, np.int64] and y.nunique() > 10 else "classification"

# === Missing Value Imputation Agent ===
def handle_missing_values(df):
    if df.isnull().sum().sum() == 0:
        return df
    msg = "Missing values detected in dataset. Do you want to impute them? Options: mean, median, mode, drop."
    response = ask_data_scientist_agent(msg)
    strategy = "mean" if "mean" in response.lower() else "median" if "median" in response.lower() else "mode" if "mode" in response.lower() else "drop"
    for col in df.columns:
        if df[col].isnull().any():
            if strategy == "mean" and df[col].dtype != 'O':
                df[col].fillna(df[col].mean(), inplace=True)
            elif strategy == "median" and df[col].dtype != 'O':
                df[col].fillna(df[col].median(), inplace=True)
            elif strategy == "mode":
                df[col].fillna(df[col].mode()[0], inplace=True)
            elif strategy == "drop":
                df.dropna(inplace=True)
    return df

# === Imbalanced Classification Handler ===
def handle_imbalance(X, y):
    class_counts = y.value_counts(normalize=True)
    if class_counts.min() < 0.1:
        response = ask_data_scientist_agent("Imbalanced classification target detected. Apply SMOTE, ADASYN, or under-sampling?")
        if "adasyn" in response.lower():
            sm = ADASYN()
        elif "smoteenn" in response.lower():
            sm = SMOTEENN()
        elif "under" in response.lower():
            sm = RandomUnderSampler()
        else:
            sm = SMOTE()
        X_res, y_res = sm.fit_resample(X, y)
        return X_res, y_res
    return X, y
