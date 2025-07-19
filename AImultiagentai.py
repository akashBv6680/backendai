# ✅ Agentic AutoML Chat - All-in-One App

import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import os
import smtplib
from email.message import EmailMessage

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import RandomUnderSampler

from langchain.llms import Together
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# === Setup API Keys ===
together_api_keys = [
    "your_together_api_key_1",
    "your_together_api_key_2"
]

# === LangChain Agents ===
def ask_agent(prompt, model, key=0):
    response = requests.post(
        "https://api.together.xyz/v1/chat/completions",
        headers={"Authorization": f"Bearer {together_api_keys[key]}"},
        json={"model": model, "messages": [{"role": "user", "content": prompt}]}
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
    chain = LLMChain(llm=llm, prompt=PromptTemplate(input_variables=["query"], template="You are a smart agent. {query}"))
    return chain.run(prompt)

# === Streamlit UI ===
st.title("🧠 Agentic AutoML with EDA + Email Notifications")
uploaded_file = st.file_uploader("📤 Upload your CSV file")

client_email = st.text_input("📧 Enter Client Email for EDA Reports")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Preview Data")
    st.dataframe(df.head())

    # === Missing Values Handler ===
    if df.isnull().sum().sum() > 0:
        st.warning("⚠️ Missing values detected.")
        if st.checkbox("Impute missing values (agent-assisted)?"):
            response = ask_data_scientist_agent("Missing values detected. Impute with mean, median, mode, or drop?")
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
            st.success(f"✅ Missing values handled with strategy: {strategy}")

    st.subheader("🎯 Select Target Column")
    target = st.selectbox("Target Variable", df.columns)

    if target:
        X = df.drop(columns=[target])
        y = df[target]

        # Encode categoricals
        for col in X.select_dtypes(include='object').columns:
            X[col] = LabelEncoder().fit_transform(X[col])

        if y.dtype == 'O':
            y = LabelEncoder().fit_transform(y)

        task_type = "regression" if y.dtype in [np.float64, np.int64] and y.nunique() > 10 else "classification"
        st.info(f"🔍 Detected task: **{task_type}**")

        if task_type == "classification":
            class_counts = pd.Series(y).value_counts(normalize=True)
            if class_counts.min() < 0.1:
                st.warning("⚠️ Imbalanced dataset detected.")
                if st.checkbox("Apply sampling (SMOTE/ADASYN/under)?"):
                    response = ask_data_scientist_agent("Imbalanced target. Apply SMOTE, ADASYN, or under-sampling?")
                    if "adasyn" in response.lower():
                        sm = ADASYN()
                    elif "smoteenn" in response.lower():
                        sm = SMOTEENN()
                    elif "under" in response.lower():
                        sm = RandomUnderSampler()
                    else:
                        sm = SMOTE()
                    X, y = sm.fit_resample(X, y)
                    st.success("✅ Resampling applied.")

        # === Train/Test Split ===
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        st.subheader("🧪 Model Training")
        model = RandomForestRegressor() if task_type == "regression" else RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # === Metrics ===
        st.subheader("📈 Model Results")
        if task_type == "regression":
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            st.write(f"**MSE:** {mse:.4f} | **R²:** {r2:.4f}")
        else:
            precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
            recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
            st.write(f"**Precision:** {precision:.4f} | **Recall:** {recall:.4f} | **F1:** {f1:.4f}")
            st.text("Confusion Matrix:")
            st.text(confusion_matrix(y_test, y_pred))

        # === Email Report ===
        if st.button("📩 Send Email Report"):
            plt.figure(figsize=(8, 5))
            if task_type == "regression":
                sns.scatterplot(x=y_test, y=y_pred)
                plt.xlabel("Actual")
                plt.ylabel("Predicted")
                plt.title("Actual vs Predicted")
            else:
                sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d")
                plt.title("Confusion Matrix")

            plot_path = "report_plot.png"
            plt.savefig(plot_path)
            plt.close()

            if client_email:
                msg = EmailMessage()
                msg["Subject"] = "📊 AutoML Model Report"
                msg["From"] = st.secrets["EMAIL_ADDRESS"]
                msg["To"] = client_email
                msg.set_content("Attached is the model evaluation report.")

                with open(plot_path, "rb") as f:
                    msg.add_attachment(f.read(), maintype="image", subtype="png", filename="report_plot.png")

                with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
                    smtp.login(st.secrets["EMAIL_ADDRESS"], st.secrets["EMAIL_PASSWORD"])
                    smtp.send_message(msg)

                st.success("✅ Report emailed!")

# === Smart Agent Panel ===
st.sidebar.subheader("🧠 LangChain Agent")
user_query = st.sidebar.text_area("Ask the AI (optional):")
if st.sidebar.button("Run Smart Agent") and user_query:
    response = ask_langchain_agent(user_query)
    st.sidebar.write(response)
