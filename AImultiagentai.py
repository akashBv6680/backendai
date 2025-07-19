# ✅ Enhanced Agentic AutoML with Chat, Explainability, Tuning, and PDF Export

import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle
import smtplib
import seaborn as sns
import matplotlib.pyplot as plt
import os
import base64
from fpdf import FPDF
from email.message import EmailMessage
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB
from imblearn.over_sampling import SMOTE
import xgboost as xgb

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import langchain
    from langchain.chat_models import ChatOpenAI
    from langchain.agents import initialize_agent, Tool
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# === Agent Class ===
class AutoMLAgent:
    def __init__(self, X, y):
        self.X_raw = X.copy()
        self.X = pd.get_dummies(X)
        self.y = y
        self.classification = self._detect_task_type()
        self.models = self._load_models()
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_score = -np.inf
        self.best_info = {}
        self.results = []

    def _detect_task_type(self):
        return self.y.dtype == 'object' or len(np.unique(self.y)) <= 20

    def _load_models(self):
        return {
            "classification": {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Extra Trees": ExtraTreesClassifier(),
                "AdaBoost": AdaBoostClassifier(),
                "KNN": KNeighborsClassifier(),
                "SVC": SVC(),
                "Naive Bayes (Gaussian)": GaussianNB(),
                "Naive Bayes (Multinomial)": MultinomialNB(),
                "Naive Bayes (Complement)": ComplementNB(),
                "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
            },
            "regression": {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "ElasticNet": ElasticNet(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Extra Trees": ExtraTreesRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "KNN": KNeighborsRegressor(),
                "SVR": SVR(),
                "XGBoost": xgb.XGBRegressor(),
                "Polynomial Linear Regression": make_pipeline(PolynomialFeatures(2), LinearRegression())
            }
        }["classification" if self.classification else "regression"]

    def run(self):
        for test_size in [0.1, 0.2, 0.3]:
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=42)

            if self.classification and len(np.unique(y_train)) > 2:
                sampler = SMOTE()
                X_train, y_train = sampler.fit_resample(X_train, y_train)

            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

            for name, model in self.models.items():
                try:
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    score = accuracy_score(y_test, preds) if self.classification else r2_score(y_test, preds)

                    info = {
                        "Model": name,
                        "Score": round(score, 4),
                        "Test Size": test_size,
                        "Type": "Classification" if self.classification else "Regression"
                    }

                    self.results.append(info)
                    if score > self.best_score:
                        self.best_score = score
                        self.best_model = model
                        self.best_info = info
                except Exception:
                    continue

        return pd.DataFrame(self.results).sort_values(by="Score", ascending=False), self.best_info

    def save_best_model(self):
        with open("best_model.pkl", "wb") as f:
            pickle.dump(self.best_model, f)
# === Streamlit UI ===
st.set_page_config(page_title="Agentic AutoML AI", layout="wide")
st.title("🤖 Multi-Agent AutoML System with Email Intelligence")

uploaded_file = st.file_uploader("📁 Upload CSV Dataset", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("📉 Basic EDA")
    st.write(df.describe())
    st.write("Missing Values:")
    st.write(df.isnull().sum())

    all_visuals = []

    fig, ax = plt.subplots()
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis", ax=ax)
    plt.title("Missing Data Visualization")
    plt.tight_layout()
    plt.savefig("eda_missing.png")
    all_visuals.append("eda_missing.png")
    st.pyplot(fig)

    st.subheader("📈 Client-Friendly Visual Insights")
    num_cols = df.select_dtypes(include=np.number).columns
    cat_cols = df.select_dtypes(include='object').columns

    if not num_cols.empty:
        st.markdown("### 🔢 Numeric Feature Distributions")
        for col in num_cols:
            fig, ax = plt.subplots()
            df[col].hist(ax=ax, bins=20, color='skyblue', edgecolor='black')
            ax.set_title(f"Histogram of {col}")
            plt.savefig(f"hist_{col}.png")
            all_visuals.append(f"hist_{col}.png")
            st.pyplot(fig)

        st.markdown("### 🧮 Box Plots (Outlier Detection)")
        for col in num_cols:
            fig, ax = plt.subplots()
            sns.boxplot(data=df, x=col, ax=ax, color='lightcoral')
            ax.set_title(f"Box Plot of {col}")
            plt.savefig(f"box_{col}.png")
            all_visuals.append(f"box_{col}.png")
            st.pyplot(fig)

    if not cat_cols.empty:
        st.markdown("### 🧾 Categorical Feature Breakdown")
        for col in cat_cols:
            fig, ax = plt.subplots()
            df[col].value_counts().plot(kind='bar', ax=ax, color='lightgreen')
            ax.set_title(f"Bar Chart of {col}")
            plt.savefig(f"bar_{col}.png")
            all_visuals.append(f"bar_{col}.png")
            st.pyplot(fig)

            fig, ax = plt.subplots()
            df[col].value_counts().plot(kind='pie', ax=ax, autopct='%1.1f%%', startangle=90)
            ax.set_ylabel("")
            ax.set_title(f"Pie Chart of {col}")
            plt.savefig(f"pie_{col}.png")
            all_visuals.append(f"pie_{col}.png")
            st.pyplot(fig)

    problem_detected = df.isnull().sum().any() or df.select_dtypes(include=np.number).apply(lambda x: ((x - x.mean())/x.std()).abs().gt(3).sum()).sum() > 0

    if problem_detected and client_email:
        eda_summary = """
Dear Client,

Our system has completed the initial analysis of your dataset. Here are the key observations:

- ❗ Potential data quality issues found (missing values or outliers)
- 🧹 Visuals attached for your review (see insights)

Please confirm if you'd like us to proceed with data cleaning and model training.

Regards,
Akash
        """
        send_email_report("Initial Data Quality Report", eda_summary, client_email, all_visuals)
        st.warning("Initial report emailed to client for confirmation before continuing.")

        proceed = st.checkbox("✅ Client confirmed. Proceed with model training?")
        if proceed:
            target = st.selectbox("🎯 Select Target Variable", df.columns)
            if target:
                X = df.drop(columns=[target])
                y = df[target]

                agent = AutoMLAgent(X, y)
                results_df, best_info = agent.run()

                st.subheader("🏆 Model Leaderboard")
                st.dataframe(results_df)

                agent.save_best_model()
                st.success(f"Best Model: {best_info['Model']} with score: {best_info['Score']}")

                model_summary = f"""
Dear Client,

The AutoML process is complete. Here are the results:

✅ Best Model: {best_info['Model']}
📈 Score: {best_info['Score']}
📊 Type: {best_info['Type']}
🔎 Test Size: {best_info['Test Size']}

Thank you for using our AI service.

Regards,
Akash
"""
                send_email_report("Final AutoML Model Report", model_summary, client_email)
                st.info("📬 Final report emailed to client.")
