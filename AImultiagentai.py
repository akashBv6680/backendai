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

# === Together AI Keys ===
together_api_keys = [
    "tgp_v1_ecSsk1__FlO2mB_gAaaP2i-Affa6Dv8OCVngkWzBJUY",
    "tgp_v1_4hJBRX0XDlwnw_hhUnhP0e_lpI-u92Xhnqny2QIDAIM"
]

client_email = st.sidebar.text_input("📨 Enter Client Email")

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

def ask_interactive_agent(role, prompt):
    prompt = f"[{role.upper()}] {prompt}"
    return ask_agent(prompt, "mistralai/Mistral-7B-Instruct-v0.1")

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

# === PDF Export ===
def generate_pdf_report(results_df, best_info):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="AutoML Model Report", ln=True, align='C')
    pdf.ln(10)
    for key, val in best_info.items():
        pdf.cell(200, 10, txt=f"{key}: {val}", ln=True)
    pdf.ln(10)
    pdf.cell(200, 10, txt="Leaderboard:", ln=True)
    for _, row in results_df.iterrows():
        line = ", ".join(f"{k}: {v}" for k, v in row.items())
        pdf.multi_cell(0, 10, line)
    path = "automl_report.pdf"
    pdf.output(path)
    return path

class AutoMLAgent:
    def __init__(self, X, y, tune=False):
        self.X_raw = X.copy()
        self.X = pd.get_dummies(X)
        self.y = y
        self.classification = self._detect_task_type()
        self.tune = tune
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
                except Exception as e:
                    continue

        return pd.DataFrame(self.results).sort_values(by="Score", ascending=False), self.best_info, X_test

    def explain_model(self, X_sample):
        if SHAP_AVAILABLE:
            try:
                if isinstance(self.best_model, (xgb.XGBClassifier, xgb.XGBRegressor,
                                                RandomForestClassifier, RandomForestRegressor,
                                                GradientBoostingClassifier, GradientBoostingRegressor,
                                                ExtraTreesClassifier, ExtraTreesRegressor,
                                                DecisionTreeClassifier, DecisionTreeRegressor)):
                    explainer = shap.TreeExplainer(self.best_model)
                    shap_values = explainer.shap_values(X_sample)
                else:
                    explainer = shap.Explainer(self.best_model, X_sample)
                    shap_values = explainer(X_sample)
                return shap_values
            except Exception as e:
                raise RuntimeError(f"SHAP explanation failed: {e}")
        else:
            raise ImportError("SHAP not available")

    def save_best_model(self):
        with open("best_model.pkl", "wb") as f:
            pickle.dump(self.best_model, f)

# === UI ===
# (UI code remains unchanged)

# === UI ===
st.set_page_config(page_title="Agentic AutoML 2.0", layout="wide")
st.title("🤖 Enhanced Multi-Agent AutoML System")

st.sidebar.markdown("## 🤖 Talk to an AI Agent")
agent_role = st.sidebar.selectbox("Choose Agent", ["Data Scientist", "ML Engineer"])
user_prompt = st.sidebar.text_area("Ask your question:")
if st.sidebar.button("💬 Ask Agent") and user_prompt:
    with st.spinner("Agent is replying..."):
        reply = ask_interactive_agent(agent_role, user_prompt)
        st.sidebar.markdown(f"**Agent Response:**\n{reply}")

uploaded_file = st.file_uploader("📁 Upload CSV Dataset", type="csv")
tune_model = st.checkbox("🔍 Enable Experimental Model Tuning")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("📉 EDA Summary")
    st.write(df.describe())
    st.write("Missing Values:", df.isnull().sum().sum())

    target = st.selectbox("🎯 Select Target Variable", df.columns)
    if target:
        X = df.drop(columns=[target])
        y = df[target]

        agent = AutoMLAgent(X, y, tune=tune_model)
        with st.spinner("Training models..."):
            results_df, best_info, X_sample = agent.run()

        st.subheader("🏆 Model Leaderboard")
        st.dataframe(results_df)

        agent.save_best_model()
        st.success(f"Best Model: {best_info['Model']} | Score: {best_info['Score']}")

        st.subheader("📌 Feature Importance (SHAP)")
        if SHAP_AVAILABLE:
            try:
                shap_values = agent.explain_model(X_sample[:50])
                st.set_option('deprecation.showPyplotGlobalUse', False)
                shap.plots.beeswarm(shap_values)
                st.pyplot(bbox_inches='tight')
            except Exception as e:
                st.warning(f"SHAP explanation failed: {e}")
        else:
            st.info("SHAP is not installed. Install it with `pip install shap` to enable model explanations.")

        pdf_path = generate_pdf_report(results_df, best_info)
        with open(pdf_path, "rb") as f:
            st.download_button("📄 Download PDF Report", f, file_name="automl_report.pdf")

        if client_email:
            send_email_report("AutoML Result", f"Best Model: {best_info['Model']}\nScore: {best_info['Score']}", client_email, [pdf_path])
            st.info("📬 Email report with PDF sent to client.")
