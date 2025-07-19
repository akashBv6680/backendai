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

# === Together AI Keys ===
together_api_keys = [
    "tgp_v1_ecSsk1__FlO2mB_gAaaP2i-Affa6Dv8OCVngkWzBJUY",
    "tgp_v1_4hJBRX0XDlwnw_hhUnhP0e_lpI-u92Xhnqny2QIDAIM"
]

client_email = st.sidebar.text_input("📨 Enter Client Email")

all_visuals = []

# === Email Report ===
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
    pdf.ln(5)
    for visual in all_visuals:
        if os.path.exists(visual):
            pdf.image(visual, w=180)
    path = "automl_report.pdf"
    pdf.output(path)
    return path

# === AI Agent ===
def get_langchain_agent():
    if not LANGCHAIN_AVAILABLE:
        return None
    llm = ChatOpenAI(temperature=0, model_name="gpt-4")
    tools = [Tool(name="AskQuestion", func=lambda x: f"I received your query: {x}", description="Answer questions.")]
    return initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

# === UI Layout ===
st.set_page_config(page_title="Agentic AutoML 2.0", layout="wide")
st.title("🤖 Enhanced Multi-Agent AutoML System")

st.sidebar.markdown("## 🤖 Talk to an AI Agent")
agent_role = st.sidebar.selectbox("Choose Agent", ["Data Scientist", "ML Engineer"])
user_prompt = st.sidebar.text_area("Ask your question:")
if st.sidebar.button("💬 Ask Agent") and user_prompt:
    with st.spinner("Agent is replying..."):
        if LANGCHAIN_AVAILABLE:
            agent = get_langchain_agent()
            reply = agent.run(user_prompt) if agent else "Langchain agent not available."
        else:
            reply = "Langchain not installed."
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

        @st.cache_resource(show_spinner="Training models...")
        def get_results(X, y, tune):
            agent = AutoMLAgent(X, y, tune=tune)
            results_df, best_info, X_sample = agent.run()
            agent.save_best_model()
            return agent, results_df, best_info, X_sample

        agent, results_df, best_info, X_sample = get_results(X, y, tune_model)

        st.subheader("🏆 Model Leaderboard")
        st.dataframe(results_df)
        st.success(f"Best Model: {best_info['Model']} | Score: {best_info['Score']}")

        st.subheader("📌 Feature Importance (SHAP)")
        if SHAP_AVAILABLE:
            try:
                shap_values = agent.explain_model(X_sample[:50])
                shap.summary_plot(shap_values, X_sample[:50], show=False)
                st.pyplot(plt.gcf())
            except Exception as e:
                st.warning(f"SHAP explanation failed: {e}")
        else:
            st.info("SHAP is not installed. Install it with `pip install shap` to enable model explanations.")

        @st.cache_data
        def get_pdf(results_df, best_info):
            return generate_pdf_report(results_df, best_info)

        pdf_path = get_pdf(results_df, best_info)
        with open(pdf_path, "rb") as f:
            st.download_button("📄 Download PDF Report", f, file_name="automl_report.pdf")

        if client_email:
            send_email_report("AutoML Result", f"Best Model: {best_info['Model']}\nScore: {best_info['Score']}", client_email, [pdf_path])
            st.info("📬 Email report with PDF sent to client.")
