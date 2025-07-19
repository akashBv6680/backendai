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
from sklearn.model_selection import train_test_split
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

# Import necessary Langchain components
from langchain.llms import Together
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import DataFrameLoader # Still useful for converting DF to LangChain Documents if needed for other ops
# from langchain_together import TogetherEmbeddings # REMOVED: No longer needed if not using embeddings
# from langchain.vectorstores import FAISS # REMOVED: No longer needed if not using embeddings
from langchain.chains.Youtubeing import load_qa_chain # Can still be used with a simple "stuff" chain
from langchain_core.documents import Document # Still useful if you manually create documents from DF info

# === Together AI ===
together_api_keys = [
    "tgp_v1_ecSsk1__FlO2mB_gAaaP2i-Affa6Dv8OCVngkWzBJUY",
    "tgp_v1_4hJBRX0XDlwnw_hhUnhP0e_lpI-u92Xhnq2QIDAIM"
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

# === Email Notification ===
def send_email_report(subject, body, to, attachment_paths=None):
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = st.secrets["EMAIL_ADDRESS"]
    msg['To'] = to
    msg.set_content(body)

    if attachment_paths:
        for path in attachment_paths:
            try:
                with open(path, 'rb') as f:
                    img_data = f.read()
                msg.add_attachment(img_data, maintype='image', subtype='png', filename=os.path.basename(path))
            except FileNotFoundError:
                st.warning(f"Attachment not found: {path}")
                continue # Skip this attachment if not found

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(st.secrets["EMAIL_ADDRESS"], st.secrets["EMAIL_PASSWORD"])
            smtp.send_message(msg)
    except Exception as e:
        st.error(f"Failed to send email: {e}. Please check your email credentials in .streamlit/secrets.toml")


# === Agent Class (Existing) ===
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
        if self.y.dtype == 'object':
            return True
        unique_ratio = len(np.unique(self.y)) / len(self.y)
        return len(np.unique(self.y)) <= 20 or unique_ratio < 0.2

    def _load_models(self):
        return {
            "classification": {
                "Logistic Regression": LogisticRegression(max_iter=1000, solver='liblinear'),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Random Forest": RandomForestClassifier(random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                "Extra Trees": ExtraTreesClassifier(random_state=42),
                "AdaBoost": AdaBoostClassifier(random_state=42),
                "KNN": KNeighborsClassifier(),
                "SVC": SVC(probability=True, random_state=42),
                "Naive Bayes (Gaussian)": GaussianNB(),
                "Naive Bayes (Multinomial)": MultinomialNB(),
                "Naive Bayes (Complement)": ComplementNB(),
                "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
            },
            "regression": {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(random_state=42),
                "Ridge": Ridge(random_state=42),
                "ElasticNet": ElasticNet(random_state=42),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Random Forest": RandomForestRegressor(random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                "Extra Trees": ExtraTreesRegressor(random_state=42),
                "AdaBoost": AdaBoostRegressor(random_state=42),
                "KNN": KNeighborsRegressor(),
                "SVR": SVR(),
                "XGBoost": xgb.XGBRegressor(random_state=42),
                "Polynomial Linear Regression": make_pipeline(PolynomialFeatures(2), LinearRegression())
            }
        }["classification" if self.classification else "regression"]

    def run(self):
        if self.classification and self.y.dtype == 'object':
            le = LabelEncoder()
            self.y = le.fit_transform(self.y)
            self.label_encoder = le

        for test_size in [0.1, 0.2, 0.3]:
            if self.classification:
                X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=42, stratify=self.y)
            else:
                X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=42)

            if self.classification and len(np.unique(y_train)) > 2:
                try:
                    sampler = SMOTE(random_state=42)
                    X_train, y_train = sampler.fit_resample(X_train, y_train)
                except ValueError as e:
                    st.warning(f"SMOTE could not be applied due to insufficient samples in some classes: {e}")

            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            for name, model in self.models.items():
                try:
                    if isinstance(model, (MultinomialNB, ComplementNB)) and (X_train_scaled < 0).any():
                        st.info(f"Skipping {name} as it requires non-negative input and data contains negative values after scaling.")
                        continue

                    model.fit(X_train_scaled, y_train)
                    preds = model.predict(X_test_scaled)
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
                    print(f"Error training {name} with test size {test_size}: {e}")
                    continue

        return pd.DataFrame(self.results).sort_values(by="Score", ascending=False), self.best_info

    def save_best_model(self):
        try:
            with open("best_model.pkl", "wb") as f:
                pickle.dump(self.best_model, f)
            st.success("Best model saved as best_model.pkl")
        except Exception as e:
            st.error(f"Failed to save model: {e}")


# === MODIFIED: DatasetAgent Class (No Embeddings) ===
class DatasetAgent:
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe
        self.llm = Together(model="mistralai/Mixtral-8x7B-Instruct-v0.1", together_api_key=together_api_keys[0])
        # We can still use load_qa_chain but it will operate in "stuff" mode,
        # meaning it directly "stuffs" the provided context into the prompt.
        # No retrieval is happening here.
        self.qa_chain = load_qa_chain(self.llm, chain_type="stuff")

    def _get_dataframe_summary(self):
        """Generates a text summary of the DataFrame's structure and a few rows."""
        summary_parts = []

        summary_parts.append(f"The dataset has {self.dataframe.shape[0]} rows and {self.dataframe.shape[1]} columns.")
        summary_parts.append("\n**Columns and their types/statistics:**")
        for col in self.dataframe.columns:
            col_info = f"- Column '{col}' (Type: {self.dataframe[col].dtype})"
            if self.dataframe[col].isnull().any():
                missing_count = self.dataframe[col].isnull().sum()
                col_info += f", Missing: {missing_count} ({missing_count/self.dataframe.shape[0]*100:.2f}%)"
            if self.dataframe[col].dtype == 'object' or self.dataframe[col].nunique() < 20:
                unique_vals = self.dataframe[col].nunique()
                top_5_unique = self.dataframe[col].value_counts().head(5).index.tolist()
                col_info += f", Unique Values: {unique_vals}, Top 5: {top_5_unique}"
            elif self.dataframe[col].dtype in ['int64', 'float64']:
                col_info += f", Min: {self.dataframe[col].min():.2f}, Max: {self.dataframe[col].max():.2f}, Mean: {self.dataframe[col].mean():.2f}"
            summary_parts.append(col_info)

        # Add a few sample rows (adjust based on your LLM's context window limits)
        summary_parts.append("\n**First 5 rows of the dataset:**")
        summary_parts.append(self.dataframe.head().to_markdown(index=False))

        return "\n".join(summary_parts)

    def answer_question(self, query: str):
        # The "context" here will be the generated DataFrame summary, not retrieved documents
        df_context = self._get_dataframe_summary()

        # Create a single Langchain Document object from the DataFrame summary
        # This replaces the need for FAISS and similarity search
        context_document = Document(page_content=df_context, metadata={"source": "dataframe_summary"})

        try:
            qa_template = """
            You are an expert data analyst. Based on the following dataset information, answer the question comprehensively.
            If the answer is not available in the provided context, state that you don't have enough information.

            Dataset Information:
            {context}

            Question: {question}

            Answer:
            """
            qa_prompt = PromptTemplate(template=qa_template, input_variables=["context", "question"])
            # Pass the single context document directly to the chain
            response = self.qa_chain.run(input_documents=[context_document], question=query)
            return response
        except Exception as e:
            st.error(f"Error processing your question: {e}. Please try rephrasing.")
            return "I apologize, but I encountered an error while processing your question. Could you please rephrase it or ask something else?"


# Streamlit App Initialization and UI
st.set_page_config(page_title="Agentic AutoML AI", layout="wide")
st.title("🤖 Multi-Agent AutoML System with Email Intelligence")

# Use a unique key for the file uploader to help with state reset
uploaded_file = st.file_uploader("📁 Upload CSV Dataset", type="csv", key="csv_uploader")

# --- IMPORTANT: Handle file upload and state reset ---
if uploaded_file is not None:
    # Check if a new file has been uploaded compared to the one in session state
    if 'last_uploaded_file_id' not in st.session_state or st.session_state.last_uploaded_file_id != uploaded_file.file_id:
        st.session_state.last_uploaded_file_id = uploaded_file.file_id
        st.session_state.clear() # Clear all session state to reset the app completely for a new file
        # Re-initialize only necessary components after clearing
        st.session_state.messages = [] # Initialize chat history
        st.session_state.df = pd.read_csv(uploaded_file) # Store new dataframe

        st.success("New dataset uploaded! Application state reset.")
        st.experimental_rerun()
    else:
        if 'df' not in st.session_state:
            st.session_state.df = pd.read_csv(uploaded_file)
            st.warning("Same file re-uploaded. Using existing data.")


# Ensure df exists in session state before proceeding
if 'df' in st.session_state and st.session_state.df is not None:
    df = st.session_state.df # Use the dataframe from session state

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("📉 Basic EDA")
    st.write(df.describe())
    st.write("Missing Values:")
    st.write(df.isnull().sum())

    all_visuals = []

    plot_dir = "temp_plots"
    os.makedirs(plot_dir, exist_ok=True)

    fig, ax = plt.subplots()
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis", ax=ax)
    plt.title("Missing Data Visualization")
    plt.tight_layout()
    missing_plot_path = os.path.join(plot_dir, "eda_missing.png")
    plt.savefig(missing_plot_path)
    all_visuals.append(missing_plot_path)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("📈 Client-Friendly Visual Insights")
    num_cols = df.select_dtypes(include=np.number).columns
    cat_cols = df.select_dtypes(include='object').columns

    if not num_cols.empty:
        st.markdown("### 🔢 Numeric Feature Distributions")
        for col in num_cols:
            fig, ax = plt.subplots()
            df[col].hist(ax=ax, bins=20, color='skyblue', edgecolor='black')
            ax.set_title(f"Histogram of {col}")
            hist_plot_path = os.path.join(plot_dir, f"hist_{col}.png")
            plt.savefig(hist_plot_path)
            all_visuals.append(hist_plot_path)
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("### 🧮 Box Plots (Outlier Detection)")
        for col in num_cols:
            fig, ax = plt.subplots()
            sns.boxplot(data=df, x=col, ax=ax, color='lightcoral')
            ax.set_title(f"Box Plot of {col}")
            box_plot_path = os.path.join(plot_dir, f"box_{col}.png")
            plt.savefig(box_plot_path)
            all_visuals.append(box_plot_path)
            st.pyplot(fig)
            plt.close(fig)

    if not cat_cols.empty:
        st.markdown("### 🧾 Categorical Feature Breakdown")
        for col in cat_cols:
            fig, ax = plt.subplots()
            df[col].value_counts().plot(kind='bar', ax=ax, color='lightgreen')
            ax.set_title(f"Bar Chart of {col}")
            bar_plot_path = os.path.join(plot_dir, f"bar_{col}.png")
            plt.savefig(bar_plot_path)
            all_visuals.append(bar_plot_path)
            st.pyplot(fig)
            plt.close(fig)

            fig, ax = plt.subplots()
            df[col].value_counts().plot(kind='pie', ax=ax, autopct='%1.1f%%', startangle=90)
            ax.set_ylabel("")
            ax.set_title(f"Pie Chart of {col}")
            pie_plot_path = os.path.join(plot_dir, f"pie_{col}.png")
            plt.savefig(pie_plot_path)
            all_visuals.append(pie_plot_path)
            st.pyplot(fig)
            plt.close(fig)


    problem_detected = df.isnull().sum().any() or df.select_dtypes(include=np.number).apply(lambda x: ((x - x.mean())/x.std()).abs().gt(3).sum()).sum() > 0

    if problem_detected and client_email:
        if 'initial_report_sent' not in st.session_state:
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
            st.session_state.initial_report_sent = True
    elif not client_email:
        st.warning("Enter client email in the sidebar to enable email notifications.")

    if problem_detected:
        proceed = st.checkbox("✅ Client confirmed. Proceed with model training?", key="proceed_training")
        if proceed:
            target = st.selectbox("🎯 Select Target Variable", df.columns, key="target_select")
            if target:
                X = df.drop(columns=[target])
                y = df[target]

                if 'automl_results' not in st.session_state or st.session_state.get('last_target') != target:
                    st.session_state.last_target = target
                    st.info("Running AutoML models... This may take a moment.")
                    agent = AutoMLAgent(X, y)
                    results_df, best_info = agent.run()

                    st.session_state.automl_results = results_df
                    st.session_state.best_model_info = best_info
                    agent.save_best_model()

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

                st.subheader("🏆 Model Leaderboard")
                st.dataframe(st.session_state.automl_results)

                st.success(f"Best Model: {st.session_state.best_model_info['Model']} with score: {st.session_state.best_model_info['Score']}")

    # --- Dataset Agent Chat Interface ---
    st.markdown("---")
    st.subheader("💬 Ask Your Dataset Agent!")
    st.write("Type your questions about the dataset below:")

    # Initialize DatasetAgent only once per loaded dataframe
    if "dataset_agent" not in st.session_state:
        st.session_state.dataset_agent = DatasetAgent(df)

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask me about the dataset...", key="chat_input"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.dataset_agent.answer_question(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

else:
    st.info("Upload a CSV file to begin the AutoML process and interact with the Dataset Agent.")
    if 'df' in st.session_state:
        del st.session_state.df
    if 'dataset_agent' in st.session_state:
        del st.session_state.dataset_agent
    if 'messages' in st.session_state:
        del st.session_state.messages
