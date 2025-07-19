import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# === Import backend logic ===
from Agentic_Automl_Chat import (
    handle_missing_values,
    detect_task_type,
    handle_imbalance,
    regression_metrics,
    classification_metrics,
    send_email_report,
    client_email
)

st.set_page_config(page_title="Agentic AutoML Chat", layout="wide")
st.title("🤖 Agentic AutoML System with Multi-Agent Support")

# === 1. Upload Dataset ===
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.dataframe(df)

    # === 2. Handle Missing Values ===
    df = handle_missing_values(df)

    # === 3. Select Target Column ===
    target = st.selectbox("🎯 Select Target Column", df.columns)
    if target:
        X = df.drop(columns=[target])
        y = df[target]

        # === 4. Task Type Detection ===
        task_type = detect_task_type(y)
        st.info(f"🔍 Detected task: **{task_type.upper()}**")

        # === 5. Encode categorical
        X = pd.get_dummies(X, drop_first=True)
        if y.dtype == 'O':
            y = y.astype('category').cat.codes

        # === 6. Handle Class Imbalance if classification ===
        if task_type == "classification":
            X, y = handle_imbalance(X, y)

        # === 7. Train/Test Split ===
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # === 8. Train Model ===
        if task_type == "regression":
            model = RandomForestRegressor()
        else:
            model = RandomForestClassifier()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # === 9. Show Metrics ===
        st.subheader("📈 Evaluation Metrics")
        if task_type == "regression":
            result = regression_metrics(y_test, y_pred, X_test)
        else:
            result = classification_metrics(y_test, y_pred)

        st.text(result)

        # === 10. Optional: Email Client ===
        if client_email:
            st.success(f"📤 Email will be sent to: {client_email}")
            if st.button("Send Report"):
                send_email_report(
                    subject="Your AutoML Report",
                    body=result,
                    to=client_email
                )
                st.success("✅ Report sent!")
