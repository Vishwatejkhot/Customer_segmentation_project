import streamlit as st
import pandas as pd
import joblib
import os
from groq import Groq
from dotenv import load_dotenv


load_dotenv() 

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if GROQ_API_KEY is None:
    st.error("GROQ_API_KEY not found. Please set it in .env file.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

st.set_page_config(page_title="Telco Churn Predictor", layout="centered")


model = joblib.load("models/churn_model.pkl")
feature_list = joblib.load("models/feature_list.pkl")
label_encoders = joblib.load("models/label_encoders.pkl")

try:
    threshold = joblib.load("models/decision_threshold.pkl")
except:
    threshold = 0.5


def encode_binary_columns(df):
    binary_map = {
        "Yes": 1,
        "No": 0,
        "Female": 0,
        "Male": 1
    }

    binary_cols = [
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling"
    ]

    for col in binary_cols:
        df[col] = df[col].map(binary_map)

    return df


def encode_inputs(df):
    
    df = encode_binary_columns(df)

    for col, encoder in label_encoders.items():
        df[col] = encoder.transform(df[col])

    return df


def add_rss_features(df):
    df["RetentionScore"] = df["tenure"] / 72
    df["StabilityScore"] = df["Contract"] / 2
    df["SpendScore"] = df["MonthlyCharges"] / 120

    df["RSS_Score"] = (
        0.4 * df["RetentionScore"]
        + 0.3 * df["StabilityScore"]
        + 0.3 * df["SpendScore"]
    )
    return df


def call_genai(customer_data, churn_prob):
    prompt = f"""
You are a telecom business analyst.

Customer profile:
{customer_data}

Predicted churn probability: {churn_prob:.2%}

1. Explain clearly why this customer may churn or stay.
2. Suggest ONE best retention action.

Keep the answer short and business-friendly.
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content


st.title("🤖 Customer Retention Intelligence Platform")
st.markdown("Predict customer churn and get AI-powered retention insights.")

with st.form("customer_form"):
    gender = st.selectbox("Gender", ["Female", "Male"])
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)

    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    PaymentMethod = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    MonthlyCharges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
    TotalCharges = st.number_input("Total Charges", 0.0, 10000.0, 1500.0)

    submitted = st.form_submit_button("Predict Churn")


if submitted:
    input_dict = {
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }

    input_df = pd.DataFrame([input_dict])

    input_df = encode_inputs(input_df)
    input_df = add_rss_features(input_df)


    input_df = input_df.reindex(columns=feature_list, fill_value=0)
    input_df = input_df.astype(float)

    churn_prob = model.predict_proba(input_df)[0][1]
    churn_pred = int(churn_prob >= threshold)

    st.subheader("🔮 Prediction Result")
    st.metric("Churn Probability", f"{churn_prob:.2%}")

    if churn_pred == 1:
        st.error("⚠️ Customer is likely to churn")
    else:
        st.success("✅ Customer is likely to stay")

    with st.spinner("Generating AI insights..."):
        ai_response = call_genai(input_dict, churn_prob)

    st.subheader("🤖 Intelligent Retention Advisor")
    st.write(ai_response)
