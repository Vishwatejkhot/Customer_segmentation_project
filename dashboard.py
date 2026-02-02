import streamlit as st
import pandas as pd
import joblib
import os


st.set_page_config(
    page_title="📊 Retention Team Dashboard",
    layout="wide"
)

st.title("📊 Retention Team Dashboard")
st.caption("Customer risk prioritization for retention teams")


model = joblib.load("models/churn_model.pkl")
feature_list = joblib.load("models/feature_list.pkl")
label_encoders = joblib.load("models/label_encoders.pkl")

try:
    threshold = joblib.load("models/decision_threshold.pkl")
except:
    threshold = 0.5

@st.cache_data
def load_raw_data():
    return pd.read_csv("Data/Telco_customer_churn.csv")

df = load_raw_data()


df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)


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

X = df.copy()
X = encode_inputs(X)
X = add_rss_features(X)

X = X.reindex(columns=feature_list, fill_value=0)
X = X.astype(float)


df["churn_prob"] = model.predict_proba(X)[:, 1]
df["Churn_Risk"] = pd.cut(
    df["churn_prob"],
    bins=[0, 0.25, 0.5, 1],
    labels=["Low", "Medium", "High"]
)


st.subheader("📌 Executive Summary")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Customers", len(df))
col2.metric("High-Risk Customers", (df["Churn_Risk"] == "High").sum())
col3.metric("Avg Churn Probability", f"{df['churn_prob'].mean()*100:.1f}%")
col4.metric(
    "Revenue at Risk (Monthly)",
    f"${df.loc[df['Churn_Risk']=='High', 'MonthlyCharges'].sum():,.0f}"
)

st.divider()

st.subheader("📉 Churn Risk Distribution")
st.bar_chart(df["Churn_Risk"].value_counts().reindex(["Low", "Medium", "High"]))

st.divider()


st.subheader("🚨 Priority Customers (Retention Action List)")

priority_df = (
    df[df["Churn_Risk"] == "High"]
    .sort_values("churn_prob", ascending=False)
    .head(25)
)

st.dataframe(
    priority_df[
        [
            "customerID",
            "churn_prob",
            "Contract",
            "PaymentMethod",
            "tenure",
            "MonthlyCharges"
        ]
    ]
    .assign(churn_prob=lambda x: (x["churn_prob"] * 100).round(1))
    .rename(columns={"churn_prob": "Churn Probability (%)"}),
    use_container_width=True
)

st.divider()

st.subheader("🔍 Customer Deep Dive")

selected_customer = st.selectbox("Select Customer ID", df["customerID"])
cust = df[df["customerID"] == selected_customer].iloc[0]

col1, col2, col3 = st.columns(3)

col1.metric("Churn Probability", f"{cust.churn_prob*100:.1f}%")
col2.metric("Risk Level", cust.Churn_Risk)
col3.metric("Monthly Charges", f"${cust.MonthlyCharges:.2f}")

st.markdown("### Customer Snapshot")

st.table(
    cust[
        [
            "tenure",
            "Contract",
            "PaymentMethod",
            "MonthlyCharges",
            "TotalCharges"
        ]
    ].to_frame(name="Value")
)

st.subheader("🎯 Recommended Retention Action")

def retention_action(row):
    if row["Churn_Risk"] == "High" and row["Contract"] == "Month-to-month":
        return "Offer 12-month discounted contract + auto-pay incentive"
    elif row["Churn_Risk"] == "High":
        return "Short-term discount or service bundle optimization"
    elif row["Churn_Risk"] == "Medium":
        return "Engagement offer or loyalty reward"
    else:
        return "No action required – monitor customer"

st.success(retention_action(cust))

st.divider()
st.caption("Customer Retention Intelligence Platform | Retention Team Dashboard")
