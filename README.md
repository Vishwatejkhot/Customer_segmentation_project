# 🤖 Customer Retention Intelligence Platform

An end-to-end **customer churn prediction and retention decision system** built using **Machine Learning, customer segmentation (RFM & RSS), and GenAI-powered insights**.

This project focuses on turning churn predictions into **actionable retention decisions** for both individual agents and retention teams.

---

## 📌 Problem Statement

Customer churn is a major revenue risk for subscription-based businesses like telecom.  
Predicting churn alone is not enough — businesses must understand **why customers churn** and **what actions to take**.

This project addresses:
- Who is likely to churn
- Why they are at risk
- What retention action should be taken

---

## 🎯 Business Objectives

- Predict **customer churn probability**
- Segment customers by **value (RFM)** and **risk (RSS)**
- Prioritize **high-risk, high-value customers**
- Support retention teams with **clear insights**
- Convert ML outputs into **business decisions**

---

## 📊 Dataset

**Telco Customer Churn Dataset**

Includes:
- Customer demographics
- Subscription tenure
- Services and add-ons
- Contract & payment details
- Monthly and total charges
- Target variable: `Churn`

---

## 🧠 Solution Overview

The system is designed as a **two-layer product**:

Analytics & Modeling  
↓  
Decision Intelligence Layer

### Components
- Jupyter Notebook – Analysis, segmentation, modeling
- app.py – Individual customer prediction + GenAI advisor
- dashboard.py – Retention team dashboard

---

## 🧮 Customer Segmentation

### RFM Analysis (Adapted for Subscription Business)

Recency → Customer tenure  
Frequency → Billing continuity  
Monetary → Total lifetime charges  

Used to identify:
- High-value loyal customers
- Potential loyalists
- At-risk customers

### RSS Framework (Custom)

Retention → Length of relationship  
Stability → Contract type  
Spend → Monthly charges  

RSS = 0.4 × Retention + 0.3 × Stability + 0.3 × Spend

---

## 🤖 Churn Prediction Model

Models evaluated:
- Logistic Regression (final model)
- Random Forest (comparison)

Metrics:
- Recall (primary)
- ROC-AUC

Logistic Regression was selected for higher recall and interpretability.

---

## 🖥️ Applications

### app.py
- Individual customer churn prediction
- Risk classification
- GenAI-powered Intelligent Retention Advisor

### dashboard.py
- Executive summary
- Risk distribution
- High-risk customer prioritization
- Customer deep dive
- Retention recommendations

---

## 🛠️ Tech Stack

- Python 3.10+
- Pandas, NumPy
- Scikit-learn
- Streamlit
- Joblib
- Groq API (GPT-OSS-120)
- python-dotenv

---

## ▶️ How to Run

pip install -r requirements.txt

Create .env file:
GROQ_API_KEY=your_groq_api_key_here

Run:
streamlit run app.py
streamlit run dashboard.py

---

## ✨ Author

Vishwatej Khot  
Customer Retention • Applied Machine Learning • GenAI Decision Systems
