import streamlit as st
import numpy as np
import pickle
import shap
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- LOAD MODEL ---------------- #
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# ✅ EXACT TRAINING FEATURES (FINAL CORRECT)
feature_names = [
    "credit_score",
    "age",
    "tenure",
    "balance",
    "products_number",
    "credit_card",
    "active_member",
    "estimated_salary",
    "country_Germany",
    "country_Spain",
    "gender_Male"
]

# ---------------- SHAP ---------------- #
try:
    explainer = shap.TreeExplainer(model)
except:
    explainer = shap.Explainer(model)

# ---------------- UI ---------------- #
st.set_page_config(page_title="Churn System", layout="wide")

st.title("🏦 AI Customer Retention System")

page = st.sidebar.selectbox(
    "Select Page",
    ["Single Prediction", "Bulk Prediction", "Dashboard"]
)

# =========================================================
# 🔹 SINGLE PREDICTION
# =========================================================
if page == "Single Prediction":

    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.slider("Credit Score", 300, 900, 600)
        age = st.slider("Age", 18, 90, 40)
        tenure = st.slider("Tenure", 0, 10, 3)
        balance = st.number_input("Balance", 0.0, 250000.0, 50000.0)

    with col2:
        products_number = st.slider("Products", 1, 4, 1)
        credit_card = st.selectbox("Has Credit Card", ["No", "Yes"])
        active_member = st.selectbox("Active Member", ["No", "Yes"])
        estimated_salary = st.number_input("Salary", 0.0, 200000.0, 50000.0)
        geography = st.selectbox("Country", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Male", "Female"])

    credit_card = 1 if credit_card == "Yes" else 0
    active_member = 1 if active_member == "Yes" else 0

    country_Germany = 1 if geography == "Germany" else 0
    country_Spain = 1 if geography == "Spain" else 0
    gender_Male = 1 if gender == "Male" else 0

    input_df = pd.DataFrame([[
        credit_score, age, tenure, balance, products_number,
        credit_card, active_member, estimated_salary,
        country_Germany, country_Spain, gender_Male
    ]], columns=feature_names)

    input_scaled = scaler.transform(input_df)

    if st.button("Predict"):

        prob = model.predict_proba(input_scaled)[0][1]

        # -------- RETENTION RECOMMENDATION -------- #
        st.subheader("🧠 Retention Recommendation")

        if prob > 0.7:

                recommendations = []

                if active_member == 0:
                    recommendations.append("📞 Contact customer with engagement offer")

                if products_number <= 1:
                    recommendations.append("🎁 Recommend additional banking products")

                if balance > 100000:
                    recommendations.append("💰 Offer premium wealth benefits")

                if age > 45:
                    recommendations.append("🏦 Provide loyalty membership plan")

                if len(recommendations) == 0:
                    recommendations.append("✅ General retention campaign recommended")

                for rec in recommendations:
                    st.write(rec)

        else:
                st.success("✅ Customer likely to stay")

        # SHAP
        shap_values = explainer(input_df)

        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[0, :, 1], show=False)
        st.pyplot(fig)

# =========================================================
# 🔹 BULK PREDICTION
# =========================================================
elif page == "Bulk Prediction":

    st.subheader("📂 Upload CSV")

    sample_csv = """CreditScore,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary,Geography,Gender
600,40,3,50000,1,1,1,50000,France,Male
700,50,5,100000,2,1,0,80000,Germany,Female
"""
    st.download_button("📥 Download Sample CSV", sample_csv, "sample.csv")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file is not None:

        data = pd.read_csv(file)
        data.columns = data.columns.str.strip().str.lower()

        # -------- COLUMN VALIDATION -------- #
        required_columns = [
            "creditscore",
            "age",
            "tenure",
            "balance",
            "numofproducts",
            "hascrcard",
            "isactivemember",
            "estimatedsalary",
            "geography",
            "gender"
            ]

        missing_cols = [
            col for col in required_columns
            if col not in data.columns
         ]

        if missing_cols:
            st.error(f"❌ Missing columns: {missing_cols}" )
            st.stop()



        try:
            # 🔥 CREATE PERFECT DATAFRAME
            df = pd.DataFrame()

            df["credit_score"] = data["creditscore"]
            df["age"] = data["age"]
            df["tenure"] = data["tenure"]
            df["balance"] = data["balance"]
            df["products_number"] = data["numofproducts"]
            df["credit_card"] = data["hascrcard"]
            df["active_member"] = data["isactivemember"]
            df["estimated_salary"] = data["estimatedsalary"]

            df["country_Germany"] = (data["geography"] == "Germany").astype(int)
            df["country_Spain"] = (data["geography"] == "Spain").astype(int)
            df["gender_Male"] = (data["gender"] == "Male").astype(int)

            # FORCE ORDER
            df = df[feature_names]

            X_scaled = scaler.transform(df)
            probs = model.predict_proba(X_scaled)[:, 1]

            data["Churn_Probability"] = probs
            data["Risk"] = ["HIGH" if p>0.7 else "MEDIUM" if p>0.4 else "LOW" for p in probs]

            st.dataframe(data.head())

            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Results", csv, "results.csv")

        except Exception as e:
            st.error(f"❌ Error: {e}")

# =========================================================
# 🔹 DASHBOARD
# =========================================================
elif page == "Dashboard":

    st.subheader("📊 Dashboard")

    file = st.file_uploader("Upload CSV (with churn/Exited column)", type=["csv"])

    if file is not None:

        data = pd.read_csv(file)
        data.columns = data.columns.str.strip().str.lower()

        # Detect target column
        target_col = None
        if "exited" in data.columns:
            target_col = "exited"
        elif "churn" in data.columns:
            target_col = "churn"

        if target_col is None:
            st.warning("⚠️ No churn column found. Showing preview only.")
            st.dataframe(data.head())

        else:
            st.success("✅ Dataset loaded successfully")

            # -------- PIE -------- #
            st.subheader("🥧 Churn Distribution")
            fig1, ax1 = plt.subplots(figsize=(7,7))
            ax1.pie(
                    data[target_col].value_counts(),
                    labels=["Stay", "Churn"],
                    autopct="%1.1f%%",
                    colors=["#4CAF50", "#FF5252"])
            plt.tight_layout()
            st.pyplot(fig1)

            st.markdown("---")

            # -------- AGE -------- #
            if "age" in data.columns:
                st.subheader("📈 Age vs Churn")
                fig2, ax2 = plt.subplots()
                data.groupby(target_col)["age"].mean().plot(kind="bar", ax=ax2)
                st.pyplot(fig2)

            st.markdown("---")

            # -------- BALANCE -------- #
            if "balance" in data.columns:
                st.subheader("💰 Balance vs Churn")
                fig3, ax3 = plt.subplots()
                data.groupby(target_col)["balance"].mean().plot(kind="bar", ax=ax3)
                st.pyplot(fig3)

            st.markdown("---")

            # -------- COUNTRY -------- #
            if "country" in data.columns:
                st.subheader("🌍 Country vs Churn")
                fig4, ax4 = plt.subplots(figsize=(6,4))
                pd.crosstab(data["country"], data[target_col]).plot(kind="bar", ax=ax4)
                plt.xticks(rotation=0)
                st.pyplot(fig4)

            st.markdown("---")

            # -------- PREVIEW -------- #
            st.subheader("📄 Data Preview")
            st.dataframe(data.head())

   
# ---------------- FOOTER ---------------- #
st.markdown("---")
st.markdown(
    "<center>🚀 Built by <b>Veerendra Challa</b></center>",
    unsafe_allow_html=True
)