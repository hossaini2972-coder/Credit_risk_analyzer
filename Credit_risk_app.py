import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Model_Training import train_and_save_model, load_model, predict

st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")

st.title("📊 Credit Risk Analytics Dashboard")

# ---------------- Sidebar ----------------
st.sidebar.title("⚙️ Controls")
st.sidebar.write("Upload your dataset to analyze risk")

uploaded_file = st.file_uploader("Upload Dataset", type=["csv"])

# ---------------- MAIN ----------------
if uploaded_file:

    # Load Data
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Data Preview")
    st.dataframe(df.head())

    # ---------------- Model ----------------
    model_name, auc = train_and_save_model(df)
    st.success(f"Best Model: {model_name} | ROC-AUC: {round(auc,3)}")

    model, le_dict, columns = load_model()

    result_df = predict(df, model, le_dict, columns)

    # ---------------- Feature Engineering ----------------
    result_df['DTI'] = result_df['loan_percent_income']

    # ---------------- KPIs ----------------
    st.markdown("## 📊 Key Metrics")

    total = len(result_df)
    high_risk = (result_df['risk_category'] == "High").sum()
    avg_risk = result_df['default_probability'].mean()
    default_rate = avg_risk * 100
    high_risk_pct = (high_risk/total)*100
    

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("🔵 Total Customers", total)
    col2.metric("🔴 High Risk", high_risk)
    col3.metric("🟢 Avg Default Prob", f"{avg_risk:.2f}")
    col4.metric("🟠 Default Rate (%)", f"{default_rate:.2f}%")
    # ---------------- Visualizations ----------------
    st.markdown("## 📈 Insights & Visualizations")

    colA, colB = st.columns(2)

    # Risk Distribution
    with colA:
        st.subheader("Risk Distribution")
        st.bar_chart(result_df['risk_category'].value_counts())

    # Default Probability Trend
    with colB:
        st.subheader("Default Probability Trend")
        st.line_chart(result_df['default_probability'])

    # Income vs Risk
    st.subheader("Income vs Risk")
    fig1, ax1 = plt.subplots()
    sns.scatterplot(
        x=result_df['person_income'],
        y=result_df['default_probability'],
        ax=ax1
    )
    st.pyplot(fig1)

    # Loan Amount vs Risk
    st.subheader("Loan Amount vs Risk")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(
        x=result_df['loan_amnt'],
        y=result_df['default_probability'],
        ax=ax2
    )
    st.pyplot(fig2)

    # Loan Intent vs Risk
    st.subheader("Loan Intent vs Risk")
    fig3, ax3 = plt.subplots()
    sns.boxplot(
        x=result_df['loan_intent'],
        y=result_df['default_probability'],
        ax=ax3
    )
    st.pyplot(fig3)

    # ---------------- Risk Segmentation ----------------
    st.markdown("## ⚠️ Risk Segmentation")

    def risk_level(row):
        if row['DTI'] > 0.5:
            return "High Risk"
        elif row['DTI'] > 0.3:
            return "Medium Risk"
        else:
            return "Low Risk"

    result_df['Risk_Level'] = result_df.apply(risk_level, axis=1)

    st.bar_chart(result_df['Risk_Level'].value_counts())

    # ---------------- Insights ----------------
    st.markdown("## 💡 Key Insights")

    st.markdown("""
    - High **loan_percent_income (>40%)** increases default risk  
    - Customers with higher loan burden are more likely to default  
    - Past defaults strongly influence future risk  
    - Income alone is not a strong predictor without considering loan exposure  
    """)

    # ---------------- Recommendations ----------------
    st.markdown("## 🧠 Recommendations")

    st.markdown("""
    - Limit loan to **≤ 4x income**  
    - Flag high loan_percent_income applicants (>40%)  
    - Monitor customers with past default history  
    - Implement a risk-based approval system  
    """)

    # ---------------- Download Data ----------------
    st.markdown("## ⬇️ Download Data")

    csv = result_df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="Download Cleaned Dataset",
        data=csv,
        file_name='cleaned_credit_data.csv',
        mime='text/csv'
    )

    # ---------------- Report ----------------
    report = f"""
LOAN DEFAULT RISK ANALYSIS REPORT

1. Business Problem
The objective is to identify high-risk loan applicants and reduce default rates while maintaining loan approvals.

2. Key Metrics
Total Customers: {total}
Default Rate: {default_rate:.2f}%
High Risk Customers: {high_risk}
Average Default Probability: {avg_risk:.2f}

3. Key Findings
- High loan_percent_income (>40%) increases default risk
- Loan burden is a stronger predictor than income alone
- Past defaults strongly impact future behavior

4. Risk Segmentation
High Risk: {round(high_risk_pct,2)}%
Medium & Low Risk: Remaining population

5. Observations
- Higher loan exposure increases risk
- Certain loan purposes show higher defaults
- Risk increases when multiple factors combine

6. Recommendations
- Limit loan based on income
- Flag high-risk applicants
- Apply risk-based approval system

7. Conclusion
Default risk is driven by loan burden and borrower behavior. A risk-based approach can reduce defaults and improve decision-making.
"""

    st.download_button(
        label="Download Report",
        data=report,
        file_name="risk_analysis_report.txt",
        mime="text/plain"
    )

else:
    st.info("Please upload a dataset to begin analysis.")