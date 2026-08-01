import streamlit as st
import pandas as pd
import joblib
import shap

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

# =========================================================
# CredWise AI - STREAMLIT APPLICATION
# =========================================================

st.set_page_config(
    page_title="CredWise AI",
    page_icon="🏦",
    layout="wide"
)
# =========================================================
# LOAD CredWise AI PRODUCTION MODELS
# =========================================================

@st.cache_resource
def load_models():

    xgb_pipeline = joblib.load(
        "models/xgboost_pipeline.pkl"
    )

    calibrator = joblib.load(
        "models/probability_calibrator.pkl"
    )

    return xgb_pipeline, calibrator


xgb_pipeline, calibrator = load_models()

# =========================================================
# LOAD SHAP EXPLAINER
# =========================================================

@st.cache_resource
def load_shap_explainer():

    return joblib.load(
        "models/shap_explainer.pkl"
    )


shap_explainer = load_shap_explainer()

# =========================================================
# LOAD BANKING POLICY FAISS VECTOR STORE
# =========================================================

@st.cache_resource
def load_policy_vector_store():

    # IMPORTANT:
    # Must use the same embedding model that was used
    # when the FAISS index was originally created.
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vector_store


policy_vector_store = load_policy_vector_store()

# =========================================================
# LOAD LLAMA 3.2
# =========================================================

@st.cache_resource
def load_llm():

    return ChatOllama(
        model="llama3.2",
        temperature=0
    )


llm = load_llm()


# =========================================================
# HEADER
# =========================================================

st.title("🏦 CredWise AI")

st.subheader(
    "Explainable AI Credit Risk & Loan Intelligence System"
)

st.caption(
    "Credit Risk ML/DL • SHAP Explainability • "
    "Banking Policy RAG • Grounded LLM Reports"
)

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("🏦 CredWise AI")

    st.write(
        "AI-powered credit-risk decision-support system."
    )

    st.divider()

    st.markdown("### System")

    st.write("✓ XGBoost Credit Risk")
    st.write("✓ PyTorch Benchmark")
    st.write("✓ SHAP Explainability")
    st.write("✓ FAISS Policy RAG")
    st.write("✓ Llama 3.2")

    st.divider()

    st.warning(
        "Decision-support prototype. "
        "This system does not automatically approve "
        "or reject loan applications."
    )


# =========================================================
# APPLICANT INFORMATION
# =========================================================

st.header("Applicant & Loan Information")

st.write(
    "Enter the applicant's information to generate "
    "an explainable credit-risk assessment."
)


col1, col2, col3 = st.columns(3)


# =========================================================
# COLUMN 1 - LOAN INFORMATION
# =========================================================

with col1:

    st.markdown("### 💰 Loan Details")

    loan_amnt = st.number_input(
        "Loan Amount ($)",
        min_value=500.0,
        max_value=50000.0,
        value=10000.0,
        step=500.0
    )

    term = st.selectbox(
        "Loan Term",
        options=[36, 60],
        format_func=lambda x: f"{x} months"
    )

    installment = st.number_input(
        "Monthly Installment ($)",
        min_value=0.0,
        value=300.0,
        step=10.0
    )

    purpose = st.selectbox(
        "Loan Purpose",
        options=[
            "debt_consolidation",
            "credit_card",
            "home_improvement",
            "major_purchase",
            "small_business",
            "medical",
            "car",
            "moving",
            "house",
            "vacation",
            "wedding",
            "renewable_energy",
            "educational",
            "other"
        ]
    )


# =========================================================
# COLUMN 2 - FINANCIAL PROFILE
# =========================================================

with col2:

    st.markdown("### 📊 Financial Profile")

    annual_inc = st.number_input(
        "Annual Income ($)",
        min_value=0.0,
        value=60000.0,
        step=1000.0
    )

    dti = st.number_input(
        "Debt-to-Income Ratio",
        min_value=0.0,
        value=15.0,
        step=0.5
    )

    revol_bal = st.number_input(
        "Revolving Credit Balance ($)",
        min_value=0.0,
        value=10000.0,
        step=500.0
    )

    revol_util = st.number_input(
        "Revolving Credit Utilization (%)",
        min_value=0.0,
        max_value=150.0,
        value=40.0,
        step=1.0
    )

    verification_status = st.selectbox(
        "Income Verification",
        options=[
            "Verified",
            "Source Verified",
            "Not Verified"
        ]
    )


# =========================================================
# COLUMN 3 - CREDIT PROFILE
# =========================================================

with col3:

    st.markdown("### 💳 Credit Profile")

    emp_length = st.slider(
        "Employment Length (Years)",
        min_value=0,
        max_value=10,
        value=5
    )

    home_ownership = st.selectbox(
        "Home Ownership",
        options=[
            "RENT",
            "MORTGAGE",
            "OWN",
            "OTHER",
            "NONE"
        ]
    )

    delinq_2yrs = st.number_input(
        "Delinquencies — Last 2 Years",
        min_value=0,
        value=0,
        step=1
    )

    inq_last_6mths = st.number_input(
        "Credit Inquiries — Last 6 Months",
        min_value=0,
        value=1,
        step=1
    )

    open_acc = st.number_input(
        "Open Credit Accounts",
        min_value=0,
        value=10,
        step=1
    )

    pub_rec = st.number_input(
        "Derogatory Public Records",
        min_value=0,
        value=0,
        step=1
    )

    total_acc = st.number_input(
        "Total Credit Accounts",
        min_value=0,
        value=20,
        step=1
    )


st.divider()


# =========================================================
# ANALYZE BUTTON
# =========================================================

analyze_button = st.button(
    "🔍 Analyze Credit Risk",
    type="primary",
    use_container_width=True
)


# =========================================================
# BUILD EXACT 16-FEATURE INPUT
# =========================================================

# =========================================================
# CREDIT RISK ANALYSIS
# =========================================================

if analyze_button:

    # -----------------------------------------------------
    # Build exact 16-feature applicant DataFrame
    # -----------------------------------------------------

    applicant_data = pd.DataFrame(
        [{
            "loan_amnt": loan_amnt,
            "term": term,
            "installment": installment,
            "emp_length": emp_length,
            "home_ownership": home_ownership,
            "annual_inc": annual_inc,
            "verification_status": verification_status,
            "purpose": purpose,
            "dti": dti,
            "delinq_2yrs": delinq_2yrs,
            "inq_last_6mths": inq_last_6mths,
            "open_acc": open_acc,
            "pub_rec": pub_rec,
            "revol_bal": revol_bal,
            "revol_util": revol_util,
            "total_acc": total_acc
        }]
    )


    # -----------------------------------------------------
    # XGBoost raw risk score
    # -----------------------------------------------------

    raw_risk_score = (
        xgb_pipeline
        .predict_proba(applicant_data)[0, 1]
    )


    # -----------------------------------------------------
    # Calibrated Probability of Default
    # -----------------------------------------------------

    default_probability = float(
        calibrator.predict(
            [raw_risk_score]
        )[0]
    )


    # -----------------------------------------------------
    # Simple display risk bands
    # -----------------------------------------------------

    if default_probability < 0.15:

        risk_level = "Low Risk"
        risk_icon = "🟢"

    elif default_probability < 0.30:

        risk_level = "Moderate Risk"
        risk_icon = "🟡"

    else:

        risk_level = "High Risk"
        risk_icon = "🔴"


    # -----------------------------------------------------
    # Results
    # -----------------------------------------------------

    st.divider()

    st.header("Credit Risk Assessment")


    metric1, metric2, metric3 = st.columns(3)


    with metric1:

        st.metric(
            "Probability of Default",
            f"{default_probability:.1%}"
        )


    with metric2:

        st.metric(
            "Risk Level",
            f"{risk_icon} {risk_level}"
        )


    with metric3:

        st.metric(
            "Model",
            "XGBoost"
        )


    st.progress(
        min(
            max(default_probability, 0.0),
            1.0
        )
    )


    st.caption(
        "Probability is generated by the calibrated "
        "CredWise AI credit-risk model."
    )


    with st.expander(
        "View Applicant Model Input"
    ):

        st.dataframe(
            applicant_data,
            use_container_width=True
        )

            # =====================================================
    # SHAP EXPLANATION FOR CURRENT APPLICANT
    # =====================================================

    # Get fitted preprocessing step from XGBoost pipeline
    fitted_preprocessor = (
        xgb_pipeline.named_steps["preprocessor"]
    )


    # Transform applicant exactly as during training
    applicant_processed = (
        fitted_preprocessor.transform(
            applicant_data
        )
    )


    # Convert sparse matrix to dense if necessary
    if hasattr(applicant_processed, "toarray"):
        applicant_processed = (
            applicant_processed.toarray()
        )


    # Generate SHAP explanation
    applicant_shap = shap_explainer(
        applicant_processed
    )


    # Get transformed feature names
    feature_names = (
        fitted_preprocessor
        .get_feature_names_out()
    )


    # Create SHAP contribution table
    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP_Value": applicant_shap.values[0]
    })


    shap_df["Impact"] = (
        shap_df["SHAP_Value"].abs()
    )


    shap_df = (
        shap_df
        .sort_values(
            "Impact",
            ascending=False
        )
        .reset_index(drop=True)
    )


    st.divider()

    st.header("Why This Risk Score?")

    st.caption(
        "SHAP explains which applicant features pushed "
        "the model toward higher or lower default risk."
    )


    # Top factors increasing model risk
    risk_factors_ui = (
        shap_df[
            shap_df["SHAP_Value"] > 0
        ]
        .head(5)
    )


    # Top factors reducing model risk
    protective_factors_ui = (
        shap_df[
            shap_df["SHAP_Value"] < 0
        ]
        .head(5)
    )


    col_risk, col_protective = st.columns(2)


    with col_risk:

        st.subheader("🔺 Risk-Raising Factors")

        for i in range(len(risk_factors_ui)):

            row = risk_factors_ui.iloc[i]

            feature = (
                row["Feature"]
                .replace("num__", "")
                .replace("cat__", "")
                .replace("_", " ")
                .title()
            )

            st.write(
                f"• {feature}"
            )


    with col_protective:

        st.subheader("🔻 Risk-Reducing Factors")

        for i in range(
            len(protective_factors_ui)
        ):

            row = protective_factors_ui.iloc[i]

            feature = (
                row["Feature"]
                .replace("num__", "")
                .replace("cat__", "")
                .replace("_", " ")
                .title()
            )

            st.write(
                f"• {feature}"
            )

                # =====================================================
    # POLICY RAG - RETRIEVE RELEVANT LENDING POLICY
    # =====================================================

    risk_feature_names = risk_factors_ui["Feature"].tolist()

    readable_risk_features = []

    for feature in risk_feature_names:

        clean_feature = (
            feature
            .replace("num__", "")
            .replace("cat__", "")
            .replace("_", " ")
        )

        readable_risk_features.append(clean_feature)


    # Build retrieval query
    rag_query = (
        "Lending policy related to borrower credit risk: "
        + ", ".join(readable_risk_features)
    )


    # Retrieve relevant policy chunks
    policy_docs = policy_vector_store.similarity_search(
        rag_query,
        k=2
    )


    # Combine policy context for Llama 3.2 later
    policy_context = "\n\n".join(
        [
            doc.page_content
            for doc in policy_docs
        ]
    )


    # =====================================================
    # DISPLAY RETRIEVED POLICY
    # =====================================================

    st.divider()

    st.header("📚 Relevant Lending Policy")

    st.caption(
        "Policy evidence retrieved from the CredWise AI"
        "lending-policy knowledge base."
    )


    for i in range(len(policy_docs)):

        with st.expander(
            f"Policy Evidence {i + 1}"
        ):

            st.write(
                policy_docs[i].page_content
            )

                # =====================================================
    # LLAMA 3.2 - GROUNDED CREDIT RISK REPORT
    # =====================================================

    # Prepare SHAP factors for the LLM
    raising_factors_text = ", ".join(
        risk_factors_ui["Feature"].tolist()
    )

    reducing_factors_text = ", ".join(
        protective_factors_ui["Feature"].tolist()
    )


    # Build grounded prompt
    report_prompt = f"""
You are CredWise AI, an explainable AI credit-risk decision-support system.

Generate a concise professional credit-risk assessment using ONLY the
information provided below.

MODEL ASSESSMENT
Probability of Default: {default_probability * 100:.1f}%
Risk Level: {risk_level}
Model: XGBoost

SHAP EXPLANATION
Risk-Raising Factors:
{raising_factors_text}

Risk-Reducing Factors:
{reducing_factors_text}

RETRIEVED LENDING POLICY
{policy_context}

INSTRUCTIONS:
1. Explain the predicted credit risk in simple professional language.
2. Mention the most important risk-raising factors.
3. Mention relevant risk-reducing factors.
4. Connect the assessment to the retrieved lending-policy evidence.
5. Do not invent facts, policies, applicant information, or numerical values.
6. Do not automatically approve or reject the loan.
7. State that the output is decision support and requires human review.
8. Keep the report concise.
9. SHAP indicates feature contribution, not causation.
10. Do not infer or invent the meaning, cause, or financial implication of a SHAP feature unless that interpretation is explicitly supported by the retrieved policy context.

Use these headings:
Risk Summary
Key Risk Drivers
Policy Context
Decision-Support Note
"""


    # Generate report with local Llama 3.2
    with st.spinner(
        "Generating grounded credit-risk report..."
    ):

        llm_response = llm.invoke(
            report_prompt
        )

        grounded_report = llm_response.content


    # =====================================================
    # DISPLAY GROUNDED REPORT
    # =====================================================

    st.divider()

    st.header("🤖 CredWise AI Credit Risk Report")

    st.caption(
        "Generated by Llama 3.2 using model predictions, "
        "SHAP explanations, and retrieved lending-policy evidence."
    )

    st.markdown(
        grounded_report
    )

    st.info(
        "Decision-support output only. "
        "Final lending decisions require appropriate human review."
    )