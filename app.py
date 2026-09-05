from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from config.settings import SETTINGS, FEATURES
from database import (
    init_db, save_application, save_prediction, save_shap_explanations,
    save_policy_evidence, save_report, save_human_review,
    get_application_history, get_dashboard_stats, get_connection, get_global_shap_importance,
)
from models.risk_engine import RiskEngine
from explainability.shap_service import clean_feature_name, ExplainabilityService
from agents.risk_agent import RiskAgent
from agents.policy_agent import PolicyAgent
from agents.decision_agent import DecisionIntelligenceAgent
from agents.supervisor import SupervisorAgent, build_langgraph_workflow, LANGGRAPH_AVAILABLE
from analytics.sql_agent import SQLAgent

init_db()

st.set_page_config(page_title="CredWise AI", page_icon="🏦", layout="wide")


def load_risk_engine():
    return RiskEngine()


def load_policy_service():
    try:
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name=SETTINGS.embedding_model)
        vector_store = FAISS.load_local(str(SETTINGS.faiss_path), embeddings, allow_dangerous_deserialization=True)
        from rag.policy_service import PolicyRAGService
        return PolicyRAGService(vector_store)
    except Exception:
        return None


def load_llm():
    try:
        from langchain_ollama import ChatOllama
        return ChatOllama(model=SETTINGS.llm_model, temperature=SETTINGS.llm_temperature)
    except Exception:
        return None


@st.cache_resource
def get_services():
    risk_engine = load_risk_engine()
    policy_service = load_policy_service()
    llm = load_llm()
    risk_agent = RiskAgent(risk_engine)
    policy_agent = PolicyAgent(policy_service) if policy_service else None
    decision_agent = DecisionIntelligenceAgent(llm)
    sql_agent = SQLAgent(llm)
    supervisor = SupervisorAgent(risk_agent, policy_agent, sql_agent, decision_agent)
    workflow = build_langgraph_workflow(supervisor) if policy_agent else supervisor
    return risk_engine, policy_agent, decision_agent, sql_agent, supervisor, workflow, llm


try:
    risk_engine, policy_agent, decision_agent, sql_agent, supervisor, workflow, llm = get_services()
except Exception as exc:
    st.error(f"CredWise initialization failed: {exc}")
    st.stop()

# ------------------------------ Sidebar ------------------------------
with st.sidebar:
    st.title("🏦 CredWise AI")
    st.caption("Agentic Banking Decision Intelligence & Explainable Risk Platform")
    st.markdown("### Architecture")
    st.write("✓ XGBoost + calibrated PD")
    st.write("✓ SHAP explainability")
    st.write("✓ Policy RAG + evidence IDs")
    st.write("✓ Read-only SQL analytics")
    st.write("✓ LangGraph supervisor" if LANGGRAPH_AVAILABLE else "○ LangGraph not installed; safe router fallback")
    st.write("✓ Human-in-the-loop")
    st.write("✓ Agent audit trail")
    if llm is None:
        st.warning("Ollama/Llama is unavailable. Deterministic model, SQL templates, and evidence retrieval can still operate where supported.")
    if policy_agent is None:
        st.error("Policy RAG is unavailable. Install RAG dependencies and ensure the local FAISS index is present.")
    st.divider()
    st.warning("Educational/research decision-support prototype. It does not automatically approve or reject loans and is not production lending software.")

st.title("🏦 CredWise AI")
st.subheader("Agentic Banking Decision Intelligence & Explainable Risk Platform")
st.caption("Existing ML + SHAP + policy RAG + SQLite foundation, extended with controlled agentic analytics, provenance, auditability, data-quality checks, and human review.")

# ------------------------------ Dashboard ------------------------------
st.header("📊 Banking Risk Overview")
stats = get_dashboard_stats()
a, b, c, d = st.columns(4)
a.metric("Applications", stats["total_applications"])
b.metric("High Risk", stats["high_risk"])
c.metric("Human Reviews", stats["total_reviews"])
d.metric("Avg. Default Probability", f"{stats['average_probability']:.1%}")

with st.expander("Recent Assessments"):
    history = get_application_history(20)
    if history.empty:
        st.info("No assessments stored yet.")
    else:
        view = history.copy()
        view["default_probability"] = view["default_probability"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "-")
        st.dataframe(view, use_container_width=True)

with st.expander("📈 Empirical SHAP Importance from Stored Assessments"):
    global_shap = get_global_shap_importance(15)
    if global_shap.empty:
        st.info("Global importance will appear after applicant assessments have been stored.")
    else:
        st.dataframe(global_shap, use_container_width=True)
        st.caption("This view aggregates absolute SHAP impact from stored assessments; it is not a replacement for the model artifact's training-time global explanation.")

st.divider()

# ------------------------------ Tabs ------------------------------
tab_risk, tab_analytics, tab_audit = st.tabs(["🔍 Applicant Risk", "💬 Analyst Intelligence", "🧾 Audit History"])

with tab_risk:
    st.header("Applicant & Loan Information")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### 💰 Loan Details")
        loan_amnt = st.number_input("Loan Amount ($)", 500.0, 50000.0, 10000.0, 500.0)
        term = st.selectbox("Loan Term", [36, 60], format_func=lambda x: f"{x} months")
        installment = st.number_input("Monthly Installment ($)", 0.0, value=300.0, step=10.0)
        purpose = st.selectbox("Loan Purpose", ["debt_consolidation", "credit_card", "home_improvement", "major_purchase", "small_business", "medical", "car", "moving", "house", "vacation", "wedding", "renewable_energy", "educational", "other"])
    with c2:
        st.markdown("### 📊 Financial Profile")
        annual_inc = st.number_input("Annual Income ($)", 0.0, value=60000.0, step=1000.0)
        dti = st.number_input("Debt-to-Income Ratio", 0.0, value=15.0, step=0.5)
        revol_bal = st.number_input("Revolving Credit Balance ($)", 0.0, value=10000.0, step=500.0)
        revol_util = st.number_input("Revolving Credit Utilization (%)", 0.0, 150.0, 40.0, 1.0)
        verification_status = st.selectbox("Income Verification", ["Verified", "Source Verified", "Not Verified"])
    with c3:
        st.markdown("### 💳 Credit Profile")
        emp_length = st.slider("Employment Length (Years)", 0, 10, 5)
        home_ownership = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER", "NONE"])
        delinq_2yrs = st.number_input("Delinquencies — Last 2 Years", 0, value=0, step=1)
        inq_last_6mths = st.number_input("Credit Inquiries — Last 6 Months", 0, value=1, step=1)
        open_acc = st.number_input("Open Credit Accounts", 0, value=10, step=1)
        pub_rec = st.number_input("Derogatory Public Records", 0, value=0, step=1)
        total_acc = st.number_input("Total Credit Accounts", 0, value=20, step=1)

    applicant = {
        "loan_amnt": loan_amnt, "term": term, "installment": installment, "emp_length": emp_length,
        "home_ownership": home_ownership, "annual_inc": annual_inc, "verification_status": verification_status,
        "purpose": purpose, "dti": dti, "delinq_2yrs": delinq_2yrs, "inq_last_6mths": inq_last_6mths,
        "open_acc": open_acc, "pub_rec": pub_rec, "revol_bal": revol_bal, "revol_util": revol_util, "total_acc": total_acc,
    }

    analyze = st.button("🔍 Analyze Credit Risk", type="primary", use_container_width=True)
    if analyze:
        try:
            with st.spinner("Running controlled risk assessment..."):
                application_id = save_application(applicant)
                risk_result = risk_engine.predict(applicant)
                prediction_id = save_prediction(application_id, risk_result.raw_risk_score, risk_result.default_probability, risk_result.risk_level)
                shap_service = ExplainabilityService(risk_engine)
                explanation = shap_service.explain(applicant, top_n=5)
                save_shap_explanations(prediction_id, explanation["all_factors"])
                policy_query = "Lending policy related to borrower credit risk: " + ", ".join(clean_feature_name(x) for x in explanation["risk_raising"]["Feature"].tolist())
                try:
                    policy = policy_agent.run(policy_query, top_k=SETTINGS.policy_top_k) if policy_agent else {"evidence": [], "context": ""}
                except Exception as policy_exc:
                    policy = {"evidence": [], "context": "", "error": str(policy_exc)}
                save_policy_evidence(prediction_id, policy["evidence"])
                analytics_context = get_dashboard_stats()
                report = decision_agent.run(risk_result.to_dict(), explanation["risk_raising"].to_dict(orient="records"), explanation["risk_reducing"].to_dict(orient="records"), policy["evidence"], analytics_context)
                save_report(prediction_id, report)
                st.session_state["last_applicant"] = applicant
                st.session_state["last_application_id"] = application_id
                st.session_state["last_prediction_id"] = prediction_id
                st.session_state["last_risk"] = risk_result.to_dict()
                st.session_state["last_explanation"] = explanation
                st.session_state["last_policy"] = policy
                st.session_state["last_report"] = report
        except Exception as exc:
            st.error(f"Assessment failed: {exc}")

    if "last_risk" in st.session_state:
        risk = st.session_state["last_risk"]
        explanation = st.session_state["last_explanation"]
        policy = st.session_state.get("last_policy", {"evidence": []})
        application_id = st.session_state["last_application_id"]
        prediction_id = st.session_state["last_prediction_id"]
        st.divider(); st.header("Credit Risk Assessment")
        m1, m2, m3 = st.columns(3)
        m1.metric("Probability of Default", f"{risk['default_probability']:.1%}")
        m2.metric("Risk Level", risk["risk_level"])
        m3.metric("Model", risk["model_name"])
        st.progress(min(max(risk["default_probability"], 0.0), 1.0))
        st.caption("The probability is produced by the existing calibrated XGBoost model. The LLM cannot change it.")
        if risk.get("data_quality_warnings"):
            st.warning("Data-quality warnings: " + " | ".join(risk["data_quality_warnings"]))

        st.subheader("🔎 SHAP Explanation")
        rcol, pcol = st.columns(2)
        with rcol:
            st.markdown("**Risk-Raising Factors**")
            for row in explanation["risk_raising"].to_dict(orient="records"):
                st.write(f"▲ {clean_feature_name(row['Feature'])} — SHAP {row['SHAP_Value']:+.4f} — value: {row['Feature_Value']}")
        with pcol:
            st.markdown("**Risk-Reducing Factors**")
            for row in explanation["risk_reducing"].to_dict(orient="records"):
                st.write(f"▼ {clean_feature_name(row['Feature'])} — SHAP {row['SHAP_Value']:+.4f} — value: {row['Feature_Value']}")
        st.caption("SHAP values describe model contribution, not causation.")

        st.subheader("📚 Policy Evidence")
        if policy["evidence"]:
            for item in policy["evidence"]:
                with st.expander(f"[{item['evidence_id']}] {item['section']}"):
                    st.caption(f"Source: {item['source']} | Retrieval score: {item.get('score')}")
                    st.write(item["content"])
        else:
            st.warning("No policy evidence was retrieved; policy claims should not be generated.")

        st.subheader("🤖 Decision-Intelligence Report")
        st.markdown(st.session_state["last_report"])
        st.info("Decision-support only. Human review is required for any final lending judgment.")

        st.subheader("👤 Human Review")
        rd1, rd2 = st.columns([1, 2])
        with rd1:
            decision = st.selectbox("Reviewer Decision", ["Approve Recommendation", "Reject Recommendation", "Further Review"], key=f"review_{application_id}")
        with rd2:
            comment = st.text_area("Reviewer Comment", key=f"comment_{application_id}")
        if st.button("💾 Save Human Review", key=f"save_{application_id}"):
            review_id = save_human_review(application_id, decision, comment)
            st.success(f"Human review #{review_id} saved for Application #{application_id}.")

with tab_analytics:
    st.header("💬 Natural-Language Analyst")
    st.caption("Ask for structured analytics, applicant risk, or policy evidence. SQL is validated as read-only before execution.")
    st.info('Examples: “How many high-risk applicants are present?” · “What is the average predicted default probability?” · “Which applicants have the highest predicted default probability?” · “What policy applies to DTI?” · “Why is this applicant risky?”')
    query = st.text_area("Analyst question", placeholder="Ask a banking risk analytics question...")
    if st.button("Run Analyst Query", type="primary") and query.strip():
        applicant_context = st.session_state.get("last_applicant")
        with st.spinner("Routing analyst request..."):
            result = workflow.invoke({"query": query, "applicant": applicant_context}) if LANGGRAPH_AVAILABLE and hasattr(workflow, "invoke") else supervisor.run(query, applicant_context)
        if result.get("error"):
            st.error(result["error"])
        else:
            st.success(f"Route: {result.get('route', supervisor.route(query))}")
            st.caption(f"Trace: {result.get('trace_id', 'n/a')} · Confidence: {result.get('route_confidence', 0):.0%} · {result.get('route_reason', '')}")
            if result.get("analytics"):
                analytics = result["analytics"]
                if analytics.get("sql"):
                    st.code(analytics["sql"], language="sql")
                if analytics.get("rows"):
                    st.dataframe(pd.DataFrame(analytics["rows"]), use_container_width=True)
                if analytics.get("validation_notes"):
                    st.caption("SQL safeguards: " + " · ".join(analytics["validation_notes"]))
                else:
                    st.info("Query executed successfully but returned no rows.")
            if result.get("policy"):
                for item in result["policy"].get("evidence", []):
                    with st.expander(f"[{item['evidence_id']}] {item['section']}"):
                        st.caption(f"Source: {item['source']}")
                        st.write(item["content"])
            if result.get("risk"):
                st.json(result["risk"]["prediction"])
                if result["risk"]["prediction"].get("data_quality_warnings"):
                    st.warning("Data-quality warnings: " + " | ".join(result["risk"]["prediction"]["data_quality_warnings"]))
                st.write("Risk-raising factors")
                st.dataframe(pd.DataFrame(result["risk"]["risk_raising"]), use_container_width=True)
            if result.get("report"):
                st.markdown(result["report"])

with tab_audit:
    st.header("🧾 Agent & Assessment Audit")
    conn = get_connection()
    audit = pd.read_sql_query("SELECT event_id, trace_id, agent_name, action, status, input_summary, output_summary, error_message, created_at FROM agent_audit_log ORDER BY event_id DESC LIMIT 100", conn)
    queries = pd.read_sql_query("SELECT query_id, query_text, route, generated_sql, execution_status, error_message, created_at FROM analyst_queries ORDER BY query_id DESC LIMIT 50", conn)
    conn.close()
    st.subheader("Agent Events")
    st.dataframe(audit, use_container_width=True)
    st.subheader("Analyst Queries")
    st.dataframe(queries, use_container_width=True)
