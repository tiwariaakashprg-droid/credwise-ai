from __future__ import annotations

import uuid
from typing import TypedDict

from database import save_agent_event

try:
    from langgraph.graph import END, START, StateGraph
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False


class AgentState(TypedDict, total=False):
    trace_id: str
    query: str
    applicant: dict
    route: str
    route_confidence: float
    route_reason: str
    risk: dict
    policy: dict
    analytics: dict
    report: str
    decision_package: dict
    error: str


class SupervisorAgent:
    name = "supervisor"

    def __init__(self, risk_agent=None, policy_agent=None, sql_agent=None, decision_agent=None):
        self.risk_agent = risk_agent
        self.policy_agent = policy_agent
        self.sql_agent = sql_agent
        self.decision_agent = decision_agent

    def route_details(self, query: str) -> tuple[str, float, str]:
        q = query.lower()
        if any(k in q for k in ["summarize", "summary", "reviewer", "decision support", "decision-intelligence"]):
            return "decision", 0.95, "The query requests a case summary or decision-support report."
        if any(k in q for k in ["how many", "average", "distribution", "segment", "portfolio", "compare", "highest", "lowest", "by loan", "by income"]):
            return "sql", 0.90, "The query asks for aggregate, comparative, portfolio, or database analytics."
        if any(k in q for k in ["policy", "rule", "guideline", "dti", "utilization", "inquiries", "delinquenc", "employment history", "loan term"]):
            return "policy", 0.90, "The query asks for policy or lending-rule evidence."
        if any(k in q for k in ["why", "risk factor", "shap", "default probability", "risk", "applicant", "borrower"]):
            return "risk", 0.85, "The query asks about applicant-level risk or model explanation."
        return "decision", 0.55, "No specialized intent matched; using the controlled decision-support fallback."

    def route(self, query: str) -> str:
        return self.route_details(query)[0]

    def run(self, query: str, applicant: dict | None = None) -> AgentState:
        trace_id = str(uuid.uuid4())
        route, confidence, reason = self.route_details(query)
        state: AgentState = {"trace_id": trace_id, "query": query, "applicant": applicant or {}, "route": route, "route_confidence": confidence, "route_reason": reason}
        save_agent_event(trace_id, self.name, "route_query", "success", query, f"route={route}; confidence={confidence:.2f}; reason={reason}")
        try:
            if route == "sql":
                result = self.sql_agent.run(query)
                state["analytics"] = {"status": result.status, "sql": result.sql, "rows": result.dataframe.to_dict(orient="records"), "error": result.error, "validation_notes": result.validation_notes or []}
                save_agent_event(trace_id, "sql_agent", "execute", result.status, query, str(state["analytics"])[:4000], result.error)
            elif route == "policy":
                result = self.policy_agent.run(query)
                state["policy"] = result
                save_agent_event(trace_id, "policy_agent", "retrieve", "success" if result.get("evidence") else "no_evidence", query, str(result)[:4000])
            elif route == "risk":
                if not applicant:
                    raise ValueError("An applicant is required for applicant-level risk questions.")
                state["risk"] = self.risk_agent.run(applicant)
                save_agent_event(trace_id, "risk_agent", "predict_and_explain", "success", "applicant", str(state["risk"])[:4000])
            else:
                if not applicant:
                    raise ValueError("An applicant is required for decision-intelligence questions.")
                state["risk"] = self.risk_agent.run(applicant)
                features = ", ".join(x["Feature"] for x in state["risk"]["risk_raising"][:5])
                state["policy"] = self.policy_agent.run(f"Lending policy related to {features}")
                if hasattr(self.decision_agent, "build_package"):
                    state["decision_package"] = self.decision_agent.build_package(state["risk"]["prediction"], state["risk"]["risk_raising"], state["risk"]["risk_reducing"], state["policy"]["evidence"], state.get("analytics")).to_dict()
                state["report"] = self.decision_agent.run(state["risk"]["prediction"], state["risk"]["risk_raising"], state["risk"]["risk_reducing"], state["policy"]["evidence"], state.get("analytics"))
                save_agent_event(trace_id, "decision_agent", "generate_report", "success", "evidence bundle", state["report"][:4000])
            return state
        except Exception as exc:
            state["error"] = str(exc)
            save_agent_event(trace_id, route, "execution", "error", query, error_message=str(exc))
            return state


def build_langgraph_workflow(supervisor: SupervisorAgent):
    if not LANGGRAPH_AVAILABLE:
        return supervisor
    graph = StateGraph(AgentState)

    def route_node(state):
        if not state.get("trace_id"):
            state["trace_id"] = str(uuid.uuid4())
        route, confidence, reason = supervisor.route_details(state["query"])
        state["route"], state["route_confidence"], state["route_reason"] = route, confidence, reason
        save_agent_event(state["trace_id"], "supervisor", "route", "success", state["query"], f"{route} ({confidence:.2f})")
        return state

    def sql_node(state):
        result = supervisor.sql_agent.run(state["query"])
        state["analytics"] = {"status": result.status, "sql": result.sql, "rows": result.dataframe.to_dict(orient="records"), "error": result.error, "validation_notes": result.validation_notes or []}
        save_agent_event(state["trace_id"], "sql_agent", "execute", result.status, state["query"], str(state["analytics"])[:4000], result.error)
        return state

    def risk_node(state):
        state["risk"] = supervisor.risk_agent.run(state["applicant"])
        save_agent_event(state["trace_id"], "risk_agent", "predict_and_explain", "success", "applicant", str(state["risk"])[:4000])
        return state

    def policy_node(state):
        state["policy"] = supervisor.policy_agent.run(state["query"])
        save_agent_event(state["trace_id"], "policy_agent", "retrieve", "success" if state["policy"].get("evidence") else "no_evidence", state["query"], str(state["policy"])[:4000])
        return state

    def decision_node(state):
        if "risk" not in state:
            state["risk"] = supervisor.risk_agent.run(state["applicant"])
        if "policy" not in state:
            features = ", ".join(x["Feature"] for x in state["risk"]["risk_raising"][:5])
            state["policy"] = supervisor.policy_agent.run(f"Lending policy related to {features}")
        state["decision_package"] = supervisor.decision_agent.build_package(state["risk"]["prediction"], state["risk"]["risk_raising"], state["risk"]["risk_reducing"], state["policy"]["evidence"], state.get("analytics")).to_dict()
        state["report"] = supervisor.decision_agent.run(state["risk"]["prediction"], state["risk"]["risk_raising"], state["risk"]["risk_reducing"], state["policy"]["evidence"], state.get("analytics"))
        save_agent_event(state["trace_id"], "decision_agent", "generate_report", "success", "evidence bundle", state["report"][:4000])
        return state

    graph.add_node("route", route_node); graph.add_node("sql", sql_node); graph.add_node("risk", risk_node); graph.add_node("policy", policy_node); graph.add_node("decision", decision_node)
    graph.add_edge(START, "route")
    graph.add_conditional_edges("route", lambda s: s["route"], {"sql": "sql", "risk": "risk", "policy": "policy", "decision": "decision"})
    graph.add_edge("sql", END); graph.add_edge("risk", "decision"); graph.add_edge("policy", END); graph.add_edge("decision", END)
    return graph.compile()
