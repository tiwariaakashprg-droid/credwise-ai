from agents.supervisor import SupervisorAgent


def test_routing():
    s = SupervisorAgent()
    assert s.route("How many high-risk applicants are present?") == "sql"
    assert s.route("What policy applies to DTI?") == "policy"
    assert s.route("Why is this applicant risky?") == "risk"
    assert s.route("Summarize this applicant for a reviewer") == "decision"


def test_route_confidence_and_reason():
    route, confidence, reason = SupervisorAgent().route_details("What policy applies to DTI?")
    assert route == "policy"
    assert confidence > 0.0
    assert reason
