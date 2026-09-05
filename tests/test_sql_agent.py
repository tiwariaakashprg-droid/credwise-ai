from analytics.sql_agent import SQLAgent


def test_rejects_write_sql():
    agent = SQLAgent()
    try:
        agent.validate_sql("DELETE FROM risk_predictions")
    except ValueError:
        return
    raise AssertionError("destructive SQL should be rejected")


def test_accepts_read_query():
    notes = SQLAgent().validate_sql("SELECT COUNT(*) FROM risk_predictions")
    assert "SELECT/WITH only" in notes
