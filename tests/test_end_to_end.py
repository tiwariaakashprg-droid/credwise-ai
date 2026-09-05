from agents.supervisor import SupervisorAgent

class FakeRisk:
    def run(self, applicant):
        return {"prediction": {"default_probability": .2, "risk_level": "Moderate Risk", "model_name":"XGBoost"}, "risk_raising": [], "risk_reducing": [], "all_shap": []}
class FakePolicy:
    def run(self, query, top_k=3):
        return {"evidence": [{"evidence_id":"policy-1","section":"TEST","content":"test","source":"test"}], "context":"test"}
class FakeSQL:
    def run(self, query):
        class R:
            status='success'; sql='SELECT 1'; error=None
            class DF:
                def to_dict(self, orient='records'): return [{'value':1}]
            dataframe=DF()
        return R()
class FakeDecision:
    def run(self,*args,**kwargs): return 'report'

def test_decision_workflow():
    s=SupervisorAgent(FakeRisk(),FakePolicy(),FakeSQL(),FakeDecision())
    result=s.run('Summarize this applicant', {'x':1})
    assert result['report']=='report'
