from database import get_global_shap_importance, get_readonly_connection


def test_readonly_connection_is_query_only():
    conn = get_readonly_connection()
    value = conn.execute("PRAGMA query_only").fetchone()[0]
    conn.close()
    assert value == 1


def test_global_shap_query_returns_dataframe():
    df = get_global_shap_importance()
    assert list(df.columns) == ["feature_name", "mean_abs_shap", "observations"]
