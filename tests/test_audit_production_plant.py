from scripts.audit_production_plant import build_audit_query


def test_build_audit_query_filters_one_plant_and_is_bounded():
    queries = build_audit_query(plant_id=234, timezone="Asia/Shanghai", sample_size=20)
    summary = queries["summary"]
    sample = queries["devices"]
    assert summary["query"] == {"term": {"plant_id": 234}}
    assert summary["size"] == 0
    assert summary["track_total_hits"] is True
    assert sample["size"] == 20
    assert sample["_source"]
    assert sample["query"] == summary["query"]
    assert "missing_string_current_01" in summary["aggs"]["missing_string_currents"]["filters"]["filters"]

