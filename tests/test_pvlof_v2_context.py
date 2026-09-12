import pandas as pd

from scripts.extract_pvlof_v2_alarm_context import extract_context


def test_extract_context_includes_peer_and_warmup_rows(tmp_path):
    production = tmp_path / "production" / "plant_id=234" / "date=2026-06-01"
    production.mkdir(parents=True)
    rows = []
    for timestamp in pd.date_range("2026-06-01 00:00", periods=3, freq="5min", tz="UTC"):
        for device in ["a", "b"]:
            rows.append({
                "event_time": timestamp,
                "plant_id": 234,
                "device_no": device,
                "main_string_count": 2,
                "string_current_01": 10.0,
                "string_current_02": 10.0,
            })
    pd.DataFrame(rows).to_parquet(production / "part-00000.parquet", index=False)
    alarm_path = tmp_path / "alarm_points.parquet"
    pd.DataFrame([{
        "alarm_event_id": "e1",
        "plant_id": 234,
        "device_no": "a",
        "event_time": pd.Timestamp("2026-06-01 00:05", tz="UTC"),
    }]).to_parquet(alarm_path, index=False)

    report = extract_context(
        production,
        alarm_path,
        tmp_path / "output",
        warmup_minutes=5,
    )
    context = pd.read_parquet(tmp_path / "output" / "context_currents.parquet")
    assert report["target_rows"] == 1
    assert len(context) == 4
    assert context["is_alarm_target"].sum() == 1
    assert context["is_warmup"].sum() == 2

