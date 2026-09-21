import sys

import pandas as pd

from scripts.compare_pvlof_v14_v15 import main


def test_compare_v14_v15_classifies_memory_addition(tmp_path, monkeypatch):
    common = {
        "plant_id": [234, 234],
        "device_no": ["dev-a", "dev-a"],
        "event_time": pd.to_datetime(["2026-06-01T00:00Z", "2026-06-01T00:05Z"]),
        "string_no": pd.Series([1, 1], dtype="Int64"),
    }
    v14 = pd.DataFrame({
        **common,
        "pvlof_v2_hier_strict_alert": pd.Series([1, 0], dtype="Int64"),
    })
    v15 = pd.DataFrame({
        **common,
        "pvlof_v15_alert": pd.Series([1, 1], dtype="Int64"),
        "pvlof_v15_memory_reactivated_alert": pd.Series([0, 1], dtype="Int64"),
        "pvlof_v2_hier_strict_alert": pd.Series([1, 0], dtype="Int64"),
        "pvlof_v15_raw_anomaly": pd.Series([1, 1], dtype="Int64"),
        "string_current": [8.0, 8.0],
        "expected_current": [10.0, 10.0],
    })
    v14_path = tmp_path / "v14.parquet"
    v15_path = tmp_path / "v15.parquet"
    output = tmp_path / "comparison"
    v14.to_parquet(v14_path, index=False)
    v15.to_parquet(v15_path, index=False)
    monkeypatch.setattr(sys, "argv", [
        "compare_pvlof_v14_v15.py",
        "--v1-4", str(v14_path),
        "--v1-5", str(v15_path),
        "--output-directory", str(output),
    ])
    main()
    additions = pd.read_csv(output / "pvlof_v15_additions.csv")
    assert len(additions) == 1
    assert additions.loc[0, "addition_reason"] == "memory_reactivation"


def test_customer_events_backfill_candidates_and_split_memory_reactivation(
    tmp_path, monkeypatch
):
    times = pd.date_range("2026-06-01T00:00Z", periods=8, freq="5min")
    common = {
        "plant_id": [234] * 8,
        "device_no": ["dev-a"] * 8,
        "event_time": times,
        "string_no": pd.Series([1] * 8, dtype="Int64"),
    }
    v14 = pd.DataFrame({
        **common,
        "isolated_directional_raw_alert": [1, 1, 1, 1, 0, 1, 0, 0],
        "isolated_directional_alert": [0, 0, 1, 1, 0, 0, 0, 0],
        "isolated_directional_consecutive": [1, 2, 3, 4, 0, 1, 0, 0],
        "pvlof_v2_hier_strict_alert": [0, 0, 1, 1, 0, 0, 0, 0],
    })
    v15 = pd.DataFrame({
        **common,
        "pvlof_v15_raw_anomaly": [1, 1, 1, 1, 0, 1, 0, 0],
        "pvlof_v15_alert": [0, 0, 1, 1, 0, 1, 0, 0],
        "pvlof_v15_memory_active": [0, 0, 1, 1, 1, 1, 1, 0],
        "pvlof_v15_memory_reactivated_alert": [0, 0, 0, 0, 0, 1, 0, 0],
        "pvlof_v15_entry_streak": [1, 2, 3, 4, 4, 4, 4, 0],
        "pvlof_v15_normal_streak": [0, 0, 0, 0, 1, 0, 1, 0],
        "pvlof_v15_memory_clear_code": [0, 0, 0, 0, 0, 0, 0, 1],
        "pvlof_v2_hier_strict_alert": [0, 0, 1, 1, 0, 0, 0, 0],
    })
    v14_path = tmp_path / "v14.parquet"
    v15_path = tmp_path / "v15.parquet"
    output = tmp_path / "comparison"
    v14.to_parquet(v14_path, index=False)
    v15.to_parquet(v15_path, index=False)
    monkeypatch.setattr(sys, "argv", [
        "compare_pvlof_v14_v15.py",
        "--v1-4", str(v14_path),
        "--v1-5", str(v15_path),
        "--output-directory", str(output),
        "--customer-only",
    ])
    main()

    customer = pd.read_csv(output / "pvlof_v14_v15_customer_events.csv")
    assert list(customer.columns) == [
        "row", "event_id", "plant_id", "device_no", "raise_time_local",
        "end_time_local", "PVLOF_V1_4_DUAL_GATE",
        "PVLOF_V1_5_MEMORY_5PCT", "comparison_case", "alert_time_points",
        "duration_minutes",
    ]
    assert len(customer) == 2
    assert customer.loc[0, "raise_time_local"] == "2026-06-01 08:00:00"
    assert customer.loc[0, "end_time_local"] == "2026-06-01 08:15:00"
    assert customer.loc[0, "alert_time_points"] == 4
    assert customer.loc[0, "duration_minutes"] == 20
    assert customer.loc[1, "raise_time_local"] == "2026-06-01 08:25:00"
    assert customer.loc[1, "end_time_local"] == "2026-06-01 08:25:00"
    assert pd.isna(customer.loc[1, "PVLOF_V1_4_DUAL_GATE"])
    assert customer.loc[1, "PVLOF_V1_5_MEMORY_5PCT"] == 1
    assert not (output / "pvlof_v14_v15_events.csv").exists()
