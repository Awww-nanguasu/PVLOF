import sys

import pandas as pd

from scripts.compare_pvlof_v13_v14 import main


def test_compare_v13_v14_exports_point_and_event_tables(tmp_path, monkeypatch):
    common = {
        "plant_id": [33, 33],
        "device_no": ["dev-a", "dev-a"],
        "event_time": pd.to_datetime(
            ["2026-08-01T00:00:00Z", "2026-08-01T00:05:00Z"]
        ),
        "string_no": pd.Series([1, 2], dtype="Int64"),
    }
    v13 = pd.DataFrame(
        {
            **common,
            "pvlof_v12_combined_alert": pd.Series([1, 1], dtype="Int64"),
            "string_current": [9.7, 8.0],
            "expected_current": [10.0, 10.0],
            "isolated_relative_drop": [0.12, 0.20],
        }
    )
    v14 = pd.DataFrame(
        {**common, "pvlof_v12_combined_alert": pd.Series([0, 1], dtype="Int64")}
    )
    v13_path = tmp_path / "v13.parquet"
    v14_path = tmp_path / "v14.parquet"
    output = tmp_path / "comparison"
    v13.to_parquet(v13_path, index=False)
    v14.to_parquet(v14_path, index=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_pvlof_v13_v14.py",
            "--v1-3", str(v13_path),
            "--v1-4", str(v14_path),
            "--output-directory", str(output),
        ],
    )

    main()

    points = pd.read_csv(output / "pvlof_v13_v14_dual_gate_points.csv")
    events = pd.read_csv(output / "pvlof_v13_v14_dual_gate_events.csv")
    filtered = pd.read_csv(output / "pvlof_v13_only_filtered_strings.csv")
    assert points["comparison_case"].tolist() == ["v1_3_only", "both_same"]
    assert events.loc[0, "PVLOF_V1_3_EFFECT_GATE"] == "01,02"
    assert events.loc[0, "PVLOF_V1_4_DUAL_GATE"] == 2
    assert events.loc[0, "comparison_case"] == "both_different"
    assert len(filtered) == 1
    assert abs(filtered.loc[0, "isolated_absolute_drop"] - 0.3) < 1e-6
