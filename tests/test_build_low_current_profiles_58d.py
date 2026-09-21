import json
import sys

import pandas as pd

from pv_anomaly.low_current_profiles import LowCurrentProfileConfig
from scripts.build_low_current_profiles_58d import main


def test_build_profile_pipeline_writes_review_outputs(tmp_path, monkeypatch):
    input_root = tmp_path / "replay"
    plant = input_root / "plant_id=234"
    plant.mkdir(parents=True)
    times = pd.date_range("2026-06-01 00:00", periods=3, freq="5min", tz="UTC")
    pd.DataFrame(
        {
            "plant_id": ["234"] * 3,
            "device_no": ["dev-a"] * 3,
            "event_time": times,
            "string_no": [1] * 3,
            "pvlof_v17_improved_raw_anomaly": [1, 1, 1],
            "pvlof_v17_improved_alert": [0, 0, 1],
            "pvlof_v17_improved_valid_point": [1, 1, 1],
            "pvlof_v17_improved_segmentation_raw_candidate": [0, 0, 0],
            "pvlof_v16_raw_anomaly": [1, 1, 1],
            "string_current": [5.0, 4.8, 4.9],
            "expected_current": [10.0, 10.0, 10.0],
            "residual_ratio": [0.50, 0.48, 0.49],
        }
    ).to_parquet(
        plant / "pvlof_v17_improved_v2_full_points.parquet",
        index=False,
    )
    config = LowCurrentProfileConfig(
        minimum_valid_points_per_day=1,
        minimum_valid_history_days=1,
        minimum_prior_alarm_days_long_term=1,
        batch_size=2,
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(config.to_dict(), ensure_ascii=False),
        encoding="utf-8",
    )
    output = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_low_current_profiles_58d.py",
            "--input-root",
            str(input_root),
            "--config",
            str(config_path),
            "--output-directory",
            str(output),
            "--skip-workbook",
        ],
    )

    main()

    events = pd.read_parquet(output / "low_current_string_events.parquet")
    assert len(events) == 1
    assert events.loc[0, "candidate_points"] == 3
    assert events.loc[0, "meets_three_point_rule"] == 1
    profiles = pd.read_parquet(output / "low_current_string_profiles.parquet")
    assert len(profiles) == 1
    assert profiles.loc[0, "alarm_days"] == 1
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["counts"]["candidate_string_points"] == 3
    assert summary["counts"]["string_events"] == 1
