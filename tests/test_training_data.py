from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd

from pv_anomaly.training_data import TrainingConfig, add_quality_and_targets


def config() -> TrainingConfig:
    return TrainingConfig(
        timezone_name="Asia/Shanghai",
        device_root=Path("unused"),
        weather_root=Path("unused"),
        aligned_output=Path("unused"),
        transformer_output=Path("unused"),
        audit_output=Path("unused"),
        train_end=date(2026, 6, 1),
        validation_end=date(2026, 7, 1),
        test_end=date(2026, 7, 23),
        running_status_code=1,
        interval_minutes=5,
        maximum_power_ratio=1.1,
        minimum_temperature=0,
        maximum_temperature=80,
        weather_tolerance_minutes=15,
    )


def test_candidate_requires_running_positive_continuous_next_step():
    frame = pd.DataFrame(
        {
            "event_time": pd.to_datetime(
                ["2026-05-01T00:00:00Z", "2026-05-01T00:05:00Z", "2026-05-01T00:10:00Z"],
                utc=True,
            ),
            "device_no": ["d1", "d1", "d1"],
            "active_power": [10.0, 11.0, 0.0],
            "dc_power": [10.5, 11.5, 0.0],
            "total_power": [10.0, 11.0, 0.0],
            "rated_power": [100.0, 100.0, 100.0],
            "device_temperature": [30.0, 31.0, 30.0],
            "status_code": [1, 1, 0],
        }
    )
    result = add_quality_and_targets(frame, config())
    assert result["candidate_normal"].tolist() == [True, False, False]
    assert result.loc[0, "target_active_power"] == 11.0


def test_cross_split_target_is_rejected():
    frame = pd.DataFrame(
        {
            "event_time": pd.to_datetime(
                ["2026-05-31T15:55:00Z", "2026-05-31T16:00:00Z"], utc=True
            ),
            "device_no": ["d1", "d1"],
            "active_power": [10.0, 11.0],
            "dc_power": [10.5, 11.5],
            "total_power": [10.0, 11.0],
            "rated_power": [100.0, 100.0],
            "device_temperature": [30.0, 31.0],
            "status_code": [1, 1],
        }
    )
    result = add_quality_and_targets(frame, replace(config()))
    assert result["candidate_normal"].tolist() == [False, False]


def test_targets_do_not_cross_plants_when_device_numbers_match():
    frame = pd.DataFrame(
        {
            "event_time": pd.to_datetime(
                [
                    "2026-05-01T00:00:00Z",
                    "2026-05-01T00:05:00Z",
                    "2026-05-01T00:00:00Z",
                    "2026-05-01T00:05:00Z",
                ],
                utc=True,
            ),
            "plant_id": [234, 234, 892, 892],
            "device_no": ["same", "same", "same", "same"],
            "active_power": [10.0, 11.0, 20.0, 21.0],
            "dc_power": [10.5, 11.5, 20.5, 21.5],
            "total_power": [10.0, 11.0, 20.0, 21.0],
            "rated_power": [100.0] * 4,
            "device_temperature": [30.0] * 4,
            "status_code": [1] * 4,
        }
    )

    result = add_quality_and_targets(frame, config())

    assert sorted(result["device_key"].unique()) == ["234::same", "892::same"]
    targets = result.dropna(subset=["target_active_power"]).set_index("device_key")
    assert targets.loc["234::same", "target_active_power"] == 11.0
    assert targets.loc["892::same", "target_active_power"] == 21.0


def test_missing_optional_total_power_does_not_reject_candidate():
    frame = pd.DataFrame(
        {
            "event_time": pd.to_datetime(
                ["2026-05-01T00:00:00Z", "2026-05-01T00:05:00Z"],
                utc=True,
            ),
            "plant_id": [892, 892],
            "device_no": ["d1", "d1"],
            "active_power": [10.0, 11.0],
            "dc_power": [10.5, 11.5],
            "total_power": [float("nan"), float("nan")],
            "rated_power": [100.0, 100.0],
            "device_temperature": [30.0, 31.0],
            "status_code": [1, 1],
        }
    )

    result = add_quality_and_targets(frame, config())

    assert result["quality_finite"].tolist() == [True, True]
    assert result["candidate_normal"].tolist() == [True, False]
