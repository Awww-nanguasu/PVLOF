import numpy as np
import pandas as pd

from pv_anomaly.pvlof import (
    PVLOFCalibration,
    apply_pvlof,
    collapse_pvlof_events,
    fit_pvlof_calibration,
    prepare_pvlof_frame,
)


def _wide_frame(currents: list[list[float]], statuses: list[list[int]] | None = None):
    rows = len(currents)
    data = {
        "event_time": pd.date_range("2026-06-01", periods=rows, freq="5min", tz="UTC"),
        "plant_id": [33] * rows,
        "device_no": ["device-a"] * rows,
        "active_power": [50.0] * rows,
        "rated_power": [100.0] * rows,
        "status_code": [1] * rows,
    }
    for index in range(len(currents[0])):
        data[f"string_current_{index + 1:02d}"] = [row[index] for row in currents]
        if statuses is not None:
            data[f"string_status_{index + 1:02d}"] = [row[index] for row in statuses]
    return pd.DataFrame(data)


def _calibration(**overrides) -> PVLOFCalibration:
    values = {
        "n_neighbors": 2,
        "quantile": 0.99,
        "threshold": 2.0,
        "minimum_power_ratio": 0.1,
        "maximum_power_ratio": 1.1,
        "zero_current_threshold": 0.0,
        "minimum_relative_drop": 0.1,
        "minimum_strings": 4,
        "minimum_consecutive": 2,
        "expected_interval_minutes": 5,
        "distance_floor": 0.01,
        "eligible_status_codes": (1, 4),
    }
    values.update(overrides)
    return PVLOFCalibration(**values)


def test_prepare_scores_low_nonzero_string_as_outlier():
    frame = _wide_frame([[10.0, 10.1, 9.9, 10.0, 4.0]])
    scored = prepare_pvlof_frame(
        frame,
        n_neighbors=2,
        minimum_strings=4,
    )
    low = scored.loc[scored["string_no"] == 5].iloc[0]
    assert low["string_current_ratio"] < 0.5
    assert low["pvlof_score"] > 2.0
    assert scored["pvlof_eligible"].all()


def test_zero_current_is_direct_rule_and_not_lof_sample():
    frame = _wide_frame(
        [[10.0, 10.1, 9.9, 10.0, 0.0]],
        statuses=[[1, 1, 1, 1, 4]],
    )
    scored = prepare_pvlof_frame(frame, n_neighbors=2, minimum_strings=4)
    zero = scored.loc[scored["string_no"] == 5].iloc[0]
    assert zero["zero_current_alert"] == 1
    assert not zero["pvlof_eligible"]
    assert np.isnan(zero["pvlof_score"])
    assert zero["weak_zero_current_label"]


def test_low_power_rows_are_excluded_as_night_like():
    frame = _wide_frame([[0.0, 0.0, 0.0, 0.0, 0.0]])
    frame["active_power"] = 1.0
    scored = prepare_pvlof_frame(frame, n_neighbors=2, minimum_strings=4)
    assert scored.empty


def test_apply_requires_consecutive_directional_low_outliers():
    currents = [[10.0, 10.1, 9.9, 10.0, 4.0]] * 3
    scored = apply_pvlof(_wide_frame(currents), _calibration())
    low = scored[scored["string_no"] == 5].sort_values("event_time")
    assert low["pvlof_raw_alert"].tolist() == [1, 1, 1]
    assert low["pvlof_alert"].tolist() == [0, 1, 1]
    high = scored[scored["string_no"] != 5]
    assert not high["pvlof_alert"].any()


def test_fit_and_event_collapse():
    normal = _wide_frame(
        [
            [10.0, 10.1, 9.9, 10.0, 10.05],
            [11.0, 11.1, 10.9, 11.0, 11.05],
        ],
        statuses=[[1] * 5, [1] * 5],
    )
    calibration, report = fit_pvlof_calibration(
        normal,
        n_neighbors=2,
        minimum_strings=4,
        quantile=0.9,
    )
    assert calibration.threshold >= 1.0
    assert report["normal_calibration_samples"] == 10

    scored = apply_pvlof(
        _wide_frame([[10.0, 10.1, 9.9, 10.0, 0.0]] * 2),
        calibration,
    )
    events = collapse_pvlof_events(scored)
    assert len(events) == 1
    assert events.iloc[0]["string_no"] == 5
    assert events.iloc[0]["points"] == 2
