import numpy as np
import pandas as pd

from pv_anomaly.pvlof_v17 import (
    PVLOFV17Config,
    _memory_state,
    apply_pvlof_v17,
    penalized_segments,
)


def _points(residuals, *, periods=3, expected_current=8.0):
    rows = []
    times = pd.date_range("2026-06-01", periods=periods, freq="5min", tz="UTC")
    residual_median = float(np.median(residuals))
    for event_time in times:
        for string_no, residual in enumerate(residuals, 1):
            rows.append({
                "plant_id": "234",
                "device_no": "dev-a",
                "event_time": event_time,
                "string_no": string_no,
                "v2_eligible": 1,
                "response_known": 1,
                "string_current": expected_current * residual,
                "expected_current": expected_current,
                "residual_ratio": residual,
                "residual_median": residual_median,
                "pvlof_score": 1.0,
                "pvlof_v16_raw_anomaly": 0,
                "pvlof_v16_alert": 0,
                "pvlof_v16_memory_active": 0,
                "pvlof_v16_memory_reactivated_alert": 0,
                "pvlof_v16_memory_clear_code": 0,
                "pvlof_v16_entry_streak": 0,
                "pvlof_v16_normal_streak": 0,
            })
    return pd.DataFrame(rows)


def test_penalized_segments_supports_multiple_staircases():
    values = np.array([0.50, 0.51, 0.52, 0.74, 0.75, 0.76, 0.99, 1.00, 1.01])
    segments = penalized_segments(
        values,
        penalty=0.01,
        minimum_segment_size=2,
    )
    assert segments == [(0, 3), (3, 6), (6, 9)]


def test_v17_adds_dense_low_group_after_three_points():
    frame = _points([0.50, 0.52, 0.54, 0.98, 0.99, 1.00, 1.01, 1.02, 1.03, 1.04])
    result = apply_pvlof_v17(frame, PVLOFV17Config())

    low = result[result["string_no"].isin([1, 2, 3])]
    high = result[~result["string_no"].isin([1, 2, 3])]
    assert low["pvlof_v17_segmentation_raw_candidate"].eq(1).all()
    assert high["pvlof_v17_segmentation_raw_candidate"].eq(0).all()
    assert low.groupby("string_no")["pvlof_v17_segmentation_alert"].agg(list).tolist() == [
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
    ]
    assert result["pvlof_v17_added_alert"].sum() == 3


def test_v17_does_not_alert_homogeneous_whole_device_drop():
    frame = _points([0.68, 0.69, 0.70, 0.70, 0.71, 0.71, 0.72, 0.72])
    result = apply_pvlof_v17(frame, PVLOFV17Config())

    assert result["pvlof_v17_no_internal_contrast"].eq(1).all()
    assert result["pvlof_v17_segmentation_raw_candidate"].eq(0).all()
    assert result["pvlof_v17_alert"].eq(0).all()


def test_v17_physical_gate_blocks_small_segment_difference():
    frame = _points(
        [0.90, 0.91, 0.92, 0.99, 1.00, 1.00, 1.01, 1.01],
        expected_current=4.0,
    )
    config = PVLOFV17Config(segmentation_penalty=0.001)
    result = apply_pvlof_v17(frame, config)

    low = result[result["string_no"].isin([1, 2, 3])]
    assert low["pvlof_v17_segmentation_structural_member"].eq(1).all()
    assert low["pvlof_v17_physical_effect_eligible"].eq(0).all()
    assert low["pvlof_v17_segmentation_raw_candidate"].eq(0).all()


def test_v17_preserves_existing_v16_alerts():
    frame = _points([0.98, 0.99, 1.00, 1.01, 1.02, 1.03, 1.04], periods=1)
    frame.loc[frame["string_no"].eq(1), "pvlof_v16_alert"] = 1
    result = apply_pvlof_v17(frame, PVLOFV17Config())

    preserved = result.loc[result["string_no"].eq(1)].iloc[0]
    assert preserved["pvlof_v17_alert"] == 1
    assert preserved["pvlof_v17_added_alert"] == 0


def _memory_result(raw, valid, *, minutes=None, maximum_unknown_intervals=2):
    size = len(raw)
    offsets = minutes if minutes is not None else range(0, size * 5, 5)
    frame = pd.DataFrame({
        "plant_id": ["234"] * size,
        "device_no": ["dev-a"] * size,
        "string_no": [1] * size,
        "event_time": [
            pd.Timestamp("2026-06-01", tz="UTC") + pd.Timedelta(minutes=value)
            for value in offsets
        ],
    })
    return _memory_state(
        frame,
        raw=pd.Series(raw, dtype=bool),
        valid=pd.Series(valid, dtype=bool),
        config=PVLOFV17Config(
            maximum_unknown_intervals=maximum_unknown_intervals,
        ),
    )


def test_memory_suspends_for_two_invalid_points_and_resumes_alert():
    state = _memory_result(
        [1, 1, 1, 0, 0, 1],
        [1, 1, 1, 0, 0, 1],
    )

    assert state["memory_active"].tolist() == [0, 0, 1, 1, 1, 1]
    assert state["memory_alert"].tolist() == [0, 0, 1, 0, 0, 1]
    assert state["unknown_streak"].tolist() == [0, 0, 0, 1, 2, 0]
    assert state["memory_suspended"].tolist() == [0, 0, 0, 1, 1, 0]
    assert state["memory_resumed_alert"].tolist() == [0, 0, 0, 0, 0, 1]
    assert state["memory_clear_code"].tolist() == [0, 0, 0, 0, 0, 0]


def test_third_invalid_point_ends_memory_and_requires_fresh_confirmation():
    state = _memory_result(
        [1, 1, 1, 0, 0, 0, 1, 1, 1],
        [1, 1, 1, 0, 0, 0, 1, 1, 1],
    )

    assert state["memory_active"].tolist() == [0, 0, 1, 1, 1, 0, 0, 0, 1]
    assert state["memory_alert"].tolist() == [0, 0, 1, 0, 0, 0, 0, 0, 1]
    assert state["memory_clear_code"].tolist() == [0, 0, 0, 0, 0, 3, 0, 0, 0]


def test_short_timestamp_gap_preserves_memory_but_long_gap_times_out():
    short = _memory_result(
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        minutes=[0, 5, 10, 20],
    )
    assert short["memory_alert"].tolist() == [0, 0, 1, 1]
    assert short["memory_resumed_alert"].tolist() == [0, 0, 0, 1]
    assert short["memory_clear_code"].tolist() == [0, 0, 0, 0]

    long = _memory_result(
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        minutes=[0, 5, 10, 30, 35, 40],
    )
    assert long["memory_alert"].tolist() == [0, 0, 1, 0, 0, 1]
    assert long["entry_streak"].tolist() == [1, 2, 3, 1, 2, 3]
    assert long["memory_clear_code"].tolist() == [0, 0, 0, 2, 0, 0]
