import pandas as pd

from pv_anomaly.pvlof_v15 import PVLOFV15MemoryConfig, apply_confirmed_anomaly_memory


def _frame(raw, *, valid=None, times=None):
    size = len(raw)
    if valid is None:
        valid = [1] * size
    if times is None:
        times = pd.date_range("2026-06-01", periods=size, freq="5min", tz="UTC")
    return pd.DataFrame(
        {
            "plant_id": ["234"] * size,
            "device_no": ["dev-a"] * size,
            "event_time": times,
            "string_no": [1] * size,
            "v2_eligible": valid,
            "response_known": valid,
            "string_current": [8.0] * size,
            "pvlof_score": [10.0] * size,
            "isolated_directional_raw_alert": raw,
            "isolated_hier_raw_alert": raw,
            "pvlof_v2_hier_strict_alert": [0] * size,
        }
    )


def test_confirmed_memory_realerts_until_three_consecutive_normal_points():
    raw = [1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1]
    result = apply_confirmed_anomaly_memory(
        _frame(raw), PVLOFV15MemoryConfig()
    )
    assert result["pvlof_v15_memory_alert"].tolist() == [
        0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0
    ]
    assert result["pvlof_v15_memory_active"].tolist() == [
        0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0
    ]
    assert result["pvlof_v15_normal_streak"].tolist() == [
        0, 0, 0, 1, 0, 1, 0, 0, 1, 2, 0, 0
    ]
    assert result["pvlof_v15_memory_clear_code"].tolist() == [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0
    ]


def test_normal_memory_points_are_not_alerts():
    result = apply_confirmed_anomaly_memory(
        _frame([1, 1, 1, 0, 0, 0]), PVLOFV15MemoryConfig()
    )
    assert result["pvlof_v15_alert"].tolist() == [0, 0, 1, 0, 0, 0]


def test_time_gap_clears_memory():
    times = pd.to_datetime(
        [
            "2026-06-01T00:00:00Z",
            "2026-06-01T00:05:00Z",
            "2026-06-01T00:10:00Z",
            "2026-06-01T00:30:00Z",
        ]
    )
    result = apply_confirmed_anomaly_memory(
        _frame([1, 1, 1, 1], times=times), PVLOFV15MemoryConfig()
    )
    assert result["pvlof_v15_memory_alert"].tolist() == [0, 0, 1, 0]
    assert result["pvlof_v15_entry_streak"].tolist() == [1, 2, 3, 1]


def test_unevaluable_point_clears_memory():
    result = apply_confirmed_anomaly_memory(
        _frame([1, 1, 1, 0, 1], valid=[1, 1, 1, 0, 1]),
        PVLOFV15MemoryConfig(),
    )
    assert result["pvlof_v15_memory_alert"].tolist() == [0, 0, 1, 0, 0]
    assert result["pvlof_v15_entry_streak"].tolist() == [1, 2, 3, 0, 1]
