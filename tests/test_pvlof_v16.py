import pandas as pd

from pv_anomaly.pvlof_v16 import (
    PVLOFV16MemoryConfig,
    apply_confirmed_anomaly_memory_v16,
)


def test_v16_memory_uses_independent_column_names():
    raw = [1, 1, 1, 0, 1, 0, 0, 0]
    frame = pd.DataFrame({
        "plant_id": ["234"] * len(raw),
        "device_no": ["dev-a"] * len(raw),
        "event_time": pd.date_range(
            "2026-06-01", periods=len(raw), freq="5min", tz="UTC"
        ),
        "string_no": [1] * len(raw),
        "v2_eligible": [1] * len(raw),
        "response_known": [1] * len(raw),
        "string_current": [8.0] * len(raw),
        "pvlof_score": [10.0] * len(raw),
        "isolated_directional_raw_alert": raw,
        "isolated_hier_raw_alert": raw,
        "pvlof_v2_hier_strict_alert": [0] * len(raw),
    })
    result = apply_confirmed_anomaly_memory_v16(
        frame, PVLOFV16MemoryConfig()
    )
    assert "pvlof_v15_alert" not in result
    assert result["pvlof_v16_alert"].tolist() == [0, 0, 1, 0, 1, 0, 0, 0]
    assert result["pvlof_v16_memory_active"].tolist() == [0, 0, 1, 1, 1, 1, 1, 0]
    assert result["pvlof_v16_memory_clear_code"].tolist()[-1] == 1
