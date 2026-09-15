import pandas as pd

from pv_anomaly.pvlof_v2 import PVLOFV2Calibration
from scripts.run_pvlof_58d_versions import (
    _filter_plant,
    derive_v15_calibration_from_v16,
)


def test_derive_v15_calibration_from_v16_restores_frozen_gate():
    v16 = PVLOFV2Calibration(
        version="pvlof-v1.6-hybrid-gate",
        minimum_isolated_relative_drop=0.20,
        minimum_isolated_absolute_drop=0.50,
        isolated_effect_gate_mode="any",
    )

    v15 = derive_v15_calibration_from_v16(v16)

    assert v15.version == "pvlof-v1.5-memory-5pct"
    assert v15.minimum_isolated_relative_drop == 0.05
    assert v15.minimum_isolated_absolute_drop == 0.50
    assert v15.isolated_effect_gate_mode == "all"
    assert v15.lof_threshold == v16.lof_threshold
    assert v15.collective_gap_threshold == v16.collective_gap_threshold


def test_filter_plant_keeps_only_requested_context():
    frame = pd.DataFrame({
        "plant_id": [234, 791, 234],
        "device_no": ["a", "b", "c"],
        "event_time": pd.date_range("2026-06-01", periods=3, freq="5min", tz="UTC"),
    })

    selected, report = _filter_plant(frame, "234")

    assert set(selected["plant_id"]) == {"234"}
    assert set(selected["device_no"]) == {"a", "c"}
    assert report["rows_before_filter"] == 3
    assert report["rows_after_filter"] == 2
    assert report["device_ids"] == ["a", "c"]
