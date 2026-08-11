import numpy as np
import pandas as pd

from pv_anomaly.pvlof_v2 import fit_pvlof_v2_calibration
from pv_anomaly.pvlof_v12 import (
    build_conditioned_virtual_context,
    fit_weather_calibration,
)


def _frame() -> pd.DataFrame:
    rows = []
    times = pd.date_range("2026-06-01", periods=8, freq="5min", tz="UTC")
    for step, timestamp in enumerate(times):
        level = 5.0 + step
        for device, scale in (("a", 1.0), ("b", 1.5), ("c", 2.0)):
            row = {"event_time": timestamp, "plant_id": 234, "device_no": device}
            row.update({f"string_current_{number:02d}": level * scale for number in range(1, 6)})
            rows.append(row)
    return pd.DataFrame(rows)


def test_weather_conditioning_is_bounded_and_missing_weather_falls_back():
    frame = _frame()
    base, _ = fit_pvlof_v2_calibration(
        frame,
        n_neighbors=2,
        minimum_peer_devices=2,
        minimum_strings=4,
        max_score_rows=1000,
    )
    weather = pd.DataFrame(
        {
            "time": pd.date_range("2026-06-01", periods=3, freq="15min", tz="UTC"),
            "plant_id": 234,
            "forecast_ghi": [100.0, 400.0, 800.0],
        }
    )
    calibration, report = fit_weather_calibration(
        frame,
        base,
        weather,
        candidate_source_offsets_minutes=(0,),
        minimum_mapping_samples=2,
        mapping_bins=3,
    )
    assert report["usable_plants"] == 1
    context, context_report = build_conditioned_virtual_context(
        frame, base, calibration, weather
    )
    both = context["raw_virtual_irradiance"].notna() & context["forecast_available"]
    relative = (
        context.loc[both, "conditioned_virtual_irradiance"]
        / context.loc[both, "raw_virtual_irradiance"]
    )
    assert relative.between(0.8, 1.2).all()
    assert context_report["forecast_available_rows"] > 0

    empty_weather = weather.iloc[0:0]
    fallback, _ = build_conditioned_virtual_context(frame, base, calibration, empty_weather)
    assert np.allclose(
        fallback["conditioned_virtual_irradiance"],
        fallback["raw_virtual_irradiance"],
        equal_nan=True,
    )


def test_forecast_can_supply_context_when_peer_virtual_is_missing():
    frame = _frame()
    base, _ = fit_pvlof_v2_calibration(
        frame,
        n_neighbors=2,
        minimum_peer_devices=2,
        minimum_strings=4,
        max_score_rows=1000,
    )
    weather = pd.DataFrame(
        {
            "time": pd.date_range("2026-06-01", periods=3, freq="15min", tz="UTC"),
            "plant_id": 234,
            "forecast_ghi": [100.0, 400.0, 800.0],
        }
    )
    calibration, _ = fit_weather_calibration(
        frame, base, weather,
        candidate_source_offsets_minutes=(0,), minimum_mapping_samples=2, mapping_bins=3,
    )
    single = frame[frame["device_no"].eq("a")].copy()
    context, _ = build_conditioned_virtual_context(single, base, calibration, weather)
    available = context["forecast_available"]
    assert available.any()
    assert context.loc[available, "raw_virtual_irradiance"].isna().all()
    assert context.loc[available, "conditioned_virtual_irradiance"].notna().all()
    assert context.loc[available, "conditioned_peer_count"].ge(base.minimum_peer_devices).all()
