import numpy as np
import pandas as pd
import pytest

from pv_anomaly.ewma import (
    apply_ewma,
    fit_ewma_calibration,
    prepare_residual_frame,
)


def _residual_frame(values: list[float], *, ratio: float = 0.5) -> pd.DataFrame:
    target = pd.date_range("2026-01-01", periods=len(values), freq="5min", tz="UTC")
    actual = np.full(len(values), ratio * 100.0)
    predicted = actual + np.asarray(values) * 100.0
    return pd.DataFrame(
        {
            "target_time": target,
            "device_no": ["a"] * len(values),
            "rated_power": [100.0] * len(values),
            "actual_power": actual,
            "predicted_power": predicted,
            "actual_power_ratio": [ratio] * len(values),
        }
    )


def test_prepare_residual_uses_predicted_minus_actual_for_underproduction():
    frame = _residual_frame([0.1, -0.2])
    prepared = prepare_residual_frame(frame, minimum_power_ratio=0.1)
    assert prepared["underproduction_residual"].tolist() == pytest.approx([0.1, -0.2])
    assert prepared["ewma_eligible"].tolist() == [True, True]


def test_low_power_rows_are_ineligible():
    frame = _residual_frame([0.1, 0.2], ratio=0.05)
    prepared = prepare_residual_frame(frame, minimum_power_ratio=0.1)
    assert not prepared["ewma_eligible"].any()


def test_ewma_requires_consecutive_exceedances():
    calibration, _ = fit_ewma_calibration(
        _residual_frame([0.0] * 10),
        lambda_=0.2,
        quantile=0.99,
        minimum_consecutive=2,
    )
    scored = apply_ewma(_residual_frame([0.0, 0.0, 1.0, 0.0, 1.0, 1.0]), calibration)
    assert scored.loc[2, "ewma_raw_alert"] == 1
    assert scored.loc[2, "ewma_alert"] == 0
    assert scored.loc[4, "ewma_raw_alert"] == 1
    assert scored.loc[5, "ewma_alert"] == 1


def test_fit_rejects_empty_eligible_set():
    with pytest.raises(ValueError, match="No eligible"):
        fit_ewma_calibration(_residual_frame([0.0], ratio=0.01))
