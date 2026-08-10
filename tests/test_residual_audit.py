import pandas as pd
import pytest

from pv_anomaly.models.residuals import audit_residual_frame


def test_residual_audit_identifies_mae_delta_by_power_bin():
    frame = pd.DataFrame(
        {
            "target_time": pd.to_datetime(
                ["2026-07-01T00:00:00Z", "2026-07-01T00:05:00Z"],
                utc=True,
            ),
            "device_no": ["a", "a"],
            "actual_power": [10.0, 50.0],
            "predicted_power": [12.0, 40.0],
            "baseline_prediction": [15.0, 48.0],
            "absolute_error": [2.0, 10.0],
            "baseline_absolute_error": [5.0, 2.0],
            "actual_power_ratio": [0.1, 0.5],
        }
    )
    report, tables = audit_residual_frame(frame)
    assert report["samples"] == 2
    assert report["mae_delta"] == pytest.approx(2.5)
    assert report["transformer_win_rate"] == 0.5
    assert set(tables) == {
        "by_date",
        "by_device",
        "by_power_bin",
        "by_device_power_bin",
    }
    bins = tables["by_power_bin"].set_index("power_bin")
    assert bins.loc["10-20%", "mae_delta"] == pytest.approx(-3.0)
    assert bins.loc["40-60%", "mae_delta"] == pytest.approx(8.0)
