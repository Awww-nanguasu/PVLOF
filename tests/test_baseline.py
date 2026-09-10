from pathlib import Path

import pandas as pd

from pv_anomaly.models.baseline import evaluate_persistence


def test_persistence_uses_current_power_as_prediction(tmp_path: Path):
    path = tmp_path / "data.parquet"
    pd.DataFrame(
        {
            "device_no": ["a", "a", "b"],
            "active_power": [10.0, 20.0, 30.0],
            "target_active_power": [12.0, 18.0, 30.0],
        }
    ).to_parquet(path, index=False)
    report = evaluate_persistence(path)
    assert report["overall"]["mae"] == 4 / 3
    assert report["per_device"]["b"]["mae"] == 0


def test_persistence_reports_plants_and_composite_devices(tmp_path: Path):
    path = tmp_path / "global.parquet"
    pd.DataFrame(
        {
            "plant_id": [234, 234, 892, 892],
            "device_no": ["same"] * 4,
            "device_key": ["234::same"] * 2 + ["892::same"] * 2,
            "active_power": [10.0, 20.0, 30.0, 40.0],
            "target_active_power": [11.0, 21.0, 33.0, 43.0],
        }
    ).to_parquet(path, index=False)

    report = evaluate_persistence(path)

    assert set(report["per_plant"]) == {"234", "892"}
    assert set(report["per_device"]) == {"234::same", "892::same"}
    assert report["per_plant"]["234"]["mae"] == 1.0
    assert report["per_plant"]["892"]["mae"] == 3.0
