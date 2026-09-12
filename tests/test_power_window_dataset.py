import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from pv_anomaly.models.dataset import (  # noqa: E402
    DeviceVocabulary,
    FeatureScaler,
    PowerWindowDataset,
)


def test_dataset_rejects_windows_across_time_gaps():
    frame = pd.DataFrame(
        {
            "event_time": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T00:05:00Z",
                    "2026-01-01T00:10:00Z",
                    "2026-01-01T01:00:00Z",
                    "2026-01-01T01:05:00Z",
                    "2026-01-01T01:10:00Z",
                ],
                utc=True,
            ),
            "device_no": ["a"] * 6,
            "feature": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "rated_power": [100.0] * 6,
            "target_active_power": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        }
    )
    dataset = PowerWindowDataset(
        frame,
        feature_scaler=FeatureScaler.fit(frame, ["feature"]),
        device_vocabulary=DeviceVocabulary.fit(frame),
        window_size=3,
    )
    assert len(dataset) == 2
    assert dataset[0]["features"].shape == (3, 1)


def test_dataset_uses_explicit_cross_plant_device_key():
    frame = pd.DataFrame(
        {
            "event_time": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T00:05:00Z",
                    "2026-01-01T00:10:00Z",
                ]
                * 2,
                utc=True,
            ),
            "device_no": ["same"] * 6,
            "device_key": ["234::same"] * 3 + ["892::same"] * 3,
            "plant_id": [234] * 3 + [892] * 3,
            "feature": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
            "rated_power": [100.0] * 6,
            "target_active_power": [2.0, 3.0, 4.0, 20.0, 30.0, 40.0],
        }
    )
    vocabulary = DeviceVocabulary.fit(frame)
    dataset = PowerWindowDataset(
        frame,
        feature_scaler=FeatureScaler.fit(frame, ["feature"]),
        device_vocabulary=vocabulary,
        window_size=3,
    )

    assert vocabulary.values == ("234::same", "892::same")
    assert len(dataset) == 2
    assert {int(dataset[index]["plant_id"]) for index in range(2)} == {234, 892}
