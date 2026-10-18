import numpy as np
import pandas as pd
import pytest

from pv_anomaly.peer_features import _leave_one_out_median, add_peer_power_features


def test_leave_one_out_median_excludes_each_device():
    medians, counts = _leave_one_out_median(np.array([0.2, 0.4, 0.8]))
    assert medians.tolist() == pytest.approx([0.6, 0.5, 0.3])
    assert counts.tolist() == [2, 2, 2]


def test_peer_features_are_grouped_by_plant_and_exact_time():
    frame = pd.DataFrame(
        {
            "event_time": pd.to_datetime(
                [
                    "2026-01-01T01:00:00Z",
                    "2026-01-01T01:00:00Z",
                    "2026-01-01T01:00:00Z",
                    "2026-01-01T01:00:00Z",
                    "2026-01-01T01:05:00Z",
                ],
                utc=True,
            ),
            "plant_id": [33, 33, 33, 34, 33],
            "device_no": ["a", "b", "c", "d", "a"],
            "active_power_ratio": [0.2, 0.4, 0.8, 0.9, 0.3],
        }
    )
    result = add_peer_power_features(frame, minimum_peers=2)
    assert result.loc[:2, "peer_median_power_ratio"].tolist() == pytest.approx(
        [0.6, 0.5, 0.3]
    )
    assert result.loc[:2, "device_peer_power_gap"].tolist() == pytest.approx(
        [-0.4, -0.1, 0.5]
    )
    assert result.loc[:2, "peer_available_count"].tolist() == [2.0, 2.0, 2.0]
    assert result.loc[:2, "peer_power_available"].tolist() == [1.0, 1.0, 1.0]
    assert result.loc[3:, "peer_power_available"].tolist() == [0.0, 0.0]
    assert result.loc[3, "peer_median_power_ratio"] == pytest.approx(0.9)
    assert result.loc[4, "device_peer_power_gap"] == 0.0


def test_peer_features_handle_missing_target_ratio_without_using_it_as_peer():
    frame = pd.DataFrame(
        {
            "event_time": pd.to_datetime(["2026-01-01T01:00:00Z"] * 3, utc=True),
            "plant_id": [33, 33, 33],
            "device_no": ["a", "b", "c"],
            "active_power_ratio": [0.2, 0.4, np.nan],
        }
    )
    result = add_peer_power_features(frame, minimum_peers=2)
    assert result.loc[2, "peer_median_power_ratio"] == pytest.approx(0.3)
    assert result.loc[2, "peer_available_count"] == 2
    assert result.loc[2, "device_peer_power_gap"] == 0.0
