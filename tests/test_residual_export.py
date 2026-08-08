from pathlib import Path

import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from pv_anomaly.models.dataset import DeviceVocabulary, FeatureScaler  # noqa: E402
from pv_anomaly.models.residuals import predict_checkpoint_frame  # noqa: E402
from pv_anomaly.models.transformer import PowerTransformer  # noqa: E402


def test_predict_checkpoint_frame_exports_window_metadata(tmp_path: Path):
    frame = pd.DataFrame(
        {
            "event_time": pd.date_range("2026-01-01", periods=4, freq="5min", tz="UTC"),
            "target_time": pd.date_range("2026-01-01 00:05", periods=4, freq="5min", tz="UTC"),
            "plant_id": [1] * 4,
            "device_no": ["a"] * 4,
            "active_power": [10.0, 20.0, 30.0, 40.0],
            "active_power_ratio": [0.1, 0.2, 0.3, 0.4],
            "peer_median_power_ratio": [0.1, 0.2, 0.3, 0.4],
            "peer_power_available": [1.0] * 4,
            "rated_power": [100.0] * 4,
            "target_active_power": [20.0, 30.0, 40.0, 50.0],
            "candidate_normal": [True] * 4,
        }
    )
    data_path = tmp_path / "validation.parquet"
    frame.to_parquet(data_path, index=False)
    scaler = FeatureScaler.fit(frame, ["active_power_ratio"])
    vocabulary = DeviceVocabulary.fit(frame)
    model_config = {
        "num_features": 1,
        "num_devices": 1,
        "window_size": 2,
        "d_model": 8,
        "nhead": 2,
        "num_layers": 1,
        "dim_feedforward": 16,
        "dropout": 0.0,
        "device_embedding_dim": 4,
    }
    model = PowerTransformer(**model_config)
    checkpoint_path = tmp_path / "best.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": model_config,
            "feature_scaler": scaler.to_dict(),
            "device_vocabulary": vocabulary.to_dict(),
            "training_config": {
                "data": {
                    "window_size": 2,
                    "interval_minutes": 5,
                }
            },
        },
        checkpoint_path,
    )

    result = predict_checkpoint_frame(
        checkpoint_path,
        data_path,
        split="validation",
        batch_size=2,
        num_workers=0,
        requested_device="cpu",
    )

    assert len(result) == 3
    assert result["input_time"].iloc[0] == frame["event_time"].iloc[1]
    assert result["actual_power"].tolist() == [30.0, 40.0, 50.0]
    assert result["baseline_prediction"].tolist() == [20.0, 30.0, 40.0]
    assert result["baseline_residual"].tolist() == [10.0, 10.0, 10.0]
    assert result["peer_physical_prediction"].tolist() == pytest.approx([30.0, 40.0, 50.0])
    assert result["peer_physical_residual"].tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert result["peer_median_ramp_ratio_5m"].tolist() == pytest.approx([0.1, 0.1, 0.1])
    assert result["predicted_power"].notna().all()


def test_residual_endpoints_keep_cross_plant_device_identity(tmp_path: Path):
    frame = pd.DataFrame(
        {
            "event_time": list(
                pd.date_range("2026-01-01", periods=3, freq="5min", tz="UTC")
            )
            * 2,
            "target_time": list(
                pd.date_range("2026-01-01 00:05", periods=3, freq="5min", tz="UTC")
            )
            * 2,
            "plant_id": [234] * 3 + [892] * 3,
            "device_no": ["same"] * 6,
            "device_key": ["234::same"] * 3 + ["892::same"] * 3,
            "active_power": [10.0, 20.0, 30.0, 100.0, 110.0, 120.0],
            "active_power_ratio": [0.1, 0.2, 0.3, 0.5, 0.55, 0.6],
            "rated_power": [100.0] * 3 + [200.0] * 3,
            "target_active_power": [20.0, 30.0, 40.0, 110.0, 120.0, 130.0],
            "candidate_normal": [True] * 6,
        }
    )
    data_path = tmp_path / "validation.parquet"
    frame.to_parquet(data_path, index=False)
    scaler = FeatureScaler.fit(frame, ["active_power_ratio"])
    vocabulary = DeviceVocabulary.fit(frame)
    model_config = {
        "num_features": 1,
        "num_devices": 2,
        "window_size": 2,
        "d_model": 8,
        "nhead": 2,
        "num_layers": 1,
        "dim_feedforward": 16,
        "dropout": 0.0,
        "device_embedding_dim": 4,
    }
    model = PowerTransformer(**model_config)
    checkpoint_path = tmp_path / "best.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": model_config,
            "feature_scaler": scaler.to_dict(),
            "device_vocabulary": vocabulary.to_dict(),
            "training_config": {"data": {"window_size": 2, "interval_minutes": 5}},
        },
        checkpoint_path,
    )

    result = predict_checkpoint_frame(
        checkpoint_path,
        data_path,
        split="validation",
        batch_size=4,
        num_workers=0,
        requested_device="cpu",
    )

    assert result["plant_id"].tolist() == [234, 234, 892, 892]
    assert result["baseline_prediction"].tolist() == [20.0, 30.0, 110.0, 120.0]
    assert result["actual_power"].tolist() == [30.0, 40.0, 120.0, 130.0]
