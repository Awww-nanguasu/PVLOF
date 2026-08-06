import pytest

torch = pytest.importorskip("torch")

from pv_anomaly.models.transformer import PowerTransformer  # noqa: E402


def test_transformer_forward_shape_and_nonnegative_output():
    model = PowerTransformer(
        num_features=7,
        num_devices=3,
        window_size=24,
        d_model=32,
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
        device_embedding_dim=8,
    )
    output = model(torch.randn(5, 24, 7), torch.tensor([0, 1, 2, 0, 1]))
    assert output.shape == (5,)
    assert torch.all(output >= 0)


def test_residual_transformer_starts_as_persistence_prediction():
    model = PowerTransformer(
        num_features=3,
        num_devices=2,
        window_size=6,
        d_model=16,
        nhead=4,
        num_layers=1,
        dim_feedforward=32,
        device_embedding_dim=4,
        residual_connection=True,
    )
    features = torch.randn(4, 6, 3)
    device_ids = torch.tensor([0, 1, 0, 1])
    current_ratio = torch.tensor([0.1, 0.3, 0.7, 1.0])
    output = model(features, device_ids, current_ratio)
    assert torch.equal(output, current_ratio)


def test_residual_transformer_requires_current_ratio():
    model = PowerTransformer(
        num_features=3,
        num_devices=2,
        window_size=6,
        d_model=16,
        nhead=4,
        num_layers=1,
        dim_feedforward=32,
        device_embedding_dim=4,
        residual_connection=True,
    )
    with pytest.raises(ValueError, match="current_ratio"):
        model(torch.randn(2, 6, 3), torch.tensor([0, 1]))
