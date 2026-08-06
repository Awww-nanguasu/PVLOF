import pytest

torch = pytest.importorskip("torch")

from pv_anomaly.models.training import CompositeRatioLoss, _loss_from_config  # noqa: E402


def test_composite_ratio_loss_adds_weighted_mse():
    predicted = torch.tensor([0.0, 0.5])
    actual = torch.tensor([0.0, 1.0])
    smooth_only = CompositeRatioLoss(beta=0.05)(predicted, actual)
    composite = CompositeRatioLoss(beta=0.05, mse_weight=0.5)(predicted, actual)
    expected_mse = torch.mean(torch.square(predicted - actual))
    assert composite == pytest.approx(float(smooth_only + 0.5 * expected_mse))


def test_old_training_config_defaults_to_smooth_l1_only():
    loss = _loss_from_config({"training": {}})
    assert loss.beta == 0.05
    assert loss.mse_weight == 0.0
    assert loss.power_mse_weight == 0.0


def test_composite_ratio_loss_rejects_negative_weight():
    with pytest.raises(ValueError, match="nonnegative"):
        CompositeRatioLoss(mse_weight=-0.1)


def test_power_aware_loss_weights_ratio_error_by_rated_power():
    predicted = torch.tensor([0.5, 0.5])
    actual = torch.tensor([0.0, 0.0])
    rated_power = torch.tensor([30.0, 150.0])
    smooth_only = CompositeRatioLoss(beta=0.05)(predicted, actual)
    power_aware = CompositeRatioLoss(
        beta=0.05,
        power_mse_weight=0.5,
        power_reference=150.0,
    )(predicted, actual, rated_power)
    normalized_power_error = (predicted - actual) * rated_power / 150.0
    expected = smooth_only + 0.5 * torch.mean(torch.square(normalized_power_error))
    assert power_aware == pytest.approx(float(expected))


def test_power_aware_loss_requires_rated_power():
    loss = CompositeRatioLoss(power_mse_weight=0.5)
    with pytest.raises(ValueError, match="Rated power"):
        loss(torch.tensor([0.5]), torch.tensor([0.0]))


def test_power_aware_loss_rejects_invalid_parameters():
    with pytest.raises(ValueError, match="Power MSE weight"):
        CompositeRatioLoss(power_mse_weight=-0.1)
    with pytest.raises(ValueError, match="Power reference"):
        CompositeRatioLoss(power_reference=0.0)
