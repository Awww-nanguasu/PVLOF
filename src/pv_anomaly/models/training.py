"""GPU-aware training, checkpointing and evaluation for PowerTransformer."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from pv_anomaly.models.dataset import (
    DeviceVocabulary,
    FeatureScaler,
    PowerWindowDataset,
    read_training_frame,
)
from pv_anomaly.models.metrics import regression_metrics
from pv_anomaly.models.transformer import PowerTransformer


class CompositeRatioLoss(nn.Module):
    """Ratio Smooth L1 plus optional ratio- and power-space tail penalties."""

    def __init__(
        self,
        *,
        beta: float = 0.05,
        mse_weight: float = 0.0,
        power_mse_weight: float = 0.0,
        power_reference: float = 150.0,
    ):
        super().__init__()
        if beta <= 0:
            raise ValueError("Smooth L1 beta must be positive")
        if mse_weight < 0:
            raise ValueError("MSE weight must be nonnegative")
        if power_mse_weight < 0:
            raise ValueError("Power MSE weight must be nonnegative")
        if power_reference <= 0:
            raise ValueError("Power reference must be positive")
        self.beta = beta
        self.mse_weight = mse_weight
        self.power_mse_weight = power_mse_weight
        self.power_reference = power_reference

    def forward(
        self,
        predicted: torch.Tensor,
        actual: torch.Tensor,
        rated_power: torch.Tensor | None = None,
    ) -> torch.Tensor:
        smooth = F.smooth_l1_loss(predicted, actual, beta=self.beta)
        loss = smooth
        if self.mse_weight:
            loss = loss + self.mse_weight * F.mse_loss(predicted, actual)
        if self.power_mse_weight:
            if rated_power is None:
                raise ValueError("Rated power is required when power MSE is enabled")
            normalized_power_error = (predicted - actual) * (
                rated_power / self.power_reference
            )
            loss = loss + self.power_mse_weight * torch.mean(
                torch.square(normalized_power_error)
            )
        return loss


def _loss_from_config(config: dict[str, Any]) -> CompositeRatioLoss:
    training = config["training"]
    return CompositeRatioLoss(
        beta=float(training.get("smooth_l1_beta", 0.05)),
        mse_weight=float(training.get("mse_weight", 0.0)),
        power_mse_weight=float(training.get("power_mse_weight", 0.0)),
        power_reference=float(training.get("power_reference", 150.0)),
    )


def load_training_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _loader(dataset: PowerWindowDataset, config: dict[str, Any], *, shuffle: bool) -> DataLoader:
    workers = int(config["training"]["num_workers"])
    return DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        drop_last=False,
    )


def build_datasets(config: dict[str, Any]) -> tuple[
    dict[str, PowerWindowDataset], FeatureScaler, DeviceVocabulary
]:
    data = config["data"]
    features = [str(name) for name in data["features"]]
    frames = {
        split: read_training_frame(data[split]) for split in ("train", "validation", "test")
    }
    scaler = FeatureScaler.fit(frames["train"], features)
    vocabulary = DeviceVocabulary.fit(frames["train"])
    datasets = {
        split: PowerWindowDataset(
            frame,
            feature_scaler=scaler,
            device_vocabulary=vocabulary,
            window_size=int(data["window_size"]),
            interval_minutes=int(data["interval_minutes"]),
        )
        for split, frame in frames.items()
    }
    return datasets, scaler, vocabulary


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: tensor.to(device, non_blocking=True) for name, tensor in batch.items()}


def _predict_batch(
    model: PowerTransformer,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    current_ratio = batch["current_ratio"] if model.residual_connection else None
    return model(batch["features"], batch["device_id"], current_ratio)


@torch.inference_mode()
def evaluate_model(
    model: PowerTransformer,
    loader: DataLoader,
    device: torch.device,
    loss_function: nn.Module,
    *,
    use_amp: bool,
) -> dict[str, Any]:
    model.eval()
    losses: list[float] = []
    actual: list[np.ndarray] = []
    predicted: list[np.ndarray] = []
    plant_ids: list[np.ndarray] = []
    for batch in loader:
        batch = _move_batch(batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp and device.type == "cuda",
        ):
            predicted_ratio = _predict_batch(model, batch)
            loss = loss_function(
                predicted_ratio,
                batch["target_ratio"],
                batch["rated_power"],
            )
        losses.append(float(loss.item()) * len(batch["target_ratio"]))
        predicted_power = predicted_ratio * batch["rated_power"]
        actual.append(batch["target_power"].detach().cpu().numpy())
        predicted.append(predicted_power.detach().float().cpu().numpy())
        plant_ids.append(batch["plant_id"].detach().cpu().numpy())
    actual_values = np.concatenate(actual)
    predicted_values = np.concatenate(predicted)
    plant_values = np.concatenate(plant_ids)
    metrics = regression_metrics(actual_values, predicted_values)
    metrics["loss"] = sum(losses) / metrics["samples"]
    metrics["per_plant"] = {
        str(int(plant_id)): regression_metrics(
            actual_values[plant_values == plant_id],
            predicted_values[plant_values == plant_id],
        )
        for plant_id in np.unique(plant_values)
        if plant_id >= 0
    }
    return metrics


def _model_from_config(
    config: dict[str, Any], num_features: int, num_devices: int
) -> tuple[PowerTransformer, dict[str, Any]]:
    model_config = {
        "num_features": num_features,
        "num_devices": num_devices,
        "window_size": int(config["data"]["window_size"]),
        **{name: value for name, value in config["model"].items()},
    }
    return PowerTransformer(**model_config), model_config


def train(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config["seed"])
    seed_everything(seed)
    device = select_device(str(config["device"]))
    datasets, scaler, vocabulary = build_datasets(config)
    loaders = {
        split: _loader(dataset, config, shuffle=split == "train")
        for split, dataset in datasets.items()
    }
    model, model_config = _model_from_config(
        config,
        len(scaler.feature_names),
        len(vocabulary.values),
    )
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    loss_function = _loss_from_config(config)
    amp_enabled = bool(config["training"]["use_amp"]) and device.type == "cuda"
    grad_scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    output = Path(config["outputs"]["directory"])
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "best.pt"
    history: list[dict[str, Any]] = []
    best_validation_rmse = float("inf")
    stale_epochs = 0

    print(
        json.dumps(
            {
                "device": str(device),
                "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "windows": {name: len(value) for name, value in datasets.items()},
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        model.train()
        train_loss_sum = 0.0
        train_samples = 0
        for batch in loaders["train"]:
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                prediction = _predict_batch(model, batch)
                loss = loss_function(
                    prediction,
                    batch["target_ratio"],
                    batch["rated_power"],
                )
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["training"]["gradient_clip_norm"])
            )
            grad_scaler.step(optimizer)
            grad_scaler.update()
            train_loss_sum += float(loss.item()) * len(batch["target_ratio"])
            train_samples += len(batch["target_ratio"])

        validation = evaluate_model(
            model,
            loaders["validation"],
            device,
            loss_function,
            use_amp=amp_enabled,
        )
        epoch_result = {
            "epoch": epoch,
            "train_loss": train_loss_sum / train_samples,
            "validation": validation,
        }
        history.append(epoch_result)
        print(json.dumps(epoch_result, ensure_ascii=False), flush=True)
        if validation["rmse"] < best_validation_rmse:
            best_validation_rmse = validation["rmse"]
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": model_config,
                    "feature_scaler": scaler.to_dict(),
                    "device_vocabulary": vocabulary.to_dict(),
                    "training_config": config,
                    "epoch": epoch,
                    "validation_metrics": validation,
                    "torch_version": torch.__version__,
                    "torch_cuda_version": torch.version.cuda,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= int(config["training"]["early_stopping_patience"]):
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = evaluate_model(
        model,
        loaders["test"],
        device,
        loss_function,
        use_amp=amp_enabled,
    )
    summary = {
        "best_epoch": checkpoint["epoch"],
        "best_validation": checkpoint["validation_metrics"],
        "test": test_metrics,
        "checkpoint": str(checkpoint_path),
        "history_epochs": len(history),
    }
    (output / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    data_path: str | Path,
    *,
    batch_size: int = 1024,
    num_workers: int = 4,
    requested_device: str = "auto",
) -> dict[str, Any]:
    device = select_device(requested_device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    scaler = FeatureScaler.from_dict(checkpoint["feature_scaler"])
    vocabulary = DeviceVocabulary.from_dict(checkpoint["device_vocabulary"])
    model = PowerTransformer(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    config = checkpoint["training_config"]
    dataset = PowerWindowDataset(
        read_training_frame(data_path),
        feature_scaler=scaler,
        device_vocabulary=vocabulary,
        window_size=int(config["data"]["window_size"]),
        interval_minutes=int(config["data"]["interval_minutes"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return evaluate_model(
        model,
        loader,
        device,
        _loss_from_config(config),
        use_amp=device.type == "cuda",
    )
