"""Persistence baseline for the five-minute active-power forecast."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from pv_anomaly.models.metrics import regression_metrics


def evaluate_persistence(path: str | Path) -> dict[str, Any]:
    available = set(pq.read_schema(path).names)
    columns = ["device_no", "active_power", "target_active_power"]
    for optional in ("plant_id", "device_key"):
        if optional in available:
            columns.append(optional)
    frame = pd.read_parquet(
        path,
        columns=columns,
    )
    overall = regression_metrics(
        frame["target_active_power"].to_numpy(),
        frame["active_power"].to_numpy(),
    )
    identity = "device_key" if "device_key" in frame else "device_no"
    per_device = {
        str(device): regression_metrics(
            group["target_active_power"].to_numpy(),
            group["active_power"].to_numpy(),
        )
        for device, group in frame.groupby(identity, observed=True)
    }
    per_plant = {}
    if "plant_id" in frame:
        per_plant = {
            str(int(plant_id)): regression_metrics(
                group["target_active_power"].to_numpy(),
                group["active_power"].to_numpy(),
            )
            for plant_id, group in frame.groupby("plant_id", observed=True)
        }
    return {
        "method": "persistence_t_plus_5_minutes",
        "overall": overall,
        "per_plant": per_plant,
        "per_device": per_device,
    }
