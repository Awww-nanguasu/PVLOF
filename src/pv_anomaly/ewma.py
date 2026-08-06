"""EWMA monitoring for one-step-ahead underproduction residuals."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EWMACalibration:
    """Parameters fitted on a normal validation residual stream."""

    lambda_: float
    quantile: float
    threshold: float
    center: float
    scale: float
    minimum_power_ratio: float
    maximum_power_ratio: float
    minimum_consecutive: int
    expected_interval_minutes: int

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["lambda"] = payload.pop("lambda_")
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EWMACalibration":
        values = dict(payload)
        values["lambda_"] = values.pop("lambda")
        return cls(**values)


def prepare_residual_frame(
    frame: pd.DataFrame,
    *,
    minimum_power_ratio: float = 0.10,
    maximum_power_ratio: float = 1.10,
) -> pd.DataFrame:
    """Prepare causal underproduction residuals and valid daytime-like rows."""
    required = {
        "target_time",
        "device_no",
        "rated_power",
        "actual_power",
        "predicted_power",
        "actual_power_ratio",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing EWMA residual columns: {missing}")
    if minimum_power_ratio < 0 or maximum_power_ratio <= minimum_power_ratio:
        raise ValueError("power-ratio bounds are invalid")
    result = frame.copy()
    result["target_time"] = pd.to_datetime(result["target_time"], errors="raise", utc=True)
    result = result.sort_values(["device_no", "target_time"]).reset_index(drop=True)
    rated = pd.to_numeric(result["rated_power"], errors="coerce").to_numpy(dtype=np.float64)
    actual = pd.to_numeric(result["actual_power"], errors="coerce").to_numpy(dtype=np.float64)
    predicted = pd.to_numeric(result["predicted_power"], errors="coerce").to_numpy(dtype=np.float64)
    actual_ratio = pd.to_numeric(result["actual_power_ratio"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    underproduction = (predicted - actual) / rated
    eligible = (
        np.isfinite(rated)
        & np.isfinite(actual)
        & np.isfinite(predicted)
        & np.isfinite(actual_ratio)
        & (rated > 0)
        & (actual_ratio >= minimum_power_ratio)
        & (actual_ratio <= maximum_power_ratio)
        & np.isfinite(underproduction)
    )
    result["underproduction_residual"] = underproduction.astype(np.float32)
    result["ewma_eligible"] = eligible
    return result


def _robust_center_scale(values: np.ndarray) -> tuple[float, float]:
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-6:
        scale = float(np.std(values))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    return center, scale


def _ewma_scores(
    frame: pd.DataFrame,
    *,
    lambda_: float,
    center: float,
    scale: float,
    expected_interval_minutes: int,
) -> np.ndarray:
    values = frame["underproduction_residual"].to_numpy(dtype=np.float64)
    eligible = frame["ewma_eligible"].to_numpy(dtype=bool)
    times = frame["target_time"].to_numpy(dtype="datetime64[ns]")
    scores = np.full(len(frame), np.nan, dtype=np.float64)
    expected_ns = np.timedelta64(expected_interval_minutes, "m")
    for _, positions in frame.groupby("device_no", sort=False, observed=True).indices.items():
        indexes = np.asarray(positions, dtype=np.int64)
        previous_time: np.datetime64 | None = None
        state = 0.0
        for index in indexes:
            contiguous = previous_time is not None and times[index] - previous_time == expected_ns
            if not eligible[index] or not contiguous:
                state = 0.0
            if eligible[index]:
                standardized = (values[index] - center) / scale
                state = lambda_ * standardized + (1.0 - lambda_) * state
                scores[index] = state
            previous_time = times[index]
    return scores


def fit_ewma_calibration(
    frame: pd.DataFrame,
    *,
    lambda_: float = 0.2,
    quantile: float = 0.995,
    minimum_power_ratio: float = 0.10,
    maximum_power_ratio: float = 1.10,
    minimum_consecutive: int = 2,
    expected_interval_minutes: int = 5,
) -> tuple[EWMACalibration, dict[str, Any]]:
    """Fit robust center/scale and an empirical upper EWMA control limit."""
    if not 0 < lambda_ <= 1:
        raise ValueError("lambda must be in (0, 1]")
    if not 0.5 < quantile < 1:
        raise ValueError("quantile must be in (0.5, 1)")
    if minimum_consecutive < 1 or expected_interval_minutes < 1:
        raise ValueError("EWMA settings must be positive")
    prepared = prepare_residual_frame(
        frame,
        minimum_power_ratio=minimum_power_ratio,
        maximum_power_ratio=maximum_power_ratio,
    )
    values = prepared.loc[prepared["ewma_eligible"], "underproduction_residual"].to_numpy(
        dtype=np.float64
    )
    if not len(values):
        raise ValueError("No eligible validation residuals remain for EWMA calibration")
    center, scale = _robust_center_scale(values)
    scores = _ewma_scores(
        prepared,
        lambda_=lambda_,
        center=center,
        scale=scale,
        expected_interval_minutes=expected_interval_minutes,
    )
    finite_scores = scores[np.isfinite(scores)]
    threshold = float(np.quantile(finite_scores, quantile))
    calibration = EWMACalibration(
        lambda_=lambda_,
        quantile=quantile,
        threshold=threshold,
        center=center,
        scale=scale,
        minimum_power_ratio=minimum_power_ratio,
        maximum_power_ratio=maximum_power_ratio,
        minimum_consecutive=minimum_consecutive,
        expected_interval_minutes=expected_interval_minutes,
    )
    report = {
        "samples": len(prepared),
        "eligible_samples": int(prepared["ewma_eligible"].sum()),
        "eligible_percent": float(prepared["ewma_eligible"].mean() * 100),
        "underproduction_center": center,
        "underproduction_scale": scale,
        "ewma_score_quantile": quantile,
        "threshold": threshold,
    }
    return calibration, report


def apply_ewma(frame: pd.DataFrame, calibration: EWMACalibration) -> pd.DataFrame:
    """Apply a saved calibration and consecutive-alert rule to residual rows."""
    prepared = prepare_residual_frame(
        frame,
        minimum_power_ratio=calibration.minimum_power_ratio,
        maximum_power_ratio=calibration.maximum_power_ratio,
    )
    prepared["ewma_score"] = _ewma_scores(
        prepared,
        lambda_=calibration.lambda_,
        center=calibration.center,
        scale=calibration.scale,
        expected_interval_minutes=calibration.expected_interval_minutes,
    ).astype(np.float32)
    raw_alert = prepared["ewma_eligible"] & (prepared["ewma_score"] > calibration.threshold)
    consecutive = np.zeros(len(prepared), dtype=np.int32)
    times = prepared["target_time"].to_numpy(dtype="datetime64[ns]")
    expected_ns = np.timedelta64(calibration.expected_interval_minutes, "m")
    for _, positions in prepared.groupby("device_no", sort=False, observed=True).indices.items():
        indexes = np.asarray(positions, dtype=np.int64)
        previous_time: np.datetime64 | None = None
        run = 0
        for index in indexes:
            contiguous = previous_time is not None and times[index] - previous_time == expected_ns
            if not contiguous:
                run = 0
            run = run + 1 if bool(raw_alert.iloc[index]) else 0
            consecutive[index] = run
            previous_time = times[index]
    prepared["ewma_raw_alert"] = raw_alert.astype("int8")
    prepared["ewma_consecutive"] = consecutive
    prepared["ewma_alert"] = (
        raw_alert & (consecutive >= calibration.minimum_consecutive)
    ).astype("int8")
    return prepared


def save_calibration(calibration: EWMACalibration, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(calibration.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_calibration(path: str | Path) -> EWMACalibration:
    return EWMACalibration.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
