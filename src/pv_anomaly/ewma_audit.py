"""Event and weak-label audits for EWMA alert streams."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def read_partitioned_columns(
    path: str | Path,
    columns: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read partitioned Parquet files one-by-one to avoid mixed-schema failures."""
    source = Path(path)
    files = [source] if source.is_file() else sorted(source.rglob("*.parquet"))
    frames: list[pd.DataFrame] = []
    errors: list[str] = []
    for file in files:
        try:
            frame = pd.read_parquet(file, columns=columns)
        except Exception:
            try:
                frame = pd.read_parquet(file)
                selected = [column for column in columns if column in frame.columns]
                frame = frame[selected]
            except Exception as error:
                errors.append(f"{file}: {error}")
                continue
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
    return result, {
        "path": str(path),
        "files_found": len(files),
        "files_read": len(frames),
        "files_failed": len(errors),
        "error_examples": errors[:3],
        "rows": len(result),
        "columns_found": [column for column in columns if column in result.columns],
    }


def binary_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    truth = actual.astype(bool).to_numpy()
    guess = predicted.astype(bool).to_numpy()
    tp = int(np.sum(truth & guess))
    fp = int(np.sum(~truth & guess))
    fn = int(np.sum(truth & ~guess))
    tn = int(np.sum(~truth & ~guess))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "samples": len(actual),
        "positive_labels": int(truth.sum()),
        "positive_predictions": int(guess.sum()),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def add_weak_label(
    alert_frame: pd.DataFrame,
    source_frame: pd.DataFrame,
    *,
    label_statuses: tuple[int, ...] = (2, 4),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Align target-time weak labels to EWMA rows without using future labels."""
    required = {"device_no", "target_time"}
    missing = sorted(required - set(alert_frame.columns))
    if missing:
        raise ValueError(f"Missing alert columns: {missing}")
    source = source_frame.copy()
    if "event_time" not in source or "device_no" not in source:
        raise ValueError("Weak-label source requires event_time and device_no")
    source["target_time"] = pd.to_datetime(source["event_time"], errors="raise", utc=True)
    source["device_no"] = source["device_no"].astype(str)
    label_columns = ["device_no", "target_time"]
    if "string_overall_status" in source:
        label_columns.append("string_overall_status")
    if "low_current_count" in source:
        label_columns.append("low_current_count")
    source = source[label_columns].drop_duplicates(["device_no", "target_time"])
    source["weak_current_label"] = False
    if "string_overall_status" in source:
        status = pd.to_numeric(source["string_overall_status"], errors="coerce")
        source["weak_current_label"] |= status.isin(label_statuses)
    if "low_current_count" in source:
        count = pd.to_numeric(source["low_current_count"], errors="coerce")
        source["weak_current_label"] |= count.gt(0)
    labels = source[["device_no", "target_time", "weak_current_label"]]
    result = alert_frame.copy()
    result["device_no"] = result["device_no"].astype(str)
    result["target_time"] = pd.to_datetime(result["target_time"], errors="raise", utc=True)
    result = result.merge(labels, on=["device_no", "target_time"], how="left")
    matched = result["weak_current_label"].notna()
    result["weak_current_label"] = result["weak_current_label"].fillna(False).astype(bool)
    return result, {
        "label_statuses": list(label_statuses),
        "matched_rows": int(matched.sum()),
        "unmatched_rows": int((~matched).sum()),
        "matched_percent": float(matched.mean() * 100),
        "positive_labels_after_alignment": int(result["weak_current_label"].sum()),
        "source_status_counts": (
            source["string_overall_status"].value_counts(dropna=False).to_dict()
            if "string_overall_status" in source
            else {}
        ),
    }


def collapse_alert_events(
    frame: pd.DataFrame,
    *,
    alert_column: str,
    expected_interval_minutes: int = 5,
    event_prefix: str = "event",
) -> pd.DataFrame:
    """Collapse consecutive alerts for one device into one operational event."""
    required = {"device_no", "target_time", alert_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing event columns: {missing}")
    source = frame.copy()
    source["target_time"] = pd.to_datetime(source["target_time"], errors="raise", utc=True)
    alerts = source[source[alert_column].astype(bool)].copy()
    if alerts.empty:
        return pd.DataFrame(
            columns=[
                "event_id",
                "device_no",
                "start_time",
                "end_time",
                "points",
                "duration_minutes",
            ]
        )
    alerts = alerts.sort_values(["device_no", "target_time"]).reset_index(drop=True)
    event_ids = np.zeros(len(alerts), dtype=np.int64)
    event_number = 0
    previous_device: str | None = None
    previous_time: pd.Timestamp | None = None
    interval = pd.Timedelta(minutes=expected_interval_minutes)
    for index, row in alerts.iterrows():
        device = str(row["device_no"])
        timestamp = row["target_time"]
        if (
            previous_device != device
            or previous_time is None
            or timestamp - previous_time != interval
        ):
            event_number += 1
        event_ids[index] = event_number
        previous_device = device
        previous_time = timestamp
    alerts["_event_number"] = event_ids
    aggregations: dict[str, tuple[str, str]] = {
        "start_time": ("target_time", "min"),
        "end_time": ("target_time", "max"),
        "points": ("target_time", "size"),
    }
    if "ewma_score" in alerts:
        aggregations["max_ewma_score"] = ("ewma_score", "max")
    if "underproduction_residual" in alerts:
        aggregations["max_underproduction_residual"] = (
            "underproduction_residual",
            "max",
        )
    events = (
        alerts.groupby(["device_no", "_event_number"], observed=True)
        .agg(**aggregations)
        .reset_index()
        .drop(columns="_event_number")
    )
    events["duration_minutes"] = (
        events["end_time"] - events["start_time"]
    ).dt.total_seconds() / 60 + expected_interval_minutes
    events.insert(
        0,
        "event_id",
        [f"{event_prefix}-{index:05d}" for index in range(1, len(events) + 1)],
    )
    return events


def match_event_tables(
    predicted_events: pd.DataFrame,
    label_events: pd.DataFrame,
) -> dict[str, Any]:
    """Match events when the same device has any overlapping time point."""
    if predicted_events.empty or label_events.empty:
        predicted_hits = 0
        label_hits = 0
    else:
        predicted_hits = 0
        for _, predicted in predicted_events.iterrows():
            candidates = label_events[
                (label_events["device_no"].astype(str) == str(predicted["device_no"]))
                & (label_events["start_time"] <= predicted["end_time"])
                & (label_events["end_time"] >= predicted["start_time"])
            ]
            predicted_hits += int(not candidates.empty)
        label_hits = 0
        for _, label in label_events.iterrows():
            candidates = predicted_events[
                (predicted_events["device_no"].astype(str) == str(label["device_no"]))
                & (predicted_events["start_time"] <= label["end_time"])
                & (predicted_events["end_time"] >= label["start_time"])
            ]
            label_hits += int(not candidates.empty)
    predicted_count = len(predicted_events)
    label_count = len(label_events)
    precision = predicted_hits / predicted_count if predicted_count else 0.0
    recall = label_hits / label_count if label_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "predicted_events": predicted_count,
        "weak_label_events": label_count,
        "predicted_events_hit": predicted_hits,
        "weak_label_events_hit": label_hits,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def distribution_table(
    frame: pd.DataFrame,
    *,
    group_columns: list[str],
    alert_column: str = "ewma_alert",
) -> pd.DataFrame:
    """Count alert points and weak labels by date/device or other groups."""
    source = frame.copy()
    source["local_date"] = (
        pd.to_datetime(source["target_time"], utc=True)
        .dt.tz_convert("Asia/Shanghai")
        .dt.strftime("%Y-%m-%d")
    )
    aggregations = {
        "samples": ("target_time", "size"),
        "eligible_samples": ("ewma_eligible", "sum"),
        "alert_points": (alert_column, "sum"),
    }
    if "weak_current_label" in source:
        aggregations["weak_label_points"] = ("weak_current_label", "sum")
    return (
        source.groupby(group_columns, observed=True, dropna=False)
        .agg(**aggregations)
        .reset_index()
        .sort_values("alert_points", ascending=False)
    )


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, float) and math.isnan(value):
        return None
    raise TypeError(f"Cannot serialize {type(value).__name__}")
