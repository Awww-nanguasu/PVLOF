"""Infer stable configured string channels without assuming contiguous numbering."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


CURRENT_PATTERN = re.compile(r"^string_current_(\d{2})$")


def infer_channel_inventory(
    frame: pd.DataFrame,
    *,
    minimum_evidence_samples: int = 3,
    minimum_evidence_days: int = 1,
    timezone: str = "Asia/Shanghai",
    zero_current_threshold: float = 0.0,
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    """Infer per-device physical channels from status and historical current evidence."""
    if minimum_evidence_samples < 1 or minimum_evidence_days < 1:
        raise ValueError("channel evidence sample/day minimums must be positive")
    required = {"plant_id", "device_no", "event_time"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Channel inventory input is missing columns: {missing}")
    strings = sorted(
        (int(match.group(1)), column)
        for column in frame.columns
        if (match := CURRENT_PATTERN.match(column))
    )
    if not strings:
        raise ValueError("No string_current_XX columns were found")

    source = frame.copy()
    source["plant_id"] = source["plant_id"].astype(str)
    source["device_no"] = source["device_no"].astype(str)
    source["event_time"] = pd.to_datetime(source["event_time"], errors="raise", utc=True)
    source["_event_date"] = source["event_time"].dt.tz_convert(timezone).dt.date
    inventory: dict[str, list[int]] = {}
    device_reports: dict[str, dict[str, Any]] = {}

    for (plant, device), group in source.groupby(
        ["plant_id", "device_no"], observed=True, sort=True
    ):
        key = f"{plant}|{device}"
        selected: list[int] = []
        channel_reports: dict[str, dict[str, Any]] = {}
        for number, current_column in strings:
            current = pd.to_numeric(group[current_column], errors="coerce")
            positive = current.gt(zero_current_threshold)
            status_column = f"string_status_{number:02d}"
            if status_column in group:
                status = pd.to_numeric(group[status_column], errors="coerce")
                # Any non-zero string state proves that the physical channel exists.
                status_observed = status.notna()
                status_evidence = status_observed & status.ne(0)
            else:
                status = pd.Series(np.nan, index=group.index)
                status_observed = pd.Series(False, index=group.index)
                status_evidence = pd.Series(False, index=group.index)
            # A present status channel is authoritative.  Positive current is
            # only a fallback when status is unavailable/all-null; otherwise
            # sporadic current on status=0 channels is reported as a conflict
            # instead of silently creating a phantom configured channel.
            uses_status = bool(status_observed.any())
            evidence = status_evidence if uses_status else positive
            positive_without_status = positive & status_observed & status.eq(0)
            samples = int(evidence.sum())
            days = int(group.loc[evidence, "_event_date"].nunique())
            configured = (
                samples >= minimum_evidence_samples and days >= minimum_evidence_days
            )
            if configured:
                selected.append(number)
            conflict_samples = int(positive_without_status.sum())
            if configured or samples or conflict_samples:
                channel_reports[f"{number:02d}"] = {
                    "configured": configured,
                    "evidence_samples": samples,
                    "evidence_days": days,
                    "positive_current_samples": int(positive.sum()),
                    "nonzero_status_samples": int(status_evidence.sum()),
                    "status_observed_samples": int(status_observed.sum()),
                    "positive_with_zero_status_samples": conflict_samples,
                    "evidence_source": "status" if uses_status else "positive_current",
                }

        main_count_mode: int | None = None
        if "main_string_count" in group:
            counts = pd.to_numeric(group["main_string_count"], errors="coerce")
            counts = counts[np.isfinite(counts) & counts.gt(0)]
            if len(counts):
                main_count_mode = int(counts.mode().iloc[0])
        inventory[key] = selected
        device_reports[key] = {
            "plant_id": str(plant),
            "device_no": str(device),
            "configured_strings": selected,
            "configured_count": len(selected),
            "main_string_count_mode": main_count_mode,
            "count_difference": (
                len(selected) - main_count_mode if main_count_mode is not None else None
            ),
            "noncontiguous": bool(selected and selected != list(range(1, len(selected) + 1))),
            "channels": channel_reports,
        }

    mismatch = sum(
        report["count_difference"] not in (None, 0) for report in device_reports.values()
    )
    noncontiguous = sum(report["noncontiguous"] for report in device_reports.values())
    report = {
        "version": "pvlof-channel-inventory-v1",
        "minimum_evidence_samples": minimum_evidence_samples,
        "minimum_evidence_days": minimum_evidence_days,
        "timezone": timezone,
        "devices": len(inventory),
        "configured_channels": sum(len(values) for values in inventory.values()),
        "devices_with_count_mismatch": int(mismatch),
        "devices_with_noncontiguous_channels": int(noncontiguous),
        "device_reports": device_reports,
    }
    return inventory, report


__all__ = ["infer_channel_inventory"]
