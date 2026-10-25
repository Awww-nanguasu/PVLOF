"""Infer configured string channels for self-service calibration."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


CURRENT_PATTERN = re.compile(r"^string_current_(\d{2})$")
CHANNEL_INVENTORY_VERSION = "pvlof-channel-inventory-v2-relative-current"


def infer_channel_inventory(
    frame: pd.DataFrame,
    *,
    minimum_evidence_samples: int = 12,
    minimum_evidence_days: int = 3,
    minimum_reference_strings: int = 4,
    minimum_reference_current: float = 0.5,
    minimum_channel_current: float = 0.25,
    minimum_relative_current: float = 0.10,
    timezone: str = "Asia/Shanghai",
    zero_current_threshold: float = 0.0,
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    if minimum_evidence_samples < 1 or minimum_evidence_days < 1:
        raise ValueError("channel evidence sample/day minimums must be positive")
    if minimum_reference_strings < 1:
        raise ValueError("minimum_reference_strings must be positive")
    if minimum_reference_current <= zero_current_threshold:
        raise ValueError(
            "minimum_reference_current must exceed zero_current_threshold"
        )
    if minimum_channel_current <= zero_current_threshold:
        raise ValueError(
            "minimum_channel_current must exceed zero_current_threshold"
        )
    if not 0 < minimum_relative_current <= 1:
        raise ValueError("minimum_relative_current must be in (0, 1]")
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
    source["event_time"] = pd.to_datetime(
        source["event_time"], errors="raise", utc=True
    )
    source["_event_date"] = (
        source["event_time"].dt.tz_convert(timezone).dt.date
    )
    inventory: dict[str, list[int]] = {}
    device_reports: dict[str, dict[str, Any]] = {}

    for (plant, device), group in source.groupby(
        ["plant_id", "device_no"], observed=True, sort=True
    ):
        key = f"{plant}|{device}"
        selected: list[int] = []
        channel_reports: dict[str, dict[str, Any]] = {}
        current_columns = [column for _, column in strings]
        current_frame = group[current_columns].apply(
            pd.to_numeric, errors="coerce"
        )
        current_values = current_frame.to_numpy(dtype=float)
        positive_values = (
            np.isfinite(current_values)
            & (current_values > zero_current_threshold)
        )
        positive_counts = positive_values.sum(axis=1)
        reference_values = np.full(len(group), np.nan, dtype=float)
        reference_rows = positive_counts >= minimum_reference_strings
        if reference_rows.any():
            usable = np.where(positive_values[reference_rows], current_values[reference_rows], np.nan)
            reference_values[reference_rows] = np.nanmedian(usable, axis=1)
        reference_valid = (
            np.isfinite(reference_values)
            & (reference_values >= minimum_reference_current)
        )
        reference = pd.Series(reference_values, index=group.index)
        reference_is_valid = pd.Series(reference_valid, index=group.index)
        valid_reference_values = reference_values[reference_valid]
        device_reference_current = (
            float(np.quantile(valid_reference_values, 0.95))
            if len(valid_reference_values)
            else None
        )

        for number, current_column in strings:
            current = current_frame[current_column]
            positive = current.gt(zero_current_threshold)
            relative_threshold = reference * minimum_relative_current
            meaningful_current = (
                reference_is_valid
                & current.notna()
                & current.ge(minimum_channel_current)
                & current.ge(relative_threshold)
            )
            status_column = f"string_status_{number:02d}"
            if status_column in group:
                status = pd.to_numeric(group[status_column], errors="coerce")
                status_observed = status.notna()
                status_evidence = status_observed & status.ne(0)
            else:
                status = pd.Series(np.nan, index=group.index)
                status_observed = pd.Series(False, index=group.index)
                status_evidence = pd.Series(False, index=group.index)
            # Status fields are retained as diagnostics only. In production
            # data they can stay non-zero for an unused channel or remain zero
            # for a channel carrying normal current, so they cannot define the
            # inventory by themselves.
            evidence = meaningful_current
            positive_without_status = positive & status_observed & status.eq(0)
            samples = int(evidence.sum())
            days = int(group.loc[evidence, "_event_date"].nunique())
            configured = (
                samples >= minimum_evidence_samples
                and days >= minimum_evidence_days
            )
            if configured:
                selected.append(number)
            conflict_samples = int(positive_without_status.sum())
            status_only_samples = int((status_evidence & ~meaningful_current).sum())
            meaningful_with_zero_status = int(
                (meaningful_current & status_observed & status.eq(0)).sum()
            )
            ratios = (current / reference).where(reference_is_valid)
            finite_ratios = ratios[np.isfinite(ratios)]
            reference_currents = current[reference_is_valid & current.notna()]
            if configured or samples or conflict_samples or int(status_evidence.sum()):
                channel_reports[f"{number:02d}"] = {
                    "configured": configured,
                    "evidence_samples": samples,
                    "evidence_days": days,
                    "positive_current_samples": int(positive.sum()),
                    "meaningful_current_samples": samples,
                    "nonzero_status_samples": int(status_evidence.sum()),
                    "status_observed_samples": int(status_observed.sum()),
                    "positive_with_zero_status_samples": conflict_samples,
                    "meaningful_with_zero_status_samples": (
                        meaningful_with_zero_status
                    ),
                    "status_only_samples": status_only_samples,
                    "p95_current": (
                        float(reference_currents.quantile(0.95))
                        if len(reference_currents)
                        else None
                    ),
                    "p95_relative_to_device_reference": (
                        float(finite_ratios.quantile(0.95))
                        if len(finite_ratios)
                        else None
                    ),
                    "evidence_source": "relative_current",
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
            "reference_eligible_samples": int(reference_valid.sum()),
            "device_reference_current_p95": device_reference_current,
            "count_difference": (
                len(selected) - main_count_mode
                if main_count_mode is not None
                else None
            ),
            "noncontiguous": bool(
                selected and selected != list(range(1, len(selected) + 1))
            ),
            "channels": channel_reports,
        }

    mismatch = sum(
        report["count_difference"] not in (None, 0)
        for report in device_reports.values()
    )
    noncontiguous = sum(
        report["noncontiguous"] for report in device_reports.values()
    )
    report = {
        "version": CHANNEL_INVENTORY_VERSION,
        "minimum_evidence_samples": minimum_evidence_samples,
        "minimum_evidence_days": minimum_evidence_days,
        "minimum_reference_strings": minimum_reference_strings,
        "minimum_reference_current": minimum_reference_current,
        "minimum_channel_current": minimum_channel_current,
        "minimum_relative_current": minimum_relative_current,
        "status_fields_are_auxiliary": True,
        "timezone": timezone,
        "devices": len(inventory),
        "configured_channels": sum(len(values) for values in inventory.values()),
        "devices_with_count_mismatch": int(mismatch),
        "devices_with_noncontiguous_channels": int(noncontiguous),
        "device_reports": device_reports,
    }
    return inventory, report


__all__ = ["CHANNEL_INVENTORY_VERSION", "infer_channel_inventory"]
