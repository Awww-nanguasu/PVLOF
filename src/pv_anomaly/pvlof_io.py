"""Parquet input helpers for the PVLOF offline pipeline."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd


PARTITION_PATTERN = re.compile(r"date=(\d{4}-\d{2}-\d{2})")


def _time_bounds(start: str, end: str, timezone: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_time = pd.Timestamp(start)
    end_time = pd.Timestamp(end)
    if start_time.tzinfo is None:
        start_time = start_time.tz_localize(timezone)
    if end_time.tzinfo is None:
        end_time = end_time.tz_localize(timezone)
    start_utc = start_time.tz_convert("UTC")
    end_utc = end_time.tz_convert("UTC")
    if end_utc <= start_utc:
        raise ValueError("end must be later than start")
    return start_utc, end_utc


def _candidate_files(source: Path, start: str, end: str) -> list[Path]:
    files = [source] if source.is_file() else sorted(source.rglob("*.parquet"))
    start_date = pd.Timestamp(start).date()
    end_date = pd.Timestamp(end).date()
    selected: list[Path] = []
    for file in files:
        match = PARTITION_PATTERN.search(str(file))
        if match is None:
            selected.append(file)
            continue
        partition_date = pd.Timestamp(match.group(1)).date()
        if start_date <= partition_date < end_date:
            selected.append(file)
    return selected


def read_pvlof_source(
    path: str | Path,
    *,
    start: str,
    end: str,
    timezone: str = "Asia/Shanghai",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read only PVLOF columns from local date-partitioned raw Parquet."""
    source = Path(path)
    files = _candidate_files(source, start, end)
    if not files:
        raise FileNotFoundError(f"No Parquet files found for [{start}, {end}) under {path}")
    first_columns = pd.read_parquet(files[0]).columns.tolist()
    metadata = [
        "event_time",
        "plant_id",
        "device_no",
        "active_power",
        "rated_power",
        "status_code",
        "main_string_count",
        "valid_current_string_count",
        "string_overall_status",
    ]
    string_columns = [
        column
        for column in first_columns
        if column.startswith("string_current_") or column.startswith("string_status_")
    ]
    columns = [column for column in metadata + string_columns if column in first_columns]
    frames: list[pd.DataFrame] = []
    errors: list[str] = []
    for file in files:
        try:
            frame = pd.read_parquet(file, columns=columns)
        except Exception as error:
            errors.append(f"{file}: {error}")
            continue
        frames.append(frame)
    if not frames:
        raise ValueError(f"All PVLOF Parquet reads failed; examples: {errors[:3]}")
    result = pd.concat(frames, ignore_index=True)
    rows_read = len(result)
    result["event_time"] = pd.to_datetime(result["event_time"], errors="raise", utc=True)
    start_utc, end_utc = _time_bounds(start, end, timezone)
    result = result[(result["event_time"] >= start_utc) & (result["event_time"] < end_utc)]
    duplicate = result.duplicated(["device_no", "event_time"], keep="last")
    duplicate_rows = int(duplicate.sum())
    if duplicate_rows:
        result = result.loc[~duplicate]
    result = result.sort_values(["event_time", "device_no"]).reset_index(drop=True)
    return result, {
        "path": str(path),
        "start": start,
        "end_exclusive": end,
        "timezone": timezone,
        "files_selected": len(files),
        "files_read": len(frames),
        "files_failed": len(errors),
        "error_examples": errors[:3],
        "rows_read": rows_read,
        "rows_in_range": len(result),
        "duplicate_rows_removed": duplicate_rows,
        "devices": int(result["device_no"].nunique()) if len(result) else 0,
        "string_current_columns": sum(column.startswith("string_current_") for column in columns),
    }
