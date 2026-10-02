"""Audit one plant's exported device Parquet partitions without contacting ES."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


REQUIRED_COLUMNS = (
    "event_time",
    "plant_id",
    "device_no",
    "active_power",
    "rated_power",
    "main_string_count",
    "valid_current_string_count",
    "string_overall_status",
)
KEY_COLUMNS = ["event_time", "plant_id", "device_no"]


def _files(root: Path) -> list[Path]:
    files = sorted(root.glob("date=*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No date-partitioned Parquet files under {root}")
    return files


def _partition_date(path: Path) -> str:
    name = path.parent.name
    if not name.startswith("date="):
        raise ValueError(f"Unexpected partition directory: {path.parent}")
    return name.removeprefix("date=")


def _gap_summary(frame: pd.DataFrame, expected_interval_minutes: int) -> dict[str, int]:
    gap_segments = 0
    estimated_missing = 0
    irregular_intervals = 0
    for _, device_frame in frame.groupby("device_no", dropna=False):
        deltas = (
            device_frame.sort_values("event_time")["event_time"]
            .diff()
            .dt.total_seconds()
            .div(60)
            .dropna()
        )
        gap_segments += int((deltas > expected_interval_minutes).sum())
        estimated_missing += int(
            sum(
                max(round(float(delta) / expected_interval_minutes) - 1, 0)
                for delta in deltas
                if delta > expected_interval_minutes
            )
        )
        irregular_intervals += int(
            ((deltas > 0) & (deltas != expected_interval_minutes)).sum()
        )
    return {
        "gap_segments": gap_segments,
        "estimated_missing_points": estimated_missing,
        "non_five_minute_intervals": irregular_intervals,
    }


def audit_parquet(
    root: Path,
    *,
    plant_id: int | None = None,
    timezone_name: str = "Asia/Shanghai",
    expected_interval_minutes: int = 5,
) -> dict[str, Any]:
    """Return a local-only integrity and continuity report for one plant."""
    if plant_id is not None and (plant_id <= 0 or isinstance(plant_id, bool)):
        raise ValueError("plant_id must be a positive integer")
    if expected_interval_minutes <= 0:
        raise ValueError("expected_interval_minutes must be positive")

    files = _files(root)
    frames: list[pd.DataFrame] = []
    missing_fields: dict[str, int] = {field: 0 for field in REQUIRED_COLUMNS}
    rows_by_partition: dict[str, int] = {}
    file_summaries: list[dict[str, Any]] = []

    for path in files:
        metadata = pq.read_metadata(path)
        available_schema = set(pq.read_schema(path).names)
        missing = [field for field in REQUIRED_COLUMNS if field not in available_schema]
        for field in missing:
            missing_fields[field] += metadata.num_rows
        available = [field for field in REQUIRED_COLUMNS if field in available_schema]
        frame = pd.read_parquet(path, columns=available)
        for field in set(REQUIRED_COLUMNS) - set(missing):
            missing_fields[field] += int(frame[field].isna().sum())
        partition = _partition_date(path)
        rows_by_partition[partition] = rows_by_partition.get(partition, 0) + len(frame)
        file_summaries.append(
            {"path": str(path), "rows": len(frame), "missing_fields": missing}
        )
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    result: dict[str, Any] = {
        "root": str(root),
        "plant_id_expected": plant_id,
        "files": len(files),
        "rows": len(combined),
        "size_mb": round(sum(path.stat().st_size for path in files) / 1024 / 1024, 3),
        "rows_by_partition": dict(sorted(rows_by_partition.items())),
        "missing_required_fields": missing_fields,
        "file_summaries": file_summaries,
    }

    if "event_time" not in combined:
        return result

    timestamps = pd.to_datetime(combined["event_time"], errors="coerce", utc=True)
    result["invalid_event_time"] = int(timestamps.isna().sum())
    valid = timestamps.notna()
    if valid.any():
        local_dates = timestamps[valid].dt.tz_convert(timezone_name).dt.date.astype(str)
        result["minimum_utc"] = timestamps[valid].min().isoformat()
        result["maximum_utc"] = timestamps[valid].max().isoformat()
        result["rows_by_local_date"] = {
            str(day): int(count)
            for day, count in local_dates.value_counts().sort_index().items()
        }
    else:
        result["minimum_utc"] = None
        result["maximum_utc"] = None
        result["rows_by_local_date"] = {}

    if "plant_id" in combined:
        result["plants"] = {
            str(value): int(count)
            for value, count in combined["plant_id"].value_counts(dropna=False).items()
        }
        if plant_id is not None:
            result["plant_id_mismatches"] = int(
                (combined["plant_id"].notna() & (combined["plant_id"] != plant_id)).sum()
            )

    if all(column in combined for column in KEY_COLUMNS):
        keyed = combined.loc[:, KEY_COLUMNS].copy()
        keyed["event_time"] = timestamps
        result["duplicate_keys"] = int(keyed.duplicated(KEY_COLUMNS).sum())

    mismatches = 0
    for path in files:
        part = pd.read_parquet(path, columns=["event_time"])
        part_time = pd.to_datetime(part["event_time"], errors="coerce", utc=True)
        local = part_time.dt.tz_convert(timezone_name).dt.date.astype("string")
        mismatches += int((local.notna() & (local != _partition_date(path))).sum())
    result["partition_date_mismatches"] = mismatches

    if "device_no" in combined and valid.any():
        timed = combined.loc[valid, ["device_no"]].copy()
        timed["event_time"] = timestamps[valid]
        result["devices"] = int(timed["device_no"].nunique(dropna=True))
        result["rows_by_device"] = {
            str(value): int(count)
            for value, count in timed["device_no"].value_counts(dropna=False).items()
        }
        result["continuity"] = _gap_summary(timed, expected_interval_minutes)

    for field in (
        "main_string_count",
        "valid_current_string_count",
        "string_overall_status",
    ):
        if field in combined:
            result[f"{field}_counts"] = {
                str(value): int(count)
                for value, count in combined[field].value_counts(dropna=False).items()
            }

    return result
