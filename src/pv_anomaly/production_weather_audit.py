"""Audit one plant's exported 15-minute weather Parquet partitions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


COLUMNS = ("time", "plant_id", "forecast_ghi", "sensor_ghi")
KEY_COLUMNS = ["time", "plant_id"]


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


def _value_counts(series: pd.Series) -> dict[str, int]:
    return {
        str(value): int(count)
        for value, count in series.value_counts(dropna=False).sort_index().items()
    }


def _ghi_summary(series: pd.Series) -> dict[str, int | float | None]:
    values = pd.to_numeric(series, errors="coerce")
    present = int(values.notna().sum())
    total = len(values)
    return {
        "present": present,
        "missing": int(values.isna().sum()),
        "present_percent": round(present / total * 100, 4) if total else 0.0,
        "minimum": float(values.min()) if present else None,
        "maximum": float(values.max()) if present else None,
        "negative_values": int((values < 0).sum()),
        "zero_values": int((values == 0).sum()),
    }


def audit_weather_parquet(
    root: Path,
    *,
    plant_id: int,
    timezone_name: str = "Asia/Shanghai",
    expected_interval_minutes: int = 15,
) -> dict[str, Any]:
    """Return a local integrity, cadence and GHI-coverage report."""
    if isinstance(plant_id, bool) or not isinstance(plant_id, int) or plant_id <= 0:
        raise ValueError("plant_id must be a positive integer")
    if expected_interval_minutes <= 0:
        raise ValueError("expected_interval_minutes must be positive")

    files = _files(root)
    frames: list[pd.DataFrame] = []
    missing_columns = {column: 0 for column in COLUMNS}
    rows_by_partition: dict[str, int] = {}

    for path in files:
        metadata = pq.read_metadata(path)
        schema = set(pq.read_schema(path).names)
        missing = [column for column in COLUMNS if column not in schema]
        for column in missing:
            missing_columns[column] += metadata.num_rows
        available = [column for column in COLUMNS if column in schema]
        frame = pd.read_parquet(path, columns=available)
        for column in set(COLUMNS) - set(missing):
            missing_columns[column] += int(frame[column].isna().sum())
        for column in missing:
            frame[column] = pd.NA
        frame = frame.loc[:, list(COLUMNS)]
        partition = _partition_date(path)
        frame["partition_date"] = partition
        rows_by_partition[partition] = rows_by_partition.get(partition, 0) + len(frame)
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    timestamps = pd.to_datetime(combined["time"], errors="coerce", utc=True)
    valid = timestamps.notna()
    local_time = timestamps.dt.tz_convert(timezone_name)
    local_dates = local_time.dt.date.astype("string")

    result: dict[str, Any] = {
        "root": str(root),
        "plant_id_expected": plant_id,
        "files": len(files),
        "rows": len(combined),
        "size_mb": round(sum(path.stat().st_size for path in files) / 1024 / 1024, 3),
        "missing_columns_or_values": missing_columns,
        "rows_by_partition": dict(sorted(rows_by_partition.items())),
        "invalid_time": int((~valid).sum()),
        "minimum_utc": timestamps[valid].min().isoformat() if valid.any() else None,
        "maximum_utc": timestamps[valid].max().isoformat() if valid.any() else None,
        "minimum_local": local_time[valid].min().isoformat() if valid.any() else None,
        "maximum_local": local_time[valid].max().isoformat() if valid.any() else None,
        "partition_date_mismatches": int(
            (valid & (local_dates != combined["partition_date"].astype("string"))).sum()
        ),
    }

    result["plants"] = _value_counts(combined["plant_id"])
    result["plant_id_mismatches"] = int(
        (combined["plant_id"].notna() & (combined["plant_id"] != plant_id)).sum()
    )

    keyed = combined.loc[:, KEY_COLUMNS].copy()
    keyed["time"] = timestamps
    result["duplicate_keys"] = int(keyed.duplicated(KEY_COLUMNS).sum())
    result["off_15_minute_grid"] = int(
        (
            valid
            & (
                (local_time.dt.minute % expected_interval_minutes != 0)
                | (local_time.dt.second != 0)
            )
        ).sum()
    )

    timed = pd.DataFrame(
        {
            "time": timestamps[valid],
            "local_date": local_dates[valid],
        }
    ).sort_values("time")
    within_day_deltas = timed.groupby("local_date")["time"].diff().dt.total_seconds().div(60)
    result["within_day_interval_minutes"] = _value_counts(within_day_deltas.dropna())
    result["within_day_gap_segments"] = int(
        (within_day_deltas > expected_interval_minutes).sum()
    )
    result["estimated_missing_within_days"] = int(
        sum(
            max(round(float(delta) / expected_interval_minutes) - 1, 0)
            for delta in within_day_deltas.dropna()
            if delta > expected_interval_minutes
        )
    )
    result["rows_by_local_date"] = _value_counts(local_dates[valid])

    for field in ("forecast_ghi", "sensor_ghi"):
        result[field] = _ghi_summary(combined[field])
        present_by_date = combined[field].notna() & valid
        result[f"{field}_present_by_local_date"] = _value_counts(
            local_dates[present_by_date]
        )

    return result
