"""Export raw Parquet data as date-partitioned wide CSV files.

The input files are treated as raw data.  No PVLOF/model columns are added,
removed, renamed, or otherwise transformed.  The only operation used to
choose an output partition is converting the existing timestamp to the local
calendar date.

Output layout::

    <output>/device/plant_id=234/date=2026-03-09.csv
    <output>/weather_15min/plant_id=234/date=2026-03-09.csv
    <output>/device_alarm_70133702.csv
    <output>/manifest.json

The device and weather Parquet files are already wide: one inverter/weather
record per row and string/weather fields in columns.  CSV is written with a
UTF-8 BOM so it opens correctly in common Windows spreadsheet applications.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


DATASET_SPECS: dict[str, tuple[str, str, str]] = {
    "device": ("event_time", "plant_id", "device"),
    "weather_15min": ("time", "plant_id", "weather_15min"),
}


def _parse_plant_ids(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    result: set[str] = set()
    for value in values:
        result.update(item.strip() for item in value.split(",") if item.strip())
    return result or None


def _timestamp_column(frame: pd.DataFrame, column: str, path: Path) -> pd.Series:
    if column not in frame.columns:
        raise ValueError(f"{path} is missing required timestamp column {column!r}")
    timestamps = pd.to_datetime(frame[column], errors="coerce", utc=True)
    if timestamps.isna().any():
        count = int(timestamps.isna().sum())
        raise ValueError(f"{path} contains {count} invalid {column} values")
    return timestamps


def _safe_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    return value


def _write_dataset(
    *,
    source_root: Path,
    output_root: Path,
    dataset_name: str,
    plant_ids: set[str] | None,
    timezone: str,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
) -> dict[str, Any]:
    timestamp_field, plant_field, output_name = DATASET_SPECS[dataset_name]
    files = sorted(source_root.rglob("*.parquet")) if source_root.exists() else []
    if not files:
        return {
            "source": str(source_root),
            "output": str(output_root / output_name),
            "files_read": 0,
            "rows_read": 0,
            "rows_written": 0,
            "plants": {},
            "columns": [],
        }

    # A day contains only a few thousand device rows (or a few hundred weather
    # rows), so grouping one day at a time keeps memory bounded while avoiding
    # append-with-header problems when several Parquet parts share a date.
    grouped: dict[tuple[str, str], list[pd.DataFrame]] = defaultdict(list)
    rows_read = 0
    files_read = 0
    columns: list[str] = []
    source_min: pd.Timestamp | None = None
    source_max: pd.Timestamp | None = None

    for path in files:
        frame = pd.read_parquet(path)
        files_read += 1
        rows_read += len(frame)
        if not columns:
            columns = [str(column) for column in frame.columns]
        elif [str(column) for column in frame.columns] != columns:
            raise ValueError(f"Raw schema differs between Parquet files: {path}")

        timestamps = _timestamp_column(frame, timestamp_field, path)
        if start is not None:
            keep = timestamps >= start
        else:
            keep = pd.Series(True, index=frame.index)
        if end is not None:
            keep &= timestamps < end
        if not keep.any():
            continue
        frame = frame.loc[keep].copy()
        timestamps = timestamps.loc[keep]

        if plant_field not in frame.columns:
            raise ValueError(f"{path} is missing required plant column {plant_field!r}")
        plant_values = frame[plant_field].astype(str)
        if plant_ids is not None:
            keep_plant = plant_values.isin(plant_ids)
            frame = frame.loc[keep_plant].copy()
            timestamps = timestamps.loc[keep_plant]
            plant_values = plant_values.loc[keep_plant]
        if frame.empty:
            continue

        local_dates = timestamps.dt.tz_convert(timezone).dt.strftime("%Y-%m-%d")
        for (plant, date), indices in frame.groupby(
            [plant_values, local_dates], sort=False, dropna=False
        ).groups.items():
            grouped[(str(plant), str(date))].append(frame.loc[indices])
        current_min = timestamps.min()
        current_max = timestamps.max()
        source_min = current_min if source_min is None else min(source_min, current_min)
        source_max = current_max if source_max is None else max(source_max, current_max)

    written = 0
    plant_summary: dict[str, dict[str, Any]] = {}
    for (plant, date), frames in sorted(grouped.items()):
        destination = output_root / output_name / f"plant_id={plant}" / f"date={date}.csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        result = pd.concat(frames, ignore_index=True)
        # Keep the raw timestamp values exactly as stored in Parquet.  The
        # local date is represented by the output directory; it is not added
        # as a derived data column.
        result.to_csv(destination, index=False, encoding="utf-8-sig")
        written += len(result)
        item = plant_summary.setdefault(
            plant,
            {"files": 0, "rows": 0, "dates": [], "minimum_utc": None, "maximum_utc": None},
        )
        item["files"] += 1
        item["rows"] += len(result)
        item["dates"].append(date)
        parsed = pd.to_datetime(result[timestamp_field], utc=True)
        low, high = parsed.min().isoformat(), parsed.max().isoformat()
        item["minimum_utc"] = low if item["minimum_utc"] is None else min(item["minimum_utc"], low)
        item["maximum_utc"] = high if item["maximum_utc"] is None else max(item["maximum_utc"], high)

    return {
        "source": str(source_root),
        "output": str(output_root / output_name),
        "files_read": files_read,
        "rows_read": rows_read,
        "rows_written": written,
        "columns": columns,
        "plants": plant_summary,
        "minimum_utc": source_min.isoformat() if source_min is not None else None,
        "maximum_utc": source_max.isoformat() if source_max is not None else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-root", type=Path, default=Path("data/raw/device"))
    parser.add_argument("--weather-root", type=Path, default=Path("data/raw/weather_15min"))
    parser.add_argument("--alarm-csv", type=Path, default=Path("device_alarm_70133702.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/raw_delivery_csv"))
    parser.add_argument("--plant-ids", nargs="*", help="IDs or comma-separated IDs to include")
    parser.add_argument("--start", help="Inclusive UTC timestamp, e.g. 2026-01-01")
    parser.add_argument("--end", help="Exclusive UTC timestamp, e.g. 2026-09-01")
    parser.add_argument("--timezone", default="Asia/Shanghai")
    args = parser.parse_args()

    plant_ids = _parse_plant_ids(args.plant_ids)
    start = pd.Timestamp(args.start, tz="UTC") if args.start else None
    end = pd.Timestamp(args.end, tz="UTC") if args.end else None
    if start is not None and end is not None and start >= end:
        raise ValueError("--start must be earlier than --end")

    args.output.mkdir(parents=True, exist_ok=True)
    datasets = [
        _write_dataset(
            source_root=args.device_root,
            output_root=args.output,
            dataset_name="device",
            plant_ids=plant_ids,
            timezone=args.timezone,
            start=start,
            end=end,
        ),
        _write_dataset(
            source_root=args.weather_root,
            output_root=args.output,
            dataset_name="weather_15min",
            plant_ids=plant_ids,
            timezone=args.timezone,
            start=start,
            end=end,
        ),
    ]

    alarm_output = None
    if args.alarm_csv.exists():
        alarm_output = args.output / args.alarm_csv.name
        shutil.copy2(args.alarm_csv, alarm_output)

    manifest = {
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "purpose": "raw wide CSV delivery; no algorithm/model fields were added",
        "timezone_for_date_partitions": args.timezone,
        "timestamp_filter": {
            "start_inclusive_utc": start.isoformat() if start is not None else None,
            "end_exclusive_utc": end.isoformat() if end is not None else None,
        },
        "plant_ids": sorted(plant_ids) if plant_ids is not None else "all",
        "datasets": datasets,
        "alarm_csv": str(alarm_output) if alarm_output is not None else None,
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
