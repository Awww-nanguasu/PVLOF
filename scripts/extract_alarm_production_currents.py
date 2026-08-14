"""Extract production string currents for cleaned inverter-alarm points."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PLANT_PATTERN = re.compile(r"(?:^|[\\/])plant_id=(\d+)(?:[\\/]|$)")
DATE_PATTERN = re.compile(r"(?:^|[\\/])date=(\d{4}-\d{2}-\d{2})(?:[\\/]|$)")
KEY_COLUMNS = ["plant_id", "device_no", "event_time"]


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is pd.NA or value is None:
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _partition_value(path: Path, pattern: re.Pattern[str]) -> str | None:
    match = pattern.search(str(path))
    return match.group(1) if match else None


def _parquet_columns(path: Path) -> list[str]:
    try:
        import pyarrow.parquet as parquet

        return list(parquet.ParquetFile(path).schema.names)
    except ImportError:
        return list(pd.read_parquet(path).columns)


def _normalize_keys(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    missing = sorted(set(KEY_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")
    result = frame.copy()
    result["plant_id"] = pd.to_numeric(result["plant_id"], errors="coerce").astype("Int64")
    result["device_no"] = result["device_no"].astype(str).str.strip()
    result["event_time"] = pd.to_datetime(result["event_time"], errors="raise", utc=True)
    return result


def _production_columns(columns: list[str]) -> list[str]:
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
        "low_current_count",
        "zero_current_count",
        "current_complete",
        "current_expected_count",
        "current_finite_count",
    ]
    strings = sorted(
        column
        for column in columns
        if column.startswith("string_current_") or column.startswith("string_status_")
    )
    return [column for column in metadata + strings if column in columns]


def _candidate_files(
    root: Path,
    *,
    plant_id: int,
    local_dates: set[str],
) -> list[Path]:
    selected: list[Path] = []
    for path in sorted(root.rglob("*.parquet")):
        path_plant = _partition_value(path, PLANT_PATTERN)
        if path_plant is not None and int(path_plant) != plant_id:
            continue
        path_date = _partition_value(path, DATE_PATTERN)
        if path_date is not None and path_date not in local_dates:
            continue
        selected.append(path)
    return selected


def _read_matching_production(
    root: Path,
    label_keys: pd.DataFrame,
    *,
    timezone: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frames: list[pd.DataFrame] = []
    by_plant: dict[str, Any] = {}
    for plant_value, plant_keys in label_keys.groupby("plant_id", observed=True):
        plant_id = int(plant_value)
        local_dates = set(
            plant_keys["event_time"]
            .dt.tz_convert(timezone)
            .dt.strftime("%Y-%m-%d")
            .unique()
        )
        files = _candidate_files(root, plant_id=plant_id, local_dates=local_dates)
        if not files:
            by_plant[str(plant_id)] = {
                "label_points": int(len(plant_keys)),
                "files_read": 0,
                "matched_rows": 0,
            }
            continue
        key_index = plant_keys.set_index(KEY_COLUMNS).index
        plant_frames: list[pd.DataFrame] = []
        read_errors: list[str] = []
        for path in files:
            available = _parquet_columns(path)
            columns = _production_columns(available)
            if not set(KEY_COLUMNS).issubset(columns):
                read_errors.append(f"{path}: missing key columns")
                continue
            try:
                frame = pd.read_parquet(path, columns=columns)
            except Exception as error:
                read_errors.append(f"{path}: {error}")
                continue
            frame = _normalize_keys(frame, name=str(path))
            frame = frame[frame["plant_id"].eq(plant_id)]
            if frame.empty:
                continue
            matched = frame.set_index(KEY_COLUMNS).index.isin(key_index)
            if matched.any():
                plant_frames.append(frame.loc[matched])
        plant_source = (
            pd.concat(plant_frames, ignore_index=True) if plant_frames else pd.DataFrame()
        )
        if not plant_source.empty:
            frames.append(plant_source)
        by_plant[str(plant_id)] = {
            "label_points": int(len(plant_keys)),
            "files_read": int(len(files) - len(read_errors)),
            "files_failed": int(len(read_errors)),
            "error_examples": read_errors[:3],
            "matched_rows_before_deduplication": int(len(plant_source)),
        }
    source = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return source, by_plant


def extract_alarm_currents(
    production_root: str | Path,
    alarm_points: str | Path,
    output_directory: str | Path,
    *,
    timezone: str = "Asia/Shanghai",
) -> dict[str, Any]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    labels = _normalize_keys(pd.read_parquet(alarm_points), name="cleaned alarm points")
    required_label_columns = {"alarm_event_id", *KEY_COLUMNS}
    missing = sorted(required_label_columns - set(labels.columns))
    if missing:
        raise ValueError(f"Cleaned alarm points are missing columns: {missing}")
    labels = labels.drop_duplicates(["alarm_event_id", *KEY_COLUMNS]).reset_index(drop=True)
    unique_keys = labels[KEY_COLUMNS].drop_duplicates().reset_index(drop=True)

    production, by_plant = _read_matching_production(
        Path(production_root),
        unique_keys,
        timezone=timezone,
    )
    duplicate_production_rows = (
        int(production.duplicated(KEY_COLUMNS).sum()) if not production.empty else 0
    )
    production = production.drop_duplicates(KEY_COLUMNS, keep="last")
    pvlof_input = unique_keys.merge(production, on=KEY_COLUMNS, how="inner")
    joined = labels.merge(production, on=KEY_COLUMNS, how="left", indicator=True)
    missing_points = joined[joined["_merge"].ne("both")][
        ["alarm_event_id", *KEY_COLUMNS]
    ].copy()
    matched = joined[joined["_merge"].eq("both")].drop(columns=["_merge"])

    pvlof_input.to_parquet(output / "pvlof_input.parquet", index=False)
    matched.to_parquet(output / "alarm_production_currents.parquet", index=False)
    missing_points.to_parquet(output / "missing_alarm_points.parquet", index=False)
    matched.to_csv(
        output / "alarm_production_currents.csv",
        index=False,
        encoding="utf-8-sig",
    )
    report = {
        "production_root": str(production_root),
        "alarm_points": str(alarm_points),
        "timezone": timezone,
        "alarm_event_points": int(len(labels)),
        "unique_device_time_points": int(len(unique_keys)),
        "matched_alarm_event_points": int(len(matched)),
        "matched_unique_device_time_points": int(len(pvlof_input)),
        "missing_alarm_event_points": int(len(missing_points)),
        "production_duplicate_keys": duplicate_production_rows,
        "events": int(labels["alarm_event_id"].nunique()),
        "matched_events": int(matched["alarm_event_id"].nunique()),
        "by_plant": by_plant,
        "outputs": {
            "pvlof_input": str(output / "pvlof_input.parquet"),
            "alarm_production_currents": str(
                output / "alarm_production_currents.parquet"
            ),
            "missing_alarm_points": str(output / "missing_alarm_points.parquet"),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=_json_safe) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--production-root",
        default="data/processed/pvlof/production_device_clean_v2",
    )
    parser.add_argument(
        "--alarm-points",
        default="artifacts/reports/pvlof_cleaning_v2/cleaned_alarm_points.parquet",
    )
    parser.add_argument(
        "--output-directory",
        default="data/processed/pvlof/alarm_windows_v1",
    )
    parser.add_argument("--timezone", default="Asia/Shanghai")
    args = parser.parse_args()
    report = extract_alarm_currents(
        args.production_root,
        args.alarm_points,
        args.output_directory,
        timezone=args.timezone,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_safe))


if __name__ == "__main__":
    main()
