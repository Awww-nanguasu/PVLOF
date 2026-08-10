"""Clean production current data and classify device-alarm coverage."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CURRENT_PATTERN = re.compile(r"^string_current_(\d{2})$")
DATE_PATTERN = re.compile(r"(?:^|[\\/])date=(\d{4}-\d{2}-\d{2})(?:[\\/]|$)")
PLANT_PATTERN = re.compile(r"(?:^|[\\/])plant_id=(\d+)(?:[\\/]|$)")
DEFAULT_ALARM_CODES = ("101001",)


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


def _parse_epoch_ms(values: pd.Series) -> pd.Series:
    return pd.to_datetime(
        pd.to_numeric(values, errors="coerce"),
        unit="ms",
        errors="coerce",
        utc=True,
    )


def _load_station_mapping(path: str | Path) -> dict[str, int]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    mapping = payload.get("station_to_plant_id", payload)
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("Station mapping must be a non-empty object")
    return {str(station).strip(): int(plant_id) for station, plant_id in mapping.items()}


def _partition_value(path: Path, pattern: re.Pattern[str]) -> str | None:
    match = pattern.search(str(path))
    return match.group(1) if match else None


def _parquet_columns(path: Path) -> list[str]:
    try:
        import pyarrow.parquet as parquet

        return list(parquet.ParquetFile(path).schema.names)
    except ImportError:
        return list(pd.read_parquet(path).columns)


def _list_files(root: Path, plant_id: int) -> list[Path]:
    files: list[Path] = []
    for path in sorted(root.rglob("*.parquet")):
        path_plant = _partition_value(path, PLANT_PATTERN)
        if path_plant is not None and int(path_plant) != plant_id:
            continue
        files.append(path)
    return files


def _current_columns(columns: list[str]) -> list[str]:
    values = [column for column in columns if CURRENT_PATTERN.match(column)]
    return sorted(
        values,
        key=lambda column: int(CURRENT_PATTERN.match(column).group(1)),  # type: ignore[union-attr]
    )


def _prepare_frame(
    path: Path,
    *,
    plant_id: int,
    current_columns: list[str],
    timezone: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    frame = pd.read_parquet(path)
    if "plant_id" in frame.columns:
        frame = frame[
            pd.to_numeric(frame["plant_id"], errors="coerce").eq(plant_id)
        ].copy()
    elif _partition_value(path, PLANT_PATTERN) is None:
        raise ValueError(f"Cannot determine plant_id for unpartitioned file: {path}")
    if frame.empty:
        return frame, {"raw_rows": 0, "valid_rows": 0, "clean_rows": 0}
    required = {"event_time", "device_no"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    frame["event_time"] = pd.to_datetime(frame["event_time"], errors="coerce", utc=True)
    frame["device_no"] = frame["device_no"].astype(str).str.strip()
    frame = frame.dropna(subset=["event_time", "device_no"]).copy()
    available_currents = [column for column in current_columns if column in frame.columns]
    if not available_currents:
        frame["current_complete"] = False
    else:
        numeric = frame[available_currents].apply(pd.to_numeric, errors="coerce")
        finite_count = np.isfinite(numeric.to_numpy(dtype=np.float64)).sum(axis=1)
        if "main_string_count" in frame.columns:
            expected = pd.to_numeric(frame["main_string_count"], errors="coerce")
            expected = expected.where(expected.gt(0), len(available_currents))
        else:
            expected = pd.Series(len(available_currents), index=frame.index, dtype=float)
        expected = expected.fillna(len(available_currents)).clip(upper=len(available_currents))
        frame["current_expected_count"] = expected.astype(np.int16)
        frame["current_finite_count"] = finite_count.astype(np.int16)
        frame["current_complete"] = (
            frame["current_expected_count"] > 0
        ) & frame["current_finite_count"].ge(frame["current_expected_count"])
    clean = frame[frame["current_complete"]].copy()
    frame["event_time_local"] = frame["event_time"].dt.tz_convert(timezone)
    clean["event_time_local"] = clean["event_time"].dt.tz_convert(timezone)
    stats = {
        "raw_rows": int(len(frame)),
        "valid_rows": int(frame["event_time"].notna().sum()),
        "clean_rows": int(len(clean)),
    }
    return frame, stats


def _write_clean_partition(
    frame: pd.DataFrame,
    *,
    plant_id: int,
    output_root: Path,
    timezone: str,
    counters: dict[str, int],
) -> None:
    if frame.empty:
        return
    for day, indexes in frame.groupby(
        frame["event_time"].dt.tz_convert(timezone).dt.strftime("%Y-%m-%d"),
        sort=True,
    ).groups.items():
        destination = output_root / f"plant_id={plant_id}" / f"date={day}"
        destination.mkdir(parents=True, exist_ok=True)
        counter_key = f"{plant_id}/{day}"
        part = counters.get(counter_key, 0)
        target = destination / f"part-{part:05d}.parquet"
        subset = frame.loc[indexes].drop(columns=["event_time_local"], errors="ignore")
        subset.to_parquet(target, index=False, engine="pyarrow", compression="snappy")
        counters[counter_key] = part + 1


def _read_alarm_events(
    path: str | Path,
    mapping: dict[str, int],
    *,
    alarm_codes: tuple[str, ...],
) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    required = {
        "id",
        "alarm_code",
        "device_no",
        "station_name",
        "raise_time",
        "end_time",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Alarm CSV is missing columns: {missing}")
    frame = frame[frame["alarm_code"].astype(str).isin(set(alarm_codes))].copy()
    frame["device_no"] = frame["device_no"].astype(str).str.strip()
    frame["station_name"] = frame["station_name"].astype(str).str.strip()
    frame["plant_id"] = frame["station_name"].map(mapping).astype("Int64")
    frame["raise_time"] = _parse_epoch_ms(frame["raise_time"])
    frame["end_time"] = _parse_epoch_ms(frame["end_time"])
    frame = frame.dropna(subset=["raise_time", "end_time"]).copy()
    frame = frame[frame["end_time"] >= frame["raise_time"]].copy()
    return frame.reset_index(drop=True)


def _device_ranges(source: pd.DataFrame, *, timezone: str) -> pd.DataFrame:
    if source.empty:
        return pd.DataFrame(
            columns=[
                "plant_id",
                "device_no",
                "raw_rows",
                "clean_rows",
                "first_event_time",
                "last_event_time",
                "first_event_time_local",
                "last_event_time_local",
            ]
        )
    grouped = source.groupby(["plant_id", "device_no"], observed=True)
    result = grouped.agg(
        raw_rows=("event_time", "size"),
        clean_rows=("current_complete", "sum"),
        first_event_time=("event_time", "min"),
        last_event_time=("event_time", "max"),
    ).reset_index()
    result["first_event_time_local"] = result["first_event_time"].dt.tz_convert(timezone)
    result["last_event_time_local"] = result["last_event_time"].dt.tz_convert(timezone)
    return result.sort_values(["plant_id", "device_no"]).reset_index(drop=True)


def _classify_events(
    alarms: pd.DataFrame,
    source: pd.DataFrame,
    ranges: pd.DataFrame,
    *,
    interval_minutes: int,
    timezone: str,
) -> pd.DataFrame:
    if alarms.empty:
        return pd.DataFrame()
    range_lookup = ranges.set_index(["plant_id", "device_no"])
    source_groups: dict[tuple[int, str], dict[pd.Timestamp, bool]] = {}
    for (plant_id, device_no), group in source.groupby(
        ["plant_id", "device_no"], observed=True
    ):
        source_groups[(int(plant_id), str(device_no))] = dict(
            zip(group["event_time"], group["current_complete"].astype(bool))
        )

    frequency = f"{interval_minutes}min"
    records: list[dict[str, Any]] = []
    for alarm in alarms.itertuples(index=False):
        plant_id = alarm.plant_id
        key = (int(plant_id), str(alarm.device_no)) if pd.notna(plant_id) else None
        base: dict[str, Any] = {
            "alarm_event_id": str(alarm.id),
            "alarm_code": str(alarm.alarm_code),
            "station_name": alarm.station_name,
            "plant_id": int(plant_id) if pd.notna(plant_id) else pd.NA,
            "device_no": str(alarm.device_no),
            "raise_time": alarm.raise_time,
            "end_time": alarm.end_time,
            "raise_time_local": alarm.raise_time.tz_convert(timezone),
            "end_time_local": alarm.end_time.tz_convert(timezone),
            "classification": "outside_production_range",
            "range_reason": "station_or_device_not_found",
            "device_first_event_time": pd.NaT,
            "device_last_event_time": pd.NaT,
            "effective_start_time": pd.NaT,
            "effective_end_time": pd.NaT,
            "expected_points": 0,
            "present_points": 0,
            "missing_points": 0,
            "current_incomplete_points": 0,
        }
        if key is None or key not in range_lookup.index:
            records.append(base)
            continue
        device_range = range_lookup.loc[key]
        first = device_range["first_event_time"]
        last = device_range["last_event_time"]
        base["device_first_event_time"] = first
        base["device_last_event_time"] = last
        if alarm.end_time < first or alarm.raise_time > last:
            base["range_reason"] = "no_interval_overlap"
            records.append(base)
            continue
        overlap_start = max(alarm.raise_time, first)
        overlap_end = min(alarm.end_time, last)
        base["effective_start_time"] = overlap_start
        base["effective_end_time"] = overlap_end
        expected = pd.date_range(
            overlap_start.ceil(frequency),
            overlap_end.floor(frequency),
            freq=frequency,
        )
        values = source_groups[key]
        present = [timestamp in values for timestamp in expected]
        complete = [values.get(timestamp, False) for timestamp in expected]
        base["expected_points"] = len(expected)
        base["present_points"] = int(sum(present))
        base["missing_points"] = int(len(expected) - sum(present))
        base["current_incomplete_points"] = int(
            sum(present[index] and not complete[index] for index in range(len(expected)))
        )
        outside_left = alarm.raise_time < first
        outside_right = alarm.end_time > last
        if outside_left or outside_right:
            base["classification"] = "partially_outside_range"
            base["range_reason"] = "left_or_right_boundary_outside"
        elif (
            base["missing_points"] > 0
            or base["current_incomplete_points"] > 0
            or not expected.size
        ):
            base["classification"] = "within_range_gap"
            base["range_reason"] = (
                "missing_timestamp_or_current" if expected.size else "no_grid_point"
            )
        else:
            base["classification"] = "complete"
            base["range_reason"] = "all_expected_points_complete"
        records.append(base)
    return pd.DataFrame(records)


def _expand_complete_alarm_points(events: pd.DataFrame, *, interval_minutes: int) -> pd.DataFrame:
    complete = events[events["classification"].eq("complete")]
    rows: list[dict[str, Any]] = []
    frequency = f"{interval_minutes}min"
    for event in complete.itertuples(index=False):
        for event_time in pd.date_range(
            event.effective_start_time.ceil(frequency),
            event.effective_end_time.floor(frequency),
            freq=frequency,
        ):
            rows.append(
                {
                    "alarm_event_id": event.alarm_event_id,
                    "alarm_code": event.alarm_code,
                    "plant_id": event.plant_id,
                    "station_name": event.station_name,
                    "device_no": event.device_no,
                    "event_time": event_time,
                }
            )
    return pd.DataFrame(rows)


def clean_production_data(
    input_root: str | Path,
    alarms: str | Path,
    station_mapping: str | Path,
    *,
    plant_ids: tuple[int, ...],
    alarm_codes: tuple[str, ...],
    output_current_root: str | Path,
    output_report_directory: str | Path,
    timezone: str = "Asia/Shanghai",
    interval_minutes: int = 5,
) -> dict[str, Any]:
    current_root = Path(output_current_root)
    report_root = Path(output_report_directory)
    if current_root.exists() and any(current_root.rglob("*.parquet")):
        raise FileExistsError(f"Cleaned output already contains Parquet files: {current_root}")
    current_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    mapping = _load_station_mapping(station_mapping)

    source_frames: list[pd.DataFrame] = []
    output_counters: dict[str, int] = {}
    production_report: dict[str, Any] = {}
    for plant_id in plant_ids:
        files = _list_files(Path(input_root), plant_id)
        if not files:
            production_report[str(plant_id)] = {
                "files": 0,
                "raw_rows": 0,
                "clean_rows": 0,
                "devices": 0,
            }
            continue
        current_columns = _current_columns(_parquet_columns(files[0]))
        frames: list[pd.DataFrame] = []
        raw_rows = 0
        clean_rows = 0
        for path in files:
            frame, stats = _prepare_frame(
                path,
                plant_id=plant_id,
                current_columns=current_columns,
                timezone=timezone,
            )
            raw_rows += stats["raw_rows"]
            if frame.empty:
                continue
            clean = frame[frame["current_complete"]].copy()
            clean_rows += len(clean)
            _write_clean_partition(
                clean,
                plant_id=plant_id,
                output_root=current_root,
                timezone=timezone,
                counters=output_counters,
            )
            if not frame.empty:
                frame["plant_id"] = plant_id
                frames.append(
                    frame[
                        [
                            "plant_id",
                            "device_no",
                            "event_time",
                            "current_complete",
                        ]
                    ]
                )
        plant_source = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        source_frames.append(plant_source)
        production_report[str(plant_id)] = {
            "files": len(files),
            "raw_rows": raw_rows,
            "clean_rows": clean_rows,
            "devices": int(plant_source["device_no"].nunique()) if not plant_source.empty else 0,
        }

    source = pd.concat(source_frames, ignore_index=True) if source_frames else pd.DataFrame()
    ranges = _device_ranges(source, timezone=timezone)
    alarms_frame = _read_alarm_events(alarms, mapping, alarm_codes=alarm_codes)
    classifications = _classify_events(
        alarms_frame,
        source,
        ranges,
        interval_minutes=interval_minutes,
        timezone=timezone,
    )
    complete_points = _expand_complete_alarm_points(
        classifications,
        interval_minutes=interval_minutes,
    )

    ranges.to_csv(report_root / "device_ranges.csv", index=False, encoding="utf-8-sig")
    classifications.to_csv(
        report_root / "alarm_event_classification.csv",
        index=False,
        encoding="utf-8-sig",
    )
    classifications[classifications["classification"].eq("complete")].to_csv(
        report_root / "cleaned_alarm_events.csv",
        index=False,
        encoding="utf-8-sig",
    )
    complete_points.to_csv(
        report_root / "cleaned_alarm_points.csv",
        index=False,
        encoding="utf-8-sig",
    )
    ranges.to_parquet(report_root / "device_ranges.parquet", index=False)
    classifications.to_parquet(
        report_root / "alarm_event_classification.parquet", index=False
    )
    classifications[classifications["classification"].eq("complete")].to_parquet(
        report_root / "cleaned_alarm_events.parquet",
        index=False,
    )
    complete_points.to_parquet(report_root / "cleaned_alarm_points.parquet", index=False)

    summary = {
        "input_root": str(input_root),
        "alarms": str(alarms),
        "station_mapping": mapping,
        "plant_ids": list(plant_ids),
        "alarm_codes": list(alarm_codes),
        "timezone": timezone,
        "interval_minutes": interval_minutes,
        "production": production_report,
        "output_current_root": str(current_root),
        "output_report_directory": str(report_root),
        "devices": int(len(ranges)),
        "alarm_events": int(len(classifications)),
        "alarm_event_status": classifications["classification"].value_counts().to_dict(),
        "cleaned_alarm_points": int(len(complete_points)),
    }
    (report_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_safe) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default="data/raw/device")
    parser.add_argument("--alarms", default="device_alarm_70133702.csv")
    parser.add_argument(
        "--station-mapping",
        default="configs/production_alarm_station_mapping.json",
    )
    parser.add_argument("--plant-ids", nargs="+", type=int, default=[234, 791, 892])
    parser.add_argument("--alarm-codes", nargs="+", default=list(DEFAULT_ALARM_CODES))
    parser.add_argument("--timezone", default="Asia/Shanghai")
    parser.add_argument("--interval-minutes", type=int, default=5)
    parser.add_argument(
        "--output-current-root",
        default="data/processed/pvlof/production_device",
    )
    parser.add_argument(
        "--output-report-directory",
        default="artifacts/reports/pvlof_cleaning",
    )
    args = parser.parse_args()
    summary = clean_production_data(
        args.input_root,
        args.alarms,
        args.station_mapping,
        plant_ids=tuple(args.plant_ids),
        alarm_codes=tuple(str(code) for code in args.alarm_codes),
        output_current_root=args.output_current_root,
        output_report_directory=args.output_report_directory,
        timezone=args.timezone,
        interval_minutes=args.interval_minutes,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_safe))


if __name__ == "__main__":
    main()
