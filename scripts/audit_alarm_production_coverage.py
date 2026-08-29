"""Audit production current coverage for inverter alarm intervals."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pv_anomaly.alarm_time import alarm_time_grid


CURRENT_PATTERN = re.compile(r"^string_current_(\d{2})$")
DATE_PATTERN = re.compile(r"(?:^|[\\/])date=(\d{4}-\d{2}-\d{2})(?:[\\/]|$)")
PLANT_PATTERN = re.compile(r"(?:^|[\\/])plant_id=(\d+)(?:[\\/]|$)")
DEFAULT_ALARM_CODES = ("101001",)


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if pd.isna(value):
        return None
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _parse_epoch_ms(values: pd.Series) -> pd.Series:
    return pd.to_datetime(
        pd.to_numeric(values, errors="coerce"),
        unit="ms",
        errors="coerce",
        utc=True,
    )


def _bound_to_utc(value: str | None, timezone: str) -> pd.Timestamp | None:
    if value is None:
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(timezone)
    return timestamp.tz_convert("UTC")


def _load_station_mapping(path: str | Path) -> dict[str, int]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    mapping = payload.get("station_to_plant_id", payload)
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("Station mapping must be a non-empty object")
    result: dict[str, int] = {}
    for station, plant_id in mapping.items():
        result[str(station).strip()] = int(plant_id)
    return result


def _read_alarm_points(
    path: str | Path,
    station_mapping: dict[str, int],
    *,
    alarm_codes: tuple[str, ...],
    timezone: str,
    interval_minutes: int,
    start: str | None,
    end: str | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    required = {
        "id",
        "alarm_code",
        "device_no",
        "station_name",
        "raise_time",
        "end_time",
    }
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"Alarm CSV is missing columns: {missing}")

    source["alarm_code"] = source["alarm_code"].astype(str)
    source = source[source["alarm_code"].isin(set(alarm_codes))].copy()
    source["device_no"] = source["device_no"].astype(str).str.strip()
    source["station_name"] = source["station_name"].astype(str).str.strip()
    source["raise_time"] = _parse_epoch_ms(source["raise_time"])
    source["end_time"] = _parse_epoch_ms(source["end_time"])
    source = source.dropna(subset=["raise_time", "end_time"])
    source = source[source["end_time"] >= source["raise_time"]].copy()

    start_utc = _bound_to_utc(start, timezone)
    end_utc = _bound_to_utc(end, timezone)
    if start_utc is not None:
        source = source[source["end_time"] >= start_utc]
    if end_utc is not None:
        source = source[source["raise_time"] < end_utc]
    source = source.reset_index(drop=True)
    source["plant_id"] = source["station_name"].map(station_mapping).astype("Int64")
    source["station_mapping_status"] = np.where(
        source["plant_id"].notna(), "mapped", "station_unmapped"
    )

    rows: list[dict[str, Any]] = []
    skipped = 0
    for row in source.itertuples(index=False):
        grid = alarm_time_grid(
            row.raise_time,
            row.end_time,
            interval_minutes=interval_minutes,
        )
        if grid.empty:
            skipped += 1
            continue
        for event_time in grid:
            rows.append(
                {
                    "alarm_event_id": str(row.id),
                    "alarm_code": str(row.alarm_code),
                    "station_name": row.station_name,
                    "plant_id": row.plant_id,
                    "device_no": row.device_no,
                    "raise_time": row.raise_time,
                    "end_time": row.end_time,
                    "event_time": event_time,
                    "station_mapping_status": row.station_mapping_status,
                }
            )
    points = pd.DataFrame(rows)
    if points.empty:
        points = pd.DataFrame(
            columns=[
                "alarm_event_id",
                "alarm_code",
                "station_name",
                "plant_id",
                "device_no",
                "raise_time",
                "end_time",
                "event_time",
                "station_mapping_status",
            ]
        )
    else:
        points = points.drop_duplicates(
            ["alarm_event_id", "alarm_code", "device_no", "event_time"]
        ).reset_index(drop=True)
    report = {
        "path": str(path),
        "alarm_codes": list(alarm_codes),
        "source_rows_after_filter": int(len(source)),
        "source_unique_alarm_ids": int(source["id"].nunique()),
        "expanded_points": int(len(points)),
        "events_without_grid_points": int(skipped),
        "unmapped_station_rows": int((source["plant_id"].isna()).sum()),
    }
    return points, report


def _partition_value(path: Path, pattern: re.Pattern[str]) -> str | None:
    match = pattern.search(str(path))
    return match.group(1) if match else None


def _parquet_columns(path: Path) -> list[str]:
    try:
        import pyarrow.parquet as parquet

        return list(parquet.ParquetFile(path).schema.names)
    except ImportError:
        return list(pd.read_parquet(path).columns)


def _list_production_files(
    root: Path,
    *,
    plant_id: int,
    local_dates: set[str],
) -> list[Path]:
    files: list[Path] = []
    for path in sorted(root.rglob("*.parquet")):
        path_plant = _partition_value(path, PLANT_PATTERN)
        if path_plant is not None and int(path_plant) != plant_id:
            continue
        path_date = _partition_value(path, DATE_PATTERN)
        if path_date is not None and local_dates and path_date not in local_dates:
            continue
        files.append(path)
    return files


def _read_production_rows(
    files: list[Path],
    *,
    plant_id: int,
    local_dates: set[str],
    timezone: str,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    if not files:
        return pd.DataFrame(), [], {"files": 0, "rows": 0, "devices": 0}
    first_columns = _parquet_columns(files[0])
    current_columns = sorted(
        (column for column in first_columns if CURRENT_PATTERN.match(column)),
        key=lambda column: int(CURRENT_PATTERN.match(column).group(1)),  # type: ignore[union-attr]
    )
    required = ["event_time", "device_no"]
    optional = ["plant_id", "main_string_count", "valid_current_string_count"]
    columns = [
        column
        for column in required + optional + current_columns
        if column in first_columns
    ]
    frames: list[pd.DataFrame] = []
    for path in files:
        available = set(_parquet_columns(path))
        selected = [column for column in columns if column in available]
        frame = pd.read_parquet(path, columns=selected)
        if "plant_id" not in frame.columns:
            path_plant = _partition_value(path, PLANT_PATTERN)
            frame["plant_id"] = int(path_plant) if path_plant else pd.NA
        else:
            frame = frame[
                pd.to_numeric(frame["plant_id"], errors="coerce").eq(plant_id)
            ]
        frame["device_no"] = frame["device_no"].astype(str).str.strip()
        frame["event_time"] = pd.to_datetime(frame["event_time"], errors="coerce", utc=True)
        frame = frame.dropna(subset=["event_time", "device_no"])
        if local_dates:
            local_day = frame["event_time"].dt.tz_convert(timezone).dt.strftime("%Y-%m-%d")
            frame = frame[local_day.isin(local_dates)]
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(), current_columns, {"files": len(files), "rows": 0, "devices": 0}
    source = pd.concat(frames, ignore_index=True)
    source = source.sort_values(["device_no", "event_time"]).reset_index(drop=True)
    duplicate_count = int(source.duplicated(["device_no", "event_time"]).sum())
    source["production_row_count"] = source.groupby(
        ["device_no", "event_time"], observed=True
    )["device_no"].transform("size")
    source = source.drop_duplicates(["device_no", "event_time"], keep="first")
    report = {
        "files": len(files),
        "rows": int(len(source)),
        "devices": int(source["device_no"].nunique()),
        "duplicate_rows": duplicate_count,
        "time_start": source["event_time"].min(),
        "time_end": source["event_time"].max(),
    }
    return source, current_columns, report


def _read_production_device_set(files: list[Path], *, plant_id: int) -> set[str]:
    devices: set[str] = set()
    for path in files:
        available = set(_parquet_columns(path))
        columns = [column for column in ("device_no", "plant_id") if column in available]
        if "device_no" not in columns:
            continue
        frame = pd.read_parquet(path, columns=columns)
        if "plant_id" in frame.columns:
            frame = frame[
                pd.to_numeric(frame["plant_id"], errors="coerce").eq(plant_id)
            ]
        devices.update(frame["device_no"].dropna().astype(str).str.strip().unique())
    return devices


def _attach_coverage(
    points: pd.DataFrame,
    production: pd.DataFrame,
    current_columns: list[str],
    *,
    production_devices: set[str],
    timezone: str,
) -> pd.DataFrame:
    result = points.copy()
    result["production_device_exists"] = False
    result["production_row_count"] = 0
    result["current_expected_count"] = 0
    result["current_finite_count"] = 0
    result["current_missing_count"] = 0
    result["current_minimum"] = np.nan
    result["current_maximum"] = np.nan
    result["production_device_exists"] = result["device_no"].isin(production_devices)
    if production.empty:
        result["coverage_status"] = np.where(
            result["station_mapping_status"].eq("mapped"),
            np.where(
                result["production_device_exists"],
                "production_timestamp_missing",
                "production_device_missing",
            ),
            "station_unmapped",
        )
        return result

    source = production.copy()
    source["device_no"] = source["device_no"].astype(str).str.strip()
    source = source.set_index(["device_no", "event_time"])
    keys = pd.MultiIndex.from_frame(result[["device_no", "event_time"]])
    matched = source.reindex(keys)
    matched.index = result.index
    result["production_row_count"] = pd.to_numeric(
        matched.get("production_row_count", pd.Series(0, index=result.index)),
        errors="coerce",
    ).fillna(0).astype(int)

    available_currents = [column for column in current_columns if column in matched.columns]
    if available_currents:
        numeric = matched[available_currents].apply(pd.to_numeric, errors="coerce")
        finite = np.isfinite(numeric.to_numpy(dtype=np.float64))
        result["current_finite_count"] = finite.sum(axis=1)
        result["current_minimum"] = numeric.min(axis=1).to_numpy()
        result["current_maximum"] = numeric.max(axis=1).to_numpy()
        if "main_string_count" in matched.columns:
            expected = pd.to_numeric(matched["main_string_count"], errors="coerce")
            expected = expected.where(expected.gt(0), len(available_currents))
        else:
            expected = pd.Series(len(available_currents), index=result.index, dtype=float)
        expected = expected.fillna(len(available_currents)).clip(upper=len(available_currents))
        result["current_expected_count"] = expected.astype(int).to_numpy()
        result["current_missing_count"] = (
            result["current_expected_count"] - result["current_finite_count"]
        ).clip(lower=0)
    result["coverage_status"] = "complete"
    result.loc[result["station_mapping_status"].ne("mapped"), "coverage_status"] = (
        "station_unmapped"
    )
    result.loc[
        result["station_mapping_status"].eq("mapped")
        & ~result["production_device_exists"],
        "coverage_status",
    ] = "production_device_missing"
    result.loc[
        result["production_device_exists"] & result["production_row_count"].eq(0),
        "coverage_status",
    ] = "production_timestamp_missing"
    result.loc[
        result["production_row_count"].gt(0)
        & result["current_missing_count"].gt(0),
        "coverage_status",
    ] = "current_incomplete"
    if not available_currents:
        result.loc[result["production_row_count"].gt(0), "coverage_status"] = (
            "current_columns_missing"
        )
    result["event_time_local"] = result["event_time"].dt.tz_convert(timezone).dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    return result


def _event_summary(points: pd.DataFrame) -> pd.DataFrame:
    if points.empty:
        return pd.DataFrame(
            columns=[
                "alarm_event_id",
                "alarm_code",
                "station_name",
                "plant_id",
                "device_no",
                "raise_time",
                "end_time",
                "grid_points",
                "complete_current_points",
                "coverage_percent",
                "event_coverage_status",
            ]
        )
    grouped = points.groupby(
        [
            "alarm_event_id",
            "alarm_code",
            "station_name",
            "plant_id",
            "device_no",
            "raise_time",
            "end_time",
        ],
        dropna=False,
        observed=True,
    )
    summary = grouped.agg(
        grid_points=("event_time", "size"),
        complete_current_points=(
            "coverage_status",
            lambda values: int((values == "complete").sum()),
        ),
        current_incomplete_points=(
            "coverage_status",
            lambda values: int((values == "current_incomplete").sum()),
        ),
        mapped_points=("coverage_status", lambda values: int((values != "station_unmapped").sum())),
        device_present_points=(
            "coverage_status",
            lambda values: int((values != "production_device_missing").sum()),
        ),
        timestamp_present_points=(
            "coverage_status",
            lambda values: int(
                values.isin(["complete", "current_incomplete"]).sum()
            ),
        ),
        minimum_current=("current_minimum", "min"),
        maximum_current=("current_maximum", "max"),
    ).reset_index()
    summary["coverage_percent"] = np.where(
        summary["grid_points"].gt(0),
        summary["complete_current_points"] / summary["grid_points"] * 100,
        0.0,
    )
    summary["event_coverage_status"] = "complete"
    summary.loc[summary["mapped_points"].eq(0), "event_coverage_status"] = "station_unmapped"
    summary.loc[
        summary["mapped_points"].gt(0) & summary["device_present_points"].eq(0),
        "event_coverage_status",
    ] = "production_device_missing"
    summary.loc[
        summary["device_present_points"].gt(0)
        & summary["timestamp_present_points"].eq(0),
        "event_coverage_status",
    ] = "production_timestamp_missing"
    summary.loc[
        summary["timestamp_present_points"].gt(0)
        & summary["timestamp_present_points"].lt(summary["grid_points"]),
        "event_coverage_status",
    ] = "production_timestamp_partial"
    summary.loc[
        summary["timestamp_present_points"].eq(summary["grid_points"])
        & summary["current_incomplete_points"].gt(0),
        "event_coverage_status",
    ] = "current_incomplete"
    return summary.sort_values(["plant_id", "device_no", "raise_time"], na_position="last")


def audit_alarm_coverage(
    alarms: str | Path,
    production_root: str | Path,
    station_mapping: str | Path,
    *,
    plant_ids: tuple[int, ...] = (234, 791, 892),
    alarm_codes: tuple[str, ...] = DEFAULT_ALARM_CODES,
    timezone: str = "Asia/Shanghai",
    interval_minutes: int = 5,
    start: str | None = None,
    end: str | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    mapping = _load_station_mapping(station_mapping)
    points, alarm_report = _read_alarm_points(
        alarms,
        mapping,
        alarm_codes=alarm_codes,
        timezone=timezone,
        interval_minutes=interval_minutes,
        start=start,
        end=end,
    )
    if points.empty:
        local_dates: set[str] = set()
    else:
        local_dates = set(
            points["event_time"].dt.tz_convert(timezone).dt.strftime("%Y-%m-%d")
        )

    production_root_path = Path(production_root)
    covered_frames: list[pd.DataFrame] = []
    production_reports: dict[str, Any] = {}
    for plant_id in plant_ids:
        all_files = _list_production_files(
            production_root_path,
            plant_id=plant_id,
            local_dates=set(),
        )
        files = _list_production_files(
            production_root_path,
            plant_id=plant_id,
            local_dates=local_dates,
        )
        production_devices = _read_production_device_set(all_files, plant_id=plant_id)
        production, current_columns, production_report = _read_production_rows(
            files,
            plant_id=plant_id,
            local_dates=local_dates,
            timezone=timezone,
        )
        production_report["device_inventory_count"] = len(production_devices)
        production_reports[str(plant_id)] = production_report
        plant_points = points[points["plant_id"].eq(plant_id)].copy()
        if not plant_points.empty:
            covered_frames.append(
                _attach_coverage(
                    plant_points,
                    production,
                    current_columns,
                    production_devices=production_devices,
                    timezone=timezone,
                )
            )

    unmapped_points = points[points["plant_id"].isna()].copy()
    if not unmapped_points.empty:
        covered_frames.append(
            _attach_coverage(
                unmapped_points,
                pd.DataFrame(),
                [],
                production_devices=set(),
                timezone=timezone,
            )
        )
    covered_points = (
        pd.concat(covered_frames, ignore_index=True)
        if covered_frames
        else _attach_coverage(
            points,
            pd.DataFrame(),
            [],
            production_devices=set(),
            timezone=timezone,
        )
    )
    events = _event_summary(covered_points)

    point_status = covered_points["coverage_status"].value_counts(dropna=False).to_dict()
    event_status = events["event_coverage_status"].value_counts(dropna=False).to_dict()
    by_plant: dict[str, Any] = {}
    for plant_id in plant_ids:
        plant_points = covered_points[covered_points["plant_id"].eq(plant_id)]
        plant_events = events[events["plant_id"].eq(plant_id)]
        by_plant[str(plant_id)] = {
            "alarm_events": int(len(plant_events)),
            "alarm_points": int(len(plant_points)),
            "complete_events": int((plant_events["event_coverage_status"] == "complete").sum()),
            "complete_points": int((plant_points["coverage_status"] == "complete").sum()),
            "point_status": plant_points["coverage_status"].value_counts().to_dict(),
            "event_status": plant_events["event_coverage_status"].value_counts().to_dict(),
        }
    report = {
        "alarms": alarm_report,
        "production_root": str(production_root),
        "station_mapping": mapping,
        "plant_ids": list(plant_ids),
        "timezone": timezone,
        "interval_minutes": interval_minutes,
        "time_filter": {"start": start, "end": end},
        "production": production_reports,
        "coverage": {
            "alarm_events": int(len(events)),
            "alarm_points": int(len(covered_points)),
            "complete_events": int((events["event_coverage_status"] == "complete").sum()),
            "complete_points": int((covered_points["coverage_status"] == "complete").sum()),
            "point_status": point_status,
            "event_status": event_status,
            "by_plant": by_plant,
        },
    }
    return report, events, covered_points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alarms", default="device_alarm_70133702.csv")
    parser.add_argument("--production-root", default="data/raw/device")
    parser.add_argument(
        "--station-mapping",
        default="configs/production_alarm_station_mapping.json",
    )
    parser.add_argument("--plant-ids", nargs="+", type=int, default=[234, 791, 892])
    parser.add_argument("--alarm-codes", nargs="+", default=list(DEFAULT_ALARM_CODES))
    parser.add_argument("--start", default=None, help="Inclusive local time/date")
    parser.add_argument("--end", default=None, help="Exclusive local time/date")
    parser.add_argument("--timezone", default="Asia/Shanghai")
    parser.add_argument("--interval-minutes", type=int, default=5)
    parser.add_argument(
        "--output-directory",
        default="artifacts/reports/production_alarm_coverage",
    )
    args = parser.parse_args()

    report, events, points = audit_alarm_coverage(
        args.alarms,
        args.production_root,
        args.station_mapping,
        plant_ids=tuple(args.plant_ids),
        alarm_codes=tuple(str(code) for code in args.alarm_codes),
        timezone=args.timezone,
        interval_minutes=args.interval_minutes,
        start=args.start,
        end=args.end,
    )
    output = Path(args.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=_json_safe) + "\n",
        encoding="utf-8",
    )
    events.to_csv(output / "alarm_event_coverage.csv", index=False, encoding="utf-8-sig")
    points.to_csv(output / "alarm_point_coverage.csv", index=False, encoding="utf-8-sig")
    events.to_parquet(output / "alarm_event_coverage.parquet", index=False)
    points.to_parquet(output / "alarm_point_coverage.parquet", index=False)
    print(json.dumps(report["coverage"], ensure_ascii=False, indent=2, default=_json_safe))
    print(f"Wrote {output / 'summary.json'}")


if __name__ == "__main__":
    main()
