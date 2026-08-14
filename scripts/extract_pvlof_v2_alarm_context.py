"""Extract alarm target points plus same-plant peer inverter context for PVLOF-V2."""

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
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _partition(path: Path, pattern: re.Pattern[str]) -> str | None:
    match = pattern.search(str(path))
    return match.group(1) if match else None


def _columns(path: Path) -> list[str]:
    import pyarrow.parquet as parquet

    available = list(parquet.ParquetFile(path).schema.names)
    metadata = [
        "event_time", "plant_id", "device_no", "main_string_count",
        "valid_current_string_count", "status_code", "string_overall_status",
    ]
    strings = sorted(
        column for column in available
        if column.startswith("string_current_") or column.startswith("string_status_")
    )
    return [column for column in metadata + strings if column in available]


def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["plant_id"] = pd.to_numeric(result["plant_id"], errors="coerce").astype("Int64")
    result["device_no"] = result["device_no"].astype(str).str.strip()
    result["event_time"] = pd.to_datetime(result["event_time"], errors="raise", utc=True)
    return result


def extract_context(
    production_root: str | Path,
    alarm_points_path: str | Path,
    output_directory: str | Path,
    *,
    timezone: str = "Asia/Shanghai",
    warmup_minutes: int = 15,
) -> dict[str, Any]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    labels = _normalise(pd.read_parquet(alarm_points_path))
    required = {"alarm_event_id", *KEY_COLUMNS}
    missing = sorted(required - set(labels.columns))
    if missing:
        raise ValueError(f"Alarm points are missing columns: {missing}")
    labels = labels.drop_duplicates(["alarm_event_id", *KEY_COLUMNS]).reset_index(drop=True)
    labels["is_alarm_target"] = True
    target_key = labels.groupby(KEY_COLUMNS, observed=True)["alarm_event_id"].agg(
        lambda values: ",".join(sorted({str(value) for value in values}))
    ).rename("alarm_event_id").reset_index()
    target_key["is_alarm_target"] = True
    target_lookup = target_key.set_index(KEY_COLUMNS)
    target_lookup_dict = target_lookup["alarm_event_id"].to_dict()
    target_time_sets: dict[int, set[pd.Timestamp]] = {}
    for plant_value, group in labels.groupby("plant_id", observed=True):
        times = set(group["event_time"])
        for timestamp in group["event_time"].unique():
            for offset in range(1, warmup_minutes // 5 + 1):
                times.add(timestamp - pd.Timedelta(minutes=offset * 5))
        target_time_sets[int(plant_value)] = times

    root = Path(production_root)
    all_frames: list[pd.DataFrame] = []
    plant_report: dict[str, Any] = {}
    for plant_value, times in target_time_sets.items():
        plant_dates = {
            timestamp.tz_convert(timezone).strftime("%Y-%m-%d") for timestamp in times
        }
        files = []
        for path in sorted(root.rglob("*.parquet")):
            path_plant = _partition(path, PLANT_PATTERN)
            path_date = _partition(path, DATE_PATTERN)
            if path_plant is not None and int(path_plant) != plant_value:
                continue
            if path_date is not None and path_date not in plant_dates:
                continue
            files.append(path)
        plant_frames: list[pd.DataFrame] = []
        failed: list[str] = []
        time_index = pd.DatetimeIndex(times)
        for path in files:
            try:
                frame = _normalise(pd.read_parquet(path, columns=_columns(path)))
            except Exception as error:
                failed.append(f"{path}: {error}")
                continue
            frame = frame[frame["plant_id"].eq(plant_value)]
            if frame.empty:
                continue
            frame = frame[frame["event_time"].isin(time_index)]
            if not frame.empty:
                plant_frames.append(frame)
        if plant_frames:
            plant_frame = pd.concat(plant_frames, ignore_index=True)
            plant_frame = plant_frame.drop_duplicates(KEY_COLUMNS, keep="last")
            plant_frame["_key"] = list(plant_frame[KEY_COLUMNS].itertuples(index=False, name=None))
            plant_frame["alarm_event_id"] = plant_frame["_key"].map(target_lookup_dict)
            plant_frame["is_alarm_target"] = plant_frame["_key"].isin(set(target_lookup.index))
            plant_frame["is_warmup"] = ~plant_frame["event_time"].isin(set(labels.loc[labels["plant_id"].eq(plant_value), "event_time"]))
            plant_frame = plant_frame.drop(columns="_key")
            all_frames.append(plant_frame)
            matched_targets = int(plant_frame["is_alarm_target"].sum())
            devices = int(plant_frame["device_no"].nunique())
        else:
            matched_targets = 0
            devices = 0
        plant_report[str(plant_value)] = {
            "files_selected": len(files),
            "files_failed": len(failed),
            "error_examples": failed[:3],
            "target_points": int(labels["plant_id"].eq(plant_value).sum()),
            "matched_target_points": matched_targets,
            "context_rows": int(len(plant_frame)) if plant_frames else 0,
            "devices": devices,
        }
    context = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    context = context.sort_values(["event_time", "plant_id", "device_no"]).reset_index(drop=True)
    targets = context[context["is_alarm_target"]].copy()
    context.to_parquet(output / "context_currents.parquet", index=False)
    targets.to_parquet(output / "target_points.parquet", index=False)
    context.to_csv(output / "context_currents.csv", index=False, encoding="utf-8-sig")
    report = {
        "production_root": str(production_root),
        "alarm_points_path": str(alarm_points_path),
        "warmup_minutes": warmup_minutes,
        "alarm_points": int(len(labels)),
        "context_rows": int(len(context)),
        "target_rows": int(len(targets)),
        "context_devices": int(context[["plant_id", "device_no"]].drop_duplicates().shape[0]) if len(context) else 0,
        "plant": plant_report,
        "outputs": {
            "context": str(output / "context_currents.parquet"),
            "targets": str(output / "target_points.parquet"),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=_json_safe) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-root", default="data/processed/pvlof/production_device_clean_v2")
    parser.add_argument("--alarm-points", default="artifacts/reports/pvlof_cleaning_v2/cleaned_alarm_points.parquet")
    parser.add_argument("--output-directory", default="data/processed/pvlof/alarm_windows_v2")
    parser.add_argument("--timezone", default="Asia/Shanghai")
    parser.add_argument("--warmup-minutes", type=int, default=15)
    args = parser.parse_args()
    print(json.dumps(extract_context(
        args.production_root, args.alarm_points, args.output_directory,
        timezone=args.timezone, warmup_minutes=args.warmup_minutes,
    ), ensure_ascii=False, indent=2, default=_json_safe))


if __name__ == "__main__":
    main()
