"""Calibrate virtual-irradiance conditioned PVLOF-V2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pv_anomaly.pvlof_io import read_pvlof_source
from pv_anomaly.pvlof_v2 import fit_pvlof_v2_calibration, save_calibration


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _exclude_alarm_windows(
    frame: pd.DataFrame,
    events_path: str | Path,
    *,
    buffer_minutes: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    events = pd.read_parquet(events_path)
    required = {"plant_id", "device_no"}
    if not required.issubset(events.columns):
        raise ValueError(f"Alarm events are missing columns: {sorted(required - set(events.columns))}")
    start_column = "effective_start_time" if "effective_start_time" in events else "raise_time"
    end_column = "effective_end_time" if "effective_end_time" in events else "end_time"
    events[start_column] = pd.to_datetime(events[start_column], errors="coerce", utc=True)
    events[end_column] = pd.to_datetime(events[end_column], errors="coerce", utc=True)
    source = frame.copy()
    source["_excluded_alarm"] = False
    interval_count = 0
    for (plant, device), group in events.groupby(["plant_id", "device_no"], observed=True):
        mask_source = source["plant_id"].astype(str).eq(str(plant)) & source["device_no"].astype(str).eq(str(device))
        if not mask_source.any():
            continue
        interval_frame = group[[start_column, end_column]].dropna()
        for start, end in interval_frame.itertuples(index=False, name=None):
            start = start - pd.Timedelta(minutes=buffer_minutes)
            end = end + pd.Timedelta(minutes=buffer_minutes)
            source.loc[mask_source & source["event_time"].between(start, end), "_excluded_alarm"] = True
            interval_count += 1
    excluded = int(source["_excluded_alarm"].sum())
    source = source.loc[~source["_excluded_alarm"]].drop(columns="_excluded_alarm")
    return source, {"alarm_intervals": interval_count, "excluded_rows": excluded}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/processed/pvlof/production_device_clean_v2")
    parser.add_argument("--start", default="2026-01-30")
    parser.add_argument("--end", default="2026-07-31", help="Exclusive local date/time")
    parser.add_argument("--timezone", default="Asia/Shanghai")
    parser.add_argument("--exclude-events", default="artifacts/reports/pvlof_cleaning_v2/cleaned_alarm_events.parquet")
    parser.add_argument("--exclude-buffer-minutes", type=int, default=10)
    parser.add_argument("--output", default="artifacts/models/pvlof_v2/calibration.json")
    parser.add_argument("--report", default="artifacts/reports/pvlof_v2_calibration.json")
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--minimum-peer-devices", type=int, default=5)
    parser.add_argument("--minimum-strings", type=int, default=4)
    parser.add_argument("--collective-overlap-threshold", type=float, default=0.50)
    parser.add_argument("--minimum-consecutive", type=int, default=2)
    parser.add_argument("--max-score-rows", type=int, default=100000)
    args = parser.parse_args()

    frame, input_report = read_pvlof_source(
        args.input, start=args.start, end=args.end, timezone=args.timezone
    )
    exclusion_report: dict[str, int] = {"alarm_intervals": 0, "excluded_rows": 0}
    if args.exclude_events and Path(args.exclude_events).exists():
        frame, exclusion_report = _exclude_alarm_windows(
            frame, args.exclude_events, buffer_minutes=args.exclude_buffer_minutes
        )
    calibration, fit_report = fit_pvlof_v2_calibration(
        frame,
        n_neighbors=args.neighbors,
        minimum_peer_devices=args.minimum_peer_devices,
        minimum_strings=args.minimum_strings,
        collective_overlap_threshold=args.collective_overlap_threshold,
        minimum_consecutive=args.minimum_consecutive,
        max_score_rows=args.max_score_rows,
    )
    save_calibration(calibration, args.output)
    report = {
        "input": input_report,
        "exclusion": exclusion_report,
        "calibration": calibration.to_dict(),
        "fit": fit_report,
        "output": args.output,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=_json_safe) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_safe))


if __name__ == "__main__":
    main()
