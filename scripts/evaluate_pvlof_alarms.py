"""Evaluate PVLOF output against inverter alarm events."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pv_anomaly.pvlof_alarm import (
    DEFAULT_ALARM_CODES,
    evaluate_pvlof,
    expand_alarm_points,
    load_device_manifest,
    read_alarm_events,
    read_pvlof_points,
    restrict_alarm_events_to_predictions,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alarms", default="device_alarm_70133702.csv")
    parser.add_argument("--manifest", default="configs/test_station_devices.json")
    parser.add_argument("--pvlof", required=True, help="PVLOF point Parquet or directory")
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--start", default=None, help="Inclusive local time/date")
    parser.add_argument("--end", default=None, help="Exclusive local time/date")
    parser.add_argument("--timezone", default="Asia/Shanghai")
    parser.add_argument("--interval-minutes", type=int, default=5)
    parser.add_argument(
        "--alarm-codes",
        nargs="+",
        default=list(DEFAULT_ALARM_CODES),
        help="Alarm codes used as labels; 101001=low current, 101002=zero current",
    )
    args = parser.parse_args()

    output = Path(args.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    manifest = load_device_manifest(args.manifest)
    alarm_events, alarm_report = read_alarm_events(
        args.alarms,
        manifest,
        alarm_codes=args.alarm_codes,
        start=args.start,
        end=args.end,
        timezone=args.timezone,
    )
    predictions, prediction_report = read_pvlof_points(
        args.pvlof,
        manifest,
        start=args.start,
        end=args.end,
        timezone=args.timezone,
    )
    alarm_events, coverage_report = restrict_alarm_events_to_predictions(
        alarm_events,
        predictions,
    )
    alarm_points, point_report = expand_alarm_points(
        alarm_events,
        interval_minutes=args.interval_minutes,
    )
    summary, matches = evaluate_pvlof(
        predictions,
        alarm_events,
        alarm_points,
        interval_minutes=args.interval_minutes,
    )
    manifest_report = {
        "stations": int(manifest["station_name"].nunique()),
        "devices": int(len(manifest)),
        "devices_by_station": manifest.groupby("station_name").size().to_dict(),
    }
    report = {
        "manifest": manifest_report,
        "alarms": alarm_report,
        "alarm_coverage": coverage_report,
        "alarm_points": point_report,
        "pvlof": prediction_report,
        "evaluation": summary,
    }
    manifest.to_csv(output / "device_manifest.csv", index=False, encoding="utf-8-sig")
    alarm_events.to_parquet(output / "alarm_events.parquet", index=False)
    alarm_points.to_parquet(output / "alarm_label_points.parquet", index=False)
    predictions.to_parquet(output / "aligned_pvlof_points.parquet", index=False)
    matches.to_csv(output / "event_matches.csv", index=False, encoding="utf-8-sig")
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=_json_safe) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_safe))


def _json_safe(value: object) -> object:
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


if __name__ == "__main__":
    main()
