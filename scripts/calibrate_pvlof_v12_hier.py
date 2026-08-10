"""Calibrate hierarchical isolated thresholds for PVLOF v1.2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pv_anomaly.forecast_features import read_weather_forecast
from pv_anomaly.pvlof_io import read_pvlof_source
from pv_anomaly.pvlof_v2 import apply_pvlof_v2, load_calibration
from pv_anomaly.pvlof_v2_hier import (
    exclude_alarm_windows,
    fit_hierarchical_calibration,
    save_hier_calibration,
)
from pv_anomaly.pvlof_v12 import build_conditioned_virtual_context, load_weather_calibration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/processed/pvlof/production_device_clean_v2")
    parser.add_argument("--weather", action="append", required=True)
    parser.add_argument("--start", default="2026-01-30")
    parser.add_argument("--end", default="2026-06-01")
    parser.add_argument("--timezone", default="Asia/Shanghai")
    parser.add_argument("--base-calibration", required=True)
    parser.add_argument("--weather-calibration", required=True)
    parser.add_argument("--exclude-events")
    parser.add_argument("--exclude-buffer-minutes", type=int, default=10)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--lof-quantile", type=float, default=0.995)
    parser.add_argument("--minimum-device-samples", type=int, default=10_000)
    parser.add_argument("--minimum-device-days", type=int, default=20)
    parser.add_argument("--minimum-plant-count-samples", type=int, default=20_000)
    parser.add_argument("--minimum-plant-count-days", type=int, default=20)
    parser.add_argument("--minimum-count-samples", type=int, default=20_000)
    parser.add_argument("--minimum-count-days", type=int, default=20)
    parser.add_argument("--shrinkage-k", type=float, default=5_000.0)
    args = parser.parse_args()

    frame, input_report = read_pvlof_source(
        args.input, start=args.start, end=args.end, timezone=args.timezone
    )
    exclusion = {"alarm_intervals": 0, "excluded_rows": 0}
    if args.exclude_events and Path(args.exclude_events).exists():
        frame, exclusion = exclude_alarm_windows(
            frame, args.exclude_events, buffer_minutes=args.exclude_buffer_minutes
        )
    weather = read_weather_forecast(args.weather)
    base = load_calibration(args.base_calibration)
    weather_calibration = load_weather_calibration(args.weather_calibration)
    context, context_report = build_conditioned_virtual_context(
        frame, base, weather_calibration, weather
    )
    scored = apply_pvlof_v2(frame, base, virtual_override=context)
    hierarchical, fit_report = fit_hierarchical_calibration(
        scored,
        quantile=args.lof_quantile,
        minimum_device_samples=args.minimum_device_samples,
        minimum_device_days=args.minimum_device_days,
        minimum_plant_count_samples=args.minimum_plant_count_samples,
        minimum_plant_count_days=args.minimum_plant_count_days,
        minimum_count_samples=args.minimum_count_samples,
        minimum_count_days=args.minimum_count_days,
        shrinkage_k=args.shrinkage_k,
        minimum_consecutive=base.minimum_consecutive,
        expected_interval_minutes=base.expected_interval_minutes,
        timezone=args.timezone,
        version="pvlof-v1.2-hier-strict",
    )
    save_hier_calibration(hierarchical, args.output)
    report = {
        "input": input_report,
        "weather_roots": args.weather,
        "base_calibration": args.base_calibration,
        "weather_calibration": args.weather_calibration,
        "exclusion": exclusion,
        "conditioned_context": context_report,
        "fit": fit_report,
        "output": args.output,
    }
    destination = Path(args.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
