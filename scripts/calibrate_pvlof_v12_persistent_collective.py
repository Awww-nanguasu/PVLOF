"""Calibrate the parallel persistent mild collective branch for PVLOF v1.2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pv_anomaly.forecast_features import read_weather_forecast
from pv_anomaly.pvlof_io import read_pvlof_source
from pv_anomaly.pvlof_v2 import apply_pvlof_v2, load_calibration
from pv_anomaly.pvlof_v2_hier import exclude_alarm_windows
from pv_anomaly.pvlof_v12 import build_conditioned_virtual_context, load_weather_calibration
from pv_anomaly.pvlof_v12_persistent import (
    fit_persistent_calibration,
    save_persistent_calibration,
)


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
    parser.add_argument("--deficit-quantile", type=float, default=0.995)
    parser.add_argument("--minimum-deficit", type=float, default=0.03)
    parser.add_argument("--minimum-group-strings", type=int, default=2)
    parser.add_argument("--maximum-group-fraction", type=float, default=0.50)
    parser.add_argument("--minimum-upper-strings", type=int, default=1)
    parser.add_argument("--membership-overlap-threshold", type=float, default=0.67)
    parser.add_argument("--minimum-consecutive", type=int, default=6)
    parser.add_argument("--minimum-device-samples", type=int, default=500)
    parser.add_argument("--minimum-device-days", type=int, default=10)
    parser.add_argument("--minimum-plant-count-samples", type=int, default=2000)
    parser.add_argument("--minimum-plant-count-days", type=int, default=10)
    parser.add_argument("--minimum-count-samples", type=int, default=2000)
    parser.add_argument("--minimum-count-days", type=int, default=10)
    parser.add_argument("--shrinkage-k", type=float, default=500.0)
    parser.add_argument("--robust-mad-multiplier", type=float, default=6.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
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
    calibration, fit_report = fit_persistent_calibration(
        scored,
        deficit_quantile=args.deficit_quantile,
        minimum_deficit=args.minimum_deficit,
        minimum_group_strings=args.minimum_group_strings,
        maximum_group_fraction=args.maximum_group_fraction,
        minimum_upper_strings=args.minimum_upper_strings,
        membership_overlap_threshold=args.membership_overlap_threshold,
        minimum_consecutive=args.minimum_consecutive,
        minimum_device_samples=args.minimum_device_samples,
        minimum_device_days=args.minimum_device_days,
        minimum_plant_count_samples=args.minimum_plant_count_samples,
        minimum_plant_count_days=args.minimum_plant_count_days,
        minimum_count_samples=args.minimum_count_samples,
        minimum_count_days=args.minimum_count_days,
        shrinkage_k=args.shrinkage_k,
        robust_mad_multiplier=args.robust_mad_multiplier,
        version="pvlof-v1.2-persistent-group-v3",
    )
    save_persistent_calibration(calibration, args.output)
    report = {
        "input": input_report,
        "weather_roots": args.weather,
        "base_calibration": args.base_calibration,
        "weather_calibration": args.weather_calibration,
        "exclusion": exclusion,
        "conditioned_context": context_report,
        "fit": fit_report,
        "calibration": calibration.to_dict(),
        "output": args.output,
    }
    destination = Path(args.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
