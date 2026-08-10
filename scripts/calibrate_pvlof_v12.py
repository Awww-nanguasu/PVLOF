"""Calibrate forecast-GHI conditioned PVLOF v1.2 base parameters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pv_anomaly.forecast_features import read_weather_forecast
from pv_anomaly.pvlof_io import read_pvlof_source
from pv_anomaly.pvlof_v2 import fit_pvlof_v2_calibration, load_calibration, save_calibration
from pv_anomaly.pvlof_v2_hier import exclude_alarm_windows
from pv_anomaly.pvlof_v12 import (
    build_conditioned_virtual_context,
    fit_weather_calibration,
    save_weather_calibration,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/processed/pvlof/production_device_clean_v2")
    parser.add_argument("--weather", action="append", required=True)
    parser.add_argument("--start", default="2026-01-30")
    parser.add_argument("--end", default="2026-06-01")
    parser.add_argument("--timezone", default="Asia/Shanghai")
    parser.add_argument("--v11-calibration", required=True)
    parser.add_argument("--exclude-events")
    parser.add_argument("--exclude-buffer-minutes", type=int, default=10)
    parser.add_argument("--weather-output", required=True)
    parser.add_argument("--base-output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--max-score-rows", type=int, default=100_000)
    parser.add_argument(
        "--maximum-collective-fraction",
        type=float,
        default=0.50,
        help="Maximum lower-group share; groups above this minority limit are rejected",
    )
    parser.add_argument("--minimum-isolated-relative-drop", type=float, default=0.0)
    parser.add_argument(
        "--version",
        default="pvlof-v1.2-forecast-ghi-jaccard-group50",
    )
    parser.add_argument(
        "--forecast-source-offset-minutes",
        type=int,
        choices=(0, 15),
        default=0,
        help="Freeze forecast timestamp semantics; 0 is the causal current-time default",
    )
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
    v11 = load_calibration(args.v11_calibration)
    weather_calibration, weather_fit = fit_weather_calibration(
        frame,
        v11,
        weather,
        candidate_source_offsets_minutes=(args.forecast_source_offset_minutes,),
    )
    context, context_report = build_conditioned_virtual_context(
        frame, v11, weather_calibration, weather
    )
    v12, base_fit = fit_pvlof_v2_calibration(
        frame,
        n_neighbors=v11.n_neighbors,
        lof_quantile=v11.lof_quantile,
        residual_quantile=v11.residual_quantile,
        collective_gap_quantile=v11.collective_gap_quantile,
        minimum_collective_gap=v11.minimum_collective_gap,
        minimum_collective_strings=v11.minimum_collective_strings,
        maximum_collective_fraction=args.maximum_collective_fraction,
        minimum_isolated_relative_drop=args.minimum_isolated_relative_drop,
        collective_overlap_threshold=v11.collective_overlap_threshold,
        minimum_peer_devices=v11.minimum_peer_devices,
        minimum_strings=v11.minimum_strings,
        minimum_consecutive=v11.minimum_consecutive,
        expected_interval_minutes=v11.expected_interval_minutes,
        zero_current_threshold=v11.zero_current_threshold,
        minimum_virtual_irradiance=v11.minimum_virtual_irradiance,
        distance_floor=v11.distance_floor,
        max_score_rows=args.max_score_rows,
        configured_strings=v11.configured_strings,
        version=args.version,
        virtual_override=context,
    )
    save_weather_calibration(weather_calibration, args.weather_output)
    save_calibration(v12, args.base_output)
    report = {
        "input": input_report,
        "weather_roots": args.weather,
        "forecast_source_offset_minutes": args.forecast_source_offset_minutes,
        "v11_calibration": args.v11_calibration,
        "exclusion": exclusion,
        "weather_fit": weather_fit,
        "conditioned_context": context_report,
        "base_fit": base_fit,
        "outputs": {"weather": args.weather_output, "base": args.base_output},
    }
    destination = Path(args.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
