"""Calibrate first-version PVLOF on candidate-normal training string currents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pv_anomaly.pvlof import fit_pvlof_calibration, save_calibration
from pv_anomaly.pvlof_io import read_pvlof_source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/raw/device")
    parser.add_argument("--start", default="2026-01-23")
    parser.add_argument("--end", default="2026-06-01", help="Exclusive local date/time")
    parser.add_argument("--timezone", default="Asia/Shanghai")
    parser.add_argument("--output", default="artifacts/models/pvlof/calibration.json")
    parser.add_argument("--report", default="artifacts/reports/pvlof_calibration.json")
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--quantile", type=float, default=0.995)
    parser.add_argument("--minimum-power-ratio", type=float, default=0.10)
    parser.add_argument("--maximum-power-ratio", type=float, default=1.10)
    parser.add_argument("--zero-current-threshold", type=float, default=0.0)
    parser.add_argument("--minimum-relative-drop", type=float, default=0.10)
    parser.add_argument("--minimum-strings", type=int, default=6)
    parser.add_argument("--minimum-consecutive", type=int, default=2)
    parser.add_argument("--interval-minutes", type=int, default=5)
    parser.add_argument("--distance-floor", type=float, default=0.01)
    parser.add_argument(
        "--eligible-status-code",
        type=int,
        action="append",
        dest="eligible_status_codes",
        help="Repeat for each eligible device status; defaults to online=1 and alert=4",
    )
    args = parser.parse_args()

    frame, input_report = read_pvlof_source(
        args.input,
        start=args.start,
        end=args.end,
        timezone=args.timezone,
    )
    calibration, calibration_report = fit_pvlof_calibration(
        frame,
        n_neighbors=args.neighbors,
        quantile=args.quantile,
        minimum_power_ratio=args.minimum_power_ratio,
        maximum_power_ratio=args.maximum_power_ratio,
        zero_current_threshold=args.zero_current_threshold,
        minimum_relative_drop=args.minimum_relative_drop,
        minimum_strings=args.minimum_strings,
        minimum_consecutive=args.minimum_consecutive,
        expected_interval_minutes=args.interval_minutes,
        distance_floor=args.distance_floor,
        eligible_status_codes=tuple(args.eligible_status_codes or (1, 4)),
    )
    save_calibration(calibration, args.output)
    report = {
        "input": input_report,
        "calibration": calibration.to_dict(),
        "calibration_report": calibration_report,
        "output": args.output,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
