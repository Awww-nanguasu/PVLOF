"""Calibrate hierarchical LOF thresholds for the isolated PVLOF-V2 branch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pv_anomaly.pvlof_io import read_pvlof_source
from pv_anomaly.pvlof_v2 import apply_pvlof_v2, load_calibration
from pv_anomaly.pvlof_v2_hier import (
    exclude_alarm_windows,
    fit_hierarchical_calibration,
    save_hier_calibration,
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default="data/processed/pvlof/production_device_clean_v2"
    )
    parser.add_argument("--start", default="2026-01-30")
    parser.add_argument("--end", default="2026-06-01", help="Exclusive local date/time")
    parser.add_argument("--timezone", default="Asia/Shanghai")
    parser.add_argument(
        "--base-calibration", default="artifacts/models/pvlof_v2/calibration.json"
    )
    parser.add_argument(
        "--exclude-events",
        default="artifacts/reports/pvlof_cleaning_v2/cleaned_alarm_events.parquet",
    )
    parser.add_argument("--exclude-buffer-minutes", type=int, default=10)
    parser.add_argument("--output", default="artifacts/models/pvlof_v2_hier/calibration.json")
    parser.add_argument("--report", default="artifacts/reports/pvlof_v2_hier_calibration.json")
    parser.add_argument("--lof-quantile", type=float, default=0.995)
    parser.add_argument("--minimum-device-samples", type=int, default=10_000)
    parser.add_argument("--minimum-device-days", type=int, default=20)
    parser.add_argument("--minimum-plant-count-samples", type=int, default=20_000)
    parser.add_argument("--minimum-plant-count-days", type=int, default=20)
    parser.add_argument("--minimum-count-samples", type=int, default=20_000)
    parser.add_argument("--minimum-count-days", type=int, default=20)
    parser.add_argument("--shrinkage-k", type=float, default=5_000.0)
    parser.add_argument(
        "--calibration-version", default="pvlof-v2-hier-v1"
    )
    args = parser.parse_args()

    frame, input_report = read_pvlof_source(
        args.input, start=args.start, end=args.end, timezone=args.timezone
    )
    exclusion_report: dict[str, int] = {"alarm_intervals": 0, "excluded_rows": 0}
    if args.exclude_events and Path(args.exclude_events).exists():
        frame, exclusion_report = exclude_alarm_windows(
            frame,
            args.exclude_events,
            buffer_minutes=args.exclude_buffer_minutes,
        )

    base_calibration = load_calibration(args.base_calibration)
    scored = apply_pvlof_v2(frame, base_calibration)
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
        minimum_consecutive=base_calibration.minimum_consecutive,
        expected_interval_minutes=base_calibration.expected_interval_minutes,
        timezone=args.timezone,
        version=args.calibration_version,
    )
    save_hier_calibration(hierarchical, args.output)
    report = {
        "input": input_report,
        "base_calibration": args.base_calibration,
        "exclusion": exclusion_report,
        "hierarchical_calibration": hierarchical.to_dict(),
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
