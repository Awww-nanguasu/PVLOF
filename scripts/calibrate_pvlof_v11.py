"""Calibrate PVLOF-V2 v1.1 with a historical non-contiguous channel inventory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pv_anomaly.pvlof_channels import infer_channel_inventory
from pv_anomaly.pvlof_io import read_pvlof_source
from pv_anomaly.pvlof_v2 import fit_pvlof_v2_calibration, save_calibration
from pv_anomaly.pvlof_v2_hier import exclude_alarm_windows


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
        "--exclude-events",
        default="artifacts/reports/pvlof_cleaning_v2/cleaned_alarm_events.parquet",
    )
    parser.add_argument("--exclude-buffer-minutes", type=int, default=10)
    parser.add_argument("--minimum-evidence-samples", type=int, default=3)
    parser.add_argument("--minimum-evidence-days", type=int, default=1)
    parser.add_argument(
        "--output", default="artifacts/models/pvlof_v11/base_calibration.json"
    )
    parser.add_argument(
        "--inventory-output", default="artifacts/models/pvlof_v11/channel_inventory.json"
    )
    parser.add_argument(
        "--report", default="artifacts/reports/pvlof_v11_base_calibration.json"
    )
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--minimum-peer-devices", type=int, default=5)
    parser.add_argument("--minimum-strings", type=int, default=4)
    parser.add_argument("--collective-overlap-threshold", type=float, default=0.50)
    parser.add_argument(
        "--minimum-consecutive",
        type=int,
        default=3,
        help="Consecutive 5-minute candidates required; 3 represents 15 minutes",
    )
    parser.add_argument("--max-score-rows", type=int, default=100_000)
    args = parser.parse_args()

    frame, input_report = read_pvlof_source(
        args.input, start=args.start, end=args.end, timezone=args.timezone
    )
    inventory, inventory_report = infer_channel_inventory(
        frame,
        minimum_evidence_samples=args.minimum_evidence_samples,
        minimum_evidence_days=args.minimum_evidence_days,
        timezone=args.timezone,
    )
    inventory_path = Path(args.inventory_output)
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_path.write_text(
        json.dumps(
            {"inventory": inventory, "audit": inventory_report},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    exclusion_report: dict[str, int] = {"alarm_intervals": 0, "excluded_rows": 0}
    if args.exclude_events and Path(args.exclude_events).exists():
        frame, exclusion_report = exclude_alarm_windows(
            frame,
            args.exclude_events,
            buffer_minutes=args.exclude_buffer_minutes,
        )
    calibration, fit_report = fit_pvlof_v2_calibration(
        frame,
        n_neighbors=args.neighbors,
        minimum_peer_devices=args.minimum_peer_devices,
        minimum_strings=args.minimum_strings,
        collective_overlap_threshold=args.collective_overlap_threshold,
        minimum_consecutive=args.minimum_consecutive,
        max_score_rows=args.max_score_rows,
        configured_strings=inventory,
        version="pvlof-v2-v1.1",
    )
    save_calibration(calibration, args.output)
    report = {
        "input": input_report,
        "inventory": {
            key: inventory_report[key]
            for key in (
                "version",
                "devices",
                "configured_channels",
                "devices_with_count_mismatch",
                "devices_with_noncontiguous_channels",
            )
        },
        "inventory_output": str(inventory_path),
        "exclusion": exclusion_report,
        "calibration_version": calibration.version,
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
