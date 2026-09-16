"""Apply PVLOF to a local time range and export point, event, and weak-label reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pv_anomaly.ewma_audit import binary_metrics
from pv_anomaly.pvlof import apply_pvlof, collapse_pvlof_events, load_calibration
from pv_anomaly.pvlof_io import read_pvlof_source


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _read_input(
    path: str,
    start: str | None,
    end: str | None,
    timezone: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = Path(path)
    if source.is_file():
        frame = pd.read_parquet(source)
        frame["event_time"] = pd.to_datetime(
            frame["event_time"], errors="raise", utc=True
        )
        return frame, {"path": path, "rows": len(frame), "mode": "file"}
    if not start or not end:
        raise ValueError("--start and --end are required when --input is a directory")
    return read_pvlof_source(path, start=start, end=end, timezone=timezone)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", default="artifacts/models/pvlof/calibration.json")
    parser.add_argument("--input", default="data/raw/device")
    parser.add_argument("--start")
    parser.add_argument("--end", help="Exclusive local date/time")
    parser.add_argument("--timezone", default="Asia/Shanghai")
    parser.add_argument("--output", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    calibration = load_calibration(args.calibration)
    frame, input_report = _read_input(
        args.input, args.start, args.end, args.timezone
    )
    scored = apply_pvlof(frame, calibration)
    events = collapse_pvlof_events(
        scored,
        expected_interval_minutes=calibration.expected_interval_minutes,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(output_path, index=False)
    event_path = Path(args.events)
    event_path.parent.mkdir(parents=True, exist_ok=True)
    events.to_parquet(event_path, index=False)

    low_eligible = scored["pvlof_eligible"].astype(bool) & ~scored[
        "zero_current_alert"
    ].astype(bool)
    low_metrics = binary_metrics(
        scored.loc[low_eligible, "weak_low_current_label"],
        scored.loc[low_eligible, "pvlof_alert"],
    )
    zero_metrics = binary_metrics(
        scored["weak_zero_current_label"], scored["zero_current_alert"]
    )
    combined_label = scored["weak_low_current_label"].astype(bool) | scored[
        "weak_zero_current_label"
    ].astype(bool)
    combined_metrics = binary_metrics(combined_label, scored["combined_alert"])
    eligible_scores = pd.to_numeric(
        scored.loc[scored["pvlof_eligible"].astype(bool), "pvlof_score"], errors="coerce"
    ).dropna()
    report = {
        "input": input_report,
        "calibration": calibration.to_dict(),
        "outputs": {
            "points": str(output_path),
            "events": str(event_path),
        },
        "wide_samples": len(frame),
        "string_samples": len(scored),
        "pvlof_eligible_samples": int(scored["pvlof_eligible"].sum()),
        "zero_current_alerts": int(scored["zero_current_alert"].sum()),
        "pvlof_raw_alerts": int(scored["pvlof_raw_alert"].sum()),
        "pvlof_alerts": int(scored["pvlof_alert"].sum()),
        "combined_alerts": int(scored["combined_alert"].sum()),
        "events": len(events),
        "alert_devices": int(
            scored.loc[scored["combined_alert"].astype(bool), "device_no"].nunique()
        ),
        "alert_strings": int(
            scored.loc[scored["combined_alert"].astype(bool), ["device_no", "string_no"]]
            .drop_duplicates()
            .shape[0]
        ),
        "score_quantiles": (
            {
                str(quantile): float(eligible_scores.quantile(quantile))
                for quantile in (0.5, 0.9, 0.95, 0.99, 0.995, 1.0)
            }
            if len(eligible_scores)
            else {}
        ),
        "weak_label_metrics": {
            "nonzero_low_current_status_2": low_metrics,
            "zero_current_status_4": zero_metrics,
            "combined_status_2_or_4": combined_metrics,
            "note": "string_status is a weak label and may be derived from fixed current rules",
        },
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
