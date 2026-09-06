"""Run PVLOF-V2 and export point-level and event-level results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pv_anomaly.pvlof_io import read_pvlof_source
from pv_anomaly.pvlof_v2 import (
    apply_pvlof_v2,
    collapse_pvlof_v2_events,
    load_calibration,
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _read_input(path: str, start: str | None, end: str | None, timezone: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    if Path(path).is_file():
        frame = pd.read_parquet(path)
        frame["event_time"] = pd.to_datetime(frame["event_time"], errors="raise", utc=True)
        return frame, {"path": path, "rows": len(frame), "mode": "file"}
    if not start or not end:
        raise ValueError("--start and --end are required when --input is a directory")
    return read_pvlof_source(path, start=start, end=end, timezone=timezone)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--calibration", default="artifacts/models/pvlof_v2/calibration.json")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--timezone", default="Asia/Shanghai")
    parser.add_argument("--output", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--events-iso-mod")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    calibration = load_calibration(args.calibration)
    frame, input_report = _read_input(args.input, args.start, args.end, args.timezone)
    scored = apply_pvlof_v2(frame, calibration)
    events = collapse_pvlof_v2_events(
        scored,
        expected_interval_minutes=calibration.expected_interval_minutes,
    )
    iso_mod_events = collapse_pvlof_v2_events(
        scored,
        alert_column="pvlof_v2_iso_mod_alert",
        expected_interval_minutes=calibration.expected_interval_minutes,
    )
    output_path = Path(args.output)
    event_path = Path(args.events)
    iso_mod_event_path = Path(args.events_iso_mod) if args.events_iso_mod else None
    report_path = Path(args.report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    event_path.parent.mkdir(parents=True, exist_ok=True)
    if iso_mod_event_path is not None:
        iso_mod_event_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(output_path, index=False)
    events.to_parquet(event_path, index=False)
    if iso_mod_event_path is not None:
        iso_mod_events.to_parquet(iso_mod_event_path, index=False)
    report = {
        "input": input_report,
        "calibration": calibration.to_dict(),
        "rows": len(scored),
        "eligible_rows": int(scored["v2_eligible"].sum()),
        "virtual_irradiance_coverage": float(scored["virtual_irradiance"].notna().mean()) if len(scored) else 0.0,
        "zero_current_alerts": int(scored["zero_current_alert"].sum()),
        "isolated_alerts": int(scored["isolated_raw_alert"].sum()),
        "isolated_final_alerts": int(scored["isolated_alert"].sum()),
        "collective_alerts": int(scored["collective_raw_alert"].sum()),
        "collective_event_raw_alerts": int(scored["collective_event_raw"].sum()),
        "collective_event_alerts": int(scored["collective_event_alert"].sum()),
        "collective_member_alerts": int(scored["collective_member_alert"].sum()),
        "pvlof_v2_raw_alerts": int(scored["pvlof_v2_raw_alert"].sum()),
        "pvlof_v2_alerts": int(scored["pvlof_v2_alert"].sum()),
        "isolated_directional_alerts": int(scored["isolated_directional_raw_alert"].sum()),
        "isolated_directional_final_alerts": int(scored["isolated_directional_alert"].sum()),
        "pvlof_v2_iso_mod_alerts": int(scored["pvlof_v2_iso_mod_alert"].sum()),
        "events": len(events),
        "iso_mod_events": len(iso_mod_events),
        "outputs": {
            "points": str(output_path),
            "events": str(event_path),
            "iso_mod_events": str(iso_mod_event_path) if iso_mod_event_path else None,
        },
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=_json_safe) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_safe))


if __name__ == "__main__":
    main()
