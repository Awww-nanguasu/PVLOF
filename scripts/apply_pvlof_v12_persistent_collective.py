"""Apply the parallel persistent collective branch to a scored v1.2 Parquet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pv_anomaly.pvlof_io import parse_plant_id_mappings
from pv_anomaly.pvlof_v12_persistent import (
    apply_persistent_collective,
    collapse_persistent_events,
    load_persistent_calibration,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--timezone", default="Asia/Shanghai")
    parser.add_argument(
        "--plant-id-map",
        action="append",
        default=[],
        metavar="SOURCE=MODEL",
        help="Fallback mapping when model_plant_id is not embedded in the scored input",
    )
    args = parser.parse_args()

    scored = pd.read_parquet(args.input)
    source_plant = scored["plant_id"].astype(str)
    mappings = parse_plant_id_mappings(args.plant_id_map)
    if "model_plant_id" in scored.columns and scored["model_plant_id"].notna().any():
        model_plant = scored["model_plant_id"].astype("string").fillna(source_plant)
        mapping_mode = "embedded_model_plant_id"
    elif mappings:
        model_plant = source_plant.map(mappings).fillna(source_plant)
        mapping_mode = "command_line_mapping"
    else:
        model_plant = source_plant
        mapping_mode = "identity"
    scored["_persistent_source_plant_id"] = source_plant
    scored["plant_id"] = model_plant.astype(str)
    calibration = load_persistent_calibration(args.calibration)
    result = apply_persistent_collective(scored, calibration)
    result["model_plant_id"] = result["plant_id"].astype(str)
    result["plant_id"] = result["_persistent_source_plant_id"].astype(str)
    result = result.drop(columns="_persistent_source_plant_id")
    events = collapse_persistent_events(
        result,
        timezone=args.timezone,
        expected_interval_minutes=calibration.expected_interval_minutes,
    )
    output = Path(args.output)
    event_output = Path(args.events)
    report_output = Path(args.report)
    for destination in (output, event_output, report_output):
        destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    events.to_parquet(event_output, index=False)
    report = {
        "input": args.input,
        "calibration": args.calibration,
        "plant_id_mapping": {
            "mode": mapping_mode,
            "requested": mappings,
            "source_plants": sorted(source_plant.unique().tolist()),
            "model_plants": sorted(model_plant.astype(str).unique().tolist()),
        },
        "rows": len(result),
        "persistent_raw_device_time_points": int(
            result[["plant_id", "device_no", "event_time", "persistent_raw_alert"]]
            .drop_duplicates()
            ["persistent_raw_alert"].sum()
        ),
        "persistent_confirmed_device_time_points": int(
            result[["plant_id", "device_no", "event_time", "persistent_event_alert"]]
            .drop_duplicates()
            ["persistent_event_alert"].sum()
        ),
        "persistent_alert_points": int(result["persistent_collective_member_alert"].sum()),
        "combined_alert_points": int(result["pvlof_v12_combined_alert"].sum())
        if "pvlof_v12_combined_alert" in result
        else None,
        "events": len(events),
        "outputs": {"points": str(output), "events": str(event_output)},
    }
    report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
