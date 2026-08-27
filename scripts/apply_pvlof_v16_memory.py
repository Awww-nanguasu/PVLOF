"""Apply PVLOF v1.6 confirmed-anomaly memory to a scored Parquet file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pv_anomaly.pvlof_events import reconstruct_pvlof_events
from pv_anomaly.pvlof_v16 import (
    apply_confirmed_anomaly_memory_v16,
    load_memory_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--memory-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    source = pd.read_parquet(args.input)
    config = load_memory_config(args.memory_config)
    result = apply_confirmed_anomaly_memory_v16(source, config)
    evidence, events, unconfirmed = reconstruct_pvlof_events(
        result,
        version=config.version,
        final_alert_column="pvlof_v16_alert",
        expected_interval_minutes=config.expected_interval_minutes,
        memory_prefix="pvlof_v16",
    )

    output = Path(args.output)
    event_output = Path(args.events)
    report_output = Path(args.report)
    evidence_output = event_output.with_name("pvlof_evidence_points.parquet")
    unconfirmed_output = event_output.with_name("pvlof_unconfirmed_candidates.parquet")
    for destination in (output, event_output, report_output):
        destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    events.to_parquet(event_output, index=False)
    evidence.to_parquet(evidence_output, index=False)
    unconfirmed.to_parquet(unconfirmed_output, index=False)

    per_plant = {}
    for plant, group in result.groupby("plant_id", observed=True):
        per_plant[str(plant)] = {
            "rows": len(group),
            "raw_anomaly_points": int(group["pvlof_v16_raw_anomaly"].sum()),
            "memory_reactivated_points": int(
                group["pvlof_v16_memory_reactivated_alert"].sum()
            ),
            "v1_6_alert_points": int(group["pvlof_v16_alert"].sum()),
        }
    report = {
        "input": args.input,
        "memory_config": args.memory_config,
        "rows": len(result),
        "raw_anomaly_points": int(result["pvlof_v16_raw_anomaly"].sum()),
        "memory_active_rows": int(result["pvlof_v16_memory_active"].sum()),
        "memory_reactivated_points": int(
            result["pvlof_v16_memory_reactivated_alert"].sum()
        ),
        "v1_6_alert_points": int(result["pvlof_v16_alert"].sum()),
        "events": len(events),
        "evidence_points": len(evidence),
        "unconfirmed_candidate_runs": len(unconfirmed),
        "per_plant": per_plant,
        "outputs": {
            "points": str(output),
            "events": str(event_output),
            "evidence_points": str(evidence_output),
            "unconfirmed_candidates": str(unconfirmed_output),
        },
    }
    report_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
