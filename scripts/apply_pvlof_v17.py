"""Apply PVLOF v1.7 to a fully memory-applied v1.6 points Parquet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pv_anomaly.pvlof_events import reconstruct_pvlof_events
from pv_anomaly.pvlof_v17 import apply_pvlof_v17, load_config


def _event_compatibility_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Map v1.7 aggregate memory fields to the generic v1.6 event adapter."""

    candidates = [
        "plant_id", "device_no", "event_time", "string_no", "pvlof_v17_alert",
        "isolated_directional_raw_alert", "isolated_hier_raw_alert",
        "isolated_raw_alert", "collective_raw_alert",
        "isolated_directional_alert", "isolated_hier_strict_alert",
        "isolated_alert", "collective_member_alert", "pvlof_v2_legacy_alert",
        "isolated_directional_consecutive", "isolated_hier_strict_consecutive",
        "isolated_consecutive", "collective_event_consecutive",
        "pvlof_v17_raw_anomaly", "pvlof_v17_entry_streak",
        "pvlof_v17_normal_streak", "pvlof_v17_memory_active",
        "pvlof_v17_memory_reactivated_alert", "pvlof_v17_memory_clear_code",
        "pvlof_v17_segmentation_strict_alert",
    ]
    result = frame[[column for column in candidates if column in frame]].copy()
    mapping = {
        "pvlof_v17_raw_anomaly": "pvlof_v16_raw_anomaly",
        "pvlof_v17_entry_streak": "pvlof_v16_entry_streak",
        "pvlof_v17_normal_streak": "pvlof_v16_normal_streak",
        "pvlof_v17_memory_active": "pvlof_v16_memory_active",
        "pvlof_v17_memory_reactivated_alert": (
            "pvlof_v16_memory_reactivated_alert"
        ),
        "pvlof_v17_memory_clear_code": "pvlof_v16_memory_clear_code",
    }
    for source, destination in mapping.items():
        result[destination] = result[source]
    return result


def _mark_segmentation_branch(
    evidence: pd.DataFrame,
    events: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    for frame in (evidence, events):
        if "confirmation_branch" in frame:
            frame.loc[
                frame["confirmation_branch"].eq("final_union"),
                "confirmation_branch",
            ] = "penalized_segmentation"
    return evidence, events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--timezone", default="Asia/Shanghai")
    args = parser.parse_args()

    source = pd.read_parquet(args.input)
    config = load_config(args.config)
    result = apply_pvlof_v17(source, config)
    evidence, events, unconfirmed = reconstruct_pvlof_events(
        _event_compatibility_frame(result),
        version=config.version,
        final_alert_column="pvlof_v17_alert",
        expected_interval_minutes=config.expected_interval_minutes,
        timezone=args.timezone,
        memory_prefix="pvlof_v16",
    )
    evidence, events = _mark_segmentation_branch(evidence, events)

    output = Path(args.output)
    event_output = Path(args.events)
    report_output = Path(args.report)
    evidence_output = event_output.with_name("pvlof_v17_evidence_points.parquet")
    unconfirmed_output = event_output.with_name(
        "pvlof_v17_unconfirmed_candidates.parquet"
    )
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
            "segmentation_raw_candidates": int(
                group["pvlof_v17_segmentation_raw_candidate"].sum()
            ),
            "segmentation_alert_points": int(
                group["pvlof_v17_segmentation_alert"].sum()
            ),
            "v1_7_added_alert_points": int(group["pvlof_v17_added_alert"].sum()),
            "v1_7_alert_points": int(group["pvlof_v17_alert"].sum()),
        }
    report = {
        "input": args.input,
        "config": args.config,
        "version": config.version,
        "rows": len(result),
        "v1_6_alert_points": int(result["pvlof_v16_alert"].sum()),
        "segmentation_structural_points": int(
            result["pvlof_v17_segmentation_structural_member"].sum()
        ),
        "segmentation_raw_candidates": int(
            result["pvlof_v17_segmentation_raw_candidate"].sum()
        ),
        "segmentation_alert_points": int(
            result["pvlof_v17_segmentation_alert"].sum()
        ),
        "v1_7_added_alert_points": int(result["pvlof_v17_added_alert"].sum()),
        "v1_7_alert_points": int(result["pvlof_v17_alert"].sum()),
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
