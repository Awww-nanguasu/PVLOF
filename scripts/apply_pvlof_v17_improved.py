"""Apply improved PVLOF v1.7 to same-run v1.6 full points."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


import pandas as pd

from pv_anomaly.pvlof_v17_improved import apply_pvlof_v17_improved, load_config
from pv_anomaly.pvlof_v17_pipeline import select_alarm_targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Same-run v1.6 full points")
    parser.add_argument("--config", required=True)
    parser.add_argument("--alarm-points", required=True)
    parser.add_argument("--output", required=True, help="Improved full points Parquet")
    parser.add_argument("--alarm-output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    source = pd.read_parquet(args.input)
    config = load_config(args.config)
    result = apply_pvlof_v17_improved(source, config)
    alarm_points = pd.read_parquet(args.alarm_points)
    alarm_result, target_report = select_alarm_targets(result, alarm_points)

    output = Path(args.output)
    alarm_output = Path(args.alarm_output)
    report_output = Path(args.report)
    for destination in (output, alarm_output, report_output):
        destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    alarm_result.to_parquet(alarm_output, index=False)

    preserved_violation = (
        result["pvlof_v16_alert"].fillna(False).astype(bool)
        & ~result["pvlof_v17_improved_alert"].fillna(False).astype(bool)
    )
    report = {
        "input": args.input,
        "config": args.config,
        "version": config.version,
        "rows": int(len(result)),
        "alarm_rows": int(len(alarm_result)),
        "target_filter": target_report,
        "segmentation_raw_candidates": int(
            result["pvlof_v17_improved_segmentation_raw_candidate"].sum()
        ),
        "original_candidates_accepted": int(
            result["pvlof_v17_improved_segmentation_original_candidate_accepted"].sum()
        ),
        "fragmented_reference_rescue_candidates": int(
            result["pvlof_v17_improved_segmentation_rescue_raw_candidate"].sum()
        ),
        "partial_next_segment_candidates": int(
            result[
                "pvlof_v17_improved_segmentation_partial_next_segment_raw_candidate"
            ].sum()
        ),
        "small_candidate_rescue_candidates": int(
            result[
                "pvlof_v17_improved_segmentation_small_candidate_rescue_raw_candidate"
            ].sum()
        ),
        "segmentation_alert_points": int(
            result["pvlof_v17_improved_segmentation_alert"].sum()
        ),
        "improved_added_alert_points": int(
            result["pvlof_v17_improved_added_alert"].sum()
        ),
        "improved_alert_points": int(result["pvlof_v17_improved_alert"].sum()),
        "v16_preservation_violations": int(preserved_violation.sum()),
        "outputs": {
            "full_points": str(output),
            "alarm_points": str(alarm_output),
        },
    }
    report_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
