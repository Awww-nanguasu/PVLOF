"""Export evidence-aware PVLOF candidate, confirmation and recovery events."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pv_anomaly.pvlof_events import reconstruct_pvlof_events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--final-alert-column", required=True)
    parser.add_argument("--use-v15-memory", action="store_true")
    parser.add_argument("--interval-minutes", type=int, default=5)
    parser.add_argument("--timezone", default="Asia/Shanghai")
    parser.add_argument("--output-directory", required=True)
    args = parser.parse_args()

    frame = pd.read_parquet(args.input)
    evidence, events, unconfirmed = reconstruct_pvlof_events(
        frame,
        version=args.version,
        final_alert_column=args.final_alert_column,
        expected_interval_minutes=args.interval_minutes,
        timezone=args.timezone,
        use_v15_memory=args.use_v15_memory,
    )
    output = Path(args.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "evidence_points": output / "pvlof_evidence_points.csv",
        "string_events": output / "pvlof_string_events.csv",
        "unconfirmed_candidates": output / "pvlof_unconfirmed_candidates.csv",
    }
    evidence.to_csv(paths["evidence_points"], index=False, encoding="utf-8-sig")
    events.to_csv(paths["string_events"], index=False, encoding="utf-8-sig")
    unconfirmed.to_csv(
        paths["unconfirmed_candidates"], index=False, encoding="utf-8-sig"
    )
    report = {
        "input": args.input,
        "version": args.version,
        "final_alert_column": args.final_alert_column,
        "use_v15_memory": args.use_v15_memory,
        "evidence_points": len(evidence),
        "string_events": len(events),
        "unconfirmed_candidate_runs": len(unconfirmed),
        "point_states": (
            evidence["point_state"].value_counts().to_dict()
            if "point_state" in evidence else {}
        ),
        "confirmation_branches": (
            events["confirmation_branch"].value_counts().to_dict()
            if "confirmation_branch" in events else {}
        ),
        "outputs": {key: str(path) for key, path in paths.items()},
    }
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
