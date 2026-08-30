"""Combine per-plant PVLOF v1.4/v1.5 comparison CSV packages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


FILES = {
    "events": "pvlof_v14_v15_events.csv",
    "points": "pvlof_v14_v15_points.csv",
    "additions": "pvlof_v15_additions.csv",
    "evidence_points": "pvlof_v14_v15_evidence_points.csv",
    "string_events": "pvlof_v14_v15_string_events.csv",
    "unconfirmed_candidates": "pvlof_unconfirmed_candidates.csv",
    "customer_events": "pvlof_v14_v15_customer_events.csv",
}


def _combine(root: Path, filename: str) -> pd.DataFrame:
    paths = sorted((root / "by_plant").glob(f"plant_id=*/{filename}"))
    if not paths:
        raise FileNotFoundError(f"No {filename} files found under {root / 'by_plant'}")
    result = pd.concat(
        [pd.read_csv(path, dtype={"plant_id": "string"}) for path in paths],
        ignore_index=True,
    )
    result = result.drop(columns="row", errors="ignore")
    result.insert(0, "row", range(1, len(result) + 1))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", required=True)
    parser.add_argument(
        "--customer-events-only",
        action="store_true",
        help="Combine only compact customer event CSVs; preserve detailed outputs.",
    )
    args = parser.parse_args()
    root = Path(args.input_directory)
    root.mkdir(parents=True, exist_ok=True)

    if args.customer_events_only:
        customer_events = _combine(root, FILES["customer_events"])
        customer_events["event_id"] = (
            "plant-" + customer_events["plant_id"].astype(str) + "-"
            + customer_events["event_id"].astype(str)
        )
        customer_events.to_csv(
            root / FILES["customer_events"], index=False, encoding="utf-8-sig"
        )
        report = {
            "customer_events": len(customer_events),
            "customer_event_cases": customer_events[
                "comparison_case"
            ].value_counts().to_dict(),
            "output": str(root / FILES["customer_events"]),
        }
        (root / "customer_events_summary.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    events = _combine(root, FILES["events"])
    points = _combine(root, FILES["points"])
    additions = _combine(root, FILES["additions"])
    evidence = _combine(root, FILES["evidence_points"])
    string_events = _combine(root, FILES["string_events"])
    unconfirmed = _combine(root, FILES["unconfirmed_candidates"])
    customer_events = _combine(root, FILES["customer_events"])
    for frame in (events, points, evidence, string_events):
        if "event_id" in frame:
            frame["event_id"] = (
                "plant-" + frame["plant_id"].astype(str) + "-"
                + frame["event_id"].astype(str)
            )
    if "event_id" in customer_events:
        customer_events["event_id"] = (
            "plant-" + customer_events["plant_id"].astype(str) + "-"
            + customer_events["event_id"].astype(str)
        )
    for column in [
        column for column in events.columns if column.endswith("_source_event_ids")
    ]:
        events[column] = [
            ",".join(
                f"plant-{plant}-{event_id}"
                for event_id in str(value).split(",") if event_id
            ) if pd.notna(value) else value
            for plant, value in zip(events["plant_id"], events[column], strict=True)
        ]

    events.to_csv(root / FILES["events"], index=False, encoding="utf-8-sig")
    points.to_csv(root / FILES["points"], index=False, encoding="utf-8-sig")
    additions.to_csv(root / FILES["additions"], index=False, encoding="utf-8-sig")
    evidence.to_csv(root / FILES["evidence_points"], index=False, encoding="utf-8-sig")
    string_events.to_csv(root / FILES["string_events"], index=False, encoding="utf-8-sig")
    unconfirmed.to_csv(
        root / FILES["unconfirmed_candidates"], index=False, encoding="utf-8-sig"
    )
    customer_events.to_csv(
        root / FILES["customer_events"], index=False, encoding="utf-8-sig"
    )

    per_plant = {}
    for plant in sorted(set(points["plant_id"].astype(str))):
        plant_points = points[points["plant_id"].astype(str).eq(plant)]
        plant_events = events[events["plant_id"].astype(str).eq(plant)]
        plant_additions = additions[additions["plant_id"].astype(str).eq(plant)]
        plant_customer_events = customer_events[
            customer_events["plant_id"].astype(str).eq(plant)
        ]
        per_plant[plant] = {
            "point_cases": plant_points["comparison_case"].value_counts().to_dict(),
            "event_cases": plant_events["comparison_case"].value_counts().to_dict(),
            "customer_event_cases": plant_customer_events[
                "comparison_case"
            ].value_counts().to_dict(),
            "v1_5_addition_reasons": plant_additions["addition_reason"].value_counts().to_dict(),
        }
    report = {
        "events": len(events),
        "event_cases": events["comparison_case"].value_counts().to_dict(),
        "points": len(points),
        "point_cases": points["comparison_case"].value_counts().to_dict(),
        "v1_5_additions": len(additions),
        "v1_5_addition_reasons": additions["addition_reason"].value_counts().to_dict(),
        "evidence_points": len(evidence),
        "string_events": len(string_events),
        "unconfirmed_candidate_runs": len(unconfirmed),
        "customer_events": len(customer_events),
        "customer_event_cases": customer_events[
            "comparison_case"
        ].value_counts().to_dict(),
        "per_plant": per_plant,
        "outputs": {key: str(root / filename) for key, filename in FILES.items()},
    }
    (root / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
