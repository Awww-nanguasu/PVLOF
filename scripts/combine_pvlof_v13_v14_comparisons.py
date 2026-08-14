"""Combine per-plant PVLOF v1.3/v1.4 comparison CSV packages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


FILES = {
    "events": "pvlof_v13_v14_dual_gate_events.csv",
    "points": "pvlof_v13_v14_dual_gate_points.csv",
    "filtered": "pvlof_v13_only_filtered_strings.csv",
}


def _read_all(root: Path, filename: str) -> pd.DataFrame:
    paths = sorted((root / "by_plant").glob(f"plant_id=*/{filename}"))
    if not paths:
        raise FileNotFoundError(f"No per-plant {filename} files found under {root / 'by_plant'}")
    frames = [pd.read_csv(path, dtype={"plant_id": "string"}) for path in paths]
    result = pd.concat(frames, ignore_index=True)
    if "row" in result:
        result = result.drop(columns="row")
    result.insert(0, "row", range(1, len(result) + 1))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", required=True)
    parser.add_argument("--output-directory")
    args = parser.parse_args()

    source = Path(args.input_directory)
    output = Path(args.output_directory) if args.output_directory else source
    output.mkdir(parents=True, exist_ok=True)

    events = _read_all(source, FILES["events"])
    points = _read_all(source, FILES["points"])
    filtered = _read_all(source, FILES["filtered"])
    for frame in (events, points):
        if "event_id" in frame:
            frame["event_id"] = (
                "plant-" + frame["plant_id"].astype(str) + "-" + frame["event_id"].astype(str)
            )

    event_path = output / FILES["events"]
    point_path = output / FILES["points"]
    filtered_path = output / FILES["filtered"]
    events.to_csv(event_path, index=False, encoding="utf-8-sig")
    points.to_csv(point_path, index=False, encoding="utf-8-sig")
    filtered.to_csv(filtered_path, index=False, encoding="utf-8-sig")

    per_plant = {}
    plants = sorted(set(events["plant_id"].astype(str)) | set(points["plant_id"].astype(str)))
    for plant in plants:
        plant_events = events[events["plant_id"].astype(str).eq(plant)]
        plant_points = points[points["plant_id"].astype(str).eq(plant)]
        plant_filtered = filtered[filtered["plant_id"].astype(str).eq(plant)]
        per_plant[plant] = {
            "events": len(plant_events),
            "event_cases": plant_events["comparison_case"].value_counts().to_dict(),
            "points": len(plant_points),
            "point_cases": plant_points["comparison_case"].value_counts().to_dict(),
            "v1_3_only_filtered_strings": len(plant_filtered),
        }
    report = {
        "plants": plants,
        "events": len(events),
        "event_cases": events["comparison_case"].value_counts().to_dict(),
        "points": len(points),
        "point_cases": points["comparison_case"].value_counts().to_dict(),
        "v1_3_only_filtered_strings": len(filtered),
        "per_plant": per_plant,
        "outputs": {
            "events": str(event_path),
            "points": str(point_path),
            "v1_3_only_filtered_strings": str(filtered_path),
        },
    }
    report_path = output / "summary.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
