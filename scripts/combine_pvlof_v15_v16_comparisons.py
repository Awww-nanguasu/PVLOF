"""Combine per-plant compact PVLOF v1.5/v1.6 customer event tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


FILENAME = "pvlof_v15_v16_customer_events.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", required=True)
    args = parser.parse_args()
    root = Path(args.input_directory)
    paths = sorted((root / "by_plant").glob(f"plant_id=*/{FILENAME}"))
    if not paths:
        raise FileNotFoundError(f"No {FILENAME} files found under {root / 'by_plant'}")
    events = pd.concat(
        [pd.read_csv(path, dtype={"plant_id": "string"}) for path in paths],
        ignore_index=True,
    ).drop(columns="row", errors="ignore")
    events["event_id"] = (
        "plant-" + events["plant_id"].astype(str) + "-"
        + events["event_id"].astype(str)
    )
    events = events.sort_values(
        ["plant_id", "device_no", "raise_time_local"]
    ).reset_index(drop=True)
    events.insert(0, "row", range(1, len(events) + 1))
    output = root / FILENAME
    events.to_csv(output, index=False, encoding="utf-8-sig")
    report = {
        "events": len(events),
        "event_cases": events["comparison_case"].value_counts().to_dict(),
        "output": str(output),
    }
    (root / "customer_events_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
