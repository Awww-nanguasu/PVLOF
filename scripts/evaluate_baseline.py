"""Evaluate the five-minute persistence baseline on all chronological splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pv_anomaly.models.baseline import evaluate_persistence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/processed/transformer")
    parser.add_argument("--output", default="artifacts/models/baseline/metrics.json")
    args = parser.parse_args()
    root = Path(args.data_root)
    report = {
        split: evaluate_persistence(root / f"{split}.parquet")
        for split in ("train", "validation", "test")
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(report, ensure_ascii=False, indent=2)
    output.write_text(content + "\n", encoding="utf-8")
    print(content)


if __name__ == "__main__":
    main()

