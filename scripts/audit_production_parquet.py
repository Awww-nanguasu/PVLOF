"""Audit one plant's exported device Parquet partitions without contacting ES."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pv_anomaly.production_parquet_audit import audit_parquet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        required=True,
        help="One plant root, e.g. data/raw/device/plant_id=892",
    )
    parser.add_argument("--plant-id", type=int, required=True)
    parser.add_argument("--timezone", default="Asia/Shanghai")
    parser.add_argument("--expected-interval-minutes", type=int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = audit_parquet(
        Path(args.root),
        plant_id=args.plant_id,
        timezone_name=args.timezone,
        expected_interval_minutes=args.expected_interval_minutes,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output}")
    summary_keys = (
        "files",
        "rows",
        "plants",
        "plant_id_mismatches",
        "devices",
        "duplicate_keys",
        "partition_date_mismatches",
        "continuity",
        "missing_required_fields",
    )
    print(
        json.dumps(
            {key: report[key] for key in summary_keys if key in report},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
