"""Audit one plant's exported 15-minute production weather data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pv_anomaly.production_weather_audit import audit_weather_parquet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--plant-id", type=int, required=True)
    parser.add_argument("--timezone", default="Asia/Shanghai")
    parser.add_argument("--expected-interval-minutes", type=int, default=15)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = audit_weather_parquet(
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
    keys = (
        "files",
        "rows",
        "plants",
        "plant_id_mismatches",
        "minimum_local",
        "maximum_local",
        "duplicate_keys",
        "partition_date_mismatches",
        "off_15_minute_grid",
        "within_day_interval_minutes",
        "within_day_gap_segments",
        "estimated_missing_within_days",
        "forecast_ghi",
        "sensor_ghi",
    )
    print(
        json.dumps(
            {key: report[key] for key in keys},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
