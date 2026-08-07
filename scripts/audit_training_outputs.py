"""Verify chronological candidate-normal train/validation/test Parquet outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/processed/transformer")
    parser.add_argument("--output", default="artifacts/reports/training_output_audit.json")
    args = parser.parse_args()
    root = Path(args.root)
    results = {}
    all_columns: set[str] = set()
    for split in ("train", "validation", "test"):
        path = root / f"{split}.parquet"
        frame = pd.read_parquet(path)
        all_columns.update(frame.columns)
        event_time = pd.to_datetime(frame["event_time"], utc=True)
        target_time = pd.to_datetime(frame["target_time"], utc=True)
        results[split] = {
            "rows": len(frame),
            "size_mb": round(path.stat().st_size / 1024 / 1024, 3),
            "minimum_event_time_utc": event_time.min().isoformat(),
            "maximum_target_time_utc": target_time.max().isoformat(),
            "devices": int(frame["device_no"].nunique()),
            "duplicate_input_keys": int(
                frame.duplicated(["event_time", "plant_id", "device_no"]).sum()
            ),
            "candidate_false_rows": int((~frame["candidate_normal"]).sum()),
            "non_five_minute_targets": int(
                ((target_time - event_time) != pd.Timedelta(minutes=5)).sum()
            ),
            "non_running_rows": int((frame["status_code"] != 1).sum()),
            "non_positive_power_rows": int((frame["active_power"] <= 0).sum()),
            "weather_matched_percent": round(float(frame["weather_matched"].mean() * 100), 4),
            "sensor_ghi_null_percent": round(float(frame["sensor_ghi"].isna().mean() * 100), 4),
            "forecast_ghi_null_percent": round(float(frame["forecast_ghi"].isna().mean() * 100), 4),
        }
    report = {
        "splits": results,
        "total_rows": sum(item["rows"] for item in results.values()),
        "low_current_count_present": "low_current_count" in all_columns,
        "target": "target_active_power at exactly t+5 minutes",
    }
    content = json.dumps(report, ensure_ascii=False, indent=2)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content + "\n", encoding="utf-8")
    print(content)


if __name__ == "__main__":
    main()
