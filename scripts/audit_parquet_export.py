"""Verify exported device and weather Parquet partitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds


DATASETS = {
    "device": ("event_time", ["event_time", "plant_id", "device_no"]),
    "weather_15min": ("time", ["time", "plant_id"]),
}


def inspect_dataset(root: Path, name: str) -> tuple[dict[str, object], pd.DataFrame]:
    time_field, key = DATASETS[name]
    files = sorted(root.glob("date=*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files under {root}")
    columns = [*key, "date"]
    if name == "weather_15min":
        columns.extend(["sensor_ghi", "forecast_ghi"])
    dataset = ds.dataset(root, format="parquet", partitioning="hive")
    frame = dataset.to_table(columns=columns).to_pandas()
    timestamps = pd.to_datetime(frame[time_field], errors="coerce", utc=True)
    local_dates = timestamps.dt.tz_convert("Asia/Shanghai").dt.date.astype(str)
    result: dict[str, object] = {
        "dataset": name,
        "files": len(files),
        "rows": len(frame),
        "size_mb": round(sum(path.stat().st_size for path in files) / 1024 / 1024, 3),
        "minimum_utc": timestamps.min().isoformat(),
        "maximum_utc": timestamps.max().isoformat(),
        "invalid_timestamps": int(timestamps.isna().sum()),
        "partition_date_mismatches": int((local_dates != frame["date"].astype(str)).sum()),
        "duplicate_keys": int(frame.duplicated(key).sum()),
        "plants": int(frame["plant_id"].nunique()),
        "rows_by_date": {
            str(day): int(count) for day, count in local_dates.value_counts().sort_index().items()
        },
    }
    if "device_no" in frame:
        result["devices"] = int(frame["device_no"].nunique())
    if name == "weather_15min":
        result["sensor_ghi_null_percent"] = round(float(frame["sensor_ghi"].isna().mean() * 100), 4)
        result["forecast_ghi_null_percent"] = round(
            float(frame["forecast_ghi"].isna().mean() * 100), 4
        )
    return result, frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/raw")
    parser.add_argument("--output", help="Optional JSON report path")
    args = parser.parse_args()
    root = Path(args.root)
    inspected = [inspect_dataset(root / name, name) for name in DATASETS]
    results = [item[0] for item in inspected]
    device, weather = inspected[0][1], inspected[1][1]
    target_plants = set(device["plant_id"].dropna().unique())
    matched = weather[weather["plant_id"].isin(target_plants)]
    alignment = {
        "target_plants": len(target_plants),
        "target_plants_present_in_weather": len(target_plants & set(weather["plant_id"].unique())),
        "matched_weather_rows": len(matched),
        "matched_sensor_ghi_null_percent": round(
            float(matched["sensor_ghi"].isna().mean() * 100), 4
        ) if len(matched) else None,
        "matched_forecast_ghi_null_percent": round(
            float(matched["forecast_ghi"].isna().mean() * 100), 4
        ) if len(matched) else None,
    }
    content = json.dumps({"datasets": results, "alignment": alignment}, ensure_ascii=False, indent=2)
    print(content)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
