"""Profile fields needed to define candidate-normal training rows."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds


FIELDS = [
    "event_time",
    "plant_id",
    "device_no",
    "active_power",
    "dc_power",
    "total_power",
    "rated_power",
    "device_temperature",
    "status_code",
    "main_string_count",
    "valid_current_string_count",
    "string_overall_status",
]


def main() -> None:
    dataset = ds.dataset("data/raw/device", format="parquet", partitioning="hive")
    frame = pd.concat(
        [pd.read_parquet(path, columns=FIELDS) for path in dataset.files],
        ignore_index=True,
    )
    frame["event_time"] = pd.to_datetime(frame["event_time"], errors="coerce", utc=True)
    numeric = [field for field in FIELDS if field not in {"event_time", "device_no"}]
    summary = frame[numeric].describe(percentiles=[0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999])

    ordered = frame.sort_values(["device_no", "event_time"])
    intervals = ordered.groupby("device_no", observed=True)["event_time"].diff().dt.total_seconds()
    rated = pd.to_numeric(frame["rated_power"], errors="coerce")
    active = pd.to_numeric(frame["active_power"], errors="coerce")
    positive_rated = rated > 0
    ratio = active[positive_rated] / rated[positive_rated]
    local_hour = frame["event_time"].dt.tz_convert("Asia/Shanghai").dt.hour

    def grouped_power(field: str) -> dict[str, dict[str, float | int]]:
        result = {}
        for value, group in frame.groupby(field, dropna=False):
            power = pd.to_numeric(group["active_power"], errors="coerce")
            result[str(value)] = {
                "rows": len(group),
                "positive_power_percent": round(float((power > 0).mean() * 100), 4),
                "median_active_power": round(float(power.median()), 6),
                "mean_active_power": round(float(power.mean()), 6),
            }
        return result

    report = {
        "rows": len(frame),
        "devices": int(frame["device_no"].nunique()),
        "plants": int(frame["plant_id"].nunique()),
        "null_percent": {
            column: round(float(frame[column].isna().mean() * 100), 4) for column in FIELDS
        },
        "numeric_summary": json.loads(summary.to_json()),
        "status_code_counts": {
            str(key): int(value) for key, value in frame["status_code"].value_counts(dropna=False).items()
        },
        "string_overall_status_counts": {
            str(key): int(value)
            for key, value in frame["string_overall_status"].value_counts(dropna=False).items()
        },
        "power_by_status_code": grouped_power("status_code"),
        "power_by_string_overall_status": grouped_power("string_overall_status"),
        "valid_equals_main_percent": round(
            float((frame["valid_current_string_count"] == frame["main_string_count"]).mean() * 100),
            4,
        ),
        "interval_seconds_counts": {
            str(key): int(value) for key, value in intervals.value_counts().head(20).items()
        },
        "non_five_minute_intervals": int(((intervals.notna()) & (intervals != 300)).sum()),
        "active_power": {
            "negative": int((active < 0).sum()),
            "zero": int((active == 0).sum()),
            "positive": int((active > 0).sum()),
            "non_finite": int((~np.isfinite(active)).sum()),
        },
        "rated_power_positive_percent": round(float(positive_rated.mean() * 100), 4),
        "active_to_rated_quantiles": {
            str(key): float(value)
            for key, value in ratio.quantile([0, 0.01, 0.5, 0.99, 0.999, 1]).items()
        },
        "positive_power_by_local_hour": {
            str(hour): int(value)
            for hour, value in frame[active > 0].groupby(local_hour[active > 0]).size().items()
        },
    }
    output = Path("artifacts/reports/training_field_profile.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
