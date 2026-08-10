"""Export point/event comparison for v1.2 strict and persistent group alerts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


KEYS = ["plant_id", "device_no", "event_time"]
ALERTS = {
    "PVLOF_V1_2": "pvlof_v2_hier_strict_alert",
    "PVLOF_V1_2_PERSISTENT_GROUP": "persistent_collective_member_alert",
    "PVLOF_V1_2_COMBINED": "pvlof_v12_combined_alert",
}


def _strings(values: pd.Series) -> str:
    numbers: set[int] = set()
    for value in values:
        if pd.isna(value) or not str(value).strip():
            continue
        for item in str(value).split(","):
            if item.strip():
                numbers.add(int(float(item)))
    return ",".join(f"{number:02d}" for number in sorted(numbers))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--timezone", default="Asia/Shanghai")
    parser.add_argument("--interval-minutes", type=int, default=5)
    args = parser.parse_args()

    frame = pd.read_parquet(args.input)
    required = set(KEYS + ["string_no", *ALERTS.values()])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{args.input} is missing columns: {missing}")
    frame["plant_id"] = frame["plant_id"].astype(str)
    frame["device_no"] = frame["device_no"].astype(str)
    frame["event_time"] = pd.to_datetime(frame["event_time"], errors="raise", utc=True)
    frame["string_no"] = pd.to_numeric(frame["string_no"], errors="coerce")

    points = frame[KEYS].drop_duplicates().copy()
    for label, column in ALERTS.items():
        selected = frame[frame[column].fillna(False).astype(bool)]
        grouped = (
            selected.groupby(KEYS, observed=True)["string_no"]
            .agg(_strings)
            .rename(label)
            .reset_index()
        )
        points = points.merge(grouped, on=KEYS, how="left", validate="one_to_one")
        # An entirely empty alert selection can retain pandas' nullable Int64
        # dtype after the merge. Convert explicitly before filling with the
        # human-readable empty marker.
        points[label] = points[label].astype("string").fillna("")
    points = points.sort_values(KEYS).reset_index(drop=True)
    points["event_time_local"] = points["event_time"].dt.tz_convert(args.timezone).dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    points.insert(0, "row", range(1, len(points) + 1))

    active = points[[*ALERTS]].apply(
        lambda row: any(pd.notna(value) and str(value).strip() for value in row),
        axis=1,
    )
    event_points = points[active].copy()
    expected = pd.Timedelta(minutes=args.interval_minutes)
    event_points["_event"] = (
        event_points["plant_id"].ne(event_points["plant_id"].shift())
        | event_points["device_no"].ne(event_points["device_no"].shift())
        | event_points["event_time"].sub(event_points["event_time"].shift()).ne(expected)
    ).cumsum()
    event_points["event_id"] = event_points["_event"].map(
        {
            number: f"pvlof-v12-persistent-compare-{index:06d}"
            for index, number in enumerate(event_points["_event"].unique(), 1)
        }
    )
    events = (
        event_points.groupby(["event_id", "plant_id", "device_no", "_event"], observed=True)
        .agg(
            raise_time=("event_time", "min"),
            end_time=("event_time", "max"),
            PVLOF_V1_2=("PVLOF_V1_2", _strings),
            PVLOF_V1_2_PERSISTENT_GROUP=("PVLOF_V1_2_PERSISTENT_GROUP", _strings),
            PVLOF_V1_2_COMBINED=("PVLOF_V1_2_COMBINED", _strings),
            alert_time_points=("event_time", "size"),
        )
        .reset_index()
        .drop(columns="_event")
    )
    events["raise_time_local"] = events["raise_time"].dt.tz_convert(args.timezone).dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    events["end_time_local"] = events["end_time"].dt.tz_convert(args.timezone).dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    events["duration_minutes"] = (
        (events["end_time"] - events["raise_time"]).dt.total_seconds() / 60
        + args.interval_minutes
    ).astype(int)
    events.insert(0, "row", range(1, len(events) + 1))

    output = Path(args.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    point_path = output / "pvlof_v12_persistent_points.csv"
    event_path = output / "pvlof_v12_persistent_events.csv"
    summary_path = output / "summary.json"
    points[
        ["row", "plant_id", "device_no", "event_time_local", *ALERTS]
    ].to_csv(point_path, index=False, encoding="utf-8-sig")
    events[
        [
            "row", "event_id", "plant_id", "device_no", "raise_time_local", "end_time_local",
            *ALERTS, "alert_time_points", "duration_minutes",
        ]
    ].to_csv(event_path, index=False, encoding="utf-8-sig")
    report = {
        "input": args.input,
        "rows": len(points),
        "event_rows": len(events),
        "point_alert_counts": {
            label: int(points[label].ne("").sum()) for label in ALERTS
        },
        "outputs": {"points": str(point_path), "events": str(event_path)},
    }
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
