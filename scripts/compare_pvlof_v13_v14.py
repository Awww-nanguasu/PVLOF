"""Compare PVLOF v1.3 with the parallel v1.4 dual-effect-gated variant."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


KEYS = ["plant_id", "device_no", "event_time"]
LABELS = ["PVLOF_V1_3_EFFECT_GATE", "PVLOF_V1_4_DUAL_GATE"]


def _numbers(values: pd.Series) -> str:
    numbers: set[int] = set()
    for value in values.dropna():
        for item in str(value).split(","):
            if item.strip():
                numbers.add(int(float(item)))
    return ",".join(f"{number:02d}" for number in sorted(numbers))


def _alerts(path: str, column: str, label: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    missing = sorted({*KEYS, "string_no", column} - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    frame["plant_id"] = frame["plant_id"].astype(str)
    frame["device_no"] = frame["device_no"].astype(str)
    frame["event_time"] = pd.to_datetime(frame["event_time"], utc=True)
    selected = frame[frame[column].fillna(False).astype(bool)]
    return (
        selected.groupby(KEYS, observed=True)["string_no"]
        .agg(_numbers)
        .rename(label)
        .reset_index()
    )


def _comparison_case(frame: pd.DataFrame) -> pd.Series:
    left = frame[LABELS[0]].ne("")
    right = frame[LABELS[1]].ne("")
    result = pd.Series("both_same", index=frame.index, dtype="string")
    result.loc[left & ~right] = "v1_3_only"
    result.loc[~left & right] = "v1_4_only"
    result.loc[left & right & frame[LABELS[0]].ne(frame[LABELS[1]])] = "both_different"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-3", required=True)
    parser.add_argument("--v1-4", required=True)
    parser.add_argument("--alert-column", default="pvlof_v12_combined_alert")
    parser.add_argument("--timezone", default="Asia/Shanghai")
    parser.add_argument("--interval-minutes", type=int, default=5)
    parser.add_argument("--output-directory", required=True)
    args = parser.parse_args()

    left = _alerts(args.v1_3, args.alert_column, LABELS[0])
    right = _alerts(args.v1_4, args.alert_column, LABELS[1])
    points = left.merge(right, on=KEYS, how="outer", validate="one_to_one")
    points[LABELS] = points[LABELS].astype("string").fillna("")
    points["comparison_case"] = _comparison_case(points)
    points = points.sort_values(KEYS).reset_index(drop=True)

    expected = pd.Timedelta(minutes=args.interval_minutes)
    points["_event"] = (
        points["plant_id"].ne(points["plant_id"].shift())
        | points["device_no"].ne(points["device_no"].shift())
        | points["event_time"].sub(points["event_time"].shift()).ne(expected)
    ).cumsum()
    event_ids = {
        value: f"pvlof-v13-v14-{index:06d}"
        for index, value in enumerate(points["_event"].unique(), 1)
    }
    points["event_id"] = points["_event"].map(event_ids)
    events = (
        points.groupby(["event_id", "plant_id", "device_no", "_event"], observed=True)
        .agg(
            raise_time=("event_time", "min"),
            end_time=("event_time", "max"),
            PVLOF_V1_3_EFFECT_GATE=(LABELS[0], _numbers),
            PVLOF_V1_4_DUAL_GATE=(LABELS[1], _numbers),
            alert_time_points=("event_time", "size"),
        )
        .reset_index()
        .drop(columns="_event")
    )
    events["comparison_case"] = _comparison_case(events)
    events["raise_time_local"] = events["raise_time"].dt.tz_convert(args.timezone).dt.strftime("%Y-%m-%d %H:%M:%S")
    events["end_time_local"] = events["end_time"].dt.tz_convert(args.timezone).dt.strftime("%Y-%m-%d %H:%M:%S")
    events["duration_minutes"] = (
        (events["end_time"] - events["raise_time"]).dt.total_seconds() / 60
        + args.interval_minutes
    ).astype(int)
    events["manual_is_low_current"] = ""
    events["manual_string_no"] = ""
    events["review_note"] = ""
    events.insert(0, "row", range(1, len(events) + 1))

    points["event_time_local"] = points["event_time"].dt.tz_convert(args.timezone).dt.strftime("%Y-%m-%d %H:%M:%S")
    points.insert(0, "row", range(1, len(points) + 1))

    output = Path(args.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    event_path = output / "pvlof_v13_v14_dual_gate_events.csv"
    point_path = output / "pvlof_v13_v14_dual_gate_points.csv"
    events[[
        "row", "event_id", "plant_id", "device_no", "raise_time_local",
        "end_time_local", *LABELS, "comparison_case", "alert_time_points",
        "duration_minutes", "manual_is_low_current", "manual_string_no", "review_note",
    ]].to_csv(event_path, index=False, encoding="utf-8-sig")
    points[[
        "row", "event_id", "plant_id", "device_no", "event_time_local",
        *LABELS, "comparison_case",
    ]].to_csv(point_path, index=False, encoding="utf-8-sig")

    report = {
        "v1_3": args.v1_3,
        "v1_4": args.v1_4,
        "alert_column": args.alert_column,
        "events": len(events),
        "points": len(points),
        "event_cases": events["comparison_case"].value_counts().to_dict(),
        "point_cases": points["comparison_case"].value_counts().to_dict(),
        "outputs": {"events": str(event_path), "points": str(point_path)},
    }
    report_path = output / "summary.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
