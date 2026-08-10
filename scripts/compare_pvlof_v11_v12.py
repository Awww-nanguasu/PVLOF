"""Compare PVLOF v1.1 and v1.2 strict alerts by device-time and event."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


KEYS = ["plant_id", "device_no", "event_time"]


def _format_strings(values: pd.Series) -> str:
    numbers = {
        int(float(value))
        for value in values
        if pd.notna(value) and str(value).strip()
    }
    return ",".join(f"{number:02d}" for number in sorted(numbers))


def _alerts(path: str, alert_column: str, output_column: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = set(KEYS + ["string_no", alert_column])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    frame["plant_id"] = frame["plant_id"].astype(str)
    frame["device_no"] = frame["device_no"].astype(str)
    frame["event_time"] = pd.to_datetime(frame["event_time"], errors="raise", utc=True)
    selected = frame[frame[alert_column].fillna(False).astype(bool)]
    return (
        selected.groupby(KEYS, observed=True)["string_no"]
        .agg(_format_strings)
        .rename(output_column)
        .reset_index()
    )


def _case(v11: pd.Series, v12: pd.Series) -> pd.Series:
    left = v11.ne("")
    right = v12.ne("")
    result = pd.Series("both_same", index=v11.index, dtype="object")
    result.loc[left & ~right] = "v1_1_only"
    result.loc[~left & right] = "v1_2_only"
    result.loc[left & right & v11.ne(v12)] = "both_different"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-1", required=True)
    parser.add_argument("--v1-2", required=True)
    parser.add_argument("--alert-column", default="pvlof_v2_hier_strict_alert")
    parser.add_argument("--timezone", default="Asia/Shanghai")
    parser.add_argument("--interval-minutes", type=int, default=5)
    parser.add_argument("--output-directory", required=True)
    args = parser.parse_args()

    v11 = _alerts(args.v1_1, args.alert_column, "PVLOF_V1_1")
    v12 = _alerts(args.v1_2, args.alert_column, "PVLOF_V1_2")
    points = v11.merge(v12, on=KEYS, how="outer", validate="one_to_one")
    points[["PVLOF_V1_1", "PVLOF_V1_2"]] = points[
        ["PVLOF_V1_1", "PVLOF_V1_2"]
    ].fillna("")
    points["comparison_case"] = _case(points["PVLOF_V1_1"], points["PVLOF_V1_2"])
    points = points.sort_values(KEYS).reset_index(drop=True)
    expected = pd.Timedelta(minutes=args.interval_minutes)
    points["_event"] = (
        points["plant_id"].ne(points["plant_id"].shift())
        | points["device_no"].ne(points["device_no"].shift())
        | points["event_time"].sub(points["event_time"].shift()).ne(expected)
    ).cumsum()
    points["event_id"] = points["_event"].map(
        {number: f"pvlof-v11-v12-{index:06d}" for index, number in enumerate(points["_event"].unique(), 1)}
    )
    events = (
        points.groupby(["event_id", "plant_id", "device_no", "_event"], observed=True, sort=False)
        .agg(
            raise_time=("event_time", "min"),
            end_time=("event_time", "max"),
            PVLOF_V1_1=("PVLOF_V1_1", lambda values: _format_strings(pd.Series(",".join(values).split(",")))),
            PVLOF_V1_2=("PVLOF_V1_2", lambda values: _format_strings(pd.Series(",".join(values).split(",")))),
            alert_time_points=("event_time", "size"),
        )
        .reset_index(drop=False)
        .drop(columns="_event")
    )
    events["comparison_case"] = _case(events["PVLOF_V1_1"], events["PVLOF_V1_2"])
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
    event_columns = [
        "row", "event_id", "plant_id", "device_no", "raise_time_local", "end_time_local",
        "PVLOF_V1_1", "PVLOF_V1_2", "comparison_case", "alert_time_points",
        "duration_minutes", "manual_is_low_current", "manual_string_no", "review_note",
    ]
    point_columns = [
        "row", "event_id", "plant_id", "device_no", "event_time_local",
        "PVLOF_V1_1", "PVLOF_V1_2", "comparison_case",
    ]
    output = Path(args.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    event_path = output / "pvlof_v11_v12_alarm_events.csv"
    point_path = output / "pvlof_v11_v12_alarm_points.csv"
    events[event_columns].to_csv(event_path, index=False, encoding="utf-8-sig")
    points[point_columns].to_csv(point_path, index=False, encoding="utf-8-sig")
    report = {
        "v1_1": args.v1_1,
        "v1_2": args.v1_2,
        "events": len(events),
        "device_time_points": len(points),
        "event_comparison_cases": events["comparison_case"].value_counts().to_dict(),
        "point_comparison_cases": points["comparison_case"].value_counts().to_dict(),
        "outputs": {"events": str(event_path), "points": str(point_path)},
    }
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
