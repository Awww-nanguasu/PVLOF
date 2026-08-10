"""Compare two PVLOF strict outputs at device-event and device-time levels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _format_numbers(values: Any) -> str:
    numbers: set[int] = set()
    for value in values:
        if pd.isna(value) or str(value).strip() == "":
            continue
        for item in str(value).split(","):
            item = item.strip()
            if item:
                numbers.add(int(float(item)))
    return ",".join(f"{number:02d}" for number in sorted(numbers))


def _read_alert_points(path: str | Path, alert_column: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {"plant_id", "device_no", "event_time", "string_no", alert_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    frame["plant_id"] = frame["plant_id"].astype(str)
    frame["device_no"] = frame["device_no"].astype(str)
    frame["event_time"] = pd.to_datetime(frame["event_time"], errors="raise", utc=True)
    alerts = frame.loc[frame[alert_column].fillna(False).astype(bool)].copy()
    if alerts.empty:
        return pd.DataFrame(
            columns=["plant_id", "device_no", "event_time", "string_no"]
        )
    return (
        alerts.groupby(["plant_id", "device_no", "event_time"], observed=True)[
            "string_no"
        ]
        .agg(lambda values: _format_numbers(values))
        .rename("string_no")
        .reset_index()
    )


def _comparison_case(v1: pd.Series, v11: pd.Series) -> pd.Series:
    has_v1 = v1.ne("")
    has_v11 = v11.ne("")
    result = pd.Series("both_same", index=v1.index, dtype="object")
    result.loc[has_v1 & ~has_v11] = "v1_only"
    result.loc[~has_v1 & has_v11] = "v1_1_only"
    result.loc[has_v1 & has_v11 & v1.ne(v11)] = "both_different"
    return result


def build_comparison_tables(
    v1: pd.DataFrame,
    v11: pd.DataFrame,
    *,
    interval_minutes: int = 5,
    timezone: str = "Asia/Shanghai",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["plant_id", "device_no", "event_time"]
    points = v1.rename(columns={"string_no": "PVLOF_V1"}).merge(
        v11.rename(columns={"string_no": "PVLOF_V1_1"}),
        on=keys,
        how="outer",
        validate="one_to_one",
    )
    if points.empty:
        return pd.DataFrame(), pd.DataFrame()
    points["PVLOF_V1"] = points["PVLOF_V1"].fillna("")
    points["PVLOF_V1_1"] = points["PVLOF_V1_1"].fillna("")
    points["comparison_case"] = _comparison_case(
        points["PVLOF_V1"], points["PVLOF_V1_1"]
    )
    points = points.sort_values(keys).reset_index(drop=True)
    expected = pd.Timedelta(minutes=interval_minutes)
    points["_new_event"] = (
        points["plant_id"].ne(points["plant_id"].shift())
        | points["device_no"].ne(points["device_no"].shift())
        | points["event_time"].sub(points["event_time"].shift()).ne(expected)
    )
    points["_event_number"] = points["_new_event"].cumsum()
    event_numbers = sorted(points["_event_number"].unique())
    event_ids = {
        number: f"pvlof-compare-{index:06d}"
        for index, number in enumerate(event_numbers, start=1)
    }
    points["event_id"] = points["_event_number"].map(event_ids)

    events = (
        points.groupby(
            ["event_id", "plant_id", "device_no", "_event_number"],
            observed=True,
            sort=False,
        )
        .agg(
            raise_time_utc=("event_time", "min"),
            end_time_utc=("event_time", "max"),
            PVLOF_V1=("PVLOF_V1", _format_numbers),
            PVLOF_V1_1=("PVLOF_V1_1", _format_numbers),
            alert_time_points=("event_time", "size"),
        )
        .reset_index()
        .drop(columns="_event_number")
    )
    events["comparison_case"] = _comparison_case(
        events["PVLOF_V1"], events["PVLOF_V1_1"]
    )
    events["raise_time_local"] = events["raise_time_utc"].dt.tz_convert(
        timezone
    ).dt.strftime("%Y-%m-%d %H:%M:%S")
    events["end_time_local"] = events["end_time_utc"].dt.tz_convert(
        timezone
    ).dt.strftime("%Y-%m-%d %H:%M:%S")
    events["duration_minutes"] = (
        (events["end_time_utc"] - events["raise_time_utc"]).dt.total_seconds() / 60
        + interval_minutes
    ).astype(int)
    events["manual_is_low_current"] = ""
    events["manual_string_no"] = ""
    events["review_note"] = ""
    events.insert(0, "row", range(1, len(events) + 1))

    points["event_time_local"] = points["event_time"].dt.tz_convert(
        timezone
    ).dt.strftime("%Y-%m-%d %H:%M:%S")
    points.insert(0, "row", range(1, len(points) + 1))
    event_columns = [
        "row",
        "event_id",
        "plant_id",
        "device_no",
        "raise_time_local",
        "end_time_local",
        "PVLOF_V1",
        "PVLOF_V1_1",
        "comparison_case",
        "alert_time_points",
        "duration_minutes",
        "manual_is_low_current",
        "manual_string_no",
        "review_note",
    ]
    point_columns = [
        "row",
        "event_id",
        "plant_id",
        "device_no",
        "event_time_local",
        "PVLOF_V1",
        "PVLOF_V1_1",
        "comparison_case",
    ]
    return events[event_columns], points[point_columns]


def compare_versions(
    v1_path: str | Path,
    v11_path: str | Path,
    output_directory: str | Path,
    *,
    alert_column: str = "pvlof_v2_hier_strict_alert",
    interval_minutes: int = 5,
    timezone: str = "Asia/Shanghai",
) -> dict[str, Any]:
    v1 = _read_alert_points(v1_path, alert_column)
    v11 = _read_alert_points(v11_path, alert_column)
    events, points = build_comparison_tables(
        v1,
        v11,
        interval_minutes=interval_minutes,
        timezone=timezone,
    )
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    event_path = output / "pvlof_v1_v11_alarm_events.csv"
    point_path = output / "pvlof_v1_v11_alarm_points.csv"
    events.to_csv(event_path, index=False, encoding="utf-8-sig")
    points.to_csv(point_path, index=False, encoding="utf-8-sig")
    report = {
        "v1": str(v1_path),
        "v1_1": str(v11_path),
        "alert_column": alert_column,
        "events": len(events),
        "device_time_points": len(points),
        "event_comparison_cases": events["comparison_case"].value_counts().to_dict()
        if len(events)
        else {},
        "point_comparison_cases": points["comparison_case"].value_counts().to_dict()
        if len(points)
        else {},
        "outputs": {"events": str(event_path), "points": str(point_path)},
    }
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1", required=True)
    parser.add_argument("--v1-1", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument(
        "--alert-column", default="pvlof_v2_hier_strict_alert"
    )
    parser.add_argument("--interval-minutes", type=int, default=5)
    parser.add_argument("--timezone", default="Asia/Shanghai")
    args = parser.parse_args()
    compare_versions(
        args.v1,
        args.v1_1,
        args.output_directory,
        alert_column=args.alert_column,
        interval_minutes=args.interval_minutes,
        timezone=args.timezone,
    )


if __name__ == "__main__":
    main()
