"""Export device-level PVLOF events and their exact alert timestamps for ES review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _format_strings(values: pd.Series) -> str:
    numbers = sorted({int(value) for value in values if pd.notna(value)})
    return ",".join(f"{number:02d}" for number in numbers)


def build_review_tables(
    frame: pd.DataFrame,
    *,
    alert_column: str = "pvlof_v2_hier_strict_alert",
    interval_minutes: int = 5,
    timezone: str = "Asia/Shanghai",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collapse strict string alerts into device events and device-time points."""
    required = {"plant_id", "device_no", "event_time", "string_no", alert_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"PVLOF points are missing columns: {missing}")

    alerts = frame.loc[frame[alert_column].fillna(False).astype(bool)].copy()
    event_columns = [
        "row",
        "event_id",
        "plant_id",
        "device_no",
        "raise_time_local",
        "end_time_local",
        "string_no",
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
        "string_no",
    ]
    if alerts.empty:
        return pd.DataFrame(columns=event_columns), pd.DataFrame(columns=point_columns)

    alerts["event_time"] = pd.to_datetime(alerts["event_time"], errors="raise", utc=True)
    point_alerts = (
        alerts.groupby(["plant_id", "device_no", "event_time"], observed=True)[
            "string_no"
        ]
        .agg(_format_strings)
        .rename("string_no")
        .reset_index()
        .sort_values(["plant_id", "device_no", "event_time"])
        .reset_index(drop=True)
    )
    expected = pd.Timedelta(minutes=interval_minutes)
    point_alerts["_new_event"] = (
        point_alerts["plant_id"].ne(point_alerts["plant_id"].shift())
        | point_alerts["device_no"].ne(point_alerts["device_no"].shift())
        | point_alerts["event_time"].sub(point_alerts["event_time"].shift()).ne(expected)
    )
    point_alerts["_event_number"] = point_alerts["_new_event"].cumsum()

    events = (
        point_alerts.groupby(
            ["plant_id", "device_no", "_event_number"], observed=True
        )
        .agg(
            raise_time_utc=("event_time", "min"),
            end_time_utc=("event_time", "max"),
            alert_time_points=("event_time", "size"),
        )
        .reset_index()
        .drop(columns="_event_number")
    )
    events.insert(0, "event_id", [f"pvlof-strict-{i:06d}" for i in range(1, len(events) + 1)])
    event_ids = events[["plant_id", "device_no", "raise_time_utc", "end_time_utc", "event_id"]]
    point_alerts = point_alerts.merge(
        event_ids,
        on=["plant_id", "device_no"],
        how="left",
        validate="many_to_many",
    )
    point_alerts = point_alerts[
        point_alerts["event_time"].between(
            point_alerts["raise_time_utc"], point_alerts["end_time_utc"]
        )
    ].copy()

    all_strings = (
        point_alerts.groupby("event_id", observed=True)["string_no"]
        .agg(lambda values: _format_strings(pd.Series(",".join(values).split(","))))
        .rename("string_no")
    )
    events = events.merge(all_strings, on="event_id", how="left", validate="one_to_one")
    events["duration_minutes"] = (
        (events["end_time_utc"] - events["raise_time_utc"]).dt.total_seconds() / 60
        + interval_minutes
    ).astype(int)
    events["raise_time_local"] = events["raise_time_utc"].dt.tz_convert(
        timezone
    ).dt.strftime("%Y-%m-%d %H:%M:%S")
    events["end_time_local"] = events["end_time_utc"].dt.tz_convert(
        timezone
    ).dt.strftime("%Y-%m-%d %H:%M:%S")
    events["manual_is_low_current"] = ""
    events["manual_string_no"] = ""
    events["review_note"] = ""
    events.insert(0, "row", range(1, len(events) + 1))

    point_alerts["event_time_local"] = point_alerts["event_time"].dt.tz_convert(
        timezone
    ).dt.strftime("%Y-%m-%d %H:%M:%S")
    point_alerts = point_alerts.sort_values(["event_time", "device_no"]).reset_index(drop=True)
    point_alerts.insert(0, "row", range(1, len(point_alerts) + 1))
    return events[event_columns], point_alerts[point_columns]


def export_review(
    input_path: str | Path,
    output_directory: str | Path,
    *,
    alert_column: str = "pvlof_v2_hier_strict_alert",
    interval_minutes: int = 5,
    timezone: str = "Asia/Shanghai",
) -> dict[str, Any]:
    frame = pd.read_parquet(input_path)
    events, points = build_review_tables(
        frame,
        alert_column=alert_column,
        interval_minutes=interval_minutes,
        timezone=timezone,
    )
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    event_path = output / "pvlof_strict_alarm_events.csv"
    point_path = output / "pvlof_strict_alarm_points.csv"
    events.to_csv(event_path, index=False, encoding="utf-8-sig")
    points.to_csv(point_path, index=False, encoding="utf-8-sig")
    report = {
        "input": str(input_path),
        "alert_column": alert_column,
        "events": len(events),
        "device_time_points": len(points),
        "devices": int(events["device_no"].nunique()) if len(events) else 0,
        "outputs": {"events": str(event_path), "points": str(point_path)},
    }
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument(
        "--alert-column", default="pvlof_v2_hier_strict_alert"
    )
    parser.add_argument("--interval-minutes", type=int, default=5)
    parser.add_argument("--timezone", default="Asia/Shanghai")
    args = parser.parse_args()
    report = export_review(
        args.input,
        args.output_directory,
        alert_column=args.alert_column,
        interval_minutes=args.interval_minutes,
        timezone=args.timezone,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
