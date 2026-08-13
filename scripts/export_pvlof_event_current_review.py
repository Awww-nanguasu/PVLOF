"""Export all string currents for each PVLOF-predicted event."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _collapse_alert_events(
    points: pd.DataFrame,
    *,
    alert_column: str,
    interval_minutes: int,
) -> pd.DataFrame:
    alerts = points[points[alert_column].astype(bool)].copy()
    if alerts.empty:
        return pd.DataFrame(
            columns=[
                "event_id",
                "station_name",
                "device_no",
                "alert_string_no",
                "start_time",
                "end_time",
                "alert_points",
            ]
        )
    group_columns = ["station_name", "device_no", "string_no"]
    alerts = alerts.sort_values(group_columns + ["event_time"])
    expected = pd.Timedelta(minutes=interval_minutes)
    alerts["_new_event"] = (
        alerts[group_columns].ne(alerts[group_columns].shift()).any(axis=1)
        | alerts["event_time"].sub(alerts["event_time"].shift()).ne(expected)
    )
    alerts["_event_number"] = alerts["_new_event"].cumsum()
    events = (
        alerts.groupby(group_columns + ["_event_number"], observed=True)
        .agg(
            start_time=("event_time", "min"),
            end_time=("event_time", "max"),
            alert_points=("event_time", "size"),
            maximum_alert_score=("pvlof_score", "max"),
        )
        .reset_index()
        .drop(columns=["_event_number"])
        .rename(columns={"string_no": "alert_string_no"})
    )
    events.insert(0, "event_id", [f"pvlof-{index:06d}" for index in range(1, len(events) + 1)])
    return events


def export_review(
    report_directory: str | Path,
    *,
    output_directory: str | Path | None = None,
    alert_column: str = "pvlof_alert",
    interval_minutes: int = 5,
    timezone: str = "Asia/Shanghai",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write sorted point rows and one summary row for each predicted event."""

    report = Path(report_directory)
    output = Path(output_directory) if output_directory else report
    output.mkdir(parents=True, exist_ok=True)
    points = pd.read_parquet(report / "aligned_pvlof_points.parquet")
    required = {"event_time", "station_name", "device_no", "string_no", alert_column}
    missing = sorted(required - set(points.columns))
    if missing:
        raise ValueError(f"Aligned PVLOF output is missing columns: {missing}")
    points["event_time"] = pd.to_datetime(points["event_time"], utc=True)
    events = _collapse_alert_events(
        points,
        alert_column=alert_column,
        interval_minutes=interval_minutes,
    )

    review_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for event in events.itertuples(index=False):
        window = points[
            (points["station_name"] == event.station_name)
            & (points["device_no"] == event.device_no)
            & (points["event_time"] >= event.start_time)
            & (points["event_time"] <= event.end_time)
        ].copy()
        if window.empty:
            continue
        window["event_id"] = event.event_id
        window["alert_column"] = alert_column
        window["alert_string_no"] = event.alert_string_no
        window["event_start_time"] = event.start_time
        window["event_end_time"] = event.end_time
        window["event_start_local"] = event.start_time.tz_convert(timezone).isoformat()
        window["event_end_local"] = event.end_time.tz_convert(timezone).isoformat()
        window["event_time_local"] = window["event_time"].dt.tz_convert(timezone).dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        window["current_rank_ascending"] = (
            window.groupby("event_time", observed=True)["string_current"]
            .rank(method="first", ascending=True)
            .astype("Int64")
        )
        window["strings_at_timestamp"] = window.groupby(
            "event_time", observed=True
        )["string_no"].transform("count")
        window["is_lowest_current"] = window["current_rank_ascending"].eq(1)
        review_rows.append(window)

        lowest = window.loc[window["string_current"].idxmin()]
        summary_rows.append(
            {
                "event_id": event.event_id,
                "alert_column": alert_column,
                "station_name": event.station_name,
                "device_no": event.device_no,
                "alert_string_no": int(event.alert_string_no),
                "event_start_local": event.start_time.tz_convert(timezone).isoformat(),
                "event_end_local": event.end_time.tz_convert(timezone).isoformat(),
                "alert_points": int(event.alert_points),
                "timestamps": int(window["event_time"].nunique()),
                "rows": int(len(window)),
                "lowest_string_no": int(lowest["string_no"]),
                "minimum_current": float(lowest["string_current"]),
                "minimum_current_ratio": float(lowest["string_current_ratio"])
                if pd.notna(lowest["string_current_ratio"])
                else None,
                "maximum_alert_score": float(event.maximum_alert_score)
                if pd.notna(event.maximum_alert_score)
                else None,
            }
        )

    columns = [
        "event_id",
        "alert_column",
        "station_name",
        "device_no",
        "alert_string_no",
        "event_start_time",
        "event_end_time",
        "event_start_local",
        "event_end_local",
        "event_time",
        "event_time_local",
        "string_no",
        "string_current",
        "current_rank_ascending",
        "strings_at_timestamp",
        "is_lowest_current",
        "string_current_ratio",
        "pvlof_score",
        "pvlof_alert",
        "combined_alert",
        "zero_current_alert",
        "active_power",
        "rated_power",
        "active_power_ratio",
    ]
    sorted_points = (
        pd.concat(review_rows, ignore_index=True)[columns]
        if review_rows
        else pd.DataFrame(columns=columns)
    )
    sorted_points = sorted_points.sort_values(
        ["event_id", "event_time", "string_current", "string_no"]
    )
    summary = pd.DataFrame(summary_rows)
    if summary_rows:
        summary = summary.sort_values(["event_start_local", "device_no"])
    else:
        summary = pd.DataFrame(
            columns=[
                "event_id",
                "alert_column",
                "station_name",
                "device_no",
                "alert_string_no",
                "event_start_local",
                "event_end_local",
                "alert_points",
                "timestamps",
                "rows",
                "lowest_string_no",
                "minimum_current",
                "minimum_current_ratio",
                "maximum_alert_score",
            ]
        )
    sorted_points.to_csv(
        output / "pvlof_event_sorted_points.csv", index=False, encoding="utf-8-sig"
    )
    summary.to_csv(
        output / "pvlof_event_summary.csv", index=False, encoding="utf-8-sig"
    )
    return sorted_points, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-directory", required=True)
    parser.add_argument("--output-directory", default=None)
    parser.add_argument("--alert-column", default="pvlof_alert")
    parser.add_argument("--interval-minutes", type=int, default=5)
    parser.add_argument("--timezone", default="Asia/Shanghai")
    args = parser.parse_args()
    points, summary = export_review(
        args.report_directory,
        output_directory=args.output_directory,
        alert_column=args.alert_column,
        interval_minutes=args.interval_minutes,
        timezone=args.timezone,
    )
    print(f"sorted_points={len(points)} events={len(summary)}")


if __name__ == "__main__":
    main()
