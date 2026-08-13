"""Export all string currents sorted at each 5-minute 101001 alarm timestamp."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def export_review(
    report_directory: str | Path,
    *,
    output_directory: str | Path | None = None,
    alarm_code: str = "101001",
    timezone: str = "Asia/Shanghai",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write long sorted-current rows and one summary row per alarm event."""

    report = Path(report_directory)
    output = Path(output_directory) if output_directory else report
    output.mkdir(parents=True, exist_ok=True)
    alarms = pd.read_parquet(report / "alarm_events.parquet")
    points = pd.read_parquet(report / "aligned_pvlof_points.parquet")
    alarms = alarms[alarms["alarm_code"].astype(str) == str(alarm_code)].copy()
    alarms["raise_time"] = pd.to_datetime(alarms["raise_time"], utc=True)
    alarms["end_time"] = pd.to_datetime(alarms["end_time"], utc=True)
    points["event_time"] = pd.to_datetime(points["event_time"], utc=True)
    points["device_no"] = points["device_no"].astype(str)

    review_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for alarm in alarms.itertuples(index=False):
        window = points[
            (points["device_no"] == str(alarm.device_no))
            & (points["event_time"] >= alarm.raise_time)
            & (points["event_time"] <= alarm.end_time)
        ].copy()
        if window.empty:
            summary_rows.append(
                {
                    "alarm_event_id": alarm.alarm_event_id,
                    "station_name": alarm.station_name,
                    "device_no": alarm.device_no,
                    "alarm_code": alarm.alarm_code,
                    "raise_local": alarm.raise_time.tz_convert(timezone).isoformat(),
                    "end_local": alarm.end_time.tz_convert(timezone).isoformat(),
                    "timestamps": 0,
                    "rows": 0,
                    "lowest_string_no": None,
                    "minimum_current": None,
                    "minimum_current_ratio": None,
                    "pvlof_alert_points": 0,
                    "combined_alert_points": 0,
                }
            )
            continue

        group = ["event_time"]
        window["current_rank_ascending"] = (
            window.groupby(group, observed=True)["string_current"]
            .rank(method="first", ascending=True)
            .astype("Int64")
        )
        window["strings_at_timestamp"] = window.groupby(
            group, observed=True
        )["string_no"].transform("count")
        window["is_lowest_current"] = window["current_rank_ascending"].eq(1)
        window["alarm_event_id"] = alarm.alarm_event_id
        window["alarm_code"] = alarm.alarm_code
        window["alarm_name"] = alarm.alarm_name
        window["alarm_station_name"] = alarm.station_name
        window["alarm_raise_time"] = alarm.raise_time
        window["alarm_end_time"] = alarm.end_time
        window["event_time_local"] = window["event_time"].dt.tz_convert(timezone).dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        review_rows.append(window)

        lowest = window.loc[window["string_current"].idxmin()]
        summary_rows.append(
            {
                "alarm_event_id": alarm.alarm_event_id,
                "station_name": alarm.station_name,
                "device_no": alarm.device_no,
                "alarm_code": alarm.alarm_code,
                "raise_local": alarm.raise_time.tz_convert(timezone).isoformat(),
                "end_local": alarm.end_time.tz_convert(timezone).isoformat(),
                "timestamps": int(window["event_time"].nunique()),
                "rows": int(len(window)),
                "lowest_string_no": int(lowest["string_no"]),
                "minimum_current": float(lowest["string_current"]),
                "minimum_current_ratio": float(lowest["string_current_ratio"])
                if pd.notna(lowest["string_current_ratio"])
                else None,
                "pvlof_alert_points": int(window["pvlof_alert"].sum()),
                "combined_alert_points": int(window["combined_alert"].sum()),
            }
        )

    columns = [
        "alarm_event_id",
        "alarm_station_name",
        "device_no",
        "alarm_code",
        "alarm_name",
        "alarm_raise_time",
        "alarm_end_time",
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
    sorted_rows = (
        pd.concat(review_rows, ignore_index=True)[columns]
        if review_rows
        else pd.DataFrame(columns=columns)
    )
    sorted_rows = sorted_rows.sort_values(
        ["alarm_event_id", "event_time", "string_current", "string_no"]
    )
    summary = pd.DataFrame(summary_rows).sort_values(["raise_local", "device_no"])
    sorted_rows.to_csv(output / "low_current_sorted_points.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output / "low_current_event_summary.csv", index=False, encoding="utf-8-sig")
    return sorted_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-directory", required=True)
    parser.add_argument("--output-directory", default=None)
    parser.add_argument("--alarm-code", default="101001")
    parser.add_argument("--timezone", default="Asia/Shanghai")
    args = parser.parse_args()
    points, summary = export_review(
        args.report_directory,
        output_directory=args.output_directory,
        alarm_code=args.alarm_code,
        timezone=args.timezone,
    )
    print(f"sorted_points={len(points)} events={len(summary)}")


if __name__ == "__main__":
    main()
