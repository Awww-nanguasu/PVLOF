"""Audit EWMA points/events, weak labels, distributions and review windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pv_anomaly.ewma_audit import (
    add_weak_label,
    binary_metrics,
    collapse_alert_events,
    distribution_table,
    json_safe,
    match_event_tables,
    read_partitioned_columns,
)


def _read_curve_source(
    path: str | None,
    columns: list[str],
) -> tuple[pd.DataFrame | None, dict[str, object]]:
    if not path:
        return None, {"path": None, "rows": 0}
    result, report = read_partitioned_columns(path, columns)
    if result.empty:
        return None, report
    if "event_time" in result:
        result["event_time"] = pd.to_datetime(result["event_time"], errors="coerce", utc=True)
    if "device_no" in result:
        result["device_no"] = result["device_no"].astype(str)
    return result.dropna(subset=["event_time", "device_no"]), report


def _review_windows(
    alerts: pd.DataFrame,
    events: pd.DataFrame,
    curves: pd.DataFrame | None,
    *,
    maximum_events: int,
    window_minutes: int,
    event_type: str,
) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    selected = events.head(maximum_events)
    rows: list[pd.DataFrame] = []
    for _, event in selected.iterrows():
        start = event["start_time"] - pd.Timedelta(minutes=window_minutes)
        end = event["end_time"] + pd.Timedelta(minutes=window_minutes)
        point_rows = alerts[
            (alerts["device_no"].astype(str) == str(event["device_no"]))
            & (alerts["target_time"].between(start, end))
        ].copy()
        point_rows["review_event_id"] = event["event_id"]
        point_rows["review_event_type"] = event_type
        if curves is not None:
            curve_rows = curves[
                (curves["device_no"] == str(event["device_no"]))
                & (curves["event_time"].between(start, end))
            ].copy()
            curve_rows = curve_rows.rename(columns={"event_time": "target_time"})
            point_rows = point_rows.merge(
                curve_rows,
                on=["device_no", "target_time"],
                how="left",
                suffixes=("", "_curve"),
            )
        rows.append(point_rows)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ewma", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--curves", default=None)
    parser.add_argument("--minimum-consecutive", type=int, default=2)
    parser.add_argument("--interval-minutes", type=int, default=5)
    parser.add_argument("--review-events", type=int, default=10)
    parser.add_argument("--review-window-minutes", type=int, default=30)
    parser.add_argument(
        "--label-statuses",
        nargs="+",
        type=int,
        default=[2, 4],
        help="string_overall_status values treated as weak current anomalies",
    )
    args = parser.parse_args()

    output = Path(args.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    alerts = pd.read_parquet(args.ewma)
    labels, label_source_report = read_partitioned_columns(
        args.labels,
        ["event_time", "device_no", "string_overall_status", "low_current_count"],
    )
    merged, label_report = add_weak_label(
        alerts,
        labels,
        label_statuses=tuple(args.label_statuses),
    )

    point_all = binary_metrics(merged["weak_current_label"], merged["ewma_alert"])
    eligible = merged["ewma_eligible"].astype(bool)
    point_eligible = binary_metrics(
        merged.loc[eligible, "weak_current_label"],
        merged.loc[eligible, "ewma_alert"],
    )
    predicted_events = collapse_alert_events(
        merged,
        alert_column="ewma_alert",
        expected_interval_minutes=args.interval_minutes,
        event_prefix="ewma",
    )
    label_events = collapse_alert_events(
        merged,
        alert_column="weak_current_label",
        expected_interval_minutes=args.interval_minutes,
        event_prefix="weak",
    )
    event_metrics = match_event_tables(predicted_events, label_events)
    curves, curve_source_report = _read_curve_source(
        args.curves,
        [
            "event_time",
            "device_no",
            "active_power",
            "dc_power",
            *[f"string_current_{index:02d}" for index in range(1, 31)],
            "string_overall_status",
            *[f"string_status_{index:02d}" for index in range(1, 31)],
        ],
    )
    predicted_review = _review_windows(
        merged,
        predicted_events,
        curves,
        maximum_events=args.review_events,
        window_minutes=args.review_window_minutes,
        event_type="ewma",
    )
    label_review = _review_windows(
        merged,
        label_events,
        curves,
        maximum_events=args.review_events,
        window_minutes=args.review_window_minutes,
        event_type="weak_label",
    )
    review = pd.concat([predicted_review, label_review], ignore_index=True)

    distribution_table(merged, group_columns=["local_date"]).to_csv(
        output / "by_date.csv", index=False
    )
    distribution_table(merged, group_columns=["device_no"]).to_csv(
        output / "by_device.csv", index=False
    )
    predicted_events.to_csv(output / "ewma_events.csv", index=False)
    label_events.to_csv(output / "weak_label_events.csv", index=False)
    review.to_csv(output / "review_windows.csv", index=False)
    merged.to_parquet(output / "aligned_alerts.parquet", index=False)
    report = {
        "ewma": args.ewma,
        "labels": args.labels,
        "label_source": label_source_report,
        "samples": len(merged),
        "label_alignment": label_report,
        "point_metrics_all": point_all,
        "point_metrics_eligible": point_eligible,
        "event_metrics": event_metrics,
        "predicted_event_count": len(predicted_events),
        "weak_label_event_count": len(label_events),
        "curve_source_available": curves is not None,
        "curve_source": curve_source_report,
        "review_rows": len(review),
    }
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=json_safe) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=json_safe))


if __name__ == "__main__":
    main()
