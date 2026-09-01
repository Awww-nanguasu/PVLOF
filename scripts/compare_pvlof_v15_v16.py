"""Export the compact customer comparison of PVLOF v1.5 and v1.6."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from pv_anomaly.pvlof_events import reconstruct_pvlof_events


KEYS = ["plant_id", "device_no", "event_time"]
STRING_KEYS = [*KEYS, "string_no"]
LABELS = ["PVLOF_V1_5_MEMORY_5PCT", "PVLOF_V1_6_HYBRID_GATE"]
OUTPUT_COLUMNS = [
    "row", "event_id", "plant_id", "device_no", "raise_time_local",
    "end_time_local", *LABELS, "comparison_case", "alert_time_points",
    "duration_minutes",
]


def _read_event_input(
    path: str,
    *,
    final_alert_column: str,
    memory_prefix: str,
) -> pd.DataFrame:
    candidates = [
        *STRING_KEYS,
        final_alert_column,
        "isolated_directional_raw_alert", "isolated_hier_raw_alert",
        "isolated_raw_alert", "collective_raw_alert",
        "isolated_directional_alert", "isolated_hier_strict_alert",
        "isolated_alert", "collective_member_alert", "pvlof_v2_legacy_alert",
        "isolated_directional_consecutive", "isolated_hier_strict_consecutive",
        "isolated_consecutive", "collective_event_consecutive",
        f"{memory_prefix}_raw_anomaly", f"{memory_prefix}_entry_streak",
        f"{memory_prefix}_normal_streak", f"{memory_prefix}_memory_active",
        f"{memory_prefix}_memory_reactivated_alert",
        f"{memory_prefix}_memory_clear_code",
    ]
    available = set(pq.ParquetFile(path).schema.names)
    columns = list(dict.fromkeys(c for c in candidates if c in available))
    return pd.read_parquet(path, columns=columns)


def _numbers(values: pd.Series) -> str:
    result = {
        int(float(item))
        for value in values.dropna()
        for item in str(value).split(",")
        if item.strip()
    }
    return ",".join(f"{number:02d}" for number in sorted(result))


def _case(frame: pd.DataFrame) -> pd.Series:
    left = frame[LABELS[0]].ne("")
    right = frame[LABELS[1]].ne("")
    result = pd.Series("both_same", index=frame.index, dtype="string")
    result.loc[left & ~right] = "v1_5_only"
    result.loc[~left & right] = "v1_6_only"
    result.loc[
        left & right & frame[LABELS[0]].ne(frame[LABELS[1]])
    ] = "both_different"
    return result


def _confirmed_points(evidence: pd.DataFrame, label: str) -> pd.DataFrame:
    if evidence.empty:
        return pd.DataFrame(columns=[*KEYS, label])
    points = evidence[
        evidence["raw_evidence"].fillna(False).astype(bool)
        | evidence["final_alert"].fillna(False).astype(bool)
    ].copy()
    if points.empty:
        return pd.DataFrame(columns=[*KEYS, label])
    points["event_time"] = pd.to_datetime(points["event_time"], utc=True)
    return (
        points.groupby(KEYS, observed=True)["string_no"]
        .agg(_numbers).rename(label).reset_index()
    )


def _customer_events(
    left_evidence: pd.DataFrame,
    right_evidence: pd.DataFrame,
    *,
    timezone: str,
    interval_minutes: int,
) -> pd.DataFrame:
    left = _confirmed_points(left_evidence, LABELS[0])
    right = _confirmed_points(right_evidence, LABELS[1])
    points = left.merge(right, on=KEYS, how="outer", validate="one_to_one")
    if points.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    points[LABELS] = points[LABELS].astype("string").fillna("")
    points = points.sort_values(KEYS).reset_index(drop=True)
    expected = pd.Timedelta(minutes=interval_minutes)
    points["_event"] = (
        points["plant_id"].ne(points["plant_id"].shift())
        | points["device_no"].ne(points["device_no"].shift())
        | points["event_time"].sub(points["event_time"].shift()).ne(expected)
    ).cumsum()
    points["event_id"] = points["_event"].map({
        value: f"pvlof-v15-v16-customer-{number:06d}"
        for number, value in enumerate(points["_event"].unique(), 1)
    })
    events = (
        points.groupby(
            ["event_id", "plant_id", "device_no", "_event"], observed=True
        )
        .agg(
            raise_time=("event_time", "min"),
            end_time=("event_time", "max"),
            PVLOF_V1_5_MEMORY_5PCT=(LABELS[0], _numbers),
            PVLOF_V1_6_HYBRID_GATE=(LABELS[1], _numbers),
            alert_time_points=("event_time", "size"),
        )
        .reset_index()
        .drop(columns="_event")
    )
    events["comparison_case"] = _case(events)
    events["raise_time_local"] = (
        events["raise_time"].dt.tz_convert(timezone).dt.strftime("%Y-%m-%d %H:%M:%S")
    )
    events["end_time_local"] = (
        events["end_time"].dt.tz_convert(timezone).dt.strftime("%Y-%m-%d %H:%M:%S")
    )
    events["duration_minutes"] = (
        (events["end_time"] - events["raise_time"]).dt.total_seconds() / 60
        + interval_minutes
    ).astype(int)
    events = events.sort_values(
        ["plant_id", "device_no", "raise_time"]
    ).reset_index(drop=True)
    events.insert(0, "row", range(1, len(events) + 1))
    return events[OUTPUT_COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-5", required=True)
    parser.add_argument("--v1-6", required=True)
    parser.add_argument("--timezone", default="Asia/Shanghai")
    parser.add_argument("--interval-minutes", type=int, default=5)
    parser.add_argument("--output-directory", required=True)
    args = parser.parse_args()

    v15_evidence, _, v15_unconfirmed = reconstruct_pvlof_events(
        _read_event_input(
            args.v1_5,
            final_alert_column="pvlof_v15_alert",
            memory_prefix="pvlof_v15",
        ),
        version="v1.5",
        final_alert_column="pvlof_v15_alert",
        expected_interval_minutes=args.interval_minutes,
        memory_prefix="pvlof_v15",
        timezone=args.timezone,
    )
    v16_evidence, _, v16_unconfirmed = reconstruct_pvlof_events(
        _read_event_input(
            args.v1_6,
            final_alert_column="pvlof_v16_alert",
            memory_prefix="pvlof_v16",
        ),
        version="v1.6",
        final_alert_column="pvlof_v16_alert",
        expected_interval_minutes=args.interval_minutes,
        memory_prefix="pvlof_v16",
        timezone=args.timezone,
    )
    events = _customer_events(
        v15_evidence,
        v16_evidence,
        timezone=args.timezone,
        interval_minutes=args.interval_minutes,
    )
    output = Path(args.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    event_path = output / "pvlof_v15_v16_customer_events.csv"
    events.to_csv(event_path, index=False, encoding="utf-8-sig")
    report = {
        "v1_5": args.v1_5,
        "v1_6": args.v1_6,
        "events": len(events),
        "event_cases": events["comparison_case"].value_counts().to_dict(),
        "unconfirmed_candidate_runs_excluded": (
            len(v15_unconfirmed) + len(v16_unconfirmed)
        ),
        "output": str(event_path),
    }
    (output / "customer_events_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
