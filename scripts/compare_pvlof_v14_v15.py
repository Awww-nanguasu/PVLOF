"""Compare PVLOF v1.4 dual gate with v1.5 memory and 5% relative gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from pv_anomaly.pvlof_events import reconstruct_pvlof_events

KEYS = ["plant_id", "device_no", "event_time"]
STRING_KEYS = [*KEYS, "string_no"]
LABELS = ["PVLOF_V1_4_DUAL_GATE", "PVLOF_V1_5_MEMORY_5PCT"]
CUSTOMER_EVENT_COLUMNS = [
    "row", "event_id", "plant_id", "device_no", "raise_time_local",
    "end_time_local", *LABELS, "comparison_case", "alert_time_points",
    "duration_minutes",
]


def _read_event_input(path: str, final_alert_column: str) -> pd.DataFrame:
    candidates = [
        *STRING_KEYS,
        final_alert_column,
        "isolated_directional_raw_alert", "isolated_hier_raw_alert",
        "isolated_raw_alert", "collective_raw_alert",
        "isolated_directional_alert", "isolated_hier_strict_alert",
        "isolated_alert", "collective_member_alert", "pvlof_v2_legacy_alert",
        "isolated_directional_consecutive", "isolated_hier_strict_consecutive",
        "isolated_consecutive", "collective_event_consecutive",
        "pvlof_v15_raw_anomaly", "pvlof_v15_entry_streak",
        "pvlof_v15_normal_streak", "pvlof_v15_memory_active",
        "pvlof_v15_memory_reactivated_alert", "pvlof_v15_memory_clear_code",
    ]
    available = set(pq.ParquetFile(path).schema.names)
    columns = list(dict.fromkeys(column for column in candidates if column in available))
    return pd.read_parquet(path, columns=columns)


def _join_unique(values: pd.Series) -> str:
    return ",".join(sorted({str(value) for value in values.dropna() if str(value)}))


def _attach_event_metadata(
    comparison_events: pd.DataFrame,
    comparison_points: pd.DataFrame,
    evidence: pd.DataFrame,
    string_events: pd.DataFrame,
    *,
    prefix: str,
    timezone: str,
) -> pd.DataFrame:
    if evidence.empty or string_events.empty:
        return comparison_events
    final = evidence[evidence["final_alert"].astype(bool)][
        ["event_id", *KEYS]
    ].rename(columns={"event_id": "source_event_id"})
    links = final.merge(
        comparison_points[["event_id", *KEYS]].rename(
            columns={"event_id": "comparison_event_id"}
        ),
        on=KEYS,
        how="inner",
    )[["comparison_event_id", "source_event_id"]].drop_duplicates()
    if links.empty:
        return comparison_events
    linked = links.merge(
        string_events,
        left_on="source_event_id",
        right_on="event_id",
        how="left",
        validate="many_to_one",
    )
    metadata = linked.groupby("comparison_event_id", observed=True).agg(
        source_event_ids=("source_event_id", _join_unique),
        candidate_start_time=("candidate_start_time", "min"),
        alert_confirm_time=("alert_confirm_time", "min"),
        last_anomaly_time=("last_anomaly_time", "max"),
        clear_time=("clear_time", "max"),
        evidence_points=("evidence_points", "sum"),
        final_alert_points=("final_alert_points", "sum"),
        confirmation_branches=("confirmation_branch", _join_unique),
        clear_reasons=("clear_reason", _join_unique),
    ).reset_index()
    metadata = metadata.rename(columns={
        column: f"{prefix}_{column}"
        for column in metadata.columns if column != "comparison_event_id"
    })
    for column in (
        "candidate_start_time", "alert_confirm_time", "last_anomaly_time", "clear_time"
    ):
        source_column = f"{prefix}_{column}"
        metadata[f"{source_column}_local"] = pd.to_datetime(
            metadata[source_column], errors="coerce", utc=True
        ).dt.tz_convert(timezone).dt.strftime("%Y-%m-%d %H:%M:%S")
    return comparison_events.merge(
        metadata,
        left_on="event_id",
        right_on="comparison_event_id",
        how="left",
        validate="one_to_one",
    ).drop(columns="comparison_event_id")


def _numbers(values: pd.Series) -> str:
    numbers: set[int] = set()
    for value in values.dropna():
        for item in str(value).split(","):
            if item.strip():
                numbers.add(int(float(item)))
    return ",".join(f"{number:02d}" for number in sorted(numbers))


def _alerts(path: str, alert_column: str, label: str) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=[*STRING_KEYS, alert_column])
    frame["plant_id"] = frame["plant_id"].astype(str)
    frame["device_no"] = frame["device_no"].astype(str)
    frame["event_time"] = pd.to_datetime(frame["event_time"], utc=True)
    selected = frame[frame[alert_column].fillna(False).astype(bool)]
    return (
        selected.groupby(KEYS, observed=True)["string_no"]
        .agg(_numbers).rename(label).reset_index()
    )


def _case(frame: pd.DataFrame) -> pd.Series:
    left = frame[LABELS[0]].ne("")
    right = frame[LABELS[1]].ne("")
    result = pd.Series("both_same", index=frame.index, dtype="string")
    result.loc[left & ~right] = "v1_4_only"
    result.loc[~left & right] = "v1_5_only"
    result.loc[left & right & frame[LABELS[0]].ne(frame[LABELS[1]])] = "both_different"
    return result


def _confirmed_alert_points(
    evidence: pd.DataFrame,
    *,
    label: str,
) -> pd.DataFrame:
    """Return retrospectively confirmed abnormal points for the customer table.

    Candidate points are included only when they belong to a subsequently
    confirmed event. Recovery-observation points held by v1.5 memory are not
    alerts and are deliberately excluded.
    """

    if evidence.empty:
        return pd.DataFrame(columns=[*KEYS, label])
    abnormal = evidence[
        evidence["raw_evidence"].fillna(False).astype(bool)
        | evidence["final_alert"].fillna(False).astype(bool)
    ].copy()
    if abnormal.empty:
        return pd.DataFrame(columns=[*KEYS, label])
    abnormal["event_time"] = pd.to_datetime(
        abnormal["event_time"], errors="raise", utc=True
    )
    return (
        abnormal.groupby(KEYS, observed=True)["string_no"]
        .agg(_numbers).rename(label).reset_index()
    )


def _build_customer_events(
    v14_evidence: pd.DataFrame,
    v15_evidence: pd.DataFrame,
    *,
    timezone: str,
    interval_minutes: int,
) -> pd.DataFrame:
    """Build the compact, customer-facing confirmed event table.

    Event start is the first confirmed candidate point, not the confirmation
    time. A normal observation is absent from the alert-point stream, so a
    later v1.5 memory reactivation becomes a separate event.
    """

    left = _confirmed_alert_points(v14_evidence, label=LABELS[0])
    right = _confirmed_alert_points(v15_evidence, label=LABELS[1])
    points = left.merge(right, on=KEYS, how="outer", validate="one_to_one")
    if points.empty:
        return pd.DataFrame(columns=CUSTOMER_EVENT_COLUMNS)
    points[LABELS] = points[LABELS].astype("string").fillna("")
    points["comparison_case"] = _case(points)
    points = points.sort_values(KEYS).reset_index(drop=True)

    expected = pd.Timedelta(minutes=interval_minutes)
    points["_event"] = (
        points["plant_id"].ne(points["plant_id"].shift())
        | points["device_no"].ne(points["device_no"].shift())
        | points["event_time"].sub(points["event_time"].shift()).ne(expected)
    ).cumsum()
    points["event_id"] = points["_event"].map({
        value: f"pvlof-v14-v15-customer-{index:06d}"
        for index, value in enumerate(points["_event"].unique(), 1)
    })
    events = (
        points.groupby(
            ["event_id", "plant_id", "device_no", "_event"], observed=True
        )
        .agg(
            raise_time=("event_time", "min"),
            end_time=("event_time", "max"),
            PVLOF_V1_4_DUAL_GATE=(LABELS[0], _numbers),
            PVLOF_V1_5_MEMORY_5PCT=(LABELS[1], _numbers),
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
    return events[CUSTOMER_EVENT_COLUMNS]


def _v15_additions(
    v14_path: str,
    v15_path: str,
    v14_alert_column: str,
    v15_alert_column: str,
    timezone: str,
) -> pd.DataFrame:
    candidates = [
        "string_current", "expected_current", "isolated_absolute_drop",
        "isolated_relative_drop", "pvlof_score", "isolated_hier_lof_threshold",
        "pvlof_v15_raw_anomaly", "pvlof_v15_entry_streak",
        "pvlof_v15_normal_streak", "pvlof_v15_memory_active",
        "pvlof_v15_memory_reactivated_alert", "pvlof_v2_hier_strict_alert",
    ]
    available = set(pq.ParquetFile(v15_path).schema.names)
    diagnostics = [column for column in candidates if column in available]
    left = pd.read_parquet(
        v14_path, columns=[*STRING_KEYS, v14_alert_column]
    ).rename(columns={v14_alert_column: "_v14_alert"})
    right = pd.read_parquet(
        v15_path, columns=[*STRING_KEYS, v15_alert_column, *diagnostics]
    ).rename(columns={v15_alert_column: "_v15_alert"})
    for frame in (left, right):
        frame["plant_id"] = frame["plant_id"].astype(str)
        frame["device_no"] = frame["device_no"].astype(str)
        frame["event_time"] = pd.to_datetime(frame["event_time"], utc=True)
        frame["string_no"] = pd.to_numeric(frame["string_no"], errors="raise").astype("Int64")
    merged = right.merge(left, on=STRING_KEYS, how="left", validate="one_to_one")
    additions = merged[
        merged["_v15_alert"].fillna(False).astype(bool)
        & ~merged["_v14_alert"].fillna(False).astype(bool)
    ].copy()
    memory = (
        additions["pvlof_v15_memory_reactivated_alert"].fillna(False).astype(bool)
        if "pvlof_v15_memory_reactivated_alert" in additions
        else pd.Series(False, index=additions.index)
    )
    additions["addition_reason"] = "relative_5pct"
    additions.loc[memory, "addition_reason"] = "memory_reactivation"
    additions["event_time_local"] = additions["event_time"].dt.tz_convert(
        timezone
    ).dt.strftime("%Y-%m-%d %H:%M:%S")
    additions["string_no"] = additions["string_no"].map(
        lambda value: f"{int(value):02d}" if pd.notna(value) else ""
    )
    columns = [
        "plant_id", "device_no", "event_time_local", "string_no",
        "addition_reason", *diagnostics,
    ]
    return additions[columns].sort_values(
        ["plant_id", "device_no", "event_time_local", "string_no"]
    ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-4", required=True)
    parser.add_argument("--v1-5", required=True)
    parser.add_argument("--v1-4-alert-column", default="pvlof_v2_hier_strict_alert")
    parser.add_argument("--v1-5-alert-column", default="pvlof_v15_alert")
    parser.add_argument("--timezone", default="Asia/Shanghai")
    parser.add_argument("--interval-minutes", type=int, default=5)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument(
        "--customer-only",
        action="store_true",
        help="Write only the compact customer event CSV; preserve existing detailed CSVs.",
    )
    args = parser.parse_args()

    output = Path(args.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    customer_event_path = output / "pvlof_v14_v15_customer_events.csv"
    if args.customer_only:
        v14_evidence, _, v14_unconfirmed = reconstruct_pvlof_events(
            _read_event_input(args.v1_4, args.v1_4_alert_column),
            version="v1.4",
            final_alert_column=args.v1_4_alert_column,
            expected_interval_minutes=args.interval_minutes,
            timezone=args.timezone,
        )
        v15_evidence, _, v15_unconfirmed = reconstruct_pvlof_events(
            _read_event_input(args.v1_5, args.v1_5_alert_column),
            version="v1.5",
            final_alert_column=args.v1_5_alert_column,
            expected_interval_minutes=args.interval_minutes,
            timezone=args.timezone,
            use_v15_memory=True,
        )
        customer_events = _build_customer_events(
            v14_evidence,
            v15_evidence,
            timezone=args.timezone,
            interval_minutes=args.interval_minutes,
        )
        customer_events.to_csv(
            customer_event_path, index=False, encoding="utf-8-sig"
        )
        customer_report = {
            "v1_4": args.v1_4,
            "v1_5": args.v1_5,
            "customer_events": len(customer_events),
            "customer_event_cases": customer_events[
                "comparison_case"
            ].value_counts().to_dict(),
            "unconfirmed_candidate_runs_excluded": (
                len(v14_unconfirmed) + len(v15_unconfirmed)
            ),
            "output": str(customer_event_path),
        }
        (output / "customer_events_summary.json").write_text(
            json.dumps(customer_report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(customer_report, ensure_ascii=False, indent=2))
        return

    left = _alerts(args.v1_4, args.v1_4_alert_column, LABELS[0])
    right = _alerts(args.v1_5, args.v1_5_alert_column, LABELS[1])
    points = left.merge(right, on=KEYS, how="outer", validate="one_to_one")
    points[LABELS] = points[LABELS].astype("string").fillna("")
    points["comparison_case"] = _case(points)
    points = points.sort_values(KEYS).reset_index(drop=True)

    expected = pd.Timedelta(minutes=args.interval_minutes)
    points["_event"] = (
        points["plant_id"].ne(points["plant_id"].shift())
        | points["device_no"].ne(points["device_no"].shift())
        | points["event_time"].sub(points["event_time"].shift()).ne(expected)
    ).cumsum()
    points["event_id"] = points["_event"].map({
        value: f"pvlof-v14-v15-{index:06d}"
        for index, value in enumerate(points["_event"].unique(), 1)
    })
    events = (
        points.groupby(["event_id", "plant_id", "device_no", "_event"], observed=True)
        .agg(
            raise_time=("event_time", "min"), end_time=("event_time", "max"),
            PVLOF_V1_4_DUAL_GATE=(LABELS[0], _numbers),
            PVLOF_V1_5_MEMORY_5PCT=(LABELS[1], _numbers),
            alert_time_points=("event_time", "size"),
        ).reset_index().drop(columns="_event")
    )
    events["comparison_case"] = _case(events)
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

    additions = _v15_additions(
        args.v1_4, args.v1_5, args.v1_4_alert_column,
        args.v1_5_alert_column, args.timezone,
    )
    additions.insert(0, "row", range(1, len(additions) + 1))

    v14_evidence, v14_string_events, v14_unconfirmed = reconstruct_pvlof_events(
        _read_event_input(args.v1_4, args.v1_4_alert_column),
        version="v1.4",
        final_alert_column=args.v1_4_alert_column,
        expected_interval_minutes=args.interval_minutes,
        timezone=args.timezone,
    )
    v15_evidence, v15_string_events, v15_unconfirmed = reconstruct_pvlof_events(
        _read_event_input(args.v1_5, args.v1_5_alert_column),
        version="v1.5",
        final_alert_column=args.v1_5_alert_column,
        expected_interval_minutes=args.interval_minutes,
        timezone=args.timezone,
        use_v15_memory=True,
    )
    evidence = pd.concat([v14_evidence, v15_evidence], ignore_index=True)
    string_events = pd.concat(
        [v14_string_events, v15_string_events], ignore_index=True
    )
    unconfirmed = pd.concat([v14_unconfirmed, v15_unconfirmed], ignore_index=True)
    customer_events = _build_customer_events(
        v14_evidence,
        v15_evidence,
        timezone=args.timezone,
        interval_minutes=args.interval_minutes,
    )
    events = _attach_event_metadata(
        events, points, v14_evidence, v14_string_events,
        prefix="v1_4", timezone=args.timezone,
    )
    events = _attach_event_metadata(
        events, points, v15_evidence, v15_string_events,
        prefix="v1_5", timezone=args.timezone,
    )

    event_path = output / "pvlof_v14_v15_events.csv"
    point_path = output / "pvlof_v14_v15_points.csv"
    additions_path = output / "pvlof_v15_additions.csv"
    evidence_path = output / "pvlof_v14_v15_evidence_points.csv"
    string_event_path = output / "pvlof_v14_v15_string_events.csv"
    unconfirmed_path = output / "pvlof_unconfirmed_candidates.csv"
    event_columns = [
        "row", "event_id", "plant_id", "device_no", "raise_time_local",
        "end_time_local", *LABELS, "comparison_case", "alert_time_points",
        "duration_minutes", "manual_is_low_current", "manual_string_no", "review_note",
    ]
    metadata_columns = [
        column for column in events.columns
        if column.startswith("v1_4_") or column.startswith("v1_5_")
    ]
    events[event_columns + metadata_columns].to_csv(
        event_path, index=False, encoding="utf-8-sig"
    )
    points[[
        "row", "event_id", "plant_id", "device_no", "event_time_local",
        *LABELS, "comparison_case",
    ]].to_csv(point_path, index=False, encoding="utf-8-sig")
    additions.to_csv(additions_path, index=False, encoding="utf-8-sig")
    evidence.to_csv(evidence_path, index=False, encoding="utf-8-sig")
    string_events.to_csv(string_event_path, index=False, encoding="utf-8-sig")
    unconfirmed.to_csv(unconfirmed_path, index=False, encoding="utf-8-sig")
    customer_events.to_csv(
        customer_event_path, index=False, encoding="utf-8-sig"
    )

    per_plant = {}
    for plant in sorted(set(points["plant_id"])):
        plant_points = points[points["plant_id"].eq(plant)]
        plant_events = events[events["plant_id"].eq(plant)]
        plant_additions = additions[additions["plant_id"].eq(plant)]
        plant_customer_events = customer_events[
            customer_events["plant_id"].eq(plant)
        ]
        per_plant[str(plant)] = {
            "point_cases": plant_points["comparison_case"].value_counts().to_dict(),
            "event_cases": plant_events["comparison_case"].value_counts().to_dict(),
            "customer_event_cases": plant_customer_events[
                "comparison_case"
            ].value_counts().to_dict(),
            "v1_5_addition_reasons": plant_additions["addition_reason"].value_counts().to_dict(),
        }
    report = {
        "v1_4": args.v1_4,
        "v1_5": args.v1_5,
        "events": len(events),
        "points": len(points),
        "event_cases": events["comparison_case"].value_counts().to_dict(),
        "point_cases": points["comparison_case"].value_counts().to_dict(),
        "v1_5_addition_reasons": additions["addition_reason"].value_counts().to_dict(),
        "evidence_points": len(evidence),
        "string_events": len(string_events),
        "unconfirmed_candidate_runs": len(unconfirmed),
        "customer_events": len(customer_events),
        "customer_event_cases": customer_events[
            "comparison_case"
        ].value_counts().to_dict(),
        "per_plant": per_plant,
        "outputs": {
            "events": str(event_path), "points": str(point_path),
            "v1_5_additions": str(additions_path),
            "evidence_points": str(evidence_path),
            "string_events": str(string_event_path),
            "unconfirmed_candidates": str(unconfirmed_path),
            "customer_events": str(customer_event_path),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
