"""Export confirmed-event comparison tables for PVLOF v1.6 and v1.7."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from pv_anomaly.pvlof_events import reconstruct_pvlof_events


KEYS = ["plant_id", "device_no", "event_time"]
STRING_KEYS = [*KEYS, "string_no"]
LABELS = ["PVLOF_V1_6_HYBRID_GATE", "PVLOF_V1_7_PENALIZED_SEGMENTATION"]
OUTPUT_COLUMNS = [
    "row", "event_id", "plant_id", "device_no", "raise_time_local",
    "end_time_local", *LABELS, "comparison_case", "alert_time_points",
    "duration_minutes",
]


def _read_event_input(path: str, *, version: str) -> pd.DataFrame:
    prefix = "pvlof_v16" if version == "v1.6" else "pvlof_v17"
    final_alert = f"{prefix}_alert"
    candidates = [
        *STRING_KEYS, final_alert,
        "isolated_directional_raw_alert", "isolated_hier_raw_alert",
        "isolated_raw_alert", "collective_raw_alert",
        "isolated_directional_alert", "isolated_hier_strict_alert",
        "isolated_alert", "collective_member_alert", "pvlof_v2_legacy_alert",
        "isolated_directional_consecutive", "isolated_hier_strict_consecutive",
        "isolated_consecutive", "collective_event_consecutive",
        f"{prefix}_raw_anomaly", f"{prefix}_entry_streak",
        f"{prefix}_normal_streak", f"{prefix}_memory_active",
        f"{prefix}_memory_reactivated_alert", f"{prefix}_memory_clear_code",
    ]
    available = set(pq.ParquetFile(path).schema.names)
    columns = list(dict.fromkeys(column for column in candidates if column in available))
    result = pd.read_parquet(path, columns=columns)
    if version == "v1.7":
        for suffix in (
            "raw_anomaly", "entry_streak", "normal_streak", "memory_active",
            "memory_reactivated_alert", "memory_clear_code",
        ):
            result[f"pvlof_v16_{suffix}"] = result[f"pvlof_v17_{suffix}"]
    return result


def _reconstruct(path: str, *, version: str, interval: int, timezone: str):
    prefix = "pvlof_v16" if version == "v1.6" else "pvlof_v17"
    evidence, events, unconfirmed = reconstruct_pvlof_events(
        _read_event_input(path, version=version),
        version=version,
        final_alert_column=f"{prefix}_alert",
        expected_interval_minutes=interval,
        timezone=timezone,
        memory_prefix="pvlof_v16",
    )
    if version == "v1.7":
        for frame in (evidence, events):
            frame.loc[
                frame["confirmation_branch"].eq("final_union"),
                "confirmation_branch",
            ] = "penalized_segmentation"
    return evidence, events, unconfirmed


def _numbers(values: pd.Series) -> str:
    result = {
        int(float(item))
        for value in values.dropna()
        for item in str(value).split(",")
        if item.strip()
    }
    return ",".join(f"{number:02d}" for number in sorted(result))


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


def _case(frame: pd.DataFrame) -> pd.Series:
    left = frame[LABELS[0]].ne("")
    right = frame[LABELS[1]].ne("")
    result = pd.Series("both_same", index=frame.index, dtype="string")
    result.loc[left & ~right] = "v1_6_only"
    result.loc[~left & right] = "v1_7_only"
    result.loc[left & right & frame[LABELS[0]].ne(frame[LABELS[1]])] = (
        "both_different"
    )
    return result


def _customer_events(
    v16_evidence: pd.DataFrame,
    v17_evidence: pd.DataFrame,
    *,
    timezone: str,
    interval: int,
) -> pd.DataFrame:
    left = _confirmed_points(v16_evidence, LABELS[0])
    right = _confirmed_points(v17_evidence, LABELS[1])
    points = left.merge(right, on=KEYS, how="outer", validate="one_to_one")
    if points.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    points[LABELS] = points[LABELS].astype("string").fillna("")
    points = points.sort_values(KEYS).reset_index(drop=True)
    expected = pd.Timedelta(minutes=interval)
    points["_event"] = (
        points["plant_id"].ne(points["plant_id"].shift())
        | points["device_no"].ne(points["device_no"].shift())
        | points["event_time"].sub(points["event_time"].shift()).ne(expected)
    ).cumsum()
    points["event_id"] = points["_event"].map({
        value: f"pvlof-v16-v17-{number:06d}"
        for number, value in enumerate(points["_event"].unique(), 1)
    })
    events = (
        points.groupby(
            ["event_id", "plant_id", "device_no", "_event"], observed=True
        )
        .agg(
            raise_time=("event_time", "min"),
            end_time=("event_time", "max"),
            PVLOF_V1_6_HYBRID_GATE=(LABELS[0], _numbers),
            PVLOF_V1_7_PENALIZED_SEGMENTATION=(LABELS[1], _numbers),
            alert_time_points=("event_time", "size"),
        )
        .reset_index().drop(columns="_event")
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
        + interval
    ).astype(int)
    events = events.sort_values(
        ["plant_id", "device_no", "raise_time"]
    ).reset_index(drop=True)
    events.insert(0, "row", range(1, len(events) + 1))
    return events[OUTPUT_COLUMNS]


def _v17_additions(v16_path: str, v17_path: str, timezone: str) -> pd.DataFrame:
    diagnostics = [
        "string_current", "expected_current", "residual_ratio", "residual_median",
        "pvlof_score", "pvlof_v17_relative_drop", "pvlof_v17_absolute_drop",
        "pvlof_v17_physical_effect_eligible",
        "pvlof_v17_segmentation_structural_member",
        "pvlof_v17_segmentation_raw_candidate",
        "pvlof_v17_segmentation_memory_reactivated_alert",
        "pvlof_v17_segment_index", "pvlof_v17_segment_count",
        "pvlof_v17_segmentation_candidate_size",
        "pvlof_v17_segmentation_reference_size",
        "pvlof_v17_segmentation_reference_residual",
        "pvlof_v17_segmentation_objective", "pvlof_v17_segmentation_sse_gain",
    ]
    available = set(pq.ParquetFile(v17_path).schema.names)
    diagnostics = [column for column in diagnostics if column in available]
    left = pd.read_parquet(
        v16_path, columns=[*STRING_KEYS, "pvlof_v16_alert"]
    ).rename(columns={"pvlof_v16_alert": "_v16_alert"})
    right = pd.read_parquet(
        v17_path, columns=[*STRING_KEYS, "pvlof_v17_alert", *diagnostics]
    ).rename(columns={"pvlof_v17_alert": "_v17_alert"})
    for frame in (left, right):
        frame["plant_id"] = frame["plant_id"].astype(str)
        frame["device_no"] = frame["device_no"].astype(str)
        frame["event_time"] = pd.to_datetime(frame["event_time"], utc=True)
        frame["string_no"] = pd.to_numeric(
            frame["string_no"], errors="raise"
        ).astype("Int64")
    merged = right.merge(left, on=STRING_KEYS, how="left", validate="one_to_one")
    additions = merged[
        merged["_v17_alert"].fillna(False).astype(bool)
        & ~merged["_v16_alert"].fillna(False).astype(bool)
    ].copy()
    additions["addition_reason"] = "penalized_segmentation"
    if "pvlof_v17_segmentation_memory_reactivated_alert" in additions:
        reactivated = additions[
            "pvlof_v17_segmentation_memory_reactivated_alert"
        ].fillna(False).astype(bool)
        additions.loc[reactivated, "addition_reason"] = (
            "segmentation_memory_reactivation"
        )
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
    parser.add_argument("--v1-6", required=True)
    parser.add_argument("--v1-7", required=True)
    parser.add_argument("--timezone", default="Asia/Shanghai")
    parser.add_argument("--interval-minutes", type=int, default=5)
    parser.add_argument("--output-directory", required=True)
    args = parser.parse_args()

    v16_evidence, _, v16_unconfirmed = _reconstruct(
        args.v1_6,
        version="v1.6",
        interval=args.interval_minutes,
        timezone=args.timezone,
    )
    v17_evidence, _, v17_unconfirmed = _reconstruct(
        args.v1_7,
        version="v1.7",
        interval=args.interval_minutes,
        timezone=args.timezone,
    )
    events = _customer_events(
        v16_evidence,
        v17_evidence,
        timezone=args.timezone,
        interval=args.interval_minutes,
    )
    additions = _v17_additions(args.v1_6, args.v1_7, args.timezone)
    output = Path(args.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    event_path = output / "pvlof_v16_v17_customer_events.csv"
    additions_path = output / "pvlof_v17_additions.csv"
    events.to_csv(event_path, index=False, encoding="utf-8-sig")
    additions.to_csv(additions_path, index=False, encoding="utf-8-sig")
    report = {
        "v1_6": args.v1_6,
        "v1_7": args.v1_7,
        "events": len(events),
        "event_cases": events["comparison_case"].value_counts().to_dict(),
        "v1_7_added_alert_points": len(additions),
        "v1_7_addition_reasons": additions["addition_reason"].value_counts().to_dict(),
        "unconfirmed_candidate_runs_excluded": (
            len(v16_unconfirmed) + len(v17_unconfirmed)
        ),
        "outputs": {
            "customer_events": str(event_path),
            "v1_7_additions": str(additions_path),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
