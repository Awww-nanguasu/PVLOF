"""Export corrected alarm-grid currents with Baseline, V1, V2 and V1.6 results."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


KEYS = ["plant_id", "device_no", "event_time"]
CURRENT_PATTERN = re.compile(r"^string_current_(\d{2})$")


def _normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["plant_id"] = pd.to_numeric(result["plant_id"], errors="raise").astype(int)
    result["device_no"] = result["device_no"].astype(str).str.strip()
    result["event_time"] = pd.to_datetime(result["event_time"], errors="raise", utc=True)
    return result


def _alert_strings(
    path: str | Path,
    alert_columns: str | tuple[str, ...],
    output_column: str,
) -> pd.DataFrame:
    if isinstance(alert_columns, str):
        alert_columns = (alert_columns,)
    source = pd.read_parquet(
        path,
        columns=[*KEYS, "string_no", "string_current", *alert_columns],
    )
    source = _normalize_keys(source)
    source["string_no"] = pd.to_numeric(source["string_no"], errors="raise").astype(int)
    source["string_current"] = pd.to_numeric(source["string_current"], errors="coerce")
    alerted = pd.Series(False, index=source.index)
    for alert_column in alert_columns:
        alerted |= source[alert_column].fillna(False).astype(bool)
    source = source[alerted & source["string_current"].gt(0)]
    if source.empty:
        return pd.DataFrame(columns=[*KEYS, output_column])
    return (
        source.groupby(KEYS, observed=True)["string_no"]
        .agg(lambda values: ",".join(f"{value:02d}" for value in sorted(set(values))))
        .rename(output_column)
        .reset_index()
    )


def _local_text(values: pd.Series, timezone: str, *, fractional: bool) -> pd.Series:
    local = pd.to_datetime(values, errors="raise", utc=True).dt.tz_convert(timezone)
    if not fractional:
        return local.dt.strftime("%Y-%m-%d %H:%M:%S")
    return (
        local.dt.strftime("%Y-%m-%d %H:%M:%S.%f")
        .str.rstrip("0")
        .str.rstrip(".")
    )


def export_review(
    currents: str | Path,
    alarm_events: str | Path,
    v1_predictions: str | Path,
    v2_predictions: str | Path,
    v16_predictions: str | Path,
    output: str | Path,
    report: str | Path,
    *,
    timezone: str = "Asia/Shanghai",
) -> dict[str, Any]:
    source = _normalize_keys(pd.read_parquet(currents))
    required = {"alarm_event_id", *KEYS}
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"Current input is missing columns: {missing}")
    source["alarm_event_id"] = source["alarm_event_id"].astype(str)
    source = source.drop_duplicates(["alarm_event_id", *KEYS], keep="last")

    events = pd.read_parquet(alarm_events)
    if "classification" in events.columns:
        events = events[events["classification"].eq("complete")]
    event_required = {"alarm_event_id", "raise_time", "end_time"}
    event_missing = sorted(event_required - set(events.columns))
    if event_missing:
        raise ValueError(f"Alarm event input is missing columns: {event_missing}")
    events = events[list(event_required)].copy()
    events["alarm_event_id"] = events["alarm_event_id"].astype(str)
    events = events.drop_duplicates("alarm_event_id", keep="last")
    source = source.merge(events, on="alarm_event_id", how="left", validate="many_to_one")

    predictions = [
        (v1_predictions, "pvlof_alert", "PVLOF_V1"),
        (v2_predictions, "pvlof_v2_iso_mod_alert", "PVLOF_V2_iso_mod"),
        (
            v16_predictions,
            (
                "pvlof_v16_raw_anomaly",
                "collective_raw_alert",
                "pvlof_v16_alert",
            ),
            "PVLOF_V1_6_HYBRID_GATE",
        ),
    ]
    for path, alert_column, output_column in predictions:
        source = source.merge(
            _alert_strings(path, alert_column, output_column),
            on=KEYS,
            how="left",
            validate="many_to_one",
        )
        source[output_column] = source[output_column].fillna("FALSE")

    # Every exported row belongs to a cleaned device_alarm 101001 interval.
    # The legacy baseline label therefore comes from the alarm table itself,
    # not from production string_status_XX fields (which use different status
    # semantics and are not the source of these alarm records).
    source["Baseline"] = "TRUE"
    source["raise_time_local"] = _local_text(
        source["raise_time"], timezone, fractional=True
    )
    source["end_time_local"] = _local_text(
        source["end_time"], timezone, fractional=True
    )
    source["event_time_local"] = _local_text(
        source["event_time"], timezone, fractional=False
    )
    source = source.sort_values(
        ["plant_id", "device_no", "raise_time", "alarm_event_id", "event_time"]
    ).reset_index(drop=True)
    source.insert(0, "row", np.arange(1, len(source) + 1, dtype=np.int64))

    current_columns: list[str] = []
    rename: dict[str, str] = {}
    for string_no in range(1, 31):
        original = f"string_current_{string_no:02d}"
        if original not in source.columns:
            source[original] = np.nan
        source[original] = pd.to_numeric(source[original], errors="coerce")
        current_columns.append(original)
        rename[original] = f"{string_no:02d}"

    columns = [
        "Baseline",
        "PVLOF_V1",
        "PVLOF_V2_iso_mod",
        "PVLOF_V1_6_HYBRID_GATE",
        "row",
        "plant_id",
        "device_no",
        "raise_time_local",
        "end_time_local",
        "event_time_local",
        *current_columns,
    ]
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    source[columns].rename(columns=rename).to_csv(
        destination, index=False, encoding="utf-8-sig"
    )

    summary = {
        "currents": str(currents),
        "alarm_events": str(alarm_events),
        "output": str(destination),
        "alignment": "inclusive floor(raise_time) through floor(end_time)",
        "rows": int(len(source)),
        "events": int(source["alarm_event_id"].nunique()),
        "plants": int(source["plant_id"].nunique()),
        "devices": int(source["device_no"].nunique()),
        "baseline_definition": (
            "inside a cleaned device_alarm 101001 interval aligned by "
            "floor(raise_time) through floor(end_time)"
        ),
        "baseline_true": int(source["Baseline"].eq("TRUE").sum()),
        "prediction_true": {
            column: int(source[column].ne("FALSE").sum())
            for _, _, column in predictions
        },
        "prediction_files": {
            column: str(path) for path, _, column in predictions
        },
        "v1_6_export_definition": (
            "nonzero strings where pvlof_v16_raw_anomaly OR "
            "collective_raw_alert OR pvlof_v16_alert is true; isolated candidates, "
            "collective candidates and confirmed points are all shown"
        ),
    }
    report_path = Path(report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--currents", required=True)
    parser.add_argument("--alarm-events", required=True)
    parser.add_argument("--v1-predictions", required=True)
    parser.add_argument("--v2-predictions", required=True)
    parser.add_argument("--v16-predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--timezone", default="Asia/Shanghai")
    args = parser.parse_args()
    export_review(
        args.currents,
        args.alarm_events,
        args.v1_predictions,
        args.v2_predictions,
        args.v16_predictions,
        args.output,
        args.report,
        timezone=args.timezone,
    )


if __name__ == "__main__":
    main()
