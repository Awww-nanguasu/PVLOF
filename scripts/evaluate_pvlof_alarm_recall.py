"""Measure PVLOF recall on cleaned positive low-current alarm events."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


KEY_COLUMNS = ["plant_id", "device_no", "event_time"]


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is pd.NA or value is None:
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _read_parquet(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    source = Path(path)
    files = [source] if source.is_file() else sorted(source.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files found under {path}")
    return pd.concat(
        [pd.read_parquet(file, columns=columns) for file in files],
        ignore_index=True,
    )


def _normalize_keys(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    missing = sorted(set(KEY_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")
    result = frame.copy()
    result["plant_id"] = pd.to_numeric(result["plant_id"], errors="coerce").astype("Int64")
    result["device_no"] = result["device_no"].astype(str).str.strip()
    result["event_time"] = pd.to_datetime(result["event_time"], errors="raise", utc=True)
    return result


def _prediction_points(
    path: str | Path,
    *,
    alert_column: str = "pvlof_alert",
    eligible_column: str = "pvlof_eligible",
) -> pd.DataFrame:
    columns = [
        "plant_id",
        "device_no",
        "event_time",
        alert_column,
        eligible_column,
    ]
    source = _normalize_keys(_read_parquet(path, columns), name="PVLOF output")
    source[alert_column] = source[alert_column].astype(bool)
    source[eligible_column] = source[eligible_column].astype(bool)
    return (
        source.groupby(KEY_COLUMNS, observed=True, as_index=False)
        .agg(
            pvlof_detected=(alert_column, "max"),
            pvlof_score_eligible=(eligible_column, "max"),
            pvlof_alerted_strings=(alert_column, "sum"),
            pvlof_output_strings=(alert_column, "size"),
        )
        .sort_values(KEY_COLUMNS)
        .reset_index(drop=True)
    )


def _alarm_points(path: str | Path) -> pd.DataFrame:
    points = _normalize_keys(_read_parquet(path), name="cleaned alarm points")
    required = {"alarm_event_id", *KEY_COLUMNS}
    missing = sorted(required - set(points.columns))
    if missing:
        raise ValueError(f"Cleaned alarm points are missing columns: {missing}")
    return points.drop_duplicates(["alarm_event_id", *KEY_COLUMNS]).reset_index(drop=True)


def _alarm_events(path: str | Path) -> pd.DataFrame:
    events = _read_parquet(path)
    required = {
        "alarm_event_id",
        "plant_id",
        "device_no",
        "raise_time",
        "end_time",
    }
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError(f"Cleaned alarm events are missing columns: {missing}")
    if "classification" in events.columns:
        events = events[events["classification"].eq("complete")].copy()
    events["plant_id"] = pd.to_numeric(events["plant_id"], errors="coerce").astype("Int64")
    events["device_no"] = events["device_no"].astype(str).str.strip()
    events["raise_time"] = pd.to_datetime(events["raise_time"], errors="raise", utc=True)
    events["end_time"] = pd.to_datetime(events["end_time"], errors="raise", utc=True)
    return events.drop_duplicates("alarm_event_id").reset_index(drop=True)


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _point_summary(frame: pd.DataFrame) -> dict[str, Any]:
    total = int(len(frame))
    detected = int(frame["pvlof_detected"].sum())
    output_available = int(frame["pvlof_output_available"].sum())
    eligible = int(frame["pvlof_score_eligible"].sum())
    return {
        "alarm_points": total,
        "detected_points": detected,
        "missed_points": total - detected,
        "point_recall": _safe_ratio(detected, total),
        "pvlof_output_available_points": output_available,
        "pvlof_output_coverage": _safe_ratio(output_available, total),
        "pvlof_score_eligible_points": eligible,
        "pvlof_score_eligible_percent": _safe_ratio(eligible, total) * 100,
    }


def _event_summary(frame: pd.DataFrame) -> dict[str, Any]:
    total = int(len(frame))
    detected = int(frame["event_detected"].sum())
    delay = pd.to_numeric(
        frame.loc[frame["event_detected"], "detection_delay_minutes"],
        errors="coerce",
    ).dropna()
    return {
        "alarm_events": total,
        "detected_events": detected,
        "missed_events": total - detected,
        "event_recall": _safe_ratio(detected, total),
        "detection_delay_minutes": {
            "median": float(delay.median()) if len(delay) else None,
            "p90": float(delay.quantile(0.90)) if len(delay) else None,
            "maximum": float(delay.max()) if len(delay) else None,
        },
    }


def evaluate_alarm_recall(
    pvlof: str | Path,
    alarm_events: str | Path,
    alarm_points: str | Path,
    output_directory: str | Path,
    *,
    alert_column: str = "pvlof_alert",
    eligible_column: str = "pvlof_eligible",
) -> dict[str, Any]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    predictions = _prediction_points(
        pvlof, alert_column=alert_column, eligible_column=eligible_column
    )
    points = _alarm_points(alarm_points)
    events = _alarm_events(alarm_events)

    aligned = points.merge(predictions, on=KEY_COLUMNS, how="left", indicator=True)
    aligned["pvlof_output_available"] = aligned["_merge"].eq("both")
    aligned = aligned.drop(columns=["_merge"])
    aligned["pvlof_detected"] = aligned["pvlof_detected"].fillna(False).astype(bool)
    aligned["pvlof_score_eligible"] = (
        aligned["pvlof_score_eligible"].fillna(False).astype(bool)
    )
    aligned["pvlof_alerted_strings"] = (
        pd.to_numeric(aligned["pvlof_alerted_strings"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    aligned["pvlof_output_strings"] = (
        pd.to_numeric(aligned["pvlof_output_strings"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    grouped = aligned.groupby("alarm_event_id", observed=True)
    event_points = grouped.agg(
        alarm_points=("event_time", "size"),
        detected_points=("pvlof_detected", "sum"),
        output_available_points=("pvlof_output_available", "sum"),
        score_eligible_points=("pvlof_score_eligible", "sum"),
    ).reset_index()
    first_detection = (
        aligned[aligned["pvlof_detected"]]
        .groupby("alarm_event_id", observed=True)["event_time"]
        .min()
        .rename("first_detection_time")
        .reset_index()
    )
    event_results = events.merge(event_points, on="alarm_event_id", how="left")
    event_results = event_results.merge(first_detection, on="alarm_event_id", how="left")
    count_columns = [
        "alarm_points",
        "detected_points",
        "output_available_points",
        "score_eligible_points",
    ]
    for column in count_columns:
        event_results[column] = (
            pd.to_numeric(event_results[column], errors="coerce").fillna(0).astype(int)
        )
    event_results["event_detected"] = event_results["detected_points"].gt(0)
    event_results["event_point_recall"] = np.where(
        event_results["alarm_points"].gt(0),
        event_results["detected_points"] / event_results["alarm_points"],
        0.0,
    )
    event_results["detection_delay_minutes"] = (
        event_results["first_detection_time"] - event_results["raise_time"]
    ).dt.total_seconds() / 60.0

    per_plant: dict[str, Any] = {}
    for plant_id, plant_points in aligned.groupby("plant_id", observed=True):
        plant_events = event_results[event_results["plant_id"].eq(plant_id)]
        per_plant[str(int(plant_id))] = {
            "point_metrics": _point_summary(plant_points),
            "event_metrics": _event_summary(plant_events),
        }

    aligned.to_parquet(output / "alarm_point_recall.parquet", index=False)
    event_results.to_parquet(output / "alarm_event_recall.parquet", index=False)
    event_results.to_csv(
        output / "alarm_event_recall.csv",
        index=False,
        encoding="utf-8-sig",
    )
    report = {
        "pvlof": str(pvlof),
        "alarm_events": str(alarm_events),
        "alarm_points": str(alarm_points),
        "prediction_rule": f"any nonzero {alert_column} string at inverter-time point",
        "zero_current_rule_used": False,
        "point_metrics": _point_summary(aligned),
        "event_metrics": _event_summary(event_results),
        "per_plant": per_plant,
        "unsupported_metrics": {
            "precision": "not calculated because the evaluation set has no negative labels",
            "f1": "not calculated because precision is unavailable",
            "true_negative": "not defined for an all-positive evaluation set",
            "false_positive": "not defined for an all-positive evaluation set",
        },
        "outputs": {
            "alarm_point_recall": str(output / "alarm_point_recall.parquet"),
            "alarm_event_recall": str(output / "alarm_event_recall.parquet"),
            "alarm_event_recall_csv": str(output / "alarm_event_recall.csv"),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=_json_safe) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pvlof", required=True)
    parser.add_argument("--alarm-events", required=True)
    parser.add_argument("--alarm-points", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--alert-column", default="pvlof_alert")
    parser.add_argument("--eligible-column", default="pvlof_eligible")
    args = parser.parse_args()
    report = evaluate_alarm_recall(
        args.pvlof,
        args.alarm_events,
        args.alarm_points,
        args.output_directory,
        alert_column=args.alert_column,
        eligible_column=args.eligible_column,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_safe))


if __name__ == "__main__":
    main()
