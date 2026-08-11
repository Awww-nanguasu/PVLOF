"""Evaluate production PVLOF predictions against cleaned inverter alarms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pv_anomaly.ewma_audit import binary_metrics


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
    frames: list[pd.DataFrame] = []
    for file in files:
        frame = pd.read_parquet(file, columns=columns)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _normalize_keys(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    missing = sorted(set(KEY_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")
    result = frame.copy()
    result["plant_id"] = pd.to_numeric(result["plant_id"], errors="coerce").astype("Int64")
    result["device_no"] = result["device_no"].astype(str).str.strip()
    result["event_time"] = pd.to_datetime(result["event_time"], errors="raise", utc=True)
    return result


def _prediction_points(path: str | Path) -> tuple[pd.DataFrame, dict[str, int]]:
    columns = [
        "plant_id",
        "device_no",
        "event_time",
        "pvlof_alert",
        "pvlof_eligible",
    ]
    source = _normalize_keys(_read_parquet(path, columns), name="PVLOF output")
    source["pvlof_alert"] = source["pvlof_alert"].astype(bool)
    if "pvlof_eligible" not in source.columns:
        source["pvlof_eligible"] = False
    source["pvlof_eligible"] = source["pvlof_eligible"].astype(bool)
    grouped = (
        source.groupby(KEY_COLUMNS, observed=True, as_index=False)
        .agg(
            predicted_alert=("pvlof_alert", "max"),
            score_eligible=("pvlof_eligible", "max"),
        )
        .sort_values(KEY_COLUMNS)
        .reset_index(drop=True)
    )
    return grouped, {
        "rows_read": int(len(source)),
        "device_time_points": int(len(grouped)),
        "alert_points": int(grouped["predicted_alert"].sum()),
        "eligible_device_time_points": int(grouped["score_eligible"].sum()),
    }


def _label_points(path: str | Path) -> pd.DataFrame:
    labels = _normalize_keys(_read_parquet(path), name="cleaned alarm points")
    return labels[KEY_COLUMNS].drop_duplicates().assign(label=True)


def _label_events(path: str | Path) -> pd.DataFrame:
    labels = _read_parquet(path)
    required = {
        "alarm_event_id",
        "plant_id",
        "device_no",
        "effective_start_time",
        "effective_end_time",
    }
    missing = sorted(required - set(labels.columns))
    if missing:
        raise ValueError(f"Cleaned alarm events are missing columns: {missing}")
    if "classification" in labels.columns:
        labels = labels[labels["classification"].eq("complete")].copy()
    else:
        labels = labels.copy()
    labels["plant_id"] = pd.to_numeric(labels["plant_id"], errors="coerce").astype("Int64")
    labels["device_no"] = labels["device_no"].astype(str).str.strip()
    labels["effective_start_time"] = pd.to_datetime(
        labels["effective_start_time"], errors="raise", utc=True
    )
    labels["effective_end_time"] = pd.to_datetime(
        labels["effective_end_time"], errors="raise", utc=True
    )
    return labels.reset_index(drop=True)


def _point_evaluation(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    aligned = predictions.merge(labels, on=KEY_COLUMNS, how="left", indicator=True)
    aligned["label"] = aligned["_merge"].eq("both")
    aligned = aligned.drop(columns=["_merge"])
    all_metrics = binary_metrics(aligned["label"], aligned["predicted_alert"])
    eligible = aligned["score_eligible"].astype(bool)
    eligible_metrics = binary_metrics(
        aligned.loc[eligible, "label"], aligned.loc[eligible, "predicted_alert"]
    )
    prediction_keys = aligned.set_index(KEY_COLUMNS).index
    covered = labels.set_index(KEY_COLUMNS).index.isin(prediction_keys)
    report = {
        "all_prediction_points": all_metrics,
        "score_eligible_points": eligible_metrics,
        "label_points_total": int(len(labels)),
        "label_points_covered": int(covered.sum()),
        "label_points_uncovered": int((~covered).sum()),
        "label_coverage_percent": float(covered.mean() * 100) if len(covered) else 0.0,
    }
    return report, aligned


def _collapse_prediction_events(
    points: pd.DataFrame,
    *,
    interval_minutes: int,
) -> pd.DataFrame:
    active = points[points["predicted_alert"].astype(bool)].copy()
    if active.empty:
        return pd.DataFrame(
            columns=[
                "predicted_event_id",
                "plant_id",
                "device_no",
                "event_start_time",
                "event_end_time",
                "points",
            ]
        )
    active = active.sort_values(["plant_id", "device_no", "event_time"])
    records: list[dict[str, Any]] = []
    expected = pd.Timedelta(minutes=interval_minutes)
    event_id = 0
    for (plant_id, device_no), group in active.groupby(
        ["plant_id", "device_no"], observed=True
    ):
        group = group.sort_values("event_time")
        start = previous = None
        count = 0
        for timestamp in group["event_time"]:
            if start is None or timestamp - previous != expected:
                if start is not None:
                    records.append(
                        {
                            "predicted_event_id": event_id,
                            "plant_id": plant_id,
                            "device_no": device_no,
                            "event_start_time": start,
                            "event_end_time": previous,
                            "points": count,
                        }
                    )
                    event_id += 1
                start = timestamp
                count = 0
            previous = timestamp
            count += 1
        records.append(
            {
                "predicted_event_id": event_id,
                "plant_id": plant_id,
                "device_no": device_no,
                "event_start_time": start,
                "event_end_time": previous,
                "points": count,
            }
        )
        event_id += 1
    return pd.DataFrame(records)


def _event_evaluation(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    interval_minutes: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    predicted_events = _collapse_prediction_events(
        predictions,
        interval_minutes=interval_minutes,
    )
    matches: list[dict[str, Any]] = []
    label_hit = np.zeros(len(labels), dtype=bool)
    predicted_hit = np.zeros(len(predicted_events), dtype=bool)
    for label_index, label in labels.iterrows():
        if predicted_events.empty:
            continue
        candidates = predicted_events[
            predicted_events["plant_id"].eq(label["plant_id"])
            & predicted_events["device_no"].eq(label["device_no"])
            & (predicted_events["event_end_time"] >= label["effective_start_time"])
            & (predicted_events["event_start_time"] <= label["effective_end_time"])
        ]
        for predicted_index, predicted in candidates.iterrows():
            label_hit[label_index] = True
            predicted_hit[predicted_index] = True
            matches.append(
                {
                    "alarm_event_id": label["alarm_event_id"],
                    "predicted_event_id": predicted["predicted_event_id"],
                    "plant_id": label["plant_id"],
                    "device_no": label["device_no"],
                    "alarm_start_time": label["effective_start_time"],
                    "alarm_end_time": label["effective_end_time"],
                    "prediction_start_time": predicted["event_start_time"],
                    "prediction_end_time": predicted["event_end_time"],
                }
            )
    true_positive = int(predicted_hit.sum())
    precision = true_positive / len(predicted_events) if len(predicted_events) else 0.0
    recall = int(label_hit.sum()) / len(labels) if len(labels) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    report = {
        "predicted_events": int(len(predicted_events)),
        "label_events": int(len(labels)),
        "predicted_events_hit": true_positive,
        "label_events_hit": int(label_hit.sum()),
        "false_positive_events": int((~predicted_hit).sum()),
        "missed_label_events": int((~label_hit).sum()),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    return report, predicted_events, pd.DataFrame(matches)


def evaluate(
    pvlof: str | Path,
    labels_events: str | Path,
    labels_points: str | Path,
    output_directory: str | Path,
    *,
    interval_minutes: int = 5,
) -> dict[str, Any]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    predictions, prediction_report = _prediction_points(pvlof)
    point_labels = _label_points(labels_points)
    event_labels = _label_events(labels_events)
    point_report, aligned = _point_evaluation(predictions, point_labels)
    event_report, predicted_events, matches = _event_evaluation(
        predictions,
        event_labels,
        interval_minutes=interval_minutes,
    )
    aligned.to_parquet(output / "aligned_points.parquet", index=False)
    predicted_events.to_parquet(output / "predicted_events.parquet", index=False)
    matches.to_csv(output / "event_matches.csv", index=False, encoding="utf-8-sig")
    report = {
        "pvlof": str(pvlof),
        "labels_events": str(labels_events),
        "labels_points": str(labels_points),
        "interval_minutes": interval_minutes,
        "prediction": prediction_report,
        "point_metrics": point_report,
        "event_metrics": event_report,
        "outputs": {
            "aligned_points": str(output / "aligned_points.parquet"),
            "predicted_events": str(output / "predicted_events.parquet"),
            "event_matches": str(output / "event_matches.csv"),
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
    parser.add_argument("--labels-events", required=True)
    parser.add_argument("--labels-points", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--interval-minutes", type=int, default=5)
    args = parser.parse_args()
    report = evaluate(
        args.pvlof,
        args.labels_events,
        args.labels_points,
        args.output_directory,
        interval_minutes=args.interval_minutes,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_safe))


if __name__ == "__main__":
    main()
