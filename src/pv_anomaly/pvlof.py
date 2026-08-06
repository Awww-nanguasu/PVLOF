"""String-level photovoltaic local outlier factor (PVLOF) detection."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CURRENT_PATTERN = re.compile(r"^string_current_(\d{2})$")


@dataclass(frozen=True)
class PVLOFCalibration:
    """Saved settings and empirical alert threshold for PVLOF."""

    n_neighbors: int
    quantile: float
    threshold: float
    minimum_power_ratio: float
    maximum_power_ratio: float
    zero_current_threshold: float
    minimum_relative_drop: float
    minimum_strings: int
    minimum_consecutive: int
    expected_interval_minutes: int
    distance_floor: float
    eligible_status_codes: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["eligible_status_codes"] = list(self.eligible_status_codes)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PVLOFCalibration":
        values = dict(payload)
        values["eligible_status_codes"] = tuple(values["eligible_status_codes"])
        return cls(**values)


def _string_columns(frame: pd.DataFrame) -> list[tuple[int, str, str | None]]:
    columns: list[tuple[int, str, str | None]] = []
    for column in frame.columns:
        match = CURRENT_PATTERN.match(column)
        if not match:
            continue
        string_number = int(match.group(1))
        status_column = f"string_status_{string_number:02d}"
        columns.append(
            (
                string_number,
                column,
                status_column if status_column in frame.columns else None,
            )
        )
    columns.sort()
    if not columns:
        raise ValueError("No string_current_XX columns were found")
    return columns


def _lof_1d(
    values: np.ndarray,
    *,
    n_neighbors: int,
    distance_floor: float,
) -> np.ndarray:
    """Calculate standard LOF scores for a small one-dimensional peer group."""
    sample = np.asarray(values, dtype=np.float64)
    if sample.ndim != 1:
        raise ValueError("PVLOF peer values must be one-dimensional")
    if len(sample) < 2:
        return np.full(len(sample), np.nan, dtype=np.float64)
    neighbors = min(n_neighbors, len(sample) - 1)
    distances = np.abs(sample[:, None] - sample[None, :])
    np.fill_diagonal(distances, np.inf)
    neighbor_indexes = np.argsort(distances, axis=1, kind="stable")[:, :neighbors]
    neighbor_distances = np.take_along_axis(distances, neighbor_indexes, axis=1)
    k_distances = neighbor_distances[:, -1]
    reachability = np.maximum(
        neighbor_distances,
        k_distances[neighbor_indexes],
    )
    reachability = np.maximum(reachability, distance_floor)
    local_reachability_density = 1.0 / reachability.mean(axis=1)
    neighbor_density = local_reachability_density[neighbor_indexes].mean(axis=1)
    return neighbor_density / local_reachability_density


def prepare_pvlof_frame(
    frame: pd.DataFrame,
    *,
    n_neighbors: int = 5,
    minimum_power_ratio: float = 0.10,
    maximum_power_ratio: float = 1.10,
    zero_current_threshold: float = 0.0,
    minimum_strings: int = 4,
    distance_floor: float = 0.01,
    eligible_status_codes: tuple[int, ...] = (1, 4),
) -> pd.DataFrame:
    """Convert wide inverter rows to scored string-level daytime rows."""
    required = {"event_time", "device_no", "active_power", "rated_power"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing PVLOF source columns: {missing}")
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be positive")
    if minimum_strings < 2:
        raise ValueError("minimum_strings must be at least two")
    if minimum_power_ratio < 0 or maximum_power_ratio <= minimum_power_ratio:
        raise ValueError("power-ratio bounds are invalid")
    if distance_floor <= 0:
        raise ValueError("distance_floor must be positive")

    source = frame.copy()
    source["event_time"] = pd.to_datetime(source["event_time"], errors="raise", utc=True)
    source["device_no"] = source["device_no"].astype(str)
    strings = _string_columns(source)
    current_columns = [current for _, current, _ in strings]
    current = source[current_columns].apply(pd.to_numeric, errors="coerce").to_numpy(
        dtype=np.float64
    )

    status = np.full(current.shape, np.nan, dtype=np.float64)
    has_any_status = False
    for column_index, (_, _, status_column) in enumerate(strings):
        if status_column is None:
            continue
        has_any_status = True
        status[:, column_index] = pd.to_numeric(
            source[status_column], errors="coerce"
        ).to_numpy(dtype=np.float64)

    active_power = pd.to_numeric(source["active_power"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    rated_power = pd.to_numeric(source["rated_power"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    power_ratio = np.divide(
        active_power,
        rated_power,
        out=np.full(len(source), np.nan, dtype=np.float64),
        where=np.isfinite(rated_power) & (rated_power > 0),
    )
    row_eligible = (
        np.isfinite(active_power)
        & np.isfinite(rated_power)
        & (rated_power > 0)
        & np.isfinite(power_ratio)
        & (power_ratio >= minimum_power_ratio)
        & (power_ratio <= maximum_power_ratio)
    )
    if "status_code" in source.columns:
        device_status = pd.to_numeric(source["status_code"], errors="coerce").to_numpy()
        row_eligible &= np.isin(device_status, eligible_status_codes)
    if "main_string_count" in source.columns:
        main_count = pd.to_numeric(source["main_string_count"], errors="coerce").to_numpy()
        row_eligible &= main_count > 0
    if "valid_current_string_count" in source.columns:
        valid_count = pd.to_numeric(
            source["valid_current_string_count"], errors="coerce"
        ).to_numpy()
        row_eligible &= valid_count > 0

    finite_current = np.isfinite(current)
    configured = finite_current.copy()
    if has_any_status:
        configured &= np.isnan(status) | np.isin(status, (1, 2, 4))
    output_mask = row_eligible[:, None] & configured
    zero_alert = output_mask & (current <= zero_current_threshold)
    positive = output_mask & (current > zero_current_threshold)
    ratios = np.full(current.shape, np.nan, dtype=np.float64)
    scores = np.full(current.shape, np.nan, dtype=np.float64)
    score_eligible = np.zeros(current.shape, dtype=bool)

    required_peer_count = max(minimum_strings, n_neighbors + 1)
    for row_index in np.flatnonzero(row_eligible):
        peer_indexes = np.flatnonzero(positive[row_index])
        if len(peer_indexes) < required_peer_count:
            continue
        peer_currents = current[row_index, peer_indexes]
        peer_median = float(np.median(peer_currents))
        if not np.isfinite(peer_median) or peer_median <= zero_current_threshold:
            continue
        peer_ratios = peer_currents / peer_median
        ratios[row_index, peer_indexes] = peer_ratios
        scores[row_index, peer_indexes] = _lof_1d(
            peer_ratios,
            n_neighbors=n_neighbors,
            distance_floor=distance_floor,
        )
        score_eligible[row_index, peer_indexes] = True

    row_indexes, column_indexes = np.nonzero(output_mask)
    if not len(row_indexes):
        return pd.DataFrame(
            columns=[
                "event_time",
                "plant_id",
                "device_no",
                "string_no",
                "string_current",
                "string_status",
                "active_power",
                "rated_power",
                "active_power_ratio",
                "string_current_ratio",
                "pvlof_score",
                "pvlof_eligible",
                "zero_current_alert",
                "weak_low_current_label",
                "weak_zero_current_label",
            ]
        )

    result = pd.DataFrame(
        {
            "event_time": source["event_time"].to_numpy()[row_indexes],
            "device_no": source["device_no"].to_numpy()[row_indexes],
            "string_no": np.asarray([number for number, _, _ in strings], dtype=np.int16)[
                column_indexes
            ],
            "string_current": current[row_indexes, column_indexes].astype(np.float32),
            "string_status": status[row_indexes, column_indexes],
            "active_power": active_power[row_indexes].astype(np.float32),
            "rated_power": rated_power[row_indexes].astype(np.float32),
            "active_power_ratio": power_ratio[row_indexes].astype(np.float32),
            "string_current_ratio": ratios[row_indexes, column_indexes].astype(np.float32),
            "pvlof_score": scores[row_indexes, column_indexes].astype(np.float32),
            "pvlof_eligible": score_eligible[row_indexes, column_indexes],
            "zero_current_alert": zero_alert[row_indexes, column_indexes].astype(np.int8),
        }
    )
    if "plant_id" in source.columns:
        result.insert(1, "plant_id", source["plant_id"].to_numpy()[row_indexes])
    else:
        result.insert(1, "plant_id", pd.NA)
    status_values = pd.to_numeric(result["string_status"], errors="coerce")
    result["weak_low_current_label"] = status_values.eq(2)
    result["weak_zero_current_label"] = status_values.eq(4)
    return result.sort_values(["event_time", "device_no", "string_no"]).reset_index(drop=True)


def fit_pvlof_calibration(
    frame: pd.DataFrame,
    *,
    n_neighbors: int = 5,
    quantile: float = 0.995,
    minimum_power_ratio: float = 0.10,
    maximum_power_ratio: float = 1.10,
    zero_current_threshold: float = 0.0,
    minimum_relative_drop: float = 0.10,
    minimum_strings: int = 6,
    minimum_consecutive: int = 2,
    expected_interval_minutes: int = 5,
    distance_floor: float = 0.01,
    eligible_status_codes: tuple[int, ...] = (1, 4),
) -> tuple[PVLOFCalibration, dict[str, Any]]:
    """Fit an empirical PVLOF threshold using status-1, non-zero string rows."""
    if not 0.5 < quantile < 1:
        raise ValueError("quantile must be in (0.5, 1)")
    if not 0 <= minimum_relative_drop < 1:
        raise ValueError("minimum_relative_drop must be in [0, 1)")
    if minimum_consecutive < 1 or expected_interval_minutes < 1:
        raise ValueError("consecutive and interval settings must be positive")
    prepared = prepare_pvlof_frame(
        frame,
        n_neighbors=n_neighbors,
        minimum_power_ratio=minimum_power_ratio,
        maximum_power_ratio=maximum_power_ratio,
        zero_current_threshold=zero_current_threshold,
        minimum_strings=minimum_strings,
        distance_floor=distance_floor,
        eligible_status_codes=eligible_status_codes,
    )
    normal = prepared["pvlof_eligible"].astype(bool)
    if "string_status" in prepared.columns and prepared["string_status"].notna().any():
        normal &= pd.to_numeric(prepared["string_status"], errors="coerce").eq(1)
    values = pd.to_numeric(prepared.loc[normal, "pvlof_score"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("No eligible normal string scores remain for PVLOF calibration")
    threshold = float(np.quantile(values, quantile))
    calibration = PVLOFCalibration(
        n_neighbors=n_neighbors,
        quantile=quantile,
        threshold=threshold,
        minimum_power_ratio=minimum_power_ratio,
        maximum_power_ratio=maximum_power_ratio,
        zero_current_threshold=zero_current_threshold,
        minimum_relative_drop=minimum_relative_drop,
        minimum_strings=minimum_strings,
        minimum_consecutive=minimum_consecutive,
        expected_interval_minutes=expected_interval_minutes,
        distance_floor=distance_floor,
        eligible_status_codes=eligible_status_codes,
    )
    report = {
        "wide_samples": len(frame),
        "string_samples": len(prepared),
        "eligible_nonzero_samples": int(prepared["pvlof_eligible"].sum()),
        "normal_calibration_samples": len(values),
        "zero_current_samples": int(prepared["zero_current_alert"].sum()),
        "score_quantile": quantile,
        "threshold": threshold,
        "score_median": float(np.median(values)),
        "score_maximum": float(np.max(values)),
    }
    return calibration, report


def _add_consecutive_alerts(
    frame: pd.DataFrame,
    *,
    minimum_consecutive: int,
    expected_interval_minutes: int,
) -> pd.DataFrame:
    result = frame.sort_values(["device_no", "string_no", "event_time"]).reset_index(drop=True)
    raw = result["pvlof_raw_alert"].astype(bool).to_numpy()
    times = result["event_time"].to_numpy(dtype="datetime64[ns]")
    consecutive = np.zeros(len(result), dtype=np.int32)
    expected = np.timedelta64(expected_interval_minutes, "m")
    groups = result.groupby(["device_no", "string_no"], sort=False, observed=True).indices
    for positions in groups.values():
        indexes = np.asarray(positions, dtype=np.int64)
        previous_time: np.datetime64 | None = None
        run = 0
        for index in indexes:
            contiguous = previous_time is not None and times[index] - previous_time == expected
            if not contiguous:
                run = 0
            run = run + 1 if raw[index] else 0
            consecutive[index] = run
            previous_time = times[index]
    result["pvlof_consecutive"] = consecutive
    result["pvlof_alert"] = (
        raw & (consecutive >= minimum_consecutive)
    ).astype(np.int8)
    result["combined_alert"] = (
        result["zero_current_alert"].astype(bool) | result["pvlof_alert"].astype(bool)
    ).astype(np.int8)
    return result.sort_values(["event_time", "device_no", "string_no"]).reset_index(drop=True)


def apply_pvlof(frame: pd.DataFrame, calibration: PVLOFCalibration) -> pd.DataFrame:
    """Score wide device rows and apply directional and persistence alert rules."""
    prepared = prepare_pvlof_frame(
        frame,
        n_neighbors=calibration.n_neighbors,
        minimum_power_ratio=calibration.minimum_power_ratio,
        maximum_power_ratio=calibration.maximum_power_ratio,
        zero_current_threshold=calibration.zero_current_threshold,
        minimum_strings=calibration.minimum_strings,
        distance_floor=calibration.distance_floor,
        eligible_status_codes=calibration.eligible_status_codes,
    )
    score = pd.to_numeric(prepared["pvlof_score"], errors="coerce")
    ratio = pd.to_numeric(prepared["string_current_ratio"], errors="coerce")
    prepared["pvlof_raw_alert"] = (
        prepared["pvlof_eligible"].astype(bool)
        & score.ge(calibration.threshold)
        & ratio.le(1.0 - calibration.minimum_relative_drop)
    ).astype(np.int8)
    return _add_consecutive_alerts(
        prepared,
        minimum_consecutive=calibration.minimum_consecutive,
        expected_interval_minutes=calibration.expected_interval_minutes,
    )


def collapse_pvlof_events(
    frame: pd.DataFrame,
    *,
    alert_column: str = "combined_alert",
    expected_interval_minutes: int = 5,
) -> pd.DataFrame:
    """Collapse consecutive string alerts into reviewable events."""
    required = {"device_no", "string_no", "event_time", alert_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing PVLOF event columns: {missing}")
    alerts = frame[frame[alert_column].astype(bool)].copy()
    if alerts.empty:
        return pd.DataFrame(
            columns=[
                "event_id",
                "device_no",
                "string_no",
                "start_time",
                "end_time",
                "points",
                "duration_minutes",
            ]
        )
    alerts = alerts.sort_values(["device_no", "string_no", "event_time"]).reset_index(
        drop=True
    )
    expected = pd.Timedelta(minutes=expected_interval_minutes)
    new_event = (
        alerts["device_no"].ne(alerts["device_no"].shift())
        | alerts["string_no"].ne(alerts["string_no"].shift())
        | alerts["event_time"].sub(alerts["event_time"].shift()).ne(expected)
    )
    alerts["_event_number"] = new_event.cumsum()
    aggregations: dict[str, tuple[str, str]] = {
        "start_time": ("event_time", "min"),
        "end_time": ("event_time", "max"),
        "points": ("event_time", "size"),
        "zero_current_points": ("zero_current_alert", "sum"),
        "pvlof_points": ("pvlof_alert", "sum"),
        "minimum_current": ("string_current", "min"),
        "minimum_current_ratio": ("string_current_ratio", "min"),
        "maximum_pvlof_score": ("pvlof_score", "max"),
    }
    events = (
        alerts.groupby(["device_no", "string_no", "_event_number"], observed=True)
        .agg(**aggregations)
        .reset_index()
        .drop(columns="_event_number")
    )
    events["duration_minutes"] = (
        events["end_time"] - events["start_time"]
    ).dt.total_seconds() / 60 + expected_interval_minutes
    events.insert(0, "event_id", [f"pvlof-{index:06d}" for index in range(1, len(events) + 1)])
    return events


def save_calibration(calibration: PVLOFCalibration, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(calibration.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_calibration(path: str | Path) -> PVLOFCalibration:
    return PVLOFCalibration.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
