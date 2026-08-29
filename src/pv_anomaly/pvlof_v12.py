"""Forecast-GHI conditioned virtual irradiance for PVLOF v1.2.

The existing peer-inverter virtual irradiance remains the primary signal.
Forecast GHI is mapped onto the same dimensionless scale per plant and is
used as a bounded fallback/support signal. Missing weather therefore falls
back exactly to the peer-only context.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pv_anomaly.pvlof_v2 import (
    PVLOFV2Calibration,
    _normalise_source,
    _virtual_context,
)


KEYS = ["plant_id", "device_no", "event_time"]


@dataclass(frozen=True)
class PVLOFV12WeatherCalibration:
    version: str = "pvlof-v1.2-forecast-ghi"
    candidate_source_offsets_minutes: list[int] = field(default_factory=lambda: [0, 15])
    tolerance_minutes: int = 15
    minimum_forecast_ghi: float = 5.0
    minimum_mapping_samples: int = 100
    mapping_bins: int = 20
    weak_peer_count: int = 5
    strong_peer_count: int = 10
    weak_virtual_weight: float = 0.65
    strong_virtual_weight: float = 0.85
    maximum_forecast_correction: float = 0.20
    plants: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PVLOFV12WeatherCalibration":
        return cls(**dict(payload))


def _normalise_weather(weather: pd.DataFrame) -> pd.DataFrame:
    required = {"time", "plant_id", "forecast_ghi"}
    missing = sorted(required - set(weather.columns))
    if missing:
        raise ValueError(f"Weather data is missing columns: {missing}")
    result = weather[["time", "plant_id", "forecast_ghi"]].copy()
    result["time"] = pd.to_datetime(result["time"], errors="raise", utc=True)
    result["plant_id"] = result["plant_id"].astype(str)
    result["forecast_ghi"] = pd.to_numeric(result["forecast_ghi"], errors="coerce")
    result = result.drop_duplicates(["plant_id", "time"], keep="last")
    return result.sort_values(["plant_id", "time"]).reset_index(drop=True)


def _raw_context(frame: pd.DataFrame, base: PVLOFV2Calibration) -> pd.DataFrame:
    source, strings = _normalise_source(frame)
    virtual, peers, _, _ = _virtual_context(
        source,
        strings,
        base.device_scales,
        minimum_peer_devices=base.minimum_peer_devices,
        zero_current_threshold=base.zero_current_threshold,
        configured_strings=base.configured_strings,
    )
    result = source[KEYS].copy()
    result["raw_virtual_irradiance"] = virtual
    result["raw_peer_device_count"] = peers.astype(int)
    return result


def _align_one_plant(
    targets: pd.DataFrame,
    weather: pd.DataFrame,
    *,
    source_offset_minutes: int,
    tolerance_minutes: int,
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for plant, left_group in targets.groupby("plant_id", observed=True, sort=False):
        left = left_group.copy().sort_values("event_time")
        right = weather[weather["plant_id"].eq(str(plant))].copy()
        if right.empty:
            left["forecast_source_time"] = pd.Series(
                pd.NaT, index=left.index, dtype="datetime64[ns, UTC]"
            )
            left["forecast_ghi"] = np.nan
            pieces.append(left)
            continue
        right["forecast_lookup_time"] = right["time"] - pd.Timedelta(
            minutes=source_offset_minutes
        )
        right = right.rename(columns={"time": "forecast_source_time"})
        aligned = pd.merge_asof(
            left,
            right[["forecast_lookup_time", "forecast_source_time", "forecast_ghi"]]
            .sort_values("forecast_lookup_time"),
            left_on="event_time",
            right_on="forecast_lookup_time",
            direction="backward",
            tolerance=pd.Timedelta(minutes=tolerance_minutes),
            allow_exact_matches=True,
        ).drop(columns="forecast_lookup_time")
        pieces.append(aligned)
    if not pieces:
        return targets.assign(forecast_source_time=pd.NaT, forecast_ghi=np.nan)
    result = pd.concat(pieces, ignore_index=True)
    sort_columns = [column for column in KEYS if column in result.columns]
    return result.sort_values(sort_columns).reset_index(drop=True)


def _mapping_anchors(
    forecast: np.ndarray, virtual: np.ndarray, bins: int
) -> tuple[list[float], list[float]]:
    table = pd.DataFrame({"forecast": forecast, "virtual": virtual}).dropna()
    table = table[(table["forecast"] >= 0) & (table["virtual"] >= 0)]
    if table.empty:
        return [], []
    quantiles = np.linspace(0, 1, min(bins, len(table)) + 1)
    edges = np.unique(np.quantile(table["forecast"], quantiles))
    if len(edges) < 2:
        return [float(edges[0])], [float(table["virtual"].median())]
    groups = pd.cut(table["forecast"], bins=edges, include_lowest=True, duplicates="drop")
    anchors = table.groupby(groups, observed=True).median().dropna()
    x = anchors["forecast"].to_numpy(float)
    y = np.maximum.accumulate(anchors["virtual"].to_numpy(float))
    if len(x) and x[0] > 0:
        x = np.insert(x, 0, 0.0)
        y = np.insert(y, 0, 0.0)
    return x.tolist(), y.tolist()


def fit_weather_calibration(
    frame: pd.DataFrame,
    base: PVLOFV2Calibration,
    weather: pd.DataFrame,
    *,
    candidate_source_offsets_minutes: tuple[int, ...] = (0, 15),
    tolerance_minutes: int = 15,
    minimum_forecast_ghi: float = 5.0,
    minimum_mapping_samples: int = 100,
    mapping_bins: int = 20,
    weak_peer_count: int = 5,
    strong_peer_count: int = 10,
    weak_virtual_weight: float = 0.65,
    strong_virtual_weight: float = 0.85,
    maximum_forecast_correction: float = 0.20,
) -> tuple[PVLOFV12WeatherCalibration, dict[str, Any]]:
    context = _raw_context(frame, base)
    weather = _normalise_weather(weather)
    plant_time = (
        context.groupby(["plant_id", "event_time"], observed=True)["raw_virtual_irradiance"]
        .median().reset_index()
    )
    plants: dict[str, dict[str, Any]] = {}
    reports: dict[str, Any] = {}
    for plant, targets in plant_time.groupby("plant_id", observed=True):
        candidates: list[dict[str, Any]] = []
        for offset in candidate_source_offsets_minutes:
            aligned = _align_one_plant(
                targets,
                weather,
                source_offset_minutes=offset,
                tolerance_minutes=tolerance_minutes,
            )
            valid = (
                aligned["raw_virtual_irradiance"].notna()
                & aligned["forecast_ghi"].ge(minimum_forecast_ghi)
            )
            sample = aligned.loc[valid, ["forecast_ghi", "raw_virtual_irradiance"]]
            correlation = (
                float(sample.corr(method="spearman").iloc[0, 1])
                if len(sample) >= 3 else np.nan
            )
            candidates.append(
                {"source_offset_minutes": int(offset), "samples": len(sample), "spearman": correlation}
            )
        eligible = [
            item for item in candidates
            if item["samples"] >= minimum_mapping_samples and np.isfinite(item["spearman"])
        ]
        if not eligible:
            reports[str(plant)] = {"usable": False, "candidates": candidates}
            continue
        selected = max(eligible, key=lambda item: (item["spearman"], item["samples"]))
        aligned = _align_one_plant(
            targets,
            weather,
            source_offset_minutes=selected["source_offset_minutes"],
            tolerance_minutes=tolerance_minutes,
        )
        valid = aligned["raw_virtual_irradiance"].notna() & aligned["forecast_ghi"].ge(
            minimum_forecast_ghi
        )
        x, y = _mapping_anchors(
            aligned.loc[valid, "forecast_ghi"].to_numpy(float),
            aligned.loc[valid, "raw_virtual_irradiance"].to_numpy(float),
            mapping_bins,
        )
        if not x:
            reports[str(plant)] = {"usable": False, "candidates": candidates}
            continue
        record = {
            "source_offset_minutes": selected["source_offset_minutes"],
            "samples": int(valid.sum()),
            "spearman": selected["spearman"],
            "forecast_anchors": x,
            "virtual_anchors": y,
        }
        plants[str(plant)] = record
        reports[str(plant)] = {"usable": True, "selected": record, "candidates": candidates}
    calibration = PVLOFV12WeatherCalibration(
        candidate_source_offsets_minutes=list(candidate_source_offsets_minutes),
        tolerance_minutes=tolerance_minutes,
        minimum_forecast_ghi=minimum_forecast_ghi,
        minimum_mapping_samples=minimum_mapping_samples,
        mapping_bins=mapping_bins,
        weak_peer_count=weak_peer_count,
        strong_peer_count=strong_peer_count,
        weak_virtual_weight=weak_virtual_weight,
        strong_virtual_weight=strong_virtual_weight,
        maximum_forecast_correction=maximum_forecast_correction,
        plants=plants,
    )
    return calibration, {"plants": reports, "usable_plants": len(plants)}


def build_conditioned_virtual_context(
    frame: pd.DataFrame,
    base: PVLOFV2Calibration,
    weather_calibration: PVLOFV12WeatherCalibration,
    weather: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = _raw_context(frame, base)
    weather = _normalise_weather(weather)
    pieces: list[pd.DataFrame] = []
    for plant, group in raw.groupby("plant_id", observed=True, sort=False):
        record = weather_calibration.plants.get(str(plant))
        if record is None:
            aligned = group.copy()
            aligned["forecast_source_time"] = pd.Series(
                pd.NaT, index=aligned.index, dtype="datetime64[ns, UTC]"
            )
            aligned["forecast_ghi"] = np.nan
        else:
            aligned = _align_one_plant(
                group,
                weather,
                source_offset_minutes=int(record["source_offset_minutes"]),
                tolerance_minutes=weather_calibration.tolerance_minutes,
            )
        pieces.append(aligned)
    result = pd.concat(pieces, ignore_index=True).sort_values(KEYS).reset_index(drop=True)
    result["forecast_virtual_irradiance"] = np.nan
    for plant, indexes in result.groupby("plant_id", observed=True).groups.items():
        record = weather_calibration.plants.get(str(plant))
        if record is None:
            continue
        ghi = pd.to_numeric(result.loc[indexes, "forecast_ghi"], errors="coerce").to_numpy(float)
        valid = np.isfinite(ghi) & (ghi >= weather_calibration.minimum_forecast_ghi)
        mapped = np.full(len(ghi), np.nan)
        mapped[valid] = np.interp(
            ghi[valid], record["forecast_anchors"], record["virtual_anchors"]
        )
        result.loc[indexes, "forecast_virtual_irradiance"] = mapped
    raw_v = result["raw_virtual_irradiance"].to_numpy(float)
    forecast_v = result["forecast_virtual_irradiance"].to_numpy(float)
    peers = result["raw_peer_device_count"].to_numpy(float)
    raw_ok = np.isfinite(raw_v)
    forecast_ok = np.isfinite(forecast_v)
    capped = forecast_v.copy()
    both = raw_ok & forecast_ok
    cap = weather_calibration.maximum_forecast_correction
    capped[both] = np.clip(capped[both], raw_v[both] * (1 - cap), raw_v[both] * (1 + cap))
    span = max(weather_calibration.strong_peer_count - weather_calibration.weak_peer_count, 1)
    fraction = np.clip((peers - weather_calibration.weak_peer_count) / span, 0, 1)
    weight = weather_calibration.weak_virtual_weight + fraction * (
        weather_calibration.strong_virtual_weight - weather_calibration.weak_virtual_weight
    )
    conditioned = np.full(len(result), np.nan)
    conditioned[raw_ok & ~forecast_ok] = raw_v[raw_ok & ~forecast_ok]
    conditioned[~raw_ok & forecast_ok] = capped[~raw_ok & forecast_ok]
    conditioned[both] = weight[both] * raw_v[both] + (1 - weight[both]) * capped[both]
    result["forecast_virtual_capped"] = capped
    result["virtual_peer_weight"] = np.where(both, weight, np.where(raw_ok, 1.0, 0.0))
    result["conditioned_virtual_irradiance"] = conditioned
    result["conditioned_peer_count"] = np.where(
        forecast_ok,
        np.maximum(peers, base.minimum_peer_devices),
        peers,
    )
    result["forecast_available"] = forecast_ok
    result["forecast_offset_minutes"] = (
        result["forecast_source_time"] - result["event_time"]
    ).dt.total_seconds() / 60
    report = {
        "rows": len(result),
        "forecast_available_rows": int(forecast_ok.sum()),
        "forecast_available_percent": float(forecast_ok.mean() * 100) if len(result) else 0.0,
        "forecast_fallback_rows": int((~raw_ok & forecast_ok).sum()),
        "peer_only_rows": int((raw_ok & ~forecast_ok).sum()),
        "unavailable_rows": int((~raw_ok & ~forecast_ok).sum()),
    }
    return result, report


def save_weather_calibration(
    calibration: PVLOFV12WeatherCalibration, path: str | Path
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(calibration.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_weather_calibration(path: str | Path) -> PVLOFV12WeatherCalibration:
    return PVLOFV12WeatherCalibration.from_dict(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


__all__ = [
    "PVLOFV12WeatherCalibration",
    "build_conditioned_virtual_context",
    "fit_weather_calibration",
    "load_weather_calibration",
    "save_weather_calibration",
]
