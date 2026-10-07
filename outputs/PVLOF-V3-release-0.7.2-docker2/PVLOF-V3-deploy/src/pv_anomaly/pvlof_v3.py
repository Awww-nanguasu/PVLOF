"""Frozen PVLOF-V3 production entry point.

PVLOF-V3 is the production name of the validated
``lowest_core_expansion_v5`` detector.  This module is deliberately an
orchestrator: it loads the frozen calibration files, accepts wide inverter
current rows, executes the complete detector chain, and exposes canonical
``pvlof_v3_*`` state columns plus customer-facing alarm views.

The current implementation is batch/state-reconstruction based.  Callers must
provide a complete, ordered history for the period being evaluated.  A future
streaming adapter may persist the per-string state and call the same scoring
components incrementally.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from pv_anomaly.pvlof_alarm_views import build_pvlof_alarm_views
from pv_anomaly.pvlof_v12 import (
    PVLOFV12WeatherCalibration,
    build_conditioned_virtual_context,
    load_weather_calibration,
)
from pv_anomaly.pvlof_v16 import (
    PVLOFV16MemoryConfig,
    apply_confirmed_anomaly_memory_v16,
    load_memory_config,
)
from pv_anomaly.pvlof_v17_improved import (
    PVLOFV17ImprovedConfig,
    apply_pvlof_v17_improved,
    load_config,
)
from pv_anomaly.pvlof_v2 import (
    PVLOFV2Calibration,
    apply_pvlof_v2,
    load_calibration,
)
from pv_anomaly.pvlof_v2_hier import (
    PVLOFV2HierCalibration,
    apply_hierarchical_isolated,
    load_hier_calibration,
)


PVLOF_V3_VERSION = "pvlof-v3"
PVLOF_V3_ALGORITHM_VARIANT = "lowest_core_expansion_v5"
PVLOF_V3_PREFIX = "pvlof_v3"

DEVICE_TIME_KEYS = ["plant_id", "device_no", "event_time"]
CURRENT_PATTERN = re.compile(r"^string_current_(\d{2})$")
CONTEXT_METADATA_COLUMNS = [
    "plant_id",
    "device_no",
    "event_time",
    "raw_virtual_irradiance",
    "raw_peer_device_count",
    "forecast_ghi",
    "forecast_source_time",
    "forecast_virtual_irradiance",
    "forecast_virtual_capped",
    "virtual_peer_weight",
    "conditioned_virtual_irradiance",
    "conditioned_peer_count",
    "forecast_available",
    "forecast_offset_minutes",
]

_STATE_ALIASES = {
    "raw_anomaly": "pvlof_v17_improved_raw_anomaly",
    # The canonical V3 alert intentionally excludes the legacy V1.6 final
    # union.  It is the independently confirmed combined-candidate state.
    "alert": "pvlof_v17_improved_combined_memory_alert",
    "combined_memory_alert": "pvlof_v17_improved_combined_memory_alert",
    "valid_point": "pvlof_v17_improved_valid_point",
    "memory_active": "pvlof_v17_improved_memory_active",
    "memory_clear_code": "pvlof_v17_improved_memory_clear_code",
    "unknown_streak": "pvlof_v17_improved_unknown_streak",
    "memory_suspended": "pvlof_v17_improved_memory_suspended",
    "memory_resumed_alert": "pvlof_v17_improved_memory_resumed_alert",
}

_DIAGNOSTIC_ALIASES = {
    "v16_raw_candidate": "pvlof_v16_raw_anomaly",
    "segmentation_raw_candidate": (
        "pvlof_v17_improved_segmentation_raw_candidate"
    ),
    "lowest_core_raw_candidate": (
        "pvlof_v17_improved_segmentation_lowest_core_raw_candidate"
    ),
    "validated_expansion_raw_candidate": (
        "pvlof_v17_improved_segmentation_validated_expansion_raw_candidate"
    ),
    "v16_middle_group_veto": "pvlof_v17_improved_v16_middle_group_veto",
}


@dataclass(frozen=True)
class PVLOFV3ModelPaths:
    """Locations of the five frozen artifacts required by PVLOF-V3."""

    base_calibration: Path
    weather_calibration: Path
    hierarchy_calibration: Path
    memory_config: Path
    detector_config: Path

    @classmethod
    def from_root(cls, root: str | Path) -> "PVLOFV3ModelPaths":
        base = Path(root)
        return cls(
            base_calibration=(
                base / "artifacts/models/pvlof_v16/base_calibration.json"
            ),
            weather_calibration=(
                base / "artifacts/models/pvlof_v12_3point/weather_calibration.json"
            ),
            hierarchy_calibration=(
                base / "artifacts/models/pvlof_v12_3point/hier_calibration.json"
            ),
            memory_config=base / "artifacts/models/pvlof_v16/memory_config.json",
            detector_config=(
                base / "configs/pvlof_v17_improved_v5_lowest_core_expansion.json"
            ),
        )

    def as_dict(self) -> dict[str, Path]:
        return {
            "base_calibration": self.base_calibration,
            "weather_calibration": self.weather_calibration,
            "hierarchy_calibration": self.hierarchy_calibration,
            "memory_config": self.memory_config,
            "detector_config": self.detector_config,
        }


@dataclass(frozen=True)
class PVLOFV3Result:
    """In-memory production result; no files or databases are written."""

    full_points: pd.DataFrame
    string_states: pd.DataFrame
    string_events: pd.DataFrame
    final_alert_points: pd.DataFrame
    raw_candidate_points: pd.DataFrame
    confirmed_events: pd.DataFrame
    candidate_segments: pd.DataFrame
    report: dict[str, Any]

    def customer_outputs(self) -> dict[str, pd.DataFrame]:
        """Return only the two tables intended for customer-facing use."""

        return {
            "final_alert_points": self.final_alert_points,
            "confirmed_events": self.confirmed_events,
        }


@dataclass(frozen=True)
class PVLOFV3Detector:
    """Loaded, reusable PVLOF-V3 detector."""

    base_calibration: PVLOFV2Calibration
    weather_calibration: PVLOFV12WeatherCalibration
    hierarchy_calibration: PVLOFV2HierCalibration
    memory_config: PVLOFV16MemoryConfig
    detector_config: PVLOFV17ImprovedConfig

    def __post_init__(self) -> None:
        _validate_frozen_contract(self)

    @classmethod
    def load(cls, paths: PVLOFV3ModelPaths) -> "PVLOFV3Detector":
        missing = [
            f"{name}: {path}"
            for name, path in paths.as_dict().items()
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "PVLOF-V3 frozen artifacts are missing: " + "; ".join(missing)
            )
        return cls(
            base_calibration=load_calibration(paths.base_calibration),
            weather_calibration=load_weather_calibration(
                paths.weather_calibration
            ),
            hierarchy_calibration=load_hier_calibration(
                paths.hierarchy_calibration
            ),
            memory_config=load_memory_config(paths.memory_config),
            detector_config=load_config(paths.detector_config),
        )

    @classmethod
    def from_root(cls, root: str | Path) -> "PVLOFV3Detector":
        return cls.load(PVLOFV3ModelPaths.from_root(root))

    @property
    def supported_device_keys(self) -> frozenset[str]:
        return frozenset(self.base_calibration.configured_strings)

    @property
    def supported_plant_ids(self) -> tuple[str, ...]:
        plants = {
            key.split("|", 1)[0]
            for key in self.supported_device_keys
            if "|" in key
        }
        return tuple(sorted(plants))

    def score_points(
        self,
        frame: pd.DataFrame,
        *,
        weather: pd.DataFrame | None = None,
        timezone: str = "Asia/Shanghai",
        strict_calibration: bool = True,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Run the complete frozen detector and return string-level points.

        ``frame`` uses the existing wide input contract: one row per
        plant/device/time and columns named ``string_current_XX``. Event times
        must represent UTC instants (timezone-aware values are preferred).
        Weather is optional; when absent, V1.2 falls back to peer-only virtual
        irradiance.
        """

        source = _prepare_source(
            frame,
            supported_device_keys=self.supported_device_keys,
            strict_calibration=strict_calibration,
        )
        context, context_report = build_conditioned_virtual_context(
            source,
            self.base_calibration,
            self.weather_calibration,
            weather,
        )
        scored = apply_pvlof_v2(
            source,
            self.base_calibration,
            virtual_override=context,
        )
        scored = apply_hierarchical_isolated(
            scored,
            self.hierarchy_calibration,
            timezone=timezone,
        )
        metadata = context[
            [column for column in CONTEXT_METADATA_COLUMNS if column in context]
        ].copy()
        scored = scored.merge(
            metadata,
            on=DEVICE_TIME_KEYS,
            how="left",
            validate="many_to_one",
        )
        v16 = apply_confirmed_anomaly_memory_v16(
            scored,
            self.memory_config,
        )
        full_points = apply_pvlof_v17_improved(
            v16,
            self.detector_config,
        )
        full_points = _add_canonical_aliases(full_points)
        report = {
            "version": PVLOF_V3_VERSION,
            "internal_version": self.detector_config.version,
            "algorithm_variant": self.detector_config.algorithm_variant,
            "timezone": timezone,
            "interval_minutes": self.detector_config.expected_interval_minutes,
            "input_rows": int(len(source)),
            "input_plants": int(source["plant_id"].nunique()),
            "input_devices": int(
                source[["plant_id", "device_no"]].drop_duplicates().shape[0]
            ),
            "input_device_time_rows": int(
                source[DEVICE_TIME_KEYS].drop_duplicates().shape[0]
            ),
            "scored_string_rows": int(len(full_points)),
            "raw_candidate_string_points": int(
                full_points[f"{PVLOF_V3_PREFIX}_raw_anomaly"]
                .fillna(False)
                .astype(bool)
                .sum()
            ),
            "formal_alert_string_points": int(
                full_points[f"{PVLOF_V3_PREFIX}_alert"]
                .fillna(False)
                .astype(bool)
                .sum()
            ),
            "context": context_report,
            "canonical_columns": {
                "raw_candidate": f"{PVLOF_V3_PREFIX}_raw_anomaly",
                "formal_alert": f"{PVLOF_V3_PREFIX}_alert",
            },
        }
        return full_points, report

    def run(
        self,
        frame: pd.DataFrame,
        *,
        weather: pd.DataFrame | None = None,
        timezone: str = "Asia/Shanghai",
        strict_calibration: bool = True,
    ) -> PVLOFV3Result:
        """Run PVLOF-V3 and build current-member and event views."""

        full_points, report = self.score_points(
            frame,
            weather=weather,
            timezone=timezone,
            strict_calibration=strict_calibration,
        )
        views = build_pvlof_alarm_views(
            full_points,
            prefix=PVLOF_V3_PREFIX,
            interval_minutes=self.detector_config.expected_interval_minutes,
            entry_consecutive=self.detector_config.entry_consecutive,
            timezone=timezone,
        )
        report = {
            **report,
            "outputs": {name: int(len(value)) for name, value in views.items()},
            "customer_outputs": ["final_alert_points", "confirmed_events"],
        }
        return PVLOFV3Result(
            full_points=full_points,
            string_states=views["string_states"],
            string_events=views["string_events"],
            final_alert_points=views["final_alert_points"],
            raw_candidate_points=views["raw_candidate_points"],
            confirmed_events=views["confirmed_events"],
            candidate_segments=views["candidate_segments"],
            report=report,
        )


def _validate_frozen_contract(detector: PVLOFV3Detector) -> None:
    config = detector.detector_config
    errors: list[str] = []
    if config.algorithm_variant != PVLOF_V3_ALGORITHM_VARIANT:
        errors.append(
            "detector algorithm_variant must be "
            f"{PVLOF_V3_ALGORITHM_VARIANT!r}, got {config.algorithm_variant!r}"
        )
    expected_parameters = {
        "segmentation_penalty": 0.01,
        "minimum_segment_strings": 2,
        "minimum_reference_strings": 5,
        "minimum_candidate_strings": 2,
        "group_relative_drop_tolerance": 0.0001,
        "minimum_relative_drop": 0.20,
        "minimum_absolute_drop": 0.50,
        "effect_gate_mode": "any",
        "internal_effect_gate_mode": "all",
        "minimum_core_segment_support": 0.80,
        "minimum_expansion_segment_support": 0.80,
        "maximum_core_expansion_boundary_gap": 0.05,
        "maximum_external_expansion_reference_segments": 1,
        "entry_consecutive": 3,
        "recovery_consecutive": 3,
        "expected_interval_minutes": 5,
        "maximum_unknown_intervals": 2,
    }
    for name, expected in expected_parameters.items():
        actual = getattr(config, name)
        matches = (
            math.isclose(float(actual), float(expected))
            if isinstance(expected, float)
            else actual == expected
        )
        if not matches:
            errors.append(f"detector {name} must be {expected!r}, got {actual!r}")

    base = detector.base_calibration
    if base.isolated_effect_gate_mode != "any":
        errors.append("base isolated_effect_gate_mode must be 'any'")
    if not math.isclose(base.minimum_isolated_relative_drop, 0.20):
        errors.append("base minimum_isolated_relative_drop must be 0.20")
    if not math.isclose(base.minimum_isolated_absolute_drop, 0.50):
        errors.append("base minimum_isolated_absolute_drop must be 0.50")
    if not base.configured_strings:
        errors.append("base calibration has no configured device/string mapping")

    temporal = {
        "base": (
            base.minimum_consecutive,
            base.expected_interval_minutes,
        ),
        "hierarchy": (
            detector.hierarchy_calibration.minimum_consecutive,
            detector.hierarchy_calibration.expected_interval_minutes,
        ),
        "memory": (
            detector.memory_config.entry_consecutive,
            detector.memory_config.expected_interval_minutes,
        ),
    }
    for name, (entry, interval) in temporal.items():
        if entry != config.entry_consecutive:
            errors.append(
                f"{name} entry/minimum_consecutive={entry} differs from "
                f"detector={config.entry_consecutive}"
            )
        if interval != config.expected_interval_minutes:
            errors.append(
                f"{name} expected_interval_minutes={interval} differs from "
                f"detector={config.expected_interval_minutes}"
            )
    if detector.memory_config.recovery_consecutive != config.recovery_consecutive:
        errors.append(
            "memory recovery_consecutive differs from detector: "
            f"{detector.memory_config.recovery_consecutive} != "
            f"{config.recovery_consecutive}"
        )
    if errors:
        raise ValueError("PVLOF-V3 frozen contract mismatch: " + "; ".join(errors))


def _prepare_source(
    frame: pd.DataFrame,
    *,
    supported_device_keys: frozenset[str],
    strict_calibration: bool,
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("PVLOF-V3 input must be a pandas DataFrame")
    if frame.empty:
        raise ValueError("PVLOF-V3 input must not be empty")
    required = set(DEVICE_TIME_KEYS)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"PVLOF-V3 input is missing columns: {missing}")
    current_columns = sorted(
        column for column in frame.columns if CURRENT_PATTERN.match(str(column))
    )
    if not current_columns:
        raise ValueError("PVLOF-V3 input has no string_current_XX columns")

    source = frame.copy()
    source["plant_id"] = source["plant_id"].astype(str).str.strip()
    source["device_no"] = source["device_no"].astype(str).str.strip()
    if source[["plant_id", "device_no"]].eq("").any().any():
        raise ValueError("PVLOF-V3 plant_id/device_no must not be blank")
    source["event_time"] = pd.to_datetime(
        source["event_time"], errors="raise", utc=True
    )
    if source.duplicated(DEVICE_TIME_KEYS).any():
        examples = source.loc[
            source.duplicated(DEVICE_TIME_KEYS, keep=False), DEVICE_TIME_KEYS
        ].head(5)
        raise ValueError(
            "PVLOF-V3 input contains duplicate plant/device/time rows; "
            f"examples: {examples.to_dict('records')}"
        )
    if strict_calibration:
        input_keys = {
            f"{plant}|{device}"
            for plant, device in source[["plant_id", "device_no"]].itertuples(
                index=False, name=None
            )
        }
        unknown = sorted(input_keys - supported_device_keys)
        if unknown:
            raise ValueError(
                "PVLOF-V3 input contains devices absent from the frozen calibration: "
                f"{unknown[:10]}"
            )
    return source.sort_values(
        ["plant_id", "event_time", "device_no"]
    ).reset_index(drop=True)


def _add_canonical_aliases(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    all_aliases = {**_STATE_ALIASES, **_DIAGNOSTIC_ALIASES}
    missing = sorted(set(all_aliases.values()) - set(result.columns))
    if missing:
        raise ValueError(
            "PVLOF-V3 internal detector output is missing columns: "
            f"{missing}"
        )
    for suffix, source_column in all_aliases.items():
        result[f"{PVLOF_V3_PREFIX}_{suffix}"] = result[source_column]
    return result


def load_pvlof_v3(root: str | Path) -> PVLOFV3Detector:
    """Load and validate the frozen PVLOF-V3 artifacts under ``root``."""

    return PVLOFV3Detector.from_root(root)


def run_pvlof_v3(
    frame: pd.DataFrame,
    detector: PVLOFV3Detector,
    *,
    weather: pd.DataFrame | None = None,
    timezone: str = "Asia/Shanghai",
    strict_calibration: bool = True,
) -> PVLOFV3Result:
    """Functional production entry point for dependency-injection frameworks."""

    return detector.run(
        frame,
        weather=weather,
        timezone=timezone,
        strict_calibration=strict_calibration,
    )


__all__ = [
    "PVLOF_V3_ALGORITHM_VARIANT",
    "PVLOF_V3_PREFIX",
    "PVLOF_V3_VERSION",
    "PVLOFV3Detector",
    "PVLOFV3ModelPaths",
    "PVLOFV3Result",
    "load_pvlof_v3",
    "run_pvlof_v3",
]
