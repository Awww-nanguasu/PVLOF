"""PVLOF v1.5 confirmed-anomaly memory for intermittent isolated faults."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PVLOFV15MemoryConfig:
    version: str = "pvlof-v1.5-memory-5pct"
    entry_consecutive: int = 3
    recovery_consecutive: int = 3
    expected_interval_minutes: int = 5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PVLOFV15MemoryConfig":
        return cls(**dict(payload))


def save_memory_config(config: PVLOFV15MemoryConfig, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(config.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_memory_config(path: str | Path) -> PVLOFV15MemoryConfig:
    return PVLOFV15MemoryConfig.from_dict(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


def apply_confirmed_anomaly_memory(
    frame: pd.DataFrame,
    config: PVLOFV15MemoryConfig,
) -> pd.DataFrame:
    """Add v1.5 state without changing any existing PVLOF columns.

    Initial entry still requires consecutive raw isolated evidence.  Once
    confirmed, a later raw isolated point alerts immediately until three
    consecutive valid normal points clear the memory.  Normal points never
    become point alerts merely because memory is active.
    """

    required = {
        "plant_id",
        "device_no",
        "event_time",
        "string_no",
        "v2_eligible",
        "response_known",
        "string_current",
        "pvlof_score",
        "isolated_directional_raw_alert",
        "isolated_hier_raw_alert",
        "pvlof_v2_hier_strict_alert",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"PVLOF v1.5 input is missing columns: {missing}")
    if config.entry_consecutive < 1 or config.recovery_consecutive < 1:
        raise ValueError("entry/recovery consecutive counts must be positive")

    result = frame.copy()
    result["plant_id"] = result["plant_id"].astype(str)
    result["device_no"] = result["device_no"].astype(str)
    result["event_time"] = pd.to_datetime(result["event_time"], errors="raise", utc=True)
    result["string_no"] = pd.to_numeric(result["string_no"], errors="raise").astype("Int64")
    result = result.sort_values(
        ["plant_id", "device_no", "string_no", "event_time"]
    ).reset_index(drop=True)

    raw = (
        result["isolated_directional_raw_alert"].fillna(False).astype(bool)
        | result["isolated_hier_raw_alert"].fillna(False).astype(bool)
    )
    valid = (
        result["v2_eligible"].fillna(False).astype(bool)
        & result["response_known"].fillna(False).astype(bool)
        & pd.to_numeric(result["string_current"], errors="coerce").gt(0)
        & pd.to_numeric(result["pvlof_score"], errors="coerce").notna()
    )
    result["pvlof_v15_raw_anomaly"] = raw.astype(np.int8)
    result["pvlof_v15_valid_point"] = valid.astype(np.int8)

    row_count = len(result)
    entry_values = np.zeros(row_count, dtype=np.int32)
    normal_values = np.zeros(row_count, dtype=np.int32)
    active_values = np.zeros(row_count, dtype=np.int8)
    alert_values = np.zeros(row_count, dtype=np.int8)
    clear_values = np.zeros(row_count, dtype=np.int8)
    raw_values = raw.to_numpy(dtype=bool)
    valid_values = valid.to_numpy(dtype=bool)
    time_values = result["event_time"].to_numpy(dtype="datetime64[ns]")

    expected = np.timedelta64(config.expected_interval_minutes, "m")
    for _, positions in result.groupby(
        ["plant_id", "device_no", "string_no"], observed=True
    ).indices.items():
        indexes = np.asarray(positions, dtype=np.int64)
        previous_time: np.datetime64 | None = None
        entry_streak = 0
        normal_streak = 0
        memory_active = False

        for index in indexes:
            current_time = time_values[index]
            contiguous = (
                previous_time is not None and current_time - previous_time == expected
            )
            clear_code = 0
            if previous_time is not None and not contiguous:
                if memory_active:
                    clear_code = 2  # data_gap
                entry_streak = 0
                normal_streak = 0
                memory_active = False

            is_valid = bool(valid_values[index])
            is_raw = bool(raw_values[index]) and is_valid
            memory_alert = False

            if not is_valid:
                # Missing, night-time and otherwise unevaluable points are not
                # recovery evidence and must not carry stale state forward.
                if memory_active:
                    clear_code = 3  # not_evaluable
                entry_streak = 0
                normal_streak = 0
                memory_active = False
            elif not memory_active:
                normal_streak = 0
                entry_streak = entry_streak + 1 if is_raw and contiguous else int(is_raw)
                if entry_streak >= config.entry_consecutive:
                    memory_active = True
                    memory_alert = True
            elif is_raw:
                # Confirmed recently: a complete raw anomaly may alert at once.
                entry_streak = max(entry_streak, config.entry_consecutive)
                normal_streak = 0
                memory_alert = True
            else:
                # Normal points remain non-alerts.  Only their recovery counter
                # changes; the confirmed memory survives the first two.
                normal_streak += 1
                if normal_streak >= config.recovery_consecutive:
                    clear_code = 1  # recovered
                    entry_streak = 0
                    normal_streak = 0
                    memory_active = False

            entry_values[index] = entry_streak
            normal_values[index] = normal_streak
            active_values[index] = int(memory_active)
            alert_values[index] = int(memory_alert)
            clear_values[index] = clear_code
            previous_time = current_time

    result["pvlof_v15_entry_streak"] = entry_values
    result["pvlof_v15_normal_streak"] = normal_values
    result["pvlof_v15_memory_active"] = active_values
    result["pvlof_v15_memory_alert"] = alert_values
    result["pvlof_v15_memory_clear_code"] = clear_values

    base_alert = result["pvlof_v2_hier_strict_alert"].fillna(False).astype(bool)
    memory_alert = result["pvlof_v15_memory_alert"].astype(bool)
    result["pvlof_v15_memory_reactivated_alert"] = (
        memory_alert & ~base_alert
    ).astype(np.int8)
    result["pvlof_v15_alert"] = (base_alert | memory_alert).astype(np.int8)
    return result.sort_values(
        ["event_time", "plant_id", "device_no", "string_no"]
    ).reset_index(drop=True)
