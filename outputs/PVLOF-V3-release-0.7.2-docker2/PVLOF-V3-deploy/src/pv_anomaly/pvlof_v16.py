"""PVLOF v1.6 hybrid effect gate with confirmed-anomaly memory."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from pv_anomaly.pvlof_v15 import (
    PVLOFV15MemoryConfig,
    apply_confirmed_anomaly_memory,
)


@dataclass(frozen=True)
class PVLOFV16MemoryConfig:
    version: str = "pvlof-v1.6-hybrid-gate"
    entry_consecutive: int = 3
    recovery_consecutive: int = 3
    expected_interval_minutes: int = 5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PVLOFV16MemoryConfig":
        return cls(**dict(payload))


def save_memory_config(config: PVLOFV16MemoryConfig, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(config.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_memory_config(path: str | Path) -> PVLOFV16MemoryConfig:
    return PVLOFV16MemoryConfig.from_dict(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


def apply_confirmed_anomaly_memory_v16(
    frame: pd.DataFrame,
    config: PVLOFV16MemoryConfig,
) -> pd.DataFrame:
    """Apply unchanged v1.5 memory semantics using v1.6 column names."""

    legacy_config = PVLOFV15MemoryConfig(
        version=config.version,
        entry_consecutive=config.entry_consecutive,
        recovery_consecutive=config.recovery_consecutive,
        expected_interval_minutes=config.expected_interval_minutes,
    )
    result = apply_confirmed_anomaly_memory(frame, legacy_config)
    rename = {
        column: column.replace("pvlof_v15_", "pvlof_v16_", 1)
        for column in result.columns
        if column.startswith("pvlof_v15_")
    }
    return result.rename(columns=rename)
