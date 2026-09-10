from dataclasses import replace

import pytest

from pv_anomaly.pvlof_v16 import PVLOFV16MemoryConfig
from pv_anomaly.pvlof_v17 import PVLOFV17Config
from pv_anomaly.pvlof_v2 import PVLOFV2Calibration
from scripts.run_pvlof_v16_v17_alarm_windows import _validate_version_contract


def _v16_base():
    return replace(
        PVLOFV2Calibration(),
        minimum_isolated_relative_drop=0.20,
        minimum_isolated_absolute_drop=0.50,
        isolated_effect_gate_mode="any",
    )


def test_integrated_runner_accepts_exact_v16_gate():
    _validate_version_contract(
        _v16_base(),
        PVLOFV16MemoryConfig(),
        PVLOFV17Config(),
    )


def test_integrated_runner_rejects_non_v16_base_calibration():
    with pytest.raises(ValueError, match="not the V1.6 hybrid gate"):
        _validate_version_contract(
            PVLOFV2Calibration(),
            PVLOFV16MemoryConfig(),
            PVLOFV17Config(),
        )


def test_integrated_runner_requires_matching_intervals():
    with pytest.raises(ValueError, match="expected_interval_minutes must match"):
        _validate_version_contract(
            _v16_base(),
            PVLOFV16MemoryConfig(expected_interval_minutes=5),
            PVLOFV17Config(expected_interval_minutes=10),
        )
