import json
from pathlib import Path

import pandas as pd

from pv_anomaly.audit import audit_dataframe, load_records
from pv_anomaly.settings import DataSettings


def config() -> DataSettings:
    return DataSettings.from_yaml("configs/data.example.yaml")


def test_audit_detects_five_minute_data_and_algorithm_inputs():
    frame = pd.DataFrame(
        {
            "timestamp": ["2026-01-01T00:00:00Z", "2026-01-01T00:05:00Z"],
            "station_id": ["s1", "s1"],
            "inverter_id": ["i1", "i1"],
            "active_power": [10.0, 11.0],
            "irradiance": [500.0, 520.0],
            "string1_current": [4.1, 4.2],
            "string2_current": [4.0, 1.0],
        }
    )
    report = audit_dataframe(frame, config())
    assert report["timestamp"]["median_interval_seconds"] == 300.0
    assert report["feasibility"]["transformer_power_prediction"]["candidate"] is True
    assert report["feasibility"]["ewma_residual_detection"]["candidate"] is True
    assert report["feasibility"]["pvlof_string_localization"]["candidate"] is True


def test_pvlof_is_not_candidate_without_multiple_string_currents():
    frame = pd.DataFrame(
        {"timestamp": ["2026-01-01"], "inverter_id": ["i1"], "power": [10.0]}
    )
    report = audit_dataframe(frame, config())
    assert report["feasibility"]["pvlof_string_localization"]["candidate"] is False


def test_loads_elasticsearch_search_response(tmp_path: Path):
    path = tmp_path / "sample.json"
    path.write_text(
        json.dumps({"hits": {"hits": [{"_source": {"timestamp": "2026-01-01", "power": 1}}]}}),
        encoding="utf-8",
    )
    frame = load_records(path)
    assert frame.to_dict(orient="records") == [{"timestamp": "2026-01-01", "power": 1}]

