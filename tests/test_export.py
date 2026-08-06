from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from pv_anomaly.export import ExportDataset, export_dataset, local_date_range


class FakeClient:
    def count_range(self, index: str, field: str, start: str, end: str) -> int:
        return 2

    def iter_range(self, *args, **kwargs):
        yield [
            {"event_time": "2026-07-15T00:00:00+08:00", "plant_id": 1, "device_no": "a"},
            {"event_time": "2026-07-16T00:00:00+08:00", "plant_id": 1, "device_no": "a"},
        ]


class FilteredFakeClient:
    def __init__(self) -> None:
        self.count_plant_id = None
        self.iter_plant_id = None

    def count_range(
        self, index: str, field: str, start: str, end: str, *, plant_id=None
    ) -> int:
        self.count_plant_id = plant_id
        return 2

    def iter_range(self, *args, plant_id=None, **kwargs):
        self.iter_plant_id = plant_id
        yield [
            {
                "event_time": "2026-07-15T00:00:00+08:00",
                "plant_id": plant_id,
                "device_no": "a",
            },
            {
                "event_time": "2026-07-16T00:00:00+08:00",
                "plant_id": plant_id,
                "device_no": "a",
            },
        ]


def test_local_date_range_is_half_open_and_bounded():
    start, end = local_date_range(date(2026, 7, 15), date(2026, 7, 22), "Asia/Shanghai")
    assert start == "2026-07-15T00:00:00+08:00"
    assert end == "2026-07-22T00:00:00+08:00"
    with pytest.raises(ValueError, match="31 days"):
        local_date_range(date(2026, 1, 1), date(2026, 3, 1), "Asia/Shanghai")


def test_export_writes_date_partitions_and_checks_count(tmp_path: Path):
    dataset = ExportDataset(
        name="device",
        index="pv_device_data",
        time_field="event_time",
        sort_fields=["event_time", "plant_id", "device_no"],
        output=tmp_path / "device",
        fields=["event_time", "plant_id", "device_no"],
    )
    result = export_dataset(
        FakeClient(),
        dataset,
        start_date=date(2026, 7, 15),
        end_date=date(2026, 7, 17),
        timezone_name="Asia/Shanghai",
    )
    assert result["exported_rows"] == 2
    paths = sorted((tmp_path / "device").glob("date=*/*.parquet"))
    assert len(paths) == 2
    assert sum(len(pd.read_parquet(path)) for path in paths) == 2


def test_filtered_export_uses_plant_term_and_separate_layout(tmp_path: Path):
    client = FilteredFakeClient()
    dataset = ExportDataset(
        name="device",
        index="pv_device_data",
        time_field="event_time",
        sort_fields=["event_time", "plant_id", "device_no"],
        output=tmp_path / "device",
        fields=["event_time", "plant_id", "device_no"],
    )

    result = export_dataset(
        client,
        dataset,
        start_date=date(2026, 7, 15),
        end_date=date(2026, 7, 17),
        timezone_name="Asia/Shanghai",
        plant_id=892,
    )

    assert client.count_plant_id == 892
    assert client.iter_plant_id == 892
    assert result["plant_id"] == 892
    assert Path(result["output"]).name == "plant_id=892"
    paths = sorted((tmp_path / "device" / "plant_id=892").glob("date=*/*.parquet"))
    assert len(paths) == 2


def test_filtered_export_rejects_missing_plant_field(tmp_path: Path):
    dataset = ExportDataset(
        name="weather",
        index="weather",
        time_field="time",
        sort_fields=["time"],
        output=tmp_path / "weather",
        fields=["time"],
    )
    with pytest.raises(ValueError, match="does not declare plant_id"):
        export_dataset(
            FakeClient(),
            dataset,
            start_date=date(2026, 7, 15),
            end_date=date(2026, 7, 16),
            timezone_name="Asia/Shanghai",
            plant_id=892,
        )
