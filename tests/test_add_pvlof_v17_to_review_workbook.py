from datetime import datetime

import pandas as pd
import pytest

from scripts.add_pvlof_v17_to_review_workbook import (
    V16_HEADER,
    V17_HEADER,
    add_v17_column,
)


openpyxl = pytest.importorskip("openpyxl")


def _write_template(path):
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "pvlof_v16_vs_baseline_floor_rai"
    sheet.append(
        [
            None,
            "Baseline",
            V16_HEADER,
            "row",
            "plant_id",
            "device_no",
            "raise_time_local",
            "end_time_local",
            "event_time_local",
            "01",
        ]
    )
    sheet.append(
        [
            1,
            True,
            "01",
            1,
            234,
            "device-a",
            datetime(2026, 4, 9, 13, 40, 5),
            datetime(2026, 4, 9, 13, 45, 5),
            datetime(2026, 4, 9, 13, 40),
            4.2,
        ]
    )
    sheet.append(
        [
            1,
            True,
            "FALSE",
            2,
            234,
            "device-a",
            datetime(2026, 4, 9, 13, 40, 5),
            datetime(2026, 4, 9, 13, 45, 5),
            datetime(2026, 4, 9, 13, 45),
            12.1,
        ]
    )
    sheet.freeze_panes = "D2"
    sheet.column_dimensions["C"].width = 20
    workbook.save(path)


def _write_predictions(path, include_second=True):
    times = [pd.Timestamp("2026-04-09 05:40:00Z")]
    if include_second:
        times.append(pd.Timestamp("2026-04-09 05:45:00Z"))
    frame = pd.DataFrame(
        {
            "plant_id": [234] * len(times),
            "device_no": ["device-a"] * len(times),
            "event_time": times,
            "string_no": [1] * len(times),
            "string_current": [4.2, 12.1][: len(times)],
            "pvlof_v17_raw_anomaly": [1, 0][: len(times)],
            "pvlof_v17_segmentation_raw_candidate": [0, 0][: len(times)],
            "pvlof_v17_alert": [1, 0][: len(times)],
        }
    )
    frame.to_parquet(path, index=False)


def test_add_v17_column_preserves_review_rows_and_inserts_after_v16(tmp_path):
    template = tmp_path / "template.xlsx"
    predictions = tmp_path / "v17.parquet"
    output = tmp_path / "output.xlsx"
    report = tmp_path / "summary.json"
    _write_template(template)
    _write_predictions(predictions)

    result = add_v17_column(template, predictions, output, report)

    workbook = openpyxl.load_workbook(output)
    sheet = workbook.active
    assert sheet["C1"].value == V16_HEADER
    assert sheet["D1"].value == V17_HEADER
    assert sheet["D2"].value == "01"
    assert sheet["D3"].value == "FALSE"
    assert sheet["E1"].value == "row"
    assert sheet.freeze_panes == "E2"
    assert sheet.column_dimensions["D"].width == 20
    assert result["workbook_rows"] == 2
    assert result["missing_prediction_device_time_points"] == 0


def test_add_v17_column_rejects_prediction_time_mismatch(tmp_path):
    template = tmp_path / "template.xlsx"
    predictions = tmp_path / "v17.parquet"
    _write_template(template)
    _write_predictions(predictions, include_second=False)

    with pytest.raises(ValueError, match="do not cover every workbook"):
        add_v17_column(
            template,
            predictions,
            tmp_path / "output.xlsx",
            tmp_path / "summary.json",
        )
