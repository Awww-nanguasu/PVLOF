from datetime import datetime

import pandas as pd
import pytest

from scripts.add_pvlof_v17_improved_v2_to_review_workbook import (
    V17_IMPROVED_V2_FINAL_HEADER,
    V17_IMPROVED_V2_HEADER,
    add_v2_results,
)
from scripts.build_pvlof_v17_improved_review_workbook import (
    V16_FINAL_HEADER,
    V16_HEADER,
    V16_REPLAY_HEADER,
    V17_FINAL_HEADER,
    V17_HEADER,
    V17_IMPROVED_FINAL_HEADER,
    V17_IMPROVED_HEADER,
    build_review_workbook,
)


openpyxl = pytest.importorskip("openpyxl")


def _workbook(path):
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append([
        None, "Baseline", V16_HEADER, V17_HEADER, "row", "plant_id",
        "device_no", "raise_time_local", "end_time_local", "event_time_local", "01",
    ])
    sheet.append([
        False, True, "01", "FALSE", 1, 234, "device-a",
        datetime(2026, 6, 1, 8, 0), datetime(2026, 6, 1, 8, 5),
        datetime(2026, 6, 1, 8, 0), 4.2,
    ])
    sheet.freeze_panes = "E2"
    workbook.save(path)


def _predictions(path, prefix, raw, final):
    frame = pd.DataFrame({
        "plant_id": [234],
        "device_no": ["device-a"],
        "event_time": [pd.Timestamp("2026-06-01 00:00:00Z")],
        "string_no": [1],
        "string_current": [4.2],
    })
    if prefix == "v16":
        frame["pvlof_v16_raw_anomaly"] = raw
        frame["collective_raw_alert"] = 0
        frame["pvlof_v16_alert"] = final
    elif prefix == "v17":
        frame["pvlof_v17_raw_anomaly"] = raw
        frame["pvlof_v17_segmentation_raw_candidate"] = raw
        frame["pvlof_v17_alert"] = final
    else:
        frame["pvlof_v17_improved_raw_anomaly"] = raw
        frame["pvlof_v17_improved_segmentation_raw_candidate"] = raw
        frame["pvlof_v17_improved_alert"] = final
    frame.to_parquet(path, index=False)


def _v2_source_workbook(path):
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append([
        None,
        "Baseline",
        V16_HEADER,
        V16_REPLAY_HEADER,
        V17_HEADER,
        V17_IMPROVED_HEADER,
        V16_FINAL_HEADER,
        V17_FINAL_HEADER,
        V17_IMPROVED_FINAL_HEADER,
        "row",
        "plant_id",
        "device_no",
        "raise_time_local",
        "end_time_local",
        "event_time_local",
        "01",
    ])
    sheet.append([
        False,
        True,
        "01",
        "01",
        "FALSE",
        "01",
        "FALSE",
        "FALSE",
        "FALSE",
        1,
        234,
        "device-a",
        datetime(2026, 6, 1, 8, 0),
        datetime(2026, 6, 1, 8, 5),
        datetime(2026, 6, 1, 8, 0),
        4.2,
    ])
    sheet.freeze_panes = "J2"
    sheet.cell(1, 6).fill = openpyxl.styles.PatternFill(
        fill_type="solid",
        fgColor="D9EAF7",
    )
    sheet.cell(2, 6).fill = openpyxl.styles.PatternFill(
        fill_type="solid",
        fgColor="D9EAF7",
    )
    workbook.create_sheet("Sheet2")
    workbook.save(path)


def test_review_workbook_preserves_manual_history_and_adds_all_results(tmp_path):
    source = tmp_path / "source.xlsx"
    output = tmp_path / "output.xlsx"
    _workbook(source)
    v16, v17, improved = (tmp_path / name for name in ("v16.parquet", "v17.parquet", "improved.parquet"))
    _predictions(v16, "v16", 1, 0)
    _predictions(v17, "v17", 0, 0)
    _predictions(improved, "improved", 1, 1)

    report = build_review_workbook(
        source, v16, v17, improved, output, tmp_path / "summary.json"
    )
    sheet = openpyxl.load_workbook(output).active
    headers = {cell.value: cell.column for cell in sheet[1] if cell.value is not None}

    assert sheet.cell(2, 1).value is False
    assert sheet.cell(2, headers["Baseline"]).value is True
    assert sheet.cell(2, headers[V16_HEADER]).value == "01"
    assert sheet.cell(2, headers[V16_REPLAY_HEADER]).value == "01"
    assert sheet.cell(2, headers[V17_HEADER]).value == "FALSE"
    assert sheet.cell(2, headers[V17_IMPROVED_HEADER]).value == "01"
    assert sheet.cell(2, headers[V16_FINAL_HEADER]).value == "FALSE"
    assert sheet.cell(2, headers[V17_FINAL_HEADER]).value == "FALSE"
    assert sheet.cell(2, headers[V17_IMPROVED_FINAL_HEADER]).value == "01"
    assert report["manual_label_preserved"] is True


def test_review_workbook_rejects_missing_prediction_point(tmp_path):
    source = tmp_path / "source.xlsx"
    _workbook(source)
    v16, v17, improved = (tmp_path / name for name in ("v16.parquet", "v17.parquet", "improved.parquet"))
    for path, prefix in ((v16, "v16"), (v17, "v17"), (improved, "improved")):
        _predictions(path, prefix, 0, 0)
    frame = pd.read_parquet(improved)
    frame["event_time"] = pd.Timestamp("2026-06-01 00:05:00Z")
    frame.to_parquet(improved, index=False)

    with pytest.raises(ValueError, match="does not cover every workbook point"):
        build_review_workbook(
            source, v16, v17, improved, tmp_path / "output.xlsx", tmp_path / "summary.json"
        )


def test_v2_review_workbook_adds_new_columns_and_preserves_every_existing_column(tmp_path):
    source = tmp_path / "pvlof1.7.xlsx"
    predictions = tmp_path / "improved_v2.parquet"
    output = tmp_path / "pvlof1.7_v2.xlsx"
    _v2_source_workbook(source)
    _predictions(predictions, "improved", 1, 0)

    report = add_v2_results(
        source,
        predictions,
        output,
        tmp_path / "summary.json",
    )
    workbook = openpyxl.load_workbook(output)
    sheet = workbook.active
    headers = {cell.value: cell.column for cell in sheet[1] if cell.value is not None}

    assert workbook.sheetnames == ["Sheet", "Sheet2"]
    assert sheet.cell(2, 1).value is False
    assert sheet.cell(2, headers["Baseline"]).value is True
    assert sheet.cell(2, headers[V16_HEADER]).value == "01"
    assert sheet.cell(2, headers[V16_REPLAY_HEADER]).value == "01"
    assert sheet.cell(2, headers[V17_HEADER]).value == "FALSE"
    assert sheet.cell(2, headers[V17_IMPROVED_HEADER]).value == "01"
    assert sheet.cell(2, headers[V16_FINAL_HEADER]).value == "FALSE"
    assert sheet.cell(2, headers[V17_FINAL_HEADER]).value == "FALSE"
    assert sheet.cell(2, headers[V17_IMPROVED_FINAL_HEADER]).value == "FALSE"
    assert sheet.cell(2, headers[V17_IMPROVED_V2_HEADER]).value == "01"
    assert sheet.cell(2, headers[V17_IMPROVED_V2_FINAL_HEADER]).value == "FALSE"
    assert headers[V17_IMPROVED_V2_HEADER] == headers[V17_IMPROVED_HEADER] + 1
    assert (
        headers[V17_IMPROVED_V2_FINAL_HEADER]
        == headers[V17_IMPROVED_FINAL_HEADER] + 1
    )
    assert (
        sheet.cell(1, headers[V17_IMPROVED_V2_HEADER]).fill.fgColor.rgb
        == sheet.cell(1, headers[V17_IMPROVED_HEADER]).fill.fgColor.rgb
    )
    assert report["manual_label_preserved"] is True
    assert report["existing_columns_preserved"] is True
    assert report["coverage"]["missing"] == 0


def test_v2_review_workbook_rejects_missing_prediction_point(tmp_path):
    source = tmp_path / "pvlof1.7.xlsx"
    predictions = tmp_path / "improved_v2.parquet"
    _v2_source_workbook(source)
    _predictions(predictions, "improved", 0, 0)
    frame = pd.read_parquet(predictions)
    frame["event_time"] = pd.Timestamp("2026-06-01 00:05:00Z")
    frame.to_parquet(predictions, index=False)

    with pytest.raises(ValueError, match="does not cover every workbook point"):
        add_v2_results(
            source,
            predictions,
            tmp_path / "pvlof1.7_v2.xlsx",
            tmp_path / "summary.json",
        )
