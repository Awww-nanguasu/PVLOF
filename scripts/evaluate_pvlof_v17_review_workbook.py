"""Evaluate PVLOF review columns against the first-column manual labels."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


BASELINE_HEADER = "Baseline"
LOCALISATION_HEADERS = (
    "PVLOF_V1_6_HYBRID_GATE",
    "PVLOF_V1_6_V3_REPLAY",
    "PVLOF_V1_7_PENALIZED_SEGMENTATION",
    "PVLOF_V1_7_IMPROVED",
    "PVLOF_V1_7_IMPROVED_V2",
)
OLD_IMPROVED_HEADER = "PVLOF_V1_7_IMPROVED"
V2_HEADER = "PVLOF_V1_7_IMPROVED_V2"
FALSE_VALUES = {"", "0", "false", "none", "nan", "no", "否", "无"}
TRUE_VALUES = {"1", "true", "yes", "是", "有"}
SPLIT_PATTERN = re.compile(r"[,，;；\s]+")
DETAIL_HEADERS = (
    "row",
    "plant_id",
    "device_no",
    "raise_time_local",
    "end_time_local",
    "event_time_local",
)


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 4)


def _parse_string_set(value: Any, *, column: str, excel_row: int) -> set[int]:
    if value is None or pd.isna(value):
        return set()
    if isinstance(value, bool):
        if not value:
            return set()
        raise ValueError(
            f"{column} row {excel_row} uses TRUE without string numbers"
        )
    if isinstance(value, (int, float)):
        number = int(value)
        if float(value) == number and number > 0:
            return {number}
    text = str(value).strip()
    if text.lower() in FALSE_VALUES:
        return set()
    tokens = [token for token in SPLIT_PATTERN.split(text) if token]
    try:
        numbers = {int(token) for token in tokens}
    except ValueError as exc:
        raise ValueError(
            f"Invalid string label in {column} row {excel_row}: {value!r}"
        ) from exc
    if not numbers or min(numbers) < 1:
        raise ValueError(
            f"Invalid string label in {column} row {excel_row}: {value!r}"
        )
    return numbers


def _parse_alarm_flag(value: Any, *, column: str, excel_row: int) -> bool:
    if value is None or pd.isna(value):
        return False
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        return float(value) != 0
    text = str(value).strip().lower()
    if text in FALSE_VALUES:
        return False
    if text in TRUE_VALUES:
        return True
    return bool(_parse_string_set(value, column=column, excel_row=excel_row))


def _row_detection_metrics(expected_sets, actual_flags) -> dict[str, Any]:
    expected_flags = [bool(values) for values in expected_sets]
    tp = sum(expected and actual for expected, actual in zip(expected_flags, actual_flags))
    fp = sum(not expected and actual for expected, actual in zip(expected_flags, actual_flags))
    fn = sum(expected and not actual for expected, actual in zip(expected_flags, actual_flags))
    tn = sum(not expected and not actual for expected, actual in zip(expected_flags, actual_flags))
    precision = _rate(tp, tp + fp)
    recall = _rate(tp, tp + fn)
    f1 = _rate(2 * tp, 2 * tp + fp + fn)
    return {
        "true_positive_rows": tp,
        "false_positive_rows": fp,
        "false_negative_rows": fn,
        "true_negative_rows": tn,
        "accuracy": _rate(tp + tn, len(expected_flags)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _localisation_metrics(expected_sets, actual_sets) -> dict[str, Any]:
    exact_rows = sum(expected == actual for expected, actual in zip(expected_sets, actual_sets))
    false_positive_rows = sum(
        not expected and bool(actual)
        for expected, actual in zip(expected_sets, actual_sets)
    )
    false_negative_rows = sum(
        bool(expected) and not actual
        for expected, actual in zip(expected_sets, actual_sets)
    )
    different_string_rows = sum(
        bool(expected) and bool(actual) and expected != actual
        for expected, actual in zip(expected_sets, actual_sets)
    )
    string_tp = sum(len(expected & actual) for expected, actual in zip(expected_sets, actual_sets))
    string_fp = sum(len(actual - expected) for expected, actual in zip(expected_sets, actual_sets))
    string_fn = sum(len(expected - actual) for expected, actual in zip(expected_sets, actual_sets))
    string_precision = _rate(string_tp, string_tp + string_fp)
    string_recall = _rate(string_tp, string_tp + string_fn)
    string_f1 = _rate(
        2 * string_tp,
        2 * string_tp + string_fp + string_fn,
    )
    result = {
        "exact_rows": exact_rows,
        "exact_rate": _rate(exact_rows, len(expected_sets)),
        "false_positive_rows": false_positive_rows,
        "false_negative_rows": false_negative_rows,
        "different_string_rows": different_string_rows,
        "string_true_positives": string_tp,
        "string_false_positives": string_fp,
        "string_false_negatives": string_fn,
        "string_precision": string_precision,
        "string_recall": string_recall,
        "string_f1": string_f1,
    }
    result["row_detection"] = _row_detection_metrics(
        expected_sets,
        [bool(values) for values in actual_sets],
    )
    return result


def _format_set(values: set[int]) -> str:
    if not values:
        return "FALSE"
    return ",".join(f"{value:02d}" for value in sorted(values))


def _detail(frame, position, expected, actual) -> dict[str, Any]:
    row = frame.iloc[position]
    detail = {
        "excel_row": position + 2,
        "expected": _format_set(expected),
        "actual": _format_set(actual),
        "missing": _format_set(expected - actual),
        "extra": _format_set(actual - expected),
    }
    for header in DETAIL_HEADERS:
        if header in frame.columns:
            value = row[header]
            detail[header] = None if pd.isna(value) else str(value)
    return detail


def evaluate_workbook(path, *, sheet_name=0) -> dict[str, Any]:
    workbook_path = Path(path).resolve()
    frame = pd.read_excel(workbook_path, sheet_name=sheet_name, dtype=object)
    if frame.empty:
        raise ValueError("Review workbook has no data rows")
    manual_column = str(frame.columns[0])
    missing = sorted(
        {OLD_IMPROVED_HEADER, V2_HEADER} - set(frame.columns)
    )
    if missing:
        raise ValueError(f"Review workbook is missing columns: {missing}")

    expected_sets = [
        _parse_string_set(value, column=manual_column, excel_row=position + 2)
        for position, value in enumerate(frame.iloc[:, 0])
    ]
    localisation = {}
    parsed_columns = {}
    for header in LOCALISATION_HEADERS:
        if header not in frame.columns:
            continue
        actual_sets = [
            _parse_string_set(value, column=header, excel_row=position + 2)
            for position, value in enumerate(frame[header])
        ]
        parsed_columns[header] = actual_sets
        localisation[header] = _localisation_metrics(expected_sets, actual_sets)

    row_detection = {}
    if BASELINE_HEADER in frame.columns:
        baseline_flags = [
            _parse_alarm_flag(value, column=BASELINE_HEADER, excel_row=position + 2)
            for position, value in enumerate(frame[BASELINE_HEADER])
        ]
        row_detection[BASELINE_HEADER] = _row_detection_metrics(
            expected_sets,
            baseline_flags,
        )

    old_sets = parsed_columns[OLD_IMPROVED_HEADER]
    v2_sets = parsed_columns[V2_HEADER]
    changed_positions = [
        position
        for position, (old, new) in enumerate(zip(old_sets, v2_sets))
        if old != new
    ]
    improved_positions = [
        position
        for position, (expected, old, new) in enumerate(
            zip(expected_sets, old_sets, v2_sets)
        )
        if old != expected and new == expected
    ]
    regressed_positions = [
        position
        for position, (expected, old, new) in enumerate(
            zip(expected_sets, old_sets, v2_sets)
        )
        if old == expected and new != expected
    ]
    v2_mismatch_positions = [
        position
        for position, (expected, actual) in enumerate(zip(expected_sets, v2_sets))
        if expected != actual
    ]

    return {
        "workbook": str(workbook_path),
        "sheet": sheet_name,
        "rows": len(frame),
        "manual_column": manual_column,
        "manual_normal_rows": sum(not values for values in expected_sets),
        "manual_alarm_rows": sum(bool(values) for values in expected_sets),
        "manual_alarm_strings": sum(len(values) for values in expected_sets),
        "row_detection_algorithms": row_detection,
        "localisation_algorithms": localisation,
        "v2_vs_previous_improved": {
            "changed_rows": len(changed_positions),
            "improved_to_exact_rows": len(improved_positions),
            "regressed_from_exact_rows": len(regressed_positions),
            "unchanged_rows": len(frame) - len(changed_positions),
            "improved_cases": [
                {
                    **_detail(frame, position, expected_sets[position], v2_sets[position]),
                    "previous": _format_set(old_sets[position]),
                }
                for position in improved_positions
            ],
            "regressed_cases": [
                {
                    **_detail(frame, position, expected_sets[position], v2_sets[position]),
                    "previous": _format_set(old_sets[position]),
                }
                for position in regressed_positions
            ],
        },
        "v2_mismatches": [
            _detail(frame, position, expected_sets[position], v2_sets[position])
            for position in v2_mismatch_positions
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--sheet", default=0)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    sheet: int | str
    sheet = int(args.sheet) if str(args.sheet).isdigit() else args.sheet
    report = evaluate_workbook(args.input, sheet_name=sheet)
    destination = Path(args.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


__all__ = ["evaluate_workbook"]
