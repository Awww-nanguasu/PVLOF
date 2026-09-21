"""Apply PVLOF V1.7 Improved V5 to existing 58-day point files.

V5 keeps V4's per-string combined-candidate memory. Its segmentation branch
uses the lowest residual segment as the candidate core and permits contiguous
upward expansion only for segments with coherent external physical evidence.
The replay reuses each plant's frozen ``pvlof_v17_full_points.parquet`` and
does not recalibrate.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


import pandas as pd

from pv_anomaly.pvlof_v17_improved import apply_pvlof_v17_improved, load_config
from scripts.run_pvlof_58d_versions import (
    _reconstruct_variant_events,
    _write_version_outputs,
)


SLUG = "pvlof_v17_improved_v5"
SOURCE_NAME = "pvlof_v17_full_points.parquet"


def _select_plant_directories(root: Path, plant_ids: list[str]) -> list[Path]:
    requested = {str(value) for value in plant_ids}
    directories = sorted(path for path in root.glob("plant_id=*") if path.is_dir())
    if requested:
        directories = [
            path
            for path in directories
            if path.name.split("=", 1)[-1] in requested
        ]
        found = {path.name.split("=", 1)[-1] for path in directories}
        missing = sorted(requested - found)
        if missing:
            raise FileNotFoundError(f"Plant replay directories not found: {missing}")
    if not directories:
        raise FileNotFoundError(f"No plant_id=* replay directories found under {root}")
    return directories


def _known_outputs(directory: Path) -> list[Path]:
    return [
        directory / f"{SLUG}_full_points.parquet",
        directory / f"{SLUG}_evidence_points.parquet",
        directory / f"{SLUG}_events.parquet",
        directory / f"{SLUG}_unconfirmed_candidates.parquet",
        directory / f"{SLUG}_summary.json",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument(
        "--config",
        default="configs/pvlof_v17_improved_v5_lowest_core_expansion.json",
    )
    parser.add_argument("--plant-id", action="append", default=[])
    parser.add_argument("--timezone", default="Asia/Shanghai")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--report")
    args = parser.parse_args()

    root = Path(args.input_root)
    config = load_config(args.config)
    if config.algorithm_variant != "lowest_core_expansion_v5":
        raise ValueError(
            "--config must select algorithm_variant=lowest_core_expansion_v5"
        )

    plant_directories = _select_plant_directories(root, args.plant_id)
    for directory in plant_directories:
        plant_id = directory.name.split("=", 1)[-1]
        source_path = directory / SOURCE_NAME
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        existing = [path for path in _known_outputs(directory) if path.exists()]
        if existing and not args.overwrite:
            raise FileExistsError(
                f"V5 outputs already exist for plant_id={plant_id}: {existing}. "
                "Use --overwrite to replace only these known V5 files."
            )

    reports = {}
    for directory in plant_directories:
        plant_id = directory.name.split("=", 1)[-1]
        source_path = directory / SOURCE_NAME
        print(f"Reading {source_path}")
        source = pd.read_parquet(source_path)
        result = apply_pvlof_v17_improved(source, config)
        evidence, events, unconfirmed = _reconstruct_variant_events(
            result,
            version=config.version,
            final_alert_column="pvlof_v17_improved_combined_memory_alert",
            raw_anomaly_column="pvlof_v17_improved_raw_anomaly",
            memory_prefix="pvlof_v17_improved",
            interval_minutes=config.expected_interval_minutes,
            timezone=args.timezone,
            segmentation_branch="improved_v5_lowest_core_expansion",
        )
        report = _write_version_outputs(
            directory,
            slug=SLUG,
            frame=result,
            version=config.version,
            raw_anomaly_column="pvlof_v17_improved_raw_anomaly",
            final_alert_column="pvlof_v17_improved_combined_memory_alert",
            evidence=evidence,
            events=events,
            unconfirmed=unconfirmed,
        )
        report.update({
            "source": str(source_path),
            "segmentation_raw_candidate_string_points": int(
                result[
                    "pvlof_v17_improved_segmentation_raw_candidate"
                ].fillna(False).astype(bool).sum()
            ),
            "lowest_core_candidate_string_points": int(
                result[
                    "pvlof_v17_improved_segmentation_lowest_core_raw_candidate"
                ].fillna(False).astype(bool).sum()
            ),
            "validated_expansion_candidate_string_points": int(
                result[
                    "pvlof_v17_improved_segmentation_validated_expansion_raw_candidate"
                ].fillna(False).astype(bool).sum()
            ),
            "combined_raw_candidate_string_points": int(
                result[
                    "pvlof_v17_improved_combined_raw_candidate"
                ].fillna(False).astype(bool).sum()
            ),
            "combined_bridge_alert_points": int(
                result[
                    "pvlof_v17_improved_combined_bridge_alert"
                ].fillna(False).astype(bool).sum()
            ),
            "v16_middle_group_veto_points": int(
                result[
                    "pvlof_v17_improved_v16_middle_group_veto"
                ].fillna(False).astype(bool).sum()
            ),
            "reference_member_gate_passes": int(
                result[
                    "pvlof_v17_improved_segmentation_reference_member_gate_pass"
                ].fillna(False).astype(bool).sum()
            ),
            "expansion_gate_passes": int(
                result[
                    "pvlof_v17_improved_segmentation_expansion_gate_pass"
                ].fillna(False).astype(bool).sum()
            ),
            "expansion_external_support_passes": int(
                result[
                    "pvlof_v17_improved_segmentation_expansion_external_support_pass"
                ].fillna(False).astype(bool).sum()
            ),
            "expansion_small_gap_passes": int(
                result[
                    "pvlof_v17_improved_segmentation_expansion_small_gap_pass"
                ].fillna(False).astype(bool).sum()
            ),
            "data_quality_suspended_string_points": int(
                result["pvlof_v17_improved_memory_suspended"]
                .fillna(False).astype(bool).sum()
            ),
            "memory_resumed_alert_points": int(
                result["pvlof_v17_improved_memory_resumed_alert"]
                .fillna(False).astype(bool).sum()
            ),
            "data_gap_timeout_points": int(
                result["pvlof_v17_improved_memory_clear_code"].eq(2).sum()
            ),
            "not_evaluable_timeout_points": int(
                result["pvlof_v17_improved_memory_clear_code"].eq(3).sum()
            ),
            "group_gate_applied": False,
            "temporal_confirmation": "combined_v16_or_v5_raw_by_string",
            "maximum_unknown_intervals": config.maximum_unknown_intervals,
            "candidate_policy": (
                "lowest_core_plus_contiguous_small_gap_or_stable_reference_external_expansion"
            ),
        })
        plant_report = directory / f"{SLUG}_summary.json"
        plant_report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        reports[plant_id] = report
        del source, result, evidence, events, unconfirmed
        gc.collect()

    aggregate = {
        "version": config.version,
        "algorithm_variant": config.algorithm_variant,
        "config": str(args.config),
        "input_root": str(root),
        "plants": reports,
    }
    report_path = (
        Path(args.report)
        if args.report
        else root / f"{SLUG}_summary.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
