"""Derive a persistence-only PVLOF variant without refitting score thresholds."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from pv_anomaly.pvlof_v2 import load_calibration, save_calibration
from pv_anomaly.pvlof_v2_hier import (
    load_hier_calibration,
    save_hier_calibration,
)


def derive_variant(
    base_input: str | Path,
    hier_input: str | Path,
    base_output: str | Path,
    hier_output: str | Path,
    *,
    minimum_consecutive: int,
    base_version: str,
    hier_version: str,
) -> dict[str, object]:
    if minimum_consecutive < 2:
        raise ValueError("minimum_consecutive must be at least 2")
    base = load_calibration(base_input)
    hier = load_hier_calibration(hier_input)
    if base.expected_interval_minutes != hier.expected_interval_minutes:
        raise ValueError("Base and hierarchical expected intervals do not match")
    derived_base = replace(
        base,
        version=base_version,
        minimum_consecutive=minimum_consecutive,
    )
    derived_hier = replace(
        hier,
        version=hier_version,
        minimum_consecutive=minimum_consecutive,
    )
    save_calibration(derived_base, base_output)
    save_hier_calibration(derived_hier, hier_output)
    return {
        "base_input": str(base_input),
        "hier_input": str(hier_input),
        "base_output": str(base_output),
        "hier_output": str(hier_output),
        "minimum_consecutive": minimum_consecutive,
        "expected_interval_minutes": base.expected_interval_minutes,
        "confirmation_minutes": (
            minimum_consecutive * base.expected_interval_minutes
        ),
        "base_version": derived_base.version,
        "hier_version": derived_hier.version,
        "score_thresholds_refit": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-input", required=True)
    parser.add_argument("--hier-input", required=True)
    parser.add_argument("--base-output", required=True)
    parser.add_argument("--hier-output", required=True)
    parser.add_argument("--minimum-consecutive", type=int, default=3)
    parser.add_argument("--base-version", default="pvlof-v2-v1.1-3point")
    parser.add_argument(
        "--hier-version", default="pvlof-v2-hier-strict-v1.1-3point"
    )
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    report = derive_variant(
        args.base_input,
        args.hier_input,
        args.base_output,
        args.hier_output,
        minimum_consecutive=args.minimum_consecutive,
        base_version=args.base_version,
        hier_version=args.hier_version,
    )
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
