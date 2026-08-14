"""Derive parallel PVLOF v1.4 relative-and-absolute isolated gates."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from pv_anomaly.pvlof_v2 import load_calibration, save_calibration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-relative-drop", type=float, default=0.10)
    parser.add_argument("--minimum-absolute-drop", type=float, default=0.50)
    parser.add_argument("--version", default="pvlof-v1.4-dual-effect-gate")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    if not 0.0 <= args.minimum_relative_drop < 1.0:
        raise ValueError("--minimum-relative-drop must be in [0, 1)")
    if args.minimum_absolute_drop < 0.0:
        raise ValueError("--minimum-absolute-drop must be non-negative")

    source = load_calibration(args.input)
    derived = replace(
        source,
        version=args.version,
        minimum_isolated_relative_drop=args.minimum_relative_drop,
        minimum_isolated_absolute_drop=args.minimum_absolute_drop,
    )
    save_calibration(derived, args.output)
    report = {
        "input": args.input,
        "output": args.output,
        "source_version": source.version,
        "version": derived.version,
        "minimum_isolated_relative_drop": derived.minimum_isolated_relative_drop,
        "minimum_isolated_absolute_drop": derived.minimum_isolated_absolute_drop,
        "isolated_gate_logic": "relative_drop >= threshold AND expected_current - string_current >= threshold",
        "score_thresholds_refit": False,
        "collective_parameters_changed": False,
    }
    destination = Path(args.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
