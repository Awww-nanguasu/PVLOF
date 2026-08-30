"""Derive PVLOF v1.5 base calibration and confirmed-memory configuration."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from pv_anomaly.pvlof_v15 import PVLOFV15MemoryConfig, save_memory_config
from pv_anomaly.pvlof_v2 import load_calibration, save_calibration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--base-output", required=True)
    parser.add_argument("--memory-output", required=True)
    parser.add_argument("--minimum-relative-drop", type=float, default=0.05)
    parser.add_argument("--minimum-absolute-drop", type=float, default=0.50)
    parser.add_argument("--entry-consecutive", type=int, default=3)
    parser.add_argument("--recovery-consecutive", type=int, default=3)
    parser.add_argument("--expected-interval-minutes", type=int, default=5)
    parser.add_argument("--version", default="pvlof-v1.5-memory-5pct")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    if not 0 <= args.minimum_relative_drop < 1:
        raise ValueError("--minimum-relative-drop must be in [0, 1)")
    if args.minimum_absolute_drop < 0:
        raise ValueError("--minimum-absolute-drop must be non-negative")
    if min(args.entry_consecutive, args.recovery_consecutive) < 1:
        raise ValueError("consecutive counts must be positive")

    source = load_calibration(args.input)
    base = replace(
        source,
        version=args.version,
        minimum_isolated_relative_drop=args.minimum_relative_drop,
        minimum_isolated_absolute_drop=args.minimum_absolute_drop,
    )
    memory = PVLOFV15MemoryConfig(
        version=args.version,
        entry_consecutive=args.entry_consecutive,
        recovery_consecutive=args.recovery_consecutive,
        expected_interval_minutes=args.expected_interval_minutes,
    )
    save_calibration(base, args.base_output)
    save_memory_config(memory, args.memory_output)
    report = {
        "input": args.input,
        "source_version": source.version,
        "version": args.version,
        "minimum_isolated_relative_drop": base.minimum_isolated_relative_drop,
        "minimum_isolated_absolute_drop": base.minimum_isolated_absolute_drop,
        "memory": memory.to_dict(),
        "score_thresholds_refit": False,
        "collective_parameters_changed": False,
        "outputs": {"base": args.base_output, "memory": args.memory_output},
    }
    destination = Path(args.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
