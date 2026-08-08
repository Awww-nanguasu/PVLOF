"""Audit physical string-channel inventories without assuming contiguous numbering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pv_anomaly.pvlof_channels import infer_channel_inventory
from pv_anomaly.pvlof_io import read_pvlof_source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--timezone", default="Asia/Shanghai")
    parser.add_argument("--minimum-evidence-samples", type=int, default=3)
    parser.add_argument("--minimum-evidence-days", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    frame, input_report = read_pvlof_source(
        args.input, start=args.start, end=args.end, timezone=args.timezone
    )
    inventory, audit = infer_channel_inventory(
        frame,
        minimum_evidence_samples=args.minimum_evidence_samples,
        minimum_evidence_days=args.minimum_evidence_days,
        timezone=args.timezone,
    )
    report = {"input": input_report, "inventory": inventory, "audit": audit}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        key: audit[key]
        for key in (
            "devices",
            "configured_channels",
            "devices_with_count_mismatch",
            "devices_with_noncontiguous_channels",
        )
    }
    summary["output"] = str(output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
