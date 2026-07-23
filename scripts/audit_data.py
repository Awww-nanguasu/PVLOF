"""Generate a local data-quality and algorithm-feasibility report."""

from __future__ import annotations

import argparse

from _common import write_json
from pv_anomaly.audit import audit_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="JSON, JSONL, NDJSON or CSV input")
    parser.add_argument("--config", default="configs/data.example.yaml")
    parser.add_argument("--output", default="artifacts/reports/data_audit.json")
    args = parser.parse_args()
    write_json(audit_file(args.input, args.config), args.output)


if __name__ == "__main__":
    main()

