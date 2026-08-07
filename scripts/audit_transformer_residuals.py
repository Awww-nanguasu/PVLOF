"""Audit Transformer residuals by local date, device and actual-power range."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pv_anomaly.models.residuals import audit_residual_frame


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", default="artifacts/models/transformer/residuals")
    parser.add_argument("--output-directory", default="artifacts/reports/transformer_residuals")
    parser.add_argument("--splits", nargs="+", default=["validation", "test"])
    parser.add_argument("--timezone", default="Asia/Shanghai")
    args = parser.parse_args()

    source = Path(args.input_directory)
    output = Path(args.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {}
    for split in args.splits:
        frame = pd.read_parquet(source / f"{split}.parquet")
        split_report, tables = audit_residual_frame(frame, timezone=args.timezone)
        report[split] = split_report
        for table_name, table in tables.items():
            table.to_csv(output / f"{split}_{table_name}.csv", index=False)

    report_path = output / "summary.json"
    content = json.dumps(report, ensure_ascii=False, indent=2, default=_json_default)
    report_path.write_text(content + "\n", encoding="utf-8")
    print(content)


if __name__ == "__main__":
    main()
