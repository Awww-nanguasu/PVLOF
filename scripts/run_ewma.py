"""Apply a saved EWMA calibration to point-level residuals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pv_anomaly.ewma import apply_ewma, load_calibration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    calibration = load_calibration(args.calibration)
    frame = pd.read_parquet(args.input)
    scored = apply_ewma(frame, calibration)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(output, index=False)
    eligible = scored["ewma_eligible"].astype(bool)
    report = {
        "input": args.input,
        "output": args.output,
        "samples": len(scored),
        "eligible_samples": int(eligible.sum()),
        "eligible_percent": float(eligible.mean() * 100),
        "raw_alerts": int(scored["ewma_raw_alert"].sum()),
        "alerts": int(scored["ewma_alert"].sum()),
        "alert_percent_of_all": float(scored["ewma_alert"].mean() * 100),
        "alert_devices": int(
            scored.loc[scored["ewma_alert"].astype(bool), "device_no"].nunique()
        ),
        "calibration": calibration.to_dict(),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
