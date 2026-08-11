"""Evaluate a saved PowerTransformer checkpoint on one Parquet split."""

from __future__ import annotations

import argparse
import json

from pv_anomaly.models.training import evaluate_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="artifacts/models/transformer/best.pt")
    parser.add_argument("--data", default="data/processed/transformer/test.parquet")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    metrics = evaluate_checkpoint(
        args.checkpoint,
        args.data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        requested_device=args.device,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

