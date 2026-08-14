"""Export point-level Transformer predictions and residuals to Parquet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pv_anomaly.models.residuals import predict_checkpoint_frame
from pv_anomaly.models.training import load_training_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/transformer.yaml")
    parser.add_argument("--checkpoint", default="artifacts/models/transformer/best.pt")
    parser.add_argument("--output-directory", default="artifacts/models/transformer/residuals")
    parser.add_argument("--splits", nargs="+", default=["validation", "test"])
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    config = load_training_config(args.config)
    output = Path(args.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for split in args.splits:
        if split not in {"train", "validation", "test"}:
            raise SystemExit(f"Unsupported split: {split}")
        frame = predict_checkpoint_frame(
            args.checkpoint,
            config["data"][split],
            split=split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            requested_device=args.device,
        )
        path = output / f"{split}.parquet"
        frame.to_parquet(path, index=False)
        summaries[split] = {
            "path": str(path),
            "rows": len(frame),
            "time_start": frame["target_time"].min().isoformat(),
            "time_end": frame["target_time"].max().isoformat(),
        }
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
