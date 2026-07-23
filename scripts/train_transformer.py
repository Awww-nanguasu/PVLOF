"""Train PowerTransformer with early stopping and save its best checkpoint."""

from __future__ import annotations

import argparse
import json

from pv_anomaly.models.training import load_training_config, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/transformer.yaml")
    args = parser.parse_args()
    summary = train(load_training_config(args.config))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

