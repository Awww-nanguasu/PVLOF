"""Export a bounded Elasticsearch sample to a local JSON file."""

from __future__ import annotations

import argparse

from _common import client_from_env, require_index, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", help="Overrides ES_INDEX")
    parser.add_argument("--size", type=int, default=100, help="1-1000 documents (default: 100)")
    parser.add_argument("--output", default="data/raw/es_sample.json")
    args = parser.parse_args()
    client = client_from_env()
    write_json(client.sample(require_index(client, args.index), size=args.size), args.output)


if __name__ == "__main__":
    main()

