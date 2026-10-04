"""Read document count and minimum/maximum timestamp from an index."""

from __future__ import annotations

import argparse

from _common import client_from_env, require_index, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", help="Overrides ES_INDEX")
    parser.add_argument("--field", required=True, help="Mapped date field")
    parser.add_argument("--output", help="Optional local JSON output path")
    args = parser.parse_args()
    client = client_from_env()
    write_json(client.time_range(require_index(client, args.index), args.field), args.output)


if __name__ == "__main__":
    main()
