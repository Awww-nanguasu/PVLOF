"""List indices, aliases and data streams visible to the read-only account."""

from __future__ import annotations

import argparse

from _common import client_from_env, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="Optional local JSON output path")
    args = parser.parse_args()
    write_json(client_from_env().list_indices(), args.output)


if __name__ == "__main__":
    main()

