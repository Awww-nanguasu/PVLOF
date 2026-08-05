"""Verify read-only connectivity and show non-secret server metadata."""

from __future__ import annotations

from _common import client_from_env, write_json


def main() -> None:
    client = client_from_env()
    info = client.info()
    write_json(
        {
            "connected": True,
            "endpoint": client.settings.url,
            "cluster_name": info.get("cluster_name"),
            "cluster_uuid": info.get("cluster_uuid"),
            "version": info.get("version", {}),
            "tagline": info.get("tagline"),
        }
    )


if __name__ == "__main__":
    main()

