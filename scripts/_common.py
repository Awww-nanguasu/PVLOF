"""Shared helpers for command-line scripts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from pv_anomaly.data.es_client import ReadOnlyESClient
from pv_anomaly.settings import ConfigurationError, ESSettings


def client_from_env() -> ReadOnlyESClient:
    try:
        return ReadOnlyESClient(ESSettings.from_env())
    except ConfigurationError as exc:
        raise SystemExit(f"Configuration error: {exc}. Copy .env.example to .env first.") from exc


def require_index(client: ReadOnlyESClient, requested: str | None) -> str:
    index = requested or client.settings.index
    if not index:
        raise SystemExit("No index provided. Set ES_INDEX or pass --index.")
    return index


def write_json(payload: Any, output: str | Path | None = None) -> None:
    content = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    if output is None:
        print(content)
        return
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content + "\n", encoding="utf-8")
    print(f"Wrote {path.resolve()}", file=sys.stderr)

