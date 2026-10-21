"""Launch the existing interactive wizard inside the service container."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def prepare_wizard(
    argv: list[str],
    env_file: str | Path = "/app/PVLOF-V3-service/.env",
) -> list[str]:
    """Reuse the service key privately and save results on the host bind mount."""
    load_dotenv(dotenv_path=env_file, override=False)
    if not os.getenv("PVLOF_CLIENT_API_KEY", "").strip():
        os.environ["PVLOF_CLIENT_API_KEY"] = os.getenv("PVLOF_API_KEY", "")

    arguments = list(argv)
    if not any(
        item == "--output-directory" or item.startswith("--output-directory=")
        for item in arguments
    ):
        arguments.extend(["--output-directory", "/app/outputs/pvlof_v3_cli"])
    return arguments


def main() -> None:
    sys.argv[1:] = prepare_wizard(sys.argv[1:])
    # Import after loading the environment. Do not put secrets in argv/logs.
    from pvlof_v3_service.wizard import main as wizard_main

    wizard_main()


if __name__ == "__main__":
    main()
