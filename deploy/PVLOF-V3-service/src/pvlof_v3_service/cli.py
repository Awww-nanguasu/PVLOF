"""Console entry point."""

from __future__ import annotations

import uvicorn

from pvlof_v3_service.api import create_app
from pvlof_v3_service.config import ServiceSettings


def main() -> None:
    settings = ServiceSettings.from_env()
    uvicorn.run(
        create_app(settings=settings),
        host=settings.api_host,
        port=settings.api_port,
        workers=1,
        access_log=True,
    )


if __name__ == "__main__":
    main()

