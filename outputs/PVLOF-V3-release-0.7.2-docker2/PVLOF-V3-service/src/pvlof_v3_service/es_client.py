"""Minimal Elasticsearch REST transport that rejects mutating methods."""

from __future__ import annotations

import base64
import json
import re
import ssl
from collections.abc import Callable, Iterator
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from pvlof_v3_service.config import ESConnectionSettings


SAFE_INDEX = re.compile(r"[A-Za-z0-9_.-]+")
SAFE_FIELD = re.compile(r"[A-Za-z0-9_.@-]+")


class ESReadError(RuntimeError):
    """Raised when a read-only Elasticsearch request fails."""

    def __init__(self, message: str, *, status: int | None = None):
        super().__init__(message)
        self.status = status


class ReadOnlyESClient:
    """Elasticsearch 7.x REST client restricted to GET and HEAD."""

    _ALLOWED_METHODS = frozenset({"GET", "HEAD"})

    def __init__(self, settings: ESConnectionSettings):
        self.settings = settings
        self._ssl_context = self._build_ssl_context(settings)

    @staticmethod
    def _build_ssl_context(
        settings: ESConnectionSettings,
    ) -> ssl.SSLContext | None:
        if not settings.url.startswith("https://"):
            return None
        if not settings.verify_certs:
            return ssl._create_unverified_context()  # noqa: SLF001
        return ssl.create_default_context(
            cafile=str(settings.ca_cert) if settings.ca_cert else None
        )

    @staticmethod
    def _index_path(index: str) -> str:
        if not SAFE_INDEX.fullmatch(index):
            raise ValueError("Elasticsearch index contains unsupported characters")
        return quote(index, safe="._-")

    @staticmethod
    def _validate_fields(fields: list[str]) -> None:
        invalid = [field for field in fields if not SAFE_FIELD.fullmatch(field)]
        if invalid:
            raise ValueError(f"Unsupported Elasticsearch field names: {invalid}")

    def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, str | int | bool] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> Any:
        method = method.upper()
        if method not in self._ALLOWED_METHODS:
            raise PermissionError(f"ReadOnlyESClient rejects HTTP method {method}")
        normalized_path = "/" + path.lstrip("/")
        query = urlencode(params or {})
        url = f"{self.settings.url}{normalized_path}"
        if query:
            url = f"{url}?{query}"

        credentials = (
            f"{self.settings.username}:{self.settings.password}".encode("utf-8")
        )
        token = base64.b64encode(credentials).decode("ascii")
        data = json.dumps(json_body).encode("utf-8") if json_body is not None else None
        request = Request(
            url,
            data=data,
            method=method,
            headers={
                "Accept": "application/json",
                "Authorization": f"Basic {token}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urlopen(  # noqa: S310 - URL is validated configuration
                request,
                timeout=self.settings.timeout_seconds,
                context=self._ssl_context,
            ) as response:
                payload = response.read()
                if not payload:
                    return None
                return json.loads(payload)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            raise ESReadError(
                f"Elasticsearch returned HTTP {exc.code}: {detail}",
                status=exc.code,
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise ESReadError(
                f"Unable to reach Elasticsearch at {self.settings.url}: {exc}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise ESReadError("Elasticsearch returned a non-JSON response") from exc

    def info(self) -> dict[str, Any]:
        body = self.request("GET", "/")
        if not isinstance(body, dict):
            raise ESReadError("Elasticsearch root response is not an object")
        return body

    def index_exists(self, index: str) -> bool:
        """Return whether an index is readable without issuing a write request."""

        index_path = self._index_path(index)
        try:
            self.request("HEAD", f"/{index_path}")
            return True
        except ESReadError as exc:
            if exc.status == 404:
                return False
            raise

    def search(self, index: str, body: dict[str, Any]) -> dict[str, Any]:
        """Execute a bounded read-only `_search` request."""

        index_path = self._index_path(index)
        response = self.request(
            "GET",
            f"/{index_path}/_search",
            json_body=body,
        )
        if not isinstance(response, dict):
            raise ESReadError("Elasticsearch search response is not an object")
        return response

    def iter_search(
        self,
        index: str,
        *,
        query: dict[str, Any],
        source_fields: list[str],
        sort_fields: list[str],
        max_documents: int | None = None,
        progress_callback: Callable[[int], None] | None = None,
    ) -> Iterator[list[dict[str, Any]]]:
        """Yield `_source` documents with bounded search_after pagination."""

        if not sort_fields:
            raise ValueError("sort_fields must not be empty")
        self._validate_fields([*source_fields, *sort_fields])
        index_path = self._index_path(index)
        search_after: list[Any] | None = None
        document_count = 0
        document_limit = (
            self.settings.max_documents
            if max_documents is None
            else int(max_documents)
        )
        if document_limit <= 0:
            raise ValueError("max_documents must be positive")

        while True:
            body: dict[str, Any] = {
                "size": self.settings.page_size,
                "query": query,
                "_source": source_fields,
                "sort": [{field: "asc"} for field in sort_fields],
                "track_total_hits": False,
            }
            if search_after is not None:
                body["search_after"] = search_after

            response = self.request(
                "GET",
                f"/{index_path}/_search",
                json_body=body,
            )
            hits = (response or {}).get("hits", {}).get("hits", [])
            if not hits:
                return

            sources: list[dict[str, Any]] = []
            for hit in hits:
                source = hit.get("_source")
                if not isinstance(source, dict):
                    raise ESReadError("Elasticsearch hit is missing an object _source")
                sources.append(source)

            document_count += len(sources)
            if document_count > document_limit:
                raise ESReadError(
                    "Elasticsearch query exceeded max_documents="
                    f"{document_limit}"
                )

            if progress_callback is not None:
                progress_callback(document_count)

            last_sort = hits[-1].get("sort")
            if not isinstance(last_sort, list) or last_sort == search_after:
                raise ESReadError(
                    "search_after did not advance; source data may violate the "
                    "configured sort-key uniqueness contract"
                )
            yield sources
            if len(hits) < self.settings.page_size:
                return
            search_after = last_sort
