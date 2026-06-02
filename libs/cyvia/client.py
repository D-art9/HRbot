from __future__ import annotations
from typing import Any

import httpx


class CyviaClient:
    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "http://127.0.0.1:8000/api",
        timeout: float = 120.0,
        http_client: httpx.Client | None = None,
    ):
        self._base = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._own_client = http_client is None
        self._client = http_client or httpx.Client(
            timeout=timeout,
            limits=httpx.Limits(keepalive_expiry=30.0),
        )

    def post_json(self, path: str, body: dict[str, Any]) -> httpx.Response:
        url = f"{self._base}{path}" if path.startswith("/") else f"{self._base}/{path}"
        return self._client.post(url, json=body, headers=self._headers)

    def get_json(self, path: str) -> httpx.Response:
        url = f"{self._base}{path}" if path.startswith("/") else f"{self._base}/{path}"
        return self._client.get(url, headers=self._headers)

    def close(self) -> None:
        if self._own_client:
            self._client.close()

    def __enter__(self) -> CyviaClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
