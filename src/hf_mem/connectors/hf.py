"""Hugging Face Hub connector."""

import os
from typing import Any, Dict, List, Optional
from uuid import uuid4

import httpx

REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", 30.0))


class HFConnector:
    """Read model files from the Hugging Face Hub via API and resolve URLs."""

    def __init__(
        self,
        model_id: str,
        revision: str = "main",
        client: Optional[httpx.AsyncClient] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self._client = client
        self._headers = headers or {}
        self._own_client = client is None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(REQUEST_TIMEOUT),
                http2=True,
                follow_redirects=True,
            )
        return self._client

    def _url(self, path: str) -> str:
        return f"https://huggingface.co/{self.model_id}/resolve/{self.revision}/{path}"

    def _api_url(self) -> str:
        return f"https://huggingface.co/api/models/{self.model_id}/tree/{self.revision}?recursive=true"

    async def list_files(self) -> List[str]:
        client = self._get_client()
        r = await client.get(self._api_url(), headers=self._headers, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        return [f["path"] for f in data if f.get("path") and f.get("type") == "file"]

    async def read_file(self, path: str, offset: int = 0, limit: int | None = None) -> bytes:
        client = self._get_client()
        url = self._url(path)
        headers = dict(self._headers)
        if limit is not None:
            end = offset + limit - 1
            headers["Range"] = f"bytes={offset}-{end}"
        r = await client.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.content

    async def read_file_json(self, path: str) -> Any:
        client = self._get_client()
        r = await client.get(self._url(path), headers=self._headers, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.json()

    async def close(self) -> None:
        if self._own_client and self._client is not None:
            await self._client.aclose()
            self._client = None


def make_hf_headers(model_id: str, revision: str) -> Dict[str, str]:
    """Build HF request headers (User-Agent + optional token from env or HF_HOME)."""
    headers: Dict[str, str] = {
        "User-Agent": f"hf-mem/0.4; id={uuid4()}; model_id={model_id}; revision={revision}",
    }
    if token := os.getenv("HF_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    else:
        path = os.getenv("HF_HOME", ".cache/huggingface")
        fn = os.path.join(os.path.expanduser("~"), path, "token") if not os.path.isabs(path) else os.path.join(path, "token")
        if os.path.exists(fn):
            with open(fn, "r", encoding="utf-8") as f:
                headers["Authorization"] = f"Bearer {f.read().strip()}"
    return headers
