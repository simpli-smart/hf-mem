"""Google Cloud Storage connector."""

import asyncio
import json
from typing import Any, List

try:
    from google.cloud import storage
except ImportError:
    storage = None  # type: ignore[assignment]

_GCS_IMPORT_ERROR = "GCS connector requires google-cloud-storage. Install with: pip install hf-mem[gcs]"


class GCSConnector:
    """Read model files from a Google Cloud Storage bucket (prefix = folder path)."""

    def __init__(self, bucket: str, prefix: str = "") -> None:
        if storage is None:
            raise RuntimeError(_GCS_IMPORT_ERROR)
        self._bucket_name = bucket
        self._prefix = prefix.rstrip("/")
        if self._prefix:
            self._prefix += "/"
        self._client = storage.Client()
        self._bucket = self._client.bucket(bucket)

    async def list_files(self) -> List[str]:
        def _list() -> List[str]:
            paths: List[str] = []
            for blob in self._bucket.list_blobs(prefix=self._prefix):
                if blob.name.endswith("/"):
                    continue
                rel = blob.name[len(self._prefix) :] if self._prefix else blob.name
                paths.append(rel)
            return paths

        return await asyncio.to_thread(_list)

    async def read_file(self, path: str, offset: int = 0, limit: int | None = None) -> bytes:
        key = f"{self._prefix}{path}" if self._prefix else path
        blob = self._bucket.blob(key)

        def _read() -> bytes:
            if limit is not None:
                end = offset + limit
                return blob.download_as_bytes(start=offset, end=end - 1)
            if offset == 0:
                return blob.download_as_bytes()
            return blob.download_as_bytes(start=offset)

        return await asyncio.to_thread(_read)

    async def read_file_json(self, path: str) -> Any:
        data = await self.read_file(path)
        return json.loads(data.decode("utf-8"))
