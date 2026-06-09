"""Azure Blob Storage connector."""

import asyncio
import json
import os
from typing import Any, List

try:
    from azure.storage.blob import BlobServiceClient
    from azure.core.exceptions import ResourceNotFoundError
except ImportError:
    BlobServiceClient = None  # type: ignore[assignment, misc]
    ResourceNotFoundError = Exception  # type: ignore[assignment, misc]

_AZURE_IMPORT_ERROR = "Azure connector requires azure-storage-blob. Install with: pip install hf-mem[azure]"


def _create_blob_service_client(account: str | None) -> BlobServiceClient:
    if BlobServiceClient is None:
        raise RuntimeError(_AZURE_IMPORT_ERROR)
    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if conn_str:
        return BlobServiceClient.from_connection_string(conn_str)
    account_name = account or os.getenv("AZURE_STORAGE_ACCOUNT")
    if not account_name:
        raise RuntimeError(
            "Azure connector requires AZURE_STORAGE_CONNECTION_STRING or --azure-account (or AZURE_STORAGE_ACCOUNT)"
        )
    try:
        from azure.identity import DefaultAzureCredential
    except ImportError:
        raise RuntimeError(
            "Azure connector with account name requires azure-identity. Install with: pip install hf-mem[azure]"
        )
    account_url = f"https://{account_name.strip()}.blob.core.windows.net"
    return BlobServiceClient(account_url=account_url, credential=DefaultAzureCredential())


class AzureConnector:
    """Read model files from an Azure Blob Storage container (prefix = folder path)."""

    def __init__(self, container: str, prefix: str = "", account: str | None = None) -> None:
        self._container_name = container
        self._prefix = prefix.rstrip("/")
        if self._prefix:
            self._prefix += "/"
        self._client = _create_blob_service_client(account)
        self._container = self._client.get_container_client(container)

    def _blob_name(self, path: str) -> str:
        return f"{self._prefix}{path}" if self._prefix else path

    async def list_files(self) -> List[str]:
        def _list() -> List[str]:
            paths: List[str] = []
            for blob in self._container.list_blobs(name_starts_with=self._prefix):
                if blob.name.endswith("/"):
                    continue
                rel = blob.name[len(self._prefix) :] if self._prefix else blob.name
                paths.append(rel)
            return paths

        return await asyncio.to_thread(_list)

    async def read_file(self, path: str, offset: int = 0, limit: int | None = None) -> bytes:
        name = self._blob_name(path)

        def _read() -> bytes:
            blob_client = self._container.get_blob_client(name)
            try:
                if limit is not None:
                    stream = blob_client.download_blob(offset=offset, length=limit)
                elif offset == 0:
                    stream = blob_client.download_blob()
                else:
                    stream = blob_client.download_blob(offset=offset)
                return stream.readall()
            except ResourceNotFoundError as e:
                raise FileNotFoundError(f"No such blob: {path}") from e

        return await asyncio.to_thread(_read)

    async def read_file_json(self, path: str) -> Any:
        data = await self.read_file(path)
        return json.loads(data.decode("utf-8"))
