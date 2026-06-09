"""AWS S3 connector."""

import asyncio
import json
from typing import Any, List

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError as e:
    boto3 = None  # type: ignore[assignment]
    ClientError = Exception  # type: ignore[assignment, misc]

_S3_IMPORT_ERROR = "S3 connector requires boto3. Install with: pip install hf-mem[s3]"


class S3Connector:
    """Read model files from an AWS S3 bucket (prefix = folder path)."""

    def __init__(self, bucket: str, prefix: str = "") -> None:
        if boto3 is None:
            raise RuntimeError(_S3_IMPORT_ERROR)
        self._bucket = bucket
        self._prefix = prefix.rstrip("/")
        if self._prefix:
            self._prefix += "/"
        self._client = boto3.client("s3")

    def _key(self, path: str) -> str:
        return f"{self._prefix}{path}" if self._prefix else path

    async def list_files(self) -> List[str]:
        def _list() -> List[str]:
            paths: List[str] = []
            paginator = self._client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self._bucket, Prefix=self._prefix):
                for obj in page.get("Contents") or []:
                    key = obj["Key"]
                    if key.endswith("/"):
                        continue
                    rel = key[len(self._prefix) :] if self._prefix else key
                    paths.append(rel)
            return paths

        return await asyncio.to_thread(_list)

    async def read_file(self, path: str, offset: int = 0, limit: int | None = None) -> bytes:
        key = self._key(path)

        def _read() -> bytes:
            extra = {}
            if limit is not None:
                end = offset + limit - 1
                extra["Range"] = f"bytes={offset}-{end}"
            try:
                r = self._client.get_object(Bucket=self._bucket, Key=key, **extra)
                return r["Body"].read()
            except ClientError as e:
                if e.response["Error"]["Code"] == "NoSuchKey":
                    raise FileNotFoundError(f"No such object: {path}") from e
                raise

        return await asyncio.to_thread(_read)

    async def read_file_json(self, path: str) -> Any:
        data = await self.read_file(path)
        return json.loads(data.decode("utf-8"))
