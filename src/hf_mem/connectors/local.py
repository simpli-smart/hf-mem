"""Local filesystem connector."""

import asyncio
import json
import os
from typing import Any, List


class LocalConnector:
    """Read model files from a local directory (e.g. downloaded from HF)."""

    def __init__(self, folder: str) -> None:
        self._folder = os.path.abspath(os.path.expanduser(folder))
        if not os.path.isdir(self._folder):
            raise RuntimeError(f"Not a directory: {self._folder}")

    async def list_files(self) -> List[str]:
        def _list() -> List[str]:
            paths: List[str] = []
            for root, _dirs, files in os.walk(self._folder, topdown=True):
                rel = os.path.relpath(root, self._folder)
                if rel == ".":
                    rel = ""
                for name in files:
                    p = os.path.join(rel, name).replace("\\", "/") if rel else name
                    paths.append(p)
            return paths

        return await asyncio.to_thread(_list)

    async def read_file(self, path: str, offset: int = 0, limit: int | None = None) -> bytes:
        full = os.path.join(self._folder, path)
        if not os.path.isfile(full):
            raise FileNotFoundError(f"No such file: {path}")

        def _read() -> bytes:
            with open(full, "rb") as f:
                f.seek(offset)
                return f.read(limit) if limit is not None else f.read()

        return await asyncio.to_thread(_read)

    async def read_file_json(self, path: str) -> Any:
        data = await self.read_file(path)
        return json.loads(data.decode("utf-8"))
