"""Base protocol for model source connectors."""

from typing import Any, List, Optional, Protocol


class Connector(Protocol):
    """Protocol for reading model files (list + read with optional range)."""

    async def list_files(self) -> List[str]:
        """Return list of relative file paths (e.g. ['config.json', 'model.safetensors'])."""
        ...

    async def read_file(self, path: str, offset: int = 0, limit: Optional[int] = None) -> bytes:
        """Read file bytes. offset/limit allow range reads (e.g. for safetensors header)."""
        ...

    async def read_file_json(self, path: str) -> Any:
        """Read file and parse as JSON."""
        ...

    async def get_file_size(self, path: str) -> Optional[int]:
        """Return file size in bytes if known, else None. Used for streaming (e.g. pytorch_model.bin)."""
        ...
