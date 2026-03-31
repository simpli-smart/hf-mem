"""
Read PyTorch .bin file metadata (tensor names, shapes, dtypes) without loading PyTorch.

PyTorch 1.6+ saves to a ZIP with data.pkl. We unpickle with stub classes so no torch import is needed.
Supports streaming (range reads) to avoid loading the full file into memory.
"""

import io
import pickle
import struct
import zipfile
from typing import Any, Awaitable, Callable, Dict

# ZIP constants
_EOCD_SIG = 0x06054B50
_CD_ENTRY_SIG = 0x02014B50
_PKL_NAMES = ("data.pkl", "data")
_TAIL_SIZE = 65560  # read last ~64K to find EOCD (max comment 65535 + EOCD 22)


# Stub dtypes: objects with .name for torch_dtype_to_safetensors_dtype
class _Dtype:
    def __init__(self, name: str) -> None:
        self.name = name


_STORAGE_DTYPE_MAP = {
    "FloatStorage": _Dtype("float32"),
    "DoubleStorage": _Dtype("float64"),
    "HalfStorage": _Dtype("float16"),
    "BFloat16Storage": _Dtype("bfloat16"),
    "LongStorage": _Dtype("int64"),
    "IntStorage": _Dtype("int32"),
    "ShortStorage": _Dtype("int16"),
    "CharStorage": _Dtype("int8"),
    "ByteStorage": _Dtype("uint8"),
}


class _FakeStorage:
    """Stub for torch.*Storage. Only dtype is needed for memory estimation."""

    dtype = _Dtype("float32")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


def _make_storage_cls(dtype: _Dtype) -> type:
    """Create a storage class that unpickle can instantiate."""
    return type("Storage", (_FakeStorage,), {"dtype": dtype})


# One stub storage per dtype; pickle will reference e.g. torch.FloatStorage
_TORCH_STORAGES = {name: _make_storage_cls(d) for name, d in _STORAGE_DTYPE_MAP.items()}


class _FakeTensor:
    """Minimal tensor-like object with shape and dtype for metadata only."""

    __slots__ = ("shape", "dtype")

    def __init__(self, shape: tuple, dtype: _Dtype) -> None:
        self.shape = shape
        self.dtype = dtype


def _rebuild_tensor_v2(
    storage: _FakeStorage,
    storage_offset: int,
    size: tuple,
    stride: tuple,
    requires_grad: bool = False,
    backward_hooks: Any = None,
) -> _FakeTensor:
    """Stub for torch._utils._rebuild_tensor_v2. Returns fake tensor with shape and dtype."""
    dtype = getattr(storage, "dtype", None) or _Dtype("float32")
    return _FakeTensor(tuple(size), dtype)


# Build a fake torch module so find_class('torch', 'FloatStorage') etc. work
class _FakeTorch:
    FloatStorage = _TORCH_STORAGES["FloatStorage"]
    DoubleStorage = _TORCH_STORAGES["DoubleStorage"]
    HalfStorage = _TORCH_STORAGES["HalfStorage"]
    BFloat16Storage = _TORCH_STORAGES["BFloat16Storage"]
    LongStorage = _TORCH_STORAGES["LongStorage"]
    IntStorage = _TORCH_STORAGES["IntStorage"]
    ShortStorage = _TORCH_STORAGES["ShortStorage"]
    CharStorage = _TORCH_STORAGES["CharStorage"]
    ByteStorage = _TORCH_STORAGES["ByteStorage"]

    class _utils:
        _rebuild_tensor_v2 = staticmethod(_rebuild_tensor_v2)

    # torch.Size is often just a tuple in pickle
    Size = tuple


_fake_torch = _FakeTorch()


class _Placeholder:
    """Accepts any constructor args (for torch.device, etc. in pickle)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


class _StubUnpickler(pickle.Unpickler):
    """Redirect torch.* and torch._utils.* to our stubs so we can unpickle without PyTorch."""

    def find_class(self, module: str, name: str) -> Any:
        if module == "torch" and name in _TORCH_STORAGES:
            return _TORCH_STORAGES[name]
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild_tensor_v2
        if module == "torch" and name == "Size":
            return tuple
        if module.startswith("torch"):
            return _Placeholder
        return super().find_class(module, name)


def _find_eocd(tail: bytes) -> tuple[int, int] | None:
    """Find EOCD in the last bytes of the file. Returns (cd_offset, cd_size) or None."""
    # EOCD is at the end; search for signature from the end
    sig = struct.pack("<I", _EOCD_SIG)
    idx = tail.rfind(sig)
    if idx == -1:
        return None
    eocd = tail[idx:]
    if len(eocd) < 22:
        return None
    cd_size = struct.unpack("<I", eocd[12:16])[0]
    cd_offset = struct.unpack("<I", eocd[16:20])[0]
    return (cd_offset, cd_size)


def _find_pkl_in_cd(cd_bytes: bytes) -> tuple[int, int] | None:
    """
    Parse central directory and find the data.pkl / data entry.
    Returns (data_offset, compressed_size) for that entry, or None.
    """
    pos = 0
    while pos + 46 <= len(cd_bytes):
        if cd_bytes[pos : pos + 4] != struct.pack("<I", _CD_ENTRY_SIG):
            break
        compressed_size = struct.unpack("<I", cd_bytes[pos + 20 : pos + 24])[0]
        filename_len = struct.unpack("<H", cd_bytes[pos + 28 : pos + 30])[0]
        extra_len = struct.unpack("<H", cd_bytes[pos + 30 : pos + 32])[0]
        comment_len = struct.unpack("<H", cd_bytes[pos + 32 : pos + 34])[0]
        local_header_offset = struct.unpack("<I", cd_bytes[pos + 42 : pos + 46])[0]
        filename = cd_bytes[pos + 46 : pos + 46 + filename_len].decode("utf-8", errors="replace")
        if filename in _PKL_NAMES:
            # Local file header is 30 + filename + extra
            data_offset = local_header_offset + 30 + filename_len + extra_len
            return (data_offset, compressed_size)
        pos += 46 + filename_len + extra_len + comment_len
    return None


async def load_pytorch_bin_metadata_streaming(
    read_range: Callable[[int, int], Awaitable[bytes]],
    get_size: Callable[[], Awaitable[int | None]],
) -> Dict[str, Any] | None:
    """
    Load state-dict metadata using range reads only (no full file in memory).
    Returns state_dict on success, None if size unknown or not a zip (caller should full-read).
    """
    size = await get_size()
    if size is None or size <= 0:
        return None
    tail_len = min(_TAIL_SIZE, size)
    tail = await read_range(size - tail_len, tail_len)
    if len(tail) < 22 or tail[:2] != b"PK":
        return None
    eocd_result = _find_eocd(tail)
    if eocd_result is None:
        return None
    cd_offset, cd_size = eocd_result
    if cd_size <= 0 or cd_offset < 0 or cd_offset + cd_size > size:
        return None
    cd_bytes = await read_range(cd_offset, cd_size)
    pkl_range = _find_pkl_in_cd(cd_bytes)
    if pkl_range is None:
        return None
    data_offset, compressed_size = pkl_range
    if data_offset + compressed_size > size:
        return None
    pkl_bytes = await read_range(data_offset, compressed_size)
    state_dict = _StubUnpickler(io.BytesIO(pkl_bytes)).load()
    if not isinstance(state_dict, dict):
        raise ValueError("Expected state dict (dict), got %s" % type(state_dict).__name__)
    return state_dict


def load_pytorch_bin_metadata(data: bytes) -> Dict[str, Any]:
    """
    Load state-dict-like metadata (name -> fake tensor with .shape, .dtype) from a .bin file.
    No PyTorch dependency. Works for zip-based format (PyTorch 1.6+).
    """
    if data[:2] != b"PK":
        # Legacy format: single pickle file. Try stub unpickler anyway; may fail on old format.
        state_dict = _StubUnpickler(io.BytesIO(data)).load()
    else:
        with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
            # PyTorch uses "data" as the pickle entry (no .pkl extension in some versions)
            names = zf.namelist()
            pkl_name = "data.pkl" if "data.pkl" in names else "data"
            if pkl_name not in names:
                raise ValueError("Zip has no data.pkl or data entry")
            with zf.open(pkl_name) as f:
                state_dict = _StubUnpickler(io.BytesIO(f.read())).load()

    if not isinstance(state_dict, dict):
        raise ValueError("Expected state dict (dict), got %s" % type(state_dict).__name__)

    return state_dict
