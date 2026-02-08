import json
import math
import struct
from dataclasses import dataclass
from typing import Any, Dict

from hf_mem.types import SafetensorsDtypes, get_safetensors_dtype_bytes


def parse_safetensors_header_bytes(data: bytes) -> Dict[str, Any]:
    """Parse safetensors header: first 8 bytes (little-endian uint64) = metadata size, then JSON."""
    if len(data) < 8:
        raise ValueError(f"Need at least 8 bytes for safetensors header, got {len(data)}")
    metadata_size = struct.unpack("<Q", data[:8])[0]
    if len(data) >= 8 + metadata_size:
        return json.loads(data[8 : 8 + metadata_size])
    raise ValueError(f"Not enough bytes: need 8+{metadata_size}, got {len(data)}")


@dataclass
class DtypeMetadata:
    param_count: int
    bytes_count: int


@dataclass
class ComponentMetadata:
    dtypes: Dict[SafetensorsDtypes, DtypeMetadata]
    param_count: int
    bytes_count: int


@dataclass
class SafetensorsMetadata:
    components: Dict[str, ComponentMetadata]
    param_count: int
    bytes_count: int


def parse_safetensors_metadata(
    raw_metadata: Dict[str, Dict[str, Any]],
) -> SafetensorsMetadata:
    components = {}
    total_param_count, total_bytes_count = 0, 0

    for name, metadata in raw_metadata.items():
        component = ComponentMetadata(dtypes={}, param_count=0, bytes_count=0)
        for key, value in metadata.items():
            if key in {"__metadata__"}:
                continue

            dtype = value["dtype"]
            if dtype not in component.dtypes:
                component.dtypes[dtype] = DtypeMetadata(param_count=0, bytes_count=0)

            dtype_bytes = get_safetensors_dtype_bytes(dtype)
            current_shape = math.prod(value["shape"])
            current_shape_bytes = current_shape * dtype_bytes

            component.dtypes[dtype].param_count += current_shape
            component.dtypes[dtype].bytes_count += current_shape_bytes
            component.param_count += current_shape
            component.bytes_count += current_shape_bytes
            total_param_count += current_shape
            total_bytes_count += current_shape_bytes

        components[name] = component

    return SafetensorsMetadata(
        components=components,
        param_count=total_param_count,
        bytes_count=total_bytes_count,
    )
