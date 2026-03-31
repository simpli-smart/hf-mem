from typing import Literal

SafetensorsDtypes = Literal[
    "F64",
    "I64",
    "U64",
    "F32",
    "I32",
    "U32",
    "F16",
    "BF16",
    "I16",
    "U16",
    "F8_E5M2",  # NOTE: Only CUDA +11.8
    "F8_E4M3",  # NOTE: CUDA +11.8 and AMD ROCm
    "I8",
    "U8",
]


def get_safetensors_dtype_bytes(dtype: SafetensorsDtypes | str) -> int:
    match dtype:
        case "F64" | "I64" | "U64":
            return 8
        case "F32" | "I32" | "U32":
            return 4
        case "F16" | "BF16" | "I16" | "U16":
            return 2
        case "F8_E5M2" | "F8_E4M3" | "I8" | "U8":
            return 1
        case _:
            raise RuntimeError(f"DTYPE={dtype} NOT HANDLED")


TorchDtypes = Literal[
    "float32", "float16", "bfloat16", "float8_e4m3", "float8_e4m3fn", "float8_e5m2",
    "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "float64",
]


def torch_dtype_to_safetensors_dtype(dtype: TorchDtypes | str) -> SafetensorsDtypes:
    if hasattr(dtype, "name"):
        dtype = getattr(dtype, "name", str(dtype))
    if isinstance(dtype, str) and dtype.startswith("torch."):
        dtype = dtype.replace("torch.", "")
    match dtype:
        case "float32":
            return "F32"
        case "float64":
            return "F64"
        case "float16":
            return "F16"
        case "bfloat16":
            return "BF16"
        case "float8_e4m3" | "float8_e4m3fn":
            return "F8_E4M3"
        case "float8_e5m2":
            return "F8_E5M2"
        case "int8":
            return "I8"
        case "uint8":
            return "U8"
        case "int16":
            return "I16"
        case "uint16":
            return "U16"
        case "int32":
            return "I32"
        case "uint32":
            return "U32"
        case "int64":
            return "I64"
        case "uint64":
            return "U64"
        case _:
            return "F16"
