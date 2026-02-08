import argparse
import asyncio
import os
import struct
import warnings
from dataclasses import asdict
from functools import reduce
from typing import Any, Dict, List, Optional

import httpx

from hf_mem.connectors import Connector, GCSConnector, HFConnector, LocalConnector, S3Connector
from hf_mem.connectors.hf import make_hf_headers
from hf_mem.metadata import parse_safetensors_header_bytes, parse_safetensors_metadata
from hf_mem.print import print_report
from hf_mem.types import TorchDtypes, get_safetensors_dtype_bytes, torch_dtype_to_safetensors_dtype

# NOTE: Defines the bytes that will be fetched per safetensors file, but the metadata
# can indeed be larger than that
MAX_METADATA_SIZE = 100_000
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", 30.0))
MAX_CONCURRENCY = int(os.getenv("MAX_WORKERS", min(32, (os.cpu_count() or 1) + 4)))


async def read_safetensors_metadata_from_connector(connector: Connector, path: str) -> Dict[str, Any]:
    """Read safetensors header from a connector (supports two-chunk read for large metadata)."""
    data = await connector.read_file(path, 0, MAX_METADATA_SIZE)
    metadata_size = struct.unpack("<Q", data[:8])[0]
    if 8 + metadata_size <= len(data):
        return parse_safetensors_header_bytes(data)
    rest = await connector.read_file(path, MAX_METADATA_SIZE, metadata_size + 8 - MAX_METADATA_SIZE)
    return parse_safetensors_header_bytes(data[:MAX_METADATA_SIZE] + rest)


async def get_modules_and_dense_metadata_from_connector(
    connector: Connector, file_paths: List[str]
) -> Dict[str, Any]:
    """Read Dense module safetensors metadata (sentence-transformers layout) via connector."""
    dense_metadata: Dict[str, Any] = {}
    if "modules.json" not in file_paths:
        return dense_metadata
    modules = await connector.read_file_json("modules.json")
    paths = [
        module.get("path")
        for module in modules
        if "type" in module and module.get("type") == "sentence_transformers.models.Dense" and "path" in module
    ]
    for path in paths:
        subpath = f"{path}/model.safetensors"
        if subpath in file_paths:
            dense_metadata[path] = await read_safetensors_metadata_from_connector(connector, subpath)
    return dense_metadata


async def run_with_connector(
    connector: Connector,
    model_id: str,
    revision: str,
    experimental: bool = False,
    max_model_len: int | None = None,
    batch_size: int = 1,
    kv_cache_dtype: str | None = None,
    json_output: bool = False,
    ignore_table_width: bool = False,
) -> Dict[str, Any] | None:
    """Unified run: list files and read safetensors/config via the given connector."""
    file_paths = await connector.list_files()

    if "model.safetensors" in file_paths:
        raw_metadata = await read_safetensors_metadata_from_connector(connector, "model.safetensors")
        if "config_sentence_transformers.json" in file_paths:
            dense_metadata = await get_modules_and_dense_metadata_from_connector(connector, file_paths)
            raw_metadata = {"0_Transformer": raw_metadata, **dense_metadata}
        else:
            raw_metadata = {"Transformer": raw_metadata}
        metadata = parse_safetensors_metadata(raw_metadata=raw_metadata)
    elif "model.safetensors.index.json" in file_paths:
        files_index = await connector.read_file_json("model.safetensors.index.json")
        shard_paths = set(files_index["weight_map"].values())
        metadata_list = await asyncio.gather(
            *[read_safetensors_metadata_from_connector(connector, p) for p in shard_paths]
        )
        raw_metadata = reduce(lambda acc, m: acc | m, metadata_list, {})
        if "config_sentence_transformers.json" in file_paths:
            dense_metadata = await get_modules_and_dense_metadata_from_connector(connector, file_paths)
            raw_metadata = {"0_Transformer": raw_metadata, **dense_metadata}
        else:
            raw_metadata = {"Transformer": raw_metadata}
        metadata = parse_safetensors_metadata(raw_metadata=raw_metadata)
    elif "model_index.json" in file_paths:
        files_index = await connector.read_file_json("model_index.json")
        paths = {k for k in files_index if not k.startswith("_")}
        path_metadatas: Dict[str, List[Dict[str, Any]]] = {}
        for path in paths:
            if f"{path}/diffusion_pytorch_model.safetensors" in file_paths:
                path_metadatas[path] = [
                    await read_safetensors_metadata_from_connector(
                        connector, f"{path}/diffusion_pytorch_model.safetensors"
                    )
                ]
            elif f"{path}/model.safetensors" in file_paths:
                path_metadatas[path] = [
                    await read_safetensors_metadata_from_connector(connector, f"{path}/model.safetensors")
                ]
            elif f"{path}/diffusion_pytorch_model.safetensors.index.json" in file_paths:
                idx = await connector.read_file_json(
                    f"{path}/diffusion_pytorch_model.safetensors.index.json"
                )
                path_metadatas[path] = [
                    await read_safetensors_metadata_from_connector(connector, f"{path}/{f}")
                    for f in set(idx["weight_map"].values())
                ]
            elif f"{path}/model.safetensors.index.json" in file_paths:
                idx = await connector.read_file_json(f"{path}/model.safetensors.index.json")
                path_metadatas[path] = [
                    await read_safetensors_metadata_from_connector(connector, f"{path}/{f}")
                    for f in set(idx["weight_map"].values())
                ]
        raw_metadata = {
            path: reduce(lambda acc, m: acc | m, meta_list, {})
            for path, meta_list in path_metadatas.items()
        }
        metadata = parse_safetensors_metadata(raw_metadata=raw_metadata)
    else:
        raise RuntimeError(
            "NONE OF `model.safetensors`, `model.safetensors.index.json`, `model_index.json` HAS BEEN FOUND"
        )

    cache_size = None
    cache_dtype = None
    if experimental and "config.json" in file_paths:
        config: Dict[str, Any] = await connector.read_file_json("config.json")
        if "architectures" not in config or (
            "architectures" in config
            and not any(
                arch.__contains__("ForCausalLM") or arch.__contains__("ForConditionalGeneration")
                for arch in config["architectures"]
            )
        ):
            warnings.warn(
                "`--experimental` was provided, but either `config.json` doesn't have the `architectures` key meaning that the model architecture cannot be inferred, or rather that it's neither `...ForCausalLM` not `...ForConditionalGeneration`, meaning that the KV Cache estimation might not apply. If that's the case, then remove the `--experimental` flag from the command to suppress this warning."
            )
        else:
            if (
                any(arch.__contains__("ForConditionalGeneration") for arch in config["architectures"])
                and "text_config" in config
            ):
                warnings.warn(
                    f"Given that `--model-id={model_id}` is a `...ForConditionalGeneration` model, then the configuration from `config.json` will be retrieved from the key `text_config` instead."
                )
                config = config["text_config"]

            if max_model_len is None:
                max_model_len = config.get(
                    "max_position_embeddings",
                    config.get("n_positions", config.get("max_seq_len", max_model_len)),
                )

            if max_model_len is None:
                warnings.warn(
                    f"Either the `--max-model-len` was not set, is not available in `config.json` with the any of the keys: `max_position_embeddings`, `n_positions`, or `max_seq_len` (in that order of priority), or both; so the memory required to fit the context length cannot be estimated."
                )

            if not all(k in config for k in {"hidden_size", "num_hidden_layers", "num_attention_heads"}):  # type: ignore
                warnings.warn(
                    f"`config.json` doesn't contain all the keys `hidden_size`, `num_hidden_layers`, and `num_attention_heads`, but only {config.keys()}."  # type: ignore
                )

            if kv_cache_dtype in {"fp8_e5m2", "fp8_e4m3"}:
                cache_dtype = kv_cache_dtype.upper().replace("FP8", "F8")
            elif kv_cache_dtype in {"fp8", "fp8_ds_mla", "fp8_inc"}:
                warnings.warn(
                    f"--kv-cache-dtype={kv_cache_dtype}` has been provided, but given that none of those matches an actual Safetensors dtype since it should be any of `F8_E5M2` or `F8_E4M3`, the `--kv-cache-dtype` will default to `F8_E4M3` instead, which implies that the calculations are the same given that both dtypes take 1 byte despite the quantization scheme of it, or the hardware compatibility; so the estimations should be accurate enough."
                )
                cache_dtype = "F8_E4M3"
            elif kv_cache_dtype == "bfloat16":
                cache_dtype = "BF16"
            elif "quantization_config" in config and "quant_method" in config["quantization_config"]:
                _quantization_config = config["quantization_config"]
                _quant_method = _quantization_config["quant_method"]

                if _quant_method != "fp8":
                    raise RuntimeError(
                        f"Provided `--kv-cache-dtype=auto` (or unset) and given that `config.json` contains the following `quantization_config={_quantization_config}` with a `quant_method` different than `fp8` i.e., `{_quant_method}`, which is not supported; you should enforce the `--kv-cache-dtype` value to whatever quantization precision it's using, if applicable.\nAs KV cache estimation is still experimental, as that might not be the case for your model, then feel free to open an issue at https://github.com/alvarobartt/hf-mem with a report and eventually what solution you would like to see implemented."
                    )

                _fmt = _quantization_config.get("fmt", _quantization_config.get("format", None))
                if _fmt:
                    if not _fmt.startswith("float8_"):
                        _fmt = f"float8_{_fmt}"

                    if _fmt not in TorchDtypes.__args__:
                        raise RuntimeError(
                            f"Provided `--kv-cache-dtype=auto` (or unset) and given that `config.json` contains the following `quantization_config={_quantization_config}` with a `fmt` (or `format`) value of `{_fmt}` that's not supported (should be any of {TorchDtypes.__args__}), you might need to set `--kv-cache-dtype=fp8` to enforce the dtype instead of pulling it from the `config.json`.\nAs KV cache estimation is still experimental, as that might not be the case for your model, then feel free to open an issue at https://github.com/alvarobartt/hf-mem with a report and eventually what solution you would like to see implemented."
                        )

                    cache_dtype = torch_dtype_to_safetensors_dtype(_fmt)
                else:
                    cache_dtype = max(
                        (
                            l := [
                                d
                                for c in metadata.components.values()
                                for d in c.dtypes.keys()
                                if d in {"F8_E5M2", "F8_E4M3"}
                            ]
                        ),
                        key=l.count,
                        default=None,
                    )

                    if not cache_dtype:
                        raise RuntimeError(
                            f"The `config.json` file for `--model-id={model_id}` contains `quantization_config={_quantization_config}` but the `quant_method=fp8` whereas any tensor in the model weights is set to any of `F8_E4M3` nor `F8_E5M2`, which means that the `F8_` format for the Safetensors dtype cannot be inferred; so you might need to set `--kv-cache-dtype=fp8` to enforce the dtype instead of pulling it from the `config.json`.\nAs KV cache estimation is still experimental, as that might not be the case for your model, then feel free to open an issue at https://github.com/alvarobartt/hf-mem with a report and eventually what solution you would like to see implemented."
                        )
            elif _cache_dtype := config.get("torch_dtype", None):
                cache_dtype = torch_dtype_to_safetensors_dtype(_cache_dtype)
            elif _cache_dtype := config.get("dtype", None):
                cache_dtype = torch_dtype_to_safetensors_dtype(_cache_dtype)
            else:
                raise RuntimeError(
                    f"Provided `--kv-cache-dtype={kv_cache_dtype}` but it needs to be any of `auto`, `bfloat16`, `fp8`, `fp8_ds_mla`, `fp8_e4m3`, `fp8_e5m2` or `fp8_inc`. If `--kv-cache-dtype=auto` (or unset), then the `config.json` should either contain the `torch_dtype` or `dtype` fields set; or if quantized, then `quantization_config` needs to be set and contain the key `quant_method` with value `fp8` (as none of `fp32`, `fp16` or `bf16` is considered within the `quantization_config`), and optionally also contain `fmt` set to any valid FP8 format as `float8_e4m3` or `float8_e4m3fn`."
                )

            cache_size = (
                2
                * config.get("num_hidden_layers")  # type: ignore
                * config.get("num_key_value_heads", config.get("num_attention_heads"))  # type: ignore
                * (config.get("hidden_size") // config.get("num_attention_heads"))  # type: ignore
                * max_model_len
                * get_safetensors_dtype_bytes(cache_dtype)
            )

            if batch_size:
                cache_size *= batch_size

    if json_output:
        out = {"model_id": model_id, "revision": revision, **asdict(metadata)}
        if experimental and cache_size:
            out["max_model_len"] = max_model_len
            out["batch_size"] = batch_size
            out["cache_size"] = cache_size
            out["cache_dtype"] = cache_dtype  # type: ignore
        return out
    if experimental and cache_size:
        print_report(
            model_id=model_id,
            revision=revision,
            metadata=metadata,
            cache={
                "max_model_len": max_model_len,
                "cache_size": cache_size,
                "batch_size": batch_size,
                "cache_dtype": cache_dtype,  # type: ignore
            },
            ignore_table_width=ignore_table_width,
        )
    else:
        print_report(
            model_id=model_id,
            revision=revision,
            metadata=metadata,
            ignore_table_width=ignore_table_width,
        )
    return None


async def run(
    model_id: str,
    revision: str,
    experimental: bool = False,
    max_model_len: int | None = None,
    batch_size: int = 1,
    kv_cache_dtype: str | None = None,
    json_output: bool = False,
    ignore_table_width: bool = False,
) -> Dict[str, Any] | None:
    """Run against Hugging Face Hub using HFConnector."""
    client = httpx.AsyncClient(
        limits=httpx.Limits(
            max_keepalive_connections=MAX_CONCURRENCY,
            max_connections=MAX_CONCURRENCY,
        ),
        timeout=httpx.Timeout(REQUEST_TIMEOUT),
        http2=True,
        follow_redirects=True,
    )
    headers = make_hf_headers(model_id, revision)
    connector = HFConnector(model_id, revision, client=client, headers=headers)
    try:
        return await run_with_connector(
            connector,
            model_id=model_id,
            revision=revision,
            experimental=experimental,
            max_model_len=max_model_len,
            batch_size=batch_size,
            kv_cache_dtype=kv_cache_dtype,
            json_output=json_output,
            ignore_table_width=ignore_table_width,
        )
    finally:
        await connector.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate inference memory for models (safetensors). Supports HF Hub, local folder, S3, and GCS.",
    )

    parser.add_argument(
        "--connector",
        choices=["hf", "local", "s3", "gcs"],
        help="Source connector: hf (Hugging Face Hub), local (directory), s3 (AWS S3), gcs (Google Cloud Storage). Inferred from other args if omitted.",
    )
    parser.add_argument(
        "--model-id",
        help="Model ID on the Hugging Face Hub (hf), or display name for report (local/s3/gcs). Defaults to folder/bucket name when using local/s3/gcs.",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Revision (hf: branch/tag; others: e.g. 'local').",
    )
    parser.add_argument(
        "--local-path",
        metavar="DIR",
        help="Path to a local folder with safetensors (connector=local).",
    )
    parser.add_argument(
        "--s3-bucket",
        metavar="BUCKET",
        help="S3 bucket name (connector=s3). Use with optional --s3-prefix.",
    )
    parser.add_argument(
        "--s3-prefix",
        default="",
        help="Prefix (folder) inside the S3 bucket. Default: root.",
    )
    parser.add_argument(
        "--gcs-bucket",
        metavar="BUCKET",
        help="GCS bucket name (connector=gcs). Use with optional --gcs-prefix.",
    )
    parser.add_argument(
        "--gcs-prefix",
        default="",
        help="Prefix (folder) inside the GCS bucket. Default: root.",
    )

    parser.add_argument(
        "--experimental",
        action="store_true",
        help="Whether to enable the experimental KV Cache estimation or not. Only applies to `...ForCausalLM` and `...ForConditionalGeneration` models from Transformers.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        # Reference: https://docs.vllm.ai/en/stable/configuration/engine_args/#-max-model-len
        help="Model context length (prompt and output). If unspecified, will be automatically derived from the model config.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size to help estimate the required RAM for caching when running the inference. Defaults to 1.",
    )
    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        default="auto",
        # NOTE: https://docs.vllm.ai/en/stable/cli/serve/#-kv-cache-dtype
        choices={"auto", "bfloat16", "fp8", "fp8_ds_mla", "fp8_e4m3", "fp8_e5m2", "fp8_inc"},
        help="Data type for the KV cache storage. If `auto` is specified, it will use the default model dtype specified in the `config.json` (if available). Despite the FP8 data types having different formats, all those take 1 byte, meaning that the calculation would lead to the same results. Defaults to `auto`.",
    )

    parser.add_argument(
        "--json-output",
        action="store_true",
        help="Whether to provide the output as a JSON instead of printed as table.",
    )
    parser.add_argument(
        "--ignore-table-width",
        action="store_true",
        help="Whether to ignore the maximum recommended table width, in case the `--model-id` and/or `--revision` cause a row overflow when printing those.",
    )

    args = parser.parse_args()

    # Resolve connector
    connector_name = args.connector
    if not connector_name:
        if args.local_path:
            connector_name = "local"
        elif args.s3_bucket:
            connector_name = "s3"
        elif args.gcs_bucket:
            connector_name = "gcs"
        else:
            connector_name = "hf"

    if args.experimental:
        warnings.warn(
            "`--experimental` is set; KV Cache estimation uses config.json from the source when present."
        )

    async def _run() -> Dict[str, Any] | None:
        connector: Connector
        model_id: str
        revision: str = args.revision

        if connector_name == "hf":
            if not args.model_id:
                parser.error("--model-id is required for connector 'hf'")
            model_id = args.model_id
            return await run(
                model_id=model_id,
                revision=revision,
                experimental=args.experimental,
                max_model_len=args.max_model_len,
                batch_size=args.batch_size,
                kv_cache_dtype=args.kv_cache_dtype,
                json_output=args.json_output,
                ignore_table_width=args.ignore_table_width,
            )
        if connector_name == "local":
            if not args.local_path:
                parser.error("--local-path is required for connector 'local'")
            path = os.path.abspath(os.path.expanduser(args.local_path))
            model_id = args.model_id or os.path.basename(path)
            connector = LocalConnector(path)
        elif connector_name == "s3":
            if not args.s3_bucket:
                parser.error("--s3-bucket is required for connector 's3'")
            model_id = args.model_id or args.s3_bucket
            connector = S3Connector(args.s3_bucket, args.s3_prefix or "")
        else:  # gcs
            if not args.gcs_bucket:
                parser.error("--gcs-bucket is required for connector 'gcs'")
            model_id = args.model_id or args.gcs_bucket
            connector = GCSConnector(args.gcs_bucket, args.gcs_prefix or "")

        return await run_with_connector(
            connector,
            model_id=model_id,
            revision=revision,
            experimental=args.experimental,
            max_model_len=args.max_model_len,
            batch_size=args.batch_size,
            kv_cache_dtype=args.kv_cache_dtype,
            json_output=args.json_output,
            ignore_table_width=args.ignore_table_width,
        )

    asyncio.run(_run())
