import argparse
import asyncio
import json
import os
import struct
import warnings
from dataclasses import asdict
from functools import reduce
from typing import Any, Dict, List, Optional

import httpx

from hf_mem.connectors import AzureConnector, Connector, GCSConnector, HFConnector, LocalConnector, S3Connector
from hf_mem.connectors.hf import make_hf_headers
from hf_mem.metadata import parse_safetensors_header_bytes, parse_safetensors_metadata
from hf_mem.print import print_report


def resolve_quantization_config(
    quantization_config: Optional[dict] = None,
) -> Optional[str]:
    """Resolve quantization config dict to a short dtype string (e.g. int4, fp8). No transformers dependency."""
    if quantization_config is None:
        return "float16"

    if "quant_method" in quantization_config:
        quant_method = quantization_config["quant_method"]
        if quant_method == "compressed-tensors":
            weights = (
                quantization_config.get("config_groups", {})
                .get("group_0", {})
                .get("weights", None)
            )
            if weights is not None:
                return weights.get("type", "int") + str(weights.get("num_bits", 4))
            return "float16"

        if quant_method == "awq":
            bits = quantization_config.get("bits", 4)
            return "int" + str(bits)
        if quant_method == "gptq":
            bits = quantization_config.get("bits", 4)
            return "int" + str(bits)

        if quant_method == "fp8":
            return "fp8"

        if quant_method == "bitsandbytes":
            if quantization_config.get("load_in_8bit", False):
                return "int8"
            if quantization_config.get("load_in_4bit", False):
                return "int4"
            raise ValueError(
                f"No weights found in BitsAndBytes quantization config: {quantization_config}"
            )
        if quant_method == "mxfp4":
            return "mxfp4"

    if "quantization" in quantization_config:
        print("Model is quantized with TensorRT Model Optimizer")
        quant_algo = quantization_config["quantization"]["quant_algo"]
        if quant_algo == "FP8":
            return "fp8"
        if quant_algo == "NVFP4":
            return "nvfp4"
        raise ValueError(f"Invalid quantization algorithm: {quant_algo}")
    return None

def get_tp_constraints(
    config_input: str | Dict[str, Any], model_id: str = "Unknown"
) -> Dict[str, Any]:
    """
    Compute TP limits from a model config (path or dict).
    Uses num_attention_heads / num_key_value_heads only.
    For encoder-decoder (ForConditionalGeneration), uses text_config.
    """
    if isinstance(config_input, str):
        with open(config_input) as f:
            config = json.load(f)
    else:
        config = config_input

    if "text_config" in config:
        effective = config["text_config"]
    elif "llm_config" in config:
        effective = config["llm_config"]
    else:
        effective = config

    q_heads = effective.get("num_attention_heads")
    kv_heads = effective.get("num_key_value_heads", q_heads)

    if q_heads is None:
        return {"error": "Could not find attention head configuration. Ensure config is valid."}

    max_tp = q_heads
    valid_degrees = [
        i for i in range(1, max_tp + 1)
        if q_heads % i == 0 and ((i & (i - 1)) == 0)
    ]

    return {
        "model": config.get("_name_or_path", model_id),
        "query_heads": q_heads,
        "kv_heads": kv_heads,
        "max_tp": max_tp,
        "valid_degrees": valid_degrees,
        "recommended_for_single_node": [d for d in valid_degrees if d <= 8],
    }


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
    dtype_bytes: int = 2,
    json_output: bool = False,
    ignore_table_width: bool = False,
    tp_limits: bool = False,
) -> Dict[str, Any] | None:
    """Unified run: list files and read safetensors/config via the given connector.
    If tp_limits=True, TP constraints are included in the returned dict (with or without json_output).
    """
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
    quantization = None
    if experimental and "config.json" in file_paths:
        config: Dict[str, Any] = await connector.read_file_json("config.json")
        quantization = resolve_quantization_config(config.get("quantization_config"))
        
        if "architectures" not in config:
            warnings.warn(
                "`--experimental` was provided, but `config.json` doesn't have the `architectures` key meaning that the model architecture cannot be inferred. If that's the case, then remove the `--experimental` flag from the command to suppress this warning."
            )
        else:
            if  "text_config" in config:
                config = config["text_config"]
            
            elif "llm_config" in config:
                config = config["llm_config"]

            if max_model_len is None:
                max_model_len = config.get(
                    "max_position_embeddings",
                    config.get("n_positions", config.get("max_seq_len", max_model_len)),
                )

            if max_model_len is None:
                warnings.warn(
                    f"Either the `--max-model-len` was not set, is not available in `config.json` with the any of the keys: `max_position_embeddings`, `n_positions`, or `max_seq_len` (in that order of priority), or both; so the memory required to fit the context length cannot be estimated."
                )
                max_model_len = 131072
                print(f"Using default max model length of 131072")

            if not all(k in config for k in {"hidden_size", "num_hidden_layers", "num_attention_heads"}):  # type: ignore
                warnings.warn(
                    f"`config.json` doesn't contain all the keys `hidden_size`, `num_hidden_layers`, and `num_attention_heads`, but only {config.keys()}."  # type: ignore
                )

            try:
                cache_size = (
                    2
                    * config.get("num_hidden_layers")  # type: ignore
                    * config.get("num_key_value_heads", config.get("num_attention_heads"))  # type: ignore
                    * (config.get("head_dim", config.get("hidden_size") // config.get("num_attention_heads"))) # type: ignore
                    * max_model_len
                    * dtype_bytes
                )
            except Exception as e:
                print(f"Error calculating cache size: {e}")
                cache_size = 0

            if batch_size:
                cache_size *= batch_size

    if "hf_quant_config.json" in file_paths:
        hf_quant_config = await connector.read_file_json("hf_quant_config.json")
        quantization = resolve_quantization_config(hf_quant_config)

    if json_output:
        out = {"model_id": model_id, "revision": revision, **asdict(metadata)}
        if quantization:
            out["quantization"] = quantization
        if experimental and cache_size:
            out["max_model_len"] = max_model_len
            out["batch_size"] = batch_size
            out["cache_size"] = cache_size
        
        if "config.json" in file_paths and tp_limits:
            config = await connector.read_file_json("config.json")
            out["tp_constraints"] = get_tp_constraints(config, model_id)
            out["architecture"] = config.get("architectures")[0]
        
        if "model_index.json" in file_paths:
            model_index = await connector.read_file_json("model_index.json")
            out["architecture"] = model_index.get("_class_name")

        print(out)
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
    dtype_bytes: int = 2,
    json_output: bool = False,
    ignore_table_width: bool = False,
    tp_limits: bool = False,
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
            dtype_bytes=dtype_bytes,
            json_output=json_output,
            ignore_table_width=ignore_table_width,
            tp_limits=tp_limits,
        )
    finally:
        await connector.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate inference memory for models (safetensors). Supports HF Hub, local folder, S3, and GCS.",
    )

    parser.add_argument(
        "--connector",
        choices=["hf", "local", "s3", "gcs", "azure"],
        help="Source connector: hf (Hugging Face Hub), local (directory), s3 (AWS S3), gcs (GCS), azure (Azure Blob Storage). Inferred from other args if omitted.",
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
        "--azure-container",
        metavar="CONTAINER",
        help="Azure Blob Storage container name (connector=azure). Use with optional --azure-prefix and --azure-account.",
    )
    parser.add_argument(
        "--azure-prefix",
        default="",
        help="Prefix (folder) inside the Azure container. Default: root.",
    )
    parser.add_argument(
        "--azure-account",
        default="",
        help="Azure Storage account name (when not using AZURE_STORAGE_CONNECTION_STRING). Uses DefaultAzureCredential.",
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
        "--dtype-bytes",
        type=int,
        default=2,
        help="Bytes per dtype. Defaults to 2.",
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
    parser.add_argument(
        "--tp-limits",
        action="store_true",
        help="Print tensor parallel limits (max TP and valid degrees) from the model config and exit. Uses the same connector as the main command (--model-id, --local-path, etc.).",
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
        elif args.azure_container:
            connector_name = "azure"
        else:
            connector_name = "hf"

    if args.experimental:
        warnings.warn(
            "`--experimental` is set; KV Cache estimation uses config.json from the source when present."
        )

    async def _run() -> Dict[str, Any] | None:
        connector: Connector | None = None
        model_id: str = ""
        revision: str = args.revision

        if connector_name == "hf":
            if not args.model_id:
                parser.error("--model-id is required for connector 'hf'")
            model_id = args.model_id
            if args.tp_limits:
                client = httpx.AsyncClient(
                    limits=httpx.Limits(
                        max_keepalive_connections=MAX_CONCURRENCY,
                        max_connections=MAX_CONCURRENCY,
                    ),
                    timeout=httpx.Timeout(REQUEST_TIMEOUT),
                    http2=True,
                    follow_redirects=True,
                )
                connector = HFConnector(
                    model_id, revision, client=client, headers=make_hf_headers(model_id, revision)
                )
            else:
                return await run(
                    model_id=model_id,
                    revision=revision,
                    experimental=args.experimental,
                    max_model_len=args.max_model_len,
                    batch_size=args.batch_size,
                    dtype_bytes=args.dtype_bytes,
                    json_output=args.json_output,
                    ignore_table_width=args.ignore_table_width,
                    tp_limits=args.tp_limits or args.json_output,
                )
        elif connector_name == "local":
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
        elif connector_name == "gcs":
            if not args.gcs_bucket:
                parser.error("--gcs-bucket is required for connector 'gcs'")
            model_id = args.model_id or args.gcs_bucket
            connector = GCSConnector(args.gcs_bucket, args.gcs_prefix or "")
        else:  # azure
            if not args.azure_container:
                parser.error("--azure-container is required for connector 'azure'")
            model_id = args.model_id or args.azure_container
            connector = AzureConnector(
                args.azure_container,
                args.azure_prefix or "",
                account=args.azure_account or None,
            )

        if args.tp_limits and connector is not None and not args.json_output:
            config = await connector.read_file_json("config.json")
            result = get_tp_constraints(config, model_id)
            if "error" in result:
                print(result["error"])
                if hasattr(connector, "close"):
                    await connector.close()
                return None
            if hasattr(connector, "close"):
                await connector.close()
            return None

        return await run_with_connector(
            connector,
            model_id=model_id,
            revision=revision,
            experimental=args.experimental,
            max_model_len=args.max_model_len,
            batch_size=args.batch_size,
            json_output=args.json_output,
            ignore_table_width=args.ignore_table_width,
            tp_limits=args.tp_limits or args.json_output,
        )

    output = asyncio.run(_run())
    if output is not None:
        print(json.dumps(output, indent=2))
