<img src="https://github.com/user-attachments/assets/509a8244-8a91-4051-b337-41b7b2fe0e2f" />

---

> [!WARNING]
> `hf-mem` is still experimental and therefore subject to major changes across releases, so please keep in mind that breaking changes may occur until v1.0.0.

`hf-mem` is a CLI to estimate inference memory requirements for Hugging Face–style models, written in Python. It reads [Safetensors](https://github.com/huggingface/safetensors) metadata from multiple sources: **Hugging Face Hub**, **local folder**, **AWS S3**, **Google Cloud Storage (GCS)**, and **Azure Blob Storage**. The core only depends on `httpx`; S3, GCS, and Azure use optional extras. It's recommended to run with [`uv`](https://github.com/astral-sh/uv) for a better experience.

`hf-mem` lets you estimate the inference requirements for any model layout that uses Safetensors—including [Transformers](https://github.com/huggingface/transformers), [Diffusers](https://github.com/huggingface/diffusers), and [Sentence Transformers](https://github.com/huggingface/sentence-transformers)—whether the files live on the Hub, on disk, or in object storage.

Read more information about `hf-mem` in [this short-form post](https://alvarobartt.com/hf-mem).

## Installation

```bash
# Core (HF Hub + local)
pip install hf-mem

# With S3 support
pip install hf-mem[s3]

# With GCS support
pip install hf-mem[gcs]

# With Azure Blob Storage support
pip install hf-mem[azure]

# All connectors
pip install hf-mem[all]
```

Or run without installing: `uvx hf-mem ...`

## Sources (connectors)

| Connector | Description | Required args |
|-----------|-------------|---------------|
| **hf** | Hugging Face Hub (default) | `--model-id` |
| **local** | Local directory (e.g. downloaded model) | `--local-path` |
| **s3** | AWS S3 bucket | `--s3-bucket` (optional: `--s3-prefix`) |
| **gcs** | Google Cloud Storage bucket | `--gcs-bucket` (optional: `--gcs-prefix`) |
| **azure** | Azure Blob Storage container | `--azure-container` (optional: `--azure-prefix`, `--azure-account`) |

Use `--connector hf|local|s3|gcs|azure` to set the source explicitly; otherwise it is inferred from `--local-path`, `--s3-bucket`, `--gcs-bucket`, or `--azure-container`.

## Usage

### Hugging Face Hub (Transformers)

```bash
uvx hf-mem --model-id MiniMaxAI/MiniMax-M2
```

<img src="https://github.com/user-attachments/assets/530f8b14-a415-4fd6-9054-bcd81cafae09" />

### Diffusers

```bash
uvx hf-mem --model-id Qwen/Qwen-Image
```

<img src="https://github.com/user-attachments/assets/cd4234ec-bdcc-4db4-8b01-0ac9b5cd390c" />

### Sentence Transformers

```bash
uvx hf-mem --model-id google/embeddinggemma-300m
```

<img src="https://github.com/user-attachments/assets/a52c464b-a6c1-446d-9921-68aaefb9df88" />

### Local folder

Use a directory that contains the same layout as a Hub repo (e.g. a model downloaded with `huggingface-cli download` or `git clone`):

```bash
hf-mem --local-path /path/to/model
# or explicitly
hf-mem --connector local --local-path ./my-model --model-id my-model
```

### S3

Requires `hf-mem[s3]` (boto3). Uses the default AWS credentials (env, `~/.aws/credentials`, or IAM role).

```bash
pip install hf-mem[s3]
hf-mem --s3-bucket my-bucket --s3-prefix models/llama-2-7b
# or explicitly
hf-mem --connector s3 --s3-bucket my-bucket --s3-prefix models/llama-2-7b --model-id llama-2-7b
```

### GCS

Requires `hf-mem[gcs]` (google-cloud-storage). Uses default GCP credentials (e.g. `GOOGLE_APPLICATION_CREDENTIALS` or gcloud).

```bash
pip install hf-mem[gcs]
hf-mem --gcs-bucket my-bucket --gcs-prefix models/llama-2-7b
# or explicitly
hf-mem --connector gcs --gcs-bucket my-bucket --gcs-prefix models/llama-2-7b
```

### Azure Blob Storage

Requires `hf-mem[azure]` (azure-storage-blob, azure-identity). Authenticate via `AZURE_STORAGE_CONNECTION_STRING`, or set `--azure-account` (and use DefaultAzureCredential: env, managed identity, Azure CLI, etc.).

```bash
pip install hf-mem[azure]
# With connection string (env AZURE_STORAGE_CONNECTION_STRING)
hf-mem --azure-container my-container --azure-prefix models/llama-2-7b

# With account name + DefaultAzureCredential
hf-mem --connector azure --azure-account mystorageaccount --azure-container my-container --azure-prefix models/llama-2-7b
```

## Experimental

By enabling the `--experimental` flag, you can enable the KV Cache memory estimation for LLMs (`...ForCausalLM`) and VLMs (`...ForConditionalGeneration`), with optional `--max-model-len` (from `config.json` if unset), `--batch-size` (default 1), and `--kv-cache-dtype` (default `auto`, from `config.json` when available). When using local, S3, or GCS, `config.json` is read from that same source.

```bash
uvx hf-mem --model-id MiniMaxAI/MiniMax-M2 --experimental
```

<img src="https://github.com/user-attachments/assets/64eaff88-d395-4d8d-849b-78fb86411dc3" />

## (Optional) Agent Skills

Optionally, you can add `hf-mem` as an agent skill, which allows the underlying coding agent to discover and use it when provided as a [`SKILL.md`](.skills/hf-mem/SKILL.md).

More information can be found at [Anthropic Agent Skills and how to use them](https://github.com/anthropics/skills).

## References

- [Safetensors Metadata parsing](https://huggingface.co/docs/safetensors/en/metadata_parsing)
- [usgraphics - TR-100 Machine Report](https://github.com/usgraphics/usgc-machine-report)
