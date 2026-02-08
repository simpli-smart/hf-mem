"""Connectors for reading model files from HF, local disk, S3, or GCS."""

from hf_mem.connectors.base import Connector
from hf_mem.connectors.gcs import GCSConnector
from hf_mem.connectors.hf import HFConnector
from hf_mem.connectors.local import LocalConnector
from hf_mem.connectors.s3 import S3Connector

__all__ = [
    "Connector",
    "GCSConnector",
    "HFConnector",
    "LocalConnector",
    "S3Connector",
]
