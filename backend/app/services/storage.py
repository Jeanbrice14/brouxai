from __future__ import annotations

import asyncio
import io
from pathlib import Path
from urllib.parse import urlparse

import boto3
import pandas as pd
import structlog

from app.config import settings

logger = structlog.get_logger(__name__)


def _get_s3_client():
    """Crée un client boto3 S3-compatible depuis les settings."""
    return boto3.client(
        "s3",
        endpoint_url=settings.storage_endpoint,
        aws_access_key_id=settings.storage_key,
        aws_secret_access_key=settings.storage_secret,
        region_name="us-east-1",  # requis par certains endpoints S3-compatibles
    )


def _parse_ref(ref: str) -> tuple[str, str]:
    """Parse une référence s3://bucket/path/to/key → (bucket, key)."""
    parsed = urlparse(ref)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    if not bucket or not key:
        raise ValueError(f"Référence storage invalide : {ref!r}. Format attendu: s3://bucket/key")
    return bucket, key


async def upload_file(
    ref: str, data: bytes, content_type: str = "application/octet-stream"
) -> None:
    """Upload un fichier binaire vers le Blob Storage.

    Args:
        ref: Référence S3 (ex: s3://narr8-dev/uploads/file.csv)
        data: Contenu binaire du fichier.
        content_type: MIME type.
    """
    bucket, key = _parse_ref(ref)
    client = _get_s3_client()
    await asyncio.to_thread(
        client.put_object,
        Bucket=bucket,
        Key=key,
        Body=data,
        ContentType=content_type,
    )
    logger.info("storage_upload", ref=ref, size=len(data))


async def download_file(ref: str) -> bytes:
    """Télécharge un fichier depuis le Blob Storage.

    Args:
        ref: Référence S3 (ex: s3://narr8-dev/uploads/file.csv)

    Returns:
        Contenu binaire du fichier.
    """
    bucket, key = _parse_ref(ref)
    client = _get_s3_client()
    response = await asyncio.to_thread(client.get_object, Bucket=bucket, Key=key)
    data: bytes = response["Body"].read()
    logger.info("storage_download", ref=ref, size=len(data))
    return data


async def read_dataframe(ref: str) -> pd.DataFrame:
    """Lit un fichier depuis le Blob Storage et retourne un DataFrame pandas.

    Formats supportés : CSV (.csv) et Excel (.xlsx, .xls).

    Args:
        ref: Référence S3 (ex: s3://narr8-dev/uploads/ventes.csv)

    Returns:
        DataFrame pandas.

    Raises:
        ValueError: Si le format de fichier n'est pas supporté.
    """
    data = await download_file(ref)
    suffix = Path(urlparse(ref).path).suffix.lower()

    if suffix in (".xlsx", ".xls"):
        df = pd.read_excel(io.BytesIO(data))
    elif suffix == ".csv":
        df = pd.read_csv(io.BytesIO(data))
    else:
        raise ValueError(
            f"Format de fichier non supporté : {suffix!r}. Formats acceptés : .csv, .xlsx, .xls"
        )

    logger.info("storage_read_dataframe", ref=ref, rows=len(df), cols=len(df.columns))
    return df
