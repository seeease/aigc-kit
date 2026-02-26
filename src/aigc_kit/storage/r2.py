"""Cloudflare R2 存储"""

import base64
import logging
import uuid

import boto3

from .base import StorageProvider, UploadResult

logger = logging.getLogger(__name__)


class R2Storage(StorageProvider):
    """Cloudflare R2 (S3 兼容)"""

    def __init__(
        self,
        *,
        access_key_id: str,
        access_key_secret: str,
        endpoint: str,
        bucket: str,
        public_domain: str = "",
    ):
        self.bucket = bucket
        self.public_domain = public_domain.rstrip("/")
        self.s3 = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=access_key_secret,
            region_name="auto",
        )

    @property
    def name(self) -> str:
        return "r2"

    def upload_bytes(
        self,
        data: bytes,
        key: str,
        *,
        content_type: str = "image/png",
    ) -> UploadResult:
        self.s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
        )
        url = f"{self.public_domain}/{key}" if self.public_domain else key
        logger.info("R2 上传完成: %s", url)
        return UploadResult(url=url, key=key, bucket=self.bucket)

    def upload_base64(
        self,
        b64_data: str,
        *,
        key_prefix: str = "images",
        content_type: str = "image/png",
        ext: str = "png",
    ) -> UploadResult:
        """便捷方法：上传 base64 数据，自动生成 key"""
        raw = base64.b64decode(b64_data)
        key = f"{key_prefix}/{uuid.uuid4().hex}.{ext}"
        return self.upload_bytes(raw, key, content_type=content_type)

    def close(self) -> None:
        pass
