"""统一图片生成客户端

自动路由到对应 provider，如果结果是 base64 则自动上传到存储获取 URL。
"""

import base64
import io
import logging
import uuid
from typing import Any

from PIL import Image

from ..storage.base import StorageProvider
from .base import ImageProvider, ImageResult

logger = logging.getLogger(__name__)

# 已注册的 provider 工厂
_PROVIDER_FACTORIES: dict[str, type] = {}


def _ensure_registered() -> None:
    """延迟注册，避免循环导入"""
    if _PROVIDER_FACTORIES:
        return
    from .providers.dashscope import DashScopeProvider
    from .providers.gemini import GeminiProvider
    from .providers.volcengine import VolcEngineProvider

    _PROVIDER_FACTORIES["volcengine"] = VolcEngineProvider
    _PROVIDER_FACTORIES["gemini"] = GeminiProvider
    _PROVIDER_FACTORIES["dashscope"] = DashScopeProvider

_THUMB_WIDTHS = [800, 600, 320]


def _convert_to_webp(data: bytes, *, max_width: int | None = None) -> bytes:
    """将图片转为 webp，可选 resize"""
    img = Image.open(io.BytesIO(data))
    if max_width and img.width > max_width:
        ratio = max_width / img.width
        new_size = (max_width, int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=85)
    return buf.getvalue()



class ImageClient:
    """统一图片生成客户端

    用法:
        client = ImageClient(provider="gemini", storage=r2, model="gemini-3-pro-image-preview")
        result = client.text_to_image("一只猫")
        print(result.url)  # 已上传到 R2 的 URL
    """

    def __init__(
        self,
        *,
        provider: str | ImageProvider,
        storage: StorageProvider | None = None,
        storage_key_prefix: str = "aigc",
        **provider_kwargs: Any,
    ):
        if isinstance(provider, ImageProvider):
            self._provider = provider
        else:
            _ensure_registered()
            factory = _PROVIDER_FACTORIES.get(provider)
            if not factory:
                raise ValueError(
                    f"未知 provider: {provider}，可选: {list(_PROVIDER_FACTORIES.keys())}"
                )
            self._provider = factory(**provider_kwargs)

        self._storage = storage
        self._storage_key_prefix = storage_key_prefix

    @property
    def provider_name(self) -> str:
        return self._provider.name

    def _ensure_url(self, result: ImageResult) -> ImageResult:
        """如果结果只有 base64 没有 url，转 webp 并生成缩略图后上传"""
        if result.has_url or not result.has_base64:
            return result

        if not self._storage:
            logger.warning("图片结果为 base64 但未配置 storage，无法生成 URL")
            return result

        raw_data = base64.b64decode(result.base64)
        oss_key = f"{self._storage_key_prefix}/{uuid.uuid4().hex}.webp"

        # 原图转 webp
        webp_data = _convert_to_webp(raw_data)
        upload = self._storage.upload_bytes(
            webp_data, oss_key, content_type="image/webp"
        )
        logger.info("上传原图 webp: %s", oss_key)

        # 生成缩略图
        for width in _THUMB_WIDTHS:
            thumb_data = _convert_to_webp(raw_data, max_width=width)
            thumb_key = f"{oss_key}.{width}.webp"
            self._storage.upload_bytes(
                thumb_data, thumb_key, content_type="image/webp"
            )
            logger.info("上传缩略图: %s", thumb_key)

        result.url = upload.url
        result.mime_type = "image/webp"
        return result

    def text_to_image(
        self,
        prompt: str,
        *,
        size: str = "1024x1024",
        **kwargs,
    ) -> ImageResult:
        """文生图，返回带 URL 的结果"""
        result = self._provider.text_to_image(prompt, size=size, **kwargs)
        return self._ensure_url(result)

    def image_to_image(
        self,
        prompt: str,
        *,
        reference_images: list[str],
        size: str = "1024x1024",
        **kwargs,
    ) -> ImageResult:
        """图生图，返回带 URL 的结果"""
        result = self._provider.image_to_image(
            prompt, reference_images=reference_images, size=size, **kwargs
        )
        return self._ensure_url(result)

    def close(self) -> None:
        self._provider.close()
        if self._storage:
            self._storage.close()
