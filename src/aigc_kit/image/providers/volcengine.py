"""火山引擎 Seedream 图片生成"""

import logging
from typing import Any

import httpx

from ..base import ImageProvider, ImageResult

logger = logging.getLogger(__name__)


class VolcEngineProvider(ImageProvider):
    """火山引擎豆包 Seedream"""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://ark.cn-beijing.volces.com/api/v3",
        model: str = "doubao-seedream-4-5-251128",
        timeout: int = 120,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.client = httpx.Client(timeout=timeout)

    @property
    def name(self) -> str:
        return "volcengine"

    def _call(self, payload: dict[str, Any]) -> ImageResult:
        resp = self.client.post(
            f"{self.base_url}/images/generations",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        if not resp.is_success:
            logger.error("VolcEngine 图片生成失败: %s %s", resp.status_code, resp.text)
            raise RuntimeError(f"VolcEngine 图片生成失败: {resp.status_code}")

        data = resp.json()
        images = data.get("data", [])
        if not images:
            raise RuntimeError("VolcEngine 未返回图片")

        img = images[0]
        return ImageResult(
            url=img.get("url", ""),
            base64=img.get("b64_json", ""),
            provider=self.name,
        )

    # 最小尺寸 2048x2048，小于此值自动提升
    MIN_SIZE = 2048

    @staticmethod
    def _clamp_size(size: str) -> str:
        """确保尺寸不低于最小值，保持比例"""
        try:
            w, h = (int(x) for x in size.split("x"))
        except ValueError:
            return f"{VolcEngineProvider.MIN_SIZE}x{VolcEngineProvider.MIN_SIZE}"
        mn = VolcEngineProvider.MIN_SIZE
        if w < mn or h < mn:
            scale = max(mn / w, mn / h)
            w, h = int(w * scale), int(h * scale)
        return f"{w}x{h}"

    def text_to_image(
        self,
        prompt: str,
        *,
        size: str = "1024x1024",
        **kwargs,
    ) -> ImageResult:
        size = self._clamp_size(size)
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "size": size,
            "watermark": False,
        }
        logger.info("VolcEngine 文生图: model=%s, size=%s", self.model, size)
        return self._call(payload)

    def image_to_image(
        self,
        prompt: str,
        *,
        reference_images: list[str],
        size: str = "1024x1024",
        **kwargs,
    ) -> ImageResult:
        size = self._clamp_size(size)
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "size": size,
            "watermark": False,
            "image": reference_images,
        }
        logger.info(
            "VolcEngine 图生图: model=%s, refs=%d", self.model, len(reference_images)
        )
        return self._call(payload)

    def close(self) -> None:
        self.client.close()
