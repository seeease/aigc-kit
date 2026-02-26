"""阿里云 DashScope 图片生成（通义万相）"""

import logging
from typing import Any

import httpx

from ..base import ImageProvider, ImageResult

logger = logging.getLogger(__name__)

# WxH → DashScope 支持的尺寸映射（按比例匹配最近的）
_SUPPORTED_SIZES: list[tuple[int, int]] = [
    (1664, 928),  # 16:9
    (1472, 1104),  # 4:3
    (1328, 1328),  # 1:1
    (1104, 1472),  # 3:4
    (928, 1664),  # 9:16
]


def _normalize_size(size: str) -> str:
    """将 WxH 转为 DashScope 支持的最近尺寸（W*H 格式）"""
    try:
        w, h = (int(x) for x in size.split("x"))
    except ValueError:
        return "1328*1328"

    target = w / h
    best = min(_SUPPORTED_SIZES, key=lambda s: abs(s[0] / s[1] - target))
    return f"{best[0]}*{best[1]}"


class DashScopeProvider(ImageProvider):
    """阿里云 DashScope 图片生成"""

    BASE_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "qwen-image-max",
        timeout: int = 120,
    ):
        self.api_key = api_key
        self.model = model
        self.client = httpx.Client(timeout=timeout)

    @property
    def name(self) -> str:
        return "dashscope"

    def _call(self, prompt: str, *, size: str, **kwargs: Any) -> ImageResult:
        ds_size = _normalize_size(size)
        payload: dict[str, Any] = {
            "model": self.model,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"text": prompt}],
                    }
                ]
            },
            "parameters": {
                "watermark": False,
                "size": ds_size,
            },
        }

        resp = self.client.post(
            self.BASE_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )

        if not resp.is_success:
            logger.error("DashScope 图片生成失败: %s %s", resp.status_code, resp.text)
            raise RuntimeError(f"DashScope 图片生成失败: {resp.status_code}")

        data = resp.json()

        # 检查业务错误
        if "code" in data:
            raise RuntimeError(
                f"DashScope 错误: {data.get('code')} - {data.get('message')}"
            )

        choices = data.get("output", {}).get("choices", [])
        if not choices:
            raise RuntimeError("DashScope 未返回图片")

        content = choices[0].get("message", {}).get("content", [])
        for item in content:
            if "image" in item:
                return ImageResult(url=item["image"], provider=self.name)

        raise RuntimeError("DashScope 响应中未找到图片 URL")

    def text_to_image(
        self,
        prompt: str,
        *,
        size: str = "1024x1024",
        **kwargs,
    ) -> ImageResult:
        logger.info("DashScope 文生图: model=%s, size=%s", self.model, size)
        return self._call(prompt, size=size, **kwargs)

    def image_to_image(
        self,
        prompt: str,
        *,
        reference_images: list[str],
        size: str = "1024x1024",
        **kwargs,
    ) -> ImageResult:
        # 图生图稍后集成，先抛出提示
        raise NotImplementedError("DashScope 图生图尚未集成")

    def close(self) -> None:
        self.client.close()
