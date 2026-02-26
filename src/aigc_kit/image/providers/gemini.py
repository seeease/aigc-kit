"""Google Gemini 图片生成 (Vertex AI)"""

import base64
import logging

from google import genai
from google.genai import types

from ..base import ImageProvider, ImageResult

logger = logging.getLogger(__name__)

# Gemini 支持的比例
_SUPPORTED_RATIOS = [
    (1, 1),
    (2, 3),
    (3, 2),
    (3, 4),
    (4, 3),
    (4, 5),
    (5, 4),
    (9, 16),
    (16, 9),
    (21, 9),
]


def _size_to_aspect_ratio(size: str) -> str:
    """将 WxH 转为 Gemini 最接近的 aspect_ratio 字符串"""
    try:
        w, h = (int(x) for x in size.split("x"))
    except ValueError:
        return "1:1"

    target = w / h
    best = min(_SUPPORTED_RATIOS, key=lambda r: abs(r[0] / r[1] - target))
    return f"{best[0]}:{best[1]}"


def _size_to_resolution(size: str) -> str:
    """将 WxH 转为 Gemini ImageConfig.image_size (1K/2K/4K)"""
    try:
        w, h = (int(x) for x in size.split("x"))
    except ValueError:
        return "1K"

    max_dim = max(w, h)
    thresholds = [(1024, "1K"), (2048, "2K"), (4096, "4K")]
    best = min(thresholds, key=lambda t: abs(t[0] - max_dim))
    return best[1]


class GeminiProvider(ImageProvider):
    """Google Gemini 图片生成，支持文生图和图生图"""

    def __init__(
        self,
        *,
        model: str = "gemini-3-pro-image-preview",
        project: str | None = None,
        location: str | None = None,
    ):
        self.model = model
        kwargs = {}
        if project:
            kwargs["project"] = project
        if location:
            kwargs["location"] = location
        self.client = genai.Client(**kwargs)

    @property
    def name(self) -> str:
        return "gemini"

    def _generate(
        self,
        contents: list,
        *,
        size: str = "1024x1024",
    ) -> ImageResult:
        """调用 Gemini 生成图片"""
        aspect_ratio = _size_to_aspect_ratio(size)
        resolution = _size_to_resolution(size)

        logger.debug(
            "Gemini generate_content 参数: model=%s, aspect_ratio=%s, image_size=%s, contents=%s",
            self.model,
            aspect_ratio,
            resolution,
            contents,
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
                image_config=types.ImageConfig(
                    aspect_ratio=aspect_ratio,
                    image_size=resolution,
                ),
            ),
        )

        for part in response.candidates[0].content.parts:
            if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                raw = part.inline_data.data
                b64 = (
                    base64.b64encode(raw).decode("utf-8")
                    if isinstance(raw, bytes)
                    else raw
                )
                return ImageResult(
                    base64=b64,
                    mime_type=part.inline_data.mime_type,
                    provider=self.name,
                )

        raise RuntimeError("Gemini 未返回图片")

    def text_to_image(
        self,
        prompt: str,
        *,
        size: str = "1024x1024",
        **kwargs,
    ) -> ImageResult:
        logger.info("Gemini 文生图: model=%s, size=%s", self.model, size)
        return self._generate([prompt], size=size)

    def image_to_image(
        self,
        prompt: str,
        *,
        reference_images: list[str],
        size: str = "1024x1024",
        **kwargs,
    ) -> ImageResult:
        """图生图：下载参考图 + prompt 一起发给 Gemini"""
        import httpx

        contents: list = []
        for img_url in reference_images:
            resp = httpx.get(img_url, timeout=30)
            resp.raise_for_status()
            ct = resp.headers.get("content-type", "image/png")
            contents.append(types.Part.from_bytes(data=resp.content, mime_type=ct))
        contents.append(prompt)

        logger.info(
            "Gemini 图生图: model=%s, size=%s, refs=%d",
            self.model,
            size,
            len(reference_images),
        )
        return self._generate(contents, size=size)
