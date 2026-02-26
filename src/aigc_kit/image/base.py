"""图片生成抽象接口"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ImageResult:
    """图片生成结果"""

    url: str = ""
    base64: str = ""
    mime_type: str = "image/png"
    provider: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def has_url(self) -> bool:
        return bool(self.url)

    @property
    def has_base64(self) -> bool:
        return bool(self.base64)


class ImageProvider(ABC):
    """图片生成 Provider 抽象基类"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider 名称"""

    @abstractmethod
    def text_to_image(
        self,
        prompt: str,
        *,
        size: str = "1024x1024",
        **kwargs,
    ) -> ImageResult:
        """文生图"""

    @abstractmethod
    def image_to_image(
        self,
        prompt: str,
        *,
        reference_images: list[str],
        size: str = "1024x1024",
        **kwargs,
    ) -> ImageResult:
        """图生图"""

    def close(self) -> None:
        """释放资源"""
