"""存储抽象接口"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class UploadResult:
    """上传结果"""

    url: str
    key: str
    bucket: str = ""


class StorageProvider(ABC):
    """存储 Provider 抽象基类"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider 名称"""

    @abstractmethod
    def upload_bytes(
        self,
        data: bytes,
        key: str,
        *,
        content_type: str = "image/png",
    ) -> UploadResult:
        """上传二进制数据"""

    def close(self) -> None:
        """释放资源"""
