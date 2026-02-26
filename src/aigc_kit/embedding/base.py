"""Embedding Provider 抽象接口"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Embedding 结果"""

    vectors: list[list[float]]
    provider: str = ""
    model: str = ""
    dimensions: int = 0
    usage: dict[str, int] = field(default_factory=dict)


class EmbeddingProvider(ABC):
    """Embedding Provider 抽象基类

    子类实现时应从环境变量读取默认配置。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider 名称"""

    @property
    @abstractmethod
    def default_model(self) -> str:
        """当前使用的模型名"""

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """向量维度"""

    @abstractmethod
    def embed(self, texts: list[str], *, model: str | None = None) -> EmbeddingResult:
        """将文本列表转为向量

        Args:
            texts: 待向量化的文本列表
            model: 可选模型覆盖
        """

    def close(self) -> None:
        """释放资源（可选覆盖）"""
