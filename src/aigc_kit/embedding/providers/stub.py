"""Stub Embedding Provider — 开发/测试用

返回固定维度的零向量，不调用任何外部 API。
"""

import logging

from ..base import EmbeddingProvider, EmbeddingResult

logger = logging.getLogger(__name__)


class StubEmbeddingProvider(EmbeddingProvider):
    """桩实现，返回零向量"""

    def __init__(self, *, dimensions: int = 1536) -> None:
        self._dimensions = dimensions

    @property
    def name(self) -> str:
        return "stub"

    @property
    def default_model(self) -> str:
        return "stub-embedding"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, texts: list[str], *, model: str | None = None) -> EmbeddingResult:
        logger.debug("Stub embedding %d texts, dimensions=%d", len(texts), self._dimensions)
        vectors = [[0.0] * self._dimensions for _ in texts]
        return EmbeddingResult(
            vectors=vectors,
            provider=self.name,
            model=model or self.default_model,
            dimensions=self._dimensions,
            usage={"total_tokens": sum(len(t) for t in texts)},
        )
