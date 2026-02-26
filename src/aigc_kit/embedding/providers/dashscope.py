"""DashScope Embedding（OpenAI 兼容模式）"""

import logging
import os

from openai import OpenAI

from ..base import EmbeddingProvider, EmbeddingResult

logger = logging.getLogger(__name__)


class DashScopeEmbeddingProvider(EmbeddingProvider):
    """通义百炼 DashScope Embedding，使用 OpenAI 兼容接口"""

    def __init__(
        self,
        *,
        model: str = "text-embedding-v4",
        api_key: str | None = None,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        dimensions: int = 1024,
    ):
        self._model = model
        self._dimensions = dimensions
        self._client = OpenAI(
            api_key=api_key or os.environ.get("DASHSCOPE_API_KEY", ""),
            base_url=base_url,
        )

    @property
    def name(self) -> str:
        return "dashscope"

    @property
    def default_model(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, texts: list[str], *, model: str | None = None) -> EmbeddingResult:
        use_model = model or self._model
        resp = self._client.embeddings.create(
            model=use_model,
            input=texts,
            dimensions=self._dimensions,
            encoding_format="float",
        )

        vectors = [item.embedding for item in resp.data]
        total_tokens = resp.usage.total_tokens if resp.usage else 0

        logger.debug(
            "DashScope embedding: model=%s, texts=%d, tokens=%d",
            use_model,
            len(texts),
            total_tokens,
        )

        return EmbeddingResult(
            vectors=vectors,
            provider=self.name,
            model=use_model,
            dimensions=self._dimensions,
            usage={"total_tokens": total_tokens},
        )

    def close(self) -> None:
        self._client.close()
