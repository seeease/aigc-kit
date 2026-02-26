"""AWS Bedrock Titan Embedding"""

import json
import logging
import os
from typing import Any

import boto3

from ..base import EmbeddingProvider, EmbeddingResult

logger = logging.getLogger(__name__)

# Titan Embedding 模型默认维度
_MODEL_DIMENSIONS: dict[str, int] = {
    "amazon.titan-embed-text-v2:0": 1024,
    "amazon.titan-embed-text-v1": 1536,
}


class BedrockEmbeddingProvider(EmbeddingProvider):
    """AWS Bedrock Titan Embedding"""

    def __init__(
        self,
        *,
        model: str = "amazon.titan-embed-text-v2:0",
        region: str | None = None,
        profile: str | None = None,
        dimensions: int | None = None,
    ):
        self._model = model
        _region = region or os.environ.get("BEDROCK_REGION", "us-east-1")
        _profile = profile or os.environ.get("AWS_PROFILE")
        session_kwargs: dict[str, Any] = {"region_name": _region}
        if _profile:
            session_kwargs["profile_name"] = _profile
        self._client = boto3.Session(**session_kwargs).client("bedrock-runtime")
        self._dimensions = dimensions or _MODEL_DIMENSIONS.get(self._model, 1024)

    @property
    def name(self) -> str:
        return "bedrock"

    @property
    def default_model(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, texts: list[str], *, model: str | None = None) -> EmbeddingResult:
        """调用 Titan Embedding，逐条请求后合并结果"""
        use_model = model or self._model
        vectors: list[list[float]] = []
        total_tokens = 0

        for text in texts:
            body: dict[str, Any] = {
                "inputText": text,
                "dimensions": self._dimensions,
            }
            resp = self._client.invoke_model(
                modelId=use_model,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body),
            )
            result = json.loads(resp["body"].read())
            vectors.append(result["embedding"])
            total_tokens += result.get("inputTextTokenCount", 0)

        logger.debug(
            "Bedrock embedding: model=%s, texts=%d, tokens=%d",
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
