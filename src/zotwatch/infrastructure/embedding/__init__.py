"""Embedding providers and caching infrastructure."""

from .base import BaseEmbeddingProvider, BaseReranker
from .cache import EmbeddingCache
from .cached import CachingEmbeddingProvider
from .dashscope import DashScopeEmbedding, DashScopeReranker
from .factory import (
    SUPPORTED_EMBEDDING_PROVIDERS,
    SUPPORTED_RERANK_PROVIDERS,
    create_embedding_provider,
    create_reranker,
)
from .faiss_index import FaissIndex
from .voyage import VoyageEmbedding, VoyageReranker

__all__ = [
    # Base classes
    "BaseEmbeddingProvider",
    "BaseReranker",
    "CachingEmbeddingProvider",
    "EmbeddingCache",
    "FaissIndex",
    # Voyage AI
    "VoyageEmbedding",
    "VoyageReranker",
    # DashScope (Alibaba Cloud)
    "DashScopeEmbedding",
    "DashScopeReranker",
    # Factory functions
    "create_embedding_provider",
    "create_reranker",
    "SUPPORTED_EMBEDDING_PROVIDERS",
    "SUPPORTED_RERANK_PROVIDERS",
]
