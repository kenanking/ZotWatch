"""Factory functions for creating embedding providers and rerankers.

Provides a single point of creation for embedding and reranking services,
supporting multiple providers (Voyage AI, DashScope) based on configuration.
"""

from zotwatch.config.settings import EmbeddingConfig, ScoringConfig
from zotwatch.core.exceptions import ConfigurationError

from .base import BaseEmbeddingProvider, BaseReranker
from .dashscope import DashScopeEmbedding, DashScopeReranker
from .voyage import VoyageEmbedding, VoyageReranker

# Supported embedding providers
SUPPORTED_EMBEDDING_PROVIDERS = frozenset({"voyage", "dashscope"})

# Supported rerank providers
SUPPORTED_RERANK_PROVIDERS = frozenset({"voyage", "dashscope"})


def create_embedding_provider(config: EmbeddingConfig) -> BaseEmbeddingProvider:
    """Create embedding provider based on configuration.

    Args:
        config: Embedding configuration containing provider and settings.

    Returns:
        Configured embedding provider instance.

    Raises:
        ConfigurationError: If the provider is not supported.
    """
    provider = config.provider.lower()

    if provider == "voyage":
        return VoyageEmbedding(
            model_name=config.model,
            api_key=config.api_key,
            batch_size=config.batch_size,
        )
    elif provider == "dashscope":
        return DashScopeEmbedding(
            model_name=config.model,
            api_key=config.api_key,
            batch_size=config.batch_size,
        )
    else:
        raise ConfigurationError(
            f"Unknown embedding provider: {config.provider}. "
            f"Supported providers: {', '.join(sorted(SUPPORTED_EMBEDDING_PROVIDERS))}"
        )


def create_reranker(
    rerank_config: ScoringConfig.RerankConfig,
    embedding_config: EmbeddingConfig,
) -> BaseReranker:
    """Create reranker based on configuration.

    Args:
        rerank_config: Rerank configuration containing provider and model.
        embedding_config: Embedding configuration (provider must match).

    Returns:
        Configured reranker instance.

    Raises:
        ConfigurationError: If providers don't match or are not supported.
    """
    # Validate provider coupling
    if rerank_config.provider != embedding_config.provider:
        raise ConfigurationError(
            f"Rerank provider '{rerank_config.provider}' must match "
            f"embedding provider '{embedding_config.provider}'. "
            f"ZotWatch requires both to use the same provider."
        )

    provider = rerank_config.provider.lower()
    api_key = embedding_config.api_key  # Share API key from embedding

    if provider == "voyage":
        return VoyageReranker(
            api_key=api_key,
            model=rerank_config.model,
        )
    elif provider == "dashscope":
        return DashScopeReranker(
            api_key=api_key,
            model=rerank_config.model,
        )
    else:
        raise ConfigurationError(
            f"Unknown rerank provider: {rerank_config.provider}. "
            f"Supported providers: {', '.join(sorted(SUPPORTED_RERANK_PROVIDERS))}"
        )


__all__ = [
    "create_embedding_provider",
    "create_reranker",
    "SUPPORTED_EMBEDDING_PROVIDERS",
    "SUPPORTED_RERANK_PROVIDERS",
]
