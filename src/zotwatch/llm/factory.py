"""LLM client factory.

Provides a single point of creation for LLM clients,
eliminating duplicate creation code throughout the codebase.
"""

from zotwatch.config.settings import LLMConfig
from zotwatch.core.exceptions import ConfigurationError
from zotwatch.llm.base import BaseLLMProvider
from zotwatch.llm.deepseek import DeepSeekClient
from zotwatch.llm.kimi import KimiClient
from zotwatch.llm.openrouter import OpenRouterClient

# Supported providers
SUPPORTED_PROVIDERS = frozenset({"openrouter", "kimi", "deepseek"})


def create_llm_client(config: LLMConfig) -> BaseLLMProvider:
    """Create LLM client based on provider configuration.

    Args:
        config: LLM configuration containing provider and API settings.

    Returns:
        Configured LLM client instance.

    Raises:
        ConfigurationError: If the provider is not supported.
    """
    provider = config.provider.lower()

    if provider == "kimi":
        return KimiClient.from_config(config)
    elif provider == "openrouter":
        return OpenRouterClient.from_config(config)
    elif provider == "deepseek":
        return DeepSeekClient.from_config(config)
    else:
        raise ConfigurationError(
            f"Unknown LLM provider: {config.provider}. Supported providers: {', '.join(sorted(SUPPORTED_PROVIDERS))}"
        )


__all__ = ["create_llm_client", "SUPPORTED_PROVIDERS"]
