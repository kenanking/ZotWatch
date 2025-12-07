"""DeepSeek LLM provider implementation."""

import logging

from zotwatch.config.settings import LLMConfig
from zotwatch.core.protocols import LLMResponse

from .http_client import BaseHTTPLLMClient

logger = logging.getLogger(__name__)


class DeepSeekClient(BaseHTTPLLMClient):
    """DeepSeek API client.

    Supports both standard models (deepseek-chat) and reasoning models (deepseek-reasoner).
    Reasoning models automatically disable temperature and use higher max_tokens.
    """

    BASE_URL = "https://api.deepseek.com"
    # Models that use the reasoning/thinking feature
    REASONING_MODELS = ("deepseek-reasoner",)
    MIN_REASONING_TOKENS = 8192

    def __init__(
        self,
        api_key: str,
        default_model: str = "deepseek-chat",
        timeout: float = 60.0,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
    ) -> None:
        """Initialize DeepSeek client.

        Args:
            api_key: DeepSeek API key.
            default_model: Default model to use.
            timeout: Request timeout in seconds (higher for reasoning models).
            max_retries: Maximum retry attempts.
            backoff_factor: Exponential backoff factor.
        """
        super().__init__(
            api_key=api_key,
            default_model=default_model,
            timeout=timeout,
            max_retries=max_retries,
            backoff_factor=backoff_factor,
        )

    @classmethod
    def from_config(cls, config: LLMConfig) -> "DeepSeekClient":
        """Create client from LLM configuration."""
        # Use longer timeout for reasoning models
        timeout = 120.0 if "reasoner" in config.model else 60.0
        return cls(
            api_key=config.api_key,
            default_model=config.model,
            timeout=timeout,
            max_retries=config.retry.max_attempts,
            backoff_factor=config.retry.backoff_factor,
        )

    @property
    def name(self) -> str:
        return "deepseek"

    def _is_reasoning_model(self, model: str) -> bool:
        """Check if the model is a reasoning model."""
        return any(model.startswith(prefix) for prefix in self.REASONING_MODELS)

    def _adjust_parameters(
        self,
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, int, float]:
        """Adjust parameters for reasoning models.

        Reasoning models don't support temperature/top_p parameters,
        and need higher max_tokens for chain-of-thought output.
        """
        if self._is_reasoning_model(model):
            # Reasoning models ignore temperature, but we still pass it through
            # (will be excluded in _build_payload)
            if max_tokens < self.MIN_REASONING_TOKENS:
                max_tokens = self.MIN_REASONING_TOKENS
                logger.debug(
                    "Increased max_tokens to %d for reasoning model %s",
                    max_tokens,
                    model,
                )
        return model, max_tokens, temperature

    def _build_headers(self) -> dict[str, str]:
        """Build HTTP headers for DeepSeek API."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _build_payload(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> dict:
        """Build JSON payload for DeepSeek API.

        Note: For reasoning models, temperature and top_p are not supported
        and will not be included in the payload.
        """
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }

        # Reasoning models don't support temperature/top_p parameters
        if not self._is_reasoning_model(model):
            payload["temperature"] = temperature

        return payload

    def _extract_response(self, data: dict, model: str) -> LLMResponse:
        """Extract LLMResponse from DeepSeek API response."""
        message = data["choices"][0]["message"]
        # Extract content, ignoring reasoning_content for reasoning models
        content = message.get("content", "")
        tokens_used = data.get("usage", {}).get("total_tokens", 0)

        return LLMResponse(
            content=content,
            model=data.get("model", model),
            tokens_used=tokens_used,
        )

    def available_models(self) -> list[str]:
        """List available DeepSeek models."""
        return [
            "deepseek-chat",
            "deepseek-reasoner",
        ]


__all__ = ["DeepSeekClient"]
