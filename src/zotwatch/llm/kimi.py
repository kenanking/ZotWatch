"""Kimi (Moonshot AI) LLM provider implementation."""

import logging

from zotwatch.config.settings import LLMConfig
from zotwatch.core.protocols import LLMResponse

from .http_client import BaseHTTPLLMClient

logger = logging.getLogger(__name__)


class KimiClient(BaseHTTPLLMClient):
    """Kimi (Moonshot AI) API client.

    Supports both thinking models (kimi-k2-thinking-*) and standard models.
    Thinking models automatically use temperature=1.0 and max_tokens>=16000.
    """

    BASE_URL = "https://api.moonshot.cn/v1"
    # Models that use the thinking/reasoning feature
    THINKING_MODEL_PREFIXES = ("kimi-k2-thinking",)
    MIN_THINKING_TOKENS = 16000

    def __init__(
        self,
        api_key: str,
        default_model: str = "kimi-k2-thinking-turbo",
        timeout: float = 120.0,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
    ) -> None:
        """Initialize Kimi client.

        Args:
            api_key: Moonshot API key.
            default_model: Default model to use.
            timeout: Request timeout in seconds (higher for thinking models).
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
    def from_config(cls, config: LLMConfig) -> "KimiClient":
        """Create client from LLM configuration."""
        return cls(
            api_key=config.api_key,
            default_model=config.model,
            timeout=120.0,  # Thinking models need longer timeout
            max_retries=config.retry.max_attempts,
            backoff_factor=config.retry.backoff_factor,
        )

    @property
    def name(self) -> str:
        return "kimi"

    def _is_thinking_model(self, model: str) -> bool:
        """Check if the model is a thinking model."""
        return any(model.startswith(prefix) for prefix in self.THINKING_MODEL_PREFIXES)

    def _adjust_parameters(
        self,
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, int, float]:
        """Adjust parameters for thinking models.

        Thinking models require temperature=1.0 and max_tokens >= 16000.
        """
        if self._is_thinking_model(model):
            temperature = 1.0
            if max_tokens < self.MIN_THINKING_TOKENS:
                max_tokens = self.MIN_THINKING_TOKENS
                logger.debug(
                    "Increased max_tokens to %d for thinking model %s",
                    max_tokens,
                    model,
                )
        return model, max_tokens, temperature

    def _build_headers(self) -> dict[str, str]:
        """Build HTTP headers for Kimi API."""
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
        """Build JSON payload for Kimi API."""
        return {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

    def _extract_response(self, data: dict, model: str) -> LLMResponse:
        """Extract LLMResponse from Kimi API response."""
        try:
            choices = data.get("choices")
            if not choices or not isinstance(choices, list):
                raise ValueError(f"Empty or invalid 'choices' in response: {data}")
            message = choices[0]["message"]
            content = message.get("content", "")
            tokens_used = data.get("usage", {}).get("total_tokens", 0)

            return LLMResponse(
                content=content,
                model=data.get("model", model),
                tokens_used=tokens_used,
            )
        except (KeyError, IndexError, ValueError) as e:
            from zotwatch.core.exceptions import LLMError

            raise LLMError(
                provider="kimi",
                message=f"API returned malformed response: {e}",
            ) from e

    def available_models(self) -> list[str]:
        """List available Kimi models."""
        # Kimi doesn't have a models endpoint, return known models
        return [
            "kimi-k2-thinking-turbo",
            "kimi-k2-thinking",
            "kimi-k2-turbo-preview",
        ]


__all__ = ["KimiClient"]
