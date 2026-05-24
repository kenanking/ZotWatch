"""DeepSeek LLM provider implementation."""

import logging

from zotwatch.config.settings import LLMConfig
from zotwatch.core.protocols import LLMResponse

from .http_client import BaseHTTPLLMClient

logger = logging.getLogger(__name__)


class DeepSeekClient(BaseHTTPLLMClient):
    """DeepSeek API client.

    Supports standard models (deepseek-v4-flash) and thinking models (deepseek-v4-pro).
    Thinking models use the ``thinking`` parameter to enable chain-of-thought reasoning.

    Legacy model names (deepseek-chat, deepseek-reasoner) are still accepted but
    deprecated upstream — they map to v4-flash and v4-pro respectively.
    """

    BASE_URL = "https://api.deepseek.com"
    # Models that support the thinking/reasoning feature
    THINKING_MODELS = ("deepseek-v4-pro", "deepseek-reasoner")
    MIN_THINKING_TOKENS = 8192

    def __init__(
        self,
        api_key: str,
        default_model: str = "deepseek-v4-flash",
        timeout: float = 60.0,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
    ) -> None:
        """Initialize DeepSeek client.

        Args:
            api_key: DeepSeek API key.
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
    def from_config(cls, config: LLMConfig) -> "DeepSeekClient":
        """Create client from LLM configuration."""
        # Use longer timeout for thinking models
        is_thinking = any(
            config.model.startswith(p) for p in cls.THINKING_MODELS
        )
        timeout = 120.0 if is_thinking else 60.0
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

    def _is_thinking_model(self, model: str) -> bool:
        """Check if the model supports thinking/reasoning mode."""
        return any(model.startswith(prefix) for prefix in self.THINKING_MODELS)

    def _adjust_parameters(
        self,
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, int, float]:
        """Adjust parameters for thinking models.

        Thinking models need higher max_tokens for chain-of-thought output.
        Temperature is still passed through (excluded in _build_payload for thinking models).
        """
        if self._is_thinking_model(model):
            if max_tokens < self.MIN_THINKING_TOKENS:
                max_tokens = self.MIN_THINKING_TOKENS
                logger.debug(
                    "Increased max_tokens to %d for thinking model %s",
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

        Thinking models (deepseek-v4-pro) get the ``thinking`` parameter
        and omit temperature/top_p. Standard models use temperature as usual.
        """
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }

        if self._is_thinking_model(model):
            payload["thinking"] = {"type": "enabled"}
        else:
            payload["temperature"] = temperature

        return payload

    def _extract_response(self, data: dict, model: str) -> LLMResponse:
        """Extract LLMResponse from DeepSeek API response."""
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
                provider="deepseek",
                message=f"API returned malformed response: {e}",
            ) from e

    def available_models(self) -> list[str]:
        """List available DeepSeek models."""
        return [
            "deepseek-v4-flash",
            "deepseek-v4-pro",
        ]


__all__ = ["DeepSeekClient"]
