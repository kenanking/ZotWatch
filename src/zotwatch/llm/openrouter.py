"""OpenRouter LLM provider implementation."""

import logging

from zotwatch.config.settings import LLMConfig
from zotwatch.core.protocols import LLMResponse

from .http_client import BaseHTTPLLMClient

logger = logging.getLogger(__name__)


class OpenRouterClient(BaseHTTPLLMClient):
    """OpenRouter API client supporting multiple LLM providers."""

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        api_key: str,
        default_model: str = "anthropic/claude-3.5-sonnet",
        site_url: str = "https://github.com/zotwatch/zotwatch",
        app_name: str = "ZotWatch",
        timeout: float = 60.0,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
    ) -> None:
        """Initialize OpenRouter client.

        Args:
            api_key: OpenRouter API key.
            default_model: Default model to use.
            site_url: Site URL for OpenRouter attribution.
            app_name: Application name for OpenRouter attribution.
            timeout: Request timeout in seconds.
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
        self.site_url = site_url
        self.app_name = app_name

    @classmethod
    def from_config(cls, config: LLMConfig) -> "OpenRouterClient":
        """Create client from LLM configuration."""
        return cls(
            api_key=config.api_key,
            default_model=config.model,
            max_retries=config.retry.max_attempts,
            backoff_factor=config.retry.backoff_factor,
        )

    @property
    def name(self) -> str:
        return "openrouter"

    def _build_headers(self) -> dict[str, str]:
        """Build HTTP headers with OpenRouter-specific headers."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": self.site_url,
            "X-Title": self.app_name,
            "Content-Type": "application/json",
        }

    def _build_payload(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> dict:
        """Build JSON payload for OpenRouter API."""
        return {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

    def _extract_response(self, data: dict, model: str) -> LLMResponse:
        """Extract LLMResponse from OpenRouter API response."""
        try:
            choices = data.get("choices")
            if not choices or not isinstance(choices, list):
                raise ValueError(f"Empty or invalid 'choices' in response: {data}")
            content = choices[0]["message"]["content"]
            tokens_used = data.get("usage", {}).get("total_tokens", 0)

            return LLMResponse(
                content=content,
                model=data.get("model", model),
                tokens_used=tokens_used,
            )
        except (KeyError, IndexError, ValueError) as e:
            from zotwatch.core.exceptions import LLMError

            raise LLMError(
                provider="openrouter",
                message=f"API returned malformed response: {e}",
            ) from e

    def available_models(self) -> list[str]:
        """Get available models from OpenRouter."""
        try:
            response = self._session.get(
                f"{self.BASE_URL}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            return [m["id"] for m in data.get("data", [])]
        except Exception as e:
            logger.warning("Failed to fetch available models: %s", e)
            return []


__all__ = ["OpenRouterClient"]
