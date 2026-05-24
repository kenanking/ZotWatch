"""Base class for HTTP-based LLM providers."""

import logging
from abc import abstractmethod

import requests

from zotwatch.config.settings import LLMConfig
from zotwatch.core.protocols import LLMResponse

from .base import BaseLLMProvider
from .retry import with_retry

logger = logging.getLogger(__name__)


class BaseHTTPLLMClient(BaseLLMProvider):
    """Abstract base class for HTTP-based LLM providers.

    Provides common functionality for API key management, session handling,
    retry logic, and request/response processing.

    Subclasses must implement:
    - BASE_URL: Class variable with API endpoint
    - name property: Provider name
    - _build_headers(): Build request headers
    - _build_payload(): Build request JSON payload
    - _extract_response(): Extract LLMResponse from API response
    """

    BASE_URL: str  # Subclass must define

    def __init__(
        self,
        api_key: str,
        default_model: str,
        timeout: float = 60.0,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
    ) -> None:
        """Initialize HTTP LLM client.

        Args:
            api_key: API key for authentication.
            default_model: Default model to use for completions.
            timeout: Request timeout in seconds.
            max_retries: Maximum retry attempts.
            backoff_factor: Exponential backoff factor.
        """
        self.api_key = api_key
        self.default_model = default_model
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self._session = requests.Session()

    @classmethod
    def from_config(cls, config: LLMConfig) -> "BaseHTTPLLMClient":
        """Create client from LLM configuration.

        Args:
            config: LLM configuration object.

        Returns:
            Configured client instance.
        """
        return cls(
            api_key=config.api_key,
            default_model=config.model,
            max_retries=config.retry.max_attempts,
            backoff_factor=config.retry.backoff_factor,
        )

    @abstractmethod
    def _build_headers(self) -> dict[str, str]:
        """Build HTTP headers for the request.

        Returns:
            Dict of headers including authorization.
        """
        ...

    @abstractmethod
    def _build_payload(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> dict:
        """Build JSON payload for the request.

        Args:
            prompt: User prompt text.
            model: Model identifier.
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.

        Returns:
            Dict of request payload.
        """
        ...

    @abstractmethod
    def _extract_response(self, data: dict, model: str) -> LLMResponse:
        """Extract LLMResponse from API response data.

        Args:
            data: Parsed JSON response from API.
            model: Model that was used.

        Returns:
            Structured LLMResponse object.
        """
        ...

    def complete(
        self,
        prompt: str,
        *,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> LLMResponse:
        """Send completion request to the API.

        Args:
            prompt: User prompt text.
            model: Optional model override.
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.

        Returns:
            LLMResponse with completion result.
        """
        return self._complete_with_retry(
            prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    @with_retry(max_attempts=3, backoff_factor=2.0, initial_delay=1.0, from_instance=True)
    def _complete_with_retry(
        self,
        prompt: str,
        *,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> LLMResponse:
        """Internal completion with retry logic."""
        use_model = model or self.default_model

        # Allow subclasses to adjust parameters
        use_model, max_tokens, temperature = self._adjust_parameters(use_model, max_tokens, temperature)

        response = self._session.post(
            f"{self.BASE_URL}/chat/completions",
            headers=self._build_headers(),
            json=self._build_payload(prompt, use_model, max_tokens, temperature),
            timeout=self.timeout,
        )
        response.raise_for_status()

        data = response.json()
        return self._extract_response(data, use_model)

    def _adjust_parameters(
        self,
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, int, float]:
        """Adjust parameters before sending request.

        Override in subclass to modify parameters (e.g., for thinking models).

        Args:
            model: Model identifier.
            max_tokens: Maximum tokens.
            temperature: Sampling temperature.

        Returns:
            Tuple of (model, max_tokens, temperature).
        """
        return model, max_tokens, temperature


__all__ = ["BaseHTTPLLMClient"]
