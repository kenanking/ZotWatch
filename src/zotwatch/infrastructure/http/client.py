"""HTTP client with retry logic."""

import logging
import time
from typing import Any

import requests

from zotwatch.core.exceptions import NetworkError

logger = logging.getLogger(__name__)


class HTTPClient:
    """HTTP client with session management and retry logic."""

    def __init__(
        self,
        headers: dict[str, str] | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        retryable_statuses: set[int] | None = None,
    ):
        self.session = requests.Session()
        if headers:
            self.session.headers.update(headers)
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.retryable_statuses = retryable_statuses or {429, 502, 503}

    def get(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        **kwargs,
    ) -> requests.Response:
        """Make GET request with retry logic."""
        return self._request("GET", url, params=params, headers=headers, **kwargs)

    def post(
        self,
        url: str,
        json: Any | None = None,
        headers: dict[str, str] | None = None,
        **kwargs,
    ) -> requests.Response:
        """Make POST request with retry logic."""
        return self._request("POST", url, json=json, headers=headers, **kwargs)

    def _request(self, method: str, url: str, **kwargs) -> requests.Response:
        """Make request with retry logic."""
        kwargs.setdefault("timeout", self.timeout)
        last_exception = None
        delay = 1.0

        for attempt in range(self.max_retries):
            try:
                response = self.session.request(method, url, **kwargs)
                if response.status_code in self.retryable_statuses:
                    retry_after = self._get_retry_delay(response, delay)
                    logger.warning(
                        "%s %s returned %d, retrying in %.1fs (attempt %d/%d)",
                        method,
                        url,
                        response.status_code,
                        retry_after,
                        attempt + 1,
                        self.max_retries,
                    )
                    if attempt < self.max_retries - 1:
                        time.sleep(retry_after)
                        delay *= self.backoff_factor
                        continue
                    # All retries exhausted for retryable status
                    raise NetworkError(
                        f"{method} {url}: failed after {self.max_retries} retries "
                        f"(HTTP {response.status_code})",
                        url=url,
                    )
                return response
            except requests.exceptions.RequestException as e:
                last_exception = e
                logger.warning(
                    "Request failed: %s (attempt %d/%d)",
                    str(e),
                    attempt + 1,
                    self.max_retries,
                )
                if attempt < self.max_retries - 1:
                    time.sleep(delay)
                    delay *= self.backoff_factor

        if last_exception:
            raise NetworkError(f"Request failed after {self.max_retries} retries: {last_exception}", url=url)
        raise NetworkError(f"Request failed after {self.max_retries} retries", url=url)

    @staticmethod
    def _get_retry_delay(response: requests.Response, default: float) -> float:
        """Get retry delay from Retry-After header or use default."""
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                pass
        return default


__all__ = ["HTTPClient"]
