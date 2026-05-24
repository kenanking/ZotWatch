"""Retry logic for LLM API calls."""

import functools
import logging
import random
import time
from typing import Callable, ParamSpec, TypeVar

import requests

from zotwatch.core.exceptions import NetworkError

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")

# HTTP status codes that should trigger a retry
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

# Jitter range as fraction of delay (0.1 = ±10%)
DEFAULT_JITTER = 0.1


def _add_jitter(delay: float, jitter: float = DEFAULT_JITTER) -> float:
    """Add random jitter to delay to prevent thundering herd.

    Args:
        delay: Base delay in seconds.
        jitter: Jitter range as fraction of delay.

    Returns:
        Delay with random jitter applied.
    """
    return delay * (1 + random.uniform(-jitter, jitter))


def _get_retry_after(response: requests.Response | None, default: float) -> float:
    """Extract Retry-After header value if present.

    Args:
        response: HTTP response object.
        default: Default delay if header not present.

    Returns:
        Delay in seconds from Retry-After header or default.
    """
    if response is None:
        return default
    retry_after = response.headers.get("Retry-After")
    if retry_after:
        try:
            return float(retry_after)
        except ValueError:
            pass
    return default


def with_retry(
    max_attempts: int = 3,
    backoff_factor: float = 2.0,
    initial_delay: float = 1.0,
    jitter: float = DEFAULT_JITTER,
    from_instance: bool = False,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator for retry logic with exponential backoff and jitter.

    Args:
        max_attempts: Maximum number of retry attempts.
        backoff_factor: Multiplier for delay between retries.
        initial_delay: Initial delay before first retry in seconds.
        jitter: Random jitter range as fraction of delay (default 0.1 = ±10%).
        from_instance: If True, read max_attempts and backoff_factor from
            the first argument (self) as self.max_retries and self.backoff_factor.

    Returns:
        Decorated function with retry logic.

    Raises:
        NetworkError: If all retry attempts fail.
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Allow instance-level override of retry parameters
            attempts = max_attempts
            backoff = backoff_factor
            if from_instance and args:
                self = args[0]
                if hasattr(self, "max_retries"):
                    attempts = self.max_retries
                if hasattr(self, "backoff_factor"):
                    backoff = self.backoff_factor

            last_exception: Exception | None = None
            last_status_code: int | None = None
            delay = initial_delay
            func_name = func.__qualname__

            for attempt in range(attempts):
                try:
                    return func(*args, **kwargs)
                except requests.exceptions.HTTPError as e:
                    status_code = e.response.status_code if e.response is not None else None
                    if status_code is not None and status_code not in RETRYABLE_STATUS_CODES:
                        # Non-retryable HTTP error
                        raise NetworkError(
                            f"{func_name}: HTTP {status_code} error",
                            url=str(e.response.url) if e.response is not None else None,
                        ) from e

                    last_exception = e
                    last_status_code = status_code

                    # Use Retry-After header for 429 rate limiting
                    if status_code == 429:
                        delay = _get_retry_after(e.response, delay)

                    logger.warning(
                        "%s: attempt %d/%d failed with HTTP %s, retrying in %.1fs",
                        func_name,
                        attempt + 1,
                        attempts,
                        status_code or "unknown",
                        delay,
                    )
                except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                    last_exception = e
                    logger.warning(
                        "%s: attempt %d/%d failed with %s, retrying in %.1fs",
                        func_name,
                        attempt + 1,
                        attempts,
                        type(e).__name__,
                        delay,
                    )

                if attempt < attempts - 1:
                    time.sleep(_add_jitter(delay, jitter))
                    delay *= backoff

            # All retries exhausted - raise NetworkError with context
            error_detail = f"HTTP {last_status_code}" if last_status_code else type(last_exception).__name__
            raise NetworkError(
                f"{func_name}: failed after {attempts} attempts ({error_detail})",
            ) from last_exception

        return wrapper

    return decorator


__all__ = ["with_retry", "RETRYABLE_STATUS_CODES"]
