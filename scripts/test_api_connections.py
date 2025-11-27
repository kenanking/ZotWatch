#!/usr/bin/env python3
"""Test API connections for ZotWatch.

This script checks if all required environment variables are set
and tests the connection to each API service.

Usage:
    uv run python scripts/test_api_connections.py
"""

import os
import sys
from dataclasses import dataclass
from enum import Enum


class Status(Enum):
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TestResult:
    name: str
    status: Status
    message: str


# Environment variable definitions
ENV_VARS = {
    "ZOTERO_API_KEY": {"required": True, "description": "Zotero API key"},
    "ZOTERO_USER_ID": {"required": True, "description": "Zotero user ID"},
    "VOYAGE_API_KEY": {"required": True, "description": "Voyage AI API key"},
    "CROSSREF_MAILTO": {"required": False, "description": "Crossref polite pool email"},
    "OPENROUTER_API_KEY": {"required": False, "description": "OpenRouter API key"},
    "MOONSHOT_API_KEY": {"required": False, "description": "Kimi (Moonshot) API key"},
}


def print_header():
    """Print the test header."""
    print()
    print("=" * 64)
    print("            ZotWatch API Connection Test")
    print("=" * 64)
    print()


def print_section(title: str):
    """Print a section header."""
    print(f"\n{title}")
    print("-" * 40)


def format_status(status: Status, message: str) -> str:
    """Format status with icon."""
    icons = {
        Status.SUCCESS: "\u2705",  # Green checkmark
        Status.FAILED: "\u274c",  # Red X
        Status.SKIPPED: "\u26a0\ufe0f",  # Warning sign
    }
    return f"{icons[status]} {message}"


def check_env_vars() -> dict[str, bool]:
    """Check which environment variables are set."""
    print_section("Environment Variables")

    results = {}
    for var_name, config in ENV_VARS.items():
        value = os.environ.get(var_name)
        is_set = bool(value and value.strip())
        results[var_name] = is_set

        required_tag = "(required)" if config["required"] else "(optional)"
        if is_set:
            # Mask the value for security
            masked = value[:4] + "..." + value[-4:] if len(value) > 10 else "***"
            print(f"  {var_name:22} \u2705 Set [{masked}] {required_tag}")
        else:
            icon = "\u274c" if config["required"] else "\u26a0\ufe0f"
            status = "Not set" if config["required"] else "Not set"
            print(f"  {var_name:22} {icon} {status} {required_tag}")

    return results


def test_zotero() -> TestResult:
    """Test Zotero API connection."""
    import requests

    api_key = os.environ.get("ZOTERO_API_KEY")
    user_id = os.environ.get("ZOTERO_USER_ID")

    if not api_key or not user_id:
        return TestResult("Zotero API", Status.FAILED, "Missing API key or user ID")

    try:
        session = requests.Session()
        session.headers.update(
            {
                "Zotero-API-Version": "3",
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "ZotWatch/0.2",
            }
        )

        resp = session.get(
            f"https://api.zotero.org/users/{user_id}/items",
            params={"limit": 1},
            timeout=30,
        )

        if resp.status_code == 200:
            total = resp.headers.get("Total-Results", "unknown")
            return TestResult("Zotero API", Status.SUCCESS, f"Connected (library has {total} items)")
        elif resp.status_code == 403:
            return TestResult("Zotero API", Status.FAILED, "Invalid API key or insufficient permissions")
        elif resp.status_code == 404:
            return TestResult("Zotero API", Status.FAILED, f"User ID '{user_id}' not found")
        else:
            return TestResult("Zotero API", Status.FAILED, f"HTTP {resp.status_code}: {resp.text[:100]}")

    except requests.exceptions.Timeout:
        return TestResult("Zotero API", Status.FAILED, "Connection timeout")
    except requests.exceptions.RequestException as e:
        return TestResult("Zotero API", Status.FAILED, f"Connection error: {e}")


def test_voyage() -> TestResult:
    """Test Voyage AI API connection."""
    api_key = os.environ.get("VOYAGE_API_KEY")

    if not api_key:
        return TestResult("Voyage AI", Status.FAILED, "Missing API key")

    try:
        import voyageai
        import numpy as np

        client = voyageai.Client(api_key=api_key)
        result = client.embed(
            ["test connection"],
            model="voyage-3.5",
            input_type="document",
        )

        embeddings = np.asarray(result.embeddings, dtype=np.float32)
        dim = embeddings.shape[1]

        if dim == 1024:
            return TestResult("Voyage AI", Status.SUCCESS, f"Connected (embedding dim: {dim})")
        else:
            return TestResult("Voyage AI", Status.FAILED, f"Unexpected embedding dimension: {dim}")

    except voyageai.error.AuthenticationError:
        return TestResult("Voyage AI", Status.FAILED, "Invalid API key")
    except voyageai.error.RateLimitError:
        return TestResult("Voyage AI", Status.FAILED, "Rate limit exceeded")
    except Exception as e:
        return TestResult("Voyage AI", Status.FAILED, f"Error: {e}")


def test_crossref() -> TestResult:
    """Test Crossref API connection."""
    import requests

    mailto = os.environ.get("CROSSREF_MAILTO")

    if not mailto:
        return TestResult("Crossref", Status.SKIPPED, "CROSSREF_MAILTO not set")

    try:
        params = {
            "rows": 1,
            "mailto": mailto,
        }

        resp = requests.get(
            "https://api.crossref.org/works",
            params=params,
            timeout=30,
        )

        if resp.status_code == 200:
            data = resp.json()
            total = data.get("message", {}).get("total-results", 0)
            return TestResult("Crossref", Status.SUCCESS, f"Connected (total works: {total:,})")
        else:
            return TestResult("Crossref", Status.FAILED, f"HTTP {resp.status_code}")

    except requests.exceptions.Timeout:
        return TestResult("Crossref", Status.FAILED, "Connection timeout")
    except requests.exceptions.RequestException as e:
        return TestResult("Crossref", Status.FAILED, f"Connection error: {e}")


def test_arxiv() -> TestResult:
    """Test arXiv API connection."""
    import time
    import requests
    import feedparser

    try:
        # arXiv has strict rate limits, wait a bit before request
        time.sleep(1)

        params = {
            "search_query": "cat:cs.LG",
            "max_results": 1,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }

        resp = requests.get(
            "https://export.arxiv.org/api/query",
            params=params,
            timeout=30,
        )

        if resp.status_code == 200:
            feed = feedparser.parse(resp.text)
            if feed.entries:
                title = feed.entries[0].get("title", "")[:50]
                return TestResult("arXiv", Status.SUCCESS, f"Connected (latest: {title}...)")
            else:
                return TestResult("arXiv", Status.SUCCESS, "Connected (no entries found)")
        elif resp.status_code == 429:
            # Rate limited - this is not a configuration error
            return TestResult("arXiv", Status.SUCCESS, "Connected (rate limited, but API is reachable)")
        else:
            return TestResult("arXiv", Status.FAILED, f"HTTP {resp.status_code}")

    except requests.exceptions.Timeout:
        return TestResult("arXiv", Status.FAILED, "Connection timeout")
    except requests.exceptions.RequestException as e:
        return TestResult("arXiv", Status.FAILED, f"Connection error: {e}")


def test_openrouter() -> TestResult:
    """Test OpenRouter API connection."""
    import requests

    api_key = os.environ.get("OPENROUTER_API_KEY")

    if not api_key:
        return TestResult("OpenRouter", Status.SKIPPED, "OPENROUTER_API_KEY not set")

    try:
        # Use a minimal request to test authentication
        # We'll use the models endpoint which doesn't cost tokens
        resp = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={
                "Authorization": f"Bearer {api_key}",
            },
            timeout=30,
        )

        if resp.status_code == 200:
            data = resp.json()
            model_count = len(data.get("data", []))
            return TestResult("OpenRouter", Status.SUCCESS, f"Connected ({model_count} models available)")
        elif resp.status_code == 401:
            return TestResult("OpenRouter", Status.FAILED, "Invalid API key")
        else:
            return TestResult("OpenRouter", Status.FAILED, f"HTTP {resp.status_code}")

    except requests.exceptions.Timeout:
        return TestResult("OpenRouter", Status.FAILED, "Connection timeout")
    except requests.exceptions.RequestException as e:
        return TestResult("OpenRouter", Status.FAILED, f"Connection error: {e}")


def test_kimi() -> TestResult:
    """Test Kimi (Moonshot) API connection."""
    import requests

    api_key = os.environ.get("MOONSHOT_API_KEY")

    if not api_key:
        return TestResult("Kimi", Status.SKIPPED, "MOONSHOT_API_KEY not set")

    try:
        # Use the models endpoint to test authentication
        resp = requests.get(
            "https://api.moonshot.cn/v1/models",
            headers={
                "Authorization": f"Bearer {api_key}",
            },
            timeout=30,
        )

        if resp.status_code == 200:
            data = resp.json()
            model_count = len(data.get("data", []))
            return TestResult("Kimi", Status.SUCCESS, f"Connected ({model_count} models available)")
        elif resp.status_code == 401:
            return TestResult("Kimi", Status.FAILED, "Invalid API key")
        else:
            return TestResult("Kimi", Status.FAILED, f"HTTP {resp.status_code}: {resp.text[:100]}")

    except requests.exceptions.Timeout:
        return TestResult("Kimi", Status.FAILED, "Connection timeout")
    except requests.exceptions.RequestException as e:
        return TestResult("Kimi", Status.FAILED, f"Connection error: {e}")


def run_tests() -> list[TestResult]:
    """Run all API connection tests."""
    print_section("API Connection Tests")

    tests = [
        ("Zotero API", test_zotero),
        ("Voyage AI", test_voyage),
        ("Crossref", test_crossref),
        ("arXiv", test_arxiv),
        ("OpenRouter", test_openrouter),
        ("Kimi", test_kimi),
    ]

    results = []
    for i, (name, test_func) in enumerate(tests, 1):
        print(f"  [{i}/{len(tests)}] {name:15} ", end="", flush=True)
        result = test_func()
        results.append(result)
        print(format_status(result.status, result.message))

    return results


def print_summary(results: list[TestResult]) -> int:
    """Print test summary and return exit code."""
    passed = sum(1 for r in results if r.status == Status.SUCCESS)
    failed = sum(1 for r in results if r.status == Status.FAILED)
    skipped = sum(1 for r in results if r.status == Status.SKIPPED)

    print()
    print("=" * 64)
    print(f"  Result: {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 64)
    print()

    # Show failed tests details
    failed_tests = [r for r in results if r.status == Status.FAILED]
    if failed_tests:
        print("Failed tests:")
        for r in failed_tests:
            print(f"  \u274c {r.name}: {r.message}")
        print()

    return 1 if failed > 0 else 0


def main():
    """Main entry point."""
    print_header()

    # Check environment variables
    env_status = check_env_vars()

    # Check required variables
    missing_required = [name for name, config in ENV_VARS.items() if config["required"] and not env_status.get(name)]

    if missing_required:
        print()
        print("\u274c Missing required environment variables:")
        for var in missing_required:
            print(f"   - {var}")
        print()
        print("Please set these variables before running the tests.")
        sys.exit(1)

    # Run API tests
    results = run_tests()

    # Print summary and exit
    exit_code = print_summary(results)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
