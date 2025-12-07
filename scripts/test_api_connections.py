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
from pathlib import Path

import requests

# Add dotenv support for local testing
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Import ZotWatch config loader
from zotwatch.config.settings import Settings, load_settings
from zotwatch.core.exceptions import ConfigurationError


class Status(Enum):
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TestResult:
    name: str
    status: Status
    message: str


# Environment variables are now detected dynamically based on config.yaml
# See detect_required_env_vars() function below


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


def load_config() -> Settings | None:
    """Load ZotWatch configuration.

    Returns:
        Settings object if config exists, None otherwise.
    """
    try:
        # Try to load from current directory
        base_dir = Path.cwd()
        config_path = base_dir / "config" / "config.yaml"

        if not config_path.exists():
            print_section("Configuration Loading")
            print("  ⚠️  Warning: config/config.yaml not found")
            print("  Falling back to basic API tests (Voyage + Zotero)")
            print()
            return None

        settings = load_settings(base_dir)
        return settings

    except ConfigurationError as e:
        print_section("Configuration Error")
        print(f"  ❌ Error loading config: {e}")
        print("  Please fix config.yaml before running tests")
        sys.exit(1)

    except Exception as e:
        print_section("Configuration Loading")
        print(f"  ⚠️  Warning: Failed to load config: {e}")
        print("  Falling back to basic API tests")
        print()
        return None


def detect_required_env_vars(settings: Settings | None) -> dict[str, dict]:
    """Detect required environment variables based on configuration.

    Args:
        settings: Loaded settings or None for fallback mode.

    Returns:
        Dict mapping env var names to their config (required, description, reason).
    """
    env_vars = {}

    # Zotero is always required
    env_vars["ZOTERO_API_KEY"] = {
        "required": True,
        "description": "Zotero API key",
        "reason": "Required for all ZotWatch operations",
    }
    env_vars["ZOTERO_USER_ID"] = {
        "required": True,
        "description": "Zotero user ID",
        "reason": "Required for all ZotWatch operations",
    }

    if settings is None:
        # Fallback mode: assume Voyage (default provider)
        env_vars["VOYAGE_API_KEY"] = {
            "required": True,
            "description": "Voyage AI API key",
            "reason": "Default embedding provider (config not loaded)",
        }
        env_vars["MOONSHOT_API_KEY"] = {
            "required": False,
            "description": "Kimi (Moonshot) API key",
            "reason": "LLM provider (status unknown, config not loaded)",
        }
        env_vars["OPENROUTER_API_KEY"] = {
            "required": False,
            "description": "OpenRouter API key",
            "reason": "LLM provider (status unknown, config not loaded)",
        }
        env_vars["DEEPSEEK_API_KEY"] = {
            "required": False,
            "description": "DeepSeek API key",
            "reason": "LLM provider (status unknown, config not loaded)",
        }
    else:
        # Config-aware mode: detect based on settings

        # Embedding provider
        if settings.embedding.provider == "voyage":
            env_vars["VOYAGE_API_KEY"] = {
                "required": True,
                "description": "Voyage AI API key",
                "reason": f"Configured embedding provider: {settings.embedding.model}",
            }
        elif settings.embedding.provider == "dashscope":
            env_vars["DASHSCOPE_API_KEY"] = {
                "required": True,
                "description": "DashScope API key",
                "reason": f"Configured embedding provider: {settings.embedding.model}",
            }

        # LLM provider (only if enabled)
        if settings.llm.enabled:
            if settings.llm.provider == "kimi":
                env_vars["MOONSHOT_API_KEY"] = {
                    "required": True,
                    "description": "Kimi (Moonshot) API key",
                    "reason": f"Configured LLM provider: {settings.llm.model}",
                }
            elif settings.llm.provider == "openrouter":
                env_vars["OPENROUTER_API_KEY"] = {
                    "required": True,
                    "description": "OpenRouter API key",
                    "reason": f"Configured LLM provider: {settings.llm.model}",
                }
            elif settings.llm.provider == "deepseek":
                env_vars["DEEPSEEK_API_KEY"] = {
                    "required": True,
                    "description": "DeepSeek API key",
                    "reason": f"Configured LLM provider: {settings.llm.model}",
                }
        else:
            # LLM disabled - add as optional for informational purposes
            env_vars["MOONSHOT_API_KEY"] = {
                "required": False,
                "description": "Kimi API key",
                "reason": "LLM disabled in config (llm.enabled=false)",
            }
            env_vars["OPENROUTER_API_KEY"] = {
                "required": False,
                "description": "OpenRouter API key",
                "reason": "LLM disabled in config (llm.enabled=false)",
            }
            env_vars["DEEPSEEK_API_KEY"] = {
                "required": False,
                "description": "DeepSeek API key",
                "reason": "LLM disabled in config (llm.enabled=false)",
            }

    # Crossref mailto (always optional but recommended)
    env_vars["CROSSREF_MAILTO"] = {
        "required": False,
        "description": "Crossref polite pool email",
        "reason": "Optional but recommended for faster Crossref API access",
    }

    return env_vars


def check_env_vars(env_vars: dict[str, dict]) -> dict[str, bool]:
    """Check which environment variables are set.

    Args:
        env_vars: Dictionary of environment variable configurations.

    Returns:
        Dict mapping var names to whether they are set.
    """
    print_section("Environment Variables")

    results = {}
    for var_name, config in env_vars.items():
        value = os.environ.get(var_name)
        is_set = bool(value and value.strip())
        results[var_name] = is_set

        required_tag = "(required)" if config["required"] else "(optional)"
        if is_set:
            # Mask the value for security
            masked = value[:4] + "..." + value[-4:] if len(value) > 10 else "***"
            print(f"  {var_name:22} ✅ Set [{masked}] {required_tag}")
            # Show reason if provided
            if config.get("reason"):
                print(f"  {'':22}    → {config['reason']}")
        else:
            icon = "❌" if config["required"] else "⚠️"
            status = "Not set"
            print(f"  {var_name:22} {icon} {status} {required_tag}")
            if config.get("reason"):
                print(f"  {'':22}    → {config['reason']}")

    return results


def test_zotero() -> TestResult:
    """Test Zotero API connection."""
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


def test_voyage_embedding(api_key: str, model: str) -> TestResult:
    """Test Voyage AI embedding API connection.

    Args:
        api_key: Voyage API key.
        model: Model name (e.g., 'voyage-3.5').

    Returns:
        TestResult with connection status.
    """
    if not api_key:
        return TestResult("Voyage Embedding", Status.FAILED, "Missing API key")

    try:
        import numpy as np
        import voyageai

        client = voyageai.Client(api_key=api_key)
        result = client.embed(
            ["test connection"],
            model=model,
            input_type="document",
        )

        embeddings = np.asarray(result.embeddings, dtype=np.float32)
        dim = embeddings.shape[1]

        if dim == 1024:
            return TestResult(
                "Voyage Embedding",
                Status.SUCCESS,
                f"Connected (model: {model}, dim: {dim})",
            )
        else:
            return TestResult(
                "Voyage Embedding",
                Status.FAILED,
                f"Unexpected embedding dimension: {dim}",
            )

    except voyageai.error.AuthenticationError:
        return TestResult("Voyage Embedding", Status.FAILED, "Invalid API key")
    except voyageai.error.RateLimitError:
        return TestResult("Voyage Embedding", Status.FAILED, "Rate limit exceeded")
    except Exception as e:
        return TestResult("Voyage Embedding", Status.FAILED, f"Error: {str(e)[:100]}")


def test_voyage_rerank(api_key: str, model: str) -> TestResult:
    """Test Voyage rerank API connection.

    Args:
        api_key: Voyage API key.
        model: Model name (e.g., 'rerank-2', 'rerank-2.5').

    Returns:
        TestResult with connection status.
    """
    if not api_key:
        return TestResult("Voyage Rerank", Status.FAILED, "Missing API key")

    try:
        import voyageai

        client = voyageai.Client(api_key=api_key)
        result = client.rerank(
            query="machine learning research",
            documents=["deep learning paper", "cooking recipe"],
            model=model,
            top_k=2,
        )

        if result.results:
            top_score = result.results[0].relevance_score
            return TestResult(
                "Voyage Rerank",
                Status.SUCCESS,
                f"Connected (model: {model}, top_score: {top_score:.3f})",
            )
        else:
            return TestResult("Voyage Rerank", Status.FAILED, "No results returned")

    except voyageai.error.AuthenticationError:
        return TestResult("Voyage Rerank", Status.FAILED, "Invalid API key")
    except voyageai.error.RateLimitError:
        return TestResult("Voyage Rerank", Status.FAILED, "Rate limit exceeded")
    except Exception as e:
        return TestResult("Voyage Rerank", Status.FAILED, f"Error: {str(e)[:100]}")


def test_dashscope_embedding(api_key: str, model: str) -> TestResult:
    """Test DashScope embedding API connection.

    Args:
        api_key: DashScope API key.
        model: Model name (e.g., 'text-embedding-v4').

    Returns:
        TestResult with connection status.
    """
    if not api_key:
        return TestResult("DashScope Embedding", Status.FAILED, "Missing API key")

    try:
        from http import HTTPStatus

        from dashscope import TextEmbedding

        resp = TextEmbedding.call(
            model=model,
            input=["test connection"],
            api_key=api_key,
        )

        if resp.status_code == HTTPStatus.OK:
            embeddings = resp.output["embeddings"]
            dim = len(embeddings[0]["embedding"]) if embeddings else 0
            return TestResult(
                "DashScope Embedding",
                Status.SUCCESS,
                f"Connected (model: {model}, dim: {dim})",
            )
        else:
            return TestResult(
                "DashScope Embedding",
                Status.FAILED,
                f"API error: {resp.code} - {resp.message}",
            )

    except ImportError:
        return TestResult(
            "DashScope Embedding",
            Status.FAILED,
            "dashscope package not installed",
        )
    except Exception as e:
        return TestResult(
            "DashScope Embedding",
            Status.FAILED,
            f"Error: {str(e)[:100]}",
        )


def test_dashscope_rerank(api_key: str, model: str) -> TestResult:
    """Test DashScope rerank API connection.

    Args:
        api_key: DashScope API key.
        model: Model name (e.g., 'qwen3-rerank').

    Returns:
        TestResult with connection status.
    """
    if not api_key:
        return TestResult("DashScope Rerank", Status.FAILED, "Missing API key")

    try:
        from http import HTTPStatus

        from dashscope import TextReRank

        resp = TextReRank.call(
            model=model,
            query="machine learning research",
            documents=["deep learning paper", "cooking recipe"],
            top_n=2,
            return_documents=False,
            api_key=api_key,
        )

        if resp.status_code == HTTPStatus.OK:
            results = resp.output["results"]
            if results:
                top_score = results[0]["relevance_score"]
                return TestResult(
                    "DashScope Rerank",
                    Status.SUCCESS,
                    f"Connected (model: {model}, top_score: {top_score:.3f})",
                )
            else:
                return TestResult("DashScope Rerank", Status.FAILED, "No results returned")
        else:
            return TestResult(
                "DashScope Rerank",
                Status.FAILED,
                f"API error: {resp.code} - {resp.message}",
            )

    except ImportError:
        return TestResult(
            "DashScope Rerank",
            Status.FAILED,
            "dashscope package not installed",
        )
    except Exception as e:
        return TestResult(
            "DashScope Rerank",
            Status.FAILED,
            f"Error: {str(e)[:100]}",
        )


def test_crossref() -> TestResult:
    """Test Crossref API connection."""
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
    import feedparser

    try:
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


def test_deepseek() -> TestResult:
    """Test DeepSeek API connection."""
    api_key = os.environ.get("DEEPSEEK_API_KEY")

    if not api_key:
        return TestResult("DeepSeek", Status.SKIPPED, "DEEPSEEK_API_KEY not set")

    try:
        # Use the models endpoint to test authentication (no tokens consumed)
        resp = requests.get(
            "https://api.deepseek.com/models",
            headers={
                "Authorization": f"Bearer {api_key}",
            },
            timeout=30,
        )

        if resp.status_code == 200:
            data = resp.json()
            model_count = len(data.get("data", []))
            return TestResult("DeepSeek", Status.SUCCESS, f"Connected ({model_count} models available)")
        elif resp.status_code == 401:
            return TestResult("DeepSeek", Status.FAILED, "Invalid API key")
        else:
            return TestResult("DeepSeek", Status.FAILED, f"HTTP {resp.status_code}: {resp.text[:100]}")

    except requests.exceptions.Timeout:
        return TestResult("DeepSeek", Status.FAILED, "Connection timeout")
    except requests.exceptions.RequestException as e:
        return TestResult("DeepSeek", Status.FAILED, f"Connection error: {e}")


def run_tests(settings: Settings | None) -> list[TestResult]:
    """Run all API connection tests based on configuration.

    Args:
        settings: Loaded settings or None for fallback mode.

    Returns:
        List of test results.
    """
    print_section("API Connection Tests")

    results = []
    test_count = 0

    # Always test Zotero
    test_count += 1
    print(f"  [{test_count}] Zotero API      ", end="", flush=True)
    result = test_zotero()
    results.append(result)
    print(format_status(result.status, result.message))

    # Test embedding provider based on config
    if settings is None:
        # Fallback: test Voyage
        test_count += 1
        print(f"  [{test_count}] Voyage Embedding", end="", flush=True)
        api_key = os.environ.get("VOYAGE_API_KEY", "")
        result = test_voyage_embedding(api_key, "voyage-3.5")
        results.append(result)
        print(format_status(result.status, result.message))

    else:
        # Config-aware testing
        if settings.embedding.provider == "voyage":
            test_count += 1
            print(f"  [{test_count}] Voyage Embedding", end="", flush=True)
            api_key = os.environ.get("VOYAGE_API_KEY", "")
            result = test_voyage_embedding(api_key, settings.embedding.model)
            results.append(result)
            print(format_status(result.status, result.message))

            # Test rerank only if interests enabled
            if settings.scoring.interests.enabled:
                test_count += 1
                print(f"  [{test_count}] Voyage Rerank   ", end="", flush=True)
                result = test_voyage_rerank(api_key, settings.scoring.rerank.model)
                results.append(result)
                print(format_status(result.status, result.message))

        elif settings.embedding.provider == "dashscope":
            test_count += 1
            print(f"  [{test_count}] DashScope Embed ", end="", flush=True)
            api_key = os.environ.get("DASHSCOPE_API_KEY", "")
            result = test_dashscope_embedding(api_key, settings.embedding.model)
            results.append(result)
            print(format_status(result.status, result.message))

            # Test rerank only if interests enabled
            if settings.scoring.interests.enabled:
                test_count += 1
                print(f"  [{test_count}] DashScope Rerank", end="", flush=True)
                result = test_dashscope_rerank(api_key, settings.scoring.rerank.model)
                results.append(result)
                print(format_status(result.status, result.message))

        # Test LLM provider only if enabled
        if settings.llm.enabled:
            if settings.llm.provider == "kimi":
                test_count += 1
                print(f"  [{test_count}] Kimi LLM        ", end="", flush=True)
                result = test_kimi()
                results.append(result)
                print(format_status(result.status, result.message))
            elif settings.llm.provider == "openrouter":
                test_count += 1
                print(f"  [{test_count}] OpenRouter LLM  ", end="", flush=True)
                result = test_openrouter()
                results.append(result)
                print(format_status(result.status, result.message))
            elif settings.llm.provider == "deepseek":
                test_count += 1
                print(f"  [{test_count}] DeepSeek LLM    ", end="", flush=True)
                result = test_deepseek()
                results.append(result)
                print(format_status(result.status, result.message))

    # Always test data sources
    test_count += 1
    print(f"  [{test_count}] Crossref        ", end="", flush=True)
    result = test_crossref()
    results.append(result)
    print(format_status(result.status, result.message))

    test_count += 1
    print(f"  [{test_count}] arXiv           ", end="", flush=True)
    result = test_arxiv()
    results.append(result)
    print(format_status(result.status, result.message))

    return results


def print_summary(results: list[TestResult], settings: Settings | None) -> int:
    """Print test summary and return exit code.

    Args:
        results: List of test results.
        settings: Loaded settings or None for fallback mode.

    Returns:
        Exit code (0 = success, 1 = failure).
    """
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
            print(f"  ❌ {r.name}: {r.message}")
        print()

    # Show configuration summary
    if settings:
        print("Configuration Summary:")
        print(f"  Embedding: {settings.embedding.provider} ({settings.embedding.model})")
        llm_status = settings.llm.provider if settings.llm.enabled else "disabled"
        if settings.llm.enabled:
            print(f"  LLM: {llm_status} ({settings.llm.model})")
        else:
            print(f"  LLM: {llm_status}")
        if settings.scoring.interests.enabled:
            print(f"  Rerank: {settings.scoring.rerank.provider} ({settings.scoring.rerank.model})")
        else:
            print("  Rerank: disabled (interests.enabled=false)")
        print()

    return 1 if failed > 0 else 0


def main():
    """Main entry point."""
    print_header()

    # Load configuration
    settings = load_config()

    # Detect required env vars based on config
    env_vars = detect_required_env_vars(settings)

    # Check env vars
    env_status = check_env_vars(env_vars)

    # Check for missing required vars
    missing_required = [name for name, config in env_vars.items() if config["required"] and not env_status.get(name)]

    if missing_required:
        print()
        print("❌ Missing required environment variables:")
        for var in missing_required:
            print(f"   - {var}")
            if env_vars[var].get("reason"):
                print(f"     Reason: {env_vars[var]['reason']}")
        print()
        print("Please set these variables before running the tests.")
        sys.exit(1)

    # Validate provider coupling if config loaded
    if settings and settings.scoring.interests.enabled:
        if settings.scoring.rerank.provider != settings.embedding.provider:
            print()
            print("❌ Configuration Error:")
            print("   When interests.enabled=true, embedding and rerank providers must match")
            print(f"   Current: embedding={settings.embedding.provider}, rerank={settings.scoring.rerank.provider}")
            print()
            sys.exit(1)

    # Run API tests
    results = run_tests(settings)

    # Print summary and exit
    exit_code = print_summary(results, settings)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
