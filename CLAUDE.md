# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ZotWatch is a personalized academic paper recommendation system that builds a research interest profile from a user's Zotero library and continuously monitors academic sources for relevant new publications. It supports AI-powered summaries, incremental embedding computation, and runs daily via GitHub Actions to output RSS/HTML feeds.

## Commands

```bash
# Install dependencies
uv sync

# Build/rebuild user profile from Zotero library (full rebuild)
uv run zotwatch profile --full

# Incremental profile update (only new/changed items)
uv run zotwatch profile

# Daily watch: fetch candidates, score, and generate RSS + HTML + AI summaries
uv run zotwatch watch

# Only generate RSS feed
uv run zotwatch watch --rss

# Only generate HTML report
uv run zotwatch watch --report

# Custom recommendation count
uv run zotwatch watch --top 50

# Push top recommendations back to Zotero
uv run zotwatch watch --push
```

## Architecture

### Pipeline Flow

1. **Ingest** (`pipeline/ingest.py`): Fetches items from Zotero Web API, stores in SQLite
2. **Profile Build** (`pipeline/profile.py`): Vectorizes library items using Voyage AI API (voyage-3.5), builds FAISS index, extracts top authors/venues
3. **Candidate Fetch** (`pipeline/fetch.py`): Pulls recent papers from Crossref and arXiv
4. **Deduplication** (`pipeline/dedupe.py`): Filters out papers already in the user's library
5. **Scoring** (`pipeline/score.py`): Ranks candidates using weighted combination of similarity, recency, citations, journal quality, and whitelist bonuses
6. **Summarization** (`llm/summarizer.py`): Generates AI summaries via OpenRouter API
7. **Output** (`output/rss.py`, `output/html.py`): Generates RSS feed and/or HTML report

### Directory Structure

```
src/zotwatch/
├── core/               # Core models, protocols, exceptions
├── config/             # Configuration loading and settings
├── infrastructure/     # External service integrations
│   ├── storage/        # SQLite storage
│   ├── embedding/      # Voyage AI + FAISS
│   └── http/           # HTTP client
├── sources/            # Data sources (arXiv, Crossref, Zotero)
├── llm/                # LLM integration (OpenRouter, summarizer)
├── pipeline/           # Processing pipeline (ingest, profile, fetch, dedupe, score)
├── output/             # Output generation (RSS, HTML, push to Zotero)
├── cli/                # Click CLI
└── utils/              # Utilities (logging, datetime, hashing, text)
```

### Key Data Artifacts

- `data/profile.sqlite`: SQLite database storing Zotero items and embeddings
- `data/faiss.index`: FAISS vector index for similarity search
- `data/profile.json`: Profile summary with top authors, venues, and centroid vector
- `data/embeddings.sqlite`: Embedding cache for reusing computed vectors

### Configuration Files (config/)

- `config.yaml`: Unified configuration file containing all settings:
  - `zotero`: Zotero API settings (user_id uses `${ZOTERO_USER_ID}` env var expansion)
  - `sources`: Data source toggles and parameters (days_back, categories, max_results)
  - `scoring`: Score weights, thresholds, decay settings, author/venue whitelists
  - `embedding`: Embedding model configuration (provider, model, batch_size)
  - `llm`: LLM configuration for AI summaries (provider, model, retry settings)
  - `output`: RSS and HTML output settings
  - `watch`: Watch pipeline settings (recent_days, preprint ratio, top_k)

For detailed configuration guides (embedding provider switching, LLM provider switching, threshold modes, etc.), see the "配置指南" section in README.md.

### Configuration Options

#### Dynamic Thresholds (`scoring.thresholds`)

Controls how papers are labeled as `must_read`, `consider`, or `ignore`:

- `mode`: Threshold computation mode
  - `"fixed"`: Use static threshold values (default behavior)
  - `"dynamic"`: Compute thresholds from score distribution per batch
- `must_read`: Fixed threshold for must_read label (default: 0.75)
- `consider`: Fixed threshold for consider label (default: 0.55)
- `dynamic`: Settings for dynamic mode
  - `must_read_percentile`: Top N% marked as must_read (default: 95, meaning top 5%)
  - `consider_percentile`: Percentile for consider threshold (default: 70)
  - `min_must_read`: Minimum score for must_read even in dynamic mode (default: 0.60)
  - `min_consider`: Minimum score for consider even in dynamic mode (default: 0.40)

#### Watch Pipeline (`watch`)

Controls the watch command behavior:

- `recent_days`: Filter papers older than N days (default: 7)
- `max_preprint_ratio`: Maximum ratio of preprints in final results (default: 0.9)
- `top_k`: Default number of recommendations (default: 20)
- `require_abstract`: Filter out candidates without abstracts (default: true)

### Switching Embedding Providers

ZotWatch supports two embedding providers:
- **Voyage AI**: High-quality embeddings, recommended for English papers
- **DashScope** (Alibaba Cloud): Alternative provider with Chinese language support

**Important:** When `scoring.interests.enabled=true`, both `embedding.provider` and `scoring.rerank.provider` MUST use the same provider.

#### Voyage AI (Default)

```yaml
# config/config.yaml
embedding:
  provider: "voyage"
  model: "voyage-3.5"
  api_key: "${VOYAGE_API_KEY}"
  batch_size: 128

scoring:
  rerank:
    provider: "voyage"  # Must match embedding.provider
    model: "rerank-2.5"
```

Environment variables:
```bash
# .env
VOYAGE_API_KEY=your_voyage_api_key_here
```

#### DashScope (Alibaba Cloud)

```yaml
# config/config.yaml
embedding:
  provider: "dashscope"
  model: "text-embedding-v4"
  api_key: "${DASHSCOPE_API_KEY}"
  batch_size: 25  # DashScope uses smaller batches

scoring:
  rerank:
    provider: "dashscope"  # Must match embedding.provider
    model: "qwen3-rerank"
```

Environment variables:
```bash
# .env
DASHSCOPE_API_KEY=your_dashscope_api_key_here
```

**After switching providers**, rebuild your profile:
```bash
uv run zotwatch profile --full
```

This is necessary because embeddings from different providers are not compatible.

### Switching LLM Providers

ZotWatch supports three LLM providers for AI summaries:
- **OpenRouter**: Access to multiple LLM providers (Claude, GPT-4, etc.)
- **Kimi** (Moonshot AI): Chinese LLM with thinking model support
- **DeepSeek**: Cost-effective LLM with reasoning model support

#### OpenRouter

```yaml
# config/config.yaml
llm:
  enabled: true
  provider: "openrouter"
  api_key: "${OPENROUTER_API_KEY}"
  model: "anthropic/claude-3.5-sonnet"
```

Environment variable:
```bash
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

#### Kimi (Moonshot AI)

```yaml
# config/config.yaml
llm:
  enabled: true
  provider: "kimi"
  api_key: "${MOONSHOT_API_KEY}"
  model: "kimi-k2-turbo-preview"  # or "kimi-k2-thinking-turbo" for thinking mode
```

Environment variable:
```bash
MOONSHOT_API_KEY=your_moonshot_api_key_here
```

#### DeepSeek

```yaml
# config/config.yaml
llm:
  enabled: true
  provider: "deepseek"
  api_key: "${DEEPSEEK_API_KEY}"
  model: "deepseek-chat"  # or "deepseek-reasoner" for reasoning mode
  max_tokens: 4096
  temperature: 0.3
```

Environment variable:
```bash
DEEPSEEK_API_KEY=your_deepseek_api_key_here
```

**Note on Reasoning Models:**
- DeepSeek's `deepseek-reasoner` model uses chain-of-thought reasoning
- Temperature parameter is automatically disabled for reasoning models
- Reasoning models may require longer timeouts (120s default)

### Core Components

- `VoyageEmbedder` (`infrastructure/embedding/voyage.py`): Wraps Voyage AI API (voyage-3.5, 1024-dim embeddings)
- `FaissIndex` (`infrastructure/embedding/faiss_index.py`): Manages FAISS index for semantic similarity
- `SQLiteStorage` (`infrastructure/storage/sqlite.py`): SQLite abstraction for items and embeddings
- `Settings` (`config/settings.py`): Pydantic models for configuration with env var expansion
- `OpenRouterClient` (`llm/openrouter.py`): OpenRouter API client for LLM calls
- `KimiClient` (`llm/kimi.py`): Kimi (Moonshot AI) API client for LLM calls
- `DeepSeekClient` (`llm/deepseek.py`): DeepSeek API client for LLM calls
- `LLMSummarizer` (`llm/summarizer.py`): Generates structured paper summaries

## Environment Variables

Required:
- `ZOTERO_API_KEY`: Zotero Web API key
- `ZOTERO_USER_ID`: Zotero user ID
- `VOYAGE_API_KEY` or `DASHSCOPE_API_KEY`: Embedding provider API key (depending on `embedding.provider` in config.yaml)
- `MOONSHOT_API_KEY`, `OPENROUTER_API_KEY`, or `DEEPSEEK_API_KEY`: LLM provider API key (at least one required, depending on `llm.provider` in config.yaml)

Optional:
- `CROSSREF_MAILTO`: Crossref polite pool email

## Key Constraints

- Preprint ratio is configurable via `watch.max_preprint_ratio` (default: 0.9)
- Recent paper filter is configurable via `watch.recent_days` (default: 7 days)
- GitHub Actions caches profile artifacts monthly to avoid full rebuilds
- AI summaries require LLM API key (`MOONSHOT_API_KEY`, `OPENROUTER_API_KEY`, or `DEEPSEEK_API_KEY`) and `llm.enabled: true` in config
- Embedding and rerank providers must use the same provider when interests.enabled=true (both Voyage or both DashScope)
- When writing code, please use English for all comments
- Use Python 3.10+ type annotation syntax: `list[X]`, `dict[K, V]`, `X | None` instead of `List`, `Dict`, `Optional` from typing module

## Commit Message Conventions

This project follows Conventional Commits style for commit messages:

- **Type**: lowercase verb category (feat, fix, refactor, chore, docs, test, perf, etc.)
  - Optional scope in parentheses to indicate submodule: `feat(profile): ...`
- **Subject**: concise imperative sentence, lowercase first letter, no ending period
  - Keep the entire message within ~72 characters
- **Structure**: `type(scope?): subject`
  - Examples: `feat: add temporal clustering` or `fix(api): handle timeouts`
- **Body** (optional): for complex changes, add a blank line after subject then write:
  - Bullet points explaining the changes
  - Breaking changes (if any)
  - Related issues
- **Voice**: use present tense/imperative mood
  - Focus on "what" and "why" rather than implementation details
  - Avoid lengthy descriptions and ending punctuation

## Troubleshooting

### Provider Mismatch Error

**Error:**
```
Configuration error: When interests.enabled=true, rerank provider 'dashscope' must match embedding provider 'voyage'
```

**Cause:** This error occurs when `scoring.interests.enabled=true` but the providers don't match.

**Solution 1:** If you need interest-based recommendations, update `config.yaml` so both providers match:
```yaml
embedding:
  provider: "voyage"  # or "dashscope"

scoring:
  rerank:
    provider: "voyage"  # Must be same as embedding.provider
```

**Solution 2:** If you don't need interest-based recommendations, disable interests:
```yaml
scoring:
  interests:
    enabled: false
```

### Missing API Key

**Error:**
```
DashScope API key is required. Set DASHSCOPE_API_KEY environment variable.
```

**Solution:** Add the API key to your `.env` file:
```bash
DASHSCOPE_API_KEY=your_key_here
```

### Incompatible Embeddings After Provider Switch

**Symptom:** Errors when running `zotwatch watch` after changing providers

**Solution:** Rebuild your profile with the new provider:
```bash
uv run zotwatch profile --full
```

This regenerates all embeddings using the new provider.
