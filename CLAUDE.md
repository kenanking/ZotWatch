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
3. **Candidate Fetch** (`pipeline/fetch.py`): Pulls recent papers from Crossref, arXiv, bioRxiv/medRxiv, OpenAlex
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
│   ├── storage/        # SQLite storage, cache
│   ├── embedding/      # Voyage AI + FAISS
│   └── http/           # HTTP client
├── sources/            # Data sources (arXiv, Crossref, OpenAlex, bioRxiv, Zotero)
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
- `data/cache/candidate_cache.json`: 12-hour cache of fetched candidates
- `data/journal_metrics.csv`: Optional SJR journal metrics for quality scoring

### Configuration Files (config/)

- `config.yaml`: Unified configuration file containing all settings:
  - `zotero`: Zotero API settings (user_id uses `${ZOTERO_USER_ID}` env var expansion)
  - `sources`: Data source toggles and parameters (days_back, categories, max_results)
  - `scoring`: Score weights, thresholds, decay settings, author/venue whitelists
  - `embedding`: Embedding model configuration (provider, model, batch_size)
  - `llm`: LLM configuration for AI summaries (provider, model, retry settings)
  - `output`: RSS and HTML output settings

### Core Components

- `VoyageEmbedder` (`infrastructure/embedding/voyage.py`): Wraps Voyage AI API (voyage-3.5, 1024-dim embeddings)
- `FaissIndex` (`infrastructure/embedding/faiss_index.py`): Manages FAISS index for semantic similarity
- `SQLiteStorage` (`infrastructure/storage/sqlite.py`): SQLite abstraction for items and embeddings
- `Settings` (`config/settings.py`): Pydantic models for configuration with env var expansion
- `OpenRouterClient` (`llm/openrouter.py`): OpenRouter API client for LLM calls
- `LLMSummarizer` (`llm/summarizer.py`): Generates structured paper summaries

## Environment Variables

Required:
- `ZOTERO_API_KEY`: Zotero Web API key
- `ZOTERO_USER_ID`: Zotero user ID
- `VOYAGE_API_KEY`: Voyage AI API key for text embeddings

Optional:
- `OPENROUTER_API_KEY`: OpenRouter API key for AI summaries
- `CROSSREF_MAILTO`: Crossref polite pool email
- `OPENALEX_MAILTO`: OpenAlex polite pool email

## Key Constraints

- Preprint ratio is capped at 30% in final results (`_limit_preprints`)
- Results are filtered to papers within last 7 days (`_filter_recent`)
- Candidate cache expires after 12 hours
- GitHub Actions caches profile artifacts monthly to avoid full rebuilds
- AI summaries require `OPENROUTER_API_KEY` and `llm.enabled: true` in config
