"""Configuration settings models."""

from pathlib import Path
from typing import Dict, List

from pydantic import BaseModel, Field, field_validator

from .loader import _load_yaml


# Zotero Configuration
class ZoteroApiConfig(BaseModel):
    """Zotero API configuration."""

    user_id: str
    api_key: str
    page_size: int = 100
    polite_delay_ms: int = 200


class ZoteroConfig(BaseModel):
    """Zotero connection configuration."""

    mode: str = "api"
    api: ZoteroApiConfig = Field(default_factory=lambda: ZoteroApiConfig(user_id="", api_key=""))

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, value: str) -> str:
        allowed = {"api", "bbt"}
        if value not in allowed:
            raise ValueError(f"Unsupported Zotero mode '{value}'. Allowed: {sorted(allowed)}")
        return value


# Source Configuration
class CrossRefConfig(BaseModel):
    """CrossRef source configuration."""

    enabled: bool = True
    mailto: str = "you@example.com"
    days_back: int = 7
    max_results: int = 500


class ArxivConfig(BaseModel):
    """arXiv source configuration."""

    enabled: bool = True
    categories: List[str] = Field(default_factory=lambda: ["cs.LG"])
    days_back: int = 7
    max_results: int = 500


class ScraperConfig(BaseModel):
    """Abstract scraper configuration with sequential fetching and rule-based extraction."""

    enabled: bool = True
    rate_limit_delay: float = 1.0  # Seconds between requests
    timeout: int = 60000  # Page load timeout in milliseconds
    max_retries: int = 2  # Maximum retry attempts per URL
    max_html_chars: int = 15000  # Max HTML chars to send to LLM
    llm_max_tokens: int = 1024  # Max tokens for LLM response
    llm_temperature: float = 0.1  # LLM temperature for extraction
    use_llm_fallback: bool = True  # Use LLM when rule extraction fails


class SourcesConfig(BaseModel):
    """Data sources configuration."""

    crossref: CrossRefConfig = Field(default_factory=CrossRefConfig)
    arxiv: ArxivConfig = Field(default_factory=ArxivConfig)
    scraper: ScraperConfig = Field(default_factory=ScraperConfig)


# Scoring Configuration
class Thresholds(BaseModel):
    """Score thresholds for labeling."""

    must_read: float = 0.65
    consider: float = 0.45


class InterestsConfig(BaseModel):
    """User research interests configuration."""

    enabled: bool = False
    description: str = ""  # Natural language interest description
    top_k_recall: int = 50  # FAISS recall count, -1 to skip FAISS and use all candidates
    top_k_featured: int = 5  # Final featured papers count


class RerankConfig(BaseModel):
    """Voyage Rerank configuration."""

    enabled: bool = True
    model: str = "rerank-2"


class ScoringConfig(BaseModel):
    """Scoring and ranking configuration."""

    thresholds: Thresholds = Field(default_factory=Thresholds)
    interests: InterestsConfig = Field(default_factory=InterestsConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)


# Embedding Configuration
class EmbeddingConfig(BaseModel):
    """Text embedding configuration."""

    provider: str = "voyage"
    model: str = "voyage-3.5"
    api_key: str = ""
    input_type: str = "document"
    batch_size: int = 128
    candidate_ttl_days: int = 7  # TTL for candidate embedding cache


# LLM Configuration
class LLMRetryConfig(BaseModel):
    """LLM retry configuration."""

    max_attempts: int = 3
    backoff_factor: float = 2.0
    initial_delay: float = 1.0


class LLMSummarizeConfig(BaseModel):
    """LLM summarization settings."""

    top_n: int = 20
    cache_expiry_days: int = 30


class LLMConfig(BaseModel):
    """LLM provider configuration."""

    enabled: bool = True
    provider: str = "openrouter"
    api_key: str = ""
    model: str = "deepseek/deepseek-chat-v3-0324"
    max_tokens: int = 1024
    temperature: float = 0.3
    retry: LLMRetryConfig = Field(default_factory=LLMRetryConfig)
    summarize: LLMSummarizeConfig = Field(default_factory=LLMSummarizeConfig)


# Output Configuration
class RSSConfig(BaseModel):
    """RSS output configuration."""

    title: str = "ZotWatch Feed"
    link: str = "https://example.com"
    description: str = "AI-assisted literature watch"


class HTMLConfig(BaseModel):
    """HTML output configuration."""

    template: str = "report.html"
    include_summaries: bool = True


class OutputConfig(BaseModel):
    """Output generation configuration."""

    rss: RSSConfig = Field(default_factory=RSSConfig)
    html: HTMLConfig = Field(default_factory=HTMLConfig)


# Main Settings
class Settings(BaseModel):
    """Main configuration settings."""

    zotero: ZoteroConfig
    sources: SourcesConfig = Field(default_factory=SourcesConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)


def load_settings(base_dir: Path | str) -> Settings:
    """Load settings from configuration file."""
    base = Path(base_dir)
    config_path = base / "config" / "config.yaml"
    config = _load_yaml(config_path)

    return Settings(
        zotero=ZoteroConfig(**config.get("zotero", {})),
        sources=SourcesConfig(**config.get("sources", {})),
        scoring=ScoringConfig(**config.get("scoring", {})),
        embedding=EmbeddingConfig(**config.get("embedding", {})),
        llm=LLMConfig(**config.get("llm", {})),
        output=OutputConfig(**config.get("output", {})),
    )


__all__ = [
    "Settings",
    "load_settings",
    "ZoteroConfig",
    "ZoteroApiConfig",
    "SourcesConfig",
    "ScraperConfig",
    "ScoringConfig",
    "InterestsConfig",
    "RerankConfig",
    "EmbeddingConfig",
    "LLMConfig",
    "OutputConfig",
]
