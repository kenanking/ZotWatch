"""Configuration settings models."""

from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator

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
    categories: list[str] = Field(default_factory=lambda: ["cs.LG"])
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

    class DynamicConfig(BaseModel):
        """Dynamic percentile-based threshold configuration."""

        must_read_percentile: float = 95.0  # Top 5% are must_read
        consider_percentile: float = 70.0  # 70th-95th percentile are consider
        min_must_read: float = 0.60  # Minimum score for must_read
        min_consider: float = 0.40  # Minimum score for consider

    mode: str = "fixed"  # "fixed" or "dynamic"
    must_read: float = 0.65
    consider: float = 0.45
    dynamic: DynamicConfig = Field(default_factory=DynamicConfig)

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, value: str) -> str:
        allowed = {"fixed", "dynamic"}
        if value not in allowed:
            raise ValueError(f"Unsupported threshold mode '{value}'. Allowed: {sorted(allowed)}")
        return value


class ScoringConfig(BaseModel):
    """Scoring and ranking configuration."""

    class InterestsConfig(BaseModel):
        """User research interests configuration."""

        enabled: bool = False
        description: str = ""  # Natural language interest description
        max_documents: int = 500  # Max documents for FAISS recall (must not exceed rerank API limit)
        top_k_interest: int = 5  # Final interest-based papers count

    class RerankConfig(BaseModel):
        """Rerank configuration (supports Voyage AI and DashScope).

        Note: Rerank is only used when interests.enabled=true.
        Provider must match embedding.provider when interests are enabled.
        Ensure interests.max_documents does not exceed the API limit
        (Voyage: 1000, DashScope: 500).
        """

        provider: str = "voyage"  # "voyage" or "dashscope"
        model: str = "rerank-2"  # Voyage: "rerank-2", DashScope: "qwen3-rerank"

        @field_validator("provider")
        @classmethod
        def validate_provider(cls, value: str) -> str:
            allowed = {"voyage", "dashscope"}
            if value.lower() not in allowed:
                raise ValueError(f"Unsupported rerank provider '{value}'. Allowed: {sorted(allowed)}")
            return value.lower()

    class FusionScoringConfig(BaseModel):
        """Micro/Macro fusion scoring.

        - Micro: recency-weighted k-NN similarity S_micro
        - Macro: cluster-size-weighted similarity S_macro = max_k(sim_k * ln(1 + E_k))
        - Final similarity: similarity = α * S_micro + (1 - α) * S_macro
        """

        micro_weight: float = 0.65  # α: weight for micro-level score
        knn_neighbors: int = 5  # L: neighbor count used for micro-level scoring

    thresholds: Thresholds = Field(default_factory=Thresholds)
    interests: InterestsConfig = Field(default_factory=InterestsConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    fusion: FusionScoringConfig = Field(default_factory=FusionScoringConfig)


# Embedding Configuration
class EmbeddingConfig(BaseModel):
    """Text embedding configuration (supports Voyage AI and DashScope)."""

    provider: str = "voyage"  # "voyage" or "dashscope"
    model: str = "voyage-3.5"  # Voyage: "voyage-3.5", DashScope: "text-embedding-v4"
    api_key: str = ""
    batch_size: int = 128
    candidate_ttl_days: int = 7  # TTL for candidate embedding cache

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, value: str) -> str:
        allowed = {"voyage", "dashscope"}
        if value.lower() not in allowed:
            raise ValueError(f"Unsupported embedding provider '{value}'. Allowed: {sorted(allowed)}")
        return value.lower()

    @property
    def signature(self) -> str:
        """Return embedding provider and model signature (e.g., 'voyage:voyage-3.5')."""
        return f"{self.provider}:{self.model}"


# LLM Configuration
class LLMConfig(BaseModel):
    """LLM provider configuration."""

    class RetryConfig(BaseModel):
        """LLM retry configuration."""

        max_attempts: int = 3
        backoff_factor: float = 2.0
        initial_delay: float = 1.0

    class SummarizeConfig(BaseModel):
        """LLM summarization settings."""

        top_n: int = 20
        cache_expiry_days: int = 30

    class TranslationConfig(BaseModel):
        """Title translation configuration."""

        enabled: bool = False

    enabled: bool = True
    provider: str = "openrouter"
    api_key: str = ""
    model: str = "deepseek/deepseek-chat-v3-0324"
    max_tokens: int = 1024
    temperature: float = 0.3
    retry: RetryConfig = Field(default_factory=RetryConfig)
    summarize: SummarizeConfig = Field(default_factory=SummarizeConfig)
    translation: TranslationConfig = Field(default_factory=TranslationConfig)


# Output Configuration
class OutputConfig(BaseModel):
    """Output generation configuration."""

    class RSSConfig(BaseModel):
        """RSS output configuration."""

        title: str = "ZotWatch Feed"
        link: str = "https://example.com"
        description: str = "AI-assisted literature watch"

    class HTMLConfig(BaseModel):
        """HTML output configuration."""

        template: str = "report.html"
        include_summaries: bool = True

    timezone: str = "UTC"  # IANA timezone name, e.g., "Asia/Shanghai"
    rss: RSSConfig = Field(default_factory=RSSConfig)
    html: HTMLConfig = Field(default_factory=HTMLConfig)


# Profile Configuration


class TemporalConfig(BaseModel):
    """Temporal weighting configuration for time-decay of paper relevance.

    Uses exponential decay: w = exp(-ln(2) / T_half * age_days)
    Papers at halflife_days age have weight = 0.5.
    """

    enabled: bool = True
    halflife_days: float = 180.0  # T_half: papers half as relevant after this many days
    min_weight: float = 0.05  # Floor weight to prevent zero weights for very old papers


class ClusteringConfig(BaseModel):
    """Configuration for profile clustering.

    Uses adaptive Silhouette-based clustering with automatic k selection.
    The optimal cluster count is determined by maximizing Silhouette score
    within the range [2, min(max_clusters, n_samples // 39)].

    K selection uses biased selection: within tolerance of the best score,
    prefer the largest k value for finer-grained research domains. Tolerance
    is expressed as a percentage of the best Silhouette score.
    """

    enabled: bool = True
    max_clusters: int = 35  # Upper limit on cluster count
    min_cluster_size: int = 1  # Minimum papers per valid cluster (1 = allow single-paper clusters)
    biased_k_tolerance_percent: float = 0.10  # Relative tolerance: within (1 - pct) of best Silhouette, select max k

    # Temporal weighting
    temporal: TemporalConfig = Field(default_factory=TemporalConfig)

    # LLM labeling
    generate_labels: bool = True  # Use LLM to generate cluster labels

    # K-means algorithm parameters
    kmeans_iterations: int = 20  # Number of k-means iterations
    subsample_threshold: int = 5000  # Subsample above this for silhouette search
    representative_title_count: int = 5  # Number of representative titles per cluster

    @field_validator("biased_k_tolerance_percent")
    @classmethod
    def validate_biased_k_tolerance_percent(cls, value: float) -> float:
        if not 0 <= value <= 1:
            raise ValueError("biased_k_tolerance_percent must be between 0 and 1 (representing a percentage)")
        return value


class ProfileConfig(BaseModel):
    """Profile analysis configuration."""

    exclude_tags: list[str] = Field(default_factory=list)  # Tags to drop during ingest
    author_min_count: int = 10  # Minimum appearances for "frequent author"
    clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)


# Watch Pipeline Configuration
class WatchPipelineConfig(BaseModel):
    """Watch pipeline configuration.

    Externalizes magic numbers previously hardcoded in cli/main.py.
    """

    recent_days: int = 7  # Filter papers older than this many days
    max_preprint_ratio: float = 0.9  # Maximum ratio of preprints in results
    top_k: int = 20  # Default number of recommendations
    require_abstract: bool = True  # Filter out candidates without abstracts


# Main Settings
class Settings(BaseModel):
    """Main configuration settings."""

    zotero: ZoteroConfig
    sources: SourcesConfig = Field(default_factory=SourcesConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    profile: ProfileConfig = Field(default_factory=ProfileConfig)
    watch: WatchPipelineConfig = Field(default_factory=WatchPipelineConfig)

    @model_validator(mode="after")
    def validate_embedding_rerank_coupling(self) -> "Settings":
        """Ensure embedding and rerank use the same provider when interests are enabled.

        This constraint is only enforced when interests.enabled=true because:
        - The reranker is only used for interest-based recommendations
        - If interests are disabled, rerank configuration is ignored
        - This prevents confusing validation errors for unused configurations
        """
        if self.scoring.interests.enabled:
            if self.scoring.rerank.provider != self.embedding.provider:
                raise ValueError(
                    f"Configuration error: When interests.enabled=true, "
                    f"rerank provider '{self.scoring.rerank.provider}' "
                    f"must match embedding provider '{self.embedding.provider}'. "
                    f"Update config.yaml to use the same provider for both.\n\n"
                    f"Example:\n"
                    f"  embedding:\n"
                    f'    provider: "{self.embedding.provider}"\n'
                    f"  scoring:\n"
                    f"    rerank:\n"
                    f'      provider: "{self.embedding.provider}"\n\n'
                    f"Alternatively, set scoring.interests.enabled=false if you don't need "
                    f"interest-based recommendations."
                )
        return self


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
        profile=ProfileConfig(**config.get("profile", {})),
        watch=WatchPipelineConfig(**config.get("watch", {})),
    )


__all__ = [
    "Settings",
    "load_settings",
    "ZoteroConfig",
    "ZoteroApiConfig",
    "SourcesConfig",
    "CrossRefConfig",
    "ArxivConfig",
    "ScraperConfig",
    "ScoringConfig",
    "Thresholds",
    "EmbeddingConfig",
    "LLMConfig",
    "OutputConfig",
    "ProfileConfig",
    "ClusteringConfig",
    "TemporalConfig",
    "WatchPipelineConfig",
]
