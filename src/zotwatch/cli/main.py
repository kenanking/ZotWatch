"""Main CLI entry point using Click."""

import logging
from datetime import timedelta
from pathlib import Path

import click
from dotenv import load_dotenv

from zotwatch import __version__
from zotwatch.config import Settings, load_settings
from zotwatch.config.settings import LLMConfig
from zotwatch.core.models import InterestWork, RankedWork, ResearcherProfile
from zotwatch.infrastructure.embedding import EmbeddingCache, VoyageEmbedding, VoyageReranker
from zotwatch.infrastructure.enrichment.cache import MetadataCache
from zotwatch.infrastructure.storage import ProfileStorage
from zotwatch.llm import (
    InterestRefiner,
    KimiClient,
    LibraryAnalyzer,
    OpenRouterClient,
    OverallSummarizer,
    PaperSummarizer,
)
from zotwatch.llm.base import BaseLLMProvider
from zotwatch.output import render_html, write_rss
from zotwatch.output.push import ZoteroPusher
from zotwatch.pipeline import DedupeEngine, InterestRanker, ProfileBuilder, ProfileRanker, ProfileStatsExtractor
from zotwatch.pipeline.enrich import AbstractEnricher
from zotwatch.pipeline.fetch import CandidateFetcher
from zotwatch.sources.zotero import ZoteroIngestor
from zotwatch.utils.datetime import utc_today_start
from zotwatch.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def _get_base_dir() -> Path:
    """Get base directory from current working directory or git root."""
    cwd = Path.cwd()
    # Check for config/config.yaml to identify project root
    if (cwd / "config" / "config.yaml").exists():
        return cwd
    # Try parent directories
    for parent in cwd.parents:
        if (parent / "config" / "config.yaml").exists():
            return parent
    return cwd


def _get_embedding_cache(base_dir: Path) -> EmbeddingCache:
    """Get or create embedding cache for the given base directory."""
    cache_db_path = base_dir / "data" / "embeddings.sqlite"
    return EmbeddingCache(cache_db_path)


@click.group()
@click.option("--base-dir", type=click.Path(exists=True), default=None, help="Repository base directory")
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging")
@click.version_option(version=__version__, prog_name="zotwatch")
@click.pass_context
def cli(ctx: click.Context, base_dir: str | None, verbose: bool) -> None:
    """ZotWatch - Personalized academic paper recommendations."""
    ctx.ensure_object(dict)

    base = Path(base_dir) if base_dir else _get_base_dir()
    load_dotenv(base / ".env")
    setup_logging(verbose=verbose)

    ctx.obj["base_dir"] = base
    ctx.obj["verbose"] = verbose

    # Load settings lazily (some commands may not need them)
    ctx.obj["_settings"] = None
    ctx.obj["_embedding_cache"] = None


def _get_settings(ctx: click.Context) -> Settings:
    """Get or load settings."""
    if ctx.obj["_settings"] is None:
        ctx.obj["_settings"] = load_settings(ctx.obj["base_dir"])
    return ctx.obj["_settings"]


def _get_cache(ctx: click.Context) -> EmbeddingCache:
    """Get or create embedding cache."""
    if ctx.obj["_embedding_cache"] is None:
        ctx.obj["_embedding_cache"] = _get_embedding_cache(ctx.obj["base_dir"])
    return ctx.obj["_embedding_cache"]


def _create_llm_client(config: LLMConfig) -> BaseLLMProvider:
    """Create LLM client based on provider configuration."""
    provider = config.provider.lower()
    if provider == "kimi":
        return KimiClient.from_config(config)
    elif provider == "openrouter":
        return OpenRouterClient.from_config(config)
    else:
        raise ValueError(f"Unknown LLM provider: {config.provider}. Supported providers: 'openrouter', 'kimi'")


def _profile_exists(base_dir: Path) -> bool:
    """Check if profile artifacts exist."""
    faiss_path = base_dir / "data" / "faiss.index"
    sqlite_path = base_dir / "data" / "profile.sqlite"
    return faiss_path.exists() and sqlite_path.exists()


def _build_profile(
    base_dir: Path,
    settings: Settings,
    embedding_cache: EmbeddingCache,
    full: bool = True,
) -> None:
    """Build user profile from Zotero library."""
    storage = ProfileStorage(base_dir / "data" / "profile.sqlite")
    storage.initialize()

    # Ingest from Zotero
    click.echo("Ingesting items from Zotero...")
    ingestor = ZoteroIngestor(storage, settings)
    stats = ingestor.run(full=full)
    click.echo(f"  Fetched: {stats.fetched}, Updated: {stats.updated}, Removed: {stats.removed}")

    # Count items
    total_items = storage.count_items()
    if total_items == 0:
        raise click.ClickException(
            "No items found in your Zotero library. Please add some papers to Zotero before running ZotWatch."
        )

    click.echo(f"Building profile from {total_items} items...")

    # Build profile with unified cache
    vectorizer = VoyageEmbedding(
        model_name=settings.embedding.model,
        api_key=settings.embedding.api_key,
        input_type=settings.embedding.input_type,
        batch_size=settings.embedding.batch_size,
    )
    builder = ProfileBuilder(
        base_dir,
        storage,
        settings,
        vectorizer=vectorizer,
        embedding_cache=embedding_cache,
    )
    artifacts = builder.run(full=full)

    click.echo("Profile built successfully:")
    click.echo(f"  SQLite: {artifacts.sqlite_path}")
    click.echo(f"  FAISS: {artifacts.faiss_path}")


@cli.command()
@click.option("--full", is_flag=True, help="Full rebuild of profile (recompute all embeddings)")
@click.pass_context
def profile(ctx: click.Context, full: bool) -> None:
    """Build or update user research profile.

    By default, uses cached embeddings where available.
    Use --full to invalidate cache and recompute all embeddings.
    """
    settings = _get_settings(ctx)
    base_dir = ctx.obj["base_dir"]
    storage = ProfileStorage(base_dir / "data" / "profile.sqlite")
    storage.initialize()
    embedding_cache = _get_cache(ctx)

    # Ingest from Zotero
    click.echo("Ingesting items from Zotero...")
    ingestor = ZoteroIngestor(storage, settings)
    stats = ingestor.run(full=full)
    click.echo(f"  Fetched: {stats.fetched}, Updated: {stats.updated}, Removed: {stats.removed}")

    # Count items
    total_items = storage.count_items()
    cached_profile = embedding_cache.count(source_type="profile", model=settings.embedding.model)

    if full:
        click.echo("Building profile (full rebuild)...")
    elif cached_profile < total_items:
        click.echo(f"Building profile ({total_items - cached_profile}/{total_items} items need embedding)...")
    else:
        click.echo(f"Building profile (all {total_items} embeddings cached)...")

    # Build profile with unified cache
    vectorizer = VoyageEmbedding(
        model_name=settings.embedding.model,
        api_key=settings.embedding.api_key,
        input_type=settings.embedding.input_type,
        batch_size=settings.embedding.batch_size,
    )
    builder = ProfileBuilder(
        base_dir,
        storage,
        settings,
        vectorizer=vectorizer,
        embedding_cache=embedding_cache,
    )
    artifacts = builder.run(full=full)

    click.echo("Profile built successfully:")
    click.echo(f"  SQLite: {artifacts.sqlite_path}")
    click.echo(f"  FAISS: {artifacts.faiss_path}")


@cli.command()
@click.option("--rss", is_flag=True, help="Generate RSS feed only")
@click.option("--report", is_flag=True, help="Generate HTML report only")
@click.option("--top", type=int, default=20, help="Number of top results (default: 20)")
@click.option("--push", is_flag=True, help="Push recommendations to Zotero")
@click.pass_context
def watch(
    ctx: click.Context,
    rss: bool,
    report: bool,
    top: int,
    push: bool,
) -> None:
    """Fetch, score, and output paper recommendations.

    By default, generates RSS feed and HTML report with AI summaries.
    Use --rss or --report to generate specific output formats.
    """
    # If none specified, generate all
    if not rss and not report:
        rss = True
        report = True
    settings = _get_settings(ctx)
    base_dir = ctx.obj["base_dir"]
    storage = ProfileStorage(base_dir / "data" / "profile.sqlite")
    storage.initialize()
    embedding_cache = _get_cache(ctx)

    # Check if profile exists, build if missing
    if not _profile_exists(base_dir):
        click.echo("No profile found. Building profile from your Zotero library...")
        click.echo("(This is a one-time setup that may take a few minutes)\n")
        _build_profile(base_dir, settings, embedding_cache, full=True)
        click.echo()  # Add blank line after profile build

    # Incremental ingest
    click.echo("Syncing with Zotero...")
    ingestor = ZoteroIngestor(storage, settings)
    ingestor.run(full=False)

    # Generate researcher profile analysis
    researcher_profile: ResearcherProfile | None = None
    if settings.llm.enabled:
        click.echo("Analyzing researcher profile...")
        all_items = storage.get_all_items()

        if all_items:
            # Check cache first
            stats_extractor = ProfileStatsExtractor()
            current_hash = stats_extractor.compute_library_hash(all_items)
            cached_profile = storage.get_profile_analysis(current_hash)

            if cached_profile:
                click.echo("  Using cached profile analysis")
                researcher_profile = cached_profile
            else:
                # Extract statistics
                click.echo("  Extracting library statistics...")
                researcher_profile = stats_extractor.extract_all(
                    all_items,
                    exclude_keywords=settings.profile.exclude_keywords,
                    author_min_count=settings.profile.author_min_count,
                )

                # Use LLM for domain classification and insights
                try:
                    llm_client = _create_llm_client(settings.llm)
                    analyzer = LibraryAnalyzer(llm_client, model=settings.llm.model)

                    click.echo("  Classifying research domains...")
                    researcher_profile.domains = analyzer.classify_domains(all_items)

                    click.echo("  Generating AI insights...")
                    researcher_profile.insights = analyzer.generate_insights(researcher_profile)
                    researcher_profile.model_used = settings.llm.model

                    # Cache the result
                    storage.save_profile_analysis(researcher_profile)
                    click.echo("  Profile analysis complete and cached")
                except Exception as e:
                    logger.warning("Failed to generate profile insights: %s", e)
                    click.echo(f"  AI insights skipped (error: {e})")
        else:
            click.echo("  No items in library, skipping profile analysis")

    # Fetch candidates
    click.echo("Fetching candidates from sources...")
    fetcher = CandidateFetcher(settings, base_dir)
    candidates = fetcher.fetch_all()
    click.echo(f"  Found {len(candidates)} candidates")

    # Enrich missing abstracts
    if settings.sources.scraper.enabled:
        click.echo("Enriching missing abstracts via scraper...")
        # Create LLM client for scraper fallback if enabled
        llm_for_enrichment = None
        if settings.llm.enabled and settings.sources.scraper.use_llm_fallback:
            try:
                llm_for_enrichment = _create_llm_client(settings.llm)
                logger.debug("LLM client created for abstract enrichment scraper")
            except Exception as e:
                logger.warning("Failed to create LLM client for enrichment: %s", e)
        enricher = AbstractEnricher(settings, base_dir, llm=llm_for_enrichment)
        candidates, enrich_stats = enricher.enrich(candidates)
        # Display before/after comparison
        click.echo(
            f"  Before: {enrich_stats.with_abstract}/{enrich_stats.total_candidates} "
            f"({enrich_stats.original_rate:.1f}%) have abstracts"
        )
        click.echo(
            f"  After:  {enrich_stats.with_abstract + enrich_stats.enriched}/{enrich_stats.total_candidates} "
            f"({enrich_stats.final_rate:.1f}%) have abstracts"
        )
        if enrich_stats.enriched > 0 or enrich_stats.failed > 0:
            click.echo(
                f"  Result: +{enrich_stats.enriched} enriched (cache: {enrich_stats.cache_hits}, "
                f"scraper: {enrich_stats.scraper_fetched}, failed: {enrich_stats.failed})"
            )

    # Deduplicate
    dedupe = DedupeEngine(storage)
    filtered = dedupe.filter(candidates)
    click.echo(f"  After dedup: {len(filtered)} candidates")

    # Filter out candidates without abstracts (required for accurate similarity scoring)
    before_abstract_filter = len(filtered)
    filtered = [c for c in filtered if c.abstract]
    removed_no_abstract = before_abstract_filter - len(filtered)
    if removed_no_abstract > 0:
        click.echo(f"  Filtered: {removed_no_abstract} candidates without abstracts removed")
        logger.info("Removed %d candidates without abstracts", removed_no_abstract)

    # Interest-based paper selection (optional)
    interest_works: list[InterestWork] = []
    interests_config = settings.scoring.interests

    if interests_config.enabled and interests_config.description.strip():
        click.echo("Selecting interest-based papers...")
        try:
            # Create LLM client for interest refinement
            llm_client = _create_llm_client(settings.llm)
            refiner = InterestRefiner(llm_client, model=settings.llm.model)

            # Create reranker
            reranker = VoyageReranker(
                api_key=settings.embedding.api_key,
                model=settings.scoring.rerank.model,
            )

            # Create vectorizer for feature selector
            vectorizer = VoyageEmbedding(
                model_name=settings.embedding.model,
                api_key=settings.embedding.api_key,
                input_type=settings.embedding.input_type,
                batch_size=settings.embedding.batch_size,
            )

            # Select featured papers
            selector = InterestRanker(
                settings=settings,
                vectorizer=vectorizer,
                reranker=reranker,
                interest_refiner=refiner,
                base_dir=base_dir,
            )
            interest_works = selector.select(filtered)
            click.echo(f"  Selected {len(interest_works)} interest papers")

        except Exception as e:
            logger.warning("Interest selection failed: %s", e)
            click.echo(f"  Interest selection skipped (error: {e})")

    # Rank (with unified embedding cache)
    click.echo("Ranking candidates...")
    ranker = ProfileRanker(base_dir, settings, embedding_cache=embedding_cache)
    ranked = ranker.rank(filtered)

    # Cleanup expired cache entries
    removed = embedding_cache.cleanup_expired()
    if removed > 0:
        click.echo(f"  Cleaned up {removed} expired embedding cache entries")

    # Cleanup expired metadata cache entries
    metadata_cache = MetadataCache(base_dir / "data" / "metadata.sqlite")
    removed_meta = metadata_cache.cleanup_expired()
    if removed_meta > 0:
        click.echo(f"  Cleaned up {removed_meta} expired metadata cache entries")
    metadata_cache.close()

    # Filter
    ranked = _filter_recent(ranked, days=7)
    ranked = _limit_preprints(ranked, max_ratio=0.9)

    if top and len(ranked) > top:
        ranked = ranked[:top]

    if not ranked:
        click.echo("No recommendations found")
        if rss:
            write_rss([], base_dir / "reports" / "feed.xml")
        if report:
            render_html([], base_dir / "reports" / "report-empty.html", timezone_name=settings.output.timezone)
        return

    click.echo(f"\nTop {min(10, len(ranked))} recommendations:")
    for idx, work in enumerate(ranked[:10], start=1):
        click.echo(f"  {idx:02d} | {work.score:.3f} | {work.label} | {work.title[:60]}...")

    # Generate AI summaries for all ranked papers and interest papers
    overall_summaries = {}
    if settings.llm.enabled:
        click.echo(f"\nGenerating AI summaries for {len(ranked)} similarity papers...")
        llm_client = _create_llm_client(settings.llm)
        summarizer = PaperSummarizer(llm_client, storage, model=settings.llm.model)
        summaries = summarizer.summarize_batch(ranked)
        click.echo(f"  Generated {len(summaries)} summaries")

        # Attach summaries to ranked works
        summary_map = {s.paper_id: s for s in summaries}
        for work in ranked:
            if work.identifier in summary_map:
                work.summary = summary_map[work.identifier]

        # Generate summaries for interest works
        if interest_works:
            click.echo(f"Generating AI summaries for {len(interest_works)} interest papers...")
            interest_summaries = summarizer.summarize_batch(interest_works)
            click.echo(f"  Generated {len(interest_summaries)} interest summaries")

            # Attach summaries to interest works
            interest_summary_map = {s.paper_id: s for s in interest_summaries}
            for work in interest_works:
                if work.identifier in interest_summary_map:
                    work.summary = interest_summary_map[work.identifier]

        # Generate overall summaries for report header
        click.echo("Generating overall summaries for report...")
        overall_summarizer = OverallSummarizer(llm_client, model=settings.llm.model)

        if interest_works:
            click.echo("  Summarizing interest papers...")
            overall_summaries["interest"] = overall_summarizer.summarize_section(interest_works, "interest")

        if ranked:
            click.echo("  Summarizing similarity papers...")
            overall_summaries["similarity"] = overall_summarizer.summarize_section(ranked, "similarity")
    else:
        click.echo("\nAI summaries disabled (llm.enabled=false in config)")

    # Generate outputs
    if rss:
        rss_path = base_dir / "reports" / "feed.xml"
        write_rss(
            ranked,
            rss_path,
            title=settings.output.rss.title,
            link=settings.output.rss.link,
            description=settings.output.rss.description,
        )
        click.echo(f"RSS feed: {rss_path}")

    if report:
        # Use UTC date for report filename
        report_name = f"report-{utc_today_start():%Y%m%d}.html"
        report_path = base_dir / "reports" / report_name
        template_dir = base_dir / "templates"
        render_html(
            ranked,
            report_path,
            template_dir=template_dir if template_dir.exists() else None,
            timezone_name=settings.output.timezone,
            interest_works=interest_works if interest_works else None,
            overall_summaries=overall_summaries if overall_summaries else None,
            researcher_profile=researcher_profile,
        )
        click.echo(f"HTML report: {report_path}")

    if push:
        pusher = ZoteroPusher(settings)
        pusher.push(ranked)
        click.echo("Pushed recommendations to Zotero")


def _filter_recent(ranked: list[RankedWork], *, days: int) -> list[RankedWork]:
    """Filter to recent papers only."""
    if days <= 0:
        return ranked
    cutoff = utc_today_start() - timedelta(days=days)
    kept = [work for work in ranked if work.published and work.published >= cutoff]
    removed = len(ranked) - len(kept)
    if removed > 0:
        logger.info("Dropped %d items older than %d days", removed, days)
    return kept


def _limit_preprints(ranked: list[RankedWork], *, max_ratio: float) -> list[RankedWork]:
    """Limit preprints to a maximum ratio."""
    if not ranked or max_ratio <= 0:
        return ranked
    preprint_sources = {"arxiv", "biorxiv", "medrxiv"}
    filtered: list[RankedWork] = []
    preprint_count = 0
    for work in ranked:
        source = work.source.lower()
        proposed_total = len(filtered) + 1
        if source in preprint_sources:
            proposed_preprints = preprint_count + 1
            if (proposed_preprints / proposed_total) > max_ratio:
                continue
            preprint_count = proposed_preprints
        filtered.append(work)
    removed = len(ranked) - len(filtered)
    if removed > 0:
        logger.info("Preprint cap removed %d items to respect %.0f%% limit", removed, max_ratio * 100)
    return filtered


if __name__ == "__main__":
    cli()
