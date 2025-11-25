"""Main CLI entry point using Click."""

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import click
from dotenv import load_dotenv

from zotwatch import __version__
from zotwatch.config import Settings, load_settings
from zotwatch.core.models import RankedWork
from zotwatch.infrastructure.embedding import EmbeddingCache, VoyageEmbedding
from zotwatch.infrastructure.storage import ProfileStorage
from zotwatch.llm import OpenRouterClient, PaperSummarizer
from zotwatch.output import render_html, write_rss
from zotwatch.output.push import ZoteroPusher
from zotwatch.pipeline import DedupeEngine, ProfileBuilder, WorkRanker
from zotwatch.pipeline.fetch import CandidateFetcher
from zotwatch.sources.zotero import ZoteroIngestor
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
def cli(ctx: click.Context, base_dir: Optional[str], verbose: bool) -> None:
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
    click.echo(f"  JSON: {artifacts.profile_json_path}")


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

    By default, generates both RSS feed and HTML report with AI summaries.
    Use --rss or --report to generate only one output format.
    """
    # If neither specified, generate both
    if not rss and not report:
        rss = True
        report = True
    settings = _get_settings(ctx)
    base_dir = ctx.obj["base_dir"]
    storage = ProfileStorage(base_dir / "data" / "profile.sqlite")
    storage.initialize()
    embedding_cache = _get_cache(ctx)

    # Incremental ingest
    click.echo("Syncing with Zotero...")
    ingestor = ZoteroIngestor(storage, settings)
    ingestor.run(full=False)

    # Fetch candidates
    click.echo("Fetching candidates from sources...")
    fetcher = CandidateFetcher(settings, base_dir)
    candidates = fetcher.fetch_all()
    click.echo(f"  Found {len(candidates)} candidates")

    # Deduplicate
    dedupe = DedupeEngine(storage)
    filtered = dedupe.filter(candidates)
    click.echo(f"  After dedup: {len(filtered)} candidates")

    # Rank (with unified embedding cache)
    click.echo("Ranking candidates...")
    ranker = WorkRanker(base_dir, settings, embedding_cache=embedding_cache)
    ranked = ranker.rank(filtered)

    # Cleanup expired embedding cache entries
    removed = embedding_cache.cleanup_expired()
    if removed > 0:
        click.echo(f"  Cleaned up {removed} expired embedding cache entries")

    # Filter
    ranked = _filter_recent(ranked, days=7)
    ranked = _limit_preprints(ranked, max_ratio=1.0)

    if top and len(ranked) > top:
        ranked = ranked[:top]

    if not ranked:
        click.echo("No recommendations found")
        if rss:
            write_rss([], base_dir / "reports" / "feed.xml")
        if report:
            render_html([], base_dir / "reports" / "report-empty.html")
        return

    click.echo(f"\nTop {min(10, len(ranked))} recommendations:")
    for idx, work in enumerate(ranked[:10], start=1):
        click.echo(f"  {idx:02d} | {work.score:.3f} | {work.label} | {work.title[:60]}...")

    # Generate AI summaries for all ranked papers
    if settings.llm.enabled:
        click.echo(f"\nGenerating AI summaries for {len(ranked)} papers...")
        llm_client = OpenRouterClient.from_config(settings.llm)
        summarizer = PaperSummarizer(llm_client, storage, model=settings.llm.model)
        summaries = summarizer.summarize_batch(ranked)
        click.echo(f"  Generated {len(summaries)} summaries")

        # Attach summaries to ranked works
        summary_map = {s.paper_id: s for s in summaries}
        for work in ranked:
            if work.identifier in summary_map:
                work.summary = summary_map[work.identifier]
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
        report_name = "report.html"
        if ranked and ranked[0].published:
            report_name = f"report-{ranked[0].published:%Y%m%d}.html"
        report_path = base_dir / "reports" / report_name
        template_dir = base_dir / "templates"
        render_html(
            ranked,
            report_path,
            template_dir=template_dir if template_dir.exists() else None,
        )
        click.echo(f"HTML report: {report_path}")

    if push:
        pusher = ZoteroPusher(settings)
        pusher.push(ranked)
        click.echo("Pushed recommendations to Zotero")


@cli.command()
@click.option("--top", type=int, default=20, help="Number of papers to summarize")
@click.option("--force", is_flag=True, help="Regenerate existing summaries")
@click.option("--model", type=str, help="Override LLM model")
@click.pass_context
def summarize(ctx: click.Context, top: int, force: bool, model: Optional[str]) -> None:
    """Generate AI summaries for recent recommendations."""
    settings = _get_settings(ctx)
    base_dir = ctx.obj["base_dir"]

    if not settings.llm.enabled:
        click.echo("LLM is disabled in configuration")
        return

    storage = ProfileStorage(base_dir / "data" / "profile.sqlite")
    storage.initialize()
    embedding_cache = _get_cache(ctx)

    # Load recent ranked works from cache
    from zotwatch.infrastructure.storage.cache import FileCache

    cache_path = base_dir / "data" / "cache" / "candidate_cache.json"
    cache = FileCache(cache_path)
    result = cache.load()

    if not result:
        click.echo("No cached candidates found. Run 'zotwatch watch' first.")
        return

    _, candidates = result

    # Re-rank to get scores (with unified embedding cache)
    click.echo("Re-ranking candidates...")
    ranker = WorkRanker(base_dir, settings, embedding_cache=embedding_cache)

    dedupe = DedupeEngine(storage)
    filtered = dedupe.filter(candidates)
    ranked = ranker.rank(filtered)

    if not ranked:
        click.echo("No papers to summarize")
        return

    # Generate summaries
    click.echo(f"Generating summaries for top {top} papers...")
    llm_client = OpenRouterClient.from_config(settings.llm)
    use_model = model or settings.llm.model
    summarizer = PaperSummarizer(llm_client, storage, model=use_model)

    summaries = summarizer.summarize_batch(ranked[:top], force=force)
    click.echo(f"Generated {len(summaries)} summaries using {use_model}")


def _filter_recent(ranked: List[RankedWork], *, days: int) -> List[RankedWork]:
    """Filter to recent papers only."""
    if days <= 0:
        return ranked
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    kept = [work for work in ranked if work.published and work.published >= cutoff]
    removed = len(ranked) - len(kept)
    if removed > 0:
        logger.info("Dropped %d items older than %d days", removed, days)
    return kept


def _limit_preprints(ranked: List[RankedWork], *, max_ratio: float) -> List[RankedWork]:
    """Limit preprints to a maximum ratio."""
    if not ranked or max_ratio <= 0:
        return ranked
    preprint_sources = {"arxiv", "biorxiv", "medrxiv"}
    filtered: List[RankedWork] = []
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
