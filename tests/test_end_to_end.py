
from __future__ import annotations

from pathlib import Path

import pytest
import tenacity

pytestmark = pytest.mark.asyncio  # requires pytest-asyncio (or asyncio_mode=auto in pytest.ini)


@pytest.fixture(autouse=True)
def no_retry_delay():
    """Same rationale as in test_services.py: AIService's retry decorators
    use wait_exponential, which would make this test genuinely sleep for
    seconds if a retry is ever triggered. Zero it out for the test run.
    """
    from src.services.ai_service import AIService

    original_summarize_wait = AIService.safe_summarize.retry.wait
    original_embed_wait = AIService.safe_embed.__wrapped__.retry.wait

    AIService.safe_summarize.retry.wait = tenacity.wait_none()
    AIService.safe_embed.__wrapped__.retry.wait = tenacity.wait_none()

    yield

    AIService.safe_summarize.retry.wait = original_summarize_wait
    AIService.safe_embed.__wrapped__.retry.wait = original_embed_wait


@pytest.fixture(autouse=True)
def clear_ai_cache():
    """Same rationale as in test_services.py: safe_embed's lru_cache
    persists across tests unless cleared."""
    from src.services.ai_service import AIService

    AIService.safe_embed.cache_clear()
    yield
    AIService.safe_embed.cache_clear()


async def test_daily_digest_happy_path(
    monkeypatch,
    fake_source_client,
    fake_repository,
    sample_user,
    digests_dir: Path,
):
    """
    End-to-end, everything faked (no network, no Postgres, no real LLM):

      1. FetchService pulls 3 sources through FakeSourceClient — one RSS
         feed, its near-duplicate mirror, and one scraped HTML page.
      2. Dedup collapses the mirror pair down to 2 unique stories.
      3. AIService labels each survivor — summarize_and_label is
         monkeypatched, since AIService calls it as a module-level
         function rather than accepting it via constructor injection.
      4. Topic/source filtering keeps only what the user wants.
      5. The digest is built and written through FakeRepository, landing
         as a real file in digests_dir.
    """
    # --- under test: adjust these imports/constructors to your real classes ---
    from src.services import ai_service as ai_service_module
    from src.services.ai_service import AIService
    from src.services.fetch_service import FetchService
    from src.core.dedup import Deduplicator
    from src.core.digest_builder import build_digest
    from ai.schemas import LabeledSummary, Topic, Sentiment

    fetch_service = FetchService(client=fake_source_client)
    deduplicator = Deduplicator(threshold=0.7)

    # Deterministic fake labeling: "processor" articles -> Tech, else Science.
    def fake_summarize_and_label(article):
        topic = Topic.TECH if "processor" in article.title.lower() else Topic.SCIENCE
        return LabeledSummary(
            summary=f"Summary of: {article.title}",
            topic=topic,
            sentiment=Sentiment.NEUTRAL,
        )

    monkeypatch.setattr(ai_service_module, "summarize_and_label", fake_summarize_and_label)

    sources = [
        "https://example.com/rss/feed1",
        "https://example.com/rss/feed2-mirror",
        "https://example.com/scrape/page1",
    ]

    # 1. Fetch + parse (real parsing logic runs against faked raw bodies)
    articles = await fetch_service.fetch_all(sources)
    assert len(articles) == 3

    # 2. Dedup: the mirrored article should collapse into the original
    unique_articles = deduplicator.deduplicate(articles)
    assert len(unique_articles) == 2

    # 3. Label each survivor via the (monkeypatched) AIService
    labeled = [(a, AIService.safe_summarize(a)) for a in unique_articles]
    assert all(ls.topic for _, ls in labeled)

    # 4. Filter to the user's preferred topics / excluded sources
    #    (Topic is a str Enum, so `Topic.TECH in ["Tech", ...]` works fine.)
    relevant = [
        (a, ls)
        for a, ls in labeled
        if ls.topic in sample_user.preferred_topics
        and a.source not in sample_user.excluded_sources
    ]
    assert len(relevant) == 2

    # 5. Build the digest and persist it through the fake repository
    markdown = build_digest(sample_user, relevant, date="2026-05-06")
    out_path = await fake_repository.save_digest(
        user_name=sample_user.name, date="2026-05-06", markdown=markdown
    )

    assert out_path.exists()
    assert out_path == digests_dir / "2026-05-06-khagani.md"
    assert out_path in fake_repository.saved_digests

    content = out_path.read_text()
    assert "khagani" in content
    assert "Acme unveils new AI processor" in content
    assert "Astronomers detect unusual signal" in content
    # the deduped mirror story should not appear a second time
    assert content.count("Acme unveils new AI processor") == 1


async def test_broken_source_is_skipped_not_fatal(fake_source_client):
    """
    One source (mocked 503 via FakeSourceClient) must be logged and
    skipped, not crash the whole run — this is explicitly called out
    as a required behavior in TOPIC.md.
    """
    from src.services.fetch_service import FetchService

    fetch_service = FetchService(client=fake_source_client)

    sources = [
        "https://example.com/rss/feed1",
        "https://example.com/broken/feed3",
        "https://example.com/scrape/page1",
    ]

    articles = await fetch_service.fetch_all(sources)

    # 2 healthy sources should still produce articles; no exception raised.
    assert len(articles) == 2
    # the client should have attempted all 3, including the broken one
    assert "https://example.com/broken/feed3" in fake_source_client.calls