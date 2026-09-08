
from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.asyncio  # requires pytest-asyncio (or asyncio_mode=auto in pytest.ini)


async def test_daily_digest_happy_path(
    fake_llm,
    fake_embedder,
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
      3. AIService labels each survivor via FakeLLM (fixed JSON payload).
      4. Topic/source filtering keeps only what the user wants.
      5. The digest is built and written through FakeRepository, landing
         as a real file in digests_dir.
    """
    # --- under test: adjust these imports/constructors to your real classes ---
    from src.services.ai_service import AIService
    from src.services.fetch_service import FetchService
    from src.core.dedup import Deduplicator
    from src.core.digest_builder import build_digest

    fetch_service = FetchService(client=fake_source_client)
    ai_service = AIService(llm=fake_llm, embedder=fake_embedder)
    deduplicator = Deduplicator(threshold=0.7)

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

    # 3. Label each survivor via the injected FakeLLM
    labeled = [(a, ai_service.summarize_and_label(a)) for a in unique_articles]
    assert fake_llm.calls, "expected the (fake) LLM to actually be invoked"
    assert all(ls.topic for _, ls in labeled)

    # 4. Filter to the user's preferred topics / excluded sources
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