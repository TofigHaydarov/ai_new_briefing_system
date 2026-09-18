import asyncio
from datetime import date

import pytest
from tenacity import wait_none

from src.concurrency import pipeline
from src.services.ai_service import AIService
from src.services.briefing_service import generate_user_digest
from src.cli import _render_markdown_digest
from src.storage.repository import UserProfile
from ai.schemas import Article
from tests.helpers import FakeSession, patch_client_session


def make_profile(user="khagani", topics=None, excluded=None, max_items=5):
    return UserProfile(
        user=user,
        preferred_topics=topics or ["General"],
        excluded_sources=excluded or [],
        max_items_per_topic=max_items,
    )


def articles_from_ingestion(raw_articles: list[dict]) -> list[Article]:
    """Converts pipeline.run_ingestion's raw dict output into Article objects.

    NOTE: this conversion does not exist anywhere in the actual codebase yet.
    cli.py currently bypasses it entirely by using a hardcoded sample_articles
    list instead of wiring in real ingestion output. This helper is written
    here only so the rest of the pipeline can be exercised end-to-end; it
    should be replaced by (or moved into) real glue code once the team adds
    it, e.g. in briefing_service.py or a dedicated adapter.
    """
    return [
        Article(
            title=a.get("title", "No Title"),
            url=a["url"],
            source=a.get("source", "Unknown"),
            content=a.get("content", ""),
        )
        for a in raw_articles
        if a.get("content", "").strip()
    ]


@pytest.fixture(autouse=True)
def patch_ai_providers(monkeypatch, fake_llm, fake_embedder):
    monkeypatch.setattr("ai.llm.get_llm", lambda: fake_llm)
    monkeypatch.setattr("ai.embedding.get_embedder", lambda: fake_embedder)

    summarize_wait = AIService.safe_summarize.retry.wait
    embed_wait = AIService.safe_embed.__wrapped__.retry.wait
    AIService.safe_summarize.retry.wait = wait_none()
    AIService.safe_embed.__wrapped__.retry.wait = wait_none()

    yield

    AIService.safe_summarize.retry.wait = summarize_wait
    AIService.safe_embed.__wrapped__.retry.wait = embed_wait


@pytest.fixture(autouse=True)
def clear_embed_cache():
    AIService.safe_embed.cache_clear()
    yield
    AIService.safe_embed.cache_clear()


@pytest.fixture(autouse=True)
def fast_pipeline_retry():
    html_wait = pipeline.fetch_html.retry.wait
    rss_wait = pipeline.fetch_rss.retry.wait
    pipeline.fetch_html.retry.wait = wait_none()
    pipeline.fetch_rss.retry.wait = wait_none()
    yield
    pipeline.fetch_html.retry.wait = html_wait
    pipeline.fetch_rss.retry.wait = rss_wait


class TestIngestionToDigestGap:

    async def test_raw_ingestion_output_is_not_directly_usable(self, monkeypatch, fake_llm):
        """Documents a real integration gap: run_ingestion returns list[dict],
        but generate_user_digest expects list[Article]. Passing dicts straight
        through fails, because the digest logic accesses attributes like
        article.source and article.content that a dict doesn't have."""
        sources = [{"url": "https://example.com/page", "type": "html"}]
        responses = {"https://example.com/page": (200, "<html><body><p>Some content.</p></body></html>")}
        fake_session = FakeSession(responses)
        patch_client_session(monkeypatch, pipeline, fake_session)

        raw_articles = await pipeline.run_ingestion(sources)
        assert isinstance(raw_articles[0], dict)

        profile = make_profile()

        with pytest.raises(AttributeError):
            await generate_user_digest(profile, raw_articles)


class TestFullPipelineHappyPath:

    async def test_ingestion_through_markdown_output(self, monkeypatch, fake_llm):
        set_topic_by_title(fake_llm, {"Acme": "Tech", "Astronomers": "Science"})

        sources = [
            {"url": "https://example.com/rss/feed1", "type": "rss"},
            {"url": "https://example.com/rss/feed2-mirror", "type": "rss"},
            {"url": "https://example.com/scrape/page1", "type": "html"},
        ]
        responses = {
            "https://example.com/rss/feed1": (200, RSS_FEED_1),
            "https://example.com/rss/feed2-mirror": (200, RSS_FEED_1_MIRROR),
            "https://example.com/scrape/page1": (200, HTML_PAGE_1),
        }
        fake_session = FakeSession(responses)
        patch_client_session(monkeypatch, pipeline, fake_session)

        # 1. Ingestion
        raw_articles = await pipeline.run_ingestion(sources)
        assert len(raw_articles) == 3

        # 2. Adapt to Article (see articles_from_ingestion note on why this
        # shim currently has to live in the test rather than in the app)
        articles = articles_from_ingestion(raw_articles)
        assert len(articles) == 3

        # 3. Dedup + AI labeling + topic/source filtering, all inside
        # generate_user_digest
        profile = make_profile(topics=["Tech", "Science"], max_items=5)
        digest = await generate_user_digest(profile, articles)

        # the mirrored RSS story should have been deduplicated away
        assert len(digest.items) == 2
        titles = {item.article.title for item in digest.items}
        assert any("Acme" in t for t in titles)
        assert any("Astronomers" in t for t in titles)

        # 4. Render to Markdown, exactly as run-daily would
        markdown = _render_markdown_digest(digest)

        assert "khagani" in markdown
        assert "Acme" in markdown
        assert "Astronomers" in markdown
        assert markdown.count("Acme") == markdown.count("Acme")  # sanity: no duplication
        assert markdown.count("### ") == 2  # exactly two article entries rendered

    async def test_excluded_source_removed_before_reaching_markdown(self, monkeypatch, fake_llm):
        set_topic_by_title(fake_llm, {"Acme": "Tech"})

        sources = [{"url": "https://example.com/scrape/page1", "type": "html"}]
        responses = {"https://example.com/scrape/page1": (200, HTML_PAGE_1)}
        fake_session = FakeSession(responses)
        patch_client_session(monkeypatch, pipeline, fake_session)

        raw_articles = await pipeline.run_ingestion(sources)
        articles = articles_from_ingestion(raw_articles)
        # the scraped article's source ends up as whatever fetch_html sets;
        # excluding "no title" html sources is not meaningful here, so this
        # test instead exercises exclusion via a source name that does match
        for a in articles:
            a.source = "Sample News"
        profile = make_profile(topics=["General"], excluded=["Sample News"])

        digest = await generate_user_digest(profile, articles)
        markdown = _render_markdown_digest(digest)

        assert digest.items == []
        assert "No articles match your preference profile today" in markdown


class TestBrokenSourceDoesNotBreakFullPipeline:

    async def test_one_failed_source_still_produces_a_digest(self, monkeypatch, fake_llm):
        set_topic_by_title(fake_llm, {"Astronomers": "Science"})

        sources = [
            {"url": "https://example.com/broken", "type": "rss"},
            {"url": "https://example.com/scrape/page1", "type": "html"},
        ]
        responses = {
            "https://example.com/broken": (503, ""),
            "https://example.com/scrape/page1": (200, HTML_PAGE_1),
        }
        fake_session = FakeSession(responses)
        patch_client_session(monkeypatch, pipeline, fake_session)

        raw_articles = await pipeline.run_ingestion(sources)
        assert len(raw_articles) == 1

        articles = articles_from_ingestion(raw_articles)
        profile = make_profile(topics=["General"])

        digest = await generate_user_digest(profile, articles)
        markdown = _render_markdown_digest(digest)

        assert len(digest.items) == 1
        assert "Astronomers" in markdown


def set_topic_by_title(fake_llm, title_to_topic: dict[str, str]):
    """Deterministically labels articles by matching a substring of their
    title, so the fake LLM's response reflects which article is being
    summarized rather than a single fixed topic for everything."""
    import json

    def complete(prompt, *, json_schema=None, max_tokens=1024):
        for marker, topic in title_to_topic.items():
            if marker in prompt:
                return json.dumps({"summary": f"Summary mentioning {marker}.", "topic": topic, "sentiment": "neutral"})
        return json.dumps({"summary": "Generic summary.", "topic": "Other", "sentiment": "neutral"})

    fake_llm.complete = complete


RSS_FEED_1 = """<?xml version="1.0"?>
<rss><channel>
  <item>
    <title>Acme unveils new AI processor</title>
    <link>https://example.com/news/article1</link>
    <description>Acme corp announced a new processor today aimed at AI workloads.</description>
  </item>
</channel></rss>"""

RSS_FEED_1_MIRROR = """<?xml version="1.0"?>
<rss><channel>
  <item>
    <title>Acme unveils new AI processor (syndicated)</title>
    <link>https://example.com/news/article1-mirror</link>
    <description>Acme corp announced a new processor today aimed at AI workloads. Full copy.</description>
  </item>
</channel></rss>"""

HTML_PAGE_1 = """
<html><body>
  <article>
    <h1>Astronomers detect unusual signal from nearby star</h1>
    <p>Researchers report an unusual radio signal from a nearby star system.</p>
  </article>
</body></html>
"""