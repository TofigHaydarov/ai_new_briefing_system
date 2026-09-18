import asyncio
import pytest
from tenacity import wait_none

from src.concurrency import pipeline
from tests.helpers import FakeSession, patch_client_session


@pytest.fixture(autouse=True)
def fast_retry():
    html_wait = pipeline.fetch_html.retry.wait
    rss_wait = pipeline.fetch_rss.retry.wait
    pipeline.fetch_html.retry.wait = wait_none()
    pipeline.fetch_rss.retry.wait = wait_none()
    yield
    pipeline.fetch_html.retry.wait = html_wait
    pipeline.fetch_rss.retry.wait = rss_wait


class TestGracefulDegradation:

    async def test_run_ingestion_continues_after_one_source_fails(self, monkeypatch):
        sources = [
            {"url": "https://example.com/broken", "type": "rss"},
            {"url": "https://example.com/ok", "type": "html"},
        ]
        responses = {
            "https://example.com/broken": (503, ""),
            "https://example.com/ok": (200, "<html><body><p>Good content</p></body></html>"),
        }
        fake_session = FakeSession(responses)
        patch_client_session(monkeypatch, pipeline, fake_session)

        articles = await pipeline.run_ingestion(sources)

        assert len(articles) == 1
        assert articles[0]["url"] == "https://example.com/ok"


class TestRetryBehavior:

    async def test_fetch_rss_retries_transient_error_then_succeeds(self):
        rss_body = """<?xml version="1.0"?>
        <rss><channel><item><title>T</title><link>https://example.com/a</link>
        <description>Body</description></item></channel></rss>"""
        fake_session = FakeSession({"https://example.com/feed": (200, rss_body)}, fail_times=2)

        articles = await pipeline.fetch_rss(fake_session, "https://example.com/feed", asyncio.Semaphore(1))

        assert len(articles) == 1
        assert fake_session.calls.count("https://example.com/feed") == 3

    def test_permanent_error_is_retried_up_to_stop_limit(self):
        # Documents current behavior: a 404 is retried the full 3 times before
        # giving up, since retry_if_exception_type doesn't distinguish status codes.
        # This is not asserting desired behavior, just recording what happens today.
        fake_session = FakeSession({"https://example.com/broken": (404, "")})

        articles = asyncio.run(
            pipeline.safe_fetch(fake_session, "https://example.com/broken", "html", asyncio.Semaphore(1))
        )

        assert articles == []
        assert fake_session.calls.count("https://example.com/broken") == 3


class TestRssParsingEdgeCases:

    async def test_fetch_rss_handles_empty_content_list(self, monkeypatch):
        # Expected to fail today: entry.get('content', [...])[0]['value'] raises
        # IndexError when 'content' key exists but is an empty list.
        class FakeFeed:
            entries = [{
                "content": [],
                "summary": "Fallback summary text.",
                "link": "https://example.com/a",
                "title": "A",
            }]

        monkeypatch.setattr(pipeline.feedparser, "parse", lambda xml: FakeFeed())
        fake_session = FakeSession({"https://example.com/feed": (200, "<rss></rss>")})

        articles = await pipeline.fetch_rss(fake_session, "https://example.com/feed", asyncio.Semaphore(1))

        assert len(articles) == 1
        assert articles[0]["content"] == "Fallback summary text."


class TestValidation:

    async def test_run_ingestion_filters_empty_content_articles(self, monkeypatch):
        # Expected to fail today: no validation step drops articles with empty content.
        sources = [{"url": "https://example.com/empty", "type": "html"}]
        responses = {"https://example.com/empty": (200, "<html><body></body></html>")}
        fake_session = FakeSession(responses)
        patch_client_session(monkeypatch, pipeline, fake_session)

        articles = await pipeline.run_ingestion(sources)

        assert articles == []


class TestConcurrency:

    async def test_run_ingestion_respects_concurrency_limit(self, monkeypatch):
        tracker = {"active": 0, "peak": 0}
        sources = [{"url": f"https://example.com/feed{i}", "type": "html"} for i in range(10)]
        responses = {s["url"]: (200, "<html><body><p>Content</p></body></html>") for s in sources}
        fake_session = FakeSession(responses, tracker=tracker, delay=0.05)
        patch_client_session(monkeypatch, pipeline, fake_session)

        articles = await pipeline.run_ingestion(sources)

        assert len(articles) == 10
        assert tracker["peak"] <= pipeline.CONCURRENCY_LIMIT
        assert tracker["peak"] > 1