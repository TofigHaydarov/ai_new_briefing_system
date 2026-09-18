import pytest
from tenacity import wait_none

from src.concurrency import benchmark, pipeline
from tests.test_helpers import FakeSession, patch_client_session


@pytest.fixture(autouse=True)
def fast_retry():
    html_wait = pipeline.fetch_html.retry.wait
    pipeline.fetch_html.retry.wait = wait_none()
    yield
    pipeline.fetch_html.retry.wait = html_wait


class TestRunSequential:

    async def test_never_overlaps_requests(self, monkeypatch):
        tracker = {"active": 0, "peak": 0}
        sources = [{"url": f"https://example.com/s{i}", "type": "html"} for i in range(5)]
        responses = {s["url"]: (200, "<html><body><p>Content</p></body></html>") for s in sources}
        fake_session = FakeSession(responses, tracker=tracker, delay=0.02)
        patch_client_session(monkeypatch, benchmark, fake_session)

        articles = await benchmark.run_sequential(sources)

        assert len(articles) == 5
        assert tracker["peak"] == 1

    async def test_returns_all_articles_in_source_order_when_no_failures(self, monkeypatch):
        sources = [
            {"url": "https://example.com/a", "type": "html"},
            {"url": "https://example.com/b", "type": "html"},
        ]
        responses = {
            "https://example.com/a": (200, "<html><body><p>A content</p></body></html>"),
            "https://example.com/b": (200, "<html><body><p>B content</p></body></html>"),
        }
        fake_session = FakeSession(responses)
        patch_client_session(monkeypatch, benchmark, fake_session)

        articles = await benchmark.run_sequential(sources)

        assert [a["url"] for a in articles] == ["https://example.com/a", "https://example.com/b"]