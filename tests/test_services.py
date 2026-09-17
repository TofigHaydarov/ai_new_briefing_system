
from __future__ import annotations

import tenacity
import pytest

from src.services import ai_service as ai_service_module
from src.services.ai_service import AIService, AIIntegrationError
from ai.schemas import Article, LabeledSummary, Topic, Sentiment


@pytest.fixture(autouse=True)
def no_retry_delay():
    """
    Reconfigure both retry-decorated methods to not actually sleep
    between attempts, for every test in this file. Restores the
    original wait strategy afterwards so we don't affect other test
    files or production behavior.
    """
    original_summarize_wait = AIService.safe_summarize.retry.wait
    # lru_cache exposes the underlying (retry-decorated) function via __wrapped__
    original_embed_wait = AIService.safe_embed.__wrapped__.retry.wait

    AIService.safe_summarize.retry.wait = tenacity.wait_none()
    AIService.safe_embed.__wrapped__.retry.wait = tenacity.wait_none()

    yield

    AIService.safe_summarize.retry.wait = original_summarize_wait
    AIService.safe_embed.__wrapped__.retry.wait = original_embed_wait


@pytest.fixture(autouse=True)
def clear_ai_cache():
    """Clear safe_embed's lru_cache before every test so results can't
    leak from one test into another."""
    AIService.safe_embed.cache_clear()
    yield
    AIService.safe_embed.cache_clear()


@pytest.fixture
def sample_article() -> Article:
    return Article(
        title="Acme unveils new processor",
        url="https://example.com/news/acme-chip",
        source="Example News",
        content="Acme Corp announced a new processor today aimed at AI workloads.",
    )


# --------------------------------------------------------------------------
# safe_summarize — happy path
# --------------------------------------------------------------------------


def test_safe_summarize_returns_result_on_first_success(monkeypatch, sample_article):
    expected = LabeledSummary(summary="A new chip.", topic=Topic.TECH, sentiment=Sentiment.NEUTRAL)
    monkeypatch.setattr(ai_service_module, "summarize_and_label", lambda article: expected)

    result = AIService.safe_summarize(sample_article)

    assert result is expected


def test_safe_summarize_retries_then_succeeds(monkeypatch, sample_article):
    """First two calls fail transiently, third succeeds — retry should
    paper over this and return the successful result."""
    expected = LabeledSummary(summary="A new chip.", topic=Topic.TECH, sentiment=Sentiment.NEUTRAL)
    calls = {"count": 0}

    def flaky_summarize(article):
        calls["count"] += 1
        if calls["count"] < 3:
            raise ConnectionError("transient network error")
        return expected

    monkeypatch.setattr(ai_service_module, "summarize_and_label", flaky_summarize)

    result = AIService.safe_summarize(sample_article)

    assert result is expected
    assert calls["count"] == 3


# --------------------------------------------------------------------------
# safe_summarize — error path
# --------------------------------------------------------------------------


def test_safe_summarize_raises_ai_integration_error_after_retries_exhausted(
    monkeypatch, sample_article, caplog
):
    """If the underlying call always fails, safe_summarize should give up
    after 3 attempts and raise AIIntegrationError — not the raw exception,
    and not hang forever."""

    def always_fails(article):
        raise ConnectionError("provider is down")

    monkeypatch.setattr(ai_service_module, "summarize_and_label", always_fails)

    with pytest.raises(AIIntegrationError):
        AIService.safe_summarize(sample_article)

    assert any(record.levelname == "ERROR" for record in caplog.records)


# --------------------------------------------------------------------------
# safe_embed — happy path + caching behavior
# --------------------------------------------------------------------------


def test_safe_embed_returns_vector(monkeypatch):
    monkeypatch.setattr(ai_service_module, "embed", lambda text: [0.1, 0.2, 0.3])

    result = AIService.safe_embed("some article text")

    assert result == [0.1, 0.2, 0.3]


def test_safe_embed_caches_identical_text(monkeypatch):
    """Calling safe_embed twice with the same text should only call the
    underlying embed() once — this is the caching behavior TOPIC.md asks
    for (applied here to embeddings; summaries don't have it yet, see
    review notes)."""
    call_count = {"n": 0}

    def counting_embed(text):
        call_count["n"] += 1
        return [0.1, 0.2, 0.3]

    monkeypatch.setattr(ai_service_module, "embed", counting_embed)

    AIService.safe_embed("repeated text")
    AIService.safe_embed("repeated text")
    AIService.safe_embed("repeated text")

    assert call_count["n"] == 1, "expected embed() to be called only once due to lru_cache"


def test_safe_embed_does_not_cache_across_different_text(monkeypatch):
    call_count = {"n": 0}

    def counting_embed(text):
        call_count["n"] += 1
        return [float(len(text))]

    monkeypatch.setattr(ai_service_module, "embed", counting_embed)

    AIService.safe_embed("text one")
    AIService.safe_embed("a totally different text")

    assert call_count["n"] == 2


# --------------------------------------------------------------------------
# safe_embed — error path
# --------------------------------------------------------------------------


def test_safe_embed_raises_ai_integration_error_after_retries_exhausted(monkeypatch, caplog):
    def always_fails(text):
        raise ConnectionError("embedding provider is down")

    monkeypatch.setattr(ai_service_module, "embed", always_fails)

    with pytest.raises(AIIntegrationError):
        AIService.safe_embed("text that will fail")

    assert any(record.levelname == "ERROR" for record in caplog.records)