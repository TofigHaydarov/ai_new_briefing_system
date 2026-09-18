import pytest
from tenacity import wait_none

from ai.providers.base import ProviderError
from ai.schemas import Article
from src.services import ai_service as ai_service_module
from src.services.ai_service import AIService, AIIntegrationError


def make_article(content="Some content here.", url="https://example.com/1", title="Test"):
    return Article(title=title, source="TechCrunch", content=content, url=url)


@pytest.fixture(autouse=True)
def patch_providers(monkeypatch, fake_llm, fake_embedder):
    monkeypatch.setattr("ai.llm.get_llm", lambda: fake_llm)
    monkeypatch.setattr("ai.embedding.get_embedder", lambda: fake_embedder)


@pytest.fixture(autouse=True)
def clear_ai_cache():
    AIService.safe_embed.cache_clear()
    yield
    AIService.safe_embed.cache_clear()


@pytest.fixture(autouse=True)
def fast_retry():
    summarize_wait = AIService.safe_summarize.retry.wait
    embed_wait = AIService.safe_embed.__wrapped__.retry.wait
    AIService.safe_summarize.retry.wait = wait_none()
    AIService.safe_embed.__wrapped__.retry.wait = wait_none()
    yield
    AIService.safe_summarize.retry.wait = summarize_wait
    AIService.safe_embed.__wrapped__.retry.wait = embed_wait


@pytest.fixture
def spy_summarize(monkeypatch):
    from ai.llm import summarize_and_label as real_summarize_and_label

    calls = {"n": 0}

    def wrapper(article, *, llm=None):
        calls["n"] += 1
        return real_summarize_and_label(article, llm=llm)

    monkeypatch.setattr(ai_service_module, "summarize_and_label", wrapper)
    return calls


class TestSafeSummarizeCaching:
    # Expected to fail until content_hash caching is added to safe_summarize.

    def test_identical_content_does_not_recall_llm(self, spy_summarize):
        article1 = make_article(content="Same body text.", url="https://example.com/1")
        article2 = make_article(content="Same body text.", url="https://example.com/2")

        AIService.safe_summarize(article1)
        AIService.safe_summarize(article2)

        assert spy_summarize["n"] == 1

    def test_different_content_calls_llm_each_time(self, spy_summarize):
        article1 = make_article(content="Body A.", url="https://example.com/1")
        article2 = make_article(content="Body B.", url="https://example.com/2")

        AIService.safe_summarize(article1)
        AIService.safe_summarize(article2)

        assert spy_summarize["n"] == 2


class TestSafeSummarizeRetryBehavior:

    def test_transient_error_recovers_after_retry(self, monkeypatch):
        from ai.llm import summarize_and_label as real_summarize_and_label

        state = {"n": 0}

        def flaky(article, *, llm=None):
            state["n"] += 1
            if state["n"] < 3:
                raise ProviderError("simulated transient failure")
            return real_summarize_and_label(article, llm=llm)

        monkeypatch.setattr(ai_service_module, "summarize_and_label", flaky)

        result = AIService.safe_summarize(make_article())

        assert state["n"] == 3
        assert result.summary

    def test_persistent_error_raises_after_exhausting_retries(self, monkeypatch):
        state = {"n": 0}

        def always_fails(article, *, llm=None):
            state["n"] += 1
            raise ProviderError("simulated persistent failure")

        monkeypatch.setattr(ai_service_module, "summarize_and_label", always_fails)

        with pytest.raises(AIIntegrationError):
            AIService.safe_summarize(make_article())

        assert state["n"] == 3

    def test_empty_content_raises_without_exhausting_retries(self, spy_summarize):
        # Expected to fail until ValueError is excluded from the retry filter;
        # currently retries 3 times before raising AIIntegrationError.
        with pytest.raises(AIIntegrationError):
            AIService.safe_summarize(make_article(content=""))

        assert spy_summarize["n"] == 1


class TestSafeEmbed:

    def test_returns_plain_list_not_ndarray(self):
        # Expected to fail until safe_embed converts the provider's
        # np.ndarray result to a plain list.
        result = AIService.safe_embed("Some text to embed.")

        assert isinstance(result, list)
        assert all(isinstance(x, float) for x in result)

    def test_identical_text_uses_cache(self, fake_embedder, monkeypatch):
        calls = {"n": 0}
        original_embed = fake_embedder.embed

        def counting_embed(text):
            calls["n"] += 1
            return original_embed(text)

        monkeypatch.setattr(fake_embedder, "embed", counting_embed)

        AIService.safe_embed("Repeated text.")
        AIService.safe_embed("Repeated text.")

        assert calls["n"] == 1

    def test_different_text_not_cached_together(self, fake_embedder, monkeypatch):
        calls = {"n": 0}
        original_embed = fake_embedder.embed

        def counting_embed(text):
            calls["n"] += 1
            return original_embed(text)

        monkeypatch.setattr(fake_embedder, "embed", counting_embed)

        AIService.safe_embed("Text A.")
        AIService.safe_embed("Text B.")

        assert calls["n"] == 2

    def test_empty_text_raises_ai_integration_error(self):
        with pytest.raises(AIIntegrationError):
            AIService.safe_embed("")