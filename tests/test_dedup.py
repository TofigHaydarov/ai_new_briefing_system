"""Tests for src/core/dedup.py — ArticleDeduplicator.

NOTE: requires the `article.text` -> `article.content` bug in dedup.py to be
fixed first; until then the first test fails with AttributeError.
"""

from datetime import datetime

import pytest

from ai.schemas import Article
from src.core.dedup import ArticleDeduplicator


def make_article(
    title="Sample title",
    url="https://example.com/news/a1",
    source="Sample News",
    content="This is the body text of the article about some topic.",
    published_at=None,
):
    """Small helper to avoid repeating fields in every test."""
    return Article(
        title=title,
        url=url,
        source=source,
        content=content,
        published_at=published_at,
    )


@pytest.fixture
def deduplicator():
    """Fresh instance per test to avoid state leakage."""
    return ArticleDeduplicator(semantic_threshold=0.85)


class TestIsDuplicateUniqueArticles:
    def test_completely_different_articles_all_kept(self, deduplicator):
        a1 = make_article(url="https://a.com/1", content="Football match result today.")
        a2 = make_article(url="https://b.com/2", content="Stock market rallies sharply.")
        a3 = make_article(url="https://c.com/3", content="New vaccine trial shows promise.")

        assert deduplicator.is_duplicate(a1) is False
        assert deduplicator.is_duplicate(a2) is False
        assert deduplicator.is_duplicate(a3) is False
        assert len(deduplicator.accepted_articles) == 3

    def test_first_article_never_a_duplicate(self, deduplicator):
        a1 = make_article()
        assert deduplicator.is_duplicate(a1) is False


class TestUrlDedup:
    def test_exact_same_url_is_duplicate(self, deduplicator):
        a1 = make_article(url="https://example.com/news/a1", content="Text A")
        a2 = make_article(url="https://example.com/news/a1", content="Completely different text B")

        assert deduplicator.is_duplicate(a1) is False
        assert deduplicator.is_duplicate(a2) is True

    def test_url_with_different_tracking_params_is_duplicate(self, deduplicator):
        # url_canonicalize should strip utm_* tracking params
        a1 = make_article(url="https://example.com/news/a1?utm_source=rss", content="Text A")
        a2 = make_article(url="https://example.com/news/a1?utm_source=twitter&fbclid=xyz", content="Text A variant")

        assert deduplicator.is_duplicate(a1) is False
        assert deduplicator.is_duplicate(a2) is True


class TestContentHashDedup:
    def test_identical_content_different_url_is_duplicate(self, deduplicator):
        shared_text = "Breaking news: something important happened today in the world."
        a1 = make_article(url="https://source1.com/x", content=shared_text)
        a2 = make_article(url="https://source2.com/y", content=shared_text)

        assert deduplicator.is_duplicate(a1) is False
        assert deduplicator.is_duplicate(a2) is True

    def test_whitespace_only_difference_still_matches_hash(self, deduplicator):
        # content_hash is whitespace-normalized, so these should match
        a1 = make_article(url="https://source1.com/x", content="Hello   world  today")
        a2 = make_article(url="https://source2.com/y", content="Hello world today")

        assert deduplicator.is_duplicate(a1) is False
        assert deduplicator.is_duplicate(a2) is True


class TestSemanticDedup:
    def test_paraphrased_text_above_threshold_is_duplicate(self, deduplicator):
        a1 = make_article(
            url="https://source1.com/x",
            content="The president announced a new economic policy today during a press conference in the capital.",
        )
        # Near-paraphrase of a1, same event and mostly same wording
        a2 = make_article(
            url="https://source2.com/y",
            content="The president announced a new economic policy today during a press conference held in the capital.",
        )

        assert deduplicator.is_duplicate(a1) is False
        assert deduplicator.is_duplicate(a2) is True

    def test_similar_topic_but_below_threshold_is_not_duplicate(self, deduplicator):
        a1 = make_article(
            url="https://source1.com/x",
            content="The football team won their match yesterday with a strong second half performance.",
        )
        a2 = make_article(
            url="https://source2.com/y",
            content="Scientists discovered a new species of beetle in the Amazon rainforest this week.",
        )

        assert deduplicator.is_duplicate(a1) is False
        assert deduplicator.is_duplicate(a2) is False

    def test_threshold_is_configurable(self):
        content_a = "Local elections results were announced late last night across the region."
        content_b = "Local election results were announced late last night across the whole region."

        strict = ArticleDeduplicator(semantic_threshold=0.95)
        a1 = make_article(url="https://s1.com/a", content=content_a)
        a2 = make_article(url="https://s2.com/b", content=content_b)
        assert strict.is_duplicate(a1) is False
        # At a very high threshold even small differences may count as "different"
        result_strict = strict.is_duplicate(a2)

        lenient = ArticleDeduplicator(semantic_threshold=0.3)
        a3 = make_article(url="https://s1.com/a", content=content_a)
        a4 = make_article(url="https://s2.com/b", content=content_b)
        assert lenient.is_duplicate(a3) is False
        result_lenient = lenient.is_duplicate(a4)

        # A lower threshold should catch at least as many duplicates as strict
        assert result_lenient or not result_strict or result_lenient == result_strict


class TestProcessBatch:
    def test_empty_list_returns_empty_list(self, deduplicator):
        assert deduplicator.process_batch([]) == []

    def test_batch_removes_all_duplicate_types(self, deduplicator):
        shared_text = "Exact same content repeated across two different sources for testing."
        articles = [
            make_article(url="https://a.com/1", content="Unique article number one about space."),
            make_article(url="https://a.com/1?utm_source=rss", content="Different text but same canonical url."),
            make_article(url="https://b.com/2", content=shared_text),
            make_article(url="https://c.com/3", content=shared_text),
            make_article(url="https://d.com/4", content="Another unique article about cooking recipes."),
        ]

        result = deduplicator.process_batch(articles)

        # articles[0] & [1] share a canonical URL -> one dropped
        # articles[2] & [3] share content hash -> one dropped
        # articles[4] is unique -> kept
        assert len(result) == 3

    def test_batch_preserves_order_of_kept_articles(self, deduplicator):
        a1 = make_article(url="https://a.com/1", content="First unique article about music.")
        a2 = make_article(url="https://a.com/1", content="Duplicate url, should be dropped.")
        a3 = make_article(url="https://b.com/2", content="Second unique article about travel.")

        result = deduplicator.process_batch([a1, a2, a3])

        assert result == [a1, a3]


class TestStateIsolation:
    def test_two_separate_instances_do_not_share_state(self):
        dedup1 = ArticleDeduplicator()
        dedup2 = ArticleDeduplicator()

        a1 = make_article(url="https://a.com/1", content="Some article content here.")

        assert dedup1.is_duplicate(a1) is False
        # Separate instance hasn't seen this article -> not a duplicate
        assert dedup2.is_duplicate(a1) is False

    def test_reusing_same_instance_across_batches_keeps_memory(self, deduplicator):
        shared_text = "Repeated content across two separate process_batch calls."
        a1 = make_article(url="https://a.com/1", content=shared_text)
        a2 = make_article(url="https://b.com/2", content=shared_text)

        first_result = deduplicator.process_batch([a1])
        second_result = deduplicator.process_batch([a2])

        assert len(first_result) == 1
        # Same instance reused, so a2 should count as a duplicate of a1
        assert len(second_result) == 0