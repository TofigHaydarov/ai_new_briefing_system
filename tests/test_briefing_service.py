import pytest
from datetime import datetime
from types import MethodType

from ai.schemas import Article
from src.storage.repository import UserProfile
from src.services.briefing_service import generate_user_digest


def make_article(title="Test Article", source="TechCrunch", content="Some content here.", url="https://example.com/1"):
    return Article(title=title, source=source, content=content, url=url)


def make_profile(user="testuser", topics=None, excluded=None, max_items=2):
    return UserProfile(
        user=user,
        preferred_topics=topics or ["General"],
        excluded_sources=excluded or [],
        max_items_per_topic=max_items,
    )


def set_payload(fake_llm, topic="Tech", sentiment="neutral", summary="A summary."):
    fake_llm.payload = {"summary": summary, "topic": topic, "sentiment": sentiment}


def set_sequence(fake_llm, topics: list[str]):
    responses = iter(topics)

    def complete(self, prompt, *, json_schema=None, max_tokens=1024):
        self.calls.append(prompt)
        topic = next(responses)
        import json
        return json.dumps({"summary": "A summary.", "topic": topic, "sentiment": "neutral"})

    fake_llm.complete = MethodType(complete, fake_llm)


def fail_on_call(fake_llm, call_index: int):
    original = fake_llm.complete
    counter = {"n": 0}

    def complete(prompt, *, json_schema=None, max_tokens=1024):
        counter["n"] += 1
        if counter["n"] == call_index:
            raise RuntimeError("Simulated LLM failure")
        return original(prompt, json_schema=json_schema, max_tokens=max_tokens)

    fake_llm.complete = complete


@pytest.fixture(autouse=True)
def patch_llm(monkeypatch, fake_llm):
    monkeypatch.setattr("ai.llm.get_llm", lambda: fake_llm)
    return fake_llm


class TestTopicFiltering:

    async def test_general_accepts_any_topic(self, fake_llm):
        set_payload(fake_llm, topic="Tech")
        profile = make_profile(topics=["General"])

        digest = await generate_user_digest(profile, [make_article()])

        assert len(digest.items) == 1

    async def test_non_matching_topic_excluded(self, fake_llm):
        set_payload(fake_llm, topic="Sports")
        profile = make_profile(topics=["Tech"])

        digest = await generate_user_digest(profile, [make_article()])

        assert digest.items == []

    async def test_matching_topic_included(self, fake_llm):
        set_payload(fake_llm, topic="Tech")
        profile = make_profile(topics=["Tech"])

        digest = await generate_user_digest(profile, [make_article()])

        assert len(digest.items) == 1
        assert digest.items[0].labeled.topic.value == "Tech"


class TestExcludedSources:

    async def test_excluded_source_is_filtered(self, fake_llm):
        set_payload(fake_llm)
        profile = make_profile(topics=["General"], excluded=["TechCrunch"])

        digest = await generate_user_digest(profile, [make_article(source="TechCrunch")])

        assert digest.items == []

    async def test_excluded_source_case_insensitive(self, fake_llm):
        set_payload(fake_llm)
        profile = make_profile(topics=["General"], excluded=["techcrunch"])

        digest = await generate_user_digest(profile, [make_article(source="TechCrunch")])

        assert digest.items == []

    async def test_non_excluded_source_passes(self, fake_llm):
        set_payload(fake_llm)
        profile = make_profile(topics=["General"], excluded=["BBC"])

        digest = await generate_user_digest(profile, [make_article(source="TechCrunch")])

        assert len(digest.items) == 1


class TestMaxItemsPerTopic:
    # Expected to fail until the topic_count/continue bug in briefing_service.py is fixed.

    async def test_limit_is_enforced(self, fake_llm):
        set_payload(fake_llm, topic="Tech")
        profile = make_profile(topics=["General"], max_items=2)
        articles = [make_article(title=f"Article {i}", url=f"https://example.com/{i}") for i in range(5)]

        digest = await generate_user_digest(profile, articles)

        tech_items = [item for item in digest.items if item.labeled.topic.value == "Tech"]
        assert len(tech_items) <= 2

    async def test_limit_is_per_topic_not_global(self, fake_llm):
        profile = make_profile(topics=["General"], max_items=2)
        articles = [
            make_article(title=f"T{i}", url=f"https://example.com/t{i}") for i in range(3)
        ] + [
            make_article(title=f"S{i}", url=f"https://example.com/s{i}") for i in range(3)
        ]
        set_sequence(fake_llm, ["Tech", "Tech", "Tech", "Sports", "Sports", "Sports"])

        digest = await generate_user_digest(profile, articles)

        tech_count = sum(1 for item in digest.items if item.labeled.topic.value == "Tech")
        sports_count = sum(1 for item in digest.items if item.labeled.topic.value == "Sports")
        assert tech_count <= 2
        assert sports_count <= 2

    async def test_under_limit_all_included(self, fake_llm):
        set_payload(fake_llm, topic="Tech")
        profile = make_profile(topics=["General"], max_items=5)
        articles = [make_article(title=f"Article {i}", url=f"https://example.com/{i}") for i in range(3)]

        digest = await generate_user_digest(profile, articles)

        assert len(digest.items) == 3


class TestErrorHandling:

    async def test_ai_error_skips_article_not_whole_digest(self, fake_llm):
        set_payload(fake_llm, topic="Tech")
        fail_on_call(fake_llm, call_index=1)
        profile = make_profile(topics=["General"])
        articles = [
            make_article(title="Bad", url="https://example.com/bad"),
            make_article(title="Good", url="https://example.com/good"),
        ]

        digest = await generate_user_digest(profile, articles)

        assert len(digest.items) == 1
        assert digest.items[0].article.title == "Good"

    async def test_empty_article_list_returns_empty_digest(self):
        profile = make_profile()

        digest = await generate_user_digest(profile, [])

        assert digest.items == []
        assert digest.user == profile.user


class TestDedup:
    # Expected to fail until dedup wiring is added to briefing_service.py.

    async def test_same_url_deduplicated(self, fake_llm):
        set_payload(fake_llm)
        profile = make_profile(topics=["General"])
        articles = [
            make_article(title="Same", url="https://example.com/1"),
            make_article(title="Same again", url="https://example.com/1"),
        ]

        digest = await generate_user_digest(profile, articles)

        assert len(digest.items) == 1

    async def test_same_content_deduplicated(self, fake_llm):
        set_payload(fake_llm)
        profile = make_profile(topics=["General"])
        articles = [
            make_article(title="A", content="Identical body text.", url="https://example.com/1"),
            make_article(title="B", content="Identical body text.", url="https://example.com/2"),
        ]

        digest = await generate_user_digest(profile, articles)

        assert len(digest.items) == 1


class TestDigestMetadata:

    async def test_digest_has_correct_user(self, fake_llm):
        set_payload(fake_llm)
        profile = make_profile(user="alice")

        digest = await generate_user_digest(profile, [make_article()])

        assert digest.user == "alice"

    async def test_generated_at_is_set(self, fake_llm):
        set_payload(fake_llm)
        profile = make_profile()

        before = datetime.utcnow()
        digest = await generate_user_digest(profile, [make_article()])
        after = datetime.utcnow()

        assert before <= digest.generated_at.replace(tzinfo=None) <= after