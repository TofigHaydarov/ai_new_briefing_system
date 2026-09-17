from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ai.providers.base import LLMProvider, EmbeddingProvider
from ai.schemas import Article



class FakeLLM(LLMProvider):
    """Returns a fixed JSON response. No network."""

    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        self.payload = payload or {
            "summary": "A new chip from a major vendor was announced today.",
            "topic": "Tech",
            "sentiment": "neutral",
        }
        self.calls: list[str] = []

    def complete(
        self,
        prompt: str,
        *,
        json_schema: dict | None = None,
        max_tokens: int = 1024,
    ) -> str:
        self.calls.append(prompt)
        return json.dumps(self.payload)


class FakeEmbedder(EmbeddingProvider):
    """Deterministic 8-D unit vector from a hash; same input -> same output."""

    @property
    def dimension(self) -> int:
        return 8

    def embed(self, text: str) -> np.ndarray:
        if not text.strip():
            raise ValueError("Cannot embed empty string.")
        rng = np.random.default_rng(seed=abs(hash(text)) % (2**31))
        v = rng.standard_normal(8).astype(np.float32)
        v /= np.linalg.norm(v)
        return v


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def sample_article() -> Article:
    return Article(
        title="Acme unveils new processor",
        url="https://example.com/news/acme-chip?utm_source=twitter",
        source="Example News",
        content="Acme Corp announced a new processor today aimed at AI workloads. "
        "The chip features improved energy efficiency over the previous "
        "generation. Industry analysts welcomed the news.",
    )



class SourceClient(ABC):
    """Interface your fetch_service should depend on, not aiohttp/httpx
    directly — that's what makes it swappable with FakeSourceClient here.
    """

    @abstractmethod
    async def fetch(self, url: str) -> str:
        """Return the raw response body (RSS XML or HTML) for a URL.

        Should raise on non-2xx status so callers' retry/backoff logic
        has something to catch.
        """
        raise NotImplementedError


class FakeSourceClient(SourceClient):
    """In-memory fake: maps URL -> (status_code, body). No network.

    Lets you simulate a broken source (e.g. 503) to exercise your
    per-source try/except + retry/backoff without hitting anything real.
    """

    class SourceError(Exception):
        def __init__(self, url: str, status_code: int) -> None:
            super().__init__(f"{url} returned {status_code}")
            self.url = url
            self.status_code = status_code

    def __init__(self, responses: dict[str, tuple[int, str]]) -> None:
        self._responses = responses
        self.calls: list[str] = []

    async def fetch(self, url: str) -> str:
        self.calls.append(url)
        if url not in self._responses:
            raise self.SourceError(url, 404)
        status_code, body = self._responses[url]
        if status_code >= 400:
            raise self.SourceError(url, status_code)
        return body


class Repository(ABC):
    """Interface your storage layer should implement (real one backed by
    asyncpg/psycopg; fake one in-memory) — covers user profiles and the
    processed-URL dedup cache mentioned in TOPIC.md.
    """

    @abstractmethod
    async def get_user(self, name: str) -> Any:
        raise NotImplementedError

    @abstractmethod
    async def has_processed(self, content_hash: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def mark_processed(self, content_hash: str) -> None:
        raise NotImplementedError

    @abstractmethod
    async def save_digest(self, user_name: str, date: str, markdown: str) -> Path:
        raise NotImplementedError


class FakeRepository(Repository):
    """In-memory fake: no Postgres, no filesystem DB. Digests are still
    written to a real tmp_path so end-to-end tests can assert on an
    actual file, since that's the graded deliverable (digests/*.md).
    """

    def __init__(self, users: dict[str, Any], digests_dir: Path) -> None:
        self._users = users
        self._digests_dir = digests_dir
        self._processed: set[str] = set()
        self.saved_digests: list[Path] = []

    async def get_user(self, name: str) -> Any:
        if name not in self._users:
            raise KeyError(f"No such user: {name}")
        return self._users[name]

    async def has_processed(self, content_hash: str) -> bool:
        return content_hash in self._processed

    async def mark_processed(self, content_hash: str) -> None:
        self._processed.add(content_hash)

    async def save_digest(self, user_name: str, date: str, markdown: str) -> Path:
        path = self._digests_dir / f"{date}-{user_name}.md"
        path.write_text(markdown, encoding="utf-8")
        self.saved_digests.append(path)
        return path


@dataclass
class User:
    """Stand-in for src.models.User — swap for the real import once it exists."""

    name: str
    preferred_topics: list[str] = field(default_factory=lambda: ["Tech", "Science"])
    excluded_sources: list[str] = field(default_factory=list)


@pytest.fixture
def sample_user() -> User:
    return User(
        name="khagani",
        preferred_topics=["Tech", "Science"],
        excluded_sources=["Spammy Times"],
    )


@pytest.fixture
def digests_dir(tmp_path: Path) -> Path:
    d = tmp_path / "digests"
    d.mkdir()
    return d


@pytest.fixture
def fake_repository(sample_user: User, digests_dir: Path) -> FakeRepository:
    return FakeRepository(users={sample_user.name: sample_user}, digests_dir=digests_dir)


@pytest.fixture
def fake_source_client() -> FakeSourceClient:
    rss_body = """<?xml version="1.0"?>
    <rss><channel>
      <item>
        <title>Acme unveils new AI processor</title>
        <link>https://example.com/news/article1?utm_source=twitter</link>
        <description>Acme corp announced a new processor today aimed at AI workloads.</description>
      </item>
    </channel></rss>"""

    mirror_rss_body = """<?xml version="1.0"?>
    <rss><channel>
      <item>
        <title>Acme unveils new AI processor (syndicated)</title>
        <link>https://example.com/news/article1-mirror</link>
        <description>Acme corp announced a new processor today aimed at AI workloads. Full copy.</description>
      </item>
    </channel></rss>"""

    html_body = """
    <html><body>
      <article>
        <h1>Astronomers detect unusual signal from nearby star</h1>
        <p>Researchers report an unusual radio signal from a nearby star system.</p>
      </article>
    </body></html>
    """

    return FakeSourceClient(
        responses={
            "https://example.com/rss/feed1": (200, rss_body),
            "https://example.com/rss/feed2-mirror": (200, mirror_rss_body),
            "https://example.com/scrape/page1": (200, html_body),
            "https://example.com/broken/feed3": (503, ""),
        }
    )


@pytest.fixture
def sample_articles() -> list[Article]:
    """Pre-parsed Articles for tests that don't need to exercise the
    RSS/HTML parsing step itself (e.g. dedup or digest-builder unit tests).
    Includes one deliberate near-duplicate pair.
    """
    return [
        Article(
            title="Acme unveils new AI processor",
            url="https://example.com/news/article1?utm_source=twitter",
            source="Sample News",
            content="Acme corp announced a new processor today aimed at AI workloads.",
        ),
        Article(
            title="Acme unveils new AI processor (syndicated)",
            url="https://example.com/news/article1-mirror",
            source="Mirror Wire",
            content="Acme corp announced a new processor today aimed at AI workloads. Full copy.",
        ),
        Article(
            title="Astronomers detect unusual signal from nearby star",
            url="https://example.com/news/article2",
            source="Sample News",
            content="Researchers report an unusual radio signal from a nearby star system.",
        ),
    ]


@pytest.fixture
def user_profile_file(tmp_path: Path, sample_user: User) -> Path:
    """A user_profile.json on disk, for tests of the JSON-based profile
    loading path (if your team uses JSON instead of/alongside Postgres).
    """
    path = tmp_path / "user_profile.json"
    path.write_text(
        json.dumps(
            {
                "name": sample_user.name,
                "preferred_topics": sample_user.preferred_topics,
                "excluded_sources": sample_user.excluded_sources,
            }
        )
    )
    return path