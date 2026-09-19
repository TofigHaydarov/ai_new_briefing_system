"""Concurrent news ingestion: fetch RSS feeds and scrape HTML pages.

Public API (unchanged, so existing tests and the benchmark keep working):
    fetch_html, fetch_rss, safe_fetch, run_ingestion, CONCURRENCY_LIMIT,
    REQUEST_TIMEOUT

Configuration (environment variables):
    MAX_CONCURRENCY  upper bound of simultaneous requests (default 5)
    REQUEST_TIMEOUT  per-request timeout in seconds (default 10)
"""
from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable
from typing import TypedDict, TypeVar
from urllib.parse import urlparse

import aiohttp
import feedparser
from bs4 import BeautifulSoup
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

Number = TypeVar("Number", int, float)


def _env_number(
    name: str, default: Number, cast: Callable[[str], Number], minimum: Number
) -> Number:
    """Read a numeric env var; fall back to `default` if missing or invalid."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = cast(raw)
    except ValueError:
        logger.warning("Invalid %s=%r, using default %s", name, raw, default)
        return default
    if value < minimum:
        logger.warning("%s=%r is below %s, using default %s", name, raw, minimum, default)
        return default
    return value


CONCURRENCY_LIMIT: int = _env_number("MAX_CONCURRENCY", 5, int, 1)
REQUEST_TIMEOUT: float = _env_number("REQUEST_TIMEOUT", 10.0, float, 0.1)
MAX_ATTEMPTS = 3
USER_AGENT = "newsbrief/1.0 (course project)"
SUPPORTED_SOURCE_TYPES = ("rss", "html")


class RawArticle(TypedDict):
    url: str
    title: str
    content: str
    source_type: str


def _is_retryable(exc: BaseException) -> bool:
    """Retry network faults, timeouts, 429 and 5xx. Never retry other 4xx."""
    if isinstance(exc, aiohttp.ClientResponseError):
        return exc.status == 429 or exc.status >= 500
    return isinstance(exc, (aiohttp.ClientError, asyncio.TimeoutError))


def _retry_policy():  # type: ignore[no-untyped-def]
    return retry(
        stop=stop_after_attempt(MAX_ATTEMPTS),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception(_is_retryable),
        reraise=True,
    )


async def _get_text(
    session: aiohttp.ClientSession, url: str, semaphore: asyncio.Semaphore
) -> str:
    """GET a URL and return its body. The semaphore is held only while the
    request is in flight, never during retry backoff sleeps."""
    async with semaphore:
        logger.info("Fetching %s", url)
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
        async with session.get(url, timeout=timeout) as response:
            response.raise_for_status()
            return await response.text()


def _clean_text(markup: str) -> str:
    """Strip HTML tags and collapse whitespace."""
    text = BeautifulSoup(markup, "html.parser").get_text(" ", strip=True)
    return " ".join(text.split())


def parse_html(html: str, url: str) -> list[RawArticle]:
    """Extract one article (title + paragraph text) from an HTML page."""
    soup = BeautifulSoup(html, "html.parser")

    # Prefer <title>; fall back to the first <h1>, then a placeholder.
    if soup.title and soup.title.string and soup.title.string.strip():
        title = soup.title.string.strip()
    elif soup.h1 and soup.h1.get_text(strip=True):
        title = soup.h1.get_text(strip=True)
    else:
        title = "No Title"

    paragraphs = (p.get_text(" ", strip=True) for p in soup.find_all("p"))
    content = " ".join(" ".join(paragraphs).split())
    return [{"url": url, "title": title, "content": content, "source_type": "html"}]


def parse_rss(xml: str, feed_url: str) -> list[RawArticle]:
    """Extract articles from an RSS/Atom document. Entries without a link
    are dropped: a shared fallback URL would make distinct stories collide
    in URL-based dedup."""
    feed = feedparser.parse(xml)
    if feed.bozo and not feed.entries:
        logger.warning("Malformed feed %s: %s", feed_url, feed.get("bozo_exception"))
        return []

    articles: list[RawArticle] = []
    for entry in feed.entries:
        link = str(entry.get("link") or "").strip()
        if not link:
            logger.debug("Skipping entry without link in %s", feed_url)
            continue

        # `content` may be missing OR present-but-empty; `or` covers both.
        content_items = entry.get("content") or []
        raw = content_items[0].get("value", "") if content_items else ""
        raw = raw or entry.get("summary", "")

        articles.append(
            {
                "url": link,
                "title": str(entry.get("title") or "").strip() or "No Title",
                "content": _clean_text(str(raw)),
                "source_type": "rss",
            }
        )
    return articles


@_retry_policy()
async def fetch_html(
    session: aiohttp.ClientSession, url: str, semaphore: asyncio.Semaphore
) -> list[RawArticle]:
    """Fetch and scrape a direct HTML page."""
    return parse_html(await _get_text(session, url, semaphore), url)


@_retry_policy()
async def fetch_rss(
    session: aiohttp.ClientSession, url: str, semaphore: asyncio.Semaphore
) -> list[RawArticle]:
    """Fetch and parse an RSS feed."""
    return parse_rss(await _get_text(session, url, semaphore), url)


async def safe_fetch(
    session: aiohttp.ClientSession,
    url: str,
    source_type: str,
    semaphore: asyncio.Semaphore,
) -> list[RawArticle]:
    """Fetch one source; on failure log and return [] so a single broken
    source never aborts the whole run."""
    try:
        if source_type == "html":
            return await fetch_html(session, url, semaphore)
        if source_type == "rss":
            return await fetch_rss(session, url, semaphore)
        logger.warning("Unknown source type %r for %s", source_type, url)
        return []
    except Exception as exc:  # isolation boundary: one source must not kill the run
        logger.error(
            "Failed to fetch %s source %s: %s: %s",
            source_type, url, type(exc).__name__, exc,
        )
        return []


def _is_valid_article(article: RawArticle) -> bool:
    parsed = urlparse(article["url"])
    return (
        parsed.scheme in ("http", "https")
        and bool(parsed.netloc)
        and bool(article["title"].strip())
        and bool(article["content"].strip())
    )


def _valid_sources(sources: list[dict]) -> list[dict]:
    valid: list[dict] = []
    for source in sources:
        url = source.get("url")
        source_type = source.get("type")
        if not isinstance(url, str) or not url.strip() or not isinstance(source_type, str):
            logger.warning("Skipping malformed source config: %r", source)
            continue
        valid.append({"url": url.strip(), "type": source_type})
    return valid


async def run_ingestion(
    sources: list[dict], concurrency: int | None = None
) -> list[RawArticle]:
    """Fetch all sources concurrently and return validated articles.

    `sources` is a list of {"url": ..., "type": "rss" | "html"} dicts.
    `concurrency` overrides the semaphore bound (default: MAX_CONCURRENCY).
    """
    limit = CONCURRENCY_LIMIT if concurrency is None else concurrency
    if limit < 1:
        raise ValueError("concurrency must be >= 1")

    valid_sources = _valid_sources(sources)
    semaphore = asyncio.Semaphore(limit)

    async with aiohttp.ClientSession(headers={"User-Agent": USER_AGENT}) as session:
        batches = await asyncio.gather(
            *(
                safe_fetch(session, source["url"], source["type"], semaphore)
                for source in valid_sources
            )
        )

    fetched = [article for batch in batches for article in batch]
    articles = [article for article in fetched if _is_valid_article(article)]
    logger.info(
        "Ingestion complete: %d sources, %d fetched, %d valid, %d dropped",
        len(valid_sources), len(fetched), len(articles), len(fetched) - len(articles),
    )
    return articles