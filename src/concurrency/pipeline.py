"""
Async ingestion pipeline for fetching and processing articles.
Handles both HTML and RSS sources with concurrency limits and retries.
"""
import os
import asyncio
import logging
import aiohttp
import feedparser
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

CONCURRENCY_LIMIT = int(os.environ.get("MAX_CONCURRENCY", "5"))
REQUEST_TIMEOUT = 10  # seconds

# Retry strategy: up to 3 attempts, exponential backoff (1s to 10s)
# Triggers only on client or timeout errors.
retry_strategy = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
    reraise=True
)


@retry_strategy
async def fetch_html(session: aiohttp.ClientSession, url: str, semaphore: asyncio.Semaphore) -> list[dict]:
    """
    Fetch and parse an HTML page concurrently.
    Extracts the title and paragraphs, returning a structured dictionary.
    """
    async with semaphore:
        logger.info(f"Fetching HTML source: {url}")
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)

        async with session.get(url, timeout=timeout) as response:
            response.raise_for_status()
            html_content = await response.text()

            # Scrape content using BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')

            # Extract title with fallback to h1
            if soup.title and soup.title.string:
                title = soup.title.string.strip()
            elif soup.h1 and soup.h1.get_text(strip=True):
                title = soup.h1.get_text(strip=True)
            else:
                title = "No Title"

            # Extract and join paragraph text
            paragraphs = soup.find_all('p')
            content = " ".join([p.get_text(strip=True) for p in paragraphs])

            # Returned as a list to maintain consistency with RSS parser output
            return [{"url": url, "title": title, "content": content, "source_type": "html"}]


@retry_strategy
async def fetch_rss(session: aiohttp.ClientSession, url: str, semaphore: asyncio.Semaphore) -> list[dict]:
    """
    Fetch and parse an RSS feed concurrently.
    Extracts entries and falls back to summaries if full content is missing.
    """
    async with semaphore:
        logger.info(f"Fetching RSS source: {url}")
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)

        async with session.get(url, timeout=timeout) as response:
            response.raise_for_status()
            xml_content = await response.text()

            # Parse RSS feed using feedparser
            feed = feedparser.parse(xml_content)
            articles = []

            for entry in feed.entries:
                # Extract content or fallback to summary
                content_list = entry.get('content') or [{'value': entry.get('summary', '')}]
                raw_content = content_list[0]['value']

                # Strip HTML tags
                clean_content = BeautifulSoup(raw_content, "html.parser").get_text(strip=True)

                articles.append({
                    "url": entry.get('link', url),
                    "title": entry.get('title', 'No Title'),
                    "content": clean_content,
                    "source_type": "rss"
                })

            return articles


async def safe_fetch(session: aiohttp.ClientSession, url: str, source_type: str, semaphore: asyncio.Semaphore) -> list[dict]:
    """
    Wrapper for fetch functions providing graceful degradation on failure.
    """
    try:
        if source_type == "html":
            return await fetch_html(session, url, semaphore)
        elif source_type == "rss":
            return await fetch_rss(session, url, semaphore)
        
        logger.warning(f"Unknown source type: {source_type} for URL: {url}")
        return []
    except Exception as e:
        # Graceful degradation: catch all remaining exceptions and log them
        logger.error(f"Failed to fetch {source_type} source {url} after retries. Error: {e}")
        return []


async def run_ingestion(sources: list[dict]) -> list[dict]:
    """
    Main entry point for concurrent data ingestion.
    Coordinates fetching across multiple sources and filters empty results.
    """
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    all_articles = []

    async with aiohttp.ClientSession() as session:
        # Create a concurrent task for each source in the configuration
        tasks = [
            safe_fetch(session, source['url'], source['type'], semaphore)
            for source in sources
        ]

        # Execute all fetching tasks concurrently
        results = await asyncio.gather(*tasks)

        # Flatten the list of lists into a single dataset
        for result_list in results:
            if result_list:
                all_articles.extend(result_list)

    logger.info(f"Ingestion complete. Fetched {len(all_articles)} raw articles.")

    # Filter out articles with no text content
    valid_articles = [a for a in all_articles if a.get("content", "").strip()]
    dropped = len(all_articles) - len(valid_articles)
    
    if dropped > 0:
        logger.info(f"Dropped {dropped} article(s) with empty content.")

    return valid_articles