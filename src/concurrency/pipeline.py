import asyncio
import logging
import aiohttp
import feedparser
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Configure logging for the pipeline
logger = logging.getLogger(__name__)

# Constants for concurrency control and timeouts
CONCURRENCY_LIMIT = 5
REQUEST_TIMEOUT = 10  # seconds per request

# Define retry strategy: max 3 attempts, exponential backoff (e.g., 1s, 2s, 4s)
# Only retry on specific aiohttp client errors or timeout errors to avoid infinite loops on 404s
retry_strategy = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
    reraise=True
)

@retry_strategy
async def fetch_html(session: aiohttp.ClientSession, url: str, semaphore: asyncio.Semaphore) -> list[dict]:
    """
    Fetches and parses a direct HTML page.
    Uses a semaphore to limit concurrent connections and respects the configured timeout.
    """
    async with semaphore:
        logger.info(f"Fetching HTML source: {url}")
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
        
        async with session.get(url, timeout=timeout) as response:
            response.raise_for_status()
            html_content = await response.text()
            
            # Scrape content using BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            title = soup.title.string if soup.title else "No Title"
            
            # Extract paragraphs to form the article content (basic extraction)
            paragraphs = soup.find_all('p')
            content = " ".join([p.get_text(strip=True) for p in paragraphs])
            
            # Returned as a list to maintain consistency with RSS parser output
            return [{"url": url, "title": title, "content": content, "source_type": "html"}]

@retry_strategy
async def fetch_rss(session: aiohttp.ClientSession, url: str, semaphore: asyncio.Semaphore) -> list[dict]:
    """
    Fetches and parses an RSS feed.
    Uses a semaphore to limit concurrent connections and respects the configured timeout.
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
                # Try to get the full content, fallback to summary if missing
                raw_content = entry.get('content', [{'value': entry.get('summary', '')}])[0]['value']
                
                # Optional: Clean HTML tags from RSS content using BeautifulSoup
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
    Wrapper around fetch functions to ensure graceful degradation.
    If a source completely fails after all retries, it logs the error 
    and returns an empty list instead of crashing the entire gather operation.
    """
    try:
        if source_type == "html":
            return await fetch_html(session, url, semaphore)
        elif source_type == "rss":
            return await fetch_rss(session, url, semaphore)
        else:
            logger.warning(f"Unknown source type: {source_type} for URL: {url}")
            return []
    except Exception as e:
        # Graceful degradation: catch all remaining exceptions and log them
        logger.error(f"Failed to fetch {source_type} source {url} after retries. Error: {e}")
        return []

async def run_ingestion(sources: list[dict]) -> list[dict]:
    """
    Main entry point for concurrent ingestion.
    Takes a list of dictionaries containing 'url' and 'type' (e.g., 'rss' or 'html').
    Returns a flattened list of all fetched articles ready for AI integration.
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
                
    logger.info(f"Ingestion complete. Fetched {len(all_articles)} articles in total.")
    return all_articles