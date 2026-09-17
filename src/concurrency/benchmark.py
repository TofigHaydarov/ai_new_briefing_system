import asyncio
import time
import aiohttp
import logging
from src.concurrency.pipeline import safe_fetch, run_ingestion

# Suppress detailed logs during benchmarking to keep output clean
logging.basicConfig(level=logging.ERROR)

# Define the 5 official RSS feeds from data/rss_feeds.txt and 2 HTML sources for testing
TEST_SOURCES = [
    {"url": "http://feeds.bbci.co.uk/news/rss.xml", "type": "rss"},
    {"url": "http://rss.cnn.com/rss/edition.rss", "type": "rss"},
    {"url": "https://www.theguardian.com/world/rss", "type": "rss"},
    {"url": "https://feeds.npr.org/1001/rss.xml", "type": "rss"},
    {"url": "https://feeds.arstechnica.com/arstechnica/index/", "type": "rss"},
    {"url": "https://en.wikipedia.org/wiki/Software_engineering", "type": "html"},
    {"url": "https://en.wikipedia.org/wiki/Artificial_intelligence", "type": "html"},
]

async def run_sequential(sources: list[dict]) -> list[dict]:
    """
    Fetches articles one by one to establish a sequential baseline.
    """
    all_articles = []
    # A semaphore of 1 essentially forces sequential execution
    semaphore = asyncio.Semaphore(1) 
    
    async with aiohttp.ClientSession() as session:
        for source in sources:
            result = await safe_fetch(session, source['url'], source['type'], semaphore)
            if result:
                all_articles.extend(result)
                
    return all_articles

async def main():
    print("Starting Benchmark: Sequential vs Concurrent Ingestion...\n")
    
    # 1. Run and measure sequential fetching
    print("Running sequential fetch (one by one)...")
    start_seq = time.perf_counter()
    await run_sequential(TEST_SOURCES)
    end_seq = time.perf_counter()
    seq_time = end_seq - start_seq
    print(f"Sequential Time: {seq_time:.2f} seconds\n")

    # 2. Run and measure concurrent fetching
    print("Running concurrent fetch (asyncio.gather)...")
    start_conc = time.perf_counter()
    await run_ingestion(TEST_SOURCES)
    end_conc = time.perf_counter()
    conc_time = end_conc - start_conc
    print(f"Concurrent Time: {conc_time:.2f} seconds\n")

    # 3. Calculate speedup and generate README markdown
    speedup = seq_time / conc_time if conc_time > 0 else 0
    
    print("=== COPY THE FOLLOWING TEXT TO README.md ===")
    print("### Concurrency Benchmark")
    print(f"- **Sequential Execution Time:** {seq_time:.2f} seconds")
    print(f"- **Concurrent Execution Time:** {conc_time:.2f} seconds")
    print(f"- **Performance Gain:** {speedup:.2f}x faster using `asyncio` and `aiohttp`")
    print("- **Command to reproduce:** `python -m src.concurrency.benchmark`")
    print("============================================")

if __name__ == "__main__":
    asyncio.run(main())