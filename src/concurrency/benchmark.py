"""Sequential vs concurrent ingestion benchmark.

Reproduce:
    python -m src.concurrency.benchmark --runs 5
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.concurrency.pipeline import CONCURRENCY_LIMIT, run_ingestion

logger = logging.getLogger(__name__)

TEST_SOURCES = [
    {"url": "http://feeds.bbci.co.uk/news/rss.xml", "type": "rss"},
    {"url": "http://rss.cnn.com/rss/edition.rss", "type": "rss"},
    {"url": "https://www.theguardian.com/world/rss", "type": "rss"},
    {"url": "https://feeds.npr.org/1001/rss.xml", "type": "rss"},
    {"url": "https://feeds.arstechnica.com/arstechnica/index/", "type": "rss"},
    {"url": "https://example.com", "type": "html"},
    {"url": "https://www.iana.org/help/example-domains", "type": "html"},
]

DEFAULT_OUTPUT = Path("artefacts/benchmark.json")


async def timed_run(sources: list[dict], concurrency: int) -> tuple[float, int]:
    start = time.perf_counter()
    articles = await run_ingestion(sources, concurrency=concurrency)
    return time.perf_counter() - start, len(articles)


def summarize(times: list[float], counts: list[int]) -> dict[str, Any]:
    return {
        "times_s": [round(t, 3) for t in times],
        "median_s": round(statistics.median(times), 3),
        "min_s": round(min(times), 3),
        "max_s": round(max(times), 3),
        "articles_per_run": counts,
    }


async def run_benchmark(
    runs: int,
    concurrency: int,
    output: Path,
    sources: list[dict] | None = None,
) -> dict[str, Any]:
    sources = TEST_SOURCES if sources is None else sources
    await run_ingestion(sources, concurrency=concurrency)  # warm-up, not recorded

    modes = [("sequential", 1), ("concurrent", concurrency)]
    times: dict[str, list[float]] = {name: [] for name, _ in modes}
    counts: dict[str, list[int]] = {name: [] for name, _ in modes}

    for index in range(runs):
        # Alternate order so neither mode always benefits from a warmer cache.
        ordered = modes if index % 2 == 0 else modes[::-1]
        for name, limit in ordered:
            elapsed, count = await timed_run(sources, limit)
            times[name].append(elapsed)
            counts[name].append(count)
            logger.info("run=%d %s limit=%d %.3fs articles=%d",
                        index + 1, name, limit, elapsed, count)

    sequential = summarize(times["sequential"], counts["sequential"])
    concurrent = summarize(times["concurrent"], counts["concurrent"])
    if len(set(counts["sequential"] + counts["concurrent"])) > 1:
        logger.warning("article counts differ between runs; some sources may have failed")

    result: dict[str, Any] = {
        "sources": len(sources),
        "runs": runs,
        "concurrency_limit": concurrency,
        "sequential": sequential,
        "concurrent": concurrent,
        "speedup": round(sequential["median_s"] / max(concurrent["median_s"], 1e-9), 2),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def format_report(result: dict[str, Any]) -> str:
    seq, con = result["sequential"], result["concurrent"]
    return (
        "### Concurrency Benchmark\n"
        f"- **Sources:** {result['sources']} | **Runs:** {result['runs']} "
        f"(median reported, 1 warm-up run excluded)\n"
        f"- **Sequential (limit 1):** median {seq['median_s']:.2f} s "
        f"(min {seq['min_s']:.2f}, max {seq['max_s']:.2f})\n"
        f"- **Concurrent (limit {result['concurrency_limit']}):** median {con['median_s']:.2f} s "
        f"(min {con['min_s']:.2f}, max {con['max_s']:.2f})\n"
        f"- **Speedup:** {result['speedup']:.2f}x\n"
        "- **Command to reproduce:** `python -m src.concurrency.benchmark --runs 5`\n"
        "- **Raw numbers:** `artefacts/benchmark.json`\n"
    )


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return number


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sequential vs concurrent ingestion benchmark")
    parser.add_argument("--runs", type=_positive_int, default=5)
    parser.add_argument("--concurrency", type=_positive_int, default=CONCURRENCY_LIMIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "ERROR").upper())
    result = asyncio.run(run_benchmark(args.runs, args.concurrency, args.output))
    sys.stdout.write(format_report(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())