import asyncio
import json
from pathlib import Path

import pytest

from src.concurrency import benchmark


async def test_run_benchmark_writes_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run_ingestion(sources: list[dict], concurrency: int | None = None) -> list[dict]:
        await asyncio.sleep(0.05 if concurrency == 1 else 0.01)
        return [{"content": "x"}]

    monkeypatch.setattr(benchmark, "run_ingestion", fake_run_ingestion)
    out = tmp_path / "bench.json"

    result = await benchmark.run_benchmark(
        runs=2, concurrency=5, output=out, sources=[{"url": "u", "type": "rss"}]
    )

    saved = json.loads(out.read_text())
    assert saved["runs"] == 2
    assert len(saved["sequential"]["times_s"]) == 2
    assert result["speedup"] > 1


def test_format_report_contains_command_and_numbers() -> None:
    summary = {"times_s": [1.0], "median_s": 1.0, "min_s": 1.0, "max_s": 1.0, "articles_per_run": [3]}
    report = benchmark.format_report({
        "sources": 7, "runs": 1, "concurrency_limit": 5,
        "sequential": summary, "concurrent": summary, "speedup": 1.0,
    })
    assert "python -m src.concurrency.benchmark" in report
    assert "Speedup" in report


def test_main_rejects_zero_runs() -> None:
    with pytest.raises(SystemExit):
        benchmark.main(["--runs", "0"])