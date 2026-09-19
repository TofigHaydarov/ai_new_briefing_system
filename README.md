# AI News Briefing Service

A scheduled service that fetches articles from several news sources concurrently, removes near-identical stories, labels each article with an LLM (summary, topic, sentiment) and writes a personalized daily Markdown digest for a user.

The AI layer (`ai/`) is provided by the instructors and is used unchanged. Everything around it (ingestion, dedup wiring, storage, digest building, CLI, retries, logging, tests, CI, Docker) is our work.

## Pipeline

1. **Ingest** — at least five sources (RSS feeds plus directly scraped HTML pages) are fetched concurrently with `asyncio` and `aiohttp`. One failing source is logged and skipped; it never stops the run.
2. **Validate** — entries with no content are dropped before they can reach dedup or the LLM.
3. **Deduplicate (two stages)** — first the cheap pass (canonical URL and content hash), then `near_duplicate` (word-shingle Jaccard) on the survivors.
4. **Summarize and label** — `ai.summarize_and_label` returns a 2–3 sentence summary, a topic and a sentiment. Within a run, results are cached in memory by content hash, so the same article is never summarized twice.
5. **Personalize** — the user's preferred topics, excluded sources and per-topic item limit (stored in a JSON file, see "Storage: why JSON") filter and order the articles.
6. **Write** — the digest is saved as `digests/YYYY-MM-DD-<user>.md`.

## Project structure

```
news-briefing-project/
├── ai/                          # provided AI module (never modified)
├── src/
│   ├── cli.py                   # command line entry point (run-daily)
│   ├── config.py                # typed settings read from the environment
│   ├── scheduler.py             # optional scheduled runs
│   ├── concurrency/
│   │   ├── pipeline.py          # concurrent RSS and HTML ingestion
│   │   └── benchmark.py         # sequential vs concurrent benchmark
│   ├── core/
│   │   ├── dedup.py             # two-stage deduplication wiring
│   │   └── digest_builder.py    # Markdown digest output
│   ├── services/
│   │   ├── ai_service.py        # retries, caching and rate limiting around ai.*
│   │   └── briefing_service.py  # summarizes articles and assembles a user's digest
│   └── storage/
│       └── repository.py        # JSON user repository
├── tests/                       # all offline tests
│   ├── conftest.py              # shared fixtures and fakes
│   ├── helpers.py
│   └── test_*.py                # one file per module, plus the provided smoke tests
├── data/
│   ├── user_profile.json        # user profiles (preferred topics, excluded sources)
│   ├── rss_feeds.txt            # RSS feed list
│   └── html_samples/            # sample pages for the offline demo
├── digests/                     # generated Markdown digests
├── artefacts/                   # benchmark results
├── docs/                        # architecture notes
├── report/report.pdf            # technical report
├── .github/workflows/ci.yml     # CI pipeline
├── demo_ai.py                   # provided offline demo
├── Dockerfile
├── .dockerignore
├── requirements.txt             # every dependency pinned
├── pytest.ini
├── .env.example                 # every variable the app reads
├── .gitignore
└── README.md
```

## Setup

Requires Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # Windows: copy .env.example .env
```

Fill in `.env`. The variables are documented in `.env.example`; the main ones are:

| Variable | Purpose |
|---|---|
| `LLM_PROVIDER`, `LLM_MODEL` | LLM used for summaries and labels (for example `anthropic`, `claude-sonnet-4-6`) |
| `ANTHROPIC_API_KEY` | key for the LLM provider |
| `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL` | embedding provider (Anthropic has no first-party embeddings, so we pair it with OpenAI) |
| `OPENAI_API_KEY` | key for the embedding provider |
| `LOG_LEVEL` | logging level (default `INFO`) |
| `MAX_PARALLEL_FETCHES` | upper bound of simultaneous fetches (default 5) |
| `FETCH_TIMEOUT_SECONDS` | per-request timeout in seconds (default 10) |

Real keys are only needed for live runs. Tests and the offline demo need none.

Write each value without a trailing comment or trailing spaces (put comments on their own line). Python's dotenv reader tolerates them, but `docker run --env-file` keeps them as part of the value, which breaks for example the model name.

## Run

Offline demo (no keys, no network):

```bash
python demo_ai.py --offline
```

Full daily pipeline for one user:

```bash
python -m src.cli run-daily --user khagani
```

The digest is written to `digests/`.

### Example output

```
# Daily digest for khagani

## Tech
- **Acme unveils new AI processor** (Sample News)  [+]
  Acme corp announced a new processor today aimed at AI workloads...
  <https://example.com/news/article1>

## Science
- **Astronomers detect unusual signal from nearby star** (Sample News)  [=]
  ...
  <https://example.com/news/article2>
```

`[+]`, `[=]` and `[-]` mark positive, neutral and negative sentiment.

## Docker

```bash
docker build -t newsbrief .
docker run --rm --env-file .env -v "${PWD}/digests:/app/digests" newsbrief
```

The default command is `run-daily --user khagani`; pass `run-daily --user <name>` after the image name for another user. The `-v` option copies the digest to your local `digests/` folder (on Linux and macOS use `$(pwd)` instead of `${PWD}`). API keys are never baked into the image; they come from `--env-file` at run time.

The offline demo also runs inside the container, with networking disabled:

```bash
docker run --rm --network none --entrypoint python newsbrief demo_ai.py --offline
```

The image installs only `requirements.txt` and runs as a non-root user.

## Tests

```bash
python -m pytest --cov=src --cov-report=term-missing
python -m pytest tests/test_ai_smoke.py     # the provided smoke tests
```

All tests are offline: HTTP is mocked with `respx` and fake source clients, and LLM and embedding calls use `FakeLLM` and `FakeEmbedder` from `tests/conftest.py`. The full suite was also run with the network switched off.

**Result (2026-09-19): 96 passed, 83% coverage** (requirement: at least 60%).

| Module | Statements | Missed | Coverage |
|---|---|---|---|
| `src/cli.py` | 127 | 3 | 98% |
| `src/concurrency/pipeline.py` | 69 | 3 | 96% |
| `src/concurrency/benchmark.py` | 40 | 22 | 45% |
| `src/config.py` | 40 | 3 | 92% |
| `src/core/dedup.py` | 37 | 0 | 100% |
| `src/services/ai_service.py` | 65 | 8 | 88% |
| `src/services/briefing_service.py` | 49 | 0 | 100% |
| `src/storage/repository.py` | 64 | 21 | 67% |
| `src/scheduler.py` | 30 | 30 | 0% |
| **Total** | **521** | **90** | **83%** |

## Type checking

```bash
python -m mypy --ignore-missing-imports --explicit-package-bases src
```

Result with mypy 2.3.1: 7 errors in 3 files. Five are in `src/` (missing variable annotations in `src/core/dedup.py` and `src/services/ai_service.py`, and one return type in `ai_service.py`). Two are in the provided `ai/providers/openai.py`, which we are not allowed to modify. CI runs the same check with `--follow-imports=silent` so that only our own code is reported.

## Concurrency benchmark

Sequential and concurrent runs use the same fetch code; only the semaphore bound differs (1 versus 5). Each configuration gets one warm-up run that is not recorded, then five timed runs. The order of the two modes alternates between runs.

- **Sources:** 7 (5 RSS feeds and 2 HTML pages)
- **Articles fetched per run:** 152 in every run of both modes, so both did identical work
- **Environment:** Windows 11, Python 3.12.10, 2026-09-19

| Mode | Runs (s) | Median (s) | Min (s) | Max (s) |
|---|---|---|---|---|
| Sequential (limit 1) | 1.971, 1.813, 1.922, 2.007, 1.768 | 1.92 | 1.77 | 2.01 |
| Concurrent (limit 5) | 0.644, 0.686, 0.664, 0.660, 0.669 | 0.66 | 0.64 | 0.69 |

**Speedup: 2.89x** (median sequential divided by median concurrent). It stays below the 7x ideal because the bound is 5 and a single slow source dominates the concurrent run.

Reproduce (the benchmark script lives on the `feat/benchmark` branch; it is not part of `main`):

```bash
git checkout feat/benchmark
python -m src.concurrency.benchmark --runs 5
```

Raw numbers are stored in `artefacts/benchmark.json`. The benchmark uses the real network, so it is kept out of the test suite.

## Design decisions

- **Bounded parallelism.** Fetching is limited by an `asyncio.Semaphore` whose bound comes from `MAX_PARALLEL_FETCHES` (default 5); the per-request timeout comes from `FETCH_TIMEOUT_SECONDS` (default 10). LLM calls have their own bound of 3 concurrent requests so that many articles cannot exhaust the provider's rate limit.
- **Retries with exponential backoff.** Fetches retry up to 3 times on connection errors and timeouts. AI calls also retry up to 3 times and wait much longer after a rate-limit (HTTP 429) response. Validation errors are not retried.
- **Cheap dedup before paid calls.** URL and content-hash matching removes exact and syndicated duplicates before any LLM call is made.
- **Caching.** Summaries are cached by content hash and embeddings by text. Both caches live in memory for the duration of one run.
- **Defensive ingestion.** Every source is fetched inside its own error boundary; failures are logged and the run continues.
- **Logging.** The standard `logging` module is used everywhere; there are no `print` calls for runtime diagnostics.

## Storage: why JSON

User profiles (preferred topics, excluded sources, items per topic) are stored in a JSON file, `data/user_profile.json`, and accessed through `JSONUserRepo` in `src/storage/repository.py`. We deliberately chose JSON over PostgreSQL; the project brief allows either. The reasons:

- **The data is small and flat.** A profile is a handful of fields per user. There are no relations, queries or reports that would justify a database.
- **One process, one writer.** The service runs as a daily command-line job, so concurrent writes to the profile file are not a concern.
- **No extra infrastructure.** No database server, connection string or migrations. The Docker image stays a single container that starts with one command, and a teammate can run the project right after `pip install`.
- **Offline, deterministic tests.** Storage tests use a temporary file instead of a live database, which keeps the whole suite offline as the brief requires.
- **Easy to inspect and review.** The file is human-readable, can be changed in a pull request and doubles as the sample data for the demo.

Trade-offs: the file is read and rewritten as a whole, it gives no protection against simultaneous writers, and there is no persistent store of already processed URLs (the caches described above live in memory for one run). Because all storage access is isolated in one module, moving to a database later would only touch that module.

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs on every push to `main` and every pull request: ruff, mypy, the test suite with a 60% coverage gate, and a Docker build.

## Known limitations

- If every LLM call fails (for example when the provider quota is exhausted), the CLI still writes an empty digest and exits successfully. Live runs depend on the provider's quota.
- `src/scheduler.py` has no tests (0% coverage).
- mypy reports the five `src/` issues listed above.
- The benchmark measures real network fetches, so absolute times depend on the connection; the speedup ratio is the meaningful figure.

## Team

- Tofig — test infrastructure, CI, Dockerfile
- Mahammad — AI integration
- Ibrahim — ingestion and concurrency
- Ilham — storage, CLI