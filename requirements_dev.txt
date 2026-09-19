# Dev / CI dependencies: tests, coverage, lint, type-check
-r requirements.txt
 
pytest==8.4.1
pytest-asyncio==1.1.0
pytest-cov==6.2.1
aioresponses==0.7.8   # mocks aiohttp offline (use respx instead if the team uses httpx)
mypy==1.17.1
ruff==0.12.9