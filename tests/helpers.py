import asyncio
from types import SimpleNamespace

import aiohttp


class FakeResponse:
    def __init__(self, status, body, tracker=None, delay=0.0, url="https://example.com/fake"):
        self.status = status
        self._body = body
        self._tracker = tracker
        self._delay = delay
        self._url = url

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def raise_for_status(self):
        if self.status >= 400:
            request_info = SimpleNamespace(real_url=self._url)
            raise aiohttp.ClientResponseError(
                request_info=request_info, history=(), status=self.status
            )

    async def text(self):
        if self._tracker is not None:
            self._tracker["active"] += 1
            self._tracker["peak"] = max(self._tracker["peak"], self._tracker["active"])
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._tracker is not None:
            self._tracker["active"] -= 1
        return self._body


class FakeSession:
    """In-memory aiohttp.ClientSession stand-in. Maps url -> (status, body).
    fail_times lets a url fail N times before succeeding, to simulate transient errors."""

    def __init__(self, responses, tracker=None, delay=0.0, fail_times=0):
        self._responses = responses
        self._tracker = tracker
        self._delay = delay
        self._fail_times = fail_times
        self._attempts = {}
        self.calls = []

    def get(self, url, timeout=None):
        self.calls.append(url)
        if url not in self._responses:
            raise aiohttp.ClientConnectionError(f"No fake response configured for {url}")

        attempt = self._attempts.get(url, 0) + 1
        self._attempts[url] = attempt
        if attempt <= self._fail_times:
            raise aiohttp.ClientConnectionError("Simulated transient failure")

        status, body = self._responses[url]
        return FakeResponse(status, body, tracker=self._tracker, delay=self._delay, url=url)


class FakeClientSessionCM:
    def __init__(self, session):
        self._session = session

    async def __aenter__(self):
        return self._session

    async def __aexit__(self, *exc):
        return False


def patch_client_session(monkeypatch, module, fake_session):
    """Patches `module.aiohttp.ClientSession()` so `async with aiohttp.ClientSession()`
    yields fake_session instead of opening a real connection."""
    monkeypatch.setattr(module.aiohttp, "ClientSession", lambda *a, **kw: FakeClientSessionCM(fake_session))