"""HTTP plumbing: polite throttled sessions, a disk cache, and browser-impersonating
sessions for the TASE / MAYA endpoints, which sit behind a WAF that blocks plain clients."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import CACHE

_RETRY = Retry(total=5, backoff_factor=1.5, status_forcelist=(429, 500, 502, 503, 504),
               allowed_methods=frozenset({"GET", "POST"}), respect_retry_after_header=True)


class Throttle:
    def __init__(self, per_second: float):
        self.gap = 1.0 / per_second
        self.lock = threading.Lock()
        self.last = 0.0

    def wait(self) -> None:
        with self.lock:
            delay = self.gap - (time.monotonic() - self.last)
            if delay > 0:
                time.sleep(delay)
            self.last = time.monotonic()


def make_session(user_agent: str, extra_headers: dict | None = None) -> requests.Session:
    s = requests.Session()
    s.mount("https://", HTTPAdapter(max_retries=_RETRY, pool_maxsize=16))
    s.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
    if extra_headers:
        s.headers.update(extra_headers)
    return s


class DiskCache:
    """Key/value cache on disk. ttl=None means the entry never expires (immutable documents
    such as SEC filings or MAYA attachments)."""

    def __init__(self, namespace: str):
        self.dir = CACHE / namespace
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        h = hashlib.sha256(key.encode()).hexdigest()
        return self.dir / h[:2] / f"{h}.json"

    def get(self, key: str, ttl_seconds: float | None) -> object | None:
        p = self._path(key)
        if not p.exists():
            return None
        if ttl_seconds is not None and time.time() - p.stat().st_mtime > ttl_seconds:
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))["v"]
        except (json.JSONDecodeError, KeyError):
            return None

    def prune(self, max_age_days: float) -> int:
        cutoff = time.time() - max_age_days * 86400
        n = 0
        for p in self.dir.glob("*/*.json"):
            if p.stat().st_mtime < cutoff:
                p.unlink(missing_ok=True)
                n += 1
        return n

    def put(self, key: str, value: object) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps({"k": key, "v": value}, ensure_ascii=False), encoding="utf-8")
        tmp.replace(p)


def prune_all(max_age_days: float = 45) -> int:
    """Removes cache entries not refreshed for max_age_days, across all namespaces."""
    total = 0
    if CACHE.exists():
        for ns in CACHE.iterdir():
            if ns.is_dir():
                total += DiskCache(ns.name).prune(max_age_days)
    return total


class BrowserSession:
    """curl_cffi session impersonating Chrome's TLS fingerprint, seeded with WAF cookies
    from a landing page. Required by api.tase.co.il and maya.tase.co.il."""

    def __init__(self, seed_url: str, origin: str, per_second: float = 2.5, lang: str = "he-IL"):
        from curl_cffi import requests as creq

        self._creq = creq
        self.seed_url = seed_url
        self.origin = origin
        self.lang = lang
        self.throttle = Throttle(per_second)
        self.lock = threading.Lock()  # curl_cffi sessions are not thread-safe
        self._new()

    def _new(self) -> None:
        self.s = self._creq.Session(impersonate="chrome")
        self.s.headers.update({"Accept": "application/json, text/plain, */*",
                               "Accept-Language": self.lang, "Origin": self.origin,
                               "Referer": self.origin + "/"})
        self.s.get(self.seed_url, timeout=45)

    def request(self, method: str, url: str, *, json_body: dict | None = None,
                params: dict | None = None, tries: int = 4, raw: bool = False):
        with self.lock:
            return self._request(method, url, json_body, params, tries, raw)

    def _request(self, method, url, json_body, params, tries, raw):
        last_exc: Exception | None = None
        for attempt in range(tries):
            self.throttle.wait()
            try:
                if method == "POST":
                    r = self.s.post(url, json=json_body, params=params, timeout=60)
                else:
                    r = self.s.get(url, params=params, timeout=60)
            except Exception as exc:  # network error: back off, rebuild session
                last_exc = exc
                time.sleep(3 * (attempt + 1))
                self._new()
                continue
            if r.status_code == 404:
                return None
            if r.status_code == 400:
                r.raise_for_status()
            if r.status_code in (403, 429) or r.status_code >= 500:
                # WAF throttling: wait, re-seed cookies, retry.
                last_exc = RuntimeError(f"{url} -> HTTP {r.status_code}")
                time.sleep(5 * (attempt + 1))
                self._new()
                continue
            r.raise_for_status()
            return r if raw else r.json()
        raise RuntimeError(f"Request failed after {tries} attempts: {url}") from last_exc
