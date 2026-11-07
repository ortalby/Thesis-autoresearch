"""Gemini API client with a per-minute rate limiter, a persistent daily call budget, retries on
transient errors, and JSON-schema constrained output."""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone

from google import genai
from google.genai import errors, types

from .config import STATE, cfg, secret

USAGE_FILE = STATE / "llm_usage.json"


class BudgetExceeded(RuntimeError):
    """The daily call budget in config.yaml (llm.max_calls_per_day) is used up."""


class QuotaExhausted(RuntimeError):
    """The API itself reports the project's quota is exhausted (HTTP 429 that persists)."""


class _Usage:
    def __init__(self):
        self.lock = threading.Lock()

    def _load(self) -> dict:
        if USAGE_FILE.exists():
            return json.loads(USAGE_FILE.read_text(encoding="utf-8"))
        return {}

    def today(self) -> str:
        # Gemini daily quotas reset at midnight Pacific; UTC is close enough for a soft cap.
        return datetime.now(timezone.utc).date().isoformat()

    def reserve(self, limit: int) -> None:
        with self.lock:
            data = self._load()
            day = data.setdefault(self.today(), {"calls": 0, "input_tokens": 0, "output_tokens": 0})
            if day["calls"] >= limit:
                raise BudgetExceeded(f"Daily LLM budget of {limit} calls reached")
            day["calls"] += 1
            for k in sorted(data)[:-60]:  # keep the last 60 days
                data.pop(k)
            USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
            USAGE_FILE.write_text(json.dumps(data, indent=1), encoding="utf-8")

    def add_tokens(self, inp: int, out: int) -> None:
        with self.lock:
            data = self._load()
            day = data.setdefault(self.today(), {"calls": 0, "input_tokens": 0, "output_tokens": 0})
            day["input_tokens"] += inp
            day["output_tokens"] += out
            USAGE_FILE.write_text(json.dumps(data, indent=1), encoding="utf-8")

    def calls_today(self) -> int:
        return self._load().get(self.today(), {}).get("calls", 0)


usage = _Usage()


class RateLimiter:
    def __init__(self, per_minute: int):
        self.gap = 60.0 / max(1, per_minute)
        self.lock = threading.Lock()
        self.next_at = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            at = max(now, self.next_at)
            self.next_at = at + self.gap
        if at > now:
            time.sleep(at - now)


class Gemini:
    def __init__(self):
        c = cfg()["llm"]
        self.client = genai.Client(api_key=secret("GEMINI_API_KEY"))
        self.limiter = RateLimiter(c["requests_per_minute"])
        self.max_per_day = c["max_calls_per_day"]
        self.thinking_level = c["thinking_level"]
        self.max_input_chars = c["max_input_chars_per_call"]
        self.max_output_tokens = c["max_output_tokens"]

    def _call(self, model: str, system: str, user: str, config_extra: dict) -> types.GenerateContentResponse:
        if len(system) + len(user) > self.max_input_chars:
            raise ValueError(f"Prompt is {len(system) + len(user)} chars; limit is {self.max_input_chars}")
        conf = types.GenerateContentConfig(
            system_instruction=system or None,
            max_output_tokens=self.max_output_tokens,
            thinking_config=types.ThinkingConfig(thinking_level=self.thinking_level),
            **config_extra,
        )
        delay = 10.0
        for attempt in range(6):
            usage.reserve(self.max_per_day)
            self.limiter.wait()
            try:
                resp = self.client.models.generate_content(model=model, contents=user, config=conf)
            except errors.APIError as exc:
                code = getattr(exc, "code", None)
                if code == 429 and "PerDay" in str(exc):
                    # Daily free-tier quota: waiting will not help until it resets.
                    raise QuotaExhausted(str(exc)) from exc
                if code == 429 or (code and code >= 500):
                    if attempt == 5:
                        if code == 429:
                            raise QuotaExhausted(str(exc)) from exc
                        raise
                    time.sleep(delay)
                    delay = min(delay * 2, 300)
                    continue
                raise
            um = resp.usage_metadata
            if um:
                usage.add_tokens(um.prompt_token_count or 0,
                                 (um.candidates_token_count or 0) + (um.thoughts_token_count or 0))
            return resp
        raise RuntimeError("unreachable")

    def json(self, model: str, system: str, user: str, schema: dict) -> dict:
        last_err = None
        for _ in range(2):
            resp = self._call(model, system, user, {"response_mime_type": "application/json",
                                                     "response_json_schema": schema})
            text = resp.text or ""
            try:
                return json.loads(text)
            except json.JSONDecodeError as exc:
                last_err = exc
        raise ValueError(f"Model did not return valid JSON: {last_err}")

    def text(self, model: str, system: str, user: str) -> str:
        return self._call(model, system, user, {}).text or ""


class StockLLM:
    """What strategy.py receives: JSON-only calls to the thesis model, capped per stock."""

    def __init__(self, gemini: Gemini, max_calls: int):
        self._g = gemini
        self._left = max_calls
        self.model = cfg()["llm"]["thesis_model"]
        self.calls = 0

    @property
    def calls_left(self) -> int:
        return self._left

    def json(self, system: str, user: str, schema: dict) -> dict:
        if self._left <= 0:
            raise BudgetExceeded("Per-stock LLM call limit reached")
        self._left -= 1
        self.calls += 1
        return self._g.json(self.model, system, user, schema)
