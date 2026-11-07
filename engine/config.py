"""Loads config.yaml and resolves paths and secrets."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "state"
CACHE = STATE / "cache"


@lru_cache(maxsize=1)
def cfg() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


def secret(env_name: str, required: bool = True) -> str | None:
    value = os.environ.get(env_name)
    if required and not value:
        raise RuntimeError(f"Missing required environment variable {env_name}")
    return value


def ensure_dirs() -> None:
    for p in (STATE, CACHE, STATE / "theses", STATE / "research"):
        p.mkdir(parents=True, exist_ok=True)
