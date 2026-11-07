"""Runs strategy.analyze on a bundle and returns a validated, annotated thesis."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path
from types import ModuleType

from .config import ROOT, cfg
from .guard import check
from .llm import Gemini, StockLLM
from .schema import validate

STRATEGY_PATH = ROOT / "strategy.py"


def strategy_version(path: Path = STRATEGY_PATH) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()[:10]


def load_strategy(path: Path = STRATEGY_PATH) -> ModuleType:
    check(path.read_text(encoding="utf-8"))
    spec = importlib.util.spec_from_file_location("strategy_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(strategy: ModuleType, version: str, bundle: dict, gemini: Gemini, previous: dict | None) -> dict:
    llm = StockLLM(gemini, cfg()["llm"]["max_calls_per_stock"])
    thesis = strategy.analyze(copy.deepcopy(bundle), llm, copy.deepcopy(previous) if previous else None)
    if not isinstance(thesis, dict):
        raise TypeError("analyze() must return a dict")
    validate(thesis)
    from .bundle import source_ids

    known = source_ids(bundle)
    unknown = [e["source_id"] for e in thesis["evidence"] if e["source_id"] not in known]
    thesis["meta"] = {
        "key": bundle["stock"]["key"], "as_of": bundle["as_of"], "benchmark": bundle["benchmark"],
        "model": cfg()["llm"]["thesis_model"], "strategy_version": version, "llm_calls": llm.calls,
        "unknown_evidence_ids": unknown, "industry": (bundle.get("profile") or {}).get("industry", ""), "data_gaps": [g["part"] for g in bundle.get("data_gaps") or []],
    }
    return thesis
