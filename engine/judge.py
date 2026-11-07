"""Fixed quality judge (part of the evaluation harness; the research agent cannot change it)."""
from __future__ import annotations

import json

from .config import cfg
from .llm import Gemini

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "grounding": {"type": "integer", "minimum": 1, "maximum": 5},
        "completeness": {"type": "integer", "minimum": 1, "maximum": 5},
        "consistency": {"type": "integer", "minimum": 1, "maximum": 5},
        "falsifiability": {"type": "integer", "minimum": 1, "maximum": 5},
        "risk_balance": {"type": "integer", "minimum": 1, "maximum": 5},
        "leakage": {"type": "boolean"},
        "leakage_note": {"type": "string"},
        "comments_he": {"type": "string"},
    },
    "required": ["grounding", "completeness", "consistency", "falsifiability", "risk_balance",
                 "leakage", "leakage_note", "comments_he"],
}

SYSTEM = """You are the head of research reviewing an analyst's investment thesis. You see the data
the analyst was given (summarized) and the thesis. Score strictly, 1 (poor) to 5 (excellent):

- grounding: every material claim is supported by the provided data; cited source_ids exist and
  actually support the claim; numbers match the data. Fabricated facts or numbers => 1.
- completeness: covers the business, financial trends, competition vs the listed peers, macro
  relevance, valuation, recent filings/news, catalysts and risks — each with substance.
- consistency: stance, conviction, expected returns and probability agree with the reasoning;
  the numbers are plausible (not extreme without strong evidence).
- falsifiability: kill criteria and catalysts are concrete and measurable, with timing.
- risk_balance: the strongest counter-arguments are stated honestly, not strawmen.

leakage: true if the thesis states or relies on any event, figure or outcome dated on or after
the analysis date that is not in the provided data (the analyst must not know the future).
Explain in leakage_note. comments_he: 2–4 sentences in Hebrew on the main weaknesses.
"""


def compact_bundle(b: dict) -> dict:
    f = b.get("fundamentals") or {}
    filings = b.get("filings") or {}
    return {
        "as_of": b["as_of"], "stock": b["stock"], "benchmark": b["benchmark"],
        "price": {k: v for k, v in (b.get("price") or {}).items() if k != "monthly_closes"},
        "fundamentals": {"ttm": f.get("ttm"), "yoy_growth": f.get("yoy_growth"), "balance": f.get("balance"),
                         "last_period_end": f.get("last_period_end"), "currency": f.get("currency")},
        "valuation": b.get("valuation"),
        "competitors": b.get("competitors"),
        "macro_latest": {r: {c: {"label": s.get("label"), "date": s.get("date"), "value": s.get("value")}
                             for c, s in series.items()} for r, series in (b.get("macro") or {}).items()},
        "filings_list": filings.get("list", [])[:40],
        "documents": [{"id": d["id"], "date": d.get("date"), "form": d.get("form"),
                       "excerpt": d.get("text", "")[:1500]} for d in filings.get("documents", [])],
        "news": [{"id": n["id"], "date": n["published"][:10], "title": n["title"]} for n in b.get("news") or []],
        "data_gaps": [g["part"] for g in b.get("data_gaps") or []],
    }


def judge(gemini: Gemini, bundle: dict, thesis: dict) -> dict:
    t = {k: v for k, v in thesis.items() if k != "meta"}
    user = ("## DATA GIVEN TO THE ANALYST (summary)\n" + json.dumps(compact_bundle(bundle), ensure_ascii=False)
            + "\n\n## THESIS\n" + json.dumps(t, ensure_ascii=False)
            + f"\n\nUnknown source_ids cited (not present in the data): {thesis.get('meta', {}).get('unknown_evidence_ids')}"
            + "\n\nScore the thesis. Return JSON only.")
    r = gemini.json(cfg()["llm"]["judge_model"], SYSTEM, user, JUDGE_SCHEMA)
    dims = ("grounding", "completeness", "consistency", "falsifiability", "risk_balance")
    r["score"] = 0.0 if r.get("leakage") else (sum(r[d] for d in dims) / len(dims) - 1) / 4
    return r
