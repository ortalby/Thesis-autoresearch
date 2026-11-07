"""The fixed output contract for an investment thesis. strategy.py may change how a thesis is
produced, never what a thesis is: the harness validates every thesis against THESIS_SCHEMA."""
from __future__ import annotations

import jsonschema

_STR = {"type": "string"}

THESIS_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "summary_he": {"type": "string", "description": "תזת ההשקעה בשניים-שלושה משפטים"},
        "stance": {"type": "string", "enum": ["long", "neutral", "short"]},
        "conviction": {"type": "integer", "minimum": 1, "maximum": 5},
        "expected_excess_return_30d": {"type": "number", "minimum": -0.6, "maximum": 0.6,
                                       "description": "תשואה עודפת צפויה מול מדד הייחוס ל-30 יום, כשבר עשרוני"},
        "expected_excess_return_90d": {"type": "number", "minimum": -0.9, "maximum": 0.9,
                                       "description": "תשואה עודפת צפויה מול מדד הייחוס ל-90 יום, כשבר עשרוני"},
        "prob_outperform_90d": {"type": "number", "minimum": 0.0, "maximum": 1.0,
                                "description": "הסתברות שהמניה תכה את מדד הייחוס ב-90 הימים הבאים"},
        "business_he": _STR,
        "financials_he": _STR,
        "competition_he": _STR,
        "macro_he": _STR,
        "valuation_he": _STR,
        "news_and_filings_he": _STR,
        "catalysts": {"type": "array", "items": {
            "type": "object",
            "properties": {"event_he": _STR, "expected_date": _STR,
                           "impact": {"type": "string", "enum": ["positive", "negative", "uncertain"]}},
            "required": ["event_he", "expected_date", "impact"]}},
        "risks": {"type": "array", "items": {
            "type": "object",
            "properties": {"risk_he": _STR,
                           "probability": {"type": "string", "enum": ["low", "medium", "high"]},
                           "impact_he": _STR},
            "required": ["risk_he", "probability", "impact_he"]}},
        "kill_criteria_he": {"type": "array", "items": _STR,
                             "description": "מה יפריך את התזה — תנאים מדידים"},
        "changes_vs_previous_he": {"type": "string",
                                   "description": "מה השתנה לעומת התזה הקודמת ולמה; ריק אם אין תזה קודמת"},
        "evidence": {"type": "array", "items": {
            "type": "object",
            "properties": {"claim_he": _STR, "source_id": _STR},
            "required": ["claim_he", "source_id"]}},
    },
    "required": ["summary_he", "stance", "conviction", "expected_excess_return_30d",
                 "expected_excess_return_90d", "prob_outperform_90d", "business_he", "financials_he",
                 "competition_he", "macro_he", "valuation_he", "news_and_filings_he", "catalysts",
                 "risks", "kill_criteria_he", "changes_vs_previous_he", "evidence"],
}

_validator = jsonschema.Draft202012Validator(THESIS_SCHEMA)


class InvalidThesis(ValueError):
    pass


def validate(thesis: dict) -> None:
    errs = sorted(_validator.iter_errors(thesis), key=lambda e: list(e.path))
    if errs:
        raise InvalidThesis("; ".join(f"{'/'.join(map(str, e.path)) or '<root>'}: {e.message}" for e in errs[:8]))
    if thesis["stance"] == "long" and thesis["expected_excess_return_90d"] < 0:
        raise InvalidThesis("stance=long with negative expected 90d excess return")
    if thesis["stance"] == "short" and thesis["expected_excess_return_90d"] > 0:
        raise InvalidThesis("stance=short with positive expected 90d excess return")
