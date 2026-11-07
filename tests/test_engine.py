"""Offline tests of the parsing, point-in-time and scoring logic (no network, no API keys)."""
from __future__ import annotations

import json
from datetime import date, timedelta

import pytest

from engine import guard, metrics, prices, sec, universe
from engine.evalset import eval_dates
from engine.il_fundamentals import _visible_end
from engine.schema import InvalidThesis, validate


def _fact(start, end, val, filed, form="10-Q"):
    return {"start": start, "end": end, "val": val, "filed": filed, "form": form}


def test_quarterly_series_from_ytd_and_annual():
    entries = [
        _fact("2025-01-01", "2025-03-31", 100, "2025-05-01"),
        _fact("2025-01-01", "2025-06-30", 220, "2025-08-01"),     # YTD 6M -> Q2 = 120
        _fact("2025-01-01", "2025-09-30", 360, "2025-11-01"),     # YTD 9M -> Q3 = 140
        _fact("2025-01-01", "2025-12-31", 520, "2026-02-15", "10-K"),  # FY -> Q4 = 160
    ]
    q = sec.quarterly_series(entries, date(2026, 3, 1))
    assert q == [("2025-03-31", 100.0), ("2025-06-30", 120.0), ("2025-09-30", 140.0), ("2025-12-31", 160.0)]
    # Point in time: before the 10-K was filed, Q4 is unknown.
    q_before = sec.quarterly_series(entries, date(2026, 2, 15))
    assert q_before[-1][0] == "2025-09-30"


def test_visible_keeps_latest_filing_before_as_of():
    entries = [_fact("2025-01-01", "2025-03-31", 100, "2025-05-01"),
               _fact("2025-01-01", "2025-03-31", 90, "2026-05-01")]  # restated later
    assert sec.quarterly_series(entries, date(2026, 1, 1)) == [("2025-03-31", 100.0)]
    assert sec.quarterly_series(entries, date(2026, 6, 1)) == [("2025-03-31", 90.0)]


def test_fundamentals_pit_ttm_and_shares():
    def q(start, end, val, filed):
        return {"start": start, "end": end, "val": val, "filed": filed, "form": "10-Q"}

    rev = [q("2024-01-01", "2024-03-31", 80, "2024-05-01"), q("2024-04-01", "2024-06-30", 80, "2024-08-01"),
           q("2024-07-01", "2024-09-30", 80, "2024-11-01"), q("2024-10-01", "2024-12-31", 80, "2025-02-10"),
           q("2025-01-01", "2025-03-31", 100, "2025-05-01"), q("2025-04-01", "2025-06-30", 100, "2025-08-01"),
           q("2025-07-01", "2025-09-30", 100, "2025-11-01"), q("2025-10-01", "2025-12-31", 100, "2026-02-10")]
    facts = {"facts": {
        "us-gaap": {"Revenues": {"units": {"USD": rev}},
                    "Assets": {"units": {"USD": [{"end": "2025-12-31", "val": 999, "filed": "2026-02-10"}]}}},
        "dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": [
            {"end": "2026-01-31", "val": 600, "filed": "2026-02-10"},
            {"end": "2026-01-31", "val": 400, "filed": "2026-02-10"}]}}}}}
    f = sec.fundamentals_pit(facts, date(2026, 3, 1))
    assert f["ttm"]["revenue"] == 400
    assert f["ttm_prior"]["revenue"] == 320
    assert f["yoy_growth"]["revenue"] == pytest.approx(0.25)
    assert f["balance"]["total_assets"]["value"] == 999
    assert f["shares_outstanding"] == 1000  # two share classes summed


def test_extract_section_skips_table_of_contents():
    text = ("TABLE OF CONTENTS\nItem 7. Management's Discussion 45\nItem 7A. Quantitative 60\n"
            "PART II\nItem 7. Management's Discussion and Analysis\n" + "Revenue grew. " * 200 +
            "\nItem 7A. Quantitative and Qualitative Disclosures\nrates")
    s = sec.extract_section(text, "10-K", "mdna")
    assert s.startswith("Item 7. Management's Discussion and Analysis")
    assert "Revenue grew" in s and "Item 7A" not in s


def test_parse_ishares_csv():
    csv_text = ("﻿iShares Russell 1000 ETF\nFund Holdings as of,\"Nov 06, 2026\"\n\n"
                "Ticker,Name,Sector,Asset Class,Market Value,Weight (%),Notional Value,Quantity,Price,Location,Exchange,Currency,FX Rate,Market Currency,Accrual Date\n"
                "\"AAPL\",\"APPLE INC\",\"Information Technology\",\"Equity\",\"1,000\",\"6.50\",\"1,000\",\"10\",\"100\",\"United States\",\"NASDAQ\",\"USD\",\"1.00\",\"USD\",\"-\"\n"
                "\"BRKB\",\"BERKSHIRE HATHAWAY INC CLASS B\",\"Financials\",\"Equity\",\"500\",\"1.60\",\"500\",\"1\",\"500\",\"United States\",\"New York Stock Exchange Inc.\",\"USD\",\"1.00\",\"USD\",\"-\"\n"
                "\"USD\",\"USD CASH\",\"Cash and/or Derivatives\",\"Cash\",\"5\",\"0.01\",\"5\",\"5\",\"1\",\"United States\",\"-\",\"USD\",\"1.00\",\"USD\",\"-\"\n")
    rows = universe.parse_ishares_csv(csv_text)
    assert [r["ticker"] for r in rows] == ["AAPL", "BRKB"]
    assert rows[0]["weight"] == 6.5


def _series(n=400, start=date(2025, 1, 1), drift=0.001):
    dates, close, d, p = [], [], start, 100.0
    while len(dates) < n:
        if d.weekday() < 5:
            dates.append(d.isoformat())
            close.append(p)
            p *= 1 + drift
        d += timedelta(days=1)
    return {"dates": dates, "close": close, "volume": [1000.0] * n}


def test_prices_point_in_time_and_forward():
    rec = _series()
    as_of = date.fromisoformat(rec["dates"][300])
    last = prices.close_before(rec, as_of)
    assert last[0] == rec["dates"][299]  # strictly before as_of
    fr = prices.forward_return(rec, as_of, 30)
    assert fr is not None and fr > 0
    f = prices.features(rec, _series(drift=0.0), as_of)
    assert f["return_1m"] == pytest.approx((1.001 ** 21) - 1, rel=1e-6)
    assert f["excess_return_1m"] == pytest.approx(f["return_1m"])


def test_il_visibility_rule():
    assert not _visible_end("2026-03-31", date(2026, 5, 30))
    assert _visible_end("2026-03-31", date(2026, 6, 1))
    assert not _visible_end("2025-12-31", date(2026, 3, 30))  # annual: 90 days
    assert _visible_end("2025-12-31", date(2026, 4, 1))


def test_guard_allows_baseline_and_blocks_escapes():
    from engine.config import ROOT

    guard.check((ROOT / "strategy.py").read_text(encoding="utf-8"))
    bad = ["import os\ndef analyze(bundle, llm, previous):\n    return {}",
           "def analyze(bundle, llm, previous):\n    return open('x').read()",
           "def analyze(bundle, llm, previous):\n    return llm._g.text('m','s','u')",
           "import requests\ndef analyze(bundle, llm, previous):\n    pass",
           "def analyze(bundle, llm):\n    pass",
           "def helper():\n    pass"]
    for src in bad:
        with pytest.raises(guard.GuardError):
            guard.check(src)


def _thesis(**over):
    t = {"summary_he": "תזה", "stance": "long", "conviction": 3, "expected_excess_return_30d": 0.01,
         "expected_excess_return_90d": 0.03, "prob_outperform_90d": 0.56, "business_he": "א",
         "financials_he": "ב", "competition_he": "ג", "macro_he": "ד", "valuation_he": "ה",
         "news_and_filings_he": "ו",
         "catalysts": [{"event_he": "דוח רבעוני", "expected_date": "2026-11-20", "impact": "uncertain"}],
         "risks": [{"risk_he": "האטה", "probability": "medium", "impact_he": "ירידה בהכנסות"}],
         "kill_criteria_he": ["צמיחה מתחת ל-5%"], "changes_vs_previous_he": "",
         "evidence": [{"claim_he": "הכנסות עלו", "source_id": "fundamentals"}]}
    t.update(over)
    return t


def test_schema_validation():
    validate(_thesis())
    with pytest.raises(InvalidThesis):
        validate(_thesis(stance="long", expected_excess_return_90d=-0.05))
    with pytest.raises(InvalidThesis):
        validate(_thesis(prob_outperform_90d=1.4))
    bad = _thesis()
    del bad["risks"]
    with pytest.raises(InvalidThesis):
        validate(bad)


def _row(as_of, pred, real, prob=0.5, judge=0.5, stance="long"):
    return {"as_of": as_of, "pred": {"excess_30d": pred, "excess_90d": pred, "prob_90d": prob, "stance": stance},
            "realized": {"excess_30d": real, "excess_90d": real}, "judge_score": judge}


def test_metrics_perfect_and_inverted():
    w = {"rank_ic": 0.35, "brier_skill": 0.35, "judge": 0.30}
    good = [_row("2026-05-01", x / 100, x / 100, prob=0.9 if x > 0 else 0.1) for x in range(-5, 6) if x]
    m = metrics.combined(good, w)
    assert m["rank_ic"] == pytest.approx(1.0)
    assert m["brier_skill"] == pytest.approx(1 - 0.01 / 0.25)
    bad = [_row("2026-05-01", -x / 100, x / 100) for x in range(-5, 6) if x]
    assert metrics.combined(bad, w)["rank_ic"] == pytest.approx(-1.0)
    flat = [_row("2026-05-01", 0.0, x / 100) for x in range(-5, 6) if x]
    assert metrics.combined(flat, w)["rank_ic"] == 0.0
    assert metrics.combined(flat, w)["brier_skill"] == pytest.approx(0.0)


def test_eval_dates_after_cutoff_with_room_for_outcomes():
    ds = eval_dates(date(2026, 4, 1), date(2026, 11, 7), 4)
    assert ds[0] >= date(2026, 4, 1)
    assert ds[-1] <= date(2026, 11, 7) - timedelta(days=97)
    assert all(d.weekday() < 5 for d in ds)
    with pytest.raises(RuntimeError):
        eval_dates(date(2026, 10, 1), date(2026, 11, 7), 4)


def _bundle():
    return {
        "stock": {"key": "US:TEST", "market": "US", "symbol": "TEST", "name": "Test Corp", "name_he": "", "sector": "IT"},
        "as_of": "2026-06-01", "benchmark": "IWB", "live": False, "data_gaps": [],
        "price": prices.features(_series(), _series(drift=0.0), date(2026, 6, 1)),
        "fundamentals": {"currency": "USD", "quarters": {"revenue": [["2026-03-31", 100.0]]}, "ttm": {"revenue": 400.0},
                         "ttm_prior": {}, "yoy_growth": {"revenue": 0.1}, "balance": {"cash": {"value": 50.0, "as_of": "2026-03-31"}},
                         "shares_outstanding": 10.0, "last_period_end": "2026-03-31"},
        "valuation": {"market_cap": 1000.0, "ps_ttm": 2.5},
        "profile": {"sic": "7372", "industry": "Software"},
        "filings": {"list": [{"id": "sec:1", "date": "2026-05-01", "form": "10-Q", "description": "q", "items_8k": []}],
                    "documents": [{"id": "sec:1:mdna", "date": "2026-05-01", "form": "10-Q", "section": "mdna", "text": "Revenue grew 10%."}]},
        "news": [{"id": "news:0", "title": "Test Corp wins contract", "source": "Wire", "published": "2026-05-20T10:00:00+00:00", "link": ""}],
        "competitors": {"industry": "SIC 7372", "peers": [{"key": "US:PEER", "name": "Peer", "return_3m": 0.02}]},
        "macro": {"US": {"DGS10": {"label": "10y", "date": "2026-05-29", "value": 4.1, "change_1m": 0.1}}, "IL": {}},
    }


class _StubGemini:
    """Stands in for the API: returns a schema-valid thesis and records the prompt."""

    def __init__(self):
        self.prompts = []

    def json(self, model, system, user, schema):
        self.prompts.append(user)
        return _thesis(evidence=[{"claim_he": "הכנסות", "source_id": "fundamentals"},
                                 {"claim_he": "חוזה", "source_id": "news:0"},
                                 {"claim_he": "בדוי", "source_id": "sec:999"}])


def test_strategy_end_to_end_with_stub_model():
    from engine import thesis

    strat = thesis.load_strategy()
    g = _StubGemini()
    t = thesis.run(strat, thesis.strategy_version(), _bundle(), g, previous=None)
    assert t["meta"]["unknown_evidence_ids"] == ["sec:999"]
    prompt = g.prompts[0]
    for needle in ("ANALYSIS DATE (as_of): 2026-06-01", "[source_id=sec:1:mdna]", "[news:0]", "SIC 7372", "DGS10"):
        assert needle in prompt


def test_research_diagnostics_and_shape():
    import run_research

    report = {"metrics": {"score": 0.1}, "rows": [
        dict(_row("2026-05-01", 0.02, 0.05, prob=0.62), key="US:A", judge={"comments_he": "טוב"}),
        dict(_row("2026-05-01", -0.01, -0.02, prob=0.41), key="US:B", error="ValueError: x")]}
    d = run_research.diagnostics(report)
    assert "US:A" in d and "calibration" in d and "ValueError" in d
    shape = run_research._shape(_bundle())
    assert json.dumps(shape, ensure_ascii=False, default=str)
