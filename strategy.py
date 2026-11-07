"""THE FILE THE RESEARCH AGENT EDITS (the train.py of this autoresearch loop).

Contract (enforced by engine/guard.py and engine/thesis.py):
    analyze(bundle, llm, previous) -> dict matching engine.schema.THESIS_SCHEMA

- bundle: point-in-time data for one stock on one date (see engine/bundle.py).
- llm: has a single method llm.json(system, user, schema) -> dict, calling the thesis model.
  At most llm.calls_left calls per stock (config: llm.max_calls_per_stock).
- previous: the previous thesis for this stock (live updates) or None.

Everything else — what data to show the model, how to summarize it, how many reasoning steps,
how to calibrate the numeric predictions — is open to change.
"""
from __future__ import annotations

import json
import math

from engine.schema import THESIS_SCHEMA

CONTEXT_BUDGET_CHARS = 180_000

SYSTEM = """You are a senior equity analyst covering US and Israeli (TASE) equities.
You write an investment thesis for ONE stock as of a given date, using ONLY the data provided
(published before that date) plus general background knowledge from before that date.
Never use knowledge of anything that happened on or after the analysis date.

Output rules:
- Write every *_he field in clear professional Hebrew. Company names, tickers and figures may stay in English.
- Ground claims in the data. In "evidence", cite claims with the exact source_id shown in the context
  (e.g. "fundamentals", "valuation", "price", "competitors", "macro", "sec:0000320193-26-000010:mdna",
  "maya:1712345", "news:3"). Do not invent source ids.
- Predictions are EXCESS returns versus the stated benchmark, as decimals (0.03 = +3%).
  A single stock's 90-day excess return has a typical absolute size of about 0.10; most forecasts
  should lie within ±0.08 and should be near 0 when there is no clear edge.
- prob_outperform_90d must be calibrated: 0.5 means no edge; values beyond 0.35–0.65 need strong evidence.
- stance must agree with the sign of expected_excess_return_90d (neutral when it is close to 0).
- kill_criteria_he: concrete, measurable conditions that would prove the thesis wrong.
- If data is missing (see DATA GAPS), say so in the relevant section instead of guessing.
"""


def _num(x, pct=False, digits=2):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "n/a"
    if pct:
        return f"{x * 100:.1f}%"
    ax = abs(x)
    if ax >= 1e9:
        return f"{x / 1e9:.2f}B"
    if ax >= 1e6:
        return f"{x / 1e6:.1f}M"
    return f"{x:.{digits}f}"


def _price(p: dict) -> str:
    if not p:
        return "n/a"
    keys_pct = ["return_1m", "return_3m", "return_6m", "return_12m", "excess_return_1m", "excess_return_3m",
                "excess_return_6m", "excess_return_12m", "benchmark_return_3m", "volatility_1y",
                "max_drawdown_1y", "pct_from_52w_high", "pct_from_52w_low"]
    lines = [f"last close {p.get('last_close_date')}: {_num(p.get('last_close'))}"]
    lines += [f"{k}: {_num(p.get(k), pct=True)}" for k in keys_pct]
    lines.append(f"avg volume 20d / 120d: {_num(p.get('avg_volume_20d'))} / {_num(p.get('avg_volume_120d'))}")
    closes = ", ".join(f"{c['date'][:7]}={c['close']}" for c in p.get("monthly_closes") or [])
    lines.append(f"monthly closes: {closes}")
    return "\n".join(lines)


def _fundamentals(f: dict) -> str:
    if not f:
        return "n/a"
    out = [f"currency: {f.get('currency')}; last period end: {f.get('last_period_end')}"]
    if f.get("visibility_rule"):
        out.append(f"note: {f['visibility_rule']}")
    for k, series in (f.get("quarters") or {}).items():
        out.append(f"{k} (quarterly): " + ", ".join(f"{d}={_num(v)}" for d, v in series))
    for k, v in (f.get("ttm") or {}).items():
        g = (f.get("yoy_growth") or {}).get(k)
        out.append(f"{k} TTM: {_num(v)}" + (f" (YoY {_num(g, pct=True)})" if g is not None else ""))
    for k, v in (f.get("yoy_growth") or {}).items():
        if k.endswith("_annual"):
            out.append(f"{k} growth: {_num(v, pct=True)}")
    for k, v in (f.get("balance") or {}).items():
        out.append(f"{k}: {_num(v['value'])} (as of {v['as_of']})")
    out.append(f"shares outstanding: {_num(f.get('shares_outstanding'))}")
    return "\n".join(out)


def _valuation(v: dict) -> str:
    if not v:
        return "n/a"
    pct_keys = {"fcf_yield", "gross_profit_margin_ttm", "operating_income_margin_ttm", "net_income_margin_ttm"}
    return "\n".join(f"{k}: {_num(x, pct=k in pct_keys)}" for k, x in v.items())


def _competitors(c: dict) -> str:
    if not c or not c.get("peers"):
        return "n/a"
    rows = [f"industry: {c.get('industry')}"]
    for p in c["peers"]:
        rows.append(f"- {p['name']} ({p['key']}): 3m {_num(p.get('return_3m'), True)}, 12m {_num(p.get('return_12m'), True)}, "
                    f"excess 3m {_num(p.get('excess_return_3m'), True)}, rev growth {_num(p.get('revenue_growth_yoy'), True)}, "
                    f"P/E {_num(p.get('pe_ttm'))}, P/S {_num(p.get('ps_ttm'))}, EV/EBIT {_num(p.get('ev_ebit_ttm'))}, "
                    f"FCF yield {_num(p.get('fcf_yield'), True)}, op margin {_num(p.get('operating_income_margin_ttm'), True)}, "
                    f"mcap {_num(p.get('market_cap'))}")
    return "\n".join(rows)


def _macro(m: dict) -> str:
    if not m:
        return "n/a"
    rows = []
    for region, series in m.items():
        for code, s in series.items():
            if not s.get("date"):
                continue
            ch = ", ".join(f"Δ{k[7:]} {_num(s[k])}" for k in ("change_1m", "change_3m", "change_12m") if k in s)
            rows.append(f"[{region}] {s.get('label', code)} ({code}) {s['date']}: {_num(s['value'])}  {ch}")
    return "\n".join(rows)


def _docs(bundle: dict, budget: int) -> str:
    docs = list((bundle.get("filings") or {}).get("documents") or [])
    if not docs:
        return "none"
    order = {"mdna": 0, "8-K": 1, "business": 2, "risk": 3}
    docs.sort(key=lambda d: (order.get(d.get("section"), 1), d.get("date", "")), reverse=False)
    per = max(2000, budget // max(1, len(docs)))
    parts = []
    for d in docs:
        head = f"[source_id={d['id']}] {d.get('date')} {d.get('form', '')} {d.get('section') or d.get('title', '')}"
        parts.append(head + "\n" + d.get("text", "")[:per])
    return "\n\n".join(parts)


def render_context(bundle: dict, previous: dict | None) -> str:
    s = bundle["stock"]
    filings_list = (bundle.get("filings") or {}).get("list") or []
    fl = "\n".join(f"- [{x['id']}] {x['date']} {x.get('form', '')} {x.get('title') or x.get('description') or ''} "
                   f"{' '.join(x.get('items_8k') or [])}" for x in filings_list[:40]) or "none"
    news = "\n".join(f"- [{n['id']}] {n['published'][:10]} {n.get('source', '')}: {n['title']}"
                     for n in bundle.get("news") or []) or "none"
    prof = bundle.get("profile") or {}
    head = [
        f"ANALYSIS DATE (as_of): {bundle['as_of']} — data below was published before this date.",
        f"STOCK: {s.get('name')} {('/ ' + s['name_he']) if s.get('name_he') else ''} ({s['key']}), market {s['market']}",
        f"BENCHMARK: {bundle['benchmark']}",
        f"\n## PROFILE [source_id=profile]\n{json.dumps(prof, ensure_ascii=False)}",
        f"\n## PRICE [source_id=price]\n{_price(bundle.get('price') or {})}",
        f"\n## FUNDAMENTALS [source_id=fundamentals]\n{_fundamentals(bundle.get('fundamentals') or {})}",
        f"\n## VALUATION [source_id=valuation]\n{_valuation(bundle.get('valuation') or {})}",
        f"\n## COMPETITORS [source_id=competitors]\n{_competitors(bundle.get('competitors') or {})}",
        f"\n## MACRO [source_id=macro]\n{_macro(bundle.get('macro') or {})}",
        f"\n## FILINGS / REPORTS LIST\n{fl}",
        f"\n## NEWS HEADLINES\n{news}",
        f"\n## DATA GAPS\n{json.dumps([g['part'] + ': ' + g['error'][:200] for g in bundle.get('data_gaps') or []], ensure_ascii=False)}",
    ]
    if previous:
        prev = {k: previous.get(k) for k in ("as_of", "summary_he", "stance", "conviction",
                                             "expected_excess_return_90d", "prob_outperform_90d", "kill_criteria_he")}
        head.append(f"\n## PREVIOUS THESIS\n{json.dumps(prev, ensure_ascii=False)}")
    text = "\n".join(head)
    remaining = CONTEXT_BUDGET_CHARS - len(text) - 2000
    text += f"\n\n## FILING / REPORT DOCUMENTS (excerpts)\n{_docs(bundle, max(10000, remaining))}"
    return text[:CONTEXT_BUDGET_CHARS]


def analyze(bundle, llm, previous):
    context = render_context(bundle, previous)
    user = (context + "\n\n## TASK\nWrite the investment thesis for this stock as of the analysis date, "
            "following the output rules. Return JSON only.")
    thesis = llm.json(SYSTEM, user, THESIS_SCHEMA)
    # Keep stance consistent with the 90-day forecast.
    e90 = thesis.get("expected_excess_return_90d", 0.0)
    if abs(e90) < 0.01:
        thesis["stance"] = "neutral"
    elif e90 > 0 and thesis.get("stance") == "short":
        thesis["stance"] = "neutral"
    elif e90 < 0 and thesis.get("stance") == "long":
        thesis["stance"] = "neutral"
    return thesis
