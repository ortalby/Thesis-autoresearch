"""Builds the point-in-time data bundle for one stock on one date: everything an analyst could have
known before that date — prices, fundamentals, valuation, filings, news, competitors and macro.

Any data source that fails is recorded in bundle["data_gaps"] with the error, so the thesis (and
the judge) can see what is missing rather than having it silently replaced."""
from __future__ import annotations

import traceback
from datetime import date
from functools import lru_cache

from .config import cfg
from . import il_fundamentals, macro, maya, news, peers, prices, sec, valuation


@lru_cache(maxsize=64)
def _macro(as_of: date) -> dict:
    return macro.snapshot(as_of)


def _gap(bundle: dict, part: str, exc: Exception) -> None:
    bundle["data_gaps"].append({"part": part, "error": f"{type(exc).__name__}: {exc}",
                                "trace": traceback.format_exc(limit=2)[-600:]})


def build(stock: dict, universe: list[dict], as_of: date, live: bool) -> dict:
    """live=True for today's update (current-state profile data allowed);
    live=False for historical evaluation (only data with a publication date before as_of)."""
    u = cfg()["update"]
    bench_sym = cfg()["benchmarks"][stock["market"]]
    b: dict = {"stock": {k: stock.get(k) for k in ("key", "market", "symbol", "name", "name_he", "sector")},
               "as_of": as_of.isoformat(), "benchmark": bench_sym, "live": live, "data_gaps": []}

    recs = prices.fetch([stock["yf"], bench_sym])
    rec, bench = recs.get(stock["yf"]), recs.get(bench_sym)
    if not rec or not bench:
        raise RuntimeError(f"No price history for {stock['yf']} or benchmark {bench_sym}")
    b["price"] = prices.features(rec, bench, as_of)
    if not b["price"]:
        raise RuntimeError(f"Not enough price history before {as_of} for {stock['yf']}")

    try:
        if stock["market"] == "US":
            f = sec.fundamentals_pit(sec.companyfacts(stock["cik"]), as_of)
            price_ccy = "USD"
        else:
            f = il_fundamentals.fundamentals_pit(stock["yf"], as_of)
            price_ccy = f.get("price_currency")
        b["fundamentals"] = f
        b["valuation"] = valuation.multiples(f, b["price"].get("last_close"), as_of, price_ccy)
    except Exception as exc:
        _gap(b, "fundamentals", exc)

    try:
        if stock["market"] == "US":
            sic, sic_desc = sec.sic_of(stock["cik"])
            b["profile"] = {"sic": sic, "industry": sic_desc}
            b["filings"] = sec.recent_filings_bundle(
                stock["cik"], as_of, u["filings_lookback_days"],
                {"business": 15000, "risk": 15000, "mdna": 25000, "8-K": 12000})
        else:
            prof = maya.profile(stock["maya_company_id"])
            b["profile"] = {"industry": " · ".join(x for x in (prof["branch"], prof["sub_branch"]) if x)}
            if live:
                # The activity text and market value are current-state fields without a
                # publication date, so they are used only for today's live thesis.
                b["profile"]["activity"] = prof["activity"]
                b["profile"]["market_value_k_ils"] = prof["market_value_k_ils"]
            b["filings"] = maya.company_reports_bundle(stock["maya_company_id"], as_of,
                                                       u["filings_lookback_days"])
    except Exception as exc:
        _gap(b, "filings", exc)

    try:
        b["news"] = news.headlines(stock, as_of, u["news_lookback_days"])
    except Exception as exc:
        _gap(b, "news", exc)

    try:
        b["competitors"] = peers.peer_table(stock, universe, as_of, bench)
    except Exception as exc:
        _gap(b, "competitors", exc)

    try:
        b["macro"] = _macro(as_of)
    except Exception as exc:
        _gap(b, "macro", exc)

    return b


def source_ids(bundle: dict) -> set[str]:
    ids = {"price", "fundamentals", "valuation", "macro", "competitors", "profile"}
    for part in ("list", "documents"):
        for x in (bundle.get("filings") or {}).get(part) or []:
            ids.add(x["id"])
    for x in bundle.get("news") or []:
        ids.add(x["id"])
    return ids
