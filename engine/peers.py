"""Competitors: same SIC industry (US, from EDGAR) or same TASE branch (IL, from MAYA), within the
universe, with point-in-time price performance and valuation for comparison."""
from __future__ import annotations

from datetime import date
from functools import lru_cache

from . import il_fundamentals, maya, prices, sec, valuation

MAX_PEERS = 6


@lru_cache(maxsize=512)
def _il_industry(company_id: int) -> tuple[str, str]:
    p = maya.profile(company_id)
    return p.get("sub_branch") or p.get("branch") or "", p.get("branch") or ""


def _industry(stock: dict) -> tuple[str, str]:
    if stock["market"] == "US":
        return sec.sic_of(stock["cik"])
    return _il_industry(stock["maya_company_id"])


def peer_candidates(stock: dict, universe: list[dict]) -> tuple[list[dict], str]:
    same_market = [s for s in universe if s["market"] == stock["market"] and s["key"] != stock["key"]]
    code, label = _industry(stock)
    if not code:
        return [], ""
    if stock["market"] == "US":
        for digits in (4, 3, 2):
            group = [s for s in same_market if sec.sic_of(s["cik"])[0][:digits] == code[:digits]]
            if len(group) >= 3 or digits == 2:
                group.sort(key=lambda s: -s.get("weight", 0))
                return group[:MAX_PEERS], f"SIC {code[:digits]} — {label}"
    group = [s for s in same_market if _industry(s)[0] == code]
    if len(group) < 3:
        branch = _industry(stock)[1]
        group = [s for s in same_market if _industry(s)[1] == branch]
        code = branch
    group.sort(key=lambda s: -s.get("weight", 0))
    return group[:MAX_PEERS], code


def peer_table(stock: dict, universe: list[dict], as_of: date, bench: dict) -> dict:
    group, industry = peer_candidates(stock, universe)
    rows = []
    price_recs = prices.fetch([p["yf"] for p in group])
    for p in group:
        rec = price_recs.get(p["yf"])
        if not rec:
            continue
        feats = prices.features(rec, bench, as_of)
        if p["market"] == "US":
            f = sec.fundamentals_pit(sec.companyfacts(p["cik"]), as_of)
            pc = "USD"
        else:
            f = il_fundamentals.fundamentals_pit(p["yf"], as_of)
            pc = f.get("price_currency")
        mult = valuation.multiples(f, feats.get("last_close"), as_of, pc)
        rows.append({
            "key": p["key"], "name": p["name"],
            "return_3m": feats.get("return_3m"), "return_12m": feats.get("return_12m"),
            "excess_return_3m": feats.get("excess_return_3m"),
            "revenue_growth_yoy": (f.get("yoy_growth") or {}).get("revenue")
            or (f.get("yoy_growth") or {}).get("revenue_annual"),
            **{k: mult.get(k) for k in ("market_cap", "pe_ttm", "ps_ttm", "ev_ebit_ttm", "fcf_yield",
                                        "operating_income_margin_ttm", "gross_profit_margin_ttm")},
        })
    return {"industry": industry, "peers": rows}
