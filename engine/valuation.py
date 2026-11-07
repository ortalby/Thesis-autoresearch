"""Valuation multiples at as_of from point-in-time fundamentals and the last close before as_of."""
from __future__ import annotations

from datetime import date

from . import macro


def _usd_ils(as_of: date) -> float | None:
    pts = [v for d, v in macro.boi("EXR", "RER_USD_ILS") if d < as_of.isoformat()]
    return pts[-1] if pts else None


def price_in_financial_currency(price: float, price_currency: str | None, fin_currency: str | None,
                                as_of: date) -> float | None:
    p = price
    pc = (price_currency or "").upper()
    fc = (fin_currency or "").upper()
    if pc == "ILA":  # Tel Aviv quotes in agorot
        p, pc = p / 100.0, "ILS"
    if not fc or pc == fc:
        return p
    rate = _usd_ils(as_of)
    if rate is None:
        return None
    if pc == "ILS" and fc == "USD":
        return p / rate
    if pc == "USD" and fc == "ILS":
        return p * rate
    return None  # other currency pairs are not converted


def multiples(f: dict, last_close: float | None, as_of: date, price_currency: str | None = "USD") -> dict:
    out: dict = {}
    shares = f.get("shares_outstanding")
    if not last_close or not shares:
        return out
    p = price_in_financial_currency(last_close, price_currency, f.get("currency"), as_of)
    if p is None:
        return out
    mcap = p * shares
    ttm = f.get("ttm") or {}
    bal = f.get("balance") or {}
    debt = sum((bal.get(k) or {}).get("value", 0.0) for k in ("long_term_debt", "short_term_debt"))
    cash = (bal.get("cash") or {}).get("value", 0.0)
    ev = mcap + debt - cash
    out["market_cap"] = mcap
    out["enterprise_value"] = ev
    if ttm.get("net_income") and ttm["net_income"] > 0:
        out["pe_ttm"] = mcap / ttm["net_income"]
    if ttm.get("revenue"):
        out["ps_ttm"] = mcap / ttm["revenue"]
        out["ev_sales_ttm"] = ev / ttm["revenue"]
    if ttm.get("operating_income") and ttm["operating_income"] > 0:
        out["ev_ebit_ttm"] = ev / ttm["operating_income"]
    if ttm.get("operating_cash_flow") is not None:
        capex = ttm.get("capex") or 0.0
        fcf = ttm["operating_cash_flow"] - abs(capex)
        out["fcf_ttm"] = fcf
        out["fcf_yield"] = fcf / mcap if mcap else None
    if (bal.get("equity") or {}).get("value"):
        out["pb"] = mcap / bal["equity"]["value"]
    if ttm.get("revenue"):
        for k in ("gross_profit", "operating_income", "net_income"):
            if ttm.get(k) is not None:
                out[f"{k}_margin_ttm"] = ttm[k] / ttm["revenue"]
    out["net_debt"] = debt - cash
    return out
