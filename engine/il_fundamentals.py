"""Fundamentals for TASE stocks from Yahoo Finance statements, made point-in-time with the
Israeli reporting deadlines (Securities Regulations (Periodic and Immediate Reports), 1970):
quarterly reports within 60 days of quarter end, annual reports within 90 days of year end.
A quarter is treated as visible at as_of only once its deadline has passed."""
from __future__ import annotations

import math
from datetime import date, timedelta

import yfinance as yf

from .http import DiskCache

_cache = DiskCache("yf_statements")

QUARTER_LAG = 60
ANNUAL_LAG = 90

FLOW_ROWS = {
    "revenue": ["Total Revenue", "Operating Revenue"],
    "gross_profit": ["Gross Profit"],
    "operating_income": ["Operating Income", "Total Operating Income As Reported"],
    "net_income": ["Net Income Common Stockholders", "Net Income"],
    "eps_diluted": ["Diluted EPS"],
    "operating_cash_flow": ["Operating Cash Flow"],
    "capex": ["Capital Expenditure"],
    "dividends": ["Cash Dividends Paid"],
    "buybacks": ["Repurchase Of Capital Stock"],
}
BALANCE_ROWS = {
    "cash": ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"],
    "total_assets": ["Total Assets"],
    "total_liabilities": ["Total Liabilities Net Minority Interest"],
    "equity": ["Stockholders Equity", "Common Stock Equity"],
    "long_term_debt": ["Long Term Debt"],
    "short_term_debt": ["Current Debt"],
}


def _frame_to_dict(df) -> dict:
    if df is None or df.empty:
        return {}
    out = {}
    for row in df.index:
        vals = {}
        for col in df.columns:
            v = df.at[row, col]
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                vals[col.date().isoformat()] = float(v)
        out[str(row)] = vals
    return out


def statements(yf_symbol: str, ttl: float = 86400) -> dict:
    hit = _cache.get(yf_symbol, ttl)
    if hit is not None:
        return hit
    t = yf.Ticker(yf_symbol)
    info = t.info or {}
    rec = {
        "q_income": _frame_to_dict(t.quarterly_income_stmt),
        "q_balance": _frame_to_dict(t.quarterly_balance_sheet),
        "q_cashflow": _frame_to_dict(t.quarterly_cashflow),
        "a_income": _frame_to_dict(t.income_stmt),
        "financial_currency": info.get("financialCurrency"),
        "price_currency": info.get("currency"),
        "sector": info.get("sector"), "industry": info.get("industry"),
    }
    _cache.put(yf_symbol, rec)
    return rec


def _visible_end(end: str, as_of: date, annual: bool = False) -> bool:
    e = date.fromisoformat(end)
    lag = ANNUAL_LAG if (annual or e.month == 12) else QUARTER_LAG
    return e + timedelta(days=lag) < as_of


def _row(table: dict, names: list[str]) -> dict:
    for n in names:
        if table.get(n):
            return table[n]
    return {}


def fundamentals_pit(yf_symbol: str, as_of: date) -> dict:
    st = statements(yf_symbol)
    out: dict = {"currency": st.get("financial_currency"), "price_currency": st.get("price_currency"),
                 "quarters": {}, "ttm": {}, "ttm_prior": {}, "yoy_growth": {}, "balance": {},
                 "shares_outstanding": None, "last_period_end": None,
                 "visibility_rule": "רבעון נחשב ידוע 60 יום אחרי סופו, דוח שנתי 90 יום אחרי סוף השנה"}
    for key, names in FLOW_ROWS.items():
        merged = {}
        for tbl in ("q_income", "q_cashflow"):
            merged.update(_row(st.get(tbl) or {}, names))
        q = sorted((d, v) for d, v in merged.items() if _visible_end(d, as_of))
        if not q:
            continue
        out["quarters"][key] = q[-8:]
        last4 = q[-4:]
        if len(last4) == 4 and (date.fromisoformat(last4[-1][0]) - date.fromisoformat(last4[0][0])).days <= 290:
            out["ttm"][key] = sum(v for _, v in last4)
        out["last_period_end"] = max(out["last_period_end"] or "", q[-1][0])
    annual = _row(st.get("a_income") or {}, FLOW_ROWS["revenue"])
    years = sorted((d, v) for d, v in annual.items() if _visible_end(d, as_of, annual=True))
    if len(years) >= 2 and years[-2][1]:
        out["yoy_growth"]["revenue_annual"] = years[-1][1] / abs(years[-2][1]) - 1
        out["annual_revenue"] = years[-3:]
    ni_annual = _row(st.get("a_income") or {}, FLOW_ROWS["net_income"])
    ni_years = sorted((d, v) for d, v in ni_annual.items() if _visible_end(d, as_of, annual=True))
    if len(ni_years) >= 2 and ni_years[-2][1]:
        out["yoy_growth"]["net_income_annual"] = ni_years[-1][1] / abs(ni_years[-2][1]) - 1
    for key, names in BALANCE_ROWS.items():
        vals = sorted((d, v) for d, v in _row(st.get("q_balance") or {}, names).items() if _visible_end(d, as_of))
        if vals:
            out["balance"][key] = {"value": vals[-1][1], "as_of": vals[-1][0]}
    shares = sorted((d, v) for d, v in _row(st.get("q_balance") or {}, ["Ordinary Shares Number", "Share Issued"]).items()
                    if _visible_end(d, as_of))
    if shares:
        out["shares_outstanding"] = shares[-1][1]
    return out
