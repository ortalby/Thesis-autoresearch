"""Daily prices from Yahoo Finance (yfinance), cached per ticker, and point-in-time price features.

Convention: an analysis "as of" day D sees closes strictly before D. A prediction made on D is
scored from the last close before D to the last close on or before D + horizon."""
from __future__ import annotations

import math
from datetime import date, timedelta
from statistics import pstdev

import pandas as pd
import yfinance as yf

from .http import DiskCache

HISTORY_START = "2023-01-01"
_cache = DiskCache("prices")
TTL = 12 * 3600


def _store(sym: str, df: pd.DataFrame) -> dict:
    df = df.dropna(subset=["Close"])
    rec = {"dates": [d.date().isoformat() for d in df.index],
           "close": [float(x) for x in df["Close"]],
           "volume": [float(x) if not math.isnan(x) else 0.0 for x in df.get("Volume", pd.Series(0, index=df.index))]}
    _cache.put(sym, rec)
    return rec


def fetch(symbols: list[str], ttl: float = TTL) -> dict[str, dict]:
    out, missing = {}, []
    for s in dict.fromkeys(symbols):
        hit = _cache.get(s, ttl)
        if hit is not None:
            out[s] = hit
        else:
            missing.append(s)
    for i in range(0, len(missing), 200):
        chunk = missing[i:i + 200]
        df = yf.download(chunk, start=HISTORY_START, auto_adjust=True, progress=False,
                         threads=True, group_by="ticker")
        for s in chunk:
            try:
                sub = df[s] if isinstance(df.columns, pd.MultiIndex) else df
            except KeyError:
                continue
            if sub is None or sub.dropna(how="all").empty:
                continue
            out[s] = _store(s, sub)
    return out


def _idx_before(dates: list[str], d: date) -> int:
    """Index of the last date strictly before d, or -1."""
    lo, hi, key = 0, len(dates), d.isoformat()
    while lo < hi:
        mid = (lo + hi) // 2
        if dates[mid] < key:
            lo = mid + 1
        else:
            hi = mid
    return lo - 1


def _idx_on_or_before(dates: list[str], d: date) -> int:
    return _idx_before(dates, d + timedelta(days=1))


def close_before(rec: dict, d: date) -> tuple[str, float] | None:
    i = _idx_before(rec["dates"], d)
    return (rec["dates"][i], rec["close"][i]) if i >= 0 else None


def forward_return(rec: dict, as_of: date, horizon_days: int) -> float | None:
    i0 = _idx_before(rec["dates"], as_of)
    i1 = _idx_on_or_before(rec["dates"], as_of + timedelta(days=horizon_days))
    if i0 < 0 or i1 <= i0:
        return None
    return rec["close"][i1] / rec["close"][i0] - 1


def _ret(c: list[float], n: int) -> float | None:
    return c[-1] / c[-1 - n] - 1 if len(c) > n and c[-1 - n] else None


def features(rec: dict, bench: dict, as_of: date) -> dict:
    i = _idx_before(rec["dates"], as_of)
    if i < 20:
        return {}
    c = rec["close"][: i + 1]
    v = rec["volume"][: i + 1]
    j = _idx_before(bench["dates"], as_of)
    b = bench["close"][: j + 1]
    daily = [c[k] / c[k - 1] - 1 for k in range(max(1, len(c) - 252), len(c)) if c[k - 1]]
    vol = pstdev(daily) * math.sqrt(252) if len(daily) > 20 else None
    peak, mdd = c[max(0, len(c) - 252)], 0.0
    for x in c[max(0, len(c) - 252):]:
        peak = max(peak, x)
        mdd = min(mdd, x / peak - 1)
    hi52, lo52 = max(c[-252:]), min(c[-252:])
    out = {
        "last_close_date": rec["dates"][i], "last_close": c[-1],
        "return_1m": _ret(c, 21), "return_3m": _ret(c, 63), "return_6m": _ret(c, 126),
        "return_12m": _ret(c, 252), "volatility_1y": vol, "max_drawdown_1y": mdd,
        "pct_from_52w_high": c[-1] / hi52 - 1, "pct_from_52w_low": c[-1] / lo52 - 1,
        "avg_volume_20d": sum(v[-20:]) / 20, "avg_volume_120d": sum(v[-120:]) / min(120, len(v)),
    }
    for n, name in ((21, "1m"), (63, "3m"), (126, "6m"), (252, "12m")):
        r, rb = out[f"return_{name}"], _ret(b, n)
        out[f"excess_return_{name}"] = (r - rb) if r is not None and rb is not None else None
    out["benchmark_return_3m"] = _ret(b, 63)
    out["monthly_closes"] = [{"date": rec["dates"][k], "close": round(rec["close"][k], 4)}
                             for k in range(max(0, i - 252), i + 1, 21)]
    return out


def daily_sigma_move(rec: dict, as_of: date) -> float | None:
    """Size of the latest daily move in units of the trailing 60-day daily volatility."""
    i = _idx_before(rec["dates"], as_of)
    if i < 62:
        return None
    c = rec["close"]
    rets = [c[k] / c[k - 1] - 1 for k in range(i - 60, i) if c[k - 1]]
    sd = pstdev(rets)
    last = c[i] / c[i - 1] - 1 if c[i - 1] else 0.0
    return abs(last) / sd if sd else None
