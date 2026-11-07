"""The stock universe: Russell 1000 (iShares IWB holdings) + TA-125 (TASE index components)."""
from __future__ import annotations

import csv
import io
import json
import re
from datetime import datetime, timezone

from .config import STATE, cfg
from .http import BrowserSession, make_session
from . import sec

ISHARES_IWB = ("https://www.ishares.com/us/products/239707/ishares-russell-1000-etf/"
               "1467271812596.ajax?fileType=csv&fileName=IWB_holdings&dataType=fund")
TASE_API = "https://api.tase.co.il/api"
TASE_SEED = "https://market.tase.co.il/he/market_data/indices"
TASE_ORIGIN = "https://market.tase.co.il"
UNIVERSE_FILE = STATE / "universe.json"


def parse_ishares_csv(text: str) -> list[dict]:
    lines = text.lstrip("﻿").splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("Ticker,"))
    rows = []
    for row in csv.DictReader(io.StringIO("\n".join(lines[start:]))):
        ticker = (row.get("Ticker") or "").strip()
        if not ticker or ticker == "-" or (row.get("Asset Class") or "").strip() != "Equity":
            continue
        weight = float((row.get("Weight (%)") or "0").replace(",", "") or 0)
        rows.append({"ticker": ticker, "name": (row.get("Name") or "").strip(),
                     "sector": (row.get("Sector") or "").strip(), "weight": weight,
                     "exchange": (row.get("Exchange") or "").strip()})
    return rows


def russell1000() -> list[dict]:
    s = make_session("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
                     "Chrome/140.0 Safari/537.36")
    r = s.get(ISHARES_IWB, timeout=60)
    r.raise_for_status()
    holdings = parse_ishares_csv(r.text)
    if len(holdings) < 900:
        raise RuntimeError(f"iShares IWB returned only {len(holdings)} equity holdings")
    sec_map = sec.ticker_map()  # normalized ticker -> {cik, ticker, name, exchange}
    out = []
    for h in holdings:
        norm = re.sub(r"[^A-Z0-9]", "", h["ticker"].upper())
        m = sec_map.get(norm)
        if not m:
            continue  # not an SEC registrant under that symbol (rare: delisted between files)
        out.append({
            "key": f"US:{m['ticker']}", "market": "US", "symbol": m["ticker"],
            "yf": m["ticker"].replace(".", "-"), "name": h["name"], "name_he": "",
            "sector": h["sector"], "cik": m["cik"], "weight": h["weight"],
        })
    return out


def _tase_index_id(s: BrowserSession, wanted: str = "TA-125") -> str:
    page, items, total = 1, [], None
    while total is None or len(items) < total:
        d = s.request("POST", f"{TASE_API}/index/indiceslistmarketdata",
                      json_body={"dType": 1, "oId": 0, "TotalRec": 200, "pageNum": page, "lang": "1"})
        batch = (d or {}).get("Items") or []
        total = int((d or {}).get("TotalRec") or 0)
        items += batch
        if not batch:
            break
        page += 1
    for it in items:
        if re.sub(r"\s", "", str(it.get("Name", ""))).upper() == wanted.replace(" ", "").upper():
            return str(it["Id"])
    raise RuntimeError(f"Index {wanted} not found in TASE indices list ({len(items)} indices)")


def _tase_components(s: BrowserSession, index_id: str, lang: str) -> list[dict]:
    out, page = [], 1
    while True:
        d = s.request("POST", f"{TASE_API}/index/components",
                      json_body={"oId": index_id, "pageNum": page, "lang": lang})
        batch = (d or {}).get("Items") or []
        out += batch
        if len(batch) < 30:
            return out
        page += 1


def ta125() -> list[dict]:
    s = BrowserSession(TASE_SEED, TASE_ORIGIN)
    idx = _tase_index_id(s)
    en = _tase_components(s, idx, "1")
    he = {str(x.get("CompanyId")): x.get("ShortName", "") for x in _tase_components(s, idx, "0")}
    if not 110 <= len(en) <= 140:
        raise RuntimeError(f"TA-125 components returned {len(en)} rows")
    out = []
    for c in en:
        symbol = re.sub(r"\s+", "", str(c.get("Symbol") or "")).upper()
        if not symbol:
            continue
        cid = str(c.get("CompanyId"))
        out.append({
            "key": f"IL:{symbol}", "market": "IL", "symbol": symbol, "yf": f"{symbol}.TA",
            "name": c.get("ShortName", ""), "name_he": he.get(cid, ""), "sector": "",
            "maya_company_id": int(c["CompanyId"]), "weight": float(c.get("Weight") or 0),
        })
    return out


def load(refresh: bool = False) -> list[dict]:
    days = cfg()["universe"]["refresh_days"]
    if UNIVERSE_FILE.exists() and not refresh:
        data = json.loads(UNIVERSE_FILE.read_text(encoding="utf-8"))
        age = datetime.now(timezone.utc) - datetime.fromisoformat(data["updated"])
        if age.days < days:
            return data["stocks"]
    stocks = russell1000() + ta125()
    UNIVERSE_FILE.parent.mkdir(parents=True, exist_ok=True)
    UNIVERSE_FILE.write_text(json.dumps({"updated": datetime.now(timezone.utc).isoformat(),
                                         "stocks": stocks}, ensure_ascii=False, indent=1),
                             encoding="utf-8")
    return stocks
