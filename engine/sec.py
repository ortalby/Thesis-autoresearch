"""SEC EDGAR: ticker->CIK map, filing lists, filing text and point-in-time XBRL fundamentals.

Point-in-time rule used everywhere: a filing or XBRL fact is visible at as_of only if it was
filed strictly before as_of (filed < as_of), so nothing published on the analysis day leaks in.
"""
from __future__ import annotations

import re
import threading
from datetime import date, timedelta
from functools import lru_cache

from bs4 import BeautifulSoup

from .config import cfg, secret
from .http import DiskCache, Throttle, make_session

_cache = DiskCache("sec")
_throttle: Throttle | None = None
_session = None
_init_lock = threading.Lock()

DAY = 86400

ITEM_8K = {
    "1.01": "התקשרות בהסכם מהותי", "1.02": "סיום הסכם מהותי", "1.03": "פשיטת רגל",
    "2.01": "השלמת רכישה/מכירה של נכסים", "2.02": "תוצאות כספיות", "2.03": "התחייבות פיננסית ישירה",
    "2.05": "עלויות יציאה/ארגון מחדש", "2.06": "ירידת ערך מהותית", "3.01": "הודעת מחיקה מהמסחר",
    "3.02": "מכירת מניות לא רשומה", "4.01": "החלפת רואה חשבון", "4.02": "אי-הסתמכות על דוחות קודמים",
    "5.01": "שינוי שליטה", "5.02": "שינוי בנושאי משרה/דירקטורים", "5.03": "שינוי תקנון/שנת כספים",
    "5.07": "הצבעת בעלי מניות", "7.01": "גילוי לפי Regulation FD", "8.01": "אירועים אחרים",
    "9.01": "דוחות כספיים ונספחים",
}


def _sess():
    global _session, _throttle
    with _init_lock:
        if _session is not None:
            return _session
        ua = secret(cfg()["sec"]["user_agent_env"])
        _session = make_session(ua, {"Accept": "application/json, text/html;q=0.9"})
        _throttle = Throttle(cfg()["sec"]["requests_per_second"])
        return _session


def _get_uncached(url: str, as_json: bool = True):
    s = _sess()
    _throttle.wait()
    r = s.get(url, timeout=90)
    if r.status_code == 404:
        return {} if as_json else ""
    r.raise_for_status()
    return r.json() if as_json else r.text


def _get(url: str, ttl: float | None, as_json: bool = True):
    hit = _cache.get(url, ttl)
    if hit is not None:
        return hit
    value = _get_uncached(url, as_json)
    _cache.put(url, value)
    return value


@lru_cache(maxsize=1)
def ticker_map() -> dict[str, dict]:
    d = _get("https://www.sec.gov/files/company_tickers_exchange.json", DAY)
    fields = d["fields"]
    out: dict[str, dict] = {}
    for row in d["data"]:
        rec = dict(zip(fields, row))
        t = str(rec["ticker"]).upper()
        norm = re.sub(r"[^A-Z0-9]", "", t)
        out.setdefault(norm, {"cik": int(rec["cik"]), "ticker": t, "name": rec["name"],
                              "exchange": rec.get("exchange")})
    return out


_SUB_KEYS = ("accessionNumber", "filingDate", "reportDate", "form", "primaryDocument",
             "primaryDocDescription", "items")


def submissions(cik: int, ttl: float = 6 * 3600) -> dict:
    """Submissions metadata, trimmed to the fields used here before caching."""
    url = f"https://data.sec.gov/submissions/CIK{cik:010d}.json"
    key = "trim:" + url
    hit = _cache.get(key, ttl)
    if hit is not None:
        return hit
    full = _get_uncached(url)
    filings = full.get("filings") or {}
    trimmed = {"sic": full.get("sic"), "sicDescription": full.get("sicDescription"), "name": full.get("name"),
               "filings": {"recent": {k: (filings.get("recent") or {}).get(k, []) for k in _SUB_KEYS},
                           "files": filings.get("files") or []}}
    _cache.put(key, trimmed)
    return trimmed


def filings_between(cik: int, start: date, end_exclusive: date, ttl: float = 6 * 3600) -> list[dict]:
    """Filings with start <= filingDate < end_exclusive, newest first."""
    sub = submissions(cik, ttl)
    rec = (sub.get("filings") or {}).get("recent") or {}
    out = _rows(rec)
    # Older filings live in extra pages; fetch them only if the window reaches past 'recent'.
    if out and min(f["filingDate"] for f in out) > start.isoformat():
        for page in (sub.get("filings") or {}).get("files") or []:
            if page.get("filingTo", "9999") < start.isoformat():
                continue
            out += _rows(_get(f"https://data.sec.gov/submissions/{page['name']}", None))
    sel = [f for f in out if start.isoformat() <= f["filingDate"] < end_exclusive.isoformat()]
    sel.sort(key=lambda f: f["filingDate"], reverse=True)
    return sel


def _rows(rec: dict) -> list[dict]:
    keys = ("accessionNumber", "filingDate", "reportDate", "form", "primaryDocument",
            "primaryDocDescription", "items")
    n = len(rec.get("accessionNumber") or [])
    return [{k: (rec.get(k) or [""] * n)[i] for k in keys} for i in range(n)]


def doc_url(cik: int, accession: str, filename: str) -> str:
    return f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}/{filename}"


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for t in soup(["script", "style", "ix:header"]):
        t.decompose()
    text = soup.get_text("\n")
    text = re.sub(r"\xa0", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def filing_text(cik: int, accession: str, filename: str) -> str:
    if not filename:
        return ""
    url = doc_url(cik, accession, filename)
    hit = _cache.get("text:" + url, None)
    if hit is not None:
        return hit
    raw = _get(url, None, as_json=False)
    text = html_to_text(raw) if filename.lower().endswith((".htm", ".html")) else raw
    _cache.put("text:" + url, text)
    return text


def exhibit_99_text(cik: int, accession: str) -> str:
    idx = _get(doc_url(cik, accession, "index.json"), None)
    items = ((idx or {}).get("directory") or {}).get("item") or []
    names = [i["name"] for i in items if re.search(r"ex-?99", i.get("name", ""), re.I)
             and i["name"].lower().endswith((".htm", ".html", ".txt"))]
    return "\n\n".join(filing_text(cik, accession, n) for n in sorted(names)[:2])


_SECTIONS = {
    "10-K": {"business": (r"item\s*1\.?\s*business", r"item\s*1a\.?\s*risk\s*factors"),
             "risk": (r"item\s*1a\.?\s*risk\s*factors", r"item\s*1b\.?|item\s*1c\.?|item\s*2\.?\s*properties"),
             "mdna": (r"item\s*7\.?\s*management", r"item\s*7a\.?|item\s*8\.?\s*financial")},
    "10-Q": {"mdna": (r"item\s*2\.?\s*management", r"item\s*3\.?\s*quantitative|item\s*4\.?\s*controls"),
             "risk": (r"item\s*1a\.?\s*risk\s*factors", r"item\s*2\.?\s*unregistered|item\s*3\.?|item\s*5\.?|item\s*6\.?")},
}


def extract_section(text: str, form: str, section: str) -> str:
    """Returns the longest span between a section heading and the next heading. Picking the
    longest skips the table-of-contents occurrence, which is only a line long."""
    base = "10-K" if form.startswith("10-K") else "10-Q"
    pats = _SECTIONS[base].get(section)
    if not pats:
        return ""
    start_re, end_re = (re.compile(p, re.I) for p in pats)
    best = ""
    for m in start_re.finditer(text):
        e = end_re.search(text, m.end())
        chunk = text[m.start(): e.start() if e else len(text)]
        if len(chunk) > len(best):
            best = chunk
    return best.strip()


def companyfacts(cik: int, ttl: float = DAY) -> dict:
    """XBRL company facts, trimmed to the concepts used here (full files can be tens of MB)."""
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
    key = "trim:" + url
    hit = _cache.get(key, ttl)
    if hit is not None:
        return hit
    full = _get_uncached(url)
    wanted = {n for names in (*FLOW.values(), *INSTANT.values()) for n in names}
    wanted.add("EntityCommonStockSharesOutstanding")
    keep = ("start", "end", "val", "filed", "form", "accn", "fy", "fp")
    trimmed = {"facts": {}}
    for taxonomy, concepts in (full.get("facts") or {}).items():
        for name, node in concepts.items():
            if name in wanted:
                trimmed["facts"].setdefault(taxonomy, {})[name] = {
                    "units": {u: [{k: e[k] for k in keep if k in e} for e in rows]
                              for u, rows in node.get("units", {}).items()}}
    _cache.put(key, trimmed)
    return trimmed


# --- point-in-time fundamentals ------------------------------------------------------------

FLOW = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet",
                "InterestAndDividendIncomeOperating", "RevenuesNetOfInterestExpense"],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "eps_diluted": ["EarningsPerShareDiluted"],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "rnd": ["ResearchAndDevelopmentExpense"],
    "sbc": ["ShareBasedCompensation"],
    "dividends": ["PaymentsOfDividends", "PaymentsOfDividendsCommonStock"],
    "buybacks": ["PaymentsForRepurchaseOfCommonStock"],
}
INSTANT = {
    "cash": ["CashAndCashEquivalentsAtCarryingValue",
             "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "total_assets": ["Assets"],
    "total_liabilities": ["Liabilities"],
    "equity": ["StockholdersEquity",
               "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "long_term_debt": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "short_term_debt": ["DebtCurrent", "LongTermDebtCurrent", "ShortTermBorrowings"],
    "inventory": ["InventoryNet"],
    "receivables": ["AccountsReceivableNetCurrent"],
}


def _days(a: str, b: str) -> int:
    return (date.fromisoformat(b) - date.fromisoformat(a)).days


def _visible(entries: list[dict], as_of: date) -> list[dict]:
    """Facts filed before as_of; for each period keep the latest filing visible at as_of."""
    best: dict[tuple, dict] = {}
    for e in entries:
        if e.get("filed", "9999") >= as_of.isoformat():
            continue
        k = (e.get("start"), e["end"])
        if k not in best or e["filed"] > best[k]["filed"]:
            best[k] = e
    return list(best.values())


def quarterly_series(entries: list[dict], as_of: date) -> list[tuple[str, float]]:
    """Discrete quarterly values (end_date, value) from 3-month facts and YTD differences."""
    vis = [e for e in _visible(entries, as_of) if e.get("start")]
    direct: dict[str, float] = {}
    by_start: dict[str, list[dict]] = {}
    for e in vis:
        d = _days(e["start"], e["end"])
        if 80 <= d <= 100:
            direct[e["end"]] = float(e["val"])
        by_start.setdefault(e["start"], []).append(e)
    derived: dict[str, float] = {}
    for start, group in by_start.items():
        group.sort(key=lambda e: e["end"])
        for prev, cur in zip(group, group[1:]):
            gap = _days(prev["end"], cur["end"])
            if 80 <= gap <= 100:
                derived[cur["end"]] = float(cur["val"]) - float(prev["val"])
        first = group[0]
        if 80 <= _days(first["start"], first["end"]) <= 100:
            derived.setdefault(first["end"], float(first["val"]))
    merged = {**derived, **direct}
    return sorted(merged.items())


def _pick_concept(facts: dict, names: list[str], as_of: date, unit_hint: str | None = None):
    best, best_end = None, ""
    for taxonomy in ("us-gaap", "ifrs-full", "dei"):
        for name in names:
            node = ((facts.get("facts") or {}).get(taxonomy) or {}).get(name)
            if not node:
                continue
            for unit, entries in node["units"].items():
                if unit_hint and unit_hint not in unit:
                    continue
                vis = _visible(entries, as_of)
                if not vis:
                    continue
                end = max(e["end"] for e in vis)
                if end > best_end:
                    best, best_end = (entries, unit), end
    return best


def fundamentals_pit(facts: dict, as_of: date) -> dict:
    """Point-in-time fundamentals: last 8 quarters, TTM and YoY growth for flow items;
    latest balance for instant items; shares outstanding."""
    out: dict = {"currency": None, "quarters": {}, "ttm": {}, "ttm_prior": {}, "yoy_growth": {},
                 "balance": {}, "shares_outstanding": None, "last_period_end": None}
    if not facts:
        return out
    for key, names in FLOW.items():
        hint = "USD/shares" if key == "eps_diluted" else None
        picked = _pick_concept(facts, names, as_of, hint)
        if not picked:
            continue
        entries, unit = picked
        if key != "eps_diluted" and "/" in unit:
            continue
        out["currency"] = out["currency"] or unit.split("/")[0]
        q = quarterly_series(entries, as_of)
        if not q:
            continue
        out["quarters"][key] = q[-8:]
        last4, prior4 = q[-4:], q[-8:-4]
        if len(last4) == 4 and _days(last4[0][0], last4[-1][0]) <= 290:
            out["ttm"][key] = sum(v for _, v in last4)
            if len(prior4) == 4 and _days(prior4[0][0], prior4[-1][0]) <= 290:
                out["ttm_prior"][key] = sum(v for _, v in prior4)
                if out["ttm_prior"][key]:
                    out["yoy_growth"][key] = out["ttm"][key] / abs(out["ttm_prior"][key]) - 1
        out["last_period_end"] = max(out["last_period_end"] or "", q[-1][0])
    for key, names in INSTANT.items():
        picked = _pick_concept(facts, names, as_of)
        if not picked:
            continue
        vis = [e for e in _visible(picked[0], as_of) if not e.get("start")]
        if vis:
            last = max(vis, key=lambda e: e["end"])
            out["balance"][key] = {"value": float(last["val"]), "as_of": last["end"]}
    shares = _pick_concept(facts, ["EntityCommonStockSharesOutstanding"], as_of, "shares")
    if shares:
        vis = [e for e in shares[0] if e.get("filed", "9999") < as_of.isoformat()]
        if vis:
            # Multi-class issuers report one row per class on the same date: sum the rows of the
            # latest filing for the latest date.
            last_end = max(e["end"] for e in vis)
            rows = [e for e in vis if e["end"] == last_end]
            last_filed = max(e["filed"] for e in rows)
            unique = {tuple(sorted(e.items())) for e in rows if e["filed"] == last_filed}
            out["shares_outstanding"] = sum(float(dict(e)["val"]) for e in unique)
    return out


@lru_cache(maxsize=8192)
def sic_of(cik: int) -> tuple[str, str]:
    sub = submissions(cik, ttl=30 * DAY)
    return str(sub.get("sic") or ""), str(sub.get("sicDescription") or "")


def recent_filings_bundle(cik: int, as_of: date, lookback_days: int, max_chars: dict) -> dict:
    """Filing list plus text excerpts visible at as_of: latest 10-K business/risk/MD&A,
    latest 10-Q MD&A/risk (if newer than the 10-K), and 8-K bodies with press releases."""
    window = filings_between(cik, as_of - timedelta(days=lookback_days), as_of)
    annual_window = filings_between(cik, as_of - timedelta(days=430), as_of)
    listing = [{"id": f"sec:{f['accessionNumber']}", "date": f["filingDate"], "form": f["form"],
                "description": f["primaryDocDescription"],
                "items_8k": [f"{i} {ITEM_8K.get(i, '')}".strip() for i in str(f["items"]).split(",") if i]}
               for f in window]
    docs = []
    tenk = next((f for f in annual_window if f["form"] in ("10-K", "10-K/A", "20-F", "40-F")), None)
    tenq = next((f for f in window if f["form"] in ("10-Q", "10-Q/A")), None)
    if tenk:
        txt = filing_text(cik, tenk["accessionNumber"], tenk["primaryDocument"])
        for sec_name in ("business", "risk", "mdna"):
            part = extract_section(txt, "10-K", sec_name) if tenk["form"].startswith("10-K") else ""
            if not part and sec_name == "mdna":
                part = txt
            if part:
                docs.append({"id": f"sec:{tenk['accessionNumber']}:{sec_name}", "date": tenk["filingDate"],
                             "form": tenk["form"], "section": sec_name,
                             "text": part[: max_chars.get(sec_name, 20000)]})
    if tenq and (not tenk or tenq["filingDate"] > tenk["filingDate"]):
        txt = filing_text(cik, tenq["accessionNumber"], tenq["primaryDocument"])
        for sec_name in ("mdna", "risk"):
            part = extract_section(txt, "10-Q", sec_name)
            if part:
                docs.append({"id": f"sec:{tenq['accessionNumber']}:{sec_name}", "date": tenq["filingDate"],
                             "form": tenq["form"], "section": sec_name,
                             "text": part[: max_chars.get(sec_name, 20000)]})
    eightks = [f for f in window if f["form"] in ("8-K", "8-K/A", "6-K")][:6]
    for f in eightks:
        body = filing_text(cik, f["accessionNumber"], f["primaryDocument"])
        ex = exhibit_99_text(cik, f["accessionNumber"])
        docs.append({"id": f"sec:{f['accessionNumber']}", "date": f["filingDate"], "form": f["form"],
                     "section": "8-K", "text": (body[:6000] + ("\n\n[EX-99]\n" + ex if ex else ""))
                     [: max_chars.get("8-K", 12000)]})
    return {"list": listing, "documents": docs}
