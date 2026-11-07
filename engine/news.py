"""News headlines from Google News RSS, restricted to a date window with after:/before: operators,
so historical (point-in-time) queries return only what had been published before as_of."""
from __future__ import annotations

import re
from datetime import date, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote

import feedparser

from .http import DiskCache, Throttle, make_session

_cache = DiskCache("news")
_throttle = Throttle(1.0)
_session = make_session("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/140.0 Safari/537.36")

_SUFFIX = re.compile(r"\b(INC|CORP|CORPORATION|CO|COMPANY|LTD|LIMITED|PLC|HOLDINGS?|GROUP|CLASS [A-Z]|"
                     r"CL [A-Z]|NV|SA|AG|LP|LLC|TRUST|REIT|THE|INCORPORATED|BANCORP)\b\.?", re.I)


def clean_name(name: str) -> str:
    n = _SUFFIX.sub(" ", name or "")
    n = re.sub(r"[^\w&'\-\. ֐-׿]", " ", n)
    return re.sub(r"\s+", " ", n).strip().title() if n.isupper() else re.sub(r"\s+", " ", n).strip()


def _feed(query: str, lang: str, start: date, end_exclusive: date) -> list[dict]:
    q = f"{query} after:{start.isoformat()} before:{end_exclusive.isoformat()}"
    if lang == "he":
        params = "hl=he&gl=IL&ceid=IL:he"
    else:
        params = "hl=en-US&gl=US&ceid=US:en"
    url = f"https://news.google.com/rss/search?q={quote(q)}&{params}"
    historical = end_exclusive < date.today() - timedelta(days=2)
    hit = _cache.get(url, None if historical else 6 * 3600)
    if hit is not None:
        return hit
    _throttle.wait()
    r = _session.get(url, timeout=30)
    r.raise_for_status()
    feed = feedparser.parse(r.content)
    items = []
    for e in feed.entries:
        try:
            pub = parsedate_to_datetime(e.get("published")).astimezone(timezone.utc)
        except (TypeError, ValueError):
            continue
        source = (e.get("source") or {}).get("title", "") if isinstance(e.get("source"), dict) else ""
        title = re.sub(r"\s+-\s+[^-]+$", "", e.get("title", "")).strip() if source else e.get("title", "")
        items.append({"title": title, "source": source, "published": pub.isoformat(),
                      "link": e.get("link", "")})
    _cache.put(url, items)
    return items


def headlines(stock: dict, as_of: date, lookback_days: int, limit: int = 40) -> list[dict]:
    start = as_of - timedelta(days=lookback_days)
    queries = []
    if stock["market"] == "US":
        queries.append((f'"{clean_name(stock["name"])}" OR "{stock["symbol"]} stock"', "en"))
    else:
        if stock.get("name_he"):
            queries.append((f'"{stock["name_he"]}"', "he"))
        queries.append((f'"{clean_name(stock["name"])}" Israel', "en"))
    seen, out = set(), []
    for q, lang in queries:
        for it in _feed(q, lang, start, as_of):
            if it["published"][:10] >= as_of.isoformat():
                continue
            k = it["title"].lower()
            if k in seen:
                continue
            seen.add(k)
            it = dict(it, lang=lang)
            out.append(it)
    out.sort(key=lambda x: x["published"], reverse=True)
    for i, it in enumerate(out[:limit]):
        it["id"] = f"news:{i}"
    return out[:limit]
