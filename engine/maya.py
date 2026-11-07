"""MAYA (Tel Aviv Stock Exchange disclosure system): company profiles, report lists and report text.

The api/v1 endpoints sit behind an F5/Incapsula WAF: requests need a Chrome TLS fingerprint and
cookies seeded from the home page (see BrowserSession). Notes taken from projects that run this
in production on GitHub Actions:
- POST /api/v1/reports/companies  body {fromDate, toDate, limit, offset[, companyId]}; daily dates;
  pagination by limit/offset only (fields named page/pageNum trigger a WAF signature);
  offsets above ~1000 return HTTP 400, so long windows are split.
- GET /api/v1/reports/{id} returns the report with its attachments (htm/pdf on mayafiles).
- GET /api/v1/companies/{id}/details returns branch, activity description and securities.
"""
from __future__ import annotations

import io
import re
from datetime import date, datetime, timedelta

from .http import BrowserSession, DiskCache
from .sec import html_to_text

BASE = "https://maya.tase.co.il"
FILES = "https://mayafiles.tase.co.il/"
PAGE = 30
OFFSET_CEILING = 1000

_cache = DiskCache("maya")
_session: BrowserSession | None = None


def _s() -> BrowserSession:
    global _session
    if _session is None:
        _session = BrowserSession(f"{BASE}/he", BASE, per_second=2.0)
    return _session


def company_details(company_id: int, ttl: float = 7 * 86400) -> dict:
    key = f"details:{company_id}"
    hit = _cache.get(key, ttl)
    if hit is not None:
        return hit
    d = _s().request("GET", f"{BASE}/api/v1/companies/{company_id}/details") or {}
    _cache.put(key, d)
    return d


def profile(company_id: int) -> dict:
    d = company_details(company_id)
    return {
        "long_name": d.get("longName") or "",
        "branch": d.get("branch") or "",
        "sub_branch": d.get("subBranch") or "",
        "activity": " ".join(str(d.get("companyActivites") or "").split()),
        "market_value_k_ils": d.get("marketValue"),
        "securities": [{"security_id": s.get("securityId"), "symbol": s.get("symbol"),
                        "issued": s.get("issuedPaidUp"), "market_cap_m": s.get("marketCap")}
                       for s in (d.get("secrities") or [])],
    }


class WindowTooLarge(Exception):
    pass


def _window(start: date, end: date, company_id: int | None) -> list[dict]:
    body = {"fromDate": start.isoformat(), "toDate": end.isoformat(), "limit": PAGE}
    if company_id:
        body["companyId"] = int(company_id)
    items, seen, offset = [], set(), 0
    while True:
        r = _s().request("POST", f"{BASE}/api/v1/reports/companies",
                         json_body={**body, "offset": offset}, raw=True)
        if r is None:
            return items
        batch = r.json() or []
        total = int(r.headers.get("x-total-count") or 0)
        if total > OFFSET_CEILING:
            raise WindowTooLarge(total)
        for it in batch:
            if it["id"] not in seen:
                seen.add(it["id"])
                items.append(it)
        offset += len(batch)
        if not batch or offset >= total:
            return items


def reports(start: date, end: date, company_id: int | None = None) -> list[dict]:
    """All reports published in [start, end] (inclusive), splitting windows above the
    pagination ceiling."""
    try:
        return _window(start, end, company_id)
    except WindowTooLarge:
        if start >= end:
            raise
        mid = start + timedelta(days=(end - start).days // 2)
        left = reports(start, mid, company_id)
        right = reports(mid + timedelta(days=1), end, company_id)
        seen, out = set(), []
        for it in left + right:
            if it["id"] not in seen:
                seen.add(it["id"])
                out.append(it)
        return out


def published_at(item: dict) -> datetime:
    return datetime.fromisoformat(str(item["publishDate"])[:19])


def report_detail(report_id: str) -> dict:
    key = f"report:{report_id}"
    hit = _cache.get(key, None)
    if hit is not None:
        return hit
    d = _s().request("GET", f"{BASE}/api/v1/reports/{report_id}") or {}
    _cache.put(key, d)
    return d


def _decode(raw: bytes) -> str:
    head = raw[:800].decode("latin-1", "replace").lower()
    order = (["windows-1255"] if "windows-1255" in head else ["utf-8"] if "utf-8" in head else [])
    for enc in order + ["utf-8", "windows-1255"]:
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", "replace")


_DROP_PREFIX = ("מספר ברשם", "Corporation no", "אסמכתא", "שעת שידור", "שודר במגנא",
                "Reported via MAGNA", "www.isa.gov.il", "www.tase.co.il", "סימול:", "Security No")


def _strip_boilerplate(text: str) -> str:
    out, seen = [], set()
    for line in text.splitlines():
        s = line.strip(" ‏‎·")
        if len(s) <= 1 or s.startswith(_DROP_PREFIX) or re.fullmatch(r"[\d\s\-.:/]{1,12}", s):
            continue
        if s not in seen:
            seen.add(s)
            out.append(s)
    return "\n".join(out)


def _file(path: str) -> bytes:
    r = _s().request("GET", FILES + path.lstrip("/"), raw=True)
    return r.content if r is not None else b""


def report_text(report_id: str, max_chars: int = 15000, with_pdf: bool = True) -> str:
    key = f"text:{report_id}:{max_chars}:{int(with_pdf)}"
    hit = _cache.get(key, None)
    if hit is not None:
        return hit
    rep = report_detail(report_id)
    atts = rep.get("attachments") or []
    htm = next((a for a in atts if str(a.get("fileType", "")).lower().startswith("htm") and a.get("url")), None)
    pdfs = [a for a in atts if str(a.get("fileType", "")).lower().startswith("pdf") and a.get("url")]
    parts = []
    if htm:
        parts.append(_strip_boilerplate(html_to_text(_decode(_file(htm["url"])))))
    if with_pdf and pdfs:
        from pypdf import PdfReader
        main = max(pdfs, key=lambda a: a.get("fileSize") or 0) if len(pdfs) > 1 else pdfs[0]
        raw = _file(main["url"])
        if raw and len(raw) < 15_000_000:
            reader = PdfReader(io.BytesIO(raw))
            pages = []
            for page in reader.pages[:30]:
                try:
                    pages.append(page.extract_text() or "")
                except Exception:  # a broken page must not drop the whole report
                    continue
                if sum(map(len, pages)) > max_chars:
                    break
            parts.append(re.sub(r"[ \t]{2,}", " ", "\n".join(pages)))
    text = "\n\n".join(p for p in parts if p.strip())[:max_chars]
    _cache.put(key, text)
    return text


def company_reports_bundle(company_id: int, as_of: date, lookback_days: int,
                           max_docs: int = 8, max_chars: int = 12000) -> dict:
    """Report list visible at as_of (published before as_of) and text of the most important ones:
    priority (immediate) reports first, then the most recent."""
    items = [it for it in reports(as_of - timedelta(days=lookback_days), as_of - timedelta(days=1), company_id)
             if published_at(it).date() < as_of]
    items.sort(key=published_at, reverse=True)
    listing = [{"id": f"maya:{it['id']}", "date": published_at(it).date().isoformat(),
                "title": (it.get("title") or "").strip(), "form": it.get("formId") or "",
                "priority": bool(it.get("isPriority"))} for it in items]
    chosen = sorted(items, key=lambda it: (not it.get("isPriority"), -published_at(it).timestamp()))[:max_docs]
    docs = []
    for it in chosen:
        text = report_text(str(it["id"]), max_chars=max_chars)
        if text:
            docs.append({"id": f"maya:{it['id']}", "date": published_at(it).date().isoformat(),
                         "form": it.get("formId") or "", "title": (it.get("title") or "").strip(),
                         "text": text})
    return {"list": listing, "documents": docs}
