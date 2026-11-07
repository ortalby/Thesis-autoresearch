"""Macro data, point-in-time: FRED (US), Bank of Israel SDMX, CBS consumer price index.
Monthly series are treated as visible only after their usual publication lag."""
from __future__ import annotations

import csv
import io
from datetime import date, timedelta

from .config import cfg
from .http import DiskCache, make_session

_cache = DiskCache("macro")
_session = make_session("thesis-autoresearch/1.0 (macro data)")
TTL = 12 * 3600
START = "2023-01-01"

MONTHLY_FRED = {"FEDFUNDS": 45, "CPIAUCSL": 45, "UNRATE": 40}
BOI_LAG = {"ZCM": 20}


def _text(url: str) -> str:
    hit = _cache.get(url, TTL)
    if hit is not None:
        return hit
    r = _session.get(url, timeout=60)
    r.raise_for_status()
    if r.text.lstrip().lower().startswith(("<!doctype", "<html")):
        raise RuntimeError(f"{url} returned HTML instead of data")
    _cache.put(url, r.text)
    return r.text


def fred(series: str) -> list[tuple[str, float]]:
    txt = _text(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd={START}")
    rows = list(csv.reader(io.StringIO(txt)))
    out = []
    for r in rows[1:]:
        if len(r) >= 2 and r[1] not in (".", ""):
            out.append((r[0], float(r[1])))
    return out


def boi(dataflow: str, code: str) -> list[tuple[str, float]]:
    url = (f"https://edge.boi.gov.il/FusionEdgeServer/sdmx/v2/data/dataflow/BOI.STATISTICS/"
           f"{dataflow}/1.0/{code}?startPeriod={START}&format=csv")
    rows = list(csv.DictReader(io.StringIO(_text(url).lstrip("﻿"))))
    out = []
    for r in rows:
        v = r.get("OBS_VALUE")
        if r.get("SERIES_CODE") != code or v in (None, "", "NaN"):
            continue
        mult = int(r.get("UNIT_MULT") or 0)
        period = r["TIME_PERIOD"]
        if len(period) == 7:
            period += "-01"
        out.append((period, float(v) / (10 ** mult)))
    out.sort()
    return out


def cbs_cpi_yoy(index_id: int) -> list[tuple[str, float]]:
    import json

    url = (f"https://api.cbs.gov.il/index/data/price?id={index_id}&format=json&download=false"
           f"&startPeriod=01-{START[:4]}&PageSize=1000")
    payload = json.loads(_text(url))
    out = []
    for series in payload.get("month") or []:
        for e in series.get("date") or []:
            if e.get("percentYear") is None:
                continue
            out.append((f"{int(e['year'])}-{int(e['month']):02d}-01", float(e["percentYear"])))
    if not out:
        raise RuntimeError("CBS CPI response had no observations")
    return sorted(out)


def _pit(series: list[tuple[str, float]], as_of: date, lag_days: int) -> list[tuple[str, float]]:
    cutoff = (as_of - timedelta(days=lag_days)).isoformat()
    return [(d, v) for d, v in series if d < cutoff]


def _summary(points: list[tuple[str, float]], monthly: bool) -> dict:
    if not points:
        return {}
    last_d, last_v = points[-1]

    def back(days: int):
        target = (date.fromisoformat(last_d) - timedelta(days=days)).isoformat()
        prev = [v for d, v in points if d <= target]
        return prev[-1] if prev else None

    out = {"date": last_d, "value": last_v}
    for days, name in ((30, "1m"), (91, "3m"), (365, "12m")):
        p = back(days)
        if p is not None:
            out[f"change_{name}"] = last_v - p
    out["history"] = [{"date": d, "value": round(v, 4)} for d, v in points[-(13 if monthly else 260):][:: 1 if monthly else 20]]
    return out


def snapshot(as_of: date) -> dict:
    """Both US and Israeli macro for every stock: US rates and the dollar drive TASE valuations,
    and many US-listed firms have Israeli operations."""
    m = cfg()["macro"]
    out: dict = {"US": {}, "IL": {}}
    for code, label in m["fred_series"].items():
        lag = MONTHLY_FRED.get(code, 0)
        out["US"][code] = {"label": label, **_summary(_pit(fred(code), as_of, lag), code in MONTHLY_FRED)}
    for s in m["boi_series"]:
        lag = BOI_LAG.get(s["dataflow"], 0)
        out["IL"][s["code"]] = {"label": s["label"],
                                **_summary(_pit(boi(s["dataflow"], s["code"]), as_of, lag),
                                           s["dataflow"] == "ZCM")}
    cpi = cbs_cpi_yoy(m["cbs_cpi_id"])
    # CBS publishes month M around the 15th of M+1.
    out["IL"]["CPI_YOY"] = {"label": "אינפלציה שנתית (הלמ\"ס)", **_summary(_pit(cpi, as_of, 46), True)}
    return out
