"""Publishes theses, the change log, the live scorecard and the research log to Google Sheets
(right-to-left), using a service account that has edit access to the spreadsheet."""
from __future__ import annotations

import csv
import json

import gspread

from .config import STATE, cfg, secret
from . import store

STANCE_HE = {"long": "חיובית (לונג)", "neutral": "ניטרלית", "short": "שלילית (שורט)"}
IMPACT_HE = {"positive": "חיובי", "negative": "שלילי", "uncertain": "לא ודאי"}
PROB_HE = {"low": "נמוכה", "medium": "בינונית", "high": "גבוהה"}
CELL_MAX = 45000


def _client() -> gspread.Spreadsheet:
    s = cfg()["sheets"]
    info = json.loads(secret(s["service_account_env"]))
    gc = gspread.service_account_from_dict(info)
    return gc.open_by_key(secret(s["spreadsheet_id_env"]))


def _pct(x) -> str:
    return "" if x is None else f"{x * 100:+.1f}%"


def _cell(x) -> str:
    s = "" if x is None else str(x)
    return s[:CELL_MAX]


def _sheet(book: gspread.Spreadsheet, title: str, cols: int) -> gspread.Worksheet:
    try:
        ws = book.worksheet(title)
    except gspread.WorksheetNotFound:
        ws = book.add_worksheet(title=title, rows=100, cols=cols)
    book.batch_update({"requests": [
        {"updateSheetProperties": {"properties": {"sheetId": ws.id, "rightToLeft": True,
                                                  "gridProperties": {"frozenRowCount": 1}},
                                   "fields": "rightToLeft,gridProperties.frozenRowCount"}}]})
    return ws


def _write(ws: gspread.Worksheet, rows: list[list]) -> None:
    ws.clear()
    need_rows, need_cols = max(len(rows), 2), max(len(r) for r in rows)
    ws.resize(rows=need_rows, cols=need_cols)
    values = [[_cell(c) for c in r] for r in rows]
    # Long Hebrew theses make big payloads; write in chunks to stay under API request limits.
    i, chunk = 0, []
    while i < len(values):
        size, chunk = 0, []
        while i < len(values) and (not chunk or size < 1_500_000):
            size += sum(len(c) for c in values[i])
            chunk.append(values[i])
            i += 1
        start = i - len(chunk) + 1
        ws.update(values=chunk, range_name=f"A{start}", value_input_option="RAW")


def thesis_rows(universe: list[dict]) -> list[list]:
    by_key = {s["key"]: s for s in universe}
    header = ["מניה", "שם", "שוק", "ענף", "עמדה", "ביטחון (1-5)", "תשואה עודפת צפויה 30 יום",
              "תשואה עודפת צפויה 90 יום", "הסתברות להכות את המדד (90 יום)", "תזה", "העסק", "פיננסים",
              "תחרות", "מאקרו", "תמחור", "דיווחים וחדשות", "קטליזטורים", "סיכונים", "מה יפריך את התזה",
              "מה השתנה", "תאריך עדכון", "סיבת עדכון", "מדד ייחוס", "גרסת אסטרטגיה", "פערי נתונים"]
    rows = [header]
    theses = store.all_theses()
    for key in sorted(theses, key=lambda k: (k.split(":")[0] != "IL", k)):
        t = theses[key]["current"]
        m = t["meta"]
        s = by_key.get(key, {})
        cats = "\n".join(f"• {c['event_he']} ({c['expected_date'] or 'ללא מועד'}, {IMPACT_HE[c['impact']]})" for c in t["catalysts"])
        risks = "\n".join(f"• {r['risk_he']} — הסתברות {PROB_HE[r['probability']]}: {r['impact_he']}" for r in t["risks"])
        rows.append([
            key.split(":", 1)[1], s.get("name_he") or s.get("name", ""), "ישראל" if key.startswith("IL") else "ארה\"ב",
            s.get("sector") or m.get("industry", ""), STANCE_HE[t["stance"]], t["conviction"], _pct(t["expected_excess_return_30d"]),
            _pct(t["expected_excess_return_90d"]), f"{t['prob_outperform_90d']:.0%}", t["summary_he"],
            t["business_he"], t["financials_he"], t["competition_he"], t["macro_he"], t["valuation_he"],
            t["news_and_filings_he"], cats, risks, "\n".join(f"• {k}" for k in t["kill_criteria_he"]),
            t["changes_vs_previous_he"], m["as_of"], "; ".join(m.get("update_reasons") or []), m["benchmark"],
            m["strategy_version"], ", ".join(m.get("data_gaps") or []),
        ])
    return rows


def change_rows(limit: int = 3000) -> list[list]:
    rows = [["תאריך", "מניה", "עמדה קודמת", "עמדה חדשה", "ביטחון קודם", "ביטחון חדש",
             "צפי 90 יום קודם", "צפי 90 יום חדש", "סיבת עדכון", "מה השתנה", "תזה"]]
    for c in reversed(store.read_jsonl(store.CHANGES)[-limit:]):
        rows.append([c["date"], c["key"], STANCE_HE.get(c["from_stance"], "—"), STANCE_HE[c["to_stance"]],
                     c["from_conviction"] or "—", c["to_conviction"], _pct(c["from_e90"]), _pct(c["to_e90"]),
                     "; ".join(c["reasons"]), c["changes_he"], c["summary_he"]])
    return rows


def scorecard_rows(sc: dict) -> list[list]:
    rows = [["קבוצה", "אופק (ימים)", "תחזיות שנמדדו", "מתאם דירוג שבועי ממוצע", "שבועות עם מדידה",
             "שיעור פגיעה בכיוון", "Brier skill", "תשואה עודפת ממוצעת בלונג", "תשואה עודפת ממוצעת בשורט"]]

    def add(label, h, s):
        f = lambda x: "" if x is None else round(x, 4)
        rows.append([label, h, s["n"], f(s["mean_weekly_rank_ic"]), s["weeks_with_ic"], f(s["hit_rate"]),
                     f(s["brier_skill"]), _pct(s["long_avg_excess"]), _pct(s["short_avg_excess"])])

    for h, s in sc.get("by_horizon", {}).items():
        add("כל התחזיות", h, s)
    for ver, hs in sc.get("by_strategy", {}).items():
        for h, s in hs.items():
            add(f"אסטרטגיה {ver}", h, s)
    return rows


def research_rows() -> list[list]:
    path = STATE / "research" / "results.tsv"
    if not path.exists():
        return [["אין עדיין ניסויים"]]
    with path.open(encoding="utf-8") as fh:
        data = list(csv.reader(fh, delimiter="\t"))
    names = {"timestamp": "זמן", "experiment": "ניסוי", "strategy_version": "גרסה", "status": "תוצאה",
             "score": "ציון", "rank_ic": "מתאם דירוג", "brier_skill": "Brier skill", "judge": "שופט",
             "hit_rate": "שיעור פגיעה", "failures": "כשלים", "holdout_score": "ציון בקרה", "seconds": "שניות",
             "description": "תיאור"}
    return [[names.get(c, c) for c in data[0]]] + list(reversed(data[1:]))


def publish(universe: list[dict], scorecard: dict) -> None:
    book = _client()
    _write(_sheet(book, "תזות", 25), thesis_rows(universe))
    _write(_sheet(book, "יומן שינויים", 11), change_rows())
    _write(_sheet(book, "ביצועי תחזיות", 9), scorecard_rows(scorecard))
    _write(_sheet(book, "מחקר", 13), research_rows())
