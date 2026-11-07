"""Live update: score matured predictions, detect what changed for every stock, refresh the theses
with the highest priority within the LLM budget, and publish to Google Sheets.

Priority (highest first): no thesis yet → periodic/earnings filings → immediate MAYA reports or
other material filings → unusual price move → stale thesis.

    python run_update.py [--max N] [--no-sheets]
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

from engine.config import cfg, ensure_dirs
from engine import bundle as bundle_mod
from engine import maya, prices, sec, sheets, store, thesis, universe
from engine.http import prune_all
from engine.llm import BudgetExceeded, Gemini, QuotaExhausted

MAJOR_FORMS = {"10-K", "10-K/A", "10-Q", "10-Q/A", "20-F", "40-F"}
EVENT_FORMS = MAJOR_FORMS | {"8-K", "8-K/A", "6-K", "SC 13D", "SC 13D/A", "S-4", "DEFM14A", "SC TO-T"}


def detect(stocks: list[dict], today: date, log=print) -> dict[str, tuple[float, list[str]]]:
    u = cfg()["update"]
    theses = store.all_theses()
    queue: dict[str, tuple[float, list[str]]] = {}

    def bump(key: str, score: float, reason: str) -> None:
        s, rs = queue.get(key, (0.0, []))
        queue[key] = (max(s, score), rs + [reason])

    for s in stocks:
        if s["key"] not in theses:
            bump(s["key"], 100 + s.get("weight", 0), "אין עדיין תזה")

    def since(key: str) -> date:
        return date.fromisoformat(theses[key]["current"]["meta"]["as_of"])

    # US: new SEC filings since each thesis
    for s in stocks:
        if s["market"] != "US" or s["key"] not in theses:
            continue
        try:
            fl = sec.filings_between(s["cik"], since(s["key"]), today + timedelta(days=1), ttl=3600)
        except Exception as exc:
            log(f"SEC scan failed for {s['key']}: {exc}")
            continue
        for f in fl:
            if f["form"] not in EVENT_FORMS:
                continue
            earnings = "2.02" in str(f.get("items"))
            score = 90 if f["form"] in MAJOR_FORMS or earnings else 75
            bump(s["key"], score, f"דיווח {f['form']} מ-{f['filingDate']}"
                 + (" (תוצאות כספיות)" if earnings else ""))

    # IL: MAYA reports since the last scan, for all companies in one window
    scan = store.load_scan()
    il = [s for s in stocks if s["market"] == "IL" and s["key"] in theses]
    if il:
        start = max(today - timedelta(days=30),
                    min([since(s["key"]) for s in il] + [date.fromisoformat(scan.get("maya_scanned_to", today.isoformat()))]))
        try:
            reps = maya.reports(start, today)
            by_company: dict[int, list[dict]] = {}
            for it in reps:
                for c in it.get("companies") or []:
                    by_company.setdefault(int(c.get("companyId") or 0), []).append(it)
            for s in il:
                for it in by_company.get(s["maya_company_id"], []):
                    if maya.published_at(it).date() < since(s["key"]):
                        continue
                    pri = bool(it.get("isPriority"))
                    bump(s["key"], 85 if pri else 60,
                         f"דיווח מאיה {maya.published_at(it).date()}: {(it.get('title') or '').strip()[:80]}")
            scan["maya_scanned_to"] = today.isoformat()
        except Exception as exc:
            log(f"MAYA scan failed: {exc}")

    # Prices: unusual move since the thesis
    recs = prices.fetch([s["yf"] for s in stocks], ttl=3600)
    for s in stocks:
        rec = recs.get(s["yf"])
        if not rec or s["key"] not in theses:
            continue
        last = prices.close_before(rec, today)
        if not last or last[0] < since(s["key"]).isoformat():
            continue
        sig = prices.daily_sigma_move(rec, today)
        if sig and sig >= u["price_move_sigma"]:
            bump(s["key"], 70 + min(sig, 10), f"תנועת מחיר חריגה ({sig:.1f} סטיות תקן) ב-{last[0]}")

    # Staleness
    for s in stocks:
        if s["key"] in theses:
            age = (today - since(s["key"])).days
            if age >= u["stale_after_days"]:
                bump(s["key"], 40 + age, f"התזה בת {age} ימים")
    store.save_scan(scan)
    return queue


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=cfg()["update"]["max_theses_per_run"])
    ap.add_argument("--no-sheets", action="store_true")
    a = ap.parse_args()
    ensure_dirs()
    print(f"pruned cache entries: {prune_all()}")
    t0 = time.time()
    today = date.today()
    u = cfg()["update"]
    stocks = universe.load()
    print(f"universe: {len(stocks)} stocks")
    print(f"scored matured predictions: {store.score_matured(stocks, today)}")

    queue = detect(stocks, today)
    by_key = {s["key"]: s for s in stocks}
    todo = sorted(queue.items(), key=lambda kv: -kv[1][0])[: a.max]
    print(f"stocks needing an update: {len(queue)}; this run: {len(todo)}")

    strategy = thesis.load_strategy()
    version = thesis.strategy_version()
    gemini = Gemini()
    stop = False
    done = failed = 0

    def work(key: str, reasons: list[str]):
        if stop or (time.time() - t0) / 60 > u["time_budget_minutes"]:
            return key, None, "skipped (time or budget)"
        prev = (store.load_thesis(key) or {}).get("current")
        b = bundle_mod.build(by_key[key], stocks, today, live=True)
        return key, thesis.run(strategy, version, b, gemini, prev), reasons

    with ThreadPoolExecutor(max_workers=u["workers"]) as pool:
        futs = {pool.submit(work, k, rs): k for k, (_, rs) in todo}
        for f in as_completed(futs):
            try:
                key, t, reasons = f.result()
            except (BudgetExceeded, QuotaExhausted) as exc:
                if not stop:
                    print(f"LLM budget/quota reached: {exc}")
                stop = True
                continue
            except Exception as exc:
                failed += 1
                print(f"failed {futs[f]}: {type(exc).__name__}: {exc}\n{traceback.format_exc(limit=2)}")
                continue
            if t is None:
                continue
            store.save_thesis(t, reasons)
            done += 1
            if done % 10 == 0:
                print(f"{done} theses updated")
    sc = store.build_scorecard()
    print(f"updated {done}, failed {failed}, minutes {(time.time() - t0) / 60:.1f}")
    if not a.no_sheets:
        sheets.publish(stocks, sc)
        print("published to Google Sheets")
    return 0


if __name__ == "__main__":
    sys.exit(main())
