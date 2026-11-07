"""Persistent state (committed to the repo by the workflows): current theses with history,
predictions awaiting scoring, change log, and the live scorecard."""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import date, timedelta
from statistics import mean

from .config import STATE, cfg
from . import prices

THESES = STATE / "theses"
PREDICTIONS = STATE / "predictions.jsonl"
CHANGES = STATE / "changes.jsonl"
SCORECARD = STATE / "scorecard.json"
SCAN = STATE / "scan.json"


def _path(key: str):
    return THESES / f"{key.replace(':', '_')}.json"


def load_thesis(key: str) -> dict | None:
    p = _path(key)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def all_theses() -> dict[str, dict]:
    out = {}
    for p in sorted(THESES.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        out[d["current"]["meta"]["key"]] = d
    return out


def save_thesis(thesis: dict, reasons: list[str]) -> dict | None:
    """Stores the new thesis; returns the change-log entry."""
    key = thesis["meta"]["key"]
    THESES.mkdir(parents=True, exist_ok=True)
    rec = load_thesis(key) or {"current": None, "history": []}
    prev = rec["current"]
    thesis["meta"]["update_reasons"] = reasons
    rec["history"].append({k: thesis[k] for k in ("summary_he", "stance", "conviction",
                                                   "expected_excess_return_30d", "expected_excess_return_90d",
                                                   "prob_outperform_90d")} | {"as_of": thesis["meta"]["as_of"],
                                                                              "strategy_version": thesis["meta"]["strategy_version"]})
    rec["history"] = rec["history"][-60:]
    rec["current"] = thesis
    _path(key).write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    change = {"date": thesis["meta"]["as_of"], "key": key,
              "from_stance": prev["stance"] if prev else None, "to_stance": thesis["stance"],
              "from_conviction": prev["conviction"] if prev else None, "to_conviction": thesis["conviction"],
              "from_e90": prev["expected_excess_return_90d"] if prev else None,
              "to_e90": thesis["expected_excess_return_90d"], "reasons": reasons,
              "changes_he": thesis["changes_vs_previous_he"], "summary_he": thesis["summary_he"]}
    with CHANGES.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(change, ensure_ascii=False) + "\n")
    with PREDICTIONS.open("a", encoding="utf-8") as fh:
        for h in cfg()["update"]["horizons_days"]:
            fh.write(json.dumps({
                "key": key, "as_of": thesis["meta"]["as_of"], "horizon": h,
                "benchmark": thesis["meta"]["benchmark"],
                "pred_excess": thesis[f"expected_excess_return_{h}d"],
                "prob_outperform": thesis["prob_outperform_90d"] if h == 90 else None,
                "stance": thesis["stance"], "strategy_version": thesis["meta"]["strategy_version"],
                "scored": False}, ensure_ascii=False) + "\n")
    return change


def read_jsonl(path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def score_matured(universe: list[dict], today: date) -> int:
    """Fills realized excess returns for predictions whose horizon has passed."""
    preds = read_jsonl(PREDICTIONS)
    yf_of = {s["key"]: s["yf"] for s in universe}
    due = [p for p in preds if not p["scored"] and date.fromisoformat(p["as_of"]) + timedelta(days=p["horizon"]) < today]
    if not due:
        return 0
    recs = prices.fetch(list({yf_of[p["key"]] for p in due if p["key"] in yf_of} | {p["benchmark"] for p in due}))
    n = 0
    for p in due:
        sym = yf_of.get(p["key"])
        if not sym or sym not in recs or p["benchmark"] not in recs:
            continue
        a = date.fromisoformat(p["as_of"])
        r = prices.forward_return(recs[sym], a, p["horizon"])
        rb = prices.forward_return(recs[p["benchmark"]], a, p["horizon"])
        if r is None or rb is None:
            continue
        p.update(realized_return=r, realized_benchmark=rb, realized_excess=r - rb, scored=True)
        n += 1
    PREDICTIONS.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in preds), encoding="utf-8")
    return n


def build_scorecard() -> dict:
    from scipy.stats import spearmanr

    preds = [p for p in read_jsonl(PREDICTIONS) if p["scored"]]
    out = {"by_horizon": {}, "by_strategy": {}}
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for p in preds:
        groups[("h", p["horizon"])].append(p)
        groups[("s", p["strategy_version"], p["horizon"])].append(p)

    def stats(rows: list[dict]) -> dict:
        weeks: dict[str, list[dict]] = defaultdict(list)
        for p in rows:
            y, w, _ = date.fromisoformat(p["as_of"]).isocalendar()
            weeks[f"{y}-W{w:02d}"].append(p)
        ics = []
        for ps in weeks.values():
            if len(ps) >= 10 and len({p["pred_excess"] for p in ps}) > 1:
                ics.append(float(spearmanr([p["pred_excess"] for p in ps], [p["realized_excess"] for p in ps]).statistic))
        dirn = [p for p in rows if p["stance"] != "neutral"]
        probs = [p for p in rows if p.get("prob_outperform") is not None]
        return {
            "n": len(rows),
            "mean_weekly_rank_ic": mean(ics) if ics else None, "weeks_with_ic": len(ics),
            "hit_rate": mean(1.0 if (p["stance"] == "long") == (p["realized_excess"] > 0) else 0.0 for p in dirn) if dirn else None,
            "brier_skill": (1 - mean((p["prob_outperform"] - (1.0 if p["realized_excess"] > 0 else 0.0)) ** 2 for p in probs) / 0.25) if probs else None,
            "long_avg_excess": mean(p["realized_excess"] for p in rows if p["stance"] == "long") if any(p["stance"] == "long" for p in rows) else None,
            "short_avg_excess": mean(p["realized_excess"] for p in rows if p["stance"] == "short") if any(p["stance"] == "short" for p in rows) else None,
        }

    for k, rows in groups.items():
        if k[0] == "h":
            out["by_horizon"][str(k[1])] = stats(rows)
        else:
            out["by_strategy"].setdefault(k[1], {})[str(k[2])] = stats(rows)
    SCORECARD.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def load_scan() -> dict:
    return json.loads(SCAN.read_text(encoding="utf-8")) if SCAN.exists() else {}


def save_scan(d: dict) -> None:
    SCAN.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
