"""Fixed evaluation harness (the prepare.py of this loop). Scores a strategy file on the frozen
evaluation set and writes a JSON report. Usage:

    python -m engine.evaluate --split dev --strategy strategy.py --out state/research/last_eval.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .config import ROOT, cfg
from . import evalset, judge, metrics, thesis
from .llm import BudgetExceeded, Gemini, QuotaExhausted


def _one(strategy, version, gemini, split, item) -> dict:
    row = {"key": item["key"], "as_of": item["as_of"], "realized": item["realized"]}
    b = evalset.load_bundle(split, item["key"], item["as_of"])
    try:
        t = thesis.run(strategy, version, b, gemini, previous=None)
    except (BudgetExceeded, QuotaExhausted):
        raise
    except Exception as exc:
        row.update(error=f"{type(exc).__name__}: {exc}", trace=traceback.format_exc(limit=3)[-1500:],
                   pred={"excess_30d": 0.0, "excess_90d": 0.0, "prob_90d": 0.5, "stance": "neutral"},
                   judge_score=0.0)
        return row
    row["pred"] = {"excess_30d": t["expected_excess_return_30d"], "excess_90d": t["expected_excess_return_90d"],
                   "prob_90d": t["prob_outperform_90d"], "stance": t["stance"], "conviction": t["conviction"]}
    row["summary_he"] = t["summary_he"]
    row["unknown_evidence_ids"] = t["meta"]["unknown_evidence_ids"]
    try:
        j = judge.judge(gemini, b, t)
        row["judge"] = j
        row["judge_score"] = j["score"]
    except (BudgetExceeded, QuotaExhausted):
        raise
    except Exception as exc:
        row["judge_error"] = f"{type(exc).__name__}: {exc}"
        row["judge_score"] = 0.0
    return row


def evaluate(split: str, strategy_path: Path, workers: int = 4) -> dict:
    t0 = time.time()
    es = evalset.load()
    strategy = thesis.load_strategy(strategy_path)
    version = thesis.strategy_version(strategy_path)
    gemini = Gemini()
    items = es[split]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(lambda it: _one(strategy, version, gemini, split, it), items))
    m = metrics.combined(rows, cfg()["research"]["weights"])
    return {"split": split, "strategy_version": version, "metrics": m, "rows": rows,
            "seconds": round(time.time() - t0, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "holdout"])
    ap.add_argument("--strategy", default=str(ROOT / "strategy.py"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    try:
        report = evaluate(a.split, Path(a.strategy), a.workers)
    except (BudgetExceeded, QuotaExhausted) as exc:
        Path(a.out).write_text(json.dumps({"budget_exhausted": str(exc)}), encoding="utf-8")
        print(f"BUDGET: {exc}", file=sys.stderr)
        return 3
    Path(a.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"SCORE={report['metrics']['score']:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
