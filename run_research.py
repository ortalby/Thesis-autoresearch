"""The autoresearch loop (Karpathy-style ratchet) for strategy.py.

Each experiment: the agent reads program.md, the current strategy.py, the experiment history and
diagnostics of the current best → writes a new strategy.py → the fixed harness scores it on the
frozen dev set → if the score beats the best by config.research.min_improvement the change is
committed (kept), otherwise strategy.py is restored (discarded). Every experiment is logged to
state/research/results.tsv.

    python run_research.py [--experiments N]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

from engine.config import ROOT, STATE, cfg, ensure_dirs
from engine import evalset, universe
from engine.guard import GuardError, check
from engine.llm import BudgetExceeded, Gemini, QuotaExhausted, usage
from engine.thesis import strategy_version

RESEARCH = STATE / "research"
RESULTS = RESEARCH / "results.tsv"
BEST_REPORT = RESEARCH / "best_dev_report.json"
STRATEGY = ROOT / "strategy.py"
COLUMNS = ["timestamp", "experiment", "strategy_version", "status", "score", "rank_ic", "brier_skill",
           "judge", "hit_rate", "failures", "holdout_score", "seconds", "description"]

AGENT_SCHEMA = {
    "type": "object",
    "properties": {
        "hypothesis": {"type": "string", "description": "What you expect to improve and why"},
        "description": {"type": "string", "description": "One-line summary of the change"},
        "code": {"type": "string", "description": "The complete new contents of strategy.py"},
    },
    "required": ["hypothesis", "description", "code"],
}

AGENT_SYSTEM = """You are an autonomous quantitative research engineer running an autoresearch loop.
You improve a single Python file, strategy.py, that turns a point-in-time data bundle into an
investment thesis via an LLM. A fixed harness scores your file; only improvements are kept.
Follow program.md exactly. Return the COMPLETE new strategy.py (not a diff) in "code".
The code must keep the contract analyze(bundle, llm, previous) -> dict matching THESIS_SCHEMA,
use only allowed imports, and never touch files, network, or private attributes."""


def git(*args: str, check_rc: bool = True) -> str:
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    if check_rc and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout.strip()


def read_results() -> list[dict]:
    if not RESULTS.exists():
        return []
    with RESULTS.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def append_result(row: dict) -> None:
    new = not RESULTS.exists()
    with RESULTS.open("a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, delimiter="\t")
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in COLUMNS})


def run_eval(split: str, path: Path, out: Path, timeout_min: int) -> dict:
    cmd = [sys.executable, "-m", "engine.evaluate", "--split", split, "--strategy", str(path), "--out", str(out)]
    try:
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout_min * 60)
    except subprocess.TimeoutExpired:
        return {"crash": f"timeout after {timeout_min} minutes"}
    if r.returncode == 3:
        return {"budget_exhausted": json.loads(out.read_text())["budget_exhausted"]}
    if r.returncode != 0:
        return {"crash": (r.stderr or r.stdout)[-3000:]}
    return json.loads(out.read_text(encoding="utf-8"))


def _shape(obj, depth: int = 0):
    """Type skeleton of a bundle with short examples, so the agent knows the data layout."""
    if depth > 4:
        return "…"
    if isinstance(obj, dict):
        return {k: _shape(v, depth + 1) for k, v in list(obj.items())[:25]}
    if isinstance(obj, list):
        return [_shape(obj[0], depth + 1), f"... {len(obj)} items"] if obj else []
    if isinstance(obj, str):
        return obj[:60] + ("…" if len(obj) > 60 else "")
    return obj


def diagnostics(report: dict) -> str:
    m = report["metrics"]
    rows = report["rows"]
    lines = [f"metrics: {json.dumps(m, ensure_ascii=False)}", "", "per-sample (key, as_of, pred_ex90, prob90, real_ex90, judge, comment/error):"]
    for r in rows:
        p, real = r["pred"], r["realized"]
        note = r.get("error") or (r.get("judge") or {}).get("comments_he", "")
        lines.append(f"- {r['key']} {r['as_of']} pred={p['excess_90d']:+.3f} p={p['prob_90d']:.2f} "
                     f"real={real.get('excess_90d') if real.get('excess_90d') is None else round(real['excess_90d'], 3)} "
                     f"judge={r['judge_score']:.2f} | {note[:300]}")
    buckets: dict[str, list[float]] = {}
    for r in rows:
        if r["realized"].get("excess_90d") is None:
            continue
        b = f"{min(0.9, (r['pred']['prob_90d'] // 0.1) / 10):.1f}"
        buckets.setdefault(b, []).append(1.0 if r["realized"]["excess_90d"] > 0 else 0.0)
    lines.append("")
    lines.append("calibration (prob bucket -> realized outperform rate, n): " + ", ".join(
        f"{k}: {sum(v) / len(v):.2f} (n={len(v)})" for k, v in sorted(buckets.items())))
    return "\n".join(lines)


def propose(gemini: Gemini, best_report: dict) -> dict:
    history = read_results()[-30:]
    hist = "\n".join(f"{h['experiment']}\t{h['status']}\t{h['score']}\tIC={h['rank_ic']}\tBSS={h['brier_skill']}"
                     f"\tJ={h['judge']}\t{h['description']}" for h in history) or "none"
    es = evalset.load()
    sample = es["dev"][0]
    shape = _shape(evalset.load_bundle("dev", sample["key"], sample["as_of"]))
    user = "\n\n".join([
        "## program.md\n" + (ROOT / "program.md").read_text(encoding="utf-8"),
        "## engine/schema.py (fixed output contract)\n" + (ROOT / "engine" / "schema.py").read_text(encoding="utf-8"),
        "## allowed imports\n" + (ROOT / "engine" / "guard.py").read_text(encoding="utf-8"),
        "## bundle layout (example skeleton)\n" + json.dumps(shape, ensure_ascii=False, default=str)[:20000],
        f"## limits\nmax LLM calls per stock: {cfg()['llm']['max_calls_per_stock']}; "
        f"max prompt chars per call: {cfg()['llm']['max_input_chars_per_call']}",
        "## experiment history (most recent last)\n" + hist,
        "## diagnostics of the current best strategy on the dev set\n" + diagnostics(best_report),
        "## current strategy.py\n```python\n" + STRATEGY.read_text(encoding="utf-8") + "\n```",
        "Propose ONE experiment. Return JSON with hypothesis, description and the full new code.",
    ])
    return gemini.json(cfg()["llm"]["agent_model"], AGENT_SYSTEM, user, AGENT_SCHEMA)


def _row(exp: str, status: str, report: dict | None, desc: str, seconds: float, holdout=None,
         version: str | None = None) -> dict:
    m = (report or {}).get("metrics") or {}
    f = lambda k: "" if m.get(k) is None else f"{m[k]:.6f}"
    return {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), "experiment": exp,
            "strategy_version": version or strategy_version(), "status": status, "score": f("score"),
            "rank_ic": f("rank_ic"), "brier_skill": f("brier_skill"), "judge": f("judge"),
            "hit_rate": f("directional_hit_rate"), "failures": m.get("failures", ""),
            "holdout_score": "" if holdout is None else f"{holdout:.6f}",
            "seconds": round(seconds), "description": desc.replace("\t", " ").replace("\n", " ")[:300]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", type=int, default=cfg()["research"]["experiments_per_run"])
    a = ap.parse_args()
    ensure_dirs()
    r = cfg()["research"]
    t_start = time.time()
    if not evalset.EVALSET_FILE.exists():
        print("Building the frozen evaluation set (one time)…")
        evalset.build(universe.load(), date.today())
        git("add", "state/research/evalset.json", "state/research/bundles")
        git("commit", "-m", "autoresearch: freeze evaluation set")

    es = evalset.load()
    calls_per_eval = 2 * len(es["dev"])
    timeout = r["experiment_timeout_minutes"]

    best_report = json.loads(BEST_REPORT.read_text(encoding="utf-8")) if BEST_REPORT.exists() else None
    if best_report is None or best_report.get("strategy_version") != strategy_version():
        t0 = time.time()
        rep = run_eval("dev", STRATEGY, RESEARCH / "eval_baseline.json", timeout)
        if "budget_exhausted" in rep:
            print(f"LLM budget exhausted before the baseline: {rep['budget_exhausted']}")
            return 0
        if "crash" in rep:
            print(rep["crash"])
            raise RuntimeError("The current strategy.py fails on the dev set; fix it before running the loop")
        best_report = rep
        BEST_REPORT.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        append_result(_row("baseline", "keep", rep, "baseline of current strategy.py", time.time() - t0))
    best = best_report["metrics"]["score"]
    print(f"best dev score: {best:.4f}")

    gemini = Gemini()
    for i in range(a.experiments):
        elapsed_min = (time.time() - t_start) / 60
        if elapsed_min + timeout > r["run_time_budget_minutes"]:
            print("time budget reached")
            break
        if usage.calls_today() + calls_per_eval + 2 > cfg()["llm"]["max_calls_per_day"]:
            print("daily LLM budget would be exceeded by another experiment; stopping")
            break
        exp = f"exp-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
        t0 = time.time()
        original = STRATEGY.read_text(encoding="utf-8")
        try:
            prop = propose(gemini, best_report)
        except (BudgetExceeded, QuotaExhausted) as exc:
            print(f"budget: {exc}")
            break
        desc = f"{prop['description']} — {prop['hypothesis']}"
        cand_version = hashlib.sha1(prop["code"].encode()).hexdigest()[:10]
        try:
            check(prop["code"])
        except GuardError as exc:
            append_result(_row(exp, "invalid", None, f"{desc} [guard: {exc}]", time.time() - t0,
                                 version=cand_version))
            continue
        STRATEGY.write_text(prop["code"], encoding="utf-8")
        rep = run_eval("dev", STRATEGY, RESEARCH / f"{exp}.json", timeout)
        if "budget_exhausted" in rep:
            STRATEGY.write_text(original, encoding="utf-8")
            print(f"budget: {rep['budget_exhausted']}")
            break
        if "crash" in rep:
            STRATEGY.write_text(original, encoding="utf-8")
            append_result(_row(exp, "crash", None, f"{desc} [{rep['crash'][-200:]}]", time.time() - t0,
                                 version=cand_version))
            continue
        score = rep["metrics"]["score"]
        if score > best + r["min_improvement"]:
            hold = run_eval("holdout", STRATEGY, RESEARCH / f"{exp}_holdout.json", timeout)
            hold_score = (hold.get("metrics") or {}).get("score")
            best, best_report = score, rep
            BEST_REPORT.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
            append_result(_row(exp, "keep", rep, desc, time.time() - t0, hold_score))
            git("add", "strategy.py")
            git("commit", "-m", f"autoresearch {exp}: keep score={score:.4f} — {prop['description'][:120]}")
            print(f"{exp}: KEEP {score:.4f}")
        else:
            STRATEGY.write_text(original, encoding="utf-8")
            append_result(_row(exp, "discard", rep, desc, time.time() - t0, version=cand_version))
            print(f"{exp}: discard {score:.4f} (best {best:.4f})")
        (RESEARCH / f"{exp}.json").unlink(missing_ok=True)
        (RESEARCH / f"{exp}_holdout.json").unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
