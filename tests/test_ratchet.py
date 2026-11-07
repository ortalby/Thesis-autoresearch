"""The keep/discard ratchet of run_research, with the model and the harness replaced by stubs,
run inside a temporary git repository."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture()
def repo(tmp_path, monkeypatch):
    from engine.config import ROOT

    dst = tmp_path / "repo"
    shutil.copytree(ROOT, dst, ignore=shutil.ignore_patterns("state", ".git", "__pycache__", ".pytest_cache"))
    (dst / "state" / "research").mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=dst, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "add", "."], cwd=dst, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"], cwd=dst, check=True)
    monkeypatch.setenv("GIT_AUTHOR_NAME", "t")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "t@t")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "t")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "t@t")
    return dst


def test_keep_and_discard(repo, monkeypatch):
    import run_research as rr

    monkeypatch.setattr(rr, "ROOT", repo)
    monkeypatch.setattr(rr, "STRATEGY", repo / "strategy.py")
    monkeypatch.setattr(rr, "RESEARCH", repo / "state" / "research")
    monkeypatch.setattr(rr, "RESULTS", repo / "state" / "research" / "results.tsv")
    monkeypatch.setattr(rr, "BEST_REPORT", repo / "state" / "research" / "best_dev_report.json")
    monkeypatch.setattr(rr, "strategy_version", lambda: "v")
    ef = repo / "state" / "research" / "evalset.json"
    ef.write_text(json.dumps({"dev": [{"key": "US:A", "as_of": "2026-05-01"}] * 10, "holdout": []}))
    monkeypatch.setattr(rr.evalset, "EVALSET_FILE", ef)
    monkeypatch.setattr(rr.evalset, "load", lambda: json.loads(ef.read_text()))
    monkeypatch.setattr(rr, "Gemini", lambda: object())
    monkeypatch.setattr(rr.usage, "calls_today", lambda: 0)

    original = (repo / "strategy.py").read_text(encoding="utf-8")
    better = original.replace("CONTEXT_BUDGET_CHARS = 180_000", "CONTEXT_BUDGET_CHARS = 150_000")
    worse = original.replace("CONTEXT_BUDGET_CHARS = 180_000", "CONTEXT_BUDGET_CHARS = 90_000")
    proposals = iter([
        {"hypothesis": "h1", "description": "smaller context", "code": better},
        {"hypothesis": "h2", "description": "much smaller context", "code": worse},
        {"hypothesis": "h3", "description": "bad import", "code": "import os\n" + better},
    ])
    monkeypatch.setattr(rr, "propose", lambda g, best: next(proposals))

    def fake_eval(split, path, out, timeout):
        code = Path(path).read_text(encoding="utf-8")
        score = 0.30 if "150_000" in code else 0.10 if "90_000" in code else 0.20
        return {"metrics": {"score": score, "rank_ic": 0, "brier_skill": 0, "judge": score,
                            "directional_hit_rate": None, "failures": 0}, "rows": [], "strategy_version": "v"}

    monkeypatch.setattr(rr, "run_eval", fake_eval)
    monkeypatch.setattr("sys.argv", ["run_research.py", "--experiments", "3"])
    assert rr.main() == 0

    final = (repo / "strategy.py").read_text(encoding="utf-8")
    assert "150_000" in final  # the improvement was kept, the regression and the invalid code were not
    rows = rr.read_results()
    assert [r["status"] for r in rows] == ["keep", "keep", "discard", "invalid"]
    log = subprocess.run(["git", "log", "--oneline"], cwd=repo, capture_output=True, text=True).stdout
    assert "keep score=0.3000" in log
