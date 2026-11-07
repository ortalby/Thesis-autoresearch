"""The fixed scoring function of the autoresearch loop."""
from __future__ import annotations

from collections import defaultdict
from statistics import mean

from scipy.stats import spearmanr

MIN_PER_DATE = 5


def rank_ic(rows: list[dict], horizon: int) -> float | None:
    """Mean over analysis dates of the Spearman correlation between predicted and realized
    excess returns (cross-sectional, as in factor research)."""
    by_date: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for r in rows:
        real = r["realized"].get(f"excess_{horizon}d")
        if real is None:
            continue
        by_date[r["as_of"]].append((r["pred"][f"excess_{horizon}d"], real))
    ics = []
    for pairs in by_date.values():
        if len(pairs) < MIN_PER_DATE:
            continue
        p, y = zip(*pairs)
        if len(set(p)) == 1 or len(set(y)) == 1:
            ics.append(0.0)  # a constant forecast carries no ranking information
            continue
        ics.append(float(spearmanr(p, y).statistic))
    return mean(ics) if ics else None


def brier_skill(rows: list[dict]) -> float | None:
    """1 - Brier / Brier(0.5). Reference is the no-information forecast 0.5 (not the sample base
    rate, which would only be known after the fact)."""
    pts = [(r["pred"]["prob_90d"], 1.0 if r["realized"]["excess_90d"] > 0 else 0.0)
           for r in rows if r["realized"].get("excess_90d") is not None]
    if not pts:
        return None
    bs = mean((p - y) ** 2 for p, y in pts)
    return 1.0 - bs / 0.25


def combined(rows: list[dict], weights: dict) -> dict:
    ic30, ic90 = rank_ic(rows, 30), rank_ic(rows, 90)
    ics = [x for x in (ic30, ic90) if x is not None]
    ric = mean(ics) if ics else 0.0
    bss = brier_skill(rows) or 0.0
    judge = mean(r["judge_score"] for r in rows) if rows else 0.0
    score = weights["rank_ic"] * ric + weights["brier_skill"] * bss + weights["judge"] * judge
    hits = [r for r in rows if r["realized"].get("excess_90d") is not None and r["pred"]["stance"] != "neutral"]
    hit_rate = (mean(1.0 if (r["pred"]["stance"] == "long") == (r["realized"]["excess_90d"] > 0) else 0.0
                     for r in hits) if hits else None)
    return {"score": score, "rank_ic": ric, "rank_ic_30d": ic30, "rank_ic_90d": ic90,
            "brier_skill": bss, "judge": judge, "directional_hit_rate": hit_rate,
            "n": len(rows), "failures": sum(1 for r in rows if r.get("error"))}
