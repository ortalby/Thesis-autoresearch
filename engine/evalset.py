"""The fixed evaluation set: (stock, historical date) pairs after the model's knowledge cutoff,
with frozen point-in-time bundles and realized excess returns. Built once and committed, so every
experiment of the loop is scored on exactly the same inputs."""
from __future__ import annotations

import gzip
import json
import random
from datetime import date, timedelta

from .config import STATE, cfg
from . import bundle as bundle_mod
from . import prices

EVAL_DIR = STATE / "research"
EVALSET_FILE = EVAL_DIR / "evalset.json"
BUNDLES = EVAL_DIR / "bundles"
MAX_HORIZON = 90


def eval_dates(cutoff: date, today: date, n: int) -> list[date]:
    last = today - timedelta(days=MAX_HORIZON + 7)
    if last <= cutoff:
        raise RuntimeError(f"No room for {MAX_HORIZON}-day outcomes between cutoff {cutoff} and {today}")
    step = (last - cutoff).days / max(1, n - 1)
    out = []
    for i in range(n):
        d = cutoff + timedelta(days=round(i * step))
        while d.weekday() >= 5:
            d += timedelta(days=1)
        while d > last or d.weekday() >= 5:
            d -= timedelta(days=1)
        out.append(d)
    return sorted(set(out))


def _bundle_path(split: str, key: str, as_of: str):
    return BUNDLES / split / f"{key.replace(':', '_')}@{as_of}.json.gz"


def save_bundle(split: str, b: dict) -> None:
    p = _bundle_path(split, b["stock"]["key"], b["as_of"])
    p.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(p, "wt", encoding="utf-8") as fh:
        json.dump(b, fh, ensure_ascii=False)


def load_bundle(split: str, key: str, as_of: str) -> dict:
    with gzip.open(_bundle_path(split, key, as_of), "rt", encoding="utf-8") as fh:
        return json.load(fh)


def realized(stock: dict, as_of: date) -> dict:
    bench_sym = cfg()["benchmarks"][stock["market"]]
    recs = prices.fetch([stock["yf"], bench_sym])
    out = {}
    for h in cfg()["update"]["horizons_days"]:
        r = prices.forward_return(recs[stock["yf"]], as_of, h)
        rb = prices.forward_return(recs[bench_sym], as_of, h)
        out[f"return_{h}d"] = r
        out[f"benchmark_{h}d"] = rb
        out[f"excess_{h}d"] = (r - rb) if r is not None and rb is not None else None
    return out


def build(universe: list[dict], today: date, log=print) -> dict:
    r = cfg()["research"]
    rng = random.Random(r["seed"])
    cutoff = date.fromisoformat(r["leakage_cutoff"])
    dates = eval_dates(cutoff, today, r["eval_dates"])
    us = [s for s in universe if s["market"] == "US"]
    il = [s for s in universe if s["market"] == "IL"]
    out = {"created": today.isoformat(), "seed": r["seed"], "cutoff": cutoff.isoformat(),
           "dates": [d.isoformat() for d in dates], "dev": [], "holdout": []}
    for d in dates:
        pools = {"US": rng.sample(us, len(us)), "IL": rng.sample(il, len(il))}
        used: set[str] = set()
        for split, per_date in (("dev", r["dev_per_date"]), ("holdout", r["holdout_per_date"])):
            n_il = round(per_date * r["il_share"])
            for market, need in (("IL", n_il), ("US", per_date - n_il)):
                got = 0
                for stock in pools[market]:
                    if got >= need:
                        break
                    if stock["key"] in used:
                        continue
                    used.add(stock["key"])
                    try:
                        real = realized(stock, d)
                        if real.get("excess_90d") is None:
                            continue
                        b = bundle_mod.build(stock, universe, d, live=False)
                    except Exception as exc:  # stock lacks history on that date: take the next one
                        log(f"skip {stock['key']} @ {d}: {type(exc).__name__}: {exc}")
                        continue
                    save_bundle(split, b)
                    out[split].append({"key": stock["key"], "as_of": d.isoformat(), "stock": stock,
                                       "realized": real})
                    got += 1
                    log(f"{split} {d} {stock['key']} ({got}/{need})")
    for split, per_date in (("dev", r["dev_per_date"]), ("holdout", r["holdout_per_date"])):
        expected = per_date * len(dates)
        if len(out[split]) < 0.8 * expected:
            raise RuntimeError(f"Evaluation set {split}: built only {len(out[split])} of {expected} samples; "
                               "check data-source errors above (SEC_USER_AGENT, network access)")
    gaps: dict[str, int] = {}
    for split in ("dev", "holdout"):
        for it in out[split]:
            for g in load_bundle(split, it["key"], it["as_of"]).get("data_gaps") or []:
                gaps[g["part"]] = gaps.get(g["part"], 0) + 1
    log(f"data gaps across evaluation bundles: {gaps or 'none'}")
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    EVALSET_FILE.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def load() -> dict:
    return json.loads(EVALSET_FILE.read_text(encoding="utf-8"))
