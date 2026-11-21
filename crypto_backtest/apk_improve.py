"""Improvement study for the MA and MarketTrend strategies (APK 0.2.1 base rules, slippage 0.05%, stop-first).

Method (to avoid over-fitting): every experiment is scored on a TRAIN period only; the chosen settings are then shown on a
TEST period that was never used for selection.  Split by trade entry time:
    30m / 60m / day : train < 2025-01-01 <= test        5m : train < 2026-05-01 <= test
Phase A  one-factor-at-a-time grid (apk_improve_ofat.csv)
Phase B  greedy forward selection on train portfolio return, combining one value per factor group (apk_improve_greedy.csv)
Trend angle band stays at the user's -10..+20 degrees (not tuned).
"""
from __future__ import annotations

import copy
import itertools
from multiprocessing import Pool

import numpy as np
import pandas as pd

import apk_strategies as S
from apk_run import COINS

CUT = {"5m": pd.Timestamp("2026-05-01"), "30m": pd.Timestamp("2025-01-01"),
       "60m": pd.Timestamp("2025-01-01"), "day": pd.Timestamp("2025-01-01")}
BASE_CFG = copy.deepcopy(S.CFG)
BARS: dict = {}
SC = None


def init() -> None:
    global SC
    ends, score = S.build_market_score(COINS)
    SC = S.ScoreLookup(ends, score)
    for tf in S.FRAMES:
        BARS[tf] = {c: S.load_bars(c, tf) for c in COINS}


def merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if k in ("ma_f", "trend_f"):
            out.setdefault(k, {}).update(v)
        else:
            out[k] = v
    return out


def stats(trades: list) -> dict:
    if not trades:
        return dict(n=0, win=np.nan, avg=np.nan, pf=np.nan, port=np.nan)
    r = np.array([t.ret for t in trades])
    w, l = r[r > 0], r[r <= 0]
    tf = trades[0].tf
    return dict(n=len(r), win=float((r > 0).mean()), avg=float(r.mean()),
                pf=float(w.sum() / -l.sum()) if l.sum() < 0 else np.nan, port=S.portfolio(trades, tf)["portfolio_return"])


def run(task):
    strategy, tf, over = task
    S.CFG.clear(); S.CFG.update(copy.deepcopy(BASE_CFG)); S.CFG.update(copy.deepcopy(over))
    S.SLIP, S.ORDER_MODE = 0.0005, "stop_first"
    trades = []
    for coin in COINS:
        b = BARS[tf][coin]
        if b is None:
            continue
        trades += S.ma_trades(coin, tf, b, SC) if strategy == "ma" else S.trend_trades(coin, tf, b, SC)
    tr = [t for t in trades if t.entry_t < CUT[tf]]
    te = [t for t in trades if t.entry_t >= CUT[tf]]
    a, bb, c = stats(tr), stats(te), stats(trades)
    return dict(strategy=strategy, tf=tf, **{f"train_{k}": v for k, v in a.items()},
                **{f"test_{k}": v for k, v in bb.items()}, **{f"full_{k}": v for k, v in c.items()})


# ----------------------------------------------------------------------------------- factor groups
MA_GROUPS = {
    "f_score": [{"ma_f": {"score_min": v}} for v in (-10, 0, 10)],
    "f_bull": [{"ma_f": {"bull": True}}],
    "f_vol": [{"ma_f": {"volx": v}} for v in (1.5, 2.0, 3.0)],
    "f_c_gt_ma5": [{"ma_f": {"c_gt_ma5": True}}],
    "f_rsi14": [{"ma_f": {"rsi14_max": v}} for v in (30, 40)],
    "f_dist120": [{"ma_f": {"dist120_min": v}} for v in (0.03, 0.06, 0.10)],
    "stop_bars": [{"ma_stop_bars": v} for v in (None, 10, 20, 30, 100)],
    "hard_stop": [{"hard_stop": v} for v in (0.05, 0.08, 0.10)],
    "tp_scale": [{"tp_scale": v} for v in (0.5, 0.75, 1.5, 2.0)],
    "trail_scale": [{"trail_scale": v} for v in (0.5, 1.5, 2.0)],
    "after_first": [{"after_first": v} for v in (0.0, 0.005, 0.02, 0.03)],
    "hold_scale": [{"hold_scale": v} for v in (0.5, 2.0)],
}
TREND_GROUPS = {
    "f_score": [{"trend_f": {"score_min": v}} for v in (-10, 0, 10)],
    "f_bull": [{"trend_f": {"bull": True}}],
    "f_vol": [{"trend_f": {"volx": v}} for v in (1.5, 2.0)],
    "f_dist_mid": [{"trend_f": {"dist_mid_max": v}} for v in (0.01, 0.02, 0.04)],
    "f_rsi_min": [{"trend_f": {"rsi_min": v}} for v in (35, 40, 45)],
    "rsi_arm": [{"trend_rsi": v} for v in (25, 35, 40)],
    "watch_scale": [{"trend_watch_scale": v} for v in (0.5, 2.0, 3.0)],
    "stop_scale": [{"trend_stop_scale": v} for v in (0.25, 0.5, 2.0)] + [{"trend_stop_off": True}],
    "hard_stop": [{"hard_stop": v} for v in (0.05, 0.08, 0.10)],
    "tp_scale": [{"tp_scale": v} for v in (0.5, 0.75, 1.5, 2.0)],
    "trail_scale": [{"trail_scale": v} for v in (0.5, 1.5, 2.0)],
    "after_first": [{"after_first": v} for v in (0.0, 0.005, 0.02, 0.03)],
    "hold_scale": [{"hold_scale": v} for v in (0.5, 2.0)],
}
GROUPS = {"ma": MA_GROUPS, "trend": TREND_GROUPS}


def label(over: dict) -> str:
    return "; ".join(f"{k}={v}" for k, v in over.items()) if over else "base"


def main() -> None:
    init()
    cases = [("ma", tf) for tf in S.FRAMES] + [("trend", tf) for tf in S.FRAMES]
    with Pool(4) as pool:
        # ---------------- Phase A: one factor at a time
        tasks, meta = [], []
        for strategy, tf in cases:
            tasks.append((strategy, tf, {})); meta.append((strategy, tf, "base", "base"))
            for g, vals in GROUPS[strategy].items():
                for v in vals:
                    tasks.append((strategy, tf, v)); meta.append((strategy, tf, g, label(v)))
        res = pool.map(run, tasks, chunksize=1)
        ofat = pd.DataFrame([dict(group=m[2], setting=m[3], **r) for m, r in zip(meta, res)])
        ofat.to_csv(S.RESULTS / "apk_improve_ofat.csv", index=False)
        print("phase A done:", len(ofat), "runs", flush=True)

        # ---------------- Phase B: greedy forward selection on TRAIN portfolio return
        rows = []
        for strategy, tf in cases:
            sub = ofat[(ofat.strategy == strategy) & (ofat.tf == tf)]
            base = sub[sub.group == "base"].iloc[0]
            min_n = 20
            cur_over, cur_train = {}, base.train_port
            cur_row = base
            used = set()
            rows.append(dict(strategy=strategy, tf=tf, step=0, added="base", over="{}", **{k: base[k] for k in base.index if k.startswith(("train_", "test_", "full_"))}))
            for step in range(1, 6):
                cands = []
                for g, vals in GROUPS[strategy].items():
                    if g in used:
                        continue
                    # take the best 2 single-factor values of this group on train to limit combinations
                    gs = sub[sub.group == g].sort_values("train_port", ascending=False).head(2)
                    for _, r in gs.iterrows():
                        v = next(x for x in vals if label(x) == r.setting)
                        cands.append((g, v))
                if not cands:
                    break
                tk = [(strategy, tf, merge(cur_over, v)) for g, v in cands]
                out = pool.map(run, tk, chunksize=1)
                best = None
                for (g, v), o in zip(cands, out):
                    if o["train_n"] >= min_n and o["train_port"] > cur_train + 0.02 and (best is None or o["train_port"] > best[2]["train_port"]):
                        best = (g, v, o)
                if best is None:
                    break
                g, v, o = best
                used.add(g)
                cur_over = merge(cur_over, v); cur_train = o["train_port"]
                rows.append(dict(strategy=strategy, tf=tf, step=step, added=label(v), over=repr(cur_over), **{k: o[k] for k in o if k.startswith(("train_", "test_", "full_"))}))
            print(strategy, tf, "greedy steps:", len([r for r in rows if r["strategy"] == strategy and r["tf"] == tf]) - 1, flush=True)
        pd.DataFrame(rows).to_csv(S.RESULTS / "apk_improve_greedy.csv", index=False)


if __name__ == "__main__":
    main()
