"""Common-filter family study (user request 2026-11-21).

MA          : take the signal only if the coin's return over the last N bars lags BTC's by at least X
              (coin_ret_N - btc_ret_N <= -X).   Same filter form on 5m / 30m / 60m / day; N and X are tuned per timeframe.
MarketTrend : take the signal only if the signal-bar close is at least Y below its MA120 (close/MA120 - 1 <= -Y).
              Same filter form on every timeframe; Y is tuned per timeframe.
Candidate trades = every signal of the APK 0.2.1 base rules (slippage 0.05%, stop-first, per-frame wallets), 45 coins
(5m: 23 coins).  A grid is scored on the FULL sample (descriptive) and by expanding-year WALK-FORWARD (strict out-of-sample).
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

import apk_strategies as S
import apk_wf as W

NS = (3, 5, 10, 20, 40, 60, 120)
XS = (0.0, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30)          # percent
YS = (0.0, 1, 2, 3, 5, 7, 10, 12, 15, 20, 25, 30)       # percent
TFS = ("5m", "30m", "60m", "day")


def table(strategy: str, tf: str, sc) -> pd.DataFrame:
    S.SLIP, S.ORDER_MODE = 0.0005, "stop_first"
    S.CFG.update(ma_f={}, trend_f={})
    btc = S.load_bars("BTC", tf)
    btc_c = pd.Series(btc.c, index=btc.t)
    rows = []
    for coin in W.ALL_COINS:
        b = S.load_bars(coin, tf)
        if b is None:
            continue
        tr = S.ma_trades(coin, tf, b, sc) if strategy == "ma" else S.trend_trades(coin, tf, b, sc)
        if not tr:
            continue
        idx = np.array([t.sig_i for t in tr])
        c = b.c
        bc = btc_c.reindex(b.t).to_numpy()
        ma120 = pd.Series(c).rolling(120, min_periods=120).mean().to_numpy()
        d = {"coin": coin, "entry_t": [t.entry_t for t in tr], "exit_t": [t.exit_t for t in tr],
             "ret": [t.ret for t in tr], "size_factor": [t.size_factor for t in tr],
             "d_ma120": c[idx] / ma120[idx] - 1}
        for n in NS:
            ok = idx >= n
            rc = np.where(ok, c[idx] / c[np.clip(idx - n, 0, None)] - 1, np.nan)
            rb = np.where(ok, bc[idx] / bc[np.clip(idx - n, 0, None)] - 1, np.nan)
            d[f"rel{n}"] = rc - rb
        rows.append(pd.DataFrame(d))
    return pd.concat(rows, ignore_index=True).sort_values("entry_t").reset_index(drop=True)


def stats(df: pd.DataFrame, tf: str) -> dict:
    if len(df) == 0:
        return dict(n=0, win=np.nan, avg=np.nan, t=np.nan, pf=np.nan, wallet=np.nan)
    r = df["ret"].to_numpy()
    objs = [SimpleNamespace(entry_t=a, exit_t=b, ret=c, size_factor=d)
            for a, b, c, d in zip(df["entry_t"], df["exit_t"], df["ret"], df["size_factor"])]
    w, l = r[r > 0], r[r <= 0]
    return dict(n=len(r), win=float((r > 0).mean()), avg=float(r.mean()), t=W.tstat(r),
                pf=float(w.sum() / -l.sum()) if l.sum() < 0 else np.nan, wallet=S.portfolio(objs, tf)["portfolio_return"])


def masks(T: pd.DataFrame, strategy: str):
    """Yield (label, params, boolean mask) for every grid point; first item is the unfiltered base."""
    yield "base", {}, np.ones(len(T), bool)
    if strategy == "ma":
        for n in NS:
            x = T[f"rel{n}"].to_numpy(float)
            for X in XS:
                yield f"N={n},X={X}%", {"N": n, "X": X}, ~np.isnan(x) & (x <= -X / 100)
    else:
        x = T["d_ma120"].to_numpy(float)
        for Y in YS:
            yield f"Y={Y}%", {"Y": Y}, ~np.isnan(x) & (x <= -Y / 100)


def grid(T: pd.DataFrame, strategy: str, tf: str) -> pd.DataFrame:
    rows = []
    for label, p, m in masks(T, strategy):
        rows.append(dict(strategy=strategy, tf=tf, setting=label, **p, **stats(T[m], tf)))
    return pd.DataFrame(rows)


def walk_forward(T: pd.DataFrame, strategy: str, tf: str, min_frac: float = 0.03):
    """Per fold: pick the grid point with the best train t-stat (n >= max(30, min_frac*train)) among closed trades."""
    pooled, picks = [], []
    for a, b in W.FOLDS[tf]:
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        tr = T[T["exit_t"] < a]
        te = T[(T["entry_t"] >= a) & (T["entry_t"] < b)]
        if len(tr) < 150 or len(te) < 30:
            continue
        best = None
        for label, p, m_all in masks(T, strategy):
            if label == "base":
                continue
            m_tr = m_all[tr.index]
            sel = tr[m_tr]
            if len(sel) < max(30, int(min_frac * len(tr))):
                continue
            t = W.tstat(sel["ret"].to_numpy())
            if best is None or t > best[0]:
                best = (t, label, p, m_all)
        if best is None:
            continue
        _, label, p, m_all = best
        sel_te = te[m_all[te.index]]
        pooled.append((te, sel_te))
        picks.append(dict(strategy=strategy, tf=tf, fold=str(a.date()), chosen=label, train_n=int(m_all[tr.index].sum()),
                          test_base_n=len(te), test_sel_n=len(sel_te),
                          test_base_avg=float(te["ret"].mean()), test_sel_avg=float(sel_te["ret"].mean()) if len(sel_te) else np.nan))
    if not pooled:
        return None, pd.DataFrame(picks)
    base = pd.concat([x[0] for x in pooled]); sel = pd.concat([x[1] for x in pooled])
    return (dict(strategy=strategy, tf=tf, **{f"base_{k}": v for k, v in stats(base, tf).items()},
                 **{f"wf_{k}": v for k, v in stats(sel, tf).items()}), pd.DataFrame(picks))


def main() -> None:
    ends, score = S.build_market_score(W.SCORE_COINS)
    sc = S.ScoreLookup(ends, score)
    grids, wfs, picks = [], [], []
    plan = [("ma", tf) for tf in TFS] + [("trend", tf) for tf in TFS]
    for strategy, tf in plan:
        if strategy == "trend":
            S.TREND_WATCH_H["day"] = 120          # user setting: daily watch 5 days
        T = table(strategy, tf, sc)
        T.to_pickle(f"/tmp/claude-0/fam_{strategy}_{tf}.pkl")
        g = grid(T, strategy, tf)
        grids.append(g)
        w, p = walk_forward(T, strategy, tf)
        if w: wfs.append(w)
        picks.append(p)
        print(f"[{strategy} {tf}] candidates {len(T)}", flush=True)
    pd.concat(grids).to_csv(S.RESULTS / "apk_family_grid.csv", index=False)
    pd.DataFrame(wfs).to_csv(S.RESULTS / "apk_family_walkforward.csv", index=False)
    pd.concat(picks).to_csv(S.RESULTS / "apk_family_picks.csv", index=False)
    # extra: trend day with a 10-day watch (more candidates) for reference only
    S.TREND_WATCH_H["day"] = 240
    T = table("trend", "day", sc); T.to_pickle("/tmp/claude-0/fam_trend10_day.pkl")
    g = grid(T, "trend", "day"); g["strategy"] = "trend_watch10d"
    g.to_csv(S.RESULTS / "apk_family_grid_trend_day10d.csv", index=False)
    print("[trend day, watch 10d] candidates", len(T))


if __name__ == "__main__":
    main()
