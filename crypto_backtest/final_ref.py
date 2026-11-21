"""Final verification (2026-11-21): reference-candle (기준봉) trade table with the reference bar's features, so that the same
common filters (depth below MA120, lag vs BTC) can be tested.  Re-implements the reference_trades loop only to record the
reference bar index (apk_strategies.py is not modified)."""
from __future__ import annotations
import sys
import numpy as np, pandas as pd
import apk_strategies as S, apk_family as Fm, apk_universe as U, apk_wf as W

def ref_table(tf, sc, coins, uni, topn=35):
    S.SLIP, S.ORDER_MODE = 0.0005, "stop_first"; S.CFG.update(ma_f={}, trend_f={})
    btc = S.load_bars("BTC", tf); bc = pd.Series(btc.c, index=btc.t)
    rows = []
    for coin in coins:
        b = S.load_bars(coin, tf)
        if b is None: continue
        n = len(b.c); cand, _ = S.reference_candidates(b, tf); idx = np.flatnonzero(cand)
        life = S.hours_to_bars(S.REF_LIFE_H[tf], tf); busy = -1
        ma120 = pd.Series(b.c).rolling(120, min_periods=120).mean().to_numpy()
        bcc = bc.reindex(b.t).to_numpy()
        recs = []
        for pos, i in enumerate(idx):
            if i <= busy: continue
            limit = b.l[i]; nxt = idx[pos + 1] if pos + 1 < len(idx) else n
            last = min(n - 1, i + life, nxt); fill = None
            for j in range(i + 1, last + 1):
                if b.l[j] <= limit:
                    if not (sc.at(b.t[j]) > S.SCORE_BUY_MIN): continue
                    fill = j; break
            if fill is None: continue
            j = fill; E = min(limit, b.o[j]); tr = S.Trade("reference", tf, coin, j, b.t[j], E)
            stop = max(0.0, b.l[i] - (b.h[i] - b.l[i]))
            busy = S.manage(tr, b, tf, "reference", sc, hold_end=j + S.hours_to_bars(S.REF_HOLD_H[tf], tf), hard_stop=stop, lb_stop=None, trail=S.TRAIL[tf], entry_mode="down_only")
            d = dict(coin=coin, sig_t=b.t[i], entry_t=tr.entry_t, exit_t=tr.exit_t, ret=tr.ret, size_factor=tr.size_factor,
                     d_ma120=b.c[i] / ma120[i] - 1 if not np.isnan(ma120[i]) else np.nan,
                     fill_d_ma120=E / ma120[i] - 1 if not np.isnan(ma120[i]) else np.nan,
                     rng=(b.h[i] - b.l[i]) / b.l[i], stop_dist=(E - stop) / E)
            for N in Fm.NS:
                d[f"rel{N}"] = ((b.c[i] / b.c[i - N] - 1) - (bcc[i] / bcc[i - N] - 1)) if i >= N and not np.isnan(bcc[i]) and not np.isnan(bcc[i-N]) else np.nan
            recs.append(d)
        rows += recs
    T = pd.DataFrame(rows).sort_values("entry_t").reset_index(drop=True)
    flags = np.zeros(len(T), bool)
    for c, g in T.groupby("coin"):
        flags[g.index] = uni.in_top(c, pd.DatetimeIndex(g["sig_t"]), tf, topn)
    T["top35"] = flags
    return T

if __name__ == "__main__":
    tfs = sys.argv[1:] or ["60m", "day"]
    ends, score = S.build_market_score(W.SCORE_COINS); sc = S.ScoreLookup(ends, score)
    uni = U.Universe(U.all_coins("1d"))
    for tf in tfs:
        coins = U.all_coins({"60m": "1h", "day": "1d", "30m": "30m", "5m": "5m"}[tf])
        T = ref_table(tf, sc, coins, uni)
        T.to_pickle(S.RESULTS / "cache" / f"final_ref_{tf}.pkl.gz")
        print(tf, len(T), T.top35.sum(), flush=True)
