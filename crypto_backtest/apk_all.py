"""All-KRW-coin study with the APK's TOP-50 (by 24h trading value) universe reproduced historically.

For Trend and MA on 60m and day: (1) signal counts per year (all coins vs TOP50), (2) yearly wallet returns (TOP50),
(3) the common filters (Trend: close <= MA120*(1-Y); MA: coin N-bar return lags BTC by X) on the much larger sample,
(4) expanding-year walk-forward of the filter parameter.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import apk_family as Fm
import apk_strategies as S
import apk_universe as U
import apk_wf as W
import apk_yearly as Y

TOPN = 50


def main() -> None:
    tfs = sys.argv[1:] or ["60m", "day"]
    ends, score = S.build_market_score(W.SCORE_COINS)
    sc = S.ScoreLookup(ends, score)
    out_counts, out_year, out_grid, out_wf, out_picks = [], [], [], [], []
    for tf in tfs:
        coins = U.all_coins({"60m": "1h", "day": "1d", "30m": "30m", "5m": "5m"}[tf])
        uni = U.Universe(U.all_coins("1d"))
        print(f"[{tf}] coins with data: {len(coins)}", flush=True)
        for strategy in ("trend", "ma"):
            S.TREND_WATCH_H["day"] = 120
            T = Fm.table(strategy, tf, sc, coins=coins)
            flags = np.zeros(len(T), bool)
            for c, g in T.groupby("coin"):
                flags[g.index] = uni.in_top(c, pd.DatetimeIndex(g["sig_t"]), tf, TOPN)
            T["top50"] = flags
            T.to_pickle(f"/tmp/claude-0/all_{strategy}_{tf}.pkl")
            yrs = T.entry_t.dt.year
            cnt = pd.DataFrame({"all": T.groupby(yrs).size(), "top50": T[T.top50].groupby(yrs[T.top50]).size()}).fillna(0).astype(int)
            cnt.insert(0, "tf", tf); cnt.insert(0, "strategy", strategy)
            out_counts.append(cnt.reset_index().rename(columns={"entry_t": "year"}))
            print(strategy, tf, "signals all/top50:", len(T), int(flags.sum()), flush=True)
            T50 = T[T.top50].reset_index(drop=True)
            # yearly wallet (base) on the TOP50 universe
            y = Y.wallet_by_year(T50, tf); y.insert(0, "tf", tf); y.insert(0, "strategy", strategy); out_year.append(y)
            # filter grid + walk-forward on the TOP50 sample
            g = Fm.grid(T50, strategy, tf); out_grid.append(g)
            w, p = Fm.walk_forward(T50, strategy, tf)
            if w: out_wf.append(w)
            out_picks.append(p)
    pd.concat(out_counts).to_csv(S.RESULTS / "apk_all_counts.csv", index=False)
    pd.concat(out_year).to_csv(S.RESULTS / "apk_all_yearly_top50.csv", index=False)
    pd.concat(out_grid).to_csv(S.RESULTS / "apk_all_grid_top50.csv", index=False)
    pd.DataFrame(out_wf).to_csv(S.RESULTS / "apk_all_walkforward_top50.csv", index=False)
    pd.concat(out_picks).to_csv(S.RESULTS / "apk_all_picks_top50.csv", index=False)


if __name__ == "__main__":
    main()
