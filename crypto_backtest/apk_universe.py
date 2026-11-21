"""Historical 'TOP-N by trading value' universe (the APK scans only the top-N KRW coins by 24h trading value).

Rank = cross-sectional rank of each coin's DAILY KRW trade value among all coins with data that day (day files, since 2017).
For intraday signals the rank of the PREVIOUS completed day is used (no look-ahead); for daily signals the rank of the
signal day itself (that day is complete when the signal bar closes).
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

import apk_strategies as S


def all_coins(tf_dir: str = "1h") -> list[str]:
    return sorted(os.path.basename(f).split(".")[0] for f in glob.glob(str(S.TF_DIR / tf_dir / "*.csv.gz")))


def rank_table(coins: list[str]) -> pd.DataFrame:
    vals = {}
    for c in coins:
        f = S.TF_DIR / "1d" / f"{c}.csv.gz"
        if f.exists():
            d = pd.read_csv(f, index_col=0, parse_dates=True)
            vals[c] = d["value"]
    v = pd.DataFrame(vals).sort_index()
    return v.rank(axis=1, ascending=False, method="first")


class Universe:
    def __init__(self, coins: list[str]):
        self.rank = rank_table(coins)

    def in_top(self, coin: str, sig_times: pd.DatetimeIndex, tf: str, n: int) -> np.ndarray:
        if coin not in self.rank.columns:
            return np.zeros(len(sig_times), bool)
        day = sig_times.normalize()
        key = day if tf == "day" else day - pd.Timedelta(days=1)
        r = self.rank[coin].reindex(key).to_numpy()
        return ~np.isnan(r) & (r <= n)
