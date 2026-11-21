"""Calendar-year wallet returns (fresh wallet every year) per strategy and timeframe.

Wallet: per-frame seed 2,000,000 KRW, FRAME_CFG slots / per-symbol amount, fee 0.05% + slippage 0.05% per side, stop-first.
A trade belongs to the year of its ENTRY and is followed to its exit even if that falls after the year end.
Part A: the 12 base cases (23 coins).  Part B: 45 coins, base vs the best common filter (MA lag-vs-BTC, Trend MA120 discount).
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

import apk_run as R
import apk_strategies as S

TFS = ("5m", "30m", "60m", "day")


def wallet_by_year(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    rows = []
    for y, d in df.groupby(df["entry_t"].dt.year):
        objs = [SimpleNamespace(entry_t=a, exit_t=b, ret=c, size_factor=f)
                for a, b, c, f in zip(d["entry_t"], d["exit_t"], d["ret"], d["size_factor"])]
        p = S.portfolio(objs, tf)
        r = d["ret"].to_numpy()
        rows.append(dict(year=int(y), trades=len(r), win=float((r > 0).mean()), avg=float(r.mean()),
                         wallet=p["portfolio_return"], mdd=p["portfolio_MDD_realized"], skipped=p["skipped_no_slot"],
                         first=d["entry_t"].min(), last=d["entry_t"].max()))
    return pd.DataFrame(rows)


def part_a() -> pd.DataFrame:
    S.SLIP, S.ORDER_MODE = 0.0005, "stop_first"
    ends, score = S.build_market_score(R.COINS)
    sc = S.ScoreLookup(ends, score)
    fns = {"reference": lambda c, tf, b: S.reference_trades(c, tf, b, sc),
           "ma": lambda c, tf, b: S.ma_trades(c, tf, b, sc),
           "trend": lambda c, tf, b: S.trend_trades(c, tf, b, sc)}
    out = []
    for strat, fn in fns.items():
        data = R.collect(fn)
        for tf, (trades, start, end, n) in data.items():
            df = pd.DataFrame([dict(entry_t=t.entry_t, exit_t=t.exit_t, ret=t.ret, size_factor=t.size_factor) for t in trades])
            y = wallet_by_year(df, tf)
            y.insert(0, "tf", tf); y.insert(0, "strategy", strat); y["data_start"] = start; y["data_end"] = end
            out.append(y)
    return pd.concat(out, ignore_index=True)


FILTERS = {  # best common-filter setting per timeframe (apk_family study, full-sample, descriptive)
    ("ma", "5m"): ("rel20", 0.01), ("ma", "30m"): ("rel120", 0.15), ("ma", "60m"): ("rel40", 0.02), ("ma", "day"): ("rel60", 0.20),
    ("trend", "5m"): ("d_ma120", 0.05), ("trend", "30m"): ("d_ma120", 0.10), ("trend", "60m"): ("d_ma120", 0.12),
    ("trend", "day"): ("d_ma120", 0.03),
}


def part_b() -> pd.DataFrame:
    out = []
    for (strat, tf), (col, th) in FILTERS.items():
        T = pd.read_pickle(f"/tmp/claude-0/fam_{strat}_{tf}.pkl")
        for name, d in (("base", T), (f"filter {col}<=-{th*100:g}%", T[T[col] <= -th])):
            y = wallet_by_year(d, tf)
            y.insert(0, "variant", name); y.insert(0, "tf", tf); y.insert(0, "strategy", strat)
            out.append(y)
    return pd.concat(out, ignore_index=True)


def main() -> None:
    a = part_a(); a.to_csv(S.RESULTS / "apk_yearly_base.csv", index=False)
    b = part_b(); b.to_csv(S.RESULTS / "apk_yearly_filtered_45coins.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
    for name, df, keys in (("A (23개 코인, 기본)", a, ["strategy", "tf"]), ("B (45개 코인, 기본 vs 필터)", b, ["strategy", "tf", "variant"])):
        print(f"\n=== {name}  지갑수익률(%) [승률% / 거래수]")
        for k, g in df.groupby(keys, sort=False):
            print(" ".join(str(x) for x in k).ljust(34) + " | " + " | ".join(
                f"{int(r.year)}: {r.wallet*100:+.1f}% [{r.win*100:.0f}/{int(r.trades)}]" for r in g.itertuples()))


if __name__ == "__main__":
    main()
