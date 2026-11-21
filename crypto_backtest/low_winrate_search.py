"""Search for combos that make money even with a LOW win rate (win rate < 50%).

Pre-declared criteria (fixed before running):
  IS  (2020-09 ~ 2022-12): total return > 0, >= 30 trades, win rate < 50%
  OOS (2023-01 ~ 2026-10): total return > 0, >= 30 trades, win rate < 50%
  Selection is on IS only; OOS is the check. Ranking among passers: min(IS, OOS) total return.
  Period return is equity change within the window (includes open positions, marked to market).
"""

from __future__ import annotations

import itertools
from dataclasses import replace

import numpy as np
import pandas as pd

import backtest as bt
from search import GRID, pf_of

IS_END = pd.Timestamp("2022-12-31")
OOS_START = pd.Timestamp("2023-01-01")
MIN_TRADES = 30
MAX_WIN = 0.50


def period_return(eq: pd.Series, start=None, end=None) -> float:
    e = eq.loc[:end] if end is not None else eq
    if start is None:
        return e.iloc[-1] / e.iloc[0] - 1
    before = eq.loc[:start - pd.Timedelta(days=1)]
    base = before.iloc[-1] if len(before) else eq.iloc[0]
    after = eq.loc[start:]
    return after.iloc[-1] / base - 1 if len(after) else np.nan


def main() -> None:
    p = bt.load_panel()
    ind = bt.add_indicators(p)
    keys = list(GRID)
    rows = []
    for vals in itertools.product(*[GRID[k] for k in keys]):
        params = dict(zip(keys, vals))
        eq, inv, tr, _ = bt.simulate(p, ind, replace(bt.Config(), **params))
        is_ret = period_return(eq, end=IS_END)
        oos_ret = period_return(eq, start=OOS_START)
        _, is_n, is_win = pf_of(tr, end=IS_END)
        _, oos_n, oos_win = pf_of(tr, start=OOS_START)
        m = bt.metrics(eq, None, inv)
        rows.append({
            **params,
            "IS_ret": is_ret, "IS_trades": is_n, "IS_win": is_win,
            "OOS_ret": oos_ret, "OOS_trades": oos_n, "OOS_win": oos_win,
            "CAGR": m["CAGR"], "MDD": m["MDD"], "avg_invested": m["avg_invested"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(bt.RESULT_DIR / "low_winrate_search.csv", index=False)

    ok_is = (df.IS_ret > 0) & (df.IS_trades >= MIN_TRADES) & (df.IS_win < MAX_WIN)
    ok_oos = (df.OOS_ret > 0) & (df.OOS_trades >= MIN_TRADES) & (df.OOS_win < MAX_WIN)
    passed = df[ok_is & ok_oos].copy()
    passed["min_ret"] = passed[["IS_ret", "OOS_ret"]].min(axis=1)

    print(f"combos tested: {len(df)}")
    print(f"IS: return>0, >=30 trades, win<50%: {int(ok_is.sum())}")
    print(f"IS and OOS both pass: {len(passed)}")
    print(f"overall OOS return>0 (any trades/win): {int((df.OOS_ret > 0).sum())}")
    print(f"median OOS return all combos: {df.OOS_ret.median():.3f}")

    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
    cols = keys + ["IS_ret", "IS_trades", "IS_win", "OOS_ret", "OOS_trades", "OOS_win", "CAGR", "MDD", "avg_invested"]
    if len(passed):
        print("\nPassing both periods, ranked by min(IS, OOS) return (top 20):")
        print(passed.sort_values("min_ret", ascending=False).head(20)[cols].to_string(index=False))
    else:
        print("\nNo combo passed both periods.")
        print("\nIS-passing combos ranked by OOS return (for reference, top 10):")
        print(df[ok_is].sort_values("OOS_ret", ascending=False).head(10)[cols].to_string(index=False))
    print(f"\nFull table: {bt.RESULT_DIR / 'low_winrate_search.csv'}")


if __name__ == "__main__":
    main()
