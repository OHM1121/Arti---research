"""Search for combos that win in DOWNTRENDS: high win rate AND positive P&L on trades entered in bear regime.

Definitions (fixed before running):
  Bear day   : BTC close < BTC 200-day MA on the signal day (the day before the entry fill).
  Bear trade : trade whose signal day is a bear day.
  Protocol   : market filter must be OFF, otherwise the strategy never trades in a bear regime.
               Select on IS bear trades (2020-09 ~ 2022-12): >= 20 trades, win rate >= 50%, P&L > 0.
               Test the selected combos on OOS bear trades (2023-01 ~ 2026-10) without re-selecting.
"""

from __future__ import annotations

import itertools
from dataclasses import replace

import numpy as np
import pandas as pd

import backtest as bt
from search import GRID

IS_END = pd.Timestamp("2022-12-31")
OOS_START = pd.Timestamp("2023-01-01")
MIN_BEAR_TRADES = 20
MIN_WIN = 0.50


def bear_flag(p, ind) -> pd.Series:
    return (p["close"]["BTC"] < ind["ma200_btc"]).fillna(False)


def bear_stats(tr: pd.DataFrame, bear: pd.Series, start=None, end=None) -> dict:
    if len(tr) == 0:
        return {"n": 0, "win": np.nan, "pnl_pct": np.nan, "pf": np.nan}
    t = tr.copy()
    sig = pd.to_datetime(t["entry_date"]) - pd.Timedelta(days=1)
    t["bear"] = bear.reindex(sig.values).fillna(False).to_numpy()
    t = t[t["bear"]]
    t = t[t["exit_date"] >= start] if start is not None else t
    t = t[t["exit_date"] <= end] if end is not None else t
    if len(t) == 0:
        return {"n": 0, "win": np.nan, "pnl_pct": np.nan, "pf": np.nan}
    win = t.loc[t.pnl > 0, "pnl"].sum()
    loss = t.loc[t.pnl <= 0, "pnl"].sum()
    return {
        "n": len(t),
        "win": (t.pnl > 0).mean(),
        "pnl_pct": t["pnl"].sum() / bt.Config().capital,
        "pf": win / abs(loss) if loss != 0 else np.inf,
    }


def main() -> None:
    p = bt.load_panel()
    ind = bt.add_indicators(p)
    bear = bear_flag(p, ind)
    cap = bt.Config().capital
    is_days = bear.loc[:IS_END].iloc[366:].sum()
    oos_days = bear.loc[OOS_START:].sum()
    print(f"bear days (post-warmup): IS {is_days}, OOS {oos_days}, "
          f"BTC<MA200 share {bear.iloc[366:].mean():.2%}")

    keys = list(GRID)
    grid = {k: v for k, v in GRID.items() if k != "market_filter"}
    rows = []
    for vals in itertools.product(*[grid[k] for k in grid]):
        params = dict(zip(grid, vals))
        cfg = replace(bt.Config(), market_filter=False, **params)
        eq, inv, tr, _ = bt.simulate(p, ind, cfg)
        is_b = bear_stats(tr, bear, end=IS_END)
        oos_b = bear_stats(tr, bear, start=OOS_START)
        all_b = bear_stats(tr, bear)
        m = bt.metrics(eq, None, inv)
        rows.append({
            **params,
            "IS_bear_n": is_b["n"], "IS_bear_win": is_b["win"], "IS_bear_pnl_pct": is_b["pnl_pct"], "IS_bear_pf": is_b["pf"],
            "OOS_bear_n": oos_b["n"], "OOS_bear_win": oos_b["win"], "OOS_bear_pnl_pct": oos_b["pnl_pct"], "OOS_bear_pf": oos_b["pf"],
            "ALL_bear_n": all_b["n"], "ALL_bear_win": all_b["win"], "ALL_bear_pnl_pct": all_b["pnl_pct"],
            "CAGR": m["CAGR"], "MDD": m["MDD"], "total_return": m["total_return"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(bt.RESULT_DIR / "bear_search.csv", index=False)

    sel = df[(df.IS_bear_n >= MIN_BEAR_TRADES) & (df.IS_bear_win >= MIN_WIN) & (df.IS_bear_pnl_pct > 0)]
    test = sel[(sel.OOS_bear_n >= 1) & (sel.OOS_bear_win >= MIN_WIN) & (sel.OOS_bear_pnl_pct > 0)]
    print(f"\ncombos tested: {len(df)}")
    print(f"IS selection (>= {MIN_BEAR_TRADES} bear trades, win >= 50%, P&L > 0): {len(sel)}")
    print(f"  of those, OOS bear win >= 50% and P&L > 0: {len(test)}")
    print(f"combos with IS bear P&L > 0 (any win rate): {int((df.IS_bear_pnl_pct > 0).sum())}")
    print(f"combos with OOS bear P&L > 0 (any win rate): {int((df.OOS_bear_pnl_pct > 0).sum())}")
    print(f"median OOS bear win rate across all combos: {df.OOS_bear_win.median():.3f}")

    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
    cols = keys[:0] + [k for k in grid] + ["IS_bear_n", "IS_bear_win", "IS_bear_pnl_pct",
                                          "OOS_bear_n", "OOS_bear_win", "OOS_bear_pnl_pct", "CAGR", "MDD"]
    print("\nIS-selected combos, sorted by IS bear P&L (top 15):")
    print(sel.sort_values("IS_bear_pnl_pct", ascending=False).head(15)[cols].to_string(index=False))
    if len(test):
        print("\nPassing both IS and OOS (sorted by OOS bear P&L):")
        print(test.sort_values("OOS_bear_pnl_pct", ascending=False).head(15)[cols].to_string(index=False))
    print(f"\nFull table: {bt.RESULT_DIR / 'bear_search.csv'}")


if __name__ == "__main__":
    main()
