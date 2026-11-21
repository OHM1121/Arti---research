"""Grid search for parameter combinations with profit factor >= 1.01 (net of costs).

Protocol (decided before running):
  1. Search on the IN-SAMPLE period only (2020-09 ~ 2022-12).
  2. Keep combos with IS profit factor >= 1.01 and at least 30 IS trades.
  3. Check those combos on the OUT-OF-SAMPLE period (2023-01 ~ 2026-10). OOS is never used to pick.

Profit factor = sum of winning trade P&L / |sum of losing trade P&L|, trades assigned by exit date.
With ~1,300 combos, some will pass IS by chance. The OOS column is what shows whether any survive.
"""

from __future__ import annotations

import itertools
from dataclasses import replace

import pandas as pd

import backtest as bt

IS_START, IS_END = None, pd.Timestamp("2022-12-31")
OOS_START = pd.Timestamp("2023-01-01")
MIN_IS_TRADES = 30
TARGET_PF = 1.01

GRID = {
    "min_score": [2, 3, 4, 5],
    "top_n_candidates": [5, 10],
    "breakout_vol_mult": [1.0, 1.5, 2.0],
    "stop_atr": [1.5, 2.0, 3.0],
    "trail_atr": [2.0, 3.0, 4.0],
    "entry_mode": ["both", "breakout", "pullback"],
    "market_filter": [True, False],
    "funding_filter": [True, False],
}


def pf_of(trades: pd.DataFrame, start=None, end=None) -> tuple[float, int, float]:
    t = trades
    if start is not None:
        t = t[t["exit_date"] >= start]
    if end is not None:
        t = t[t["exit_date"] <= end]
    if len(t) == 0:
        return float("nan"), 0, float("nan")
    win = t.loc[t.pnl > 0, "pnl"].sum()
    loss = t.loc[t.pnl <= 0, "pnl"].sum()
    pf = win / abs(loss) if loss != 0 else float("inf")
    return pf, len(t), (t.pnl > 0).mean()


def main() -> None:
    p = bt.load_panel()
    ind = bt.add_indicators(p)
    keys = list(GRID)
    combos = list(itertools.product(*[GRID[k] for k in keys]))
    rows = []
    for vals in combos:
        params = dict(zip(keys, vals))
        cfg = replace(bt.Config(), **params)
        eq, inv, tr, _ = bt.simulate(p, ind, cfg)
        is_pf, is_n, is_win = pf_of(tr, end=IS_END)
        oos_pf, oos_n, oos_win = pf_of(tr, start=OOS_START)
        full_pf, full_n, _ = pf_of(tr)
        m = bt.metrics(eq, None, inv)
        rows.append({
            **params,
            "IS_PF": is_pf, "IS_trades": is_n, "IS_win": is_win,
            "OOS_PF": oos_pf, "OOS_trades": oos_n, "OOS_win": oos_win,
            "FULL_PF": full_pf, "FULL_trades": full_n,
            "CAGR": m["CAGR"], "MDD": m["MDD"], "avg_invested": m["avg_invested"],
        })
    df = pd.DataFrame(rows)
    bt.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(bt.RESULT_DIR / "grid_search.csv", index=False)

    n = len(df)
    is_pass = df[(df.IS_PF >= TARGET_PF) & (df.IS_trades >= MIN_IS_TRADES)]
    oos_pass_among = is_pass[(is_pass.OOS_PF >= TARGET_PF) & (is_pass.OOS_trades >= MIN_IS_TRADES)]
    full_pass = df[(df.FULL_PF >= TARGET_PF) & (df.FULL_trades >= MIN_IS_TRADES)]

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
    print(f"combos tested: {n}")
    print(f"IS PF>=1.01 & IS trades>=30: {len(is_pass)}")
    print(f"  of those, OOS PF>=1.01 & OOS trades>=30: {len(oos_pass_among)}")
    print(f"FULL-period PF>=1.01 (for reference only): {len(full_pass)}")
    print(f"combos with OOS PF>=1.01 regardless of IS: {int(((df.OOS_PF >= TARGET_PF) & (df.OOS_trades >= MIN_IS_TRADES)).sum())}")
    print(f"median OOS PF across all combos: {df.OOS_PF.median():.3f}")
    print(f"median IS PF across all combos: {df.IS_PF.median():.3f}")

    print("\nTop 15 by IS PF (with OOS shown for reference):")
    cols = keys + ["IS_PF", "IS_trades", "OOS_PF", "OOS_trades", "OOS_win", "CAGR", "MDD"]
    print(df.sort_values("IS_PF", ascending=False).head(15)[cols].to_string(index=False))

    if len(is_pass):
        print("\nIS-passing combos sorted by IS PF (first 20):")
        print(is_pass.sort_values("IS_PF", ascending=False).head(20)[cols].to_string(index=False))
    print(f"\nFull table: {bt.RESULT_DIR / 'grid_search.csv'}")


if __name__ == "__main__":
    main()
