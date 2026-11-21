"""Win rate and profit of each strategy over the most recent 1 year, with the rules unchanged.

Window: last 365 days of available data (default 2025-11-21 ~ 2026-11-20).
  - return   : equity change inside the window (positions open at the window start are included,
               marked to market). Base = equity at the close of the day before the window.
  - win rate : share of trades with P&L > 0, trades counted by EXIT date inside the window.
  - P&L      : sum of trade P&L (starting capital 10,000) for those trades.
Funding data is not available for 2026-11 (monthly files only), so the funding filter does not
block anything on those days.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

import backtest as bt

STRATEGIES = {
    "A 원래 규칙": {},
    "B 탐색용 min_score=3": {"min_score": 3},
    "C 탐색용 min_score=2": {"min_score": 2},
    "D 워크포워드 선택 (pullback)": dict(min_score=3, top_n_candidates=5, breakout_vol_mult=1.0, stop_atr=3.0,
                                      trail_atr=4.0, entry_mode="pullback", market_filter=True, funding_filter=True),
    "E 손익비 탐색 상위 (breakout, 필터 ON)": dict(min_score=2, top_n_candidates=10, breakout_vol_mult=2.0, stop_atr=3.0,
                                              trail_atr=4.0, entry_mode="breakout", market_filter=True, funding_filter=False),
    "F 저승률 탐색 OOS 수익 1위": dict(min_score=2, top_n_candidates=10, breakout_vol_mult=1.0, stop_atr=2.0,
                                  trail_atr=4.0, entry_mode="breakout", market_filter=False, funding_filter=False),
    "G 승률<35% 1위": dict(min_score=2, top_n_candidates=10, breakout_vol_mult=2.0, stop_atr=3.0,
                       trail_atr=3.0, entry_mode="breakout", market_filter=True, funding_filter=False),
}


def window_stats(eq: pd.Series, tr: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, capital: float) -> dict:
    before = eq.loc[: start - pd.Timedelta(days=1)]
    base = before.iloc[-1] if len(before) else eq.iloc[0]
    w = eq.loc[start:end]
    series = pd.concat([pd.Series([base], index=[start - pd.Timedelta(days=1)]), w])
    t = tr[(tr["exit_date"] >= start) & (tr["exit_date"] <= end)] if len(tr) else tr
    out = {
        "return": w.iloc[-1] / base - 1,
        "MDD": (series / series.cummax() - 1).min(),
        "trades": len(t),
    }
    if len(t):
        wins, losses = t[t.pnl > 0], t[t.pnl <= 0]
        out.update({
            "win_rate": len(wins) / len(t),
            "pnl_$": t["pnl"].sum(),
            "pnl_pct_of_capital": t["pnl"].sum() / capital,
            "avg_win_%": wins["ret"].mean() if len(wins) else np.nan,
            "avg_loss_%": losses["ret"].mean() if len(losses) else np.nan,
            "profit_factor": wins["pnl"].sum() / abs(losses["pnl"].sum()) if losses["pnl"].sum() != 0 else np.inf,
        })
    else:
        out.update({"win_rate": np.nan, "pnl_$": 0.0, "pnl_pct_of_capital": 0.0,
                    "avg_win_%": np.nan, "avg_loss_%": np.nan, "profit_factor": np.nan})
    return out


def main() -> None:
    p = bt.load_panel()
    ind = bt.add_indicators(p)
    end = p["close"].index.max()
    start = end - pd.Timedelta(days=364)
    cap = bt.Config().capital
    print(f"data through {end.date()}  |  window {start.date()} ~ {end.date()}  ({(end - start).days + 1} days)\n")

    rows = []
    for name, params in STRATEGIES.items():
        eq, inv, tr, open_pos = bt.simulate(p, ind, replace(bt.Config(), **params))
        s = window_stats(eq, tr, start, end, cap)
        s["open_at_end"] = ",".join(open_pos) if open_pos else "-"
        # what fraction of window days the strategy held any position
        s["avg_invested"] = inv.loc[start:end].mean()
        rows.append({"strategy": name, **s})

    # Benchmarks, measured from the close of the day before the window, like the strategies.
    c = p["close"].loc[start - pd.Timedelta(days=1):end]
    btc = c["BTC"] / c["BTC"].iloc[0]
    daily = c.pct_change().where(c.shift(1).notna()).iloc[1:]
    ew = pd.concat([pd.Series([1.0]), (1 + daily.mean(axis=1).fillna(0)).cumprod().reset_index(drop=True)],
                   ignore_index=True)
    for name, series in [("BTC 단순 보유", btc.reset_index(drop=True)), ("동일가중 (일별 리밸런싱)", ew)]:
        rows.append({"strategy": name, "return": series.iloc[-1] - 1,
                     "MDD": (series / series.cummax() - 1).min()})

    df = pd.DataFrame(rows)
    df.to_csv(bt.RESULT_DIR / "recent_year.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
    print(df.drop(columns=["open_at_end"]).to_string(index=False))
    print("\nOpen positions at the end of data:")
    for _, row in df.dropna(subset=["trades"]).iterrows():
        print(f"  {row['strategy']}: {row['open_at_end']}")


if __name__ == "__main__":
    main()
