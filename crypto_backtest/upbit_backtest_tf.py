"""Multi-timeframe Upbit backtest (5m / 30m / 1h / 1d) reusing the 1m engine and strategies.

Two windows:
  A) "180d": the same 180-day window as the 1m study, bars resampled from the 1m data
     (timeframes 1m, 5m, 30m, 1h, 1d) -> directly comparable across timeframes.
  B) "4y": long history downloaded from Upbit (30m, 1h, 1d; see upbit_fetch_tf.py).

Strategy parameters are the SAME BAR COUNTS on every timeframe (EMA 20/60, Donchian 60/20,
RSI 14 <25/>50, Bollinger 20/2, 60-bar max hold for mean reversion), not re-tuned per timeframe.
Costs per side: fee 0.05% + slippage 0.03% = 0.08% (also reported: 0 and fee-only).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import upbit_backtest_1m as eng

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
TF_DIR = HERE / "data" / "upbit_tf"
COINS = ["BTC", "ETH", "XRP", "SOL", "DOGE"]
FREQ = {"1m": "1min", "5m": "5min", "30m": "30min", "1h": "1h", "1d": "1D"}
COSTS = {"gross": 0.0, "fee_only": 0.0005, "base": 0.0008}
STRATS = ["ema_cross", "donchian", "rsi_revert", "boll_revert", "buy_hold"]


def resample(df1m: pd.DataFrame, tf: str) -> pd.DataFrame:
    if tf == "1m":
        return df1m[["open", "high", "low", "close", "volume"]]
    r = df1m.resample(FREQ[tf], label="left", closed="left")
    out = pd.DataFrame({"open": r["open"].first(), "high": r["high"].max(), "low": r["low"].min(),
                        "close": r["close"].last(), "volume": r["volume"].sum()})
    return out.dropna(subset=["close"])


def load_long(coin: str, tf: str) -> pd.DataFrame:
    df = pd.read_csv(TF_DIR / tf / f"{coin}.csv.gz", index_col=0, parse_dates=True)
    full = pd.date_range(df.index[0], df.index[-1], freq=FREQ[tf])
    out = df.reindex(full)
    out["close"] = out["close"].ffill()
    for c in ("open", "high", "low"):
        out[c] = out[c].fillna(out["close"])
    out["volume"] = out["volume"].fillna(0.0)
    return out[["open", "high", "low", "close", "volume"]]


def run(df: pd.DataFrame, window: str, tf: str, coin: str, rows: list, trades: list) -> None:
    for sname in STRATS:
        target = eng.STRATEGIES[sname](df)
        for cname, cost in COSTS.items():
            if sname == "buy_hold" and cname != "base":
                continue
            eq, tr = eng.simulate(df, target, cost)
            rows.append({"window": window, "timeframe": tf, "coin": coin, "strategy": sname, "cost": cname,
                         "bars": len(df), "start": df.index[0], "end": df.index[-1],
                         **eng.stats(eq, tr, float(np.mean(target)))})
            if cname == "base" and len(tr):
                trades.append(tr.assign(window=window, timeframe=tf, coin=coin, strategy=sname))


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    rows: list[dict] = []
    trades: list[pd.DataFrame] = []
    for coin in COINS:
        df1m = eng.load(coin)
        for tf in ("1m", "5m", "30m", "1h", "1d"):
            run(resample(df1m, tf), "180d", tf, coin, rows, trades)
        for tf in ("30m", "1h", "1d"):
            if (TF_DIR / tf / f"{coin}.csv.gz").exists():
                run(load_long(coin, tf), "4y", tf, coin, rows, trades)
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "upbit_tf_summary.csv", index=False)
    pd.concat(trades).to_csv(RESULTS / "upbit_tf_trades.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
    order_tf = ["1m", "5m", "30m", "1h", "1d"]
    base = res[res["cost"] == "base"]
    for window in ("180d", "4y"):
        w = base[base["window"] == window]
        if w.empty:
            continue
        print(f"\n######## window {window} : total return per coin (base cost) ########")
        for sname in STRATS:
            p = w[w.strategy == sname].pivot_table(index="timeframe", columns="coin", values="total_return")
            p = p.reindex([t for t in order_tf if t in p.index])
            p["median"] = p.median(axis=1)
            print(f"\n-- {sname}")
            print(p.to_string())
        print(f"\n-- median across coins: trades / win_rate / profit_factor / MDD (base cost)")
        g = w.groupby(["strategy", "timeframe"])[["trades", "win_rate", "profit_factor", "MDD"]].median()
        g = g.reindex(pd.MultiIndex.from_product([STRATS, order_tf]), fill_value=np.nan).dropna(how="all")
        print(g.to_string())
        print(f"\n-- median total return across coins by cost (gross / fee_only / base)")
        c = res[(res["window"] == window) & (res.strategy != "buy_hold")].pivot_table(
            index=["strategy", "timeframe"], columns="cost", values="total_return", aggfunc="median")
        print(c[["gross", "fee_only", "base"]].to_string())


if __name__ == "__main__":
    main()
