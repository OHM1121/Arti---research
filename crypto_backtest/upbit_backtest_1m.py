"""Upbit KRW 1-minute candle backtest (long-only spot, single coin all-in per strategy).

Execution model
  * Signals are computed on the CLOSE of bar t and filled at the OPEN of bar t+1 (no look-ahead).
  * Costs per side: Upbit KRW fee 0.05% + slippage 0.03% (configurable, see COSTS).
  * Minutes without trades are missing in Upbit data; they are forward-filled (flat bar, volume 0).
  * Equity is marked to market on every bar's close.

Strategies (parameters fixed in advance, NOT tuned):
  ema_cross    : long while EMA20 > EMA60
  donchian     : enter on close > prior 60-bar high, exit on close < prior 20-bar low
  rsi_revert   : enter on RSI14 < 25, exit on RSI14 > 50 or after 60 bars
  boll_revert  : enter on close < lower Bollinger(20, 2), exit on close > middle band or after 60 bars
  buy_hold     : benchmark
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "upbit_1m"
RESULTS = HERE / "results"
MIN_PER_YEAR = 365 * 24 * 60

COSTS = {  # per side
    "gross (no cost)": 0.0,
    "fee only (0.05%)": 0.0005,
    "base (fee 0.05% + slip 0.03%)": 0.0008,
}
BASE = "base (fee 0.05% + slip 0.03%)"


# ------------------------------------------------------------------ data

def load(coin: str) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"{coin}.csv.gz", index_col=0, parse_dates=True)
    full = pd.date_range(df.index[0], df.index[-1], freq="min")
    out = df.reindex(full)
    out["missing"] = out["close"].isna()
    out["close"] = out["close"].ffill()
    for c in ("open", "high", "low"):
        out[c] = out[c].fillna(out["close"])
    out[["volume", "value"]] = out[["volume", "value"]].fillna(0.0)
    return out


# ------------------------------------------------------------------ signals
# Each returns target[i] in {0,1}: desired position after the close of bar i.

def sig_ema_cross(df: pd.DataFrame, fast: int = 20, slow: int = 60) -> np.ndarray:
    c = df["close"]
    f, s = c.ewm(span=fast, adjust=False).mean(), c.ewm(span=slow, adjust=False).mean()
    t = (f > s).to_numpy(dtype=np.int8)
    t[: slow * 3] = 0
    return t


def _stateful(enter: np.ndarray, exit_: np.ndarray, max_hold: int | None = None) -> np.ndarray:
    n = len(enter)
    t = np.zeros(n, dtype=np.int8)
    pos, held = 0, 0
    for i in range(n):
        if pos:
            held += 1
            if exit_[i] or (max_hold is not None and held >= max_hold):
                pos = 0
        elif enter[i]:
            pos, held = 1, 0
        t[i] = pos
    return t


def sig_donchian(df: pd.DataFrame, n_in: int = 60, n_out: int = 20) -> np.ndarray:
    c = df["close"]
    hi = df["high"].rolling(n_in).max().shift(1)
    lo = df["low"].rolling(n_out).min().shift(1)
    return _stateful((c > hi).to_numpy(), (c < lo).to_numpy())


def sig_rsi_revert(df: pd.DataFrame, n: int = 14, lo: float = 25, hi: float = 50, max_hold: int = 60) -> np.ndarray:
    d = df["close"].diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rsi = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    rsi = rsi.fillna(50)
    rsi.iloc[: n * 5] = 50
    return _stateful((rsi < lo).to_numpy(), (rsi > hi).to_numpy(), max_hold)


def sig_boll_revert(df: pd.DataFrame, n: int = 20, k: float = 2.0, max_hold: int = 60) -> np.ndarray:
    c = df["close"]
    mid, sd = c.rolling(n).mean(), c.rolling(n).std()
    enter = (c < mid - k * sd).to_numpy()
    exit_ = (c > mid).to_numpy()
    return _stateful(enter, exit_, max_hold)


def sig_buy_hold(df: pd.DataFrame) -> np.ndarray:
    t = np.ones(len(df), dtype=np.int8)
    t[:1] = 0
    return t


STRATEGIES = {
    "ema_cross": sig_ema_cross,
    "donchian": sig_donchian,
    "rsi_revert": sig_rsi_revert,
    "boll_revert": sig_boll_revert,
    "buy_hold": sig_buy_hold,
}


# ------------------------------------------------------------------ simulation

def simulate(df: pd.DataFrame, target: np.ndarray, side_cost: float) -> tuple[pd.Series, pd.DataFrame]:
    """Return (equity series starting at 1.0, trades frame). Fill at next-bar open."""
    o, c = df["open"].to_numpy(), df["close"].to_numpy()
    n = len(df)
    eq = np.ones(n)
    pos = 0
    entry_fill, entry_i = 0.0, 0
    cur = 1.0
    trades = []
    for i in range(1, n):
        want = target[i - 1]
        prev_c = c[i - 1]
        if pos == 0 and want == 1:  # buy at open
            fill = o[i] * (1 + side_cost)
            entry_fill, entry_i = fill, i
            pos = 1
            cur *= c[i] / fill
        elif pos == 1 and want == 0:  # sell at open
            fill = o[i] * (1 - side_cost)
            cur *= fill / prev_c
            trades.append((df.index[entry_i], df.index[i], entry_fill, fill, i - entry_i))
            pos = 0
        elif pos == 1:
            cur *= c[i] / prev_c
        eq[i] = cur
    if pos == 1:  # close the open position at the last close for trade stats
        fill = c[-1] * (1 - side_cost)
        trades.append((df.index[entry_i], df.index[-1], entry_fill, fill, n - 1 - entry_i))
        eq[-1] = cur * fill / c[-1]
    tr = pd.DataFrame(trades, columns=["entry", "exit", "entry_px", "exit_px", "bars"])
    if len(tr):
        tr["ret"] = tr["exit_px"] / tr["entry_px"] - 1
    else:
        tr["ret"] = []
    return pd.Series(eq, index=df.index), tr


def stats(eq: pd.Series, tr: pd.DataFrame, exposure: float) -> dict:
    daily = eq.resample("1D").last().pct_change().dropna()
    dd = eq / eq.cummax() - 1
    wins, losses = tr.loc[tr["ret"] > 0, "ret"], tr.loc[tr["ret"] <= 0, "ret"]
    return {
        "total_return": eq.iloc[-1] / eq.iloc[0] - 1,
        "MDD": dd.min(),
        "Sharpe_daily_ann": daily.mean() / daily.std() * np.sqrt(365) if daily.std() > 0 else np.nan,
        "trades": len(tr),
        "win_rate": (tr["ret"] > 0).mean() if len(tr) else np.nan,
        "avg_trade": tr["ret"].mean() if len(tr) else np.nan,
        "profit_factor": wins.sum() / -losses.sum() if losses.sum() < 0 else np.nan,
        "avg_hold_min": tr["bars"].mean() if len(tr) else np.nan,
        "exposure": exposure,
    }


# ------------------------------------------------------------------ main

def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    coins = sorted(p.name.split(".")[0] for p in DATA.glob("*.csv.gz"))
    if not coins:
        raise SystemExit("no data; run upbit_fetch_1m.py first")
    rows, all_trades, equities = [], [], {}
    for coin in coins:
        df = load(coin)
        mid = df.index[0] + (df.index[-1] - df.index[0]) * 2 / 3  # first 2/3 vs last 1/3 (reporting only)
        print(f"{coin}: {len(df)} bars {df.index[0]} -> {df.index[-1]}, "
              f"missing minutes {df['missing'].mean():.1%}")
        for sname, fn in STRATEGIES.items():
            target = fn(df)
            for cname, cost in COSTS.items():
                if sname == "buy_hold" and cname != BASE:
                    continue
                eq, tr = simulate(df, target, cost)
                for label, sl in (("full", slice(None)), ("first 2/3", df.index < mid), ("last 1/3", df.index >= mid)):
                    mask = np.ones(len(df), bool) if label == "full" else np.asarray(sl)
                    e = eq[mask]
                    e = e / e.iloc[0]
                    t = tr[(tr["entry"] >= e.index[0]) & (tr["entry"] <= e.index[-1])]
                    exp = float(np.mean(target[mask]))
                    rows.append({"coin": coin, "strategy": sname, "cost": cname, "period": label,
                                 "start": e.index[0], "end": e.index[-1], **stats(e, t, exp)})
                if cname == BASE:
                    equities[(coin, sname)] = eq
                    if len(tr):
                        all_trades.append(tr.assign(coin=coin, strategy=sname))
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "upbit_1m_summary.csv", index=False)
    pd.concat(all_trades).to_csv(RESULTS / "upbit_1m_trades.csv", index=False)

    # equal-weight portfolio across coins (each coin sleeve rebalanced never; simple average of equity curves)
    port = []
    for sname in STRATEGIES:
        eqs = pd.concat([equities[(c, sname)] for c in coins], axis=1).dropna().mean(axis=1)
        tr = pd.concat([t for t in all_trades if (t["strategy"] == sname).all()]) if any(
            (t["strategy"] == sname).all() for t in all_trades) else pd.DataFrame(columns=["ret", "bars"])
        port.append({"strategy": sname, "cost": BASE, **stats(eqs, tr, np.nan)})
    pd.DataFrame(port).to_csv(RESULTS / "upbit_1m_portfolio.csv", index=False)

    # cost sensitivity: total return vs per-side cost (fee 0.05% + slippage), all coins/strategies
    sens = []
    for coin in coins:
        df = load(coin)
        for sname in ("ema_cross", "donchian", "rsi_revert", "boll_revert"):
            target = STRATEGIES[sname](df)
            for cost in (0.0, 0.0005, 0.0008, 0.0010, 0.0012, 0.0015, 0.0020, 0.0030):
                eq, _ = simulate(df, target, cost)
                sens.append({"coin": coin, "strategy": sname, "cost_per_side": cost,
                             "total_return": eq.iloc[-1] - 1})
    sens = pd.DataFrame(sens)
    sens.to_csv(RESULTS / "upbit_1m_cost_sensitivity.csv", index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
    base_full = res[(res["cost"] == BASE) & (res["period"] == "full")]
    print("\n=== BASE cost, full period, per coin ===")
    print(base_full[["coin", "strategy", "total_return", "MDD", "Sharpe_daily_ann", "trades",
                     "win_rate", "avg_trade", "profit_factor", "avg_hold_min", "exposure"]].to_string(index=False))
    print("\n=== Equal-weight portfolio of coins (base cost) ===")
    print(pd.DataFrame(port).drop(columns=["cost"]).to_string(index=False))
    print("\n=== Total return vs per-side cost: median across coins / DOGE ===")
    print(sens[sens.coin != "DOGE"].pivot_table(index="strategy", columns="cost_per_side",
                                                 values="total_return", aggfunc="median").to_string())
    print(sens[sens.coin == "DOGE"].pivot_table(index="strategy", columns="cost_per_side",
                                                 values="total_return").to_string())
    print("\n=== Mean total return across coins by strategy / cost (full period) ===")
    print(res[res["period"] == "full"].pivot_table(index="strategy", columns="cost",
                                                   values="total_return", aggfunc="mean").to_string())
    print("\n=== Mean total return across coins: first 2/3 vs last 1/3 (base cost) ===")
    print(res[res["cost"] == BASE].pivot_table(index="strategy", columns="period",
                                               values="total_return", aggfunc="mean").to_string())


if __name__ == "__main__":
    main()
