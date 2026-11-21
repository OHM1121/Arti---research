"""The ORIGINAL trend-following rules (README.md table) applied to Upbit KRW candles on 1m/5m/30m/1h/1d.

Rules are exactly backtest.py's (market filter BTC > 200-bar MA, 5-condition score with MA20/60/120 alignment,
volume-surge breakout or MA20 pullback entry, 2/3 ATR stops, MA20 2-bar exit, weekly score refresh) with
**every window expressed in bars** (1 bar plays the role of 1 day), so the same rule set is evaluated on each
timeframe:   200-bar BTC MA, MA20/60/120, ATR14, 20-bar breakout high, 365-bar 12-1 momentum / 52w high,
126-bar relative strength, "weekly" refresh = every 7 bars (Sunday close on 1d, as in the original).

Differences from the Binance daily study (unavoidable, see README): Upbit KRW universe (23 coins),
no funding-rate filter, fee 0.05% + slippage 0.05% per side (Upbit KRW fee; slippage ~ half-spread of the alts).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import backtest as bt

HERE = Path(__file__).resolve().parent
TF_DIR = HERE / "data" / "upbit_tf"
RESULTS = HERE / "results"
FREQ = {"1m": "1min", "5m": "5min", "30m": "30min", "1h": "1h", "1d": "1D"}
BARS_PER_YEAR = {"1d": 365}


def load_panel(tf: str) -> dict[str, pd.DataFrame]:
    files = sorted((TF_DIR / tf).glob("*.csv.gz"))
    frames: dict[str, pd.DataFrame] = {}
    for f in files:
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        if len(df) < 50:
            continue
        full = pd.date_range(df.index[0], df.index[-1], freq=FREQ[tf])
        d = df.reindex(full)
        d["close"] = d["close"].ffill()                      # no-trade bars -> flat bar
        for c in ("open", "high", "low"):
            d[c] = d[c].fillna(d["close"])
        d["value"] = d["value"].fillna(0.0)
        frames[f.name.split(".")[0]] = d
    if "BTC" not in frames:
        raise SystemExit(f"BTC missing for {tf}")
    end = min(d.index[-1] for d in frames.values())          # common end (fetch times differ)
    idx = pd.date_range(min(d.index[0] for d in frames.values()), end, freq=FREQ[tf])
    syms = sorted(frames, key=lambda s: (s != "BTC", s))     # BTC first
    p = {
        "open": pd.DataFrame({s: frames[s]["open"].reindex(idx) for s in syms}),
        "high": pd.DataFrame({s: frames[s]["high"].reindex(idx) for s in syms}),
        "low": pd.DataFrame({s: frames[s]["low"].reindex(idx) for s in syms}),
        "close": pd.DataFrame({s: frames[s]["close"].reindex(idx) for s in syms}),
        "quote_volume": pd.DataFrame({s: frames[s]["value"].reindex(idx) for s in syms}),
    }
    p["funding"] = pd.DataFrame(np.nan, index=idx, columns=syms)   # no funding data -> filter passes
    return p


def refresh_flags(idx: pd.DatetimeIndex, tf: str) -> np.ndarray | None:
    if tf == "1d":
        return None                                           # Sunday close, exactly as original
    return (np.arange(len(idx)) % 7) == 6                     # every 7 bars


def summarize(eq: pd.Series, inv: pd.Series, tr: pd.DataFrame) -> dict:
    daily = eq.resample("1D").last().dropna()
    r = daily.pct_change().dropna()
    out = {
        "start": eq.index[0], "end": eq.index[-1], "bars": len(eq),
        "total_return": eq.iloc[-1] / eq.iloc[0] - 1,
        "MDD": (eq / eq.cummax() - 1).min(),
        "Sharpe_daily_ann": r.mean() / r.std() * np.sqrt(365) if len(r) > 5 and r.std() > 0 else np.nan,
        "avg_invested": inv.mean(),
    }
    if len(tr):
        w, l = tr[tr.pnl > 0], tr[tr.pnl <= 0]
        out.update({"trades": len(tr), "win_rate": len(w) / len(tr), "avg_trade": tr["ret"].mean(),
                    "profit_factor": w.pnl.sum() / abs(l.pnl.sum()) if len(l) and l.pnl.sum() != 0 else np.nan,
                    "avg_hold_bars": tr["hold_bars"].mean()})
    else:
        out.update({"trades": 0})
    return out


def run_tf(tf: str, rows: list, trade_frames: list) -> None:
    p = load_panel(tf)
    ind = bt.add_indicators(p)
    refresh = refresh_flags(p["close"].index, tf)
    base = bt.Config(fee=0.0005, slippage=0.0005, funding_filter=False)
    variants = {
        "base (rules as-is)": base,
        "gross (no cost)": replace(base, fee=0.0, slippage=0.0),
        "no market filter": replace(base, market_filter=False),
    }
    n_ok = (p["close"]["BTC"] > ind["ma200_btc"]).mean()
    print(f"[{tf}] coins={p['close'].shape[1]} bars={len(p['close'])} "
          f"{p['close'].index[0]} -> {p['close'].index[-1]}  BTC>MA200 share={n_ok:.1%}", flush=True)
    start = None
    for name, cfg in variants.items():
        eq, inv, tr, open_pos = bt.simulate(p, ind, cfg, refresh=refresh)
        start = eq.index[0]
        rows.append({"timeframe": tf, "strategy": name, "coins": p["close"].shape[1],
                     **summarize(eq, inv, tr), "open_at_end": len(open_pos)})
        if name.startswith("base") and len(tr):
            trade_frames.append(tr.assign(timeframe=tf))
    # benchmarks over the same window
    c = p["close"].loc[start:]
    btc = c["BTC"].dropna()
    ew = c.pct_change().where(c.shift(1).notna()).mean(axis=1).fillna(0.0)
    for name, curve in (("BTC buy&hold", btc / btc.iloc[0]), ("equal-weight (rebal each bar)", (1 + ew).cumprod())):
        rows.append({"timeframe": tf, "strategy": name, "coins": c.shape[1], "start": curve.index[0],
                     "end": curve.index[-1], "bars": len(curve), "total_return": curve.iloc[-1] / curve.iloc[0] - 1,
                     "MDD": (curve / curve.cummax() - 1).min(),
                     "Sharpe_daily_ann": (lambda r: r.mean() / r.std() * np.sqrt(365))(
                         curve.resample("1D").last().pct_change().dropna())})


def main(tfs: list[str]) -> None:
    RESULTS.mkdir(exist_ok=True)
    rows: list[dict] = []
    trades: list[pd.DataFrame] = []
    for tf in tfs:
        if not (TF_DIR / tf).exists() or not any((TF_DIR / tf).glob("*.csv.gz")):
            print(f"[{tf}] no data, skipped")
            continue
        run_tf(tf, rows, trades)
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "upbit_strategy_summary.csv", index=False)
    if trades:
        pd.concat(trades).to_csv(RESULTS / "upbit_strategy_trades.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
    cols = ["timeframe", "strategy", "start", "end", "total_return", "MDD", "Sharpe_daily_ann",
            "trades", "win_rate", "profit_factor", "avg_hold_bars", "avg_invested"]
    print(res[[c for c in cols if c in res.columns]].to_string(index=False))
    if trades:
        t = pd.concat(trades)
        print("\nexit reasons (base):")
        print(t.groupby(["timeframe", "reason"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    import sys
    main(sys.argv[1:] or ["1d", "1h", "30m", "5m", "1m"])
