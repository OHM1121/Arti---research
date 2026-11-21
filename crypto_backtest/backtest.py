"""Rule-based crypto trend strategy backtest (daily bars, Binance USDT spot, UTC).

Rules (see README.md for the full write-up):
  Market filter : BTC close > BTC 200-day MA -> new entries allowed, else none.
  Weekly (Sun)  : score each coin on 5 conditions (A-E), keep score >= 4, top 10 by score then momentum.
  Entry         : coin is a candidate AND (breakout: close > 20d high and quote volume >= 1.5x 20d avg
                  OR pullback: low touches MA20 and closes back above it on higher volume)
                  AND funding filter (avg funding per 8h <= 0.1%, skipped if no data).
                  Fill at the NEXT day's open.
  Sizing        : 1% of equity risked per trade with stop = 2 ATR14, capped at 10% of equity per
                  coin and 60% of equity invested in total.
  Exit          : initial stop = entry - 2 ATR14; trailing stop = highest high since entry - 3 ATR14
                  (the stop is the higher of the two); close < MA20 two days in a row; weekly score < 3;
                  no data for the coin (delisting) -> exit at last price.
  Costs         : fee 0.10% per side + slippage 0.10% per side (configurable).

No parameter in this file was tuned. The in-sample / out-of-sample split only reports stability.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
RESULT_DIR = HERE / "results"
WARMUP = 366  # need 365 days of history for the 12-1 momentum and 365-day high
IS_END = pd.Timestamp("2022-12-31")  # in-sample / out-of-sample split (reporting only)


@dataclass(frozen=True)
class Config:
    capital: float = 10_000.0
    market_filter: bool = True
    funding_filter: bool = True
    fee: float = 0.001
    slippage: float = 0.001
    risk_per_trade: float = 0.01
    max_position: float = 0.10
    max_exposure: float = 0.60
    max_positions: int = 10
    min_score: int = 4
    exit_score: int = 3
    top_n_candidates: int = 10
    stop_atr: float = 2.0
    trail_atr: float = 3.0
    breakout_vol_mult: float = 1.5
    funding_limit_per_8h: float = 0.001
    entry_mode: str = "both"  # "both" | "breakout" | "pullback"


# ---------------------------------------------------------------- data

def load_panel() -> dict[str, pd.DataFrame]:
    kl_dir = DATA_DIR / "klines"
    fu_dir = DATA_DIR / "funding"
    syms = sorted(p.stem for p in kl_dir.glob("*.csv"))
    if "BTC" not in syms:
        raise SystemExit("BTC klines missing; run crypto_backtest/fetch_data.py first")
    klines = {s: pd.read_csv(kl_dir / f"{s}.csv", index_col=0, parse_dates=True) for s in syms}
    funding = {
        s: pd.read_csv(fu_dir / f"{s}.csv", index_col=0, parse_dates=True)["funding_sum"]
        for s in syms
        if (fu_dir / f"{s}.csv").exists()
    }
    idx = pd.date_range(min(df.index.min() for df in klines.values()),
                        max(df.index.max() for df in klines.values()), freq="D")

    def wide(field: str) -> pd.DataFrame:
        return pd.DataFrame({s: klines[s][field].reindex(idx) for s in syms}, index=idx)

    panel = {f: wide(f) for f in ["open", "high", "low", "close", "quote_volume"]}
    panel["funding"] = pd.DataFrame(
        {s: funding[s].reindex(idx) if s in funding else np.nan for s in syms}, index=idx
    )
    return panel


def add_indicators(p: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    o, h, l, c, qv = p["open"], p["high"], p["low"], p["close"], p["quote_volume"]
    ind: dict[str, pd.DataFrame] = {}
    ind["ma20"] = c.rolling(20, min_periods=20).mean()
    ind["ma60"] = c.rolling(60, min_periods=60).mean()
    ind["ma120"] = c.rolling(120, min_periods=120).mean()
    prev_c = c.shift(1)
    tr = np.fmax(np.fmax((h - l).to_numpy(), (h - prev_c).abs().to_numpy()),
                 (l - prev_c).abs().to_numpy())
    ind["atr"] = pd.DataFrame(tr, index=c.index, columns=c.columns).rolling(14, min_periods=14).mean()
    ind["hh20"] = h.rolling(20, min_periods=20).max().shift(1)          # prior 20-day high
    ind["qv_avg20"] = qv.rolling(20, min_periods=20).mean().shift(1)    # prior 20-day avg volume
    ind["qv_ratio"] = qv.rolling(20).mean() / qv.rolling(120).mean()
    ind["mom"] = c.shift(30) / c.shift(365) - 1                          # 12-1 month momentum
    ind["prox"] = c / h.rolling(365, min_periods=365).max()             # distance to 52w high
    ret126 = c / c.shift(126) - 1
    ind["rs"] = ret126.sub(ret126["BTC"], axis=0)                       # relative strength vs BTC
    ind["fund8"] = p["funding"].rolling(3, min_periods=1).mean() / 3    # avg funding per 8h
    ind["ma200_btc"] = c["BTC"].rolling(200, min_periods=200).mean()

    mom_rank = ind["mom"].rank(axis=1, pct=True)
    A = mom_rank >= 0.8
    B = ind["prox"] >= 0.90
    C = (c > ind["ma20"]) & (ind["ma20"] > ind["ma60"]) & (ind["ma60"] > ind["ma120"])
    D = ind["qv_ratio"] >= 1.2
    E = ind["rs"] > 0
    ind["score"] = (A.astype(int) + B.astype(int) + C.astype(int)
                    + D.astype(int) + E.astype(int))
    return ind


# ---------------------------------------------------------------- simulation

def simulate(p: dict[str, pd.DataFrame], ind: dict[str, pd.DataFrame], cfg: Config,
             refresh: np.ndarray | None = None, warmup: int = WARMUP):
    """refresh: bool array marking the bars on which the weekly score/candidate refresh runs
    (default: Sundays, i.e. the original daily behaviour)."""
    close = p["close"]
    syms = list(close.columns)
    idx = close.index
    T, N = len(idx), len(syms)
    btc = syms.index("BTC")

    arr = lambda df: df.to_numpy(dtype=float)
    O, H, L, C = arr(p["open"]), arr(p["high"]), arr(p["low"]), arr(p["close"])
    QV = arr(p["quote_volume"])
    MA20, MA60, MA120 = arr(ind["ma20"]), arr(ind["ma60"]), arr(ind["ma120"])
    ATR, HH20, QVAVG = arr(ind["atr"]), arr(ind["hh20"]), arr(ind["qv_avg20"])
    FUND8, SCORE, MOM = arr(ind["fund8"]), arr(ind["score"]), arr(ind["mom"])
    market_ok = (close["BTC"] > ind["ma200_btc"]).to_numpy() if cfg.market_filter else np.ones(T, bool)
    dow = idx.dayofweek.to_numpy()
    refresh = (dow == 6) if refresh is None else np.asarray(refresh, dtype=bool)

    cash = cfg.capital
    pos: dict[int, dict] = {}
    pending: list[int] = []
    candidates: list[int] = []
    trades: list[dict] = []
    equity, invested_frac = [], []

    def close_pos(j: int, raw_px: float, t: int, reason: str) -> None:
        nonlocal cash
        pp = pos.pop(j)
        sell_px = raw_px * (1 - cfg.slippage)
        proceeds = pp["qty"] * sell_px * (1 - cfg.fee)
        cash += proceeds
        trades.append({
            "symbol": syms[j], "entry_date": pp["entry_date"], "exit_date": idx[t],
            "entry_px": pp["entry"], "exit_px": sell_px, "reason": reason,
            "pnl": proceeds - pp["cost"], "ret": proceeds / pp["cost"] - 1,
            "hold_days": (idx[t] - pp["entry_date"]).days, "hold_bars": t - pp["t0"],
        })

    for t in range(warmup, T):
        # 1) fill pending buys at today's open -------------------------------
        if pending:
            prev_eq = equity[-1] if equity else cfg.capital
            prev_invested = sum(pp["qty"] * pp["last"] for pp in pos.values())
            for j in pending:
                if j in pos or len(pos) >= cfg.max_positions or np.isnan(O[t, j]):
                    continue
                stop_dist = cfg.stop_atr * ATR[t - 1, j]
                if not np.isfinite(stop_dist) or stop_dist <= 0:
                    continue
                buy_px = O[t, j] * (1 + cfg.slippage)
                notional = min(
                    cfg.risk_per_trade * prev_eq / stop_dist * buy_px,   # risk-based size
                    cfg.max_position * prev_eq,                          # per-coin cap
                    cfg.max_exposure * prev_eq - prev_invested,          # total exposure cap
                    cash / (1 + cfg.fee),
                )
                if notional < 0.01 * prev_eq:
                    continue
                qty = notional / buy_px
                cost = notional * (1 + cfg.fee)
                cash -= cost
                prev_invested += notional
                pos[j] = {
                    "qty": qty, "entry": buy_px, "entry_date": idx[t], "cost": cost,
                    "stop0": buy_px - stop_dist, "hh": H[t, j],
                    "atr_prev": ATR[t, j], "below": 0, "last": O[t, j], "t0": t,
                }
            pending = []

        # 2) manage open positions with today's bar -------------------------
        for j in list(pos):
            pp = pos[j]
            if np.isnan(C[t, j]) or np.isnan(O[t, j]):
                close_pos(j, pp["last"], t, "no_data")  # delisted / data gap
                continue
            trail = pp["hh"] - cfg.trail_atr * pp["atr_prev"]
            stop = max(pp["stop0"], trail)
            if O[t, j] <= stop:
                close_pos(j, O[t, j], t, "stop_gap")
                continue
            if L[t, j] <= stop:
                close_pos(j, stop, t, "stop")
                continue
            pp["hh"] = max(pp["hh"], H[t, j])
            if not np.isnan(ATR[t, j]):
                pp["atr_prev"] = ATR[t, j]
            pp["last"] = C[t, j]
            pp["below"] = pp["below"] + 1 if C[t, j] < MA20[t, j] else 0
            if pp["below"] >= 2:
                close_pos(j, C[t, j], t, "ma20_2d")
                continue
            if refresh[t] and SCORE[t, j] < cfg.exit_score:
                close_pos(j, C[t, j], t, "score_drop")

        # 3) entry triggers at today's close -> fill next open --------------
        if market_ok[t] and candidates:
            for j in candidates:
                if j in pos or j in pending or np.isnan(C[t, j]):
                    continue
                trend = C[t, j] > MA20[t, j] > MA60[t, j] > MA120[t, j]
                breakout = C[t, j] > HH20[t, j] and QV[t, j] >= cfg.breakout_vol_mult * QVAVG[t, j]
                pullback = (L[t, j] <= MA20[t, j] * 1.01 and C[t, j] > MA20[t, j]
                            and C[t, j] > O[t, j] and QV[t, j] > QV[t - 1, j])
                f = FUND8[t, j]
                fund_ok = (not cfg.funding_filter) or np.isnan(f) or f <= cfg.funding_limit_per_8h
                if cfg.entry_mode == "breakout":
                    trigger = breakout
                elif cfg.entry_mode == "pullback":
                    trigger = pullback
                else:
                    trigger = breakout or pullback
                if trend and trigger and fund_ok:
                    pending.append(j)
            pending = pending[: max(0, cfg.max_positions - len(pos))]

        # 4) mark to market --------------------------------------------------
        invested = 0.0
        for j, pp in pos.items():
            if not np.isnan(C[t, j]):
                pp["last"] = C[t, j]
            invested += pp["qty"] * pp["last"]
        equity.append(cash + invested)
        invested_frac.append(invested / equity[-1] if equity[-1] > 0 else 0.0)

        # 5) weekly candidate refresh (Sunday close) -------------------------
        if refresh[t]:
            scores = SCORE[t]
            moms = MOM[t]
            ok = [j for j in range(N) if scores[j] >= cfg.min_score and not np.isnan(moms[j])]
            ok.sort(key=lambda j: (-scores[j], -moms[j]))
            candidates = ok[: cfg.top_n_candidates]

    eq = pd.Series(equity, index=idx[warmup:], name="equity")
    inv = pd.Series(invested_frac, index=idx[warmup:], name="invested_frac")
    tr = pd.DataFrame(trades)
    open_pos = [syms[j] for j in pos]
    return eq, inv, tr, open_pos


# ---------------------------------------------------------------- metrics

def metrics(eq: pd.Series, trades: pd.DataFrame | None = None, inv: pd.Series | None = None) -> dict:
    eq = eq.dropna()
    r = eq.pct_change().dropna()
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    out = {
        "start": eq.index[0].date().isoformat(),
        "end": eq.index[-1].date().isoformat(),
        "total_return": eq.iloc[-1] / eq.iloc[0] - 1,
        "CAGR": (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1,
        "MDD": (eq / eq.cummax() - 1).min(),
        "vol_ann": r.std() * np.sqrt(365),
        "Sharpe": r.mean() / r.std() * np.sqrt(365) if r.std() > 0 else np.nan,
    }
    if inv is not None:
        out["avg_invested"] = inv.mean()
    if trades is not None and len(trades):
        w, l = trades[trades.pnl > 0], trades[trades.pnl <= 0]
        out.update({
            "trades": len(trades),
            "win_rate": len(w) / len(trades),
            "avg_win": w["ret"].mean() if len(w) else np.nan,
            "avg_loss": l["ret"].mean() if len(l) else np.nan,
            "profit_factor": w["pnl"].sum() / abs(l["pnl"].sum()) if len(l) and l["pnl"].sum() != 0 else np.nan,
            "avg_hold_days": trades["hold_days"].mean(),
        })
    return out


def slice_metrics(eq, trades, inv, start=None, end=None):
    e = eq.loc[start:end]
    i = inv.loc[start:end] if inv is not None else None
    tr = None
    if trades is not None and len(trades):
        mask = trades["exit_date"].between(pd.Timestamp(start) if start else trades["exit_date"].min(),
                                           pd.Timestamp(end) if end else trades["exit_date"].max())
        tr = trades[mask]
    return metrics(e, tr, i)


def benchmarks(p: dict[str, pd.DataFrame], start: pd.Timestamp, cfg: Config):
    c = p["close"].loc[start:]
    btc = c["BTC"].dropna()
    btc_eq = cfg.capital * btc / btc.iloc[0]
    rets = c.pct_change().where(c.shift(1).notna())
    ew = rets.mean(axis=1).fillna(0.0)
    ew_eq = cfg.capital * (1 + ew).cumprod()
    return btc_eq.rename("BTC buy&hold"), ew_eq.rename("equal-weight universe (daily rebal)")


def yearly(eq: pd.Series) -> pd.Series:
    ye = eq.resample("YE").last()
    first = eq.iloc[0]
    ret = ye.pct_change()
    ret.iloc[0] = ye.iloc[0] / first - 1
    ret.index = ret.index.year
    return ret.rename("return")


# ---------------------------------------------------------------- main

def run(p, ind, cfg):
    return simulate(p, ind, cfg)


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    p = load_panel()
    ind = add_indicators(p)
    base = Config()
    variants = {
        "base": base,
        "gross (no costs)": replace(base, fee=0.0, slippage=0.0),
        "no market filter": replace(base, market_filter=False),
        "no funding filter": replace(base, funding_filter=False),
        # Exploratory only: looser score threshold. Not part of the original rules.
        "EXPLORATORY min_score=3": replace(base, min_score=3),
        "EXPLORATORY min_score=2": replace(base, min_score=2),
    }

    results = {}
    for name, cfg in variants.items():
        eq, inv, tr, open_pos = run(p, ind, cfg)
        results[name] = (eq, inv, tr, open_pos)

    eq0, inv0, tr0, open0 = results["base"]
    start = eq0.index[0]
    btc_eq, ew_eq = benchmarks(p, start, base)

    rows = []
    for name, (eq, inv, tr, _) in results.items():
        rows.append({"strategy": name, "period": "full", **slice_metrics(eq, tr, inv)})
        rows.append({"strategy": name, "period": "IS (~2022)",
                     **slice_metrics(eq, tr, inv, None, IS_END)})
        rows.append({"strategy": name, "period": "OOS (2023~)",
                     **slice_metrics(eq, tr, inv, IS_END + pd.Timedelta(days=1), None)})
    for name, bm in [("BTC buy&hold", btc_eq), ("equal-weight universe", ew_eq)]:
        rows.append({"strategy": name, "period": "full", **metrics(bm)})
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULT_DIR / "summary.csv", index=False)

    eq0.to_csv(RESULT_DIR / "equity_base.csv")
    tr0.to_csv(RESULT_DIR / "trades_base.csv", index=False)
    yearly_tbl = pd.DataFrame({
        "base": yearly(eq0), "BTC": yearly(btc_eq), "EW universe": yearly(ew_eq),
    })
    yearly_tbl.to_csv(RESULT_DIR / "yearly_returns.csv")

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
    show = summary[["strategy", "period", "start", "end", "total_return", "CAGR", "MDD", "Sharpe",
                    "trades", "win_rate", "profit_factor", "avg_hold_days", "avg_invested"]]
    print(show.to_string(index=False))
    print("\nYearly returns (base vs benchmarks):")
    print(yearly_tbl.to_string())
    print("\nBase: trades by exit reason:")
    print(tr0["reason"].value_counts().to_string())
    print(f"\nOpen positions at end: {open0}")


if __name__ == "__main__":
    main()
