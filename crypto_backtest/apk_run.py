"""Run the three APK (0.2.1) coin strategies - reference / MA / MarketTrend - on Upbit candles for 5m/30m/60m/day.

Base setting: slippage 0.05%/side + fee 0.05%/side, same-bar ambiguity resolved stop-first, per-frame wallets from
FRAME_CFG (5m/30m: 10 slots 200k/100k, 60m: 5 slots 400k/200k, day: 4 slots 500k/250k), Trend day watch = 5 days.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import apk_strategies as S

COINS = ["BTC", "ETH", "XRP", "SOL", "DOGE", "ADA", "AVAX", "LINK", "DOT", "TRX", "XLM", "ATOM",
         "ETC", "BCH", "NEAR", "AAVE", "UNI", "SUI", "APT", "ARB", "OP", "HBAR", "POL"]
BASE_SLIP = 0.0005


def collect(fn):
    out = {}
    for tf in S.FRAMES:
        trades, starts, ends = [], [], []
        n = 0
        for coin in COINS:
            b = S.load_bars(coin, tf)
            if b is None:
                continue
            n += 1
            starts.append(b.t[0]); ends.append(b.t[-1])
            trades += fn(coin, tf, b)
        out[tf] = (trades, min(starts), max(ends), n)
    return out


def main() -> None:
    S.RESULTS.mkdir(exist_ok=True)
    ends, score = S.build_market_score(COINS)
    sc = S.ScoreLookup(ends, score)
    ok = score[~np.isnan(score)]
    print(f"market score: {len(ok)} hourly points, mean {ok.mean():.1f}, >0 {(ok > 0).mean():.1%}, <=-10 {(ok <= -10).mean():.1%}",
          flush=True)
    pd.Series(score, index=pd.to_datetime(ends)).dropna().to_csv(S.RESULTS / "apk_market_score_hourly.csv", header=["score"])
    nosc = S.ScoreLookup(np.array([0], dtype="int64"), np.array([100.0]))
    main_fns = {
        "reference": lambda c, tf, b: S.reference_trades(c, tf, b, sc),
        "ma": lambda c, tf, b: S.ma_trades(c, tf, b, sc),
        "trend": lambda c, tf, b: S.trend_trades(c, tf, b, sc),
    }
    rows, all_trades = [], []

    def add(strategy, variant, data, slip):
        S.SLIP = slip
        for tf, (trades, start, end, n) in data.items():
            row = S.summarize(trades, strategy, tf, start, end, n)
            row["variant"] = variant
            row["slippage_per_side"] = slip
            rows.append(row)
            if variant == "base":
                print(f"{strategy:10s} {tf:4s} trades={len(trades):5d} win={row.get('win_rate', float('nan')):6.1%} "
                      f"avg={row.get('avg_trade_ret', float('nan')):7.2%} pf={row.get('profit_factor', float('nan')):5.2f} "
                      f"port={row.get('portfolio_return', float('nan')):8.1%} skipped={row.get('skipped_no_slot', 0)}", flush=True)
                all_trades.extend(dict(strategy=strategy, tf=tf, coin=t.coin, entry_t=t.entry_t, entry=t.entry,
                                       exit_t=t.exit_t, ret=t.ret, size_factor=t.size_factor,
                                       reasons="|".join(t.reasons), open_at_end=t.open_at_end) for t in trades)

    # ---- base: stop-first, slippage 0.05%; plus slippage sensitivity on the SAME trades
    S.ORDER_MODE = "stop_first"
    base_data = {}
    for strategy, fn in main_fns.items():
        base_data[strategy] = collect(fn)
        add(strategy, "base", base_data[strategy], BASE_SLIP)
        add(strategy, "slippage_0", base_data[strategy], 0.0)
        add(strategy, "slippage_0.10pct", base_data[strategy], 0.0010)
    # ---- sensitivity: previous same-bar convention (OHLC path)
    S.ORDER_MODE = "path"
    for strategy, fn in main_fns.items():
        add(strategy, "same_bar_order_path_heuristic", collect(fn), BASE_SLIP)
    S.ORDER_MODE = "stop_first"
    # ---- sensitivity: score gate off (reference), repo angle (trend), daily watch 24h (trend)
    add("reference", "no_score_gate", collect(lambda c, tf, b: S.reference_trades(c, tf, b, nosc, use_score=False)), BASE_SLIP)
    add("trend", "angle_repo_-15_+20_strict", collect(lambda c, tf, b: S.trend_trades(c, tf, b, sc, -15.0, 20.0, strict=True)), BASE_SLIP)
    saved = S.TREND_WATCH_H["day"]
    S.TREND_WATCH_H["day"] = 24
    add("trend", "day_watch_24h", collect(lambda c, tf, b: S.trend_trades(c, tf, b, sc)), BASE_SLIP)
    S.TREND_WATCH_H["day"] = saved
    S.SLIP = BASE_SLIP
    pd.DataFrame(rows).to_csv(S.RESULTS / "apk_strategy_summary.csv", index=False)
    pd.DataFrame(all_trades).to_csv(S.RESULTS / "apk_strategy_trades.csv", index=False)


if __name__ == "__main__":
    main()
