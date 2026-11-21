"""Run the three APK coin strategies (reference / MA / MarketTrend) on Upbit candles for every timeframe."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import apk_strategies as S

COINS = ["BTC", "ETH", "XRP", "SOL", "DOGE", "ADA", "AVAX", "LINK", "DOT", "TRX", "XLM", "ATOM",
         "ETC", "BCH", "NEAR", "AAVE", "UNI", "SUI", "APT", "ARB", "OP", "HBAR", "POL"]


def main() -> None:
    S.RESULTS.mkdir(exist_ok=True)
    ends, score = S.build_market_score(COINS)
    sc = S.ScoreLookup(ends, score)
    ok = score[~np.isnan(score)]
    print(f"market score: {len(ok)} hourly points, mean {ok.mean():.1f}, >0 share {(ok > 0).mean():.1%}, "
          f"<-10 share {(ok < -10).mean():.1%}", flush=True)
    pd.Series(score, index=pd.to_datetime(ends)).dropna().to_csv(S.RESULTS / "apk_market_score_hourly.csv",
                                                                header=["score"])
    rows, all_trades = [], []
    nosc = S.ScoreLookup(np.array([0], dtype="int64"), np.array([100.0]))   # score gate disabled
    plan = [("reference", S.FRAMES), ("ma", S.FRAMES), ("trend", S.TREND["frames"]),
            ("reference_no_score_gate", S.FRAMES), ("trend_no_slope_filter", S.TREND["frames"]),
            ("ma_v067_variant", S.FRAMES)]
    for strategy, frames in plan:
        for tf in frames:
            trades, starts, ends_t, ncoin = [], [], [], 0
            for coin in COINS:
                b = S.load_bars(coin, tf)
                if b is None:
                    continue
                ncoin += 1
                starts.append(b.t[0]); ends_t.append(b.t[-1])
                if strategy == "reference":
                    trades += S.reference_trades(coin, tf, b, sc)
                elif strategy == "reference_no_score_gate":
                    trades += S.reference_trades(coin, tf, b, nosc)
                elif strategy == "trend_no_slope_filter":
                    trades += S.trend_trades(coin, tf, b, slope_filter=False)
                elif strategy == "ma":
                    trades += S.ma_trades(coin, tf, b, "new_legacy")
                elif strategy == "ma_v067_variant":
                    trades += S.ma_trades(coin, tf, b, "v067")
                else:
                    trades += S.trend_trades(coin, tf, b)
            row = S.summarize(trades, strategy, tf, min(starts), max(ends_t), ncoin)
            rows.append(row)
            all_trades += [dict(strategy=strategy, tf=tf, coin=t.coin, entry_t=t.entry_t, entry=t.entry,
                                exit_t=t.exit_t, ret=t.ret, reasons="|".join(t.reasons),
                                open_at_end=t.open_at_end) for t in trades]
            print(f"{strategy:10s} {tf:4s} coins={ncoin} trades={len(trades)} "
                  f"win={row.get('win_rate', float('nan')):.1%} avg={row.get('avg_trade_ret', float('nan')):.2%} "
                  f"pf={row.get('profit_factor', float('nan')):.2f} port={row.get('portfolio_return', float('nan')):.1%}",
                  flush=True)
    print("\nreference funnel:")
    for k, v in sorted(S.FUNNEL.items()):
        print(" ", k, v)
    res = pd.DataFrame(rows)
    res.to_csv(S.RESULTS / "apk_strategy_summary.csv", index=False)
    pd.DataFrame(all_trades).to_csv(S.RESULTS / "apk_strategy_trades.csv", index=False)


if __name__ == "__main__":
    main()
