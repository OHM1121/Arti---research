"""Run the three APK (0.2.1) coin strategies - reference / MA / MarketTrend - on Upbit candles for 5m/30m/60m/day."""
from __future__ import annotations

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
    print(f"market score: {len(ok)} hourly points, mean {ok.mean():.1f}, >0 {(ok > 0).mean():.1%}, <=-10 {(ok <= -10).mean():.1%}",
          flush=True)
    pd.Series(score, index=pd.to_datetime(ends)).dropna().to_csv(S.RESULTS / "apk_market_score_hourly.csv", header=["score"])
    nosc = S.ScoreLookup(np.array([0], dtype="int64"), np.array([100.0]))      # score never blocks / exits
    runners = {
        # the 12 requested cases
        "reference": lambda c, tf, b: S.reference_trades(c, tf, b, sc),
        "ma": lambda c, tf, b: S.ma_trades(c, tf, b, sc),
        "trend": lambda c, tf, b: S.trend_trades(c, tf, b, sc),
        # sensitivity only
        "reference_no_score_gate": lambda c, tf, b: S.reference_trades(c, tf, b, nosc, use_score=False),
        "trend_angle_repo_-15_+20_strict": lambda c, tf, b: S.trend_trades(c, tf, b, sc, -15.0, 20.0, strict=True),
    }
    rows, all_trades = [], []
    for strategy, fn in runners.items():
        for tf in S.FRAMES:
            trades, starts, ends_t, ncoin = [], [], [], 0
            for coin in COINS:
                b = S.load_bars(coin, tf)
                if b is None:
                    continue
                ncoin += 1
                starts.append(b.t[0]); ends_t.append(b.t[-1])
                trades += fn(coin, tf, b)
            row = S.summarize(trades, strategy, tf, min(starts), max(ends_t), ncoin)
            rows.append(row)
            if strategy in ("reference", "ma", "trend"):
                all_trades += [dict(strategy=strategy, tf=tf, coin=t.coin, entry_t=t.entry_t, entry=t.entry,
                                    exit_t=t.exit_t, ret=t.ret, size_factor=t.size_factor,
                                    reasons="|".join(t.reasons), open_at_end=t.open_at_end) for t in trades]
            print(f"{strategy:34s} {tf:4s} coins={ncoin} trades={len(trades):5d} win={row.get('win_rate', float('nan')):6.1%} "
                  f"avg={row.get('avg_trade_ret', float('nan')):7.2%} pf={row.get('profit_factor', float('nan')):5.2f} "
                  f"port={row.get('portfolio_return', float('nan')):8.1%}", flush=True)
    print("\nreference funnel:")
    for k, v in sorted(S.FUNNEL.items()):
        print(" ", k, v)
    pd.DataFrame(rows).to_csv(S.RESULTS / "apk_strategy_summary.csv", index=False)
    pd.DataFrame(all_trades).to_csv(S.RESULTS / "apk_strategy_trades.csv", index=False)


if __name__ == "__main__":
    main()
