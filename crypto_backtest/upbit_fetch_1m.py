"""Upbit KRW 1-minute candle downloader (public API, no key needed).

Usage: python3 upbit_fetch_1m.py [days] [COIN ...]
Saves data/upbit_1m/<COIN>.csv.gz  (index = UTC minute, columns open/high/low/close/volume/value).
Minutes without any trade are absent from Upbit's response; they are filled in the backtest.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import requests

URL = "https://api.upbit.com/v1/candles/minutes/1"
OUT = Path(__file__).resolve().parent / "data" / "upbit_1m"
DEFAULT_COINS = ["BTC", "ETH", "XRP", "SOL", "DOGE"]
MIN_INTERVAL = 0.125  # ~8 req/s, under Upbit's 10 req/s candle limit


def fetch(coin: str, days: int) -> pd.DataFrame:
    end = pd.Timestamp.now("UTC").tz_localize(None).floor("min")
    start = end - pd.Timedelta(days=days)
    to = end + pd.Timedelta(minutes=1)
    rows: list[dict] = []
    sess = requests.Session()
    last_call = 0.0
    while to > start:
        wait = MIN_INTERVAL - (time.time() - last_call)
        if wait > 0:
            time.sleep(wait)
        last_call = time.time()
        r = sess.get(URL, params={"market": f"KRW-{coin}", "count": 200,
                                  "to": to.strftime("%Y-%m-%dT%H:%M:%S")}, timeout=20)
        if r.status_code == 429:
            time.sleep(1.5)
            continue
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        to = pd.Timestamp(batch[-1]["candle_date_time_utc"])
        if len(rows) % 20000 < 200:
            print(f"{coin}: {len(rows)} candles, back to {to}", flush=True)
    df = pd.DataFrame(rows)
    df.index = pd.to_datetime(df["candle_date_time_utc"])
    df = df.rename(columns={"opening_price": "open", "high_price": "high", "low_price": "low",
                            "trade_price": "close", "candle_acc_trade_volume": "volume",
                            "candle_acc_trade_price": "value"})
    df = df[["open", "high", "low", "close", "volume", "value"]].astype(float)
    df = df[~df.index.duplicated()].sort_index()
    return df[df.index >= start]


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 180
    coins = sys.argv[2:] or DEFAULT_COINS
    OUT.mkdir(parents=True, exist_ok=True)
    for coin in coins:
        df = fetch(coin, days)
        df.index.name = "t"
        df.to_csv(OUT / f"{coin}.csv.gz")
        print(f"{coin}: saved {len(df)} candles {df.index[0]} -> {df.index[-1]}", flush=True)


if __name__ == "__main__":
    main()
