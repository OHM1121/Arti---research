"""Upbit KRW candle downloader for 30m / 1h / 1d (long history).

Usage: python3 upbit_fetch_tf.py [days] [COIN ...]
Saves data/upbit_tf/<tf>/<COIN>.csv.gz  (index = UTC bar start).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import requests

OUT = Path(__file__).resolve().parent / "data" / "upbit_tf"
DEFAULT_COINS = ["BTC", "ETH", "XRP", "SOL", "DOGE"]
ENDPOINTS = {
    "30m": "https://api.upbit.com/v1/candles/minutes/30",
    "1h": "https://api.upbit.com/v1/candles/minutes/60",
    "1d": "https://api.upbit.com/v1/candles/days",
}
MIN_INTERVAL = 0.125


def fetch(coin: str, tf: str, days: int) -> pd.DataFrame:
    end = pd.Timestamp.now("UTC").tz_localize(None).floor("min")
    start = end - pd.Timedelta(days=days)
    to = end + pd.Timedelta(days=1)
    rows: list[dict] = []
    sess = requests.Session()
    last = 0.0
    while to > start:
        wait = MIN_INTERVAL - (time.time() - last)
        if wait > 0:
            time.sleep(wait)
        last = time.time()
        r = sess.get(ENDPOINTS[tf], params={"market": f"KRW-{coin}", "count": 200,
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
    df = pd.DataFrame(rows)
    df.index = pd.to_datetime(df["candle_date_time_utc"])
    df = df.rename(columns={"opening_price": "open", "high_price": "high", "low_price": "low",
                            "trade_price": "close", "candle_acc_trade_volume": "volume",
                            "candle_acc_trade_price": "value"})
    df = df[["open", "high", "low", "close", "volume", "value"]].astype(float)
    df = df[~df.index.duplicated()].sort_index()
    return df[df.index >= start]


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1460
    coins = sys.argv[2:] or DEFAULT_COINS
    for tf in ENDPOINTS:
        (OUT / tf).mkdir(parents=True, exist_ok=True)
    for coin in coins:
        for tf in ENDPOINTS:
            df = fetch(coin, tf, days)
            df.index.name = "t"
            df.to_csv(OUT / tf / f"{coin}.csv.gz")
            print(f"{coin} {tf}: saved {len(df)} bars {df.index[0]} -> {df.index[-1]}", flush=True)


if __name__ == "__main__":
    main()
