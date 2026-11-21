"""Upbit KRW candle downloader for a coin universe (1m / 5m / 30m / 1h / 1d), parallel with a global rate limit.

Usage: python3 upbit_fetch_tf.py            # default universe and per-timeframe history
Saves data/upbit_tf/<tf>/<COIN>.csv.gz  (index = UTC bar start). Existing files are skipped.
"""

from __future__ import annotations

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

OUT = Path(__file__).resolve().parent / "data" / "upbit_tf"
UNIVERSE = ["BTC", "ETH", "XRP", "SOL", "DOGE", "ADA", "AVAX", "LINK", "DOT", "TRX", "XLM", "ATOM",
            "ETC", "BCH", "NEAR", "AAVE", "UNI", "SUI", "APT", "ARB", "OP", "HBAR", "POL",
            # extension (2026-11-21): more coins -> more out-of-sample trades
            "SC", "SHIB", "STX", "FIL", "SAND", "ALGO", "AXS", "WAVES", "KAVA", "VET", "ELF", "EGLD", "LSK", "ARK",
            "GAS", "CHZ", "BAT", "NEO", "IOTA", "IMX", "GRT", "ANKR"]
ENDPOINTS = {
    "1m": "https://api.upbit.com/v1/candles/minutes/1",
    "5m": "https://api.upbit.com/v1/candles/minutes/5",
    "30m": "https://api.upbit.com/v1/candles/minutes/30",
    "1h": "https://api.upbit.com/v1/candles/minutes/60",
    "1d": "https://api.upbit.com/v1/candles/days",
}
DAYS = {"1d": 3300, "1h": 2190, "30m": 1460, "5m": 365, "1m": 30}
MIN_INTERVAL = 0.105  # global ~9.5 req/s, under Upbit's 10 req/s candle limit
_lock = threading.Lock()
_last = [0.0]


def _throttle() -> None:
    with _lock:
        wait = MIN_INTERVAL - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()


def fetch(coin: str, tf: str, days: int) -> pd.DataFrame:
    end = pd.Timestamp.now("UTC").tz_localize(None).floor("min")
    start = end - pd.Timedelta(days=days)
    to = end + pd.Timedelta(days=1)
    rows: list[dict] = []
    sess = requests.Session()
    while to > start:
        _throttle()
        r = sess.get(ENDPOINTS[tf], params={"market": f"KRW-{coin}", "count": 200,
                                            "to": to.strftime("%Y-%m-%dT%H:%M:%S")}, timeout=30)
        if r.status_code == 429:
            time.sleep(2)
            continue
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        to = pd.Timestamp(batch[-1]["candle_date_time_utc"])
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df.index = pd.to_datetime(df["candle_date_time_utc"])
    df = df.rename(columns={"opening_price": "open", "high_price": "high", "low_price": "low",
                            "trade_price": "close", "candle_acc_trade_volume": "volume",
                            "candle_acc_trade_price": "value"})
    df = df[["open", "high", "low", "close", "volume", "value"]].astype(float)
    df = df[~df.index.duplicated()].sort_index()
    return df[df.index >= start]


def job(args: tuple[str, str]) -> None:
    coin, tf = args
    path = OUT / tf / f"{coin}.csv.gz"
    if path.exists():
        return
    for attempt in range(3):
        try:
            df = fetch(coin, tf, DAYS[tf])
            break
        except requests.RequestException as e:
            print(f"{coin} {tf}: retry {attempt} ({e})", flush=True)
            time.sleep(3)
    else:
        print(f"{coin} {tf}: FAILED", flush=True)
        return
    df.index.name = "t"
    df.to_csv(path)
    print(f"{coin} {tf}: saved {len(df)} bars {df.index[0] if len(df) else ''}", flush=True)


def main() -> None:
    tfs = sys.argv[1:] or list(ENDPOINTS)
    for tf in tfs:
        (OUT / tf).mkdir(parents=True, exist_ok=True)
    tasks = [(c, tf) for tf in tfs for c in UNIVERSE]
    with ThreadPoolExecutor(max_workers=10) as ex:
        list(ex.map(job, tasks))
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
