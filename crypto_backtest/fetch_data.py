"""Download Binance public daily data from data.binance.vision (no API key needed).

- Spot daily klines (USDT pairs), monthly zip files.
- USDT-M futures funding rates, monthly zip files (used for the overheat filter).

Output:
    crypto_backtest/data/klines/<SYM>.csv   (date, open, high, low, close, quote_volume)
    crypto_backtest/data/funding/<SYM>.csv  (date, funding_sum)  # sum of 8h settlements per day

Usage:
    python crypto_backtest/fetch_data.py
"""

from __future__ import annotations

import io
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

BASE = "https://data.binance.vision/data"
DATA_DIR = Path(__file__).resolve().parent / "data"

# Survivorship bias: this list is picked from coins that are liquid TODAY. Coins that were
# delisted or fell out of the top list in the past are missing, which inflates backtest results.
UNIVERSE = [
    "BTC", "ETH", "BNB", "XRP", "ADA", "DOGE", "LTC", "TRX", "LINK", "XLM",
    "ETC", "BCH", "DOT", "UNI", "ATOM", "FIL", "NEAR", "SOL", "AVAX", "MATIC",
    "APT", "ARB", "OP", "SUI", "AAVE",
]

KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_base", "taker_quote", "ignore",
]
FUNDING_COLS = ["calc_time", "funding_interval_hours", "last_funding_rate"]

FIRST_MONTH = pd.Period("2019-09", "M")
LAST_MONTH = pd.Period("2026-10", "M")  # last complete month available at build time


def months() -> list[str]:
    out, p = [], FIRST_MONTH
    while p <= LAST_MONTH:
        out.append(p.strftime("%Y-%m"))
        p += 1
    return out


def download(url: str) -> bytes | None:
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 404:
                return None  # coin not listed that month
            r.raise_for_status()
            return r.content
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(2**attempt)
    return None


def read_zip_csv(content: bytes, names: list[str]) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(f, header=None)
    # Some files ship with a header row; drop it if the first cell is not numeric.
    if not str(df.iat[0, 0]).isdigit():
        df = df.iloc[1:]
    df = df.iloc[:, : len(names)]
    df.columns = names[: df.shape[1]]
    return df


def to_daily_date(ts: pd.Series) -> pd.Series:
    ts = ts.astype("int64")
    unit = "us" if ts.iloc[0] > 10**14 else "ms"  # Binance switched some files to microseconds
    return pd.to_datetime(ts, unit=unit).dt.normalize()


def fetch_klines(sym: str) -> pd.DataFrame | None:
    url = lambda m: f"{BASE}/spot/monthly/klines/{sym}USDT/1d/{sym}USDT-1d-{m}.zip"
    with ThreadPoolExecutor(max_workers=8) as pool:
        blobs = list(pool.map(lambda m: (m, download(url(m))), months()))
    frames = []
    for m, blob in blobs:
        if blob is None:
            continue
        df = read_zip_csv(blob, KLINE_COLS)
        df["date"] = to_daily_date(df["open_time"])
        frames.append(df[["date", "open", "high", "low", "close", "quote_volume"]])
    if not frames:
        return None
    out = pd.concat(frames).drop_duplicates("date").set_index("date").sort_index()
    return out.astype(float)


def fetch_funding(sym: str) -> pd.DataFrame | None:
    url = lambda m: f"{BASE}/futures/um/monthly/fundingRate/{sym}USDT/{sym}USDT-fundingRate-{m}.zip"
    with ThreadPoolExecutor(max_workers=8) as pool:
        blobs = list(pool.map(lambda m: (m, download(url(m))), months()))
    frames = []
    for m, blob in blobs:
        if blob is None:
            continue
        df = read_zip_csv(blob, FUNDING_COLS)
        df["date"] = to_daily_date(df["calc_time"])
        df["last_funding_rate"] = df["last_funding_rate"].astype(float)
        frames.append(df[["date", "last_funding_rate"]])
    if not frames:
        return None
    raw = pd.concat(frames)
    daily = raw.groupby("date")["last_funding_rate"].sum().rename("funding_sum").sort_index()
    return daily.to_frame()


def main() -> None:
    (DATA_DIR / "klines").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "funding").mkdir(parents=True, exist_ok=True)
    for sym in UNIVERSE:
        k = fetch_klines(sym)
        if k is None:
            print(f"[skip] {sym}: no spot klines")
            continue
        k.to_csv(DATA_DIR / "klines" / f"{sym}.csv")
        f = fetch_funding(sym)
        if f is not None:
            f.to_csv(DATA_DIR / "funding" / f"{sym}.csv")
        print(
            f"[ok] {sym}: klines {k.index.min().date()}..{k.index.max().date()} ({len(k)} rows)"
            + (f", funding {f.index.min().date()}..{f.index.max().date()}" if f is not None else ", no funding")
        )


if __name__ == "__main__":
    main()
