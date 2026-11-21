"""ARTITRADE (APK) coin strategies, version 0.2.1 rules, replayed on Upbit KRW candles per timeframe.

Source: OHM1121/RESALE_ALPHA tag 0.2.1 (ARTITRADE_ANDROID python modules + strategy_v070 core + USER_ACTIVE_POLICY.md):
  new_legacy_1118_policy.py   exit_signal / targets / bollinger angle / holding hours   [CODE]
  new_legacy_1120_policy.py   lookback-low close stop (MA 50 bars, Trend 200/90/90)     [CODE]
  ma_v067_policy.py           MA REVERSAL_UP entry + up_exit                            [CODE]
  d_upbit.py                  reference candidate (ref_features), pending-order rules   [CODE]
Three INDEPENDENT strategies with their own per-timeframe wallets (seed 2,000,000 KRW, 4 slots, 500,000 KRW/slot):
  reference (코인 기준봉) 5m/30m/60m/day   ma (코인 MA) 5m/30m/60m/day   trend (코인 MarketTrend) 30m/60m/day (+5m on request)

Bar-level modelling (candles only, no ticks/order book):
  * intrabar order follows the usual OHLC path: bullish bar O-L-H-C, bearish bar O-H-L-C; downside triggers fill at
    their level (or the open when it gapped through), limit/TP fills at the level (or the open when gapped).
  * completed-bar close stops (lookback low) are evaluated at the bar close; MA entry = signal bar close;
    Trend entry = open of the bar after the signal bar; reference entry = limit at the reference low.
  * tick-size rounding and slippage are ignored; fee 0.05% per side.
  * signed coin market score (reference BUY gate > 0, exit <= -10, MA/Trend half-size <= -10) is rebuilt from
    market_research_v033.coin_score on the 23 downloaded coins (rolling-24h change instead of signed_change_rate).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TF_DIR = HERE / "data" / "upbit_tf"
RESULTS = HERE / "results"

FRAMES = ("5m", "30m", "60m", "day")
DIRS = {"5m": "5m", "30m": "30m", "60m": "1h", "day": "1d"}
FREQ = {"5m": "5min", "30m": "30min", "60m": "1h", "day": "1D"}
SECS = {"5m": 300, "30m": 1800, "60m": 3600, "day": 86400}
FEE = 0.0005                                         # [CODE] d_upbit.FEE_RATE per side
# USER (2026-11-21) per-frame wallet: concurrent slots and per-symbol amount when the market is good / bad (score <= -10).
FRAME_CFG = {
    "5m": dict(slots=10, good=200_000, bad=100_000),
    "30m": dict(slots=10, good=200_000, bad=100_000),
    "60m": dict(slots=5, good=400_000, bad=200_000),
    "day": dict(slots=4, good=500_000, bad=250_000),
}
SEED_PER_FRAME = {tf: c["slots"] * c["good"] for tf, c in FRAME_CFG.items()}      # 2,000,000 KRW each
MIN_ORDER_KRW = 5000
SLIP = 0.0005          # per side, applied to every fill (buy at x(1+SLIP), sell at x(1-SLIP)); USER asked for slippage
ORDER_MODE = "stop_first"   # same-bar ambiguity: downside trigger is always assumed first ("path" = bullish O-L-H-C / bearish O-H-L-C)


def hours_to_bars(hours: float, tf: str) -> int:
    return max(1, int(round(hours * 3600 / SECS[tf])))


# ----------------------------------------------------------------------------------- parameters (0.2.1)
FIRST_GAIN = {"5m": .02, "30m": .03, "60m": .035, "day": .04}      # [CODE] new_legacy_1118_policy
SECOND_GAIN = {"5m": .02, "30m": .02, "60m": .025, "day": .03}     # on top of the TP1 price
TRAIL = {"5m": .025, "30m": .03, "60m": .03, "day": .04}           # reference + MA, after TP2
TREND_TRAIL = {"5m": .04, "30m": .04, "60m": .05, "day": .06}      # 5m: not defined (retired) -> 30m value
FIRST_SHARE, SECOND_SHARE = .30, .35
AFTER_FIRST_STOP = 0.01                                            # after TP1: sell all at entry +1%
HARD_STOP = 0.15                                                   # MA / Trend
MA_HOLD_H = {"5m": 48, "30m": 80, "60m": 120, "day": 360}
TREND_HOLD_H = {"5m": 48, "30m": 100, "60m": 192, "day": 360}      # 5m: not defined -> old v043 value
REF_HOLD_H = {"5m": 24, "30m": 60, "60m": 72, "day": 1440}         # [CODE] coin_hold_policy.BASE_HOLD_HOURS
REF_LIFE_H = {"5m": 2, "30m": 12, "60m": 12, "day": 72}            # [CODE] ORDER_LIFE_SECONDS
REF_WINDOW = {"5m": 288, "30m": 96, "60m": 120, "day": 91}         # volume-record lookback in bars
REF_MIN_LOW_CLOSE = {"5m": 2.0, "30m": 3.0, "60m": 4.0, "day": None}
REF_BEAR_SPAN = {"5m": 6, "30m": 6, "60m": 9, "day": 14}
MA_STOP_BARS = 50
TREND_STOP_BARS = {"5m": 200, "30m": 200, "60m": 90, "day": 90}
RSI_ARM = 30.0
TREND_WATCH_H = {"5m": 24, "30m": 24, "60m": 24, "day": 120}      # USER: daily watch 24h -> 5 days
ANGLE_LO, ANGLE_HI = -10.0, 20.0                                   # USER (2026-11-21): -10 <= angle <= +20 degrees
SCORE_BUY_MIN, SCORE_EXIT = 0.0, -10.0


# ----------------------------------------------------------------------------------- tunable settings (improvement study)
# Defaults == the APK 0.2.1 base rules. apk_improve.py overrides them per experiment; apk_run.py never touches them.
CFG = dict(
    tp_scale=1.0,                 # multiplies TP1 / TP2 gains
    trail_scale=1.0,              # multiplies the post-TP2 trailing percentages
    after_first=AFTER_FIRST_STOP, # sell-all level after TP1 (entry * (1+x))
    hard_stop=HARD_STOP,          # MA / Trend hard stop
    hold_scale=1.0,               # multiplies the holding expiry
    first_share=FIRST_SHARE, second_share=SECOND_SHARE,
    ma_stop_bars=MA_STOP_BARS,    # None disables the lookback-low close stop
    trend_stop_scale=1.0, trend_stop_off=False,
    ma_f={}, trend_f={},          # entry filters (see _mask)
    trend_rsi=RSI_ARM, trend_watch_scale=1.0,
)


# ----------------------------------------------------------------------------------- data
@dataclass
class Bars:
    t: pd.DatetimeIndex
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    v: np.ndarray
    value: np.ndarray


def load_bars(coin: str, tf: str) -> Bars | None:
    f = TF_DIR / DIRS[tf] / f"{coin}.csv.gz"
    if not f.exists():
        return None
    d = pd.read_csv(f, index_col=0, parse_dates=True)
    if len(d) < 200:
        return None
    d = d.reindex(pd.date_range(d.index[0], d.index[-1], freq=FREQ[tf]))
    d["close"] = d["close"].ffill()
    for k in ("open", "high", "low"):
        d[k] = d[k].fillna(d["close"])
    d[["volume", "value"]] = d[["volume", "value"]].fillna(0.0)
    return Bars(d.index, *(d[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume", "value")))


# ----------------------------------------------------------------------------------- market score
def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def _rsi14_closed(closes: np.ndarray, period: int = 14) -> float | None:
    if len(closes) < period + 1:
        return None
    ch = np.diff(closes)
    gain, loss = np.maximum(ch[:period], 0).mean(), np.maximum(-ch[:period], 0).mean()
    for x in ch[period:]:
        gain = (gain * (period - 1) + max(0.0, x)) / period
        loss = (loss * (period - 1) + max(0.0, -x)) / period
    if gain == 0 and loss == 0:
        return 50.0
    return 100.0 if loss == 0 else 100.0 - 100.0 / (1.0 + gain / loss)


def build_market_score(coins: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Hourly signed score (-100..100) per market_research_v033.coin_score. Returns (end_times_ns, score)."""
    hourly = {c: load_bars(c, "60m") for c in coins}
    hourly = {c: b for c, b in hourly.items() if b is not None}
    daily = {c: load_bars(c, "day") for c in ("BTC", "ETH")}
    idx = hourly["BTC"].t
    n = len(idx)
    close = pd.DataFrame({c: pd.Series(b.c, index=b.t).reindex(idx) for c, b in hourly.items()})
    value = pd.DataFrame({c: pd.Series(b.value, index=b.t).reindex(idx) for c, b in hourly.items()})
    ch24 = close / close.shift(24) - 1
    turn24 = value.rolling(24, min_periods=24).sum()
    ok = ch24.notna() & (turn24 > 0)
    breadth = 100 * ((ch24 > 0) & ok).sum(axis=1) / ok.sum(axis=1).replace(0, np.nan)
    rising = (turn24.where(ch24 > 0, 0.0).where(ok, 0.0)).sum(axis=1)
    cashflow = 100 * rising / turn24.where(ok, 0.0).sum(axis=1).replace(0, np.nan)
    nvalid = ok.sum(axis=1).to_numpy()
    btc_d = pd.Series(daily["BTC"].c, index=daily["BTC"].t)
    eth_d = pd.Series(daily["ETH"].c, index=daily["ETH"].t)
    live_btc, live_eth = close["BTC"].to_numpy(), close["ETH"].to_numpy()
    days = idx.normalize()
    btc_arr, eth_arr = btc_d.to_numpy(), eth_d.to_numpy()
    btc_pos = btc_d.index.searchsorted(days, side="left")        # first daily bar of today -> closed = [:pos]
    eth_pos = eth_d.index.searchsorted(days, side="left")
    score = np.full(n, np.nan)
    w = (0.15, 0.25, 0.20, 0.15, 0.10, 0.15)
    for i in range(n):
        bp, ep = btc_pos[i], eth_pos[i]
        if bp < 34 or ep < 11 or nvalid[i] < 10 or math.isnan(live_btc[i]) or math.isnan(live_eth[i]):
            continue
        btc = np.append(btc_arr[bp - 34:bp], live_btc[i])          # 34 closed + live (35 candles)
        eth = np.append(eth_arr[ep - 11:ep], live_eth[i])          # 11 closed + live (12 candles)
        btc7 = (btc[-1] / btc[-8] - 1) * 100
        eth7 = (eth[-1] / eth[-8] - 1) * 100
        ma30 = btc[-30:].mean()
        rets = (btc[1:] / btc[:-1] - 1) * 100
        vol = float(np.std(rets[-20:]))
        rsi = _rsi14_closed(btc[:-1])
        if rsi is None:
            continue
        vals = (
            (_clamp(50 + (btc[-1] / ma30 - 1) * 100 * 6) + _clamp(50 + btc7 * 6)) / 2,
            breadth.iat[i], cashflow.iat[i],
            _clamp(50 + (eth7 - btc7) * 5), _clamp((10.0 - vol) * 12.5), _clamp((70.0 - rsi) * 2.5),
        )
        if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in vals):
            continue
        score[i] = _clamp(sum((2 * v - 100) * wt for v, wt in zip(vals, w)), -100, 100)
    ends = (idx + pd.Timedelta(hours=1)).to_numpy("datetime64[ns]").astype("int64")
    return ends, score


class ScoreLookup:
    def __init__(self, ends: np.ndarray, score: np.ndarray):
        self.ends, self.score = ends, score

    def at(self, ts: pd.Timestamp) -> float:
        i = np.searchsorted(self.ends, ts.value, side="right") - 1
        return float(self.score[i]) if i >= 0 else float("nan")




# ----------------------------------------------------------------------------------- trades + exit engine
@dataclass
class Trade:
    strategy: str
    tf: str
    coin: str
    entry_i: int
    entry_t: pd.Timestamp
    entry: float
    size_factor: float = 1.0
    sig_i: int = -1                              # signal bar index (for feature lookup)
    legs: list = field(default_factory=list)     # (bar_index, time, price, fraction of original)
    reasons: list = field(default_factory=list)
    open_at_end: bool = False

    @property
    def exit_t(self) -> pd.Timestamp:
        return self.legs[-1][1]

    @property
    def ret(self) -> float:
        cost = self.entry * (1 + SLIP) * (1 + FEE)
        return sum(f * p * (1 - SLIP) * (1 - FEE) for _, _, p, f in self.legs) / cost - 1


def _sell(tr: Trade, k: int, t, px: float, frac: float, why: str) -> None:
    tr.legs.append((k, t, px, frac))
    tr.reasons.append(why)


def _remaining(tr: Trade) -> float:
    return 1.0 - sum(x[3] for x in tr.legs)


def _finish_open(tr: Trade, b: Bars) -> None:
    rem = _remaining(tr)
    if rem > 1e-9:
        _sell(tr, len(b.c) - 1, b.t[-1], b.c[-1], rem, "OPEN_AT_END_MARK")
        tr.open_at_end = True


def manage(tr: Trade, b: Bars, tf: str, kind: str, sc: "ScoreLookup", *, hold_end: int, hard_stop: float,
           lb_stop: float | None, trail: float, entry_mode: str = "skip") -> int:
    """Bar-level replay of new_legacy_1118_policy.exit_signal (+ new_legacy_1120 lookback-low close stop).

    Order inside exit_signal: hard stop, hold expiry, (reference: score<=-10), lookback-low close stop,
    then by stage: 0 -> [reference: return to entry] TP1 | 1 -> entry+1% stop, TP2 | 2 -> TP1 floor, peak trail.
    Same-bar ambiguity (a downside trigger and a profit target both inside one bar): ORDER_MODE 'stop_first' always
    takes the downside first, i.e. the position is stopped out before the target counts.
    entry_mode: 'skip'      entry at the bar close (MA) -> management starts with the next bar;
                'down_only' limit fill inside the entry bar (reference) -> only the hard stop can still hit in that bar;
                'full'      entry at the bar open (Trend) -> the entry bar is managed completely.
    """
    E = tr.entry
    n = len(b.c)
    tp1 = E * (1 + FIRST_GAIN[tf] * CFG["tp_scale"])
    tp2 = tp1 * (1 + SECOND_GAIN[tf] * CFG["tp_scale"])
    stage, seen, peak = 0, False, E
    first_k = tr.entry_i if entry_mode in ("full", "down_only") else tr.entry_i + 1
    for k in range(first_k, n):
        o, h, l, c, t = b.o[k], b.h[k], b.l[k], b.c[k], b.t[k]
        entry_bar = (k == tr.entry_i)
        if not entry_bar:
            if kind == "reference":
                s = sc.at(t)
                if not math.isnan(s) and s <= SCORE_EXIT:
                    _sell(tr, k, t, o, _remaining(tr), "SCORE_MINUS_10")
                    return k
            if k >= hold_end:
                _sell(tr, k, t, o, _remaining(tr), "HOLD_EXPIRY")
                return k
        if entry_bar and entry_mode == "down_only":
            if hard_stop > 0 and l <= hard_stop:                 # fell through the limit and on to the stop in the fill bar
                _sell(tr, k, t, hard_stop if o > hard_stop else o, _remaining(tr), "HARD_STOP")
                return k
            continue
        cur = o
        if ORDER_MODE == "stop_first":
            phases = ("down", "up")
        else:
            phases = ("down", "up") if c >= o else ("up", "down")
        for phase in phases:
            if phase == "down":
                if stage == 0:
                    levels = ([(E, "RETURN_TO_ENTRY")] if (kind == "reference" and seen) else []) + [(hard_stop, "HARD_STOP")]
                elif stage == 1:
                    levels = [(E * (1 + CFG["after_first"]), "AFTER_FIRST_PLUS_1"), (hard_stop, "HARD_STOP")]
                else:
                    levels = [(max(tp1, peak * (1 - trail)), "TP1_FLOOR_OR_TRAIL"), (hard_stop, "HARD_STOP")]
                for level, why in levels:
                    if level > 0 and l <= level:
                        _sell(tr, k, t, min(level, cur), _remaining(tr), why)
                        return k
                cur = l
            else:
                if stage == 0:
                    if kind == "reference" and h > E:
                        seen = True
                    if h >= tp1:
                        _sell(tr, k, t, max(tp1, cur), CFG["first_share"], "TP1")
                        stage = 1
                if stage == 1 and h >= tp2:
                    _sell(tr, k, t, max(tp2, cur), CFG["second_share"], "TP2")
                    stage, peak = 2, tp2
                if stage == 2:
                    peak = max(peak, h)
                cur = h
        if lb_stop is not None and c < lb_stop:                      # completed-bar CLOSE below the lookback-low stop
            _sell(tr, k, t, c, _remaining(tr), "LOOKBACK_LOW_CLOSE_STOP")
            return k
    _finish_open(tr, b)
    return n


def _lookback_stop(lows: np.ndarray, end_excl: int, nbars: int, entry: float) -> float | None:
    """Lowest LOW of the last nbars completed bars before the buy; ignored if above the buy price (new_legacy_1120)."""
    seg = lows[max(0, end_excl - nbars):end_excl]
    if len(seg) < 20:
        return None
    price = float(seg.min())
    return None if price > entry else price


# ----------------------------------------------------------------------------------- reference (기준봉)
FUNNEL: dict = {}


def _fn(key: str, k: str) -> None:
    FUNNEL.setdefault(key, {}).setdefault(k, 0)
    FUNNEL[key][k] += 1


def reference_candidates(b: Bars, tf: str) -> tuple[np.ndarray, np.ndarray]:
    n = len(b.c)
    N = REF_WINDOW[tf]
    prior_max = pd.Series(b.v).shift(1).rolling(N, min_periods=N).max().to_numpy()
    width = b.h - b.l
    with np.errstate(divide="ignore", invalid="ignore"):
        rec = b.v > prior_max                                                  # strict new volume record
        mlc = REF_MIN_LOW_CLOSE[tf]
        move = np.ones(n, bool) if mlc is None else (b.c - b.l) * 100 >= b.l * mlc
        red = np.maximum(0, b.o - b.c) / width
        strong_bear = (red >= 0.60) & (100 * width / b.l >= REF_BEAR_SPAN[tf])
        upper = (b.h - np.maximum(b.o, b.c)) / width
        body_lo, body_hi = np.minimum(b.o, b.c), np.maximum(b.o, b.c)
        target = np.where(width / b.l * 100 >= 10, body_lo + (body_hi - body_lo) * 0.45, body_hi)
    cand = rec & move & ~strong_bear & (upper * 5 < 3) & (width > 0) & (target > b.l)
    if tf == "5m":
        cand &= width * 100 >= b.l * 2                                         # _enough_5m_range
    cand[:N + 1] = False
    return cand, target


def reference_trades(coin: str, tf: str, b: Bars, sc: "ScoreLookup", use_score: bool = True) -> list[Trade]:
    n = len(b.c)
    cand, _ = reference_candidates(b, tf)
    idx = np.flatnonzero(cand)
    life = hours_to_bars(REF_LIFE_H[tf], tf)
    key = f"reference/{tf}"
    trades: list[Trade] = []
    busy = -1
    for pos, i in enumerate(idx):
        if i <= busy:
            _fn(key, "rejected_already_held")
            continue
        _fn(key, "candidates")
        limit = b.l[i]
        nxt = idx[pos + 1] if pos + 1 < len(idx) else n
        last = min(n - 1, i + life, nxt)                  # order expires with its life or is REPLACED by a newer reference
        fill, blocked = None, False
        for j in range(i + 1, last + 1):
            if b.l[j] <= limit:
                if use_score and not (sc.at(b.t[j]) > SCORE_BUY_MIN):
                    blocked = True                       # touched while score <= 0: no BUY, order stays pending
                    continue
                fill = j
                break
        if fill is None:
            _fn(key, "unfilled_expired_or_replaced" + ("_after_score_block" if blocked else ""))
            continue
        _fn(key, "filled")
        j = fill
        E = min(limit, b.o[j])
        tr = Trade("reference", tf, coin, j, b.t[j], E)
        stop = max(0.0, b.l[i] - (b.h[i] - b.l[i]))       # at_hard_stop: ref low - (ref high - ref low)
        busy = manage(tr, b, tf, "reference", sc, hold_end=j + hours_to_bars(REF_HOLD_H[tf], tf),
                      hard_stop=stop, lb_stop=None, trail=TRAIL[tf], entry_mode="down_only")
        trades.append(tr)
    return trades


# ----------------------------------------------------------------------------------- entry filters (improvement study)
def _score_array(b: Bars, sc: "ScoreLookup") -> np.ndarray:
    t = b.t.to_numpy("datetime64[ns]").astype("int64")
    idx = np.searchsorted(sc.ends, t, side="right") - 1
    return np.where(idx >= 0, sc.score[np.clip(idx, 0, None)], np.nan)


def _mask(b: Bars, sc: "ScoreLookup", f: dict, ma: dict | None = None, mid: np.ndarray | None = None) -> np.ndarray:
    """Boolean mask of bars that pass the optional entry filters (empty dict = no filter)."""
    n = len(b.c)
    m = np.ones(n, bool)
    if not f:
        return m
    with np.errstate(invalid="ignore", divide="ignore"):
        if f.get("score_min") is not None:
            m &= _score_array(b, sc) > f["score_min"]
        if f.get("bull"):
            m &= b.c > b.o
        if f.get("volx") is not None:
            avg = pd.Series(b.v).rolling(20, min_periods=20).mean().shift(1).to_numpy()
            m &= b.v >= f["volx"] * avg
        if f.get("c_gt_ma5") and ma is not None:
            m &= b.c > ma[5]
        if f.get("rsi14_max") is not None:
            m &= rsi20(b.c, 14) <= f["rsi14_max"]
        if f.get("dist120_min") is not None and ma is not None:
            m &= (ma[120] - b.c) / ma[120] >= f["dist120_min"]
        if f.get("dist_mid_max") is not None and mid is not None:
            m &= (b.c - mid) / mid <= f["dist_mid_max"]
        if f.get("rsi_min") is not None:
            m &= rsi20(b.c, 20) >= f["rsi_min"]
    return m


# ----------------------------------------------------------------------------------- MA
def ma_trades(coin: str, tf: str, b: Bars, sc: "ScoreLookup") -> list[Trade]:
    """REVERSAL_UP (ma_v067_policy.prewatch_upward_alignment, 0.2.1): MA120 slope strictly negative, MA5/10/20/60 all
    below MA120 and all rising; only the FIRST bar where this alignment appears."""
    n = len(b.c)
    c = pd.Series(b.c)
    ma = {p: c.rolling(p, min_periods=p).mean().to_numpy() for p in (5, 10, 20, 60, 120)}
    d = {p: np.r_[np.nan, np.diff(ma[p])] for p in ma}
    with np.errstate(invalid="ignore"):
        qual = (d[120] < 0) & np.all([(d[p] > 0) & (ma[p] < ma[120]) for p in (5, 10, 20, 60)], axis=0)
    sig = qual & ~np.r_[False, qual[:-1]]
    sig[:125] = False
    sig &= _mask(b, sc, CFG["ma_f"], ma=ma)
    trades: list[Trade] = []
    busy = -1
    for i in np.flatnonzero(sig):
        if i <= busy or i + 1 >= n:
            continue
        E = b.c[i]                                          # bought in the last seconds of the signal bar
        s = sc.at(b.t[i])
        tr = Trade("ma", tf, coin, i, b.t[i], E, size_factor=0.5 if (not math.isnan(s) and s <= SCORE_EXIT) else 1.0)
        tr.sig_i = i
        busy = manage(tr, b, tf, "ma", sc, hold_end=i + hours_to_bars(MA_HOLD_H[tf] * CFG["hold_scale"], tf),
                      hard_stop=E * (1 - CFG["hard_stop"]),
                      lb_stop=(_lookback_stop(b.l, i, CFG["ma_stop_bars"], E) if CFG["ma_stop_bars"] else None),
                      trail=TRAIL[tf] * CFG["trail_scale"], entry_mode="skip")
        trades.append(tr)
    return trades


# ----------------------------------------------------------------------------------- MarketTrend
def rsi20(close: np.ndarray, period: int = 20) -> np.ndarray:
    n = len(close)
    out = np.full(n, np.nan)
    if n <= period:
        return out
    ch = np.diff(close)
    gain, loss = np.maximum(ch[:period], 0).mean(), np.maximum(-ch[:period], 0).mean()
    out[period] = 100.0 if loss == 0 else 100 - 100 / (1 + gain / loss)
    for i in range(period, n - 1):
        gain = (gain * (period - 1) + max(ch[i], 0.0)) / period
        loss = (loss * (period - 1) + max(-ch[i], 0.0)) / period
        out[i + 1] = 50.0 if gain == 0 and loss == 0 else (100.0 if loss == 0 else 100 - 100 / (1 + gain / loss))
    return out


def boll_angle(close: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bollinger(20) middle and its pixel-projected slope in degrees (new_legacy_1118_policy.bollinger)."""
    c = pd.Series(close)
    mid = c.rolling(20, min_periods=20).mean()
    former = mid.shift(1)
    lo, hi = c.rolling(21, min_periods=21).min(), c.rolling(21, min_periods=21).max()
    span = np.maximum(hi - lo, mid * 0.001)
    pmin, pmax = np.maximum(1e-10, lo - span * 0.08), hi + span * 0.08
    dy = (mid - former) * 180.0 / (pmax - pmin)                 # VISUAL_CHART_HEIGHT_PX = 180
    angle = np.degrees(np.arctan2(dy, 8.0))                     # VISUAL_CANDLE_PITCH_PX = 8
    return mid.to_numpy(), angle.to_numpy()


def trend_trades(coin: str, tf: str, b: Bars, sc: "ScoreLookup", angle_lo: float = ANGLE_LO, angle_hi: float = ANGLE_HI,
                 strict: bool = False) -> list[Trade]:
    """RSI20<=30 starts a 24h watch; the FIRST completed bar after it that closes above its Bollinger(20) middle may buy
    if the middle's angle is allowed; otherwise the cycle is consumed until RSI>30 then <=30 again."""
    n = len(b.c)
    rsi = rsi20(b.c)
    mid, ang = boll_angle(b.c)
    watch = hours_to_bars(TREND_WATCH_H[tf] * CFG["trend_watch_scale"], tf)
    fmask = _mask(b, sc, CFG["trend_f"], mid=mid)
    rsi_arm = CFG["trend_rsi"]
    state, w = "READY", -1
    trades: list[Trade] = []
    busy = -1
    for j in range(25, n - 1):
        r = rsi[j]
        if state == "WATCHING":
            if j - w > watch:
                state = "EXPIRED"
            elif j > w and b.c[j] > mid[j]:
                ok = (angle_lo < ang[j] < angle_hi) if strict else (angle_lo <= ang[j] <= angle_hi)
                if ok and fmask[j] and j + 1 > busy:
                    e = j + 1
                    E = b.o[e]
                    s = sc.at(b.t[e])
                    tr = Trade("trend", tf, coin, e, b.t[e], E,
                               size_factor=0.5 if (not math.isnan(s) and s <= SCORE_EXIT) else 1.0)
                    tr.sig_i = j
                    busy = manage(tr, b, tf, "trend", sc, hold_end=e + hours_to_bars(TREND_HOLD_H[tf] * CFG["hold_scale"], tf),
                                  hard_stop=E * (1 - CFG["hard_stop"]),
                                  lb_stop=(None if CFG["trend_stop_off"] else
                                           _lookback_stop(b.l, e, max(20, int(TREND_STOP_BARS[tf] * CFG["trend_stop_scale"])), E)),
                                  trail=TREND_TRAIL[tf] * CFG["trail_scale"], entry_mode="full")
                    trades.append(tr)
                    state = "BOUGHT"
                else:
                    state = "CONSUMED"
        if not math.isnan(r):
            if r > rsi_arm and state in ("EXPIRED", "BOUGHT", "CONSUMED"):
                state = "READY"
            if state == "READY" and r <= rsi_arm:
                state, w = "WATCHING", j
    return trades


# ----------------------------------------------------------------------------------- portfolio + stats
def portfolio(trades: list[Trade], tf: str) -> dict:
    """One wallet per frame (FRAME_CFG): slots, per-symbol amount good/bad market (bad = score <= -10, applies to all)."""
    cfg = FRAME_CFG[tf]
    seed = float(SEED_PER_FRAME[tf])
    cash = seed
    open_pos: list[tuple] = []
    skipped = 0
    realized = [seed]
    for tr in sorted(trades, key=lambda x: x.entry_t):
        for p in sorted([p for p in open_pos if p[0] <= tr.entry_t]):
            cash += p[1] * (1 + p[2])
            open_pos.remove(p)
            realized.append(cash + sum(q[1] for q in open_pos))
        notional = min(cfg["bad"] if tr.size_factor < 1 else cfg["good"], cash)
        if len(open_pos) >= cfg["slots"] or notional < MIN_ORDER_KRW:
            skipped += 1
            continue
        cash -= notional
        open_pos.append((tr.exit_t, notional, tr.ret))
    for p in sorted(open_pos):
        cash += p[1] * (1 + p[2])
        realized.append(cash)
    eq = np.array(realized)
    return {"portfolio_return": cash / seed - 1,
            "portfolio_MDD_realized": float((eq / np.maximum.accumulate(eq) - 1).min()),
            "skipped_no_slot": skipped}


def summarize(trades: list[Trade], strategy: str, tf: str, start, end, ncoins: int) -> dict:
    row = {"strategy": strategy, "timeframe": tf, "start": start, "end": end, "coins": ncoins, "trades": len(trades)}
    if trades:
        rets = np.array([t.ret for t in trades])
        wins, losses = rets[rets > 0], rets[rets <= 0]
        row.update({
            "win_rate": float((rets > 0).mean()), "avg_trade_ret": float(rets.mean()),
            "median_trade_ret": float(np.median(rets)),
            "avg_win": float(wins.mean()) if len(wins) else np.nan,
            "avg_loss": float(losses.mean()) if len(losses) else np.nan,
            "profit_factor": float(wins.sum() / -losses.sum()) if losses.sum() < 0 else np.nan,
            "open_at_end": int(sum(t.open_at_end for t in trades)), **portfolio(trades, tf)})
    return row
