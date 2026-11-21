"""ARTITRADE (APK) coin strategies replayed on Upbit KRW candles, one timeframe at a time.

Source of every rule: OHM1121/RESALE_ALPHA (ARTITRADE_ANDROID python modules, docs/artitrade/USER_ACTIVE_POLICY.md,
and the embedded strategy core assets/strategy_v070). The "NEW LEGACY (11.18).xlsx" workbook is NOT in the repo, so
numbers that only the workbook defines are marked  [ASSUMED]  below. Everything else is [CODE] / [POLICY].

Three INDEPENDENT strategies (the APK keeps separate wallets):
  reference  (코인 기준봉)   frames 5m/30m/60m/day
  ma         (코인 MA)       frames 5m/30m/60m/day
  trend      (코인 MarketTrend) frames 30m/60m/day only (5m new entries are forbidden)

Bar-level approximations (candles only, no ticks / order book):
  * limit fills at the limit price (or the bar open if it gapped through), stops at the stop price (or open if gapped)
  * if a stop-type and a profit-type level are both inside one bar the stop-type wins (policy: never invent the
    favourable order); a level is never both armed and triggered inside the same bar.
  * tick-size rounding is ignored; the app's forming-candle "close-10s" MA/Trend entry is modelled as a fill at the
    signal bar's close (MA) / at the Bollinger middle (Trend).
  * signed coin market score (the >0 gate / <-10 exit) is rebuilt from market_research_v033.coin_score using the
    23 downloaded coins as the KRW universe and rolling-24h change instead of Upbit's signed_change_rate.
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
FEE = 0.0005                      # [CODE] d_upbit.FEE_RATE (per side)
SEED, MAX_PER_COIN, MAX_POS = 2_000_000, 500_000, 4   # [CODE] d_upbit SEED_KRW / MAX_KRW_PER_COIN / MAX_COIN_POSITIONS


def hours_to_bars(hours: float, tf: str) -> int:
    return max(1, int(round(hours * 3600 / SECS[tf])))


# ----------------------------------------------------------------------------------- parameters
REF = dict(  # coin reference candle, [CODE] d_upbit.ref_features / coin_v24_policy / coin_hold_policy
    window_bars={"5m": 288, "30m": 96, "60m": 120, "day": 91},          # 24h / 48h / 120h / ~3 calendar months
    min_low_close_pct={"5m": 2.0, "30m": 3.0, "60m": 4.0, "day": None},
    bear_span_pct={"5m": 6, "30m": 6, "60m": 9, "day": 14},
    order_life_h={"5m": 2, "30m": 12, "60m": 12, "day": 72},
    base_hold_h={"5m": 24, "30m": 60, "60m": 72, "day": 1440},
    max_hold_h={"5m": 36, "30m": 72, "60m": 96, "day": 1440},
    rec2_wait_h={"5m": 12, "30m": 24, "60m": 36, "day": None},
    rec_trig={"5m": (0.02, 0.04), "30m": (0.03, 0.05), "60m": (0.05, 0.07), "day": (0.07, 0.10)},
    tp1={"5m": 0.02, "30m": 0.03, "60m": 0.035, "day": 0.04},            # [POLICY] USER_ACTIVE_POLICY 11.18
    tp2_extra={"5m": 0.02, "30m": 0.02, "60m": 0.025, "day": 0.03},      # [POLICY] on top of the TP1 price
    f1=0.30, f2=0.35,                                                     # [POLICY] 30% / 35% of original qty
    trail=0.05,                                                           # [CODE] V2.5 third stage: peak -5%
    score_buy_min=0.0, score_exit_below=-10.0,                            # [POLICY]
)
MA = dict(
    hold_h={"5m": 48, "30m": 80, "60m": 120, "day": 360},                 # [POLICY] 48h/80h/120h/15d
    hard_stop=0.15,                                                       # [POLICY] -15%
    tp1={"5m": 0.02, "30m": 0.025, "60m": 0.03, "day": 0.05},             # [CODE] ma_v067_policy UP_FIRST_TARGET_PCT
    tp2_extra={"5m": 0.02, "30m": 0.025, "60m": 0.03, "day": 0.03},       # [CODE] UP_SECOND_EXTRA_PCT
    f1=0.30, f2=0.35,                                                     # [POLICY] 11.18 (code v067 had 10%/50%)
    guard_after_first=0.01, trail=0.03,                                   # [CODE] v067 up_exit
)
TREND = dict(
    frames=("30m", "60m", "day"),
    rsi_arm=30.0, watch_h=24, bb_k=1.5, slope_lo=-20.0, slope_hi=30.0,    # [POLICY] RSI20<=30, 24h, BB20/1.5, (-20,+30)
    hard_stop=0.15,                                                       # [POLICY]
    hold_h={"30m": 72, "60m": 120, "day": 240},                           # [ASSUMED] market_trend_v043.HOLD_HOURS
    width_cap=0.03, f1=0.30, f2=0.35, trail=0.04,                         # [CODE] candle_targets / TrendPosition
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


# ----------------------------------------------------------------------------------- trades
@dataclass
class Trade:
    strategy: str
    tf: str
    coin: str
    entry_i: int
    entry_t: pd.Timestamp
    entry: float
    legs: list = field(default_factory=list)     # (bar_index, time, price, fraction of original)
    reasons: list = field(default_factory=list)
    open_at_end: bool = False

    @property
    def exit_t(self) -> pd.Timestamp:
        return self.legs[-1][1]

    @property
    def ret(self) -> float:
        cost = self.entry * (1 + FEE)
        proceeds = sum(frac * px * (1 - FEE) for _, _, px, frac in self.legs)
        return proceeds / cost - 1


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


# ----------------------------------------------------------------------------------- reference (기준봉)
FUNNEL: dict = {}


def _fn(key: str, k: str, inc: int = 1) -> None:
    FUNNEL.setdefault(key, {}).setdefault(k, 0)
    FUNNEL[key][k] += inc


def reference_trades(coin: str, tf: str, b: Bars, sc: ScoreLookup) -> list[Trade]:
    P = REF
    n = len(b.c)
    N = P["window_bars"][tf]
    vol = pd.Series(b.v)
    prior_max = vol.shift(1).rolling(N, min_periods=N).max().to_numpy()
    width = b.h - b.l
    with np.errstate(divide="ignore", invalid="ignore"):
        rec = b.v > prior_max
        mlc = P["min_low_close_pct"][tf]
        move = np.ones(n, bool) if mlc is None else (b.c - b.l) * 100 >= b.l * mlc
        red = np.maximum(0, b.o - b.c) / width
        strong_bear = (red >= 0.60) & (100 * width / b.l >= P["bear_span_pct"][tf])
        upper = (b.h - np.maximum(b.o, b.c)) / width
        body_lo, body_hi = np.minimum(b.o, b.c), np.maximum(b.o, b.c)
        span = body_hi - body_lo
        range_pct = width / b.l * 100
        target = np.where(range_pct >= 10, body_lo + span * 0.45, body_hi)
    cand = rec & move & ~strong_bear & (upper * 5 < 3) & (width > 0) & (target > b.l)
    cand[:N + 1] = False
    life = hours_to_bars(P["order_life_h"][tf], tf)
    trades: list[Trade] = []
    busy_until = -1
    for i in np.flatnonzero(cand):
        if i <= busy_until:
            continue
        limit = b.l[i]
        tgt_px = target[i]                               # [CODE] candidate target = body45 / body_high (pre-buy check)
        key = f"reference/{tf}"
        _fn(key, "candidates")
        fill = None
        gated = False
        reason = "EXPIRED"
        for j in range(i + 1, min(n, i + 1 + life)):
            touched_entry = b.l[j] <= limit
            touched_target = b.h[j] >= tgt_px
            if touched_target and not touched_entry:
                reason = "TARGET_FIRST"                  # candidate dropped before any buy
                break
            if touched_entry:
                s = sc.at(b.t[j])
                if not (s > P["score_buy_min"]):         # signed score must be strictly > 0 at the actual BUY
                    gated = True
                    continue
                fill = j
                break
        if fill is None:
            _fn(key, "dropped_" + reason + ("_after_score_block" if gated else ""))
            continue
        _fn(key, "filled")
        j = fill
        E = min(limit, b.o[j])
        tr = Trade("reference", tf, coin, j, b.t[j], E)
        end = _manage_reference(tr, b, sc, tf, i, E, body_lo[i], body_hi[i], b.l[i], b.h[i])
        trades.append(tr)
        busy_until = end
    return trades


def _manage_reference(tr: Trade, b: Bars, sc: ScoreLookup, tf: str, ref_i: int, E: float,
                      body_lo: float, body_hi: float, ref_low: float, ref_high: float) -> int:
    P = REF
    n = len(b.c)
    t1, t2 = P["rec_trig"][tf]
    tp1 = E * (1 + P["tp1"][tf])
    tp2 = tp1 * (1 + P["tp2_extra"][tf])
    rec_half = body_lo + (body_hi - body_lo) * 0.225
    rec2_target = E * 0.995
    mid = (ref_low + body_hi) / 2 if 0 < ref_low <= body_hi <= ref_high else 0.0
    hold_until = tr.entry_i + hours_to_bars(P["base_hold_h"][tf], tf)
    max_until = tr.entry_i + hours_to_bars(P["max_hold_h"][tf], tf)
    rec2_wait = P["rec2_wait_h"][tf]
    tier, rec2_at = 0, None
    first = second = False
    armed = False
    peak = 0.0
    for k in range(tr.entry_i + 1, n):
        o, h, l, c, t = b.o[k], b.h[k], b.l[k], b.c[k], b.t[k]
        rem = _remaining(tr)
        # ---- bar-open checks
        s = sc.at(t)
        if not math.isnan(s) and s < P["score_exit_below"]:
            _sell(tr, k, t, o, rem, "SCORE_BELOW_-10")
            return k
        if not first:
            if tier == 2 and rec2_wait is not None and rec2_at is not None and k >= rec2_at + hours_to_bars(rec2_wait, tf):
                _sell(tr, k, t, o, rem, "RECOVERY2_WAIT_END")
                return k
            if k >= hold_until:
                if hold_until < max_until and tier < 2 and o >= E * 0.99:
                    hold_until = max_until
                else:
                    _sell(tr, k, t, o, rem, "HOLD_END")
                    return k
        # ---- intrabar
        if not first:
            if tier == 1 and h >= rec_half:
                _sell(tr, k, t, max(o, rec_half), rem, "RECOVERY1_TARGET")
                return k
            if tier == 2 and h >= rec2_target:
                _sell(tr, k, t, max(o, rec2_target), rem, "RECOVERY2_TARGET")
                return k
            new_tier = tier
            if l <= E * (1 - t2):
                new_tier = 2
            elif l <= E * (1 - t1) and tier < 1:
                new_tier = 1
            if new_tier == 0 and h >= tp1:
                _sell(tr, k, t, max(o, tp1), P["f1"], "TP1")
                first = True
                peak = tp1
                continue
            if new_tier > tier:
                if new_tier == 2 and tier < 2:
                    rec2_at = k
                tier = new_tier
        else:
            if mid > 0:
                if armed and l <= mid:
                    _sell(tr, k, t, min(o, mid), _remaining(tr), "MIDPOINT_DOWNTOUCH")
                    return k
                if h >= mid:
                    armed = True
            if not second:
                if h >= tp2:
                    _sell(tr, k, t, max(o, tp2), P["f2"], "TP2")
                    second = True
                    peak = tp2
                    continue
            else:
                stop = max(tp1, peak * (1 - P["trail"]))
                if l <= stop:
                    _sell(tr, k, t, min(o, stop), _remaining(tr), "TRAIL_OR_TP1_RETURN")
                    return k
                peak = max(peak, h)
    _finish_open(tr, b)
    return n


# ----------------------------------------------------------------------------------- MA
def ma_features(b: Bars):
    c = pd.Series(b.c)
    out = {}
    for p in (5, 10, 20, 60, 120):
        m = c.rolling(p, min_periods=p).mean()
        out[p] = (m.to_numpy(), m.diff().to_numpy())
    return out


def ma_trades(coin: str, tf: str, b: Bars, variant: str = "new_legacy") -> list[Trade]:
    """variant 'new_legacy': MA120 slope turns UP while MA5/10/20/60 stay below MA120 and rise  [POLICY 11.18]
       variant 'v067'      : MA120 not rising (<= +0.005) while all four short MAs turn up (no 'below' filter) [CODE]"""
    n = len(b.c)
    m = ma_features(b)
    d120, v120 = m[120][1], m[120][0]
    short = (5, 10, 20, 60)
    short_up = np.all([m[p][1] > 0 for p in short], axis=0)
    short_below = np.all([m[p][0] < v120 for p in short], axis=0)
    with np.errstate(invalid="ignore"):
        if variant == "new_legacy":
            qual = (d120 > 0) & short_up & short_below
            first_up = np.r_[False, (d120[1:] > 0) & ~(d120[:-1] > 0)] & short_up & short_below
            sig = first_up & ~np.r_[False, qual[:-1]]
        else:
            qual = (d120 <= 0.005) & short_up
            sig = qual & ~np.r_[False, qual[:-1]]
    sig[:125] = False
    trades: list[Trade] = []
    busy = -1
    for i in np.flatnonzero(sig):
        if i <= busy or i + 1 >= n:
            continue
        E = b.c[i]
        tr = Trade("ma", tf, coin, i, b.t[i], E)
        busy = _manage_ma(tr, b, tf, m, E, b.l[i])
        trades.append(tr)
    return trades


def _manage_ma(tr: Trade, b: Bars, tf: str, m: dict, E: float, entry_low: float) -> int:
    P = MA
    n = len(b.c)
    tp1 = E * (1 + P["tp1"][tf])
    tp2 = tp1 * (1 + P["tp2_extra"][tf])
    hold_end = tr.entry_i + hours_to_bars(P["hold_h"][tf], tf)
    first = second = False
    touched_low = False
    peak = 0.0
    for k in range(tr.entry_i + 1, n):
        o, h, l, c, t = b.o[k], b.h[k], b.l[k], b.c[k], b.t[k]
        rem = _remaining(tr)
        if k >= hold_end:
            _sell(tr, k, t, o, rem, "HOLD_END")
            return k
        stop = E * (1 - P["hard_stop"])
        if l <= stop:
            _sell(tr, k, t, min(o, stop), rem, "HARD_STOP_-15")
            return k
        # completed-bar protective exits (evaluated at this bar's close)
        d120 = m[120][1][k]
        if (d120 > 0 and all(m[p][0][k] > m[120][0][k] and m[p][1][k] < 0 for p in (5, 10, 20, 60))):
            _sell(tr, k, t, c, rem, "FORCE_120UP_TOP4_DOWN")
            return k
        if l <= entry_low:
            touched_low = True
        if touched_low and c < entry_low:
            _sell(tr, k, t, c, rem, "ENTRY_BAR_LOW_BREAK")
            return k
        if not first:
            if h >= tp1:
                _sell(tr, k, t, max(o, tp1), P["f1"], "TP1")
                first = True
        elif not second:
            guard = E * (1 + P["guard_after_first"])
            if l <= guard:
                _sell(tr, k, t, min(o, guard), _remaining(tr), "AFTER_FIRST_ENTRY+1")
                return k
            if h >= tp2:
                _sell(tr, k, t, max(o, tp2), P["f2"], "TP2")
                second = True
                peak = tp2
        else:
            if l <= tp1:
                _sell(tr, k, t, min(o, tp1), _remaining(tr), "FIRST_TARGET_FLOOR")
                return k
            if l <= peak * (1 - P["trail"]):
                _sell(tr, k, t, min(o, peak * (1 - P["trail"])), _remaining(tr), "PEAK_-3")
                return k
            peak = max(peak, h)
    _finish_open(tr, b)
    return n


# ----------------------------------------------------------------------------------- MarketTrend
def rsi20(close: np.ndarray, period: int = 20) -> np.ndarray:
    """Wilder RSI(20), matching market_trend_v043.rsi20 incrementally."""
    n = len(close)
    out = np.full(n, np.nan)
    if n <= period:
        return out
    ch = np.diff(close)
    gain = np.maximum(ch[:period], 0).mean()
    loss = np.maximum(-ch[:period], 0).mean()
    out[period] = 100.0 if loss == 0 else 100 - 100 / (1 + gain / loss)
    for i in range(period, n - 1):
        gain = (gain * (period - 1) + max(ch[i], 0.0)) / period
        loss = (loss * (period - 1) + max(-ch[i], 0.0)) / period
        out[i + 1] = 50.0 if gain == 0 and loss == 0 else (100.0 if loss == 0 else 100 - 100 / (1 + gain / loss))
    return out


def trend_trades(coin: str, tf: str, b: Bars, top50_ok: bool = True, slope_filter: bool = True) -> list[Trade]:
    P = TREND
    n = len(b.c)
    rsi = rsi20(b.c)
    mid = pd.Series(b.c).rolling(20, min_periods=20).mean().to_numpy()   # Bollinger(20, k) middle = SMA20
    slope_bp = np.r_[np.nan, (mid[1:] / mid[:-1] - 1) * 1e4]
    watch_bars = hours_to_bars(P["watch_h"], tf)
    trades: list[Trade] = []
    busy = -1
    watch_until = -1
    fired = False
    for j in range(25, n - 1):
        if not (j <= busy or fired or j > watch_until or not top50_ok):
            # decision during bar j uses only bars <= j-1: middle = SMA20 of the previous 20 closes
            m_prev, s_prev = mid[j - 1], slope_bp[j - 1]
            if (not slope_filter or P["slope_lo"] < s_prev < P["slope_hi"]) and b.o[j] < m_prev <= b.h[j]:   # genuine upward touch
                fired = True
                E = m_prev
                tr = Trade("trend", tf, coin, j, b.t[j], E)
                busy = _manage_trend(tr, b, tf, rsi, E, b.l[j], b.h[j])
                trades.append(tr)
        if rsi[j] <= P["rsi_arm"]:                       # RSI20<=30 on a COMPLETED bar j arms a 24h watch from j+1
            watch_until = j + watch_bars
            fired = False
    return trades


def _manage_trend(tr: Trade, b: Bars, tf: str, rsi: np.ndarray, E: float, ebar_low: float, ebar_high: float) -> int:
    P = TREND
    n = len(b.c)
    j = tr.entry_i
    d = min(P["width_cap"], (ebar_high - ebar_low) / ebar_high)
    if d <= 0:
        _finish_open(tr, b)
        return n
    tgt1 = E * (1 + d) if E >= ebar_high else ebar_high
    tgt2 = tgt1 * (1 + d)
    deadline = j + hours_to_bars(P["hold_h"][tf], tf)
    seq = 0 if rsi[j] <= 30 else -1
    stage = 0
    peak = E
    touched_low = False
    for k in range(j + 1, n):
        o, h, l, c, t = b.o[k], b.h[k], b.l[k], b.c[k], b.t[k]
        rem = _remaining(tr)
        stop = E * (1 - P["hard_stop"])
        if l <= stop:
            _sell(tr, k, t, min(o, stop), rem, "HARD_STOP_-15")
            return k
        if k >= deadline:
            _sell(tr, k, t, o, rem, "HOLD_DEADLINE")
            return k
        if stage == 0 and h >= tgt1:
            _sell(tr, k, t, max(o, tgt1), P["f1"], "TP1")
            stage = 1
            peak = max(peak, tgt1)
        elif stage == 1:
            if l <= tgt1:
                _sell(tr, k, t, min(o, tgt1), _remaining(tr), "TP1_RETOUCH")
                return k
            if h >= tgt2:
                _sell(tr, k, t, max(o, tgt2), P["f2"], "TP2")
                stage = 2
                peak = tgt2
        elif stage >= 2:
            sl = max(tgt2, peak * (1 - P["trail"]))
            if l <= sl:
                _sell(tr, k, t, min(o, sl), _remaining(tr), "TRAIL")
                return k
            peak = max(peak, h)
        peak = max(peak, h)
        # completed-bar exits
        if l <= ebar_low:
            touched_low = True
        if touched_low and c < ebar_low:
            _sell(tr, k, t, c, _remaining(tr), "ENTRY_BAR_LOW_BREAK")
            return k
        if not math.isnan(rsi[k]):
            seq = min(2, seq + 1) if rsi[k] <= 30 else -1
            if seq == 2:
                _sell(tr, k, t, c, _remaining(tr), "RSI_3_BARS")
                return k
    _finish_open(tr, b)
    return n


# ----------------------------------------------------------------------------------- portfolio + stats
def portfolio(trades: list[Trade]) -> dict:
    """App wallet rules: 2,000,000 KRW seed, <=500,000 KRW per coin entry, <=4 concurrent positions."""
    cash = float(SEED)
    open_pos: list[tuple[pd.Timestamp, float, float]] = []   # (exit_t, notional, net_ret)
    skipped = 0
    equity_pts: list[float] = []
    for tr in sorted(trades, key=lambda x: x.entry_t):
        for p in sorted([p for p in open_pos if p[0] <= tr.entry_t]):
            cash += p[1] * (1 + p[2])
            open_pos.remove(p)
            equity_pts.append(cash + sum(q[1] for q in open_pos))
        if len(open_pos) >= MAX_POS or cash < 5000:
            skipped += 1
            continue
        notional = min(MAX_PER_COIN, cash)
        cash -= notional
        open_pos.append((tr.exit_t, notional, tr.ret))
    for p in sorted(open_pos):
        cash += p[1] * (1 + p[2])
        equity_pts.append(cash)
    eq = np.array(equity_pts) if equity_pts else np.array([SEED])
    mdd = float((eq / np.maximum.accumulate(eq) - 1).min())
    return {"portfolio_return": cash / SEED - 1, "portfolio_MDD_realized": mdd, "skipped_no_slot": skipped}


def summarize(trades: list[Trade], strategy: str, tf: str, start, end, ncoins: int) -> dict:
    rets = np.array([t.ret for t in trades])
    row = {"strategy": strategy, "timeframe": tf, "start": start, "end": end, "coins": ncoins,
           "trades": len(trades)}
    if len(rets):
        wins, losses = rets[rets > 0], rets[rets <= 0]
        row.update({
            "win_rate": float((rets > 0).mean()), "avg_trade_ret": float(rets.mean()),
            "median_trade_ret": float(np.median(rets)),
            "avg_win": float(wins.mean()) if len(wins) else np.nan,
            "avg_loss": float(losses.mean()) if len(losses) else np.nan,
            "profit_factor": float(wins.sum() / -losses.sum()) if losses.sum() < 0 else np.nan,
            "open_at_end": int(sum(t.open_at_end for t in trades)),
            **portfolio(trades)})
    return row
