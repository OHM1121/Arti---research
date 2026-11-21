"""Feature library for signal-quality filters (improvement study v2).

Every feature is computed from bars that are COMPLETE at the signal bar (no look-ahead):
  price/trend, volatility, oscillators, volume/flow, candle shape, BTC regime, signed market score,
  universe breadth (cross-sectional), and higher-timeframe (previous completed day) context.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import apk_strategies as S


def _rsi(c: pd.Series, n: int) -> pd.Series:
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def _atr_pct(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["c"].shift(1)
    tr = pd.concat([df["h"] - df["l"], (df["h"] - pc).abs(), (df["l"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / df["c"]


def frame(b: S.Bars) -> pd.DataFrame:
    return pd.DataFrame({"o": b.o, "h": b.h, "l": b.l, "c": b.c, "v": b.v, "val": b.value}, index=b.t)


def core(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l, o, v = df["c"], df["h"], df["l"], df["o"], df["v"]
    f = pd.DataFrame(index=df.index)
    for n in (1, 5, 20, 60):
        f[f"ret{n}"] = c / c.shift(n) - 1
    f["dd60"] = c / h.rolling(60).max() - 1
    f["dd120"] = c / h.rolling(120).max() - 1
    f["pos120"] = (c - l.rolling(120).min()) / (h.rolling(120).max() - l.rolling(120).min())
    ma = {n: c.rolling(n).mean() for n in (5, 20, 60, 120)}
    f["d_ma20"], f["d_ma60"], f["d_ma120"] = c / ma[20] - 1, c / ma[60] - 1, c / ma[120] - 1
    f["slope_ma20"], f["slope_ma60"], f["slope_ma120"] = (ma[20].pct_change(), ma[60].pct_change(), ma[120].pct_change())
    f["spread_5_120"] = ma[5] / ma[120] - 1
    f["atr14"] = _atr_pct(df)
    sd20 = c.rolling(20).std()
    f["bbw20"] = 4 * sd20 / ma[20]
    f["rng"] = (h - l) / c
    f["rng_x"] = (h - l) / (h - l).rolling(20).mean().shift(1)
    r = c.pct_change()
    f["vol_ratio"] = r.rolling(20).std() / r.rolling(100).std()
    f["rsi5"], f["rsi14"], f["rsi20"] = _rsi(c, 5), _rsi(c, 14), _rsi(c, 20)
    ll, hh = l.rolling(14).min(), h.rolling(14).max()
    f["stoch14"] = (c - ll) / (hh - ll)
    tp = (h + l + c) / 3
    f["cci20"] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True))
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    f["macd_h"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c
    f["roc10"] = c / c.shift(10) - 1
    up, dn = h.diff(), -l.diff()
    pdm, mdm = np.where((up > dn) & (up > 0), up, 0.0), np.where((dn > up) & (dn > 0), dn, 0.0)
    atr = _atr_pct(df) * c
    pdi = 100 * pd.Series(pdm, index=df.index).ewm(alpha=1 / 14, adjust=False).mean() / atr
    mdi = 100 * pd.Series(mdm, index=df.index).ewm(alpha=1 / 14, adjust=False).mean() / atr
    f["adx14"] = (100 * (pdi - mdi).abs() / (pdi + mdi)).ewm(alpha=1 / 14, adjust=False).mean()
    down = (c < c.shift(1)).astype(int)
    grp = (down == 0).cumsum()
    f["consec_dn"] = down.groupby(grp).cumsum().clip(upper=20)
    f["volx20"] = v / v.rolling(20).mean().shift(1)
    lv = np.log1p(v)
    f["volz60"] = (lv - lv.rolling(60).mean()) / lv.rolling(60).std()
    f["vol_tr"] = v.rolling(5).mean() / v.rolling(60).mean()
    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    f["obv_sl"] = (obv - obv.shift(20)) / (v.rolling(20).sum() + 1e-12)
    mf = tp * v
    pos, neg = mf.where(tp > tp.shift(1), 0.0).rolling(14).sum(), mf.where(tp < tp.shift(1), 0.0).rolling(14).sum()
    f["mfi14"] = 100 - 100 / (1 + pos / (neg + 1e-12))
    rg = (h - l).replace(0, np.nan)
    f["body"] = (c - o) / o
    f["lwick"] = (np.minimum(o, c) - l) / rg
    f["uwick"] = (h - np.maximum(o, c)) / rg
    f["cloc"] = (c - l) / rg
    return f


def btc_block(btc: pd.DataFrame) -> pd.DataFrame:
    c = btc["c"]
    out = pd.DataFrame(index=btc.index)
    out["btc_ret5"], out["btc_ret20"] = c / c.shift(5) - 1, c / c.shift(20) - 1
    out["btc_dd60"] = c / btc["h"].rolling(60).max() - 1
    out["btc_rsi14"] = _rsi(c, 14)
    out["btc_d_ma120"] = c / c.rolling(120).mean() - 1
    out["btc_atr14"] = _atr_pct(btc)
    return out


def htf_block(day: pd.DataFrame) -> pd.DataFrame:
    c = day["c"]
    out = pd.DataFrame(index=day.index)
    out["h_rsi14"], out["h_ret5"] = _rsi(c, 14), c / c.shift(5) - 1
    out["h_d_ma20"], out["h_d_ma60"] = c / c.rolling(20).mean() - 1, c / c.rolling(60).mean() - 1
    out["h_atr"] = _atr_pct(day)
    return out


def breadth_block(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    idx = sorted(set().union(*[set(f.index) for f in frames.values()]))
    close = pd.DataFrame({k: f["c"].reindex(idx) for k, f in frames.items()})
    ret1 = close.pct_change()
    rsi = close.apply(lambda s: _rsi(s, 14))
    out = pd.DataFrame(index=close.index)
    out["br_ma20"] = (close > close.rolling(20).mean()).where(close.notna()).mean(axis=1)
    out["br_os"] = (rsi < 30).where(rsi.notna()).mean(axis=1)
    out["br_up"] = (ret1 > 0).where(ret1.notna()).mean(axis=1)
    return out


def build_all(coins: list[str], tf: str, sc: "S.ScoreLookup") -> dict[str, pd.DataFrame]:
    """Feature frame per coin (aligned to that coin's bars)."""
    bars = {c: S.load_bars(c, tf) for c in coins}
    bars = {c: b for c, b in bars.items() if b is not None}
    frames = {c: frame(b) for c, b in bars.items()}
    btc = btc_block(frames["BTC"])
    br = breadth_block(frames)
    days = {c: S.load_bars(c, "day") for c in coins} if tf != "day" else {}
    out = {}
    for c, df in frames.items():
        f = core(df)
        f = f.join(btc.reindex(f.index))
        f["rel20"] = f["ret20"] - f["btc_ret20"]
        f = f.join(br.reindex(f.index))
        s = S._score_array(bars[c], sc)
        f["score"] = s
        t = bars[c].t.to_numpy("datetime64[ns]").astype("int64") - 24 * 3600 * 10**9
        idx = np.searchsorted(sc.ends, t, side="right") - 1
        s24 = np.where(idx >= 0, sc.score[np.clip(idx, 0, None)], np.nan)
        f["score_chg"] = s - s24
        if tf != "day" and days.get(c) is not None:
            d = htf_block(frame(days[c]))
            key = f.index.normalize() - pd.Timedelta(days=1)            # previous COMPLETED day
            hv = d.reindex(key).to_numpy()
            for j, name in enumerate(d.columns):
                f[name] = hv[:, j]
        out[c] = f.astype("float32")
    return out
